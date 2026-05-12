import numpy as np
from jaxtyping import Float

from treeffuser.sde.base_solver import BaseSDESolver
from treeffuser.sde.base_solver import _register_solver


@_register_solver(name="euler")
class EulerMaruyama(BaseSDESolver):
    """
    Euler-Maruyama solver for SDEs [1].

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    """

    def step(self, y0: Float[np.ndarray, "batch y_dim"], t0: float, t1: float) -> Float[np.ndarray, "batch y_dim"]:
        dt = t1 - t0
        t0_arr = np.broadcast_to(t0, (*y0.shape[:-1], 1))
        drift, diffusion = self.sde.drift_and_diffusion(y0, t0_arr)
        dW = self._rng.normal(size=y0.shape)
        return y0 + drift * dt + diffusion * np.sqrt(dt) * dW


@_register_solver(name="heun")
class Heun(BaseSDESolver):
    """
    Heun second-order predictor-corrector solver.

    Designed for deterministic ODEs (e.g. the probability-flow ODE associated with a
    diffusion SDE). For zero-diffusion targets the step reduces to the classical
    Heun ODE update

        y_pred  = y + drift(y, t0) * dt
        y_next  = y + 0.5 * (drift(y, t0) + drift(y_pred, t1)) * dt.

    When the underlying SDE has non-zero diffusion the noise term is added using a
    single shared Brownian increment for the predictor and corrector, recovering a
    valid stochastic-Heun update. The deterministic case is the intended path: the
    EDM and probability-flow literature consistently uses this solver for tabular
    diffusion sampling.
    """

    def step(self, y0: Float[np.ndarray, "batch y_dim"], t0: float, t1: float) -> Float[np.ndarray, "batch y_dim"]:
        dt = t1 - t0
        t0_arr = np.broadcast_to(t0, (*y0.shape[:-1], 1))
        t1_arr = np.broadcast_to(t1, (*y0.shape[:-1], 1))
        drift0, diffusion0 = self.sde.drift_and_diffusion(y0, t0_arr)
        deterministic_only = not bool(np.any(diffusion0))
        if deterministic_only:
            y_pred = y0 + drift0 * dt
            drift1, _ = self.sde.drift_and_diffusion(y_pred, t1_arr)
            return y0 + 0.5 * (drift0 + drift1) * dt
        dW = self._rng.normal(size=y0.shape)
        sqrt_dt = np.sqrt(dt)
        y_pred = y0 + drift0 * dt + diffusion0 * sqrt_dt * dW
        drift1, diffusion1 = self.sde.drift_and_diffusion(y_pred, t1_arr)
        return y0 + 0.5 * (drift0 + drift1) * dt + 0.5 * (diffusion0 + diffusion1) * sqrt_dt * dW
