"""
Conformalized quantile-regression calibrator for sample-based predictive models.

Implements split conformalized quantile regression (CQR; Romano, Patterson & Candes,
2019) on top of the empirical quantiles of a model's posterior samples. The wrapper is
post-fit and treats the underlying sampler as a black box, so it composes with any
Treeffuser configuration without changing fitting behavior.
"""

from __future__ import annotations

import math

import numpy as np
from jaxtyping import Float


def empirical_quantile_band(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    level: float,
) -> tuple[Float[np.ndarray, "batch y_dim"], Float[np.ndarray, "batch y_dim"]]:
    """Empirical (alpha/2, 1-alpha/2) quantiles per row from sampled outputs."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1).")
    alpha = 1.0 - level
    lower = np.quantile(y_samples, alpha / 2.0, axis=0)
    upper = np.quantile(y_samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


class ConformalQuantileCalibrator:
    """
    Split CQR on top of sample-based predictive intervals.

    For a target level `level` (e.g. 0.9), the calibrator computes per-row
    conformity scores

        E_i = max(q_lo(X_cal_i) - y_cal_i, y_cal_i - q_hi(X_cal_i))

    on a calibration set, where `q_lo`, `q_hi` are the empirical (alpha/2, 1-alpha/2)
    quantiles of the model's samples at `X_cal_i`. The calibration radius is the
    `ceil((n_cal + 1) * level) / n_cal` empirical quantile of `{E_i}`. New predictive
    intervals expand the sampled quantile band by that radius on each side.

    Parameters
    ----------
    level : float
        Nominal coverage in (0, 1).
    """

    def __init__(self, level: float = 0.9) -> None:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        self.level = float(level)
        self.radius: Float[np.ndarray, "y_dim"] | None = None
        self._n_cal: int | None = None

    @property
    def fitted(self) -> bool:
        return self.radius is not None

    def fit_from_samples(
        self,
        y_samples_cal: Float[np.ndarray, "n_samples batch y_dim"],
        y_cal: Float[np.ndarray, "batch y_dim"],
    ) -> "ConformalQuantileCalibrator":
        """
        Compute the per-output calibration radius from precomputed calibration samples.

        Decoupling sampling from calibration lets benchmarks reuse one sample tensor
        for both raw and conformalized metric reporting.
        """
        samples = np.asarray(y_samples_cal)
        truth = np.asarray(y_cal)
        if truth.ndim == 1:
            truth = truth.reshape(-1, 1)
        if samples.ndim == 2:
            samples = samples[:, :, None]
        if samples.shape[1] != truth.shape[0]:
            raise ValueError("Calibration samples and y_cal must share the batch dimension.")
        if samples.shape[2] != truth.shape[1]:
            raise ValueError("Calibration samples and y_cal must share the output dimension.")

        lower, upper = empirical_quantile_band(samples, self.level)
        scores = np.maximum(lower - truth, truth - upper)
        n_cal = scores.shape[0]
        if n_cal < 2:
            raise ValueError("Need at least two calibration points for split conformal.")
        # Finite-sample CQR uses the ceil((n+1)*level)/n empirical quantile.
        rank = math.ceil((n_cal + 1) * self.level) / n_cal
        if rank >= 1.0:
            # Asks for more than n_cal points -> not enough data for this level; fall back.
            radius = scores.max(axis=0)
        else:
            radius = np.quantile(scores, rank, axis=0)
        self.radius = radius
        self._n_cal = n_cal
        return self

    def fit(
        self,
        model,
        X_cal: Float[np.ndarray, "batch x_dim"],
        y_cal: Float[np.ndarray, "batch y_dim"],
        n_samples: int = 200,
        n_steps: int = 50,
        n_parallel: int = 10,
        seed: int | None = None,
    ) -> "ConformalQuantileCalibrator":
        """Sample from `model` at `X_cal` and calibrate against `y_cal`."""
        y_samples = model.sample(
            X_cal,
            n_samples=n_samples,
            n_parallel=n_parallel,
            n_steps=n_steps,
            seed=seed,
            verbose=False,
        )
        return self.fit_from_samples(y_samples_cal=y_samples, y_cal=y_cal)

    def predict_interval_from_samples(
        self,
        y_samples_test: Float[np.ndarray, "n_samples batch y_dim"],
    ) -> tuple[Float[np.ndarray, "batch y_dim"], Float[np.ndarray, "batch y_dim"]]:
        """Return calibrated `(lower, upper)` intervals expanded by the conformal radius."""
        if self.radius is None:
            raise RuntimeError("Calibrator has not been fitted.")
        samples = np.asarray(y_samples_test)
        if samples.ndim == 2:
            samples = samples[:, :, None]
        lower, upper = empirical_quantile_band(samples, self.level)
        return lower - self.radius, upper + self.radius

    def predict_interval(
        self,
        model,
        X: Float[np.ndarray, "batch x_dim"],
        n_samples: int = 200,
        n_steps: int = 50,
        n_parallel: int = 10,
        seed: int | None = None,
    ) -> tuple[Float[np.ndarray, "batch y_dim"], Float[np.ndarray, "batch y_dim"]]:
        """Sample from `model` at `X` and return the conformalized interval."""
        y_samples = model.sample(
            X,
            n_samples=n_samples,
            n_parallel=n_parallel,
            n_steps=n_steps,
            seed=seed,
            verbose=False,
        )
        return self.predict_interval_from_samples(y_samples)
