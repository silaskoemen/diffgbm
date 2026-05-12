"""
Tests for the probability-flow ODE wrapper and the Heun solver.

The Treeffuser SDE module is purely numerical so we exercise the new code paths
against analytic OU/VESDE marginals where the reverse-time density is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from treeffuser import Treeffuser
from treeffuser.sde import sdeint
from treeffuser.sde.base_sde import CustomSDE
from treeffuser.sde.base_sde import ProbabilityFlowODE
from treeffuser.sde.diffusion_sdes import VESDE


def test_heun_matches_analytical_solution_for_deterministic_drift():
    """For dy = -y dt the closed-form integral is y(t) = y0 * exp(-t)."""
    sde = CustomSDE(drift_fn=lambda y, t: -y, diffusion_fn=lambda y, t: 0.0 * y)
    y0 = np.ones((4, 1))
    samples = sdeint(sde, y0, 0.0, 1.0, n_steps=20, method="heun", seed=0)
    expected = y0 * np.exp(-1.0)
    assert np.allclose(samples, expected, atol=1e-3)


def test_heun_more_accurate_than_euler_on_smooth_ode():
    """For dy = -y dt the Heun corrector should beat Euler at the same step count."""
    sde = CustomSDE(drift_fn=lambda y, t: -y, diffusion_fn=lambda y, t: 0.0 * y)
    y0 = np.ones((1, 1))
    euler = sdeint(sde, y0, 0.0, 1.0, n_steps=5, method="euler", seed=0)
    heun = sdeint(sde, y0, 0.0, 1.0, n_steps=5, method="heun", seed=0)
    truth = y0 * np.exp(-1.0)
    assert abs(float(heun.item() - truth.item())) < abs(float(euler.item() - truth.item()))


def test_pf_ode_rejected_in_forward_time():
    sde = CustomSDE(drift_fn=lambda y, t: 0 * y, diffusion_fn=lambda y, t: 0 * y + 1.0)
    y0 = np.zeros((4, 1))
    with pytest.raises(ValueError, match="reverse-time"):
        sdeint(sde, y0, 0.0, 1.0, n_steps=5, method="euler", pf_ode=True, score_fn=lambda y, t: y)


def test_pf_ode_recovers_data_distribution_on_known_vesde_marginal():
    """Reverse a VESDE with the analytic score and check the empirical marginal of
    the PF-ODE-Heun sampler matches the training marginal in mean/std."""
    rng = np.random.default_rng(0)
    n = 4000
    rng.normal(loc=0.0, scale=1.0, size=(n, 1))
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=5.0)

    def analytic_score(y, t):
        # Conditional VESDE marginal under standard normal data:
        # y_t = y0 + sigma(t) z, y0 ~ N(0,1), so y_t ~ N(0, 1 + sigma(t)^2),
        # and score(y_t) = -y_t / (1 + sigma(t)^2). Clamp std so the Heun corrector,
        # which evaluates score at the t1 boundary, stays well-defined at t -> 0.
        t_safe = np.clip(t, 1e-4, None)
        _, std = sde.get_mean_std_pt_given_y0(np.ones_like(y), t_safe)
        var = 1.0 + std**2
        return -y / var

    y_prior = sde.sample_from_theoretical_prior((n, 1), seed=1)
    samples = sdeint(
        sde,
        y_prior,
        sde.T,
        0.0,
        n_steps=60,
        method="heun",
        score_fn=analytic_score,
        pf_ode=True,
        seed=2,
    )
    assert samples.shape == (n, 1)
    assert abs(float(samples.mean())) < 0.1
    assert abs(float(samples.std()) - 1.0) < 0.1


def test_probability_flow_ode_drift_halves_score_term():
    """Compare ProbabilityFlowODE drift to ReverseSDE drift directly. The score
    contribution should be halved and the diffusion zeroed."""
    sde = CustomSDE(
        drift_fn=lambda y, t: -0.5 * y,
        diffusion_fn=lambda y, t: 0 * y + 0.8,
    )
    pf = ProbabilityFlowODE(sde, t_reverse_origin=1.0, score_fn=lambda y, t: y * 2.0)
    y = np.array([[1.0], [2.0]])
    t = np.array([[0.25], [0.25]])
    drift, diffusion = pf.drift_and_diffusion(y, t)
    expected_drift = 0.5 * y + 0.5 * (0.8**2) * (2.0 * y)
    assert np.allclose(drift, expected_drift)
    assert np.allclose(diffusion, 0.0)


def test_treeffuser_sample_accepts_heun_and_pf_ode():
    """Integration smoke: a small Treeffuser fit followed by ODE-Heun sampling."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = (X[:, :1] + 0.3 * rng.normal(size=(120, 1))).astype(np.float64)
    m = Treeffuser(
        n_repeats=2,
        n_estimators=30,
        early_stopping_rounds=None,
        learning_rate=0.1,
        verbose=-1,
        seed=0,
    )
    m.fit(X, y)
    samples = m.sample(
        X[:8],
        n_samples=20,
        n_steps=20,
        sampler_method="heun",
        pf_ode=True,
        verbose=False,
        seed=0,
    )
    assert samples.shape == (20, 8, 1)
    assert np.all(np.isfinite(samples))
