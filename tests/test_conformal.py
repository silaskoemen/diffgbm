"""
Tests for the split-CQR conformal calibrator. We avoid retraining a tabular diffusion
inside the unit tests by feeding hand-crafted sample tensors directly into the
calibrator's `_from_samples` API. End-to-end Treeffuser integration is covered by a
short statistical regression test on a tiny synthetic dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from treeffuser import ConformalQuantileCalibrator
from treeffuser import Treeffuser
from treeffuser._conformal import empirical_quantile_band


def _make_normal_samples(means: np.ndarray, std: float, n_samples: int, seed: int) -> np.ndarray:
    """Helper: produce `n_samples` Gaussian samples per row with given conditional means."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_samples, means.shape[0], means.shape[1]))
    return means[None, :, :] + std * noise


def test_empirical_quantile_band_returns_correct_bounds():
    samples = np.arange(101, dtype=float).reshape(101, 1, 1)
    lower, upper = empirical_quantile_band(samples, level=0.9)
    assert lower.shape == (1, 1)
    assert np.isclose(lower[0, 0], 5.0)
    assert np.isclose(upper[0, 0], 95.0)


def test_empirical_quantile_band_rejects_invalid_level():
    samples = np.zeros((10, 2, 1))
    with pytest.raises(ValueError, match=r"level must be in"):
        empirical_quantile_band(samples, level=0.0)
    with pytest.raises(ValueError, match=r"level must be in"):
        empirical_quantile_band(samples, level=1.0)


def test_conformal_calibrator_rejects_invalid_level():
    with pytest.raises(ValueError, match=r"level must be in"):
        ConformalQuantileCalibrator(level=0.0)
    with pytest.raises(ValueError, match=r"level must be in"):
        ConformalQuantileCalibrator(level=1.0)


def test_predict_interval_requires_fit():
    calibrator = ConformalQuantileCalibrator(level=0.9)
    samples = np.zeros((10, 4, 1))
    with pytest.raises(RuntimeError):
        calibrator.predict_interval_from_samples(samples)


def test_fit_from_samples_validates_shapes():
    cal = ConformalQuantileCalibrator(level=0.9)
    samples = np.zeros((20, 5, 1))
    y_cal_wrong_batch = np.zeros((4, 1))
    with pytest.raises(ValueError, match=r"batch dimension"):
        cal.fit_from_samples(samples, y_cal_wrong_batch)

    y_cal_wrong_dim = np.zeros((5, 2))
    with pytest.raises(ValueError, match=r"output dimension"):
        cal.fit_from_samples(samples, y_cal_wrong_dim)


def test_conformal_radius_is_zero_when_samples_already_cover_truth():
    """If the empirical band already brackets every calibration point, the radius
    collapses to (at most) zero — confirming we are not silently widening intervals."""
    n_cal = 200
    truth = np.zeros((n_cal, 1))
    # Samples wide enough that the (alpha/2, 1-alpha/2) band straddles truth=0 with margin.
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.0, scale=2.0, size=(500, n_cal, 1))

    cal = ConformalQuantileCalibrator(level=0.9).fit_from_samples(samples, truth)
    assert cal.fitted
    assert cal.radius is not None
    # Conformity scores are negative when truth is inside the band; the empirical
    # high quantile of negative scores stays <= 0, so the radius is non-positive.
    assert float(cal.radius[0]) <= 0.0


def test_conformal_calibration_restores_target_coverage_on_undercovered_model():
    """A model whose samples are intentionally too narrow should be widened by the
    calibrator to recover near-nominal empirical coverage on held-out data."""
    rng = np.random.default_rng(42)
    n_cal, n_eval = 500, 500
    true_std = 1.0
    biased_std = 0.4  # model is overconfident
    y_cal = rng.normal(scale=true_std, size=(n_cal, 1))
    y_eval = rng.normal(scale=true_std, size=(n_eval, 1))
    samples_cal = _make_normal_samples(means=np.zeros_like(y_cal), std=biased_std, n_samples=400, seed=1)
    samples_eval = _make_normal_samples(means=np.zeros_like(y_eval), std=biased_std, n_samples=400, seed=2)

    raw_lo, raw_hi = empirical_quantile_band(samples_eval, level=0.9)
    raw_coverage = float(np.mean((y_eval >= raw_lo) & (y_eval <= raw_hi)))
    assert raw_coverage < 0.7  # confirm the model is severely undercovered

    cal = ConformalQuantileCalibrator(level=0.9).fit_from_samples(samples_cal, y_cal)
    cqr_lo, cqr_hi = cal.predict_interval_from_samples(samples_eval)
    cqr_coverage = float(np.mean((y_eval >= cqr_lo) & (y_eval <= cqr_hi)))

    assert cqr_coverage >= 0.85  # nominal 0.9 - finite-sample slack


def test_treeffuser_can_be_calibrated_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(0)
    n_train, n_cal, n_eval = 200, 200, 200
    X_train = rng.normal(size=(n_train, 2))
    y_train = (X_train[:, :1] + 0.3 * rng.normal(size=(n_train, 1))).astype(np.float64)
    X_cal = rng.normal(size=(n_cal, 2))
    y_cal = (X_cal[:, :1] + 0.3 * rng.normal(size=(n_cal, 1))).astype(np.float64)
    X_eval = rng.normal(size=(n_eval, 2))
    y_eval = (X_eval[:, :1] + 0.3 * rng.normal(size=(n_eval, 1))).astype(np.float64)

    model = Treeffuser(
        n_repeats=3,
        n_estimators=40,
        early_stopping_rounds=None,
        learning_rate=0.1,
        verbose=-1,
        seed=0,
    )
    model.fit(X_train, y_train)

    calibrator = ConformalQuantileCalibrator(level=0.9).fit(
        model=model,
        X_cal=X_cal,
        y_cal=y_cal,
        n_samples=80,
        n_steps=20,
        n_parallel=10,
        seed=0,
    )
    lower, upper = calibrator.predict_interval(
        model=model,
        X=X_eval,
        n_samples=80,
        n_steps=20,
        n_parallel=10,
        seed=1,
    )

    assert lower.shape == y_eval.shape
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(upper >= lower)
