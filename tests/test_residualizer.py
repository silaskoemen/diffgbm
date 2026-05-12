import numpy as np
import pytest

import treeffuser._residualizer as residualizer_module
from treeffuser._residualizer import ConditionalResidualizer
from treeffuser._residualizer import _winsorized_std


def _make_heteroscedastic_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    mean = X[:, :1] - 0.5 * X[:, 1:2]
    scale = 0.2 + 0.6 / (1.0 + np.exp(-X[:, :1]))
    y = mean + scale * rng.normal(size=(n, 1))
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    return X, y


def _fast_params():
    return {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 5,
    }


@pytest.mark.parametrize("mode", ["mean", "mean_scale"])
def test_conditional_residualizer_transform_inverse_identity(mode):
    X, y = _make_heteroscedastic_data()
    residualizer = ConditionalResidualizer(
        residualize=mode,
        k_folds=3,
        seed=0,
        extra_params=_fast_params(),
    )
    residualizer.fit(X, y)

    residual = residualizer.transform(X, y)
    reconstructed = residualizer.inverse_transform(X, residual)

    assert residual.shape == y.shape
    assert np.allclose(reconstructed, y)


@pytest.mark.parametrize("mode", ["mean", "mean_scale"])
def test_conditional_residualizer_oof_residuals_are_standardized(mode):
    X, y = _make_heteroscedastic_data()
    residualizer = ConditionalResidualizer(
        residualize=mode,
        k_folds=3,
        seed=0,
        extra_params=_fast_params(),
    )
    residual = residualizer.fit_transform(X, y)

    assert abs(float(np.mean(residual))) < 0.05
    assert 0.9 < float(_winsorized_std(residual)[0]) < 1.1
    assert 0.8 < float(np.std(residual)) < 1.2


def test_conditional_residualizer_is_deterministic_with_fixed_seed():
    X, y = _make_heteroscedastic_data()
    residualizer_a = ConditionalResidualizer("mean_scale", k_folds=3, seed=123, extra_params=_fast_params())
    residualizer_b = ConditionalResidualizer("mean_scale", k_folds=3, seed=123, extra_params=_fast_params())

    residual_a = residualizer_a.fit_transform(X, y)
    residual_b = residualizer_b.fit_transform(X, y)

    assert np.allclose(residual_a, residual_b)
    assert np.allclose(residualizer_a.mean_oof, residualizer_b.mean_oof)
    assert np.allclose(residualizer_a.scale_oof, residualizer_b.scale_oof)
    assert np.allclose(residualizer_a.residual_global_scale, residualizer_b.residual_global_scale)


def test_conditional_residualizer_train_and_inference_residual_gap_is_bounded():
    X, y = _make_heteroscedastic_data(n=320)
    residualizer = ConditionalResidualizer(
        "mean_scale",
        k_folds=4,
        seed=123,
        extra_params=_fast_params(),
    )

    training_residual = residualizer.fit_transform(X, y)
    inference_residual = residualizer.transform(X, y)
    gap = training_residual - inference_residual

    assert np.std(gap) < 0.5 * np.std(training_residual)
    assert abs(float(np.mean(inference_residual))) < 0.25
    assert 0.65 < float(np.std(inference_residual)) < 1.3


def test_conditional_residualizer_warns_when_k_folds_are_reduced():
    X, y = _make_heteroscedastic_data(n=120)
    residualizer = ConditionalResidualizer(
        "mean",
        k_folds=5,
        seed=0,
        extra_params=_fast_params(),
    )

    with pytest.warns(UserWarning, match="Reducing residualize_k_folds"):
        residualizer.fit(X, y)
    assert residualizer.effective_k_folds == 3


def test_conditional_residualizer_passes_cat_idx_to_lightgbm(monkeypatch):
    calls = []

    class FakeModel:
        def predict(self, X):
            return np.zeros(X.shape[0])

    def fake_fit_one_lgbm_model(**kwargs):
        calls.append(kwargs["cat_idx"])
        return FakeModel()

    monkeypatch.setattr(residualizer_module, "_fit_one_lgbm_model", fake_fit_one_lgbm_model)
    X, y = _make_heteroscedastic_data(n=120)
    residualizer = ConditionalResidualizer(
        "mean_scale",
        k_folds=3,
        seed=0,
        extra_params=_fast_params(),
    )

    residualizer.fit(X, y, cat_idx=[1])

    assert calls
    assert all(call == [1] for call in calls)


def test_conditional_residualizer_scale_floor_handles_degenerate_residuals():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 2))
    y = np.ones((120, 1))
    residualizer = ConditionalResidualizer(
        residualize="mean_scale",
        k_folds=3,
        seed=0,
        extra_params=_fast_params(),
    )

    residual = residualizer.fit_transform(X, y)
    scale = residualizer.predict_scale(X)

    assert np.all(np.isfinite(residual))
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0)


def test_conditional_residualizer_requires_enough_rows():
    X, y = _make_heteroscedastic_data(n=79)
    residualizer = ConditionalResidualizer(
        residualize="mean",
        k_folds=5,
        seed=0,
        extra_params=_fast_params(),
    )

    with pytest.raises(ValueError, match="at least 80"):
        residualizer.fit(X, y)


def test_conditional_residualizer_preserves_multioutput_shape():
    X, y1 = _make_heteroscedastic_data()
    y = np.concatenate([y1, -0.5 * y1 + 0.1 * X[:, :1]], axis=1)
    residualizer = ConditionalResidualizer(
        residualize="mean_scale",
        k_folds=3,
        seed=0,
        extra_params=_fast_params(),
    )

    residual = residualizer.fit_transform(X, y)
    reconstructed = residualizer.inverse_transform(X, residualizer.transform(X, y))

    assert residual.shape == y.shape
    assert reconstructed.shape == y.shape
    assert np.allclose(reconstructed, y)
