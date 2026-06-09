import numpy as np
import pytest
from benchmarks.baselines import QuantileRandomForestBaseline
from benchmarks.baselines import make_baseline_model


def test_quantile_forest_baseline_samples_with_expected_shape():
    pytest.importorskip("quantile_forest")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=60)).reshape(-1, 1)
    X_test = rng.normal(size=(7, 3))

    model = QuantileRandomForestBaseline(
        quantile_count=9,
        n_estimators=8,
        max_samples_leaf=None,
        n_jobs=1,
        seed=0,
    )
    model.fit(X, y)

    samples = model.sample(X_test, n_samples=11, seed=1)

    assert samples.shape == (11, 7, 1)
    assert np.isfinite(samples).all()


def test_quantile_forest_baseline_is_available_from_factory():
    model = make_baseline_model(
        model_type="quantile_forest",
        params={"n_estimators": 3, "quantile_count": 5},
        seed=0,
    )

    assert isinstance(model, QuantileRandomForestBaseline)
