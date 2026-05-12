import numpy as np
import pytest
from benchmarks.metrics import evaluate_samples


def test_evaluate_samples_reports_abs_coverage_error_and_valid_widths():
    y_true = np.arange(10, dtype=float).reshape(-1, 1)
    y_samples = np.stack(
        [
            y_true[:, 0] - 1.0,
            y_true[:, 0],
            y_true[:, 0] + 1.0,
        ],
        axis=0,
    )[:, :, None]
    X_test = y_true.copy()

    metrics = evaluate_samples(
        y_samples=y_samples,
        y_true=y_true,
        X_test=X_test,
        coverage_levels=(0.90, 0.95),
    )

    assert metrics["interval_90_coverage"] == 1.0
    assert metrics["interval_90_coverage_error"] == pytest.approx(0.10)
    assert metrics["interval_90_abs_coverage_error"] == pytest.approx(0.10)
    assert metrics["interval_90_valid_width_01"] is None
    assert metrics["interval_90_valid_width_02"] is None

    assert metrics["interval_95_coverage_error"] == pytest.approx(0.05)
    assert metrics["interval_95_abs_coverage_error"] == pytest.approx(0.05)
