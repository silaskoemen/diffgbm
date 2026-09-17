"""Per-test-point CRPS / PIT persistence for paired uncertainty estimates.

Fold-level means are enough to rank arms but not to put an interval on a
difference: a paired bootstrap needs the same test points under both arms. Each
eval run writes one compressed `.npz` alongside its JSONL, holding the unreduced
arrays per eval fold together with the split indices they align to, so two runs
can be paired by index rather than by position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PER_POINT_DIRNAME = "per_point"


def per_point_path(results_dir: Path, stem: str) -> Path:
    """Sidecar path for the run whose JSONL is `results_dir / f"{stem}.jsonl"`."""
    return results_dir / PER_POINT_DIRNAME / f"{stem}.npz"


def save_per_point(path: Path, folds: dict[int, dict[str, Any]], *, merge: bool = False) -> Path:
    """Write `{eval_fold: {"crps", "pit", "test_idx"}}` to `path`.

    CRPS is one value per test point (averaged over output dims); PIT keeps its
    `(batch, y_dim)` shape; `test_idx` indexes the concatenated (train, test)
    design matrix that `build_splits` partitions. With `merge`, folds already in
    an existing file are kept unless this call supplies them again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {}
    if merge and path.exists():
        with np.load(path) as existing:
            payload.update({k: existing[k] for k in existing.files})

    for fold_id, arrays in sorted(folds.items()):
        payload[f"fold{fold_id}_crps"] = np.asarray(arrays["crps"], dtype=np.float32)
        payload[f"fold{fold_id}_pit"] = np.asarray(arrays["pit"], dtype=np.float32)
        payload[f"fold{fold_id}_test_idx"] = np.asarray(arrays["test_idx"], dtype=np.int32)

    np.savez_compressed(path, **payload)
    return path


def load_per_point(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Inverse of `save_per_point`, keyed by eval fold."""
    folds: dict[int, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            fold_part, _, name = key.partition("_")
            folds.setdefault(int(fold_part.removeprefix("fold")), {})[name] = data[key]
    return folds
