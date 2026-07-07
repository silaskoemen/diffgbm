"""Eval-only transfer diagnostics: refit a tuned YAML with explicit param overrides.

Loads a tuned best-params YAML, applies overrides to the model params (set keys
via a JSON dict, or unset keys entirely), then refits and evaluates on the held-out
evaluation folds with the YAML's sampler. No retuning happens, so results are
"transfer" diagnostics: the tunable LightGBM surface stays at the values selected
under the original fixed params.

The search-space fingerprint check is intentionally skipped — overridden params no
longer match any registered space. Each output row records the label and the exact
overrides applied.

Usage:
    python -m benchmarks.scripts.run_param_override_eval \
        benchmarks/results/tuning/best_params/ct_slices__treeffuser_score_plus_noresid.yaml \
        --label adaptive_sigma_max \
        --unset-params sde_hyperparam_min sde_hyperparam_max \
        --folds 1 2 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from benchmarks.datasets import make_dataset
from benchmarks.harness import get_provenance
from benchmarks.metrics import crps_climatology
from benchmarks.metrics import crps_skill_score
from benchmarks.metrics import evaluate_samples
from benchmarks.tuning.evaluate import _json_safe
from benchmarks.tuning.evaluate import _sample_kwargs
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.splits import build_splits
from benchmarks.variants import Variant

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/results/param_override_eval"


def evaluate_with_overrides(
    tuned_yaml: Path,
    *,
    label: str,
    set_params: dict[str, Any],
    unset_params: list[str],
    sampler_overrides: dict[str, Any],
    fold_ids: set[int] | None,
    results_dir: Path,
    append: bool,
) -> Path:
    payload = yaml.safe_load(tuned_yaml.read_text())
    protocol = payload["protocol"]
    space_name = protocol["space_name"]
    if space_name not in SPACES:
        raise ValueError(f"Tuned YAML references unknown space {space_name!r}.")
    space = SPACES[space_name]

    dataset_name = protocol["dataset"]
    bound = logger.bind(dataset=dataset_name, space=space_name, label=label)

    params = dict(payload["params"])
    for key in unset_params:
        if key in params:
            del params[key]
        else:
            bound.warning("Unset key {!r} not present in tuned params; ignoring.", key)
    overlap = set(set_params) & set(unset_params)
    if overlap:
        raise ValueError(f"Keys both set and unset: {sorted(overlap)}")
    params.update(set_params)

    sampler = dict(payload["sampler"] or {})
    sampler.update(sampler_overrides)

    bound.info("Overrides: set={}, unset={}, sampler={}", set_params, unset_params, sampler_overrides)

    bundle = make_dataset(
        name=dataset_name,
        n_train=protocol["n_train"],
        n_test=protocol["n_test"],
        seed=protocol["master_seed"],
        x_dim=protocol["x_dim"],
    )
    X = np.concatenate([bundle.X_train, bundle.X_test], axis=0)
    y = np.concatenate([bundle.y_train, bundle.y_test], axis=0)
    splits = build_splits(
        X,
        y,
        master_seed=protocol["master_seed"],
        n_folds=protocol["n_folds"],
        name=dataset_name,
    )

    provenance = get_provenance()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name}__{space_name}__{label}.jsonl"
    if not append:
        out_path.unlink(missing_ok=True)

    sample_kwargs = _sample_kwargs(sampler)
    eval_fold_ids = [k for k in range(protocol["n_folds"]) if k != 0]

    rows_written = 0
    with out_path.open("a") as fh:
        for eval_fold, fold in zip(eval_fold_ids, splits.eval_folds, strict=True):
            if fold_ids is not None and eval_fold not in fold_ids:
                continue
            X_train, y_train = splits.slice(fold.train_idx)
            X_test, y_test = splits.slice(fold.test_idx)

            fold_bound = bound.bind(eval_fold=eval_fold)
            fold_bound.info("Fit start (n_train={}, n_test={})", X_train.shape[0], X_test.shape[0])

            variant = Variant(name=f"{space_name}__{label}", params=params, model=space.model)
            model = variant.make_model(seed=protocol["master_seed"] + 10_000)

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_samples = model.sample(
                X_test,
                seed=protocol["master_seed"] + 20_000 + eval_fold,
                **sample_kwargs,
            )
            sample_time = time.perf_counter() - t0

            metrics = evaluate_samples(y_samples=y_samples, y_true=y_test, X_test=X_test)
            clim = crps_climatology(y_train=y_train, y_true=y_test)

            row: dict[str, Any] = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "dataset": dataset_name,
                "space": space_name,
                "label": label,
                "override_set_params": set_params,
                "override_unset_params": unset_params,
                "override_sampler": sampler_overrides,
                "source_yaml": str(tuned_yaml),
                "model": space.model,
                "eval_fold": eval_fold,
                "master_seed": protocol["master_seed"],
                "n_folds": protocol["n_folds"],
                "n_train_fold": int(X_train.shape[0]),
                "n_test_fold": int(X_test.shape[0]),
                "fit_time": fit_time,
                "sample_time": sample_time,
                "crps_climatology": clim,
                "crps_skill_score": crps_skill_score(crps_model=metrics["crps"], crps_climatology_val=clim),
                "source_protocol_fingerprint": payload["protocol_fingerprint"],
                "source_best_value_crps_tuning": payload["best_value_crps"],
                "sampler": sampler,
                "params": params,
            }
            row.update(provenance)
            row.update(metrics)
            fh.write(json.dumps(_json_safe(row), sort_keys=True))
            fh.write("\n")
            fh.flush()
            rows_written += 1
            fold_bound.success(
                "CRPS={:.4f} (skill={:.3f}); fit={:.1f}s, sample={:.1f}s",
                metrics["crps"],
                row["crps_skill_score"],
                fit_time,
                sample_time,
            )

    bound.success("Wrote {} rows to {}", rows_written, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Eval-only transfer diagnostic with param overrides.")
    parser.add_argument("tuned_yaml", type=Path, help="Tuned best-params YAML to start from.")
    parser.add_argument("--label", required=True, help="Diagnostic label (recorded in rows, used in output filename).")
    parser.add_argument(
        "--set-json",
        default="{}",
        help="JSON dict merged into the tuned model params, e.g. '{\"extra_residualizer_params\": {...}}'.",
    )
    parser.add_argument(
        "--unset-params",
        nargs="+",
        default=[],
        help="Param keys removed before fitting (e.g. sde_hyperparam_min sde_hyperparam_max).",
    )
    parser.add_argument(
        "--sampler-json",
        default="{}",
        help="JSON dict merged into the YAML's sampler config.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional eval-fold filter (subset of 1..K-1); default runs all eval folds.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--append", action="store_true", help="Append to the output file instead of replacing it.")
    args = parser.parse_args(argv)

    set_params = json.loads(args.set_json)
    sampler_overrides = json.loads(args.sampler_json)
    if not isinstance(set_params, dict) or not isinstance(sampler_overrides, dict):
        raise SystemExit("--set-json and --sampler-json must be JSON objects.")
    if not set_params and not args.unset_params and not sampler_overrides:
        raise SystemExit("No overrides given; use benchmarks.tuning.evaluate for plain re-evaluation.")

    _setup_logging()
    evaluate_with_overrides(
        args.tuned_yaml,
        label=args.label,
        set_params=set_params,
        unset_params=args.unset_params,
        sampler_overrides=sampler_overrides,
        fold_ids=set(args.folds) if args.folds else None,
        results_dir=args.results_dir,
        append=args.append,
    )


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=("<green>{time:HH:mm:ss}</green> " "<level>{level: <7}</level> " "<cyan>[{extra}]</cyan> " "{message}"),
    )


if __name__ == "__main__":
    main()
