"""Run an eval-only sampler cost sweep from already tuned DiffGBM YAMLs.

This script does not retune hyperparameters. It loads per-dataset best-parameter
YAMLs, refits the selected model on each evaluation fold, and evaluates multiple
sampler step counts from the same fitted model.

Usage:
    python -m benchmarks.scripts.run_sampler_cost_sweep --dry-run
    python -m benchmarks.scripts.run_sampler_cost_sweep --datasets yacht diabetes
"""

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
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
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.splits import build_splits
from benchmarks.tuning.study import _space_fingerprint
from benchmarks.variants import Variant

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/results/sampler_cost_sweep/eval"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "sampler_cost_sweep.jsonl"


@dataclass(frozen=True)
class SourceSpec:
    family: str
    space_name: str
    yaml_dir: Path
    sampler_labels: tuple[str, ...]
    display: str


@dataclass(frozen=True)
class SamplerSpec:
    label: str
    sampler: dict[str, Any]


SAMPLERS: dict[str, SamplerSpec] = {
    "published_euler_15": SamplerSpec(
        "published_euler_15",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 15, "method": "euler", "pf_ode": False},
    ),
    "published_euler_25": SamplerSpec(
        "published_euler_25",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 25, "method": "euler", "pf_ode": False},
    ),
    "published_euler_50": SamplerSpec(
        "published_euler_50",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 50, "method": "euler", "pf_ode": False},
    ),
    "published_euler_100": SamplerSpec(
        "published_euler_100",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 100, "method": "euler", "pf_ode": False},
    ),
    "score_plus_heun_5": SamplerSpec(
        "score_plus_heun_5",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 5, "method": "heun", "pf_ode": True},
    ),
    "score_plus_heun_10": SamplerSpec(
        "score_plus_heun_10",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 10, "method": "heun", "pf_ode": True},
    ),
    "score_plus_heun_15": SamplerSpec(
        "score_plus_heun_15",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 15, "method": "heun", "pf_ode": True},
    ),
    "score_plus_heun_25": SamplerSpec(
        "score_plus_heun_25",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 25, "method": "heun", "pf_ode": True},
    ),
    "score_plus_heun_50": SamplerSpec(
        "score_plus_heun_50",
        {"n_samples": 200, "n_parallel": 20, "n_steps": 50, "method": "heun", "pf_ode": True},
    ),
    "fm_ode_3": SamplerSpec(
        "fm_ode_3",
        {
            "n_samples": 200,
            "n_parallel": 20,
            "n_steps": 3,
            "method": "heun",
            "pf_ode": False,
            "velocity_stochasticity": 0.0,
            "velocity_stochasticity_schedule": "linear",
        },
    ),
    "fm_ode_5": SamplerSpec(
        "fm_ode_5",
        {
            "n_samples": 200,
            "n_parallel": 20,
            "n_steps": 5,
            "method": "heun",
            "pf_ode": False,
            "velocity_stochasticity": 0.0,
            "velocity_stochasticity_schedule": "linear",
        },
    ),
    "fm_ode_10": SamplerSpec(
        "fm_ode_10",
        {
            "n_samples": 200,
            "n_parallel": 20,
            "n_steps": 10,
            "method": "heun",
            "pf_ode": False,
            "velocity_stochasticity": 0.0,
            "velocity_stochasticity_schedule": "linear",
        },
    ),
    "fm_ode_25": SamplerSpec(
        "fm_ode_25",
        {
            "n_samples": 200,
            "n_parallel": 20,
            "n_steps": 25,
            "method": "heun",
            "pf_ode": False,
            "velocity_stochasticity": 0.0,
            "velocity_stochasticity_schedule": "linear",
        },
    ),
    "fm_sde_25": SamplerSpec(
        "fm_sde_25",
        {
            "n_samples": 200,
            "n_parallel": 20,
            "n_steps": 25,
            "method": "heun",
            "pf_ode": False,
            "velocity_stochasticity": 1.0,
            "velocity_stochasticity_schedule": "linear",
        },
    ),
}


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        family="published",
        display="Published score Euler SDE",
        space_name="treeffuser_published",
        yaml_dir=REPO_ROOT / "benchmarks/results/tuning/best_params",
        sampler_labels=("published_euler_15", "published_euler_25", "published_euler_50", "published_euler_100"),
    ),
    SourceSpec(
        family="score_plus",
        display="Score+ Heun PF-ODE",
        space_name="treeffuser_score_plus",
        yaml_dir=REPO_ROOT / "benchmarks/results/tuning/best_params",
        sampler_labels=(
            "score_plus_heun_5",
            "score_plus_heun_10",
            "score_plus_heun_15",
            "score_plus_heun_25",
            "score_plus_heun_50",
            "published_euler_50",
        ),
    ),
    SourceSpec(
        family="score_plus_noresid",
        display="Score+ (no residualizer) Heun PF-ODE",
        space_name="treeffuser_score_plus_noresid",
        yaml_dir=REPO_ROOT / "benchmarks/results/tuning/best_params",
        sampler_labels=("published_euler_50", "score_plus_heun_50"),
    ),
    SourceSpec(
        family="fm_noresid",
        display="FM-VP (no residualizer)",
        space_name="treeffuser_fm_noresid",
        yaml_dir=REPO_ROOT / "benchmarks/results/tuning/best_params",
        sampler_labels=("fm_ode_5", "fm_ode_10", "fm_ode_25"),
    ),
    SourceSpec(
        family="fm_vp",
        display="FM-VP",
        space_name="treeffuser_fm",
        yaml_dir=REPO_ROOT / "benchmarks/results/tuning/best_params",
        sampler_labels=("fm_ode_3", "fm_ode_5", "fm_ode_10", "fm_ode_25", "fm_sde_25"),
    ),
    SourceSpec(
        family="fm_linear",
        display="FM-linear",
        space_name="ablate_fm_linear_resid_ode5",
        yaml_dir=REPO_ROOT / "benchmarks/results/mechanism_ablation/best_params",
        sampler_labels=("fm_ode_3", "fm_ode_5", "fm_ode_10", "fm_ode_25", "fm_sde_25"),
    ),
)


def run_sweep(
    *,
    output_path: Path,
    dataset_names: set[str] | None,
    family_names: set[str] | None,
    sampler_names: set[str] | None,
    fold_ids: set[int] | None,
    strict_space_match: bool,
    append: bool,
    dry_run: bool,
) -> None:
    jobs = list(_iter_jobs(dataset_names=dataset_names, family_names=family_names))
    jobs = [
        (source, yaml_path, labels) for source, yaml_path in jobs if (labels := _select_labels(source, sampler_names))
    ]
    if dry_run:
        n_folds = len(fold_ids) if fold_ids is not None else 5
        for source, yaml_path, labels in jobs:
            logger.info(
                "Would evaluate dataset={} family={} source_space={} samplers=[{}]",
                yaml_path.name.split("__", maxsplit=1)[0],
                source.family,
                source.space_name,
                ", ".join(labels),
            )
        row_count = sum(n_folds * len(labels) for _, _, labels in jobs)
        logger.info("Dry run: {} tuned YAMLs, {} output rows", len(jobs), row_count)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        output_path.unlink(missing_ok=True)

    provenance = get_provenance()
    total_rows = 0
    with output_path.open("a") as file:
        for source, yaml_path, labels in jobs:
            total_rows += _evaluate_source_yaml(
                source=source,
                tuned_yaml=yaml_path,
                sampler_labels=labels,
                fold_ids=fold_ids,
                strict_space_match=strict_space_match,
                provenance=provenance,
                file=file,
            )
    logger.success("Wrote {} rows to {}", total_rows, output_path)


def _select_labels(source: SourceSpec, sampler_names: set[str] | None) -> tuple[str, ...]:
    if sampler_names is None:
        return source.sampler_labels
    return tuple(label for label in source.sampler_labels if label in sampler_names)


def _iter_jobs(
    *,
    dataset_names: set[str] | None,
    family_names: set[str] | None,
) -> Iterable[tuple[SourceSpec, Path]]:
    for source in SOURCES:
        if family_names is not None and source.family not in family_names:
            continue
        for yaml_path in sorted(source.yaml_dir.glob(f"*__{source.space_name}.yaml")):
            dataset = yaml_path.name.rsplit(f"__{source.space_name}.yaml", maxsplit=1)[0]
            if dataset_names is None or dataset in dataset_names:
                yield source, yaml_path


def _evaluate_source_yaml(
    *,
    source: SourceSpec,
    tuned_yaml: Path,
    sampler_labels: tuple[str, ...],
    fold_ids: set[int] | None,
    strict_space_match: bool,
    provenance: dict[str, Any],
    file,
) -> int:
    payload = yaml.safe_load(tuned_yaml.read_text())
    protocol = payload["protocol"]
    if protocol["space_name"] != source.space_name:
        raise ValueError(f"{tuned_yaml} has space_name={protocol['space_name']!r}, expected {source.space_name!r}")
    if source.space_name not in SPACES:
        raise ValueError(f"Unknown source space {source.space_name!r}")
    space = SPACES[source.space_name]

    if strict_space_match:
        current_fp = _space_fingerprint(space)
        if current_fp != protocol["space_fingerprint"]:
            raise RuntimeError(
                f"Search space {source.space_name!r} changed since {tuned_yaml} was written "
                f"(stored={protocol['space_fingerprint']!r}, current={current_fp!r})."
            )

    dataset_name = protocol["dataset"]
    bound = logger.bind(dataset=dataset_name, family=source.family)
    bundle = make_dataset(
        name=dataset_name,
        n_train=protocol["n_train"],
        n_test=protocol["n_test"],
        seed=protocol["master_seed"],
        x_dim=protocol["x_dim"],
    )
    X = np.concatenate([bundle.X_train, bundle.X_test], axis=0)
    y = np.concatenate([bundle.y_train, bundle.y_test], axis=0)
    splits = build_splits(X, y, master_seed=protocol["master_seed"], n_folds=protocol["n_folds"], name=dataset_name)

    params = payload["params"]
    variant = Variant(name=source.space_name, params=params, model=space.model)
    eval_fold_ids = [fold_id for fold_id in range(protocol["n_folds"]) if fold_id != 0]
    rows_written = 0
    for eval_fold, fold in zip(eval_fold_ids, splits.eval_folds, strict=True):
        if fold_ids is not None and eval_fold not in fold_ids:
            continue
        X_train, y_train = splits.slice(fold.train_idx)
        X_test, y_test = splits.slice(fold.test_idx)

        model = variant.make_model(seed=protocol["master_seed"] + 10_000)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        clim = crps_climatology(y_train=y_train, y_true=y_test)

        for sampler_label in sampler_labels:
            sampler_spec = SAMPLERS[sampler_label]
            sampler = sampler_spec.sampler
            t0 = time.perf_counter()
            y_samples = model.sample(
                X_test,
                seed=protocol["master_seed"] + 20_000 + eval_fold,
                **_sample_kwargs(sampler),
            )
            sample_time = time.perf_counter() - t0
            metrics = evaluate_samples(y_samples=y_samples, y_true=y_test, X_test=X_test)

            row: dict[str, Any] = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "dataset": dataset_name,
                "family": source.family,
                "display": source.display,
                "source_space": source.space_name,
                "source_yaml": str(tuned_yaml.relative_to(REPO_ROOT)),
                "sampler_label": sampler_label,
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
            file.write(json.dumps(_json_safe(row), sort_keys=True))
            file.write("\n")
            file.flush()
            rows_written += 1
            bound.info(
                "fold={} sampler={} CRPS={:.4f} skill={:.3f} sample={:.1f}s",
                eval_fold,
                sampler_label,
                metrics["crps"],
                row["crps_skill_score"],
                sample_time,
            )
    return rows_written


def _sample_kwargs(sampler: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_samples": sampler["n_samples"],
        "n_parallel": sampler["n_parallel"],
        "n_steps": sampler["n_steps"],
        "verbose": False,
        "sampler_method": sampler["method"],
        "pf_ode": sampler["pf_ode"],
        "velocity_stochasticity": sampler.get("velocity_stochasticity", 0.0),
        "velocity_stochasticity_schedule": sampler.get("velocity_stochasticity_schedule", "linear"),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run eval-only sampler step/cost sweep from tuned YAMLs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset-name filter.")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=[source.family for source in SOURCES],
        default=None,
        help="Optional family filter.",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=sorted(SAMPLERS),
        default=None,
        help="Optional sampler-label filter (intersected with each source's labels).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional eval-fold filter (subset of 1..K-1); default runs all eval folds.",
    )
    parser.add_argument("--append", action="store_true", help="Append to output instead of replacing it.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without fitting models.")
    parser.add_argument(
        "--no-strict-space",
        dest="strict_space_match",
        action="store_false",
        help="Skip search-space fingerprint checks.",
    )
    args = parser.parse_args(argv)

    run_sweep(
        output_path=args.output,
        dataset_names=set(args.datasets) if args.datasets else None,
        family_names=set(args.families) if args.families else None,
        sampler_names=set(args.samplers) if args.samplers else None,
        fold_ids=set(args.folds) if args.folds else None,
        strict_space_match=args.strict_space_match,
        append=args.append,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
