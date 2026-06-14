"""Summarize the tuned Treeffuser mechanism ablation.

Reads the fold-level JSONL files emitted by
`benchmarks/configs/mechanism_ablation_manifest.yaml` and writes a compact
markdown summary for paper drafting.

Usage:
    python -m benchmarks.scripts.summarize_mechanism_ablation
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "benchmarks/results/mechanism_ablation/eval"
OUTPUT_PATH = REPO_ROOT / "benchmarks/results/selected/mechanism_ablation.md"

VARIANTS = [
    "ablate_score_noise_euler50",
    "ablate_score_noise_heun25",
    "ablate_score_resid_noise_heun25",
    "ablate_score_resid_edm_rawtime_heun25",
    "ablate_score_resid_edm_logstd_uniform_heun25",
    "ablate_score_plus_heun25",
    "ablate_fm_linear_resid_ode5",
    "ablate_fm_trig_resid_ode5",
    "ablate_fm_vp_noresid_ode5",
    "ablate_fm_vp_resid_ode5",
]

DISPLAY = {
    "ablate_score_noise_euler50": "Score noise, Euler-50",
    "ablate_score_noise_heun25": "Score noise, Heun-25",
    "ablate_score_resid_noise_heun25": "+ residualizer",
    "ablate_score_resid_edm_rawtime_heun25": "+ EDM input/target",
    "ablate_score_resid_edm_logstd_uniform_heun25": "+ log-sigma feature",
    "ablate_score_plus_heun25": "Score+ full",
    "ablate_fm_linear_resid_ode5": "FM linear + resid",
    "ablate_fm_trig_resid_ode5": "FM trig + resid",
    "ablate_fm_vp_noresid_ode5": "FM VP no resid",
    "ablate_fm_vp_resid_ode5": "FM VP + resid",
}

METRICS = [
    "crps",
    "crps_skill_score",
    "interval_90_abs_coverage_error",
    "interval_95_abs_coverage_error",
    "quantile_mace",
    "pit_ks_pvalue",
    "sample_time",
    "fit_time",
]


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open() as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def grouped_means(rows: list[dict[str, Any]]) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["dataset"], row["space"])].append(row)

    datasets = sorted({dataset for dataset, _ in buckets})
    expected = {(dataset, variant) for dataset in datasets for variant in VARIANTS}
    missing = sorted(expected - set(buckets))
    incomplete = sorted((key, len(value)) for key, value in buckets.items() if len(value) != 5)
    if missing:
        raise RuntimeError(f"Missing mechanism-ablation pairs: {missing}")
    if incomplete:
        raise RuntimeError(f"Expected 5 folds for every pair, got: {incomplete}")

    means: dict[tuple[str, str], dict[str, float]] = {}
    for key, fold_rows in buckets.items():
        means[key] = {metric: float(np.mean([row[metric] for row in fold_rows])) for metric in METRICS}

    for dataset in datasets:
        best_crps = min(means[(dataset, variant)]["crps"] for variant in VARIANTS)
        for variant in VARIANTS:
            means[(dataset, variant)]["rel_crps"] = means[(dataset, variant)]["crps"] / best_crps

    return datasets, means


def aggregate_row(
    datasets: list[str], means: dict[tuple[str, str], dict[str, float]], variant: str
) -> dict[str, float]:
    out = {
        metric: float(np.mean([means[(dataset, variant)][metric] for dataset in datasets]))
        for metric in [
            "crps_skill_score",
            "rel_crps",
            "crps",
            "quantile_mace",
            "interval_90_abs_coverage_error",
            "interval_95_abs_coverage_error",
            "sample_time",
            "fit_time",
        ]
    }
    out["ks_pass_rate"] = float(np.mean([means[(dataset, variant)]["pit_ks_pvalue"] > 0.05 for dataset in datasets]))
    out["raw_crps_wins"] = float(
        sum(
            means[(dataset, variant)]["crps"] == min(means[(dataset, candidate)]["crps"] for candidate in VARIANTS)
            for dataset in datasets
        )
    )
    return out


def render_markdown(datasets: list[str], means: dict[tuple[str, str], dict[str, float]]) -> str:
    lines: list[str] = []
    lines.append("# Treeffuser mechanism ablation\n")
    lines.append(
        "Source: `benchmarks/results/mechanism_ablation/eval/*.jsonl`. "
        "Each cell averages the five evaluation folds from the fold-0 tuning / "
        "folds-1..5 evaluation protocol. All rows use the same LightGBM "
        "hyperparameter search surface; fixed method choices and bound samplers "
        "differ by row. rel-CRPS is normalized by the best ablation row on each "
        "dataset.\n"
    )

    lines.append("## Aggregate metrics\n")
    lines.append(
        "| Variant | CRPSS | rel-CRPS | CRPS | q-MACE | \\|cE\\|@90 | \\|cE\\|@95 | "
        "KS pass | sample s | fit s | raw wins |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant in VARIANTS:
        row = aggregate_row(datasets, means, variant)
        lines.append(
            f"| {DISPLAY[variant]} | "
            f"{row['crps_skill_score']:.3f} | "
            f"{row['rel_crps']:.3f} | "
            f"{row['crps']:.3f} | "
            f"{row['quantile_mace']:.3f} | "
            f"{row['interval_90_abs_coverage_error']:.3f} | "
            f"{row['interval_95_abs_coverage_error']:.3f} | "
            f"{row['ks_pass_rate']:.2f} | "
            f"{row['sample_time']:.2f} | "
            f"{row['fit_time']:.1f} | "
            f"{int(row['raw_crps_wins'])} |"
        )
    lines.append("")

    lines.append("## Per-dataset raw CRPS winners\n")
    lines.append("| Dataset | Winner | CRPS |")
    lines.append("|---|---|---:|")
    for dataset in datasets:
        winner = min(VARIANTS, key=lambda variant: means[(dataset, variant)]["crps"])
        lines.append(f"| {dataset} | {DISPLAY[winner]} | {means[(dataset, winner)]['crps']:.6g} |")
    lines.append("")

    lines.append("## Per-dataset raw CRPS\n")
    lines.append("| Dataset | " + " | ".join(DISPLAY[variant] for variant in VARIANTS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(VARIANTS)) + "|")
    for dataset in datasets:
        cells = [f"{means[(dataset, variant)]['crps']:.4g}" for variant in VARIANTS]
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Interpretation notes\n")
    lines.append(
        "- Moving the noise-prediction baseline from Euler SDE to Heun PF-ODE without "
        "changing the score parameterization hurts CRPS and PIT pass rate; the "
        "sampler speedup is not a free model-side replacement."
    )
    lines.append(
        "- Residualization is essential for the few-step FM rows: VP without "
        "residualization is the weakest rel-CRPS row despite one diabetes win."
    )
    lines.append(
        "- EDM input/target preconditioning gives the clearest score-side accuracy "
        "and calibration jump; the explicit log-sigma feature is approximately "
        "neutral under uniform t in this tuned protocol."
    )
    lines.append(
        "- Full score+ is the best calibrated score-side row. Among residualized "
        "FM paths, linear has the best aggregate CRPSS, trig is close and wins "
        "two datasets, and VP is fastest. This revises the earlier "
        "fixed-diagnostic path story into a practical path/conditioning "
        "frontier."
    )
    return "\n".join(lines)


def main() -> None:
    rows = load_rows(INPUT_DIR)
    datasets, means = grouped_means(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_markdown(datasets, means) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Rows: {len(rows)}; datasets: {len(datasets)}; variants: {len(VARIANTS)}")


if __name__ == "__main__":
    main()
