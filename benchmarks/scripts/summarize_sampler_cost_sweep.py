"""Summarize and plot the tuned sampler-cost sweep.

Reads `benchmarks/results/sampler_cost_sweep/eval/sampler_cost_sweep.jsonl`,
then writes a compact markdown table and a two-panel speed/quality figure.

Usage:
    python -m benchmarks.scripts.summarize_sampler_cost_sweep
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "benchmarks/results/sampler_cost_sweep/eval/sampler_cost_sweep.jsonl"
SUMMARY_PATH = REPO_ROOT / "benchmarks/results/selected/sampler_cost_sweep.md"
FIGURE_PATH = REPO_ROOT / "benchmarks/results/sampler_cost_sweep/sampler_cost_tradeoff.pdf"
PNG_PATH = REPO_ROOT / "benchmarks/results/sampler_cost_sweep/sampler_cost_tradeoff.png"

METRICS = [
    "crps",
    "crps_skill_score",
    "sample_time",
    "quantile_mace",
    "interval_90_abs_coverage_error",
    "interval_95_abs_coverage_error",
]

FAMILY_ORDER = {
    "published": 0,
    "score_plus": 1,
    "fm_vp": 2,
    "fm_linear": 3,
}

DISPLAY = {
    "published": "Published Euler SDE",
    "score_plus": "Score+ PF-ODE",
    "fm_vp": "FM-VP ODE",
    "fm_linear": "FM-linear ODE",
}

COLORS = {
    "published": "#4C78A8",
    "score_plus": "#F58518",
    "fm_vp": "#54A24B",
    "fm_linear": "#B279A2",
}

MARKERS = {
    "published": "o",
    "score_plus": "s",
    "fm_vp": "^",
    "fm_linear": "D",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dataset_condition_means(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["dataset"], row["family"], row["sampler_label"])].append(row)

    missing_folds = [(key, len(value)) for key, value in buckets.items() if len(value) != 5]
    if missing_folds:
        raise RuntimeError(f"Expected 5 folds for every dataset/family/sampler, got: {missing_folds}")

    means: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, fold_rows in buckets.items():
        sampler = fold_rows[0]["sampler"]
        means[key] = {metric: sum(row[metric] for row in fold_rows) / len(fold_rows) for metric in METRICS}
        means[key].update(
            {
                "n_steps": sampler["n_steps"],
                "method": sampler["method"],
                "pf_ode": sampler["pf_ode"],
                "velocity_stochasticity": sampler.get("velocity_stochasticity", 0.0),
            }
        )

    datasets = sorted({dataset for dataset, _, _ in means})
    for dataset in datasets:
        best_crps = min(value["crps"] for (ds, _, _), value in means.items() if ds == dataset)
        for key, value in means.items():
            if key[0] == dataset:
                value["rel_crps"] = value["crps"] / best_crps

    return means


def aggregate(means: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (_, family, sampler_label), value in means.items():
        buckets[(family, sampler_label)].append(value)

    rows = []
    for (family, sampler_label), values in buckets.items():
        out = {
            "family": family,
            "sampler_label": sampler_label,
            "display": DISPLAY[family],
            "n_steps": values[0]["n_steps"],
            "method": values[0]["method"],
            "pf_ode": values[0]["pf_ode"],
            "velocity_stochasticity": values[0]["velocity_stochasticity"],
        }
        for metric in [*METRICS, "rel_crps"]:
            out[metric] = sum(value[metric] for value in values) / len(values)
        rows.append(out)

    return sorted(rows, key=lambda row: (FAMILY_ORDER[row["family"]], row["velocity_stochasticity"], row["n_steps"]))


def render_markdown(rows: list[dict[str, Any]], n_raw_rows: int) -> str:
    lines = [
        "# Sampler cost sweep",
        "",
        f"Source: `{INPUT_PATH.relative_to(REPO_ROOT)}` ({n_raw_rows} fold-level rows). "
        "Each displayed row averages folds 1--5 within each dataset, then averages over the ten UCI datasets. "
        "rel-CRPS is normalized by the best sampler-cost row on each dataset.",
        "",
        "## Aggregate metrics",
        "",
        "| Family | Sampler | Steps | CRPSS | rel-CRPS | q-MACE | \\|cE\\|@90 | \\|cE\\|@95 | sample s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        sampler = "SDE" if row["velocity_stochasticity"] > 0 else ("PF-ODE" if row["pf_ode"] else "ODE")
        if row["family"] == "published":
            sampler = "Euler SDE"
        lines.append(
            f"| {DISPLAY[row['family']]} | {sampler} | {row['n_steps']} | "
            f"{row['crps_skill_score']:.3f} | {row['rel_crps']:.3f} | "
            f"{row['quantile_mace']:.3f} | {row['interval_90_abs_coverage_error']:.3f} | "
            f"{row['interval_95_abs_coverage_error']:.3f} | {row['sample_time']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- FM ODE is effectively converged by 3--5 steps on aggregate CRPS; extra ODE steps mainly add latency.",
            "- Score+ needs about 15--25 Heun PF-ODE steps for the calibration metrics to settle; 5 steps is not competitive.",
            "- Published Euler SDE improves gradually with more steps but remains much slower at comparable CRPS.",
            "- FM-SDE at 25 steps improves VP-FM 95% coverage error, but worsens q-MACE, 90% coverage error, and CRPS.",
            "- Linear FM remains the tuned CRPS corner in this sweep; VP-FM remains the faster headline FM row.",
            "",
            f"Figure: `{FIGURE_PATH.relative_to(REPO_ROOT)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    panels = [
        ("rel_crps", "rel-CRPS (lower is better)"),
        ("interval_95_abs_coverage_error", "|cE|@95 (lower is better)"),
    ]

    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for family in FAMILY_ORDER:
            family_rows = [row for row in rows if row["family"] == family]
            ode_rows = [row for row in family_rows if row["velocity_stochasticity"] == 0.0]
            sde_rows = [row for row in family_rows if row["velocity_stochasticity"] > 0.0]
            ode_rows.sort(key=lambda row: row["n_steps"])
            ax.plot(
                [row["sample_time"] for row in ode_rows],
                [row[metric] for row in ode_rows],
                marker=MARKERS[family],
                color=COLORS[family],
                linewidth=2.6,
                markersize=8.0,
                label=DISPLAY[family],
                zorder=3,
            )
            for row in ode_rows:
                ax.annotate(
                    str(row["n_steps"]),
                    (row["sample_time"], row[metric]),
                    textcoords="offset points",
                    xytext=(5, 6),
                    fontsize=10.5,
                    fontweight="bold",
                    color=COLORS[family],
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                )
            for row in sde_rows:
                ax.scatter(
                    [row["sample_time"]],
                    [row[metric]],
                    marker="x",
                    s=90,
                    linewidths=2.6,
                    color=COLORS[family],
                    zorder=4,
                )
                ax.annotate(
                    "SDE25",
                    (row["sample_time"], row[metric]),
                    textcoords="offset points",
                    xytext=(6, -13),
                    fontsize=9.5,
                    color=COLORS[family],
                    zorder=4,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Sample-generation time (s, log scale)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=11, loc="best")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH)
    fig.savefig(PNG_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = load_rows(INPUT_PATH)
    means = dataset_condition_means(rows)
    aggregate_rows = aggregate(means)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(render_markdown(aggregate_rows, len(rows)))
    plot(aggregate_rows)
    print(f"Wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {FIGURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
