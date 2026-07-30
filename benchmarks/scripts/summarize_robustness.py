"""Summarize robustness checks from the tuned evaluation artifacts.

Reads `benchmarks/results/tuning/eval/*.jsonl` and writes rank, winner,
standard-error, size-group, calibration, and rank-test summaries for the paper.

Usage:
    python -m benchmarks.scripts.summarize_robustness
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata
from scipy.stats import studentized_range
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "benchmarks/results/tuning/eval"
OUTPUT_PATH = REPO_ROOT / "benchmarks/results/selected/robustness_summary.md"

SPACES = [
    "treeffuser_published",
    "treeffuser_score_flex_sde",
    "treeffuser_fm",
    "qreg_lightgbm",
    "catboost_uncertainty",
    "ngboost",
    "deep_ensemble",
    "ibug",
    "card",
]

TREEFFUSER_SPACES = [
    "treeffuser_published",
    "treeffuser_score_flex_sde",
    "treeffuser_fm",
]

DISPLAY = {
    "treeffuser_published": "Treeffuser (published)",
    "treeffuser_score_flex_sde": "DiffGBM (score-flex)",
    "treeffuser_score_plus": "DiffGBM-score+",
    "treeffuser_score_flex": "DiffGBM-score-flex-ODE",
    "treeffuser_fm": "DiffGBM-FM",
    "qreg_lightgbm": "QReg-LightGBM",
    "catboost_uncertainty": "CatBoost-unc.",
    "ngboost": "NGBoost",
    "deep_ensemble": "Deep ensemble",
    "ibug": "iBUG",
    "card": "CARD-style diffusion",
}

METRICS = [
    "crps",
    "crps_skill_score",
    "sample_time",
    "quantile_mace",
    "interval_50_abs_coverage_error",
    "interval_90_abs_coverage_error",
    "interval_95_abs_coverage_error",
    "pit_ks_pvalue",
    "n_train_fold",
]


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open() as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def dataset_means(rows: list[dict[str, Any]]) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["dataset"], row["space"])].append(row)

    datasets = sorted({dataset for dataset, _ in buckets})
    expected = {(dataset, space) for dataset in datasets for space in SPACES}
    missing = sorted(expected - set(buckets))
    incomplete = sorted((key, len(value)) for key, value in buckets.items() if len(value) != 5)
    if missing:
        raise RuntimeError(f"Missing tuned evaluation pairs: {missing}")
    if incomplete:
        raise RuntimeError(f"Expected 5 folds for every dataset/space pair, got: {incomplete}")

    means: dict[tuple[str, str], dict[str, float]] = {}
    for key, fold_rows in buckets.items():
        means[key] = {}
        for metric in METRICS:
            values = [row[metric] for row in fold_rows]
            means[key][metric] = float(np.mean(values))
        means[key]["pit_pass_rate"] = float(np.mean([row["pit_ks_pvalue"] > 0.05 for row in fold_rows]))

    for dataset in datasets:
        best_crps = min(means[(dataset, space)]["crps"] for space in SPACES)
        for space in SPACES:
            means[(dataset, space)]["rel_crps"] = means[(dataset, space)]["crps"] / best_crps

    return datasets, means


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def se(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def rank_values(values: list[float], lower_is_better: bool) -> list[float]:
    scores = values if lower_is_better else [-value for value in values]
    return [float(rank) for rank in rankdata(scores, method="average")]


def aggregate_rows(datasets: list[str], means: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for space in SPACES:
        crpss = [means[(dataset, space)]["crps_skill_score"] for dataset in datasets]
        rel_crps = [means[(dataset, space)]["rel_crps"] for dataset in datasets]
        q_mace = [means[(dataset, space)]["quantile_mace"] for dataset in datasets]
        ce90 = [means[(dataset, space)]["interval_90_abs_coverage_error"] for dataset in datasets]
        ce95 = [means[(dataset, space)]["interval_95_abs_coverage_error"] for dataset in datasets]
        pit_pass = [means[(dataset, space)]["pit_pass_rate"] for dataset in datasets]
        rows.append(
            {
                "space": space,
                "crpss": mean(crpss),
                "crpss_se": se(crpss),
                "rel_crps": mean(rel_crps),
                "rel_crps_se": se(rel_crps),
                "q_mace": mean(q_mace),
                "ce90": mean(ce90),
                "ce95": mean(ce95),
                "pit_pass": mean(pit_pass),
            }
        )
    return rows


def rank_rows(datasets: list[str], means: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, float | str]]:
    crps_ranks: dict[str, list[float]] = {space: [] for space in SPACES}
    crpss_ranks: dict[str, list[float]] = {space: [] for space in SPACES}
    calibration_ranks: dict[str, list[float]] = {space: [] for space in SPACES}

    for dataset in datasets:
        crps_values = [means[(dataset, space)]["crps"] for space in SPACES]
        crpss_values = [means[(dataset, space)]["crps_skill_score"] for space in SPACES]
        calibration_values = [means[(dataset, space)]["interval_95_abs_coverage_error"] for space in SPACES]
        for space, rank in zip(SPACES, rank_values(crps_values, lower_is_better=True), strict=True):
            crps_ranks[space].append(rank)
        for space, rank in zip(SPACES, rank_values(crpss_values, lower_is_better=False), strict=True):
            crpss_ranks[space].append(rank)
        for space, rank in zip(SPACES, rank_values(calibration_values, lower_is_better=True), strict=True):
            calibration_ranks[space].append(rank)

    rows = []
    for space in SPACES:
        rows.append(
            {
                "space": space,
                "crps_rank": mean(crps_ranks[space]),
                "crpss_rank": mean(crpss_ranks[space]),
                "ce95_rank": mean(calibration_ranks[space]),
            }
        )
    return sorted(rows, key=lambda row: float(row["crps_rank"]))


def winner_rows(datasets: list[str], means: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, str | float]]:
    rows = []
    for dataset in datasets:
        raw_winner = min(SPACES, key=lambda space: means[(dataset, space)]["crps"])
        treeffuser_winner = max(
            TREEFFUSER_SPACES,
            key=lambda space: means[(dataset, space)]["crps_skill_score"],
        )
        rows.append(
            {
                "dataset": dataset,
                "raw_winner": raw_winner,
                "raw_crps": means[(dataset, raw_winner)]["crps"],
                "treeffuser_winner": treeffuser_winner,
                "treeffuser_crpss": means[(dataset, treeffuser_winner)]["crps_skill_score"],
            }
        )
    return rows


def size_group_rows(
    datasets: list[str], means: dict[tuple[str, str], dict[str, float]]
) -> list[dict[str, float | str | int]]:
    groups = {
        "small": [dataset for dataset in datasets if means[(dataset, SPACES[0])]["n_train_fold"] <= 1_000],
        "medium": [dataset for dataset in datasets if 1_000 < means[(dataset, SPACES[0])]["n_train_fold"] <= 10_000],
        "large": [dataset for dataset in datasets if means[(dataset, SPACES[0])]["n_train_fold"] > 10_000],
    }
    rows: list[dict[str, float | str | int]] = []
    for group, group_datasets in groups.items():
        for space in SPACES:
            crpss_values = [means[(dataset, space)]["crps_skill_score"] for dataset in group_datasets]
            rel_crps_values = [means[(dataset, space)]["rel_crps"] for dataset in group_datasets]
            rows.append(
                {
                    "group": group,
                    "n_datasets": len(group_datasets),
                    "space": space,
                    "crpss": mean(crpss_values),
                    "rel_crps": mean(rel_crps_values),
                }
            )
    return rows


def friedman_nemenyi(
    datasets: list[str],
    means: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    rank_matrix = []
    for dataset in datasets:
        values = [means[(dataset, space)]["crps"] for space in SPACES]
        rank_matrix.append(rank_values(values, lower_is_better=True))
    ranks = np.array(rank_matrix)
    statistic, pvalue = friedmanchisquare(*[ranks[:, column] for column in range(len(SPACES))])
    q_alpha = float(studentized_range.ppf(0.95, len(SPACES), np.inf) / math.sqrt(2))
    critical_difference = q_alpha * math.sqrt(len(SPACES) * (len(SPACES) + 1) / (6 * len(datasets)))
    mean_ranks = {space: float(np.mean(ranks[:, index])) for index, space in enumerate(SPACES)}
    significant_pairs = []
    for left_index, left in enumerate(SPACES):
        for right in SPACES[left_index + 1 :]:
            diff = abs(mean_ranks[left] - mean_ranks[right])
            if diff > critical_difference:
                significant_pairs.append((left, right, diff))
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "critical_difference": float(critical_difference),
        "mean_ranks": mean_ranks,
        "significant_pairs": significant_pairs,
    }


def wilcoxon_rows(
    datasets: list[str], means: dict[tuple[str, str], dict[str, float]]
) -> list[dict[str, float | str | int]]:
    pairs = [
        ("score-flex vs published", "treeffuser_score_flex_sde", "treeffuser_published", "less"),
        ("FM vs published", "treeffuser_fm", "treeffuser_published", "less"),
        ("score-flex vs FM", "treeffuser_score_flex_sde", "treeffuser_fm", "two-sided"),
    ]
    rows = []
    for label, left, right, alternative in pairs:
        left_values = np.array([means[(dataset, left)]["crps"] for dataset in datasets])
        right_values = np.array([means[(dataset, right)]["crps"] for dataset in datasets])
        differences = left_values - right_values
        result = wilcoxon(left_values, right_values, alternative=alternative, zero_method="wilcox")
        rows.append(
            {
                "pair": label,
                "alternative": alternative,
                "wins_left": int(np.sum(differences < 0)),
                "wins_right": int(np.sum(differences > 0)),
                "median_difference": float(np.median(differences)),
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
            }
        )
    return rows


def render_markdown(
    datasets: list[str],
    means: dict[tuple[str, str], dict[str, float]],
    rows: list[dict[str, Any]],
) -> str:
    aggregate = aggregate_rows(datasets, means)
    ranks = rank_rows(datasets, means)
    winners = winner_rows(datasets, means)
    groups = size_group_rows(datasets, means)
    friedman = friedman_nemenyi(datasets, means)
    wilcoxon = wilcoxon_rows(datasets, means)

    lines = [
        "# Robustness summary",
        "",
        f"Source: `{INPUT_DIR.relative_to(REPO_ROOT)}/*.jsonl` ({len(rows)} fold-level rows). "
        "Each displayed value first averages folds 1--5 within each dataset/model family, "
        "then averages over datasets unless stated otherwise. rel-CRPS is normalized by "
        "the best displayed family on each dataset.",
        "",
        "## Aggregate metrics with dataset standard errors",
        "",
        "| Model | CRPSS | SE | rel-CRPS | SE | q-MACE | \\|cE\\|@90 | \\|cE\\|@95 | PIT pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate, key=lambda item: float(item["rel_crps"])):
        space = str(row["space"])
        lines.append(
            f"| {DISPLAY[space]} | {row['crpss']:.3f} | {row['crpss_se']:.3f} | "
            f"{row['rel_crps']:.3f} | {row['rel_crps_se']:.3f} | "
            f"{row['q_mace']:.3f} | {row['ce90']:.3f} | {row['ce95']:.3f} | "
            f"{row['pit_pass']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Mean ranks across datasets",
            "",
            "| Model | CRPS rank | CRPSS rank | \\|cE\\|@95 rank |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ranks:
        space = str(row["space"])
        lines.append(
            f"| {DISPLAY[space]} | {row['crps_rank']:.2f} | " f"{row['crpss_rank']:.2f} | {row['ce95_rank']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Friedman and Nemenyi rank diagnostic",
            "",
            f"Friedman test on per-dataset CRPS ranks over {len(SPACES)} families and "
            f"{len(datasets)} datasets: chi^2={friedman['statistic']:.2f}, "
            f"p={friedman['pvalue']:.3f}. Nemenyi critical difference at alpha=0.05: "
            f"{friedman['critical_difference']:.2f} mean-rank points.",
        ]
    )
    significant_pairs = friedman["significant_pairs"]
    if significant_pairs:
        lines.append("")
        lines.append("| Pair | mean-rank gap |")
        lines.append("|---|---:|")
        for left, right, diff in significant_pairs:
            lines.append(f"| {DISPLAY[left]} vs {DISPLAY[right]} | {diff:.2f} |")
    else:
        lines.append("No pair exceeds the Nemenyi critical difference.")

    lines.extend(
        [
            "",
            "## Paired Wilcoxon tests for DiffGBM headline rows",
            "",
            "| Pair | Alt. | left wins | right wins | median delta CRPS | W | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in wilcoxon:
        lines.append(
            f"| {row['pair']} | {row['alternative']} | {row['wins_left']} | "
            f"{row['wins_right']} | {row['median_difference']:+.4f} | "
            f"{row['statistic']:.1f} | {row['pvalue']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset winners",
            "",
            "| Dataset | Raw-CRPS winner | CRPS | DiffGBM CRPSS winner | CRPSS |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in winners:
        raw_winner = str(row["raw_winner"])
        treeffuser_winner = str(row["treeffuser_winner"])
        lines.append(
            f"| {row['dataset']} | {DISPLAY[raw_winner]} | {row['raw_crps']:.6g} | "
            f"{DISPLAY[treeffuser_winner]} | {row['treeffuser_crpss']:.3f} |"
        )

    counts: dict[str, int] = defaultdict(int)
    for row in winners:
        counts[str(row["raw_winner"])] += 1
    lines.extend(["", "Raw-CRPS winner counts:"])
    for space, count in sorted(counts.items(), key=lambda item: (-item[1], DISPLAY[item[0]])):
        lines.append(f"- {DISPLAY[space]}: {count}")

    lines.extend(
        [
            "",
            "## Dataset-size groups",
            "",
            "| Group | n datasets | Model | CRPSS | rel-CRPS |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in groups:
        space = str(row["space"])
        lines.append(
            f"| {row['group']} | {row['n_datasets']} | {DISPLAY[space]} | "
            f"{row['crpss']:.3f} | {row['rel_crps']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- DiffGBM (score-flex) leads mean CRPS rank and takes the plurality of raw-CRPS wins; the remaining per-dataset winners are DiffGBM-FM and the deep ensemble.",
            "- The DiffGBM rows lead the cross-dataset CRPSS/rel-CRPS aggregate, but the dataset standard errors are large relative to the small aggregate gaps.",
            "- DiffGBM-FM is the most robust calibration row by PIT pass rate and interval coverage error; DiffGBM (score-flex) is strongest on mean CRPS rank and rel-CRPS.",
            "- The size split shows DiffGBM (score-flex) leading every size group on CRPSS/rel-CRPS, most decisively on the large group; DiffGBM-FM is competitive on small/medium but weak on large.",
            "- Friedman/Nemenyi is useful as a rank robustness check, but with eleven datasets it should be treated as supporting context rather than a headline significance claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows(INPUT_DIR)
    datasets, means = dataset_means(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_markdown(datasets, means, rows))
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Rows: {len(rows)}; datasets: {len(datasets)}; families: {len(SPACES)}")


if __name__ == "__main__":
    main()
