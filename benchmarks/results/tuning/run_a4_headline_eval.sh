#!/usr/bin/env bash
# A4 headline arms: re-evaluate published / score_flex_sde / fm on the ten-dataset
# suite so every headline row carries per-point CRPS/PIT (B2).
#
# --no-strict-space is safe here: the eval path builds the model from the params
# dict stored in the YAML, so nothing from the SearchSpace but `model` is read.
# Verified for these arms: `fixed` and `sampler` are byte-identical to the state
# they were tuned under, and `model` changed only treeffuser -> diffgbm.
#
# Resumable: each start re-derives the outstanding cells, so an interrupted run
# is continued by invoking this script again. A cell counts as done only once its
# per-point .npz exists and its JSONL holds all five eval folds; a cell cut
# mid-run is redone from scratch (evaluate_tuned_yaml truncates its own output).
set -euo pipefail

cd "$(dirname "$0")/../../.."

LIST=benchmarks/results/tuning/a4_headline_eval_list.txt
LOG=benchmarks/results/tuning/a4_headline_eval.log
TODO=$(mktemp)
trap 'rm -f "$TODO"' EXIT

pixi run -e bench python - "$LIST" "$TODO" <<'PY'
import json, sys
from pathlib import Path

list_path, todo_path = Path(sys.argv[1]), Path(sys.argv[2])
eval_dir = Path("benchmarks/results/tuning/eval")

outstanding = []
for line in list_path.read_text().split():
    stem = Path(line).stem
    rows_path = eval_dir / f"{stem}.jsonl"
    npz_path = eval_dir / "per_point" / f"{stem}.npz"
    n_folds = sum(1 for _ in rows_path.open()) if rows_path.exists() else 0
    if not (npz_path.exists() and n_folds == 5):
        outstanding.append(line)

todo_path.write_text("\n".join(outstanding) + ("\n" if outstanding else ""))
print(f"{len(outstanding)} of {len(list_path.read_text().split())} cells outstanding")
PY

if [ ! -s "$TODO" ]; then
  echo "nothing to do"
  exit 0
fi

xargs pixi run -e bench python -m benchmarks.tuning.evaluate \
  --results-dir benchmarks/results/tuning/eval \
  --no-strict-space \
  < "$TODO" \
  >> "$LOG" 2>&1
echo "done"
