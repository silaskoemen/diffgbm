#!/usr/bin/env bash
# Launch the remaining tail (protein, ct_slices) as two concurrent workers.
# Lever 1: TREEFFUSER_N_PARALLEL grows the sampler batch (fewer sequential SDE
#          passes + better core saturation), without touching the study fingerprint.
# Lever 2: one worker per dataset, run concurrently, each capped to half the cores
#          (OMP_NUM_THREADS) so they don't oversubscribe, with separate status files.
#
# Run ONLY after the california_housing job (PID as of writing: 91616) has finished.
# Verify first:  ps -p 91616  ->  should be gone.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

SPACES="treeffuser_published treeffuser_score_plus treeffuser_fm treeffuser_score_flex treeffuser_score_flex_sde"
MANIFEST=benchmarks/configs/tuning_manifest.yaml
LOGDIR=benchmarks/results/tuning
export TREEFFUSER_N_PARALLEL=100   # drop to 50 if RSS climbs too high on ct_slices

run_worker () {
  local ds="$1"
  OMP_NUM_THREADS=5 pixi run python -m benchmarks.tuning.manifest \
    --manifest "$MANIFEST" \
    --datasets "$ds" \
    --spaces $SPACES \
    --status-path "$LOGDIR/batch_status_${ds}.jsonl" \
    --eval-after-tune > "$LOGDIR/tail_${ds}.log" 2>&1 &
  echo "launched $ds -> pid $! (log: $LOGDIR/tail_${ds}.log)"
}

run_worker protein
run_worker ct_slices
wait
echo "ALL TAIL DATASETS DONE"
