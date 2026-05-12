# Treeffuser Benchmarks

This directory contains lightweight, implementation-focused benchmarks for comparing
Treeffuser variants. It is separate from `testbed/`: `testbed/` is for broad model
comparisons, while this harness is for paired diagnostics during Treeffuser development.

The benchmark grain is one result row per:

```text
dataset x seed x variant x sampler
```

This keeps comparisons paired and makes it possible to ask whether a variant improves
coverage or CRPS without hiding the cost in wider intervals, slower sampling, or more
training rows.

## Layout

```text
benchmarks/
  run.py
  harness.py
  variants.py
  datasets.py
  metrics.py
  configs/
    smoke.yaml
    synthetic_core.yaml
    real_smoke.yaml
  results/raw/
```

## Running

The runner uses PyYAML if available and otherwise falls back to a small parser for the
simple YAML subset used by these configs:

```bash
python -m benchmarks.run --config benchmarks/configs/smoke.yaml
```

By default, results are written as JSON Lines to
`benchmarks/results/raw/<config>__<variants>_<timestamp>.jsonl`. JSONL is the preferred
format because each completed benchmark row is appended immediately and variant-specific
parameter dictionaries do not force CSV schema rewrites.

To run only selected variants from a broad config:

```bash
python -m benchmarks.run \
  --config benchmarks/configs/synthetic_core.yaml \
  --variants baseline_raw_time residualized_mean_edm_raw_time_log_std
```

CSV remains available with `--output-format csv` or by passing an `--output` path ending
in `.csv`.
The `real_smoke.yaml` config uses local datasets bundled with scikit-learn, so it does
not download external benchmark data.

## Seeding Policy

For each dataset/seed pair, the harness derives and records three seeds:

- `data_seed`: controls data generation and paired train/test splits.
- `model_seed`: controls model training randomness.
- `sampler_seed`: controls Monte Carlo sampling randomness.

The same resolved `model_seed` and `sampler_seed` are used across variants for the same
dataset/seed pair unless the config explicitly changes the offsets. This gives paired
comparisons a stable stochastic contract.

## Recommended Variant

From the 2026-05-12 decision runs (see `results/raw/synthetic_core__*_20260512_*.jsonl`
and `results/raw/real_smoke__*_20260512_*.jsonl`), the leading experimental combo is
`score_parameterization="edm"` + `residualize="mean"` + `noise_features="raw_time_log_std"`,
encoded as the `residualized_mean_edm_raw_time_log_std` variant. It ties baseline CRPS on
synthetic and beats it ~3% on real_smoke, with ~40% lower interval-90 absolute coverage
error on synthetic and ~80% lower on real_smoke. Mean residualization and EDM are
independent improvements: mean residualization mostly moves CRPS, EDM mostly moves
calibration. `mean_scale` and the plain-`x0` parameterization stayed experimental — no
clear CRPS win, higher fit cost.

The library default remains `residualize="off"` for backward compatibility.

### 2026-05-12 expanded `real_smoke` decision run

`real_smoke.yaml` now spans four datasets (`diabetes`, `california_housing`, `kin8nm`,
`wine_quality_white`). See `results/raw/real_smoke__all_20260512_194200.jsonl`. The
calibration story for `residualized_mean_edm_raw_time_log_std` holds: interval-90 absolute
coverage error drops vs `baseline_raw_time` on every dataset (avg ~50% reduction). CRPS is
mixed — wins on diabetes (-3.3%) and kin8nm (-8.9%), ties on california_housing, and loses
~5.7% on wine_quality_white. EDM-only variants tend to undercover on real data; pairing
EDM with mean residualization is what closes that gap. The next lever investigated for
CRPS uniformity is min-SNR loss weighting (see `loss_weighting`).

### 2026-05-12 min-SNR loss weighting — experimental, not adopted

Parameterization-aware min-SNR-γ weighting from Hang et al. (2023) was added behind
`loss_weighting="min_snr"` + `min_snr_gamma` and benchmarked against
`residualized_mean_edm_raw_time_log_std` with γ ∈ {1, 5}. See
`results/raw/real_smoke__*minsnr*_20260512_201658.jsonl` and
`results/raw/synthetic_core__*minsnr*_20260512_201659.jsonl`. Real data: γ=1 helps on
california_housing (CRPS −1.3%, I90 err −26%) but hurts elsewhere; γ=5 is uniformly
worse. Synthetic: γ=5 wins on bimodal_mixture (I90 err 0.014 vs 0.038) and skewed_noise
but otherwise ties or loses. Not adopted as default; surface kept opt-in for future use
on multimodal-tailed targets.

### 2026-05-12 EDM-style log-σ t sampling sweep — positive, recommended

Configurable training-time `t` distribution added behind `t_sampling="log_sigma_normal"`
with `log_sigma_p_mean` and `log_sigma_p_std`. Sweep on `residualized_mean_edm_raw_time_log_std`
across 8 datasets × 3 seeds × 5 variants. See
`results/raw/log_sigma_sweep__all_20260512_214355.jsonl`.

Result: log-σ sampling improves CRPS on **every dataset** vs `t_sampling="uniform"`. The
EDM-default `(p_mean=-1.2, p_std=1.2)` is the safest pick across all 8 sets (CRPS gains
0.5%–3.9%). Standout wins: `wine_quality_white` (the only previously regressing real
dataset) recovers a CRPS improvement under `(p_mean=-1.2, p_std=2.0)`; `bimodal_mixture`
gets the largest coverage win in the whole improvement track (interval-90 absolute
coverage error drops from 0.045 to 0.012, -74%). Tradeoff: coverage error slightly
regresses on a few small real sets (diabetes, kin8nm). This matches the GBT hypothesis
that emerged from the PF-ODE failure: tree-based score models are bin-density-limited,
so shifting `t` density beats reweighting fixed bins.

Recommended successor variant for the leading combo:
`score_parameterization="edm"` + `residualize="mean"` + `noise_features="raw_time_log_std"`
+ `t_sampling="log_sigma_normal"` (`p_mean=-1.2`, `p_std=1.2`). The library default
remains `t_sampling="uniform"` for backward compatibility; new fits should opt in.

### 2026-05-12 Residualizer-capacity sweep — high-capacity adopted for real data

Capacity sweep on the LightGBM conditional-mean model used by `residualize="mean"`,
on top of the new winning combo (EDM + log-σ t-sampling). Three points:
A — current defaults; B — regularized (shallower, more rounds, stronger min_child);
C — high-capacity (`max_depth=-1`, `num_leaves=63`, `min_child_samples=10`,
`n_estimators=300`, `learning_rate=0.05`). See
`results/raw/residualizer_sweep__all_20260512_221655.jsonl`.

Outcome: C wins on real data, A wins on synthetic. On the 4 real sets, C improves I90
absolute coverage error from `{0.044, 0.051, 0.023, 0.017}` to `{0.010, 0.010, 0.034,
0.006}` (closes the diabetes coverage regression from log-σ sampling; halves the wine
gap), and improves CRPS on 3 of 4 (slight regression only on diabetes). On synthetic,
CRPS regresses 5–11% with C — those generators have near-linear conditional means that
A already captures.

Two secondary findings worth recording:
1. **OOF MSE does not track CRPS.** C has higher OOF MSE than A on every dataset yet
   wins downstream CRPS on the real ones. The residualizer's interaction with the
   diffusion matters more than pure mean-prediction accuracy. Future residualizer
   tuning should optimize downstream CRPS via a benchmark run, not OOF MSE.
2. **Variant B (regularized) is strictly dominated.** Lower capacity doesn't help on
   any dataset; the current defaults were already on the safe side of the
   capacity axis.

Recommendation: pair the new winning combo with `extra_residualizer_params` set to the
C configuration when targeting real tabular data; keep A defaults for synthetic
diagnostics. The library defaults remain unchanged (residualizer-A) for backward
compatibility.

### 2026-05-12 Residualizer early-stopping sweep — ES helps as auto-tuner, not as a peak-quality lever

Inner-split early stopping added behind setting `early_stopping_rounds` in
`extra_residualizer_params`. The residualizer splits each fold's `train_idx` further
into 85% inner-train / 15% inner-val, with a hard gate: when inner-val < 50 rows the
residualizer warns and falls back to the empirically validated high-capacity defaults
(variant C). Sweep on A (current), C (high-cap fixed), D (ES + moderate caps), E
(ES + lifted caps). See `results/raw/residualizer_es_sweep__all_20260512_223507.jsonl`.

Findings:
1. **D is the most robust single config.** Ties or beats A on synthetic CRPS (the
   regime A used to win), and lands within 0.2–1% of C on real CRPS. Neither A nor
   C alone could span both regimes.
2. **E is dominated by D.** Lifting depth/leaf caps while running ES gains nothing —
   early stopping already controls capacity better than caps do.
3. **The size gate works cleanly.** On diabetes (inner-val = 36 < 50) the gate trips
   and D/E both fall back to C with identical results, as designed.
4. **C still wins on real-data CRPS by small margins** (0.2–0.8%). For large enough
   `n`, fixed high-capacity beats ES at moderate caps. ES's value is robustness, not
   peak performance.

Recommendation update: **C** remains the peak-quality config for real tabular data.
**D** is the new recommended "auto" config when the user does not know `n` in advance —
it self-tunes via ES on large data and falls back to C on small data via the gate.
Library defaults remain unchanged for backward compatibility.

### 2026-05-12 PF-ODE / Heun sampler sweep — negative, not adopted

A deterministic probability-flow ODE plus a Heun second-order solver were added behind
`sampler_method="heun"` + `pf_ode=True`. Step-count sweep at `n_steps ∈ {15, 25, 50, 100}`
on 8 datasets × 3 seeds, two variants (baseline + winner). See
`results/raw/pf_ode_sweep__all_20260512_205403.jsonl`. Heun PF-ODE is worse than Euler SDE
at every step count for both variants: on `residualized_mean_edm_raw_time_log_std`, Euler
@ 15 steps (CRPS 4.285, I90 err 0.022, time 0.79s) beats Heun PF-ODE @ 100 steps (CRPS
4.498, I90 err 0.085, time 6.56s) on every metric while running ~8× faster. Likely cause:
the LightGBM-based score is piecewise-constant in `t` (histogram binning), so Heun's
predictor-corrector has no smooth drift to average; removing the stochastic term also
tightens the sampling distribution beyond what the noisy score warrants, dropping
interval-90 coverage from ~0.89 to ~0.82. This negative result also closes off tree-based
flow matching as a near-term direction (it was gated on ODE-based sampling winning here).
The sampler surface is kept for completeness but defaults remain Euler/SDE.

## Provenance

Every result row records:

- `git_sha`
- `git_dirty`
- `treeffuser_source_hash`

Because `baseline_current` means "the current baseline behavior in this checkout", these
columns are required to interpret old-vs-new comparisons later.
