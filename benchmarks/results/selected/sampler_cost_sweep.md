# Sampler cost sweep

Source: `benchmarks/results/sampler_cost_sweep/eval/sampler_cost_sweep.jsonl` (950 fold-level rows). Each displayed row averages folds 1--5 within each dataset, then averages over the ten UCI datasets. rel-CRPS is normalized by the best sampler-cost row on each dataset.

## Aggregate metrics

| Family | Sampler | Steps | CRPSS | rel-CRPS | q-MACE | \|cE\|@90 | \|cE\|@95 | sample s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Published Euler SDE | Euler SDE | 15 | 0.660 | 1.218 | 0.057 | 0.050 | 0.026 | 54.95 |
| Published Euler SDE | Euler SDE | 25 | 0.666 | 1.181 | 0.049 | 0.046 | 0.023 | 92.29 |
| Published Euler SDE | Euler SDE | 50 | 0.670 | 1.156 | 0.044 | 0.044 | 0.023 | 199.14 |
| Published Euler SDE | Euler SDE | 100 | 0.672 | 1.144 | 0.042 | 0.041 | 0.023 | 373.50 |
| Score+ PF-ODE | PF-ODE | 5 | 0.473 | 1.781 | 0.158 | 0.098 | 0.048 | 9.74 |
| Score+ PF-ODE | PF-ODE | 10 | 0.672 | 1.082 | 0.057 | 0.066 | 0.034 | 15.57 |
| Score+ PF-ODE | PF-ODE | 15 | 0.679 | 1.053 | 0.035 | 0.033 | 0.018 | 20.91 |
| Score+ PF-ODE | PF-ODE | 25 | 0.681 | 1.048 | 0.028 | 0.020 | 0.015 | 31.59 |
| Score+ PF-ODE | PF-ODE | 50 | 0.681 | 1.047 | 0.027 | 0.021 | 0.016 | 58.51 |
| FM-VP ODE | ODE | 3 | 0.679 | 1.046 | 0.038 | 0.036 | 0.027 | 9.99 |
| FM-VP ODE | ODE | 5 | 0.680 | 1.037 | 0.034 | 0.028 | 0.024 | 13.69 |
| FM-VP ODE | ODE | 10 | 0.681 | 1.030 | 0.030 | 0.025 | 0.023 | 22.79 |
| FM-VP ODE | ODE | 25 | 0.681 | 1.028 | 0.028 | 0.026 | 0.022 | 49.70 |
| FM-VP ODE | SDE | 25 | 0.679 | 1.035 | 0.038 | 0.037 | 0.018 | 50.47 |
| FM-linear ODE | ODE | 3 | 0.683 | 1.031 | 0.029 | 0.023 | 0.019 | 13.80 |
| FM-linear ODE | ODE | 5 | 0.684 | 1.030 | 0.028 | 0.024 | 0.020 | 19.69 |
| FM-linear ODE | ODE | 10 | 0.684 | 1.030 | 0.028 | 0.023 | 0.019 | 34.59 |
| FM-linear ODE | ODE | 25 | 0.684 | 1.030 | 0.028 | 0.022 | 0.019 | 77.72 |
| FM-linear ODE | SDE | 25 | 0.683 | 1.034 | 0.036 | 0.034 | 0.021 | 76.64 |

## Interpretation notes

- FM ODE is effectively converged by 3--5 steps on aggregate CRPS; extra ODE steps mainly add latency.
- Score+ needs about 15--25 Heun PF-ODE steps for the calibration metrics to settle; 5 steps is not competitive.
- Published Euler SDE improves gradually with more steps but remains much slower at comparable CRPS.
- FM-SDE at 25 steps improves VP-FM 95% coverage error, but worsens q-MACE, 90% coverage error, and CRPS.
- Linear FM remains the tuned CRPS corner in this sweep; VP-FM remains the faster headline FM row.

Figure: `benchmarks/results/sampler_cost_sweep/sampler_cost_tradeoff.pdf`.
