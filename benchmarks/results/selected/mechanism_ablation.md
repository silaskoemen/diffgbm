# Treeffuser mechanism ablation

Source: `benchmarks/results/mechanism_ablation/eval/*.jsonl`. Each cell averages the five evaluation folds from the fold-0 tuning / folds-1..5 evaluation protocol. All rows use the same LightGBM hyperparameter search surface; fixed method choices and bound samplers differ by row. rel-CRPS is normalized by the best ablation row on each dataset.

## Aggregate metrics

| Variant | CRPSS | rel-CRPS | CRPS | q-MACE | \|cE\|@90 | \|cE\|@95 | KS pass | sample s | fit s | raw wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Score noise, Euler-50 | 0.670 | 1.150 | 4.279 | 0.044 | 0.044 | 0.023 | 0.60 | 217.30 | 13.1 | 2 |
| Score noise, Heun-25 | 0.629 | 1.424 | 5.490 | 0.058 | 0.049 | 0.025 | 0.20 | 190.92 | 13.2 | 0 |
| + residualizer | 0.646 | 1.104 | 5.407 | 0.041 | 0.044 | 0.027 | 0.80 | 47.58 | 14.7 | 0 |
| + EDM input/target | 0.680 | 1.042 | 4.108 | 0.034 | 0.025 | 0.016 | 0.50 | 109.46 | 18.5 | 0 |
| + log-sigma feature | 0.679 | 1.042 | 4.112 | 0.035 | 0.027 | 0.017 | 0.40 | 73.10 | 13.8 | 0 |
| Score+ full | 0.681 | 1.045 | 4.057 | 0.028 | 0.020 | 0.015 | 0.80 | 35.11 | 12.5 | 0 |
| FM linear + resid | 0.684 | 1.027 | 4.071 | 0.028 | 0.024 | 0.020 | 0.80 | 19.28 | 16.7 | 4 |
| FM trig + resid | 0.681 | 1.027 | 4.170 | 0.028 | 0.026 | 0.021 | 0.60 | 23.87 | 15.4 | 2 |
| FM VP no resid | 0.631 | 1.985 | 4.189 | 0.050 | 0.051 | 0.032 | 0.30 | 23.19 | 11.6 | 1 |
| FM VP + resid | 0.680 | 1.034 | 4.121 | 0.034 | 0.028 | 0.024 | 0.60 | 14.69 | 13.6 | 1 |

## Per-dataset raw CRPS winners

| Dataset | Winner | CRPS |
|---|---|---:|
| california_housing | Score noise, Euler-50 | 0.197089 |
| concrete | FM linear + resid | 2.11186 |
| diabetes | FM VP no resid | 32.9781 |
| energy | FM linear + resid | 0.194775 |
| kin8nm | FM trig + resid | 0.0548777 |
| naval | FM linear + resid | 0.000203752 |
| power_plant | FM VP + resid | 1.5349 |
| protein | FM linear + resid | 1.70498 |
| wine | Score noise, Euler-50 | 0.300395 |
| yacht | FM trig + resid | 0.277075 |

## Per-dataset raw CRPS

| Dataset | Score noise, Euler-50 | Score noise, Heun-25 | + residualizer | + EDM input/target | + log-sigma feature | Score+ full | FM linear + resid | FM trig + resid | FM VP no resid | FM VP + resid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| california_housing | 0.1971 | 0.2011 | 0.2069 | 0.2073 | 0.2068 | 0.2042 | 0.2066 | 0.2062 | 0.2187 | 0.2056 |
| concrete | 2.457 | 2.528 | 2.229 | 2.15 | 2.163 | 2.126 | 2.112 | 2.172 | 2.498 | 2.158 |
| diabetes | 35.68 | 47.33 | 47.38 | 34.54 | 34.54 | 34.02 | 34.26 | 35.22 | 32.98 | 34.66 |
| energy | 0.2519 | 0.3827 | 0.2173 | 0.2069 | 0.21 | 0.2145 | 0.1948 | 0.2013 | 0.8389 | 0.2039 |
| kin8nm | 0.06685 | 0.06568 | 0.05897 | 0.05536 | 0.05592 | 0.05756 | 0.05551 | 0.05488 | 0.06614 | 0.05535 |
| naval | 0.0002178 | 0.0003794 | 0.0002247 | 0.0002185 | 0.0002133 | 0.0002197 | 0.0002038 | 0.0002075 | 0.0006743 | 0.0002111 |
| power_plant | 1.671 | 1.686 | 1.579 | 1.562 | 1.567 | 1.572 | 1.557 | 1.538 | 2.029 | 1.535 |
| protein | 1.722 | 1.716 | 1.785 | 1.743 | 1.768 | 1.769 | 1.705 | 1.714 | 1.739 | 1.796 |
| wine | 0.3004 | 0.3196 | 0.3201 | 0.3242 | 0.3249 | 0.319 | 0.3197 | 0.3197 | 0.3319 | 0.3181 |
| yacht | 0.4375 | 0.664 | 0.2941 | 0.2892 | 0.2827 | 0.2843 | 0.3029 | 0.2771 | 1.187 | 0.2831 |

## Interpretation notes

- Moving the noise-prediction baseline from Euler SDE to Heun PF-ODE without changing the score parameterization hurts CRPS and PIT pass rate; the sampler speedup is not a free model-side replacement.
- Residualization is essential for the few-step FM rows: VP without residualization is the weakest rel-CRPS row despite one diabetes win.
- EDM input/target preconditioning gives the clearest score-side accuracy and calibration jump; the explicit log-sigma feature is approximately neutral under uniform t in this tuned protocol.
- Full score+ is the best calibrated score-side row. Among residualized FM paths, linear has the best aggregate CRPSS, trig is close and wins two datasets, and VP is fastest. This revises the earlier fixed-diagnostic path story into a practical path/conditioning frontier.
