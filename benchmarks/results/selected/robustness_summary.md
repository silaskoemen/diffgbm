# Robustness summary

Source: `benchmarks/results/tuning/eval/*.jsonl` (450 fold-level rows). Each displayed value first averages folds 1--5 within each dataset/model family, then averages over datasets unless stated otherwise. rel-CRPS is normalized by the best displayed family on each dataset.

## Aggregate metrics with dataset standard errors

| Model | CRPSS | SE | rel-CRPS | SE | q-MACE | \|cE\|@90 | \|cE\|@95 | PIT pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treeffuser-FM | 0.680 | 0.086 | 1.117 | 0.056 | 0.034 | 0.028 | 0.024 | 0.46 |
| Treeffuser-score+ | 0.681 | 0.085 | 1.131 | 0.062 | 0.028 | 0.020 | 0.015 | 0.62 |
| Treeffuser-published | 0.670 | 0.084 | 1.242 | 0.086 | 0.044 | 0.044 | 0.023 | 0.40 |
| CatBoost-unc. | 0.638 | 0.088 | 1.409 | 0.113 | 0.045 | 0.078 | 0.066 | 0.28 |
| QReg-LightGBM | 0.668 | 0.079 | 1.459 | 0.278 | 0.033 | 0.041 | 0.044 | 0.34 |
| iBUG | 0.649 | 0.081 | 1.471 | 0.154 | 0.069 | 0.051 | 0.065 | 0.36 |
| Deep ensemble | 0.654 | 0.082 | 1.519 | 0.201 | 0.050 | 0.032 | 0.018 | 0.26 |
| NGBoost | 0.601 | 0.094 | 1.550 | 0.190 | 0.052 | 0.114 | 0.101 | 0.16 |
| CARD-style diffusion | 0.631 | 0.091 | 1.632 | 0.349 | 0.051 | 0.057 | 0.043 | 0.24 |

## Mean ranks across datasets

| Model | CRPS rank | CRPSS rank | \|cE\|@95 rank |
|---|---:|---:|---:|
| Treeffuser-FM | 2.80 | 2.80 | 4.00 |
| Treeffuser-score+ | 3.00 | 3.00 | 2.10 |
| QReg-LightGBM | 4.00 | 4.00 | 5.60 |
| Treeffuser-published | 4.10 | 4.10 | 3.30 |
| Deep ensemble | 5.80 | 5.80 | 2.90 |
| iBUG | 5.80 | 5.80 | 6.50 |
| CatBoost-unc. | 6.00 | 6.00 | 7.10 |
| CARD-style diffusion | 6.60 | 6.60 | 5.70 |
| NGBoost | 6.90 | 6.90 | 7.80 |

## Friedman and Nemenyi rank diagnostic

Friedman test on per-dataset CRPS ranks over 9 families and 10 datasets: chi^2=25.47, p=0.001. Nemenyi critical difference at alpha=0.05: 3.80 mean-rank points.

| Pair | mean-rank gap |
|---|---:|
| Treeffuser-score+ vs NGBoost | 3.90 |
| Treeffuser-FM vs NGBoost | 4.10 |
| Treeffuser-FM vs CARD-style diffusion | 3.80 |

## Paired Wilcoxon tests for Treeffuser headline rows

| Pair | Alt. | left wins | right wins | median delta CRPS | W | p |
|---|---|---:|---:|---:|---:|---:|
| score+ vs published | less | 6 | 4 | -0.0233 | 13.0 | 0.080 |
| FM vs published | less | 7 | 3 | -0.0297 | 12.0 | 0.065 |
| FM vs score+ | two-sided | 6 | 4 | -0.0004 | 26.0 | 0.922 |

## Per-dataset winners

| Dataset | Raw-CRPS winner | CRPS | Treeffuser CRPSS winner | CRPSS |
|---|---|---:|---|---:|
| california_housing | Treeffuser-published | 0.197089 | Treeffuser-published | 0.687 |
| concrete | Treeffuser-score+ | 2.12616 | Treeffuser-score+ | 0.774 |
| diabetes | Deep ensemble | 31.2506 | Treeffuser-score+ | 0.241 |
| energy | Treeffuser-FM | 0.203919 | Treeffuser-FM | 0.964 |
| kin8nm | Deep ensemble | 0.0366765 | Treeffuser-FM | 0.631 |
| naval | CARD-style diffusion | 0.000154683 | Treeffuser-FM | 0.951 |
| power_plant | Treeffuser-FM | 1.5349 | Treeffuser-FM | 0.843 |
| protein | QReg-LightGBM | 1.6824 | Treeffuser-published | 0.494 |
| wine | Treeffuser-published | 0.300395 | Treeffuser-published | 0.346 |
| yacht | Treeffuser-FM | 0.283098 | Treeffuser-FM | 0.962 |

Raw-CRPS winner counts:
- Treeffuser-FM: 3
- Deep ensemble: 2
- Treeffuser-published: 2
- CARD-style diffusion: 1
- QReg-LightGBM: 1
- Treeffuser-score+: 1

## Dataset-size groups

| Group | n datasets | Model | CRPSS | rel-CRPS |
|---|---:|---|---:|---:|
| small | 4 | Treeffuser-published | 0.709 | 1.269 |
| small | 4 | Treeffuser-score+ | 0.735 | 1.036 |
| small | 4 | Treeffuser-FM | 0.731 | 1.031 |
| small | 4 | QReg-LightGBM | 0.711 | 1.775 |
| small | 4 | CatBoost-unc. | 0.724 | 1.278 |
| small | 4 | NGBoost | 0.718 | 1.167 |
| small | 4 | Deep ensemble | 0.699 | 1.865 |
| small | 4 | iBUG | 0.700 | 1.470 |
| small | 4 | CARD-style diffusion | 0.632 | 2.342 |
| medium | 4 | Treeffuser-published | 0.670 | 1.330 |
| medium | 4 | Treeffuser-score+ | 0.678 | 1.269 |
| medium | 4 | Treeffuser-FM | 0.683 | 1.233 |
| medium | 4 | QReg-LightGBM | 0.666 | 1.358 |
| medium | 4 | CatBoost-unc. | 0.612 | 1.645 |
| medium | 4 | NGBoost | 0.550 | 2.058 |
| medium | 4 | Deep ensemble | 0.672 | 1.346 |
| medium | 4 | iBUG | 0.637 | 1.669 |
| medium | 4 | CARD-style diffusion | 0.667 | 1.173 |
| large | 2 | Treeffuser-published | 0.591 | 1.012 |
| large | 2 | Treeffuser-score+ | 0.578 | 1.044 |
| large | 2 | Treeffuser-FM | 0.573 | 1.055 |
| large | 2 | QReg-LightGBM | 0.587 | 1.030 |
| large | 2 | CatBoost-unc. | 0.517 | 1.200 |
| large | 2 | NGBoost | 0.468 | 1.301 |
| large | 2 | Deep ensemble | 0.529 | 1.172 |
| large | 2 | iBUG | 0.571 | 1.076 |
| large | 2 | CARD-style diffusion | 0.555 | 1.129 |

## Interpretation notes

- Mean ranks and raw-CRPS wins show mixed per-dataset winners rather than a one-family sweep.
- The Treeffuser rows lead the cross-dataset CRPSS/rel-CRPS aggregate, but the dataset standard errors are large relative to the small aggregate gaps.
- Score+ is the most robust calibration row by PIT pass rate and interval coverage error; FM is strongest on mean CRPS rank and rel-CRPS.
- The size split shows the large-dataset inversion: published SDE and QReg are strongest on the two largest datasets, while score+ and FM lead on small/medium groups.
- Friedman/Nemenyi is useful as a rank robustness check, but with ten datasets it should be treated as supporting context rather than a headline significance claim.
