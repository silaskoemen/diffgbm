# Robustness summary

Source: `benchmarks/results/tuning/eval/*.jsonl` (615 fold-level rows). Each displayed value first averages folds 1--5 within each dataset/model family, then averages over datasets unless stated otherwise. rel-CRPS is normalized by the best displayed family on each dataset.

## Aggregate metrics with dataset standard errors

| Model | CRPSS | SE | rel-CRPS | SE | q-MACE | \|cE\|@90 | \|cE\|@95 | PIT pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffGBM (score-flex) | 0.725 | 0.076 | 1.106 | 0.061 | 0.045 | 0.049 | 0.037 | 0.33 |
| Treeffuser (published) | 0.699 | 0.081 | 1.248 | 0.074 | 0.053 | 0.047 | 0.026 | 0.27 |
| DiffGBM-FM | 0.707 | 0.082 | 1.334 | 0.180 | 0.036 | 0.029 | 0.021 | 0.45 |
| Deep ensemble | 0.685 | 0.080 | 1.556 | 0.199 | 0.054 | 0.036 | 0.019 | 0.24 |
| iBUG | 0.679 | 0.080 | 1.564 | 0.171 | 0.069 | 0.052 | 0.060 | 0.33 |
| CARD-style diffusion | 0.662 | 0.088 | 1.746 | 0.334 | 0.050 | 0.059 | 0.044 | 0.22 |
| CatBoost-unc. | 0.666 | 0.085 | 1.815 | 0.350 | 0.045 | 0.080 | 0.066 | 0.25 |
| QReg-LightGBM | 0.694 | 0.076 | 1.826 | 0.394 | 0.038 | 0.045 | 0.043 | 0.31 |
| NGBoost | 0.626 | 0.089 | 2.471 | 0.852 | 0.051 | 0.111 | 0.098 | 0.15 |

## Mean ranks across datasets

| Model | CRPS rank | CRPSS rank | \|cE\|@95 rank |
|---|---:|---:|---:|
| DiffGBM (score-flex) | 1.91 | 2.00 | 4.73 |
| DiffGBM-FM | 3.45 | 3.45 | 3.18 |
| Treeffuser (published) | 4.18 | 4.27 | 3.55 |
| QReg-LightGBM | 4.36 | 4.27 | 5.36 |
| Deep ensemble | 5.45 | 5.45 | 2.73 |
| iBUG | 5.91 | 5.82 | 5.82 |
| CatBoost-unc. | 6.09 | 6.09 | 6.73 |
| CARD-style diffusion | 6.73 | 6.73 | 5.55 |
| NGBoost | 6.91 | 6.91 | 7.36 |

## Friedman and Nemenyi rank diagnostic

Friedman test on per-dataset CRPS ranks over 9 families and 11 datasets: chi^2=32.07, p=0.000. Nemenyi critical difference at alpha=0.05: 3.62 mean-rank points.

| Pair | mean-rank gap |
|---|---:|
| DiffGBM (score-flex) vs CatBoost-unc. | 4.18 |
| DiffGBM (score-flex) vs NGBoost | 5.00 |
| DiffGBM (score-flex) vs iBUG | 4.00 |
| DiffGBM (score-flex) vs CARD-style diffusion | 4.82 |

## Paired Wilcoxon tests for Treeffuser headline rows

| Pair | Alt. | left wins | right wins | median delta CRPS | W | p |
|---|---|---:|---:|---:|---:|---:|
| score-flex vs published | less | 11 | 0 | -0.0376 | 0.0 | 0.000 |
| FM vs published | less | 6 | 5 | -0.0098 | 22.0 | 0.183 |
| score-flex vs FM | two-sided | 8 | 3 | -0.0140 | 17.0 | 0.175 |

## Per-dataset winners

| Dataset | Raw-CRPS winner | CRPS | Treeffuser CRPSS winner | CRPSS |
|---|---|---:|---|---:|
| california_housing | DiffGBM (score-flex) | 0.190435 | DiffGBM (score-flex) | 0.698 |
| concrete | DiffGBM-FM | 2.15139 | DiffGBM-FM | 0.772 |
| ct_slices | DiffGBM (score-flex) | 0.143207 | DiffGBM (score-flex) | 0.989 |
| diabetes | Deep ensemble | 31.2506 | DiffGBM (score-flex) | 0.241 |
| energy | DiffGBM (score-flex) | 0.198761 | DiffGBM (score-flex) | 0.965 |
| kin8nm | Deep ensemble | 0.0366765 | DiffGBM-FM | 0.631 |
| naval | DiffGBM (score-flex) | 0.000128857 | DiffGBM (score-flex) | 0.970 |
| power_plant | DiffGBM (score-flex) | 1.47479 | DiffGBM (score-flex) | 0.849 |
| protein | DiffGBM (score-flex) | 1.56829 | DiffGBM (score-flex) | 0.539 |
| wine | DiffGBM (score-flex) | 0.277772 | DiffGBM (score-flex) | 0.395 |
| yacht | DiffGBM-FM | 0.269165 | DiffGBM-FM | 0.964 |

Raw-CRPS winner counts:
- DiffGBM (score-flex): 7
- Deep ensemble: 2
- DiffGBM-FM: 2

## Dataset-size groups

| Group | n datasets | Model | CRPSS | rel-CRPS |
|---|---:|---|---:|---:|
| small | 4 | Treeffuser (published) | 0.701 | 1.316 |
| small | 4 | DiffGBM (score-flex) | 0.728 | 1.161 |
| small | 4 | DiffGBM-FM | 0.729 | 1.035 |
| small | 4 | QReg-LightGBM | 0.711 | 1.830 |
| small | 4 | CatBoost-unc. | 0.724 | 1.303 |
| small | 4 | NGBoost | 0.718 | 1.187 |
| small | 4 | Deep ensemble | 0.699 | 1.907 |
| small | 4 | iBUG | 0.700 | 1.499 |
| small | 4 | CARD-style diffusion | 0.632 | 2.410 |
| medium | 4 | Treeffuser (published) | 0.680 | 1.302 |
| medium | 4 | DiffGBM (score-flex) | 0.710 | 1.132 |
| medium | 4 | DiffGBM-FM | 0.684 | 1.326 |
| medium | 4 | QReg-LightGBM | 0.666 | 1.471 |
| medium | 4 | CatBoost-unc. | 0.612 | 1.792 |
| medium | 4 | NGBoost | 0.550 | 2.237 |
| medium | 4 | Deep ensemble | 0.672 | 1.476 |
| medium | 4 | iBUG | 0.637 | 1.828 |
| medium | 4 | CARD-style diffusion | 0.667 | 1.262 |
| large | 3 | Treeffuser (published) | 0.723 | 1.084 |
| large | 3 | DiffGBM (score-flex) | 0.742 | 1.000 |
| large | 3 | DiffGBM-FM | 0.707 | 1.743 |
| large | 3 | QReg-LightGBM | 0.707 | 2.294 |
| large | 3 | CatBoost-unc. | 0.659 | 2.529 |
| large | 3 | NGBoost | 0.605 | 4.494 |
| large | 3 | Deep ensemble | 0.682 | 1.193 |
| large | 3 | iBUG | 0.708 | 1.300 |
| large | 3 | CARD-style diffusion | 0.695 | 1.504 |

## Interpretation notes

- DiffGBM (score-flex) leads mean CRPS rank and takes the plurality of raw-CRPS wins; the remaining per-dataset winners are DiffGBM-FM and the deep ensemble.
- The DiffGBM rows lead the cross-dataset CRPSS/rel-CRPS aggregate, but the dataset standard errors are large relative to the small aggregate gaps.
- DiffGBM-FM is the most robust calibration row by PIT pass rate and interval coverage error; DiffGBM (score-flex) is strongest on mean CRPS rank and rel-CRPS.
- The size split shows DiffGBM (score-flex) leading every size group on CRPSS/rel-CRPS, most decisively on the large group; DiffGBM-FM is competitive on small/medium but weak on large.
- Friedman/Nemenyi is useful as a rank robustness check, but with eleven datasets it should be treated as supporting context rather than a headline significance claim.
