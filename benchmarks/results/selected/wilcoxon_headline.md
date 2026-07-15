# Paired Wilcoxon signed-rank tests on headline CRPS

Source: `benchmarks/results/tuning/eval/*.jsonl`. Each variant's CRPS is averaged over 5 evaluation folds per dataset; the paired test is run across the eleven benchmark datasets (n=11). Lower CRPS is better, so a negative signed difference favours the left-hand variant.

## Summary

| Pair | Alt. | n | a-wins | b-wins | ties | median Δ | W | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| score-flex vs published | less | 11 | 11 | 0 | 0 | -0.0376 | 0.0 | 4.88e-04 |
| FM vs published | less | 11 | 6 | 5 | 0 | -0.0098 | 22.0 | 0.183 |
| score-flex vs FM | two-sided | 11 | 8 | 3 | 0 | -0.0140 | 17.0 | 0.175 |

## Per-dataset mean CRPS (5-fold average)

| dataset | published | score-flex | FM |
|---|---|---|---|
| california_housing | 0.2003 | 0.1904 | 0.2044 |
| concrete | 2.7126 | 2.2702 | 2.1514 |
| ct_slices | 0.1594 | 0.1432 | 0.4332 |
| diabetes | 36.0370 | 33.9772 | 35.1901 |
| energy | 0.2364 | 0.1988 | 0.2018 |
| kin8nm | 0.0652 | 0.0560 | 0.0554 |
| naval | 0.0002 | 0.0001 | 0.0002 |
| power_plant | 1.6295 | 1.4748 | 1.5218 |
| protein | 1.7042 | 1.5683 | 1.7744 |
| wine | 0.2942 | 0.2778 | 0.3168 |
| yacht | 0.4474 | 0.4037 | 0.2692 |

## Per-dataset signed differences

### score-flex vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | -0.0099 |
| concrete | -0.4424 |
| ct_slices | -0.0162 |
| diabetes | -2.0598 |
| energy | -0.0376 |
| kin8nm | -0.0092 |
| naval | -0.0000 |
| power_plant | -0.1547 |
| protein | -0.1359 |
| wine | -0.0164 |
| yacht | -0.0436 |

### FM vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0041 |
| concrete | -0.5613 |
| ct_slices | +0.2738 |
| diabetes | -0.8468 |
| energy | -0.0345 |
| kin8nm | -0.0098 |
| naval | +0.0000 |
| power_plant | -0.1077 |
| protein | +0.0702 |
| wine | +0.0226 |
| yacht | -0.1782 |

### score-flex vs FM (alt=two-sided)

| dataset | Δ CRPS |
|---|---:|
| california_housing | -0.0140 |
| concrete | +0.1188 |
| ct_slices | -0.2900 |
| diabetes | -1.2129 |
| energy | -0.0031 |
| kin8nm | +0.0006 |
| naval | -0.0001 |
| power_plant | -0.0470 |
| protein | -0.2061 |
| wine | -0.0390 |
| yacht | +0.1346 |
