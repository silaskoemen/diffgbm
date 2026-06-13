# Paired Wilcoxon signed-rank tests on headline CRPS

Source: `benchmarks/results/tuning/eval/*.jsonl`. Each variant's CRPS is averaged over 5 evaluation folds per dataset; the paired test is run across the ten UCI datasets (n=10). Lower CRPS is better, so a negative signed difference favours the left-hand variant.

## Summary

| Pair | Alt. | n | a-wins | b-wins | ties | median Δ | W | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| score+ vs published | less | 10 | 6 | 4 | 0 | -0.0233 | 13.0 | 0.080 |
| FM vs published | less | 10 | 7 | 3 | 0 | -0.0297 | 12.0 | 0.065 |
| FM vs score+ | two-sided | 10 | 6 | 4 | 0 | -0.0004 | 26.0 | 0.922 |

## Per-dataset mean CRPS (5-fold average)

| dataset | published | score+ | FM |
|---|---|---|---|
| california_housing | 0.1971 | 0.2042 | 0.2056 |
| concrete | 2.4567 | 2.1262 | 2.1576 |
| diabetes | 35.6822 | 34.0245 | 34.6603 |
| energy | 0.2519 | 0.2145 | 0.2039 |
| kin8nm | 0.0669 | 0.0576 | 0.0553 |
| naval | 0.0002 | 0.0002 | 0.0002 |
| power_plant | 1.6711 | 1.5716 | 1.5349 |
| protein | 1.7224 | 1.7692 | 1.7958 |
| wine | 0.3004 | 0.3190 | 0.3181 |
| yacht | 0.4375 | 0.2843 | 0.2831 |

## Per-dataset signed differences

### score+ vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0071 |
| concrete | -0.3306 |
| diabetes | -1.6577 |
| energy | -0.0374 |
| kin8nm | -0.0093 |
| naval | +0.0000 |
| power_plant | -0.0996 |
| protein | +0.0468 |
| wine | +0.0186 |
| yacht | -0.1532 |

### FM vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0085 |
| concrete | -0.2991 |
| diabetes | -1.0219 |
| energy | -0.0479 |
| kin8nm | -0.0115 |
| naval | -0.0000 |
| power_plant | -0.1362 |
| protein | +0.0734 |
| wine | +0.0177 |
| yacht | -0.1544 |

### FM vs score+ (alt=two-sided)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0014 |
| concrete | +0.0315 |
| diabetes | +0.6358 |
| energy | -0.0106 |
| kin8nm | -0.0022 |
| naval | -0.0000 |
| power_plant | -0.0367 |
| protein | +0.0266 |
| wine | -0.0009 |
| yacht | -0.0012 |
