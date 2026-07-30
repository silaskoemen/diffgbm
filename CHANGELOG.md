# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.2

### Fixed

- `diffgbm.__version__` is now read from the installed distribution metadata
  instead of being hardcoded, so it can no longer drift from the version in
  `pyproject.toml` (0.1.1 shipped reporting `0.1.0`).
- README images now use absolute URLs, so the logo and example plot render on
  the PyPI project page rather than only on GitHub.

## 0.1.1

First published release on PyPI. Identical in functionality to 0.1.0, which was
tagged before the release workflow existed and was never published.

## 0.1.0

First release of `diffgbm`.

DiffGBM is a derivative of [Treeffuser](https://github.com/blei-lab/treeffuser)
(MIT, see [NOTICE](NOTICE)) that treats the noising path, parameterization,
training distribution, features, and sampler as tunable conditioning choices
rather than fixed defaults.

### Added

- **Flow matching on Gaussian paths** (`training_objective="flow_matching"`)
  with linear, trigonometric, and variance-preserving paths, a velocity-to-score
  identity for stochastic-interpolant sampling, and few-step deterministic Heun
  ODE sampling.
- **Conditional-mean residualization** (`residualize=`), which fits a
  cross-validated LightGBM mean predictor and diffuses the residual.
- **EDM-style preconditioning** and alternative score parameterizations
  (`score_parameterization=` `"noise" | "x0" | "edm"`).
- **Log-sigma time sampling** (`t_sampling=`) and an explicit log-sigma noise
  feature (`noise_features=`).
- **Loss weighting**, including min-SNR-gamma (`loss_weighting=`).
- **Probability-flow ODE and Heun samplers** alongside the Euler SDE sampler.
- **Conformal quantile calibration** via `ConformalQuantileCalibrator`.

### Changed

- Package renamed from `treeffuser` to `diffgbm`; the estimator class is now
  `DiffGBM`. There is no `Treeffuser` alias — `diffgbm` is a new distribution,
  so no existing code can depend on the old name.
- **Defaults are the score-side recipe**, not the published Treeffuser one:
  `score_parameterization="edm"`, `noise_features="raw_time_log_std"`,
  `t_sampling="log_sigma_normal"`, `residualize="mean"`. The published recipe is
  the explicit configuration `score_parameterization="noise"`,
  `noise_features="raw_time"`, `t_sampling="uniform"`, `residualize="off"`.
- Because residualization is now on by default, a fit with fewer than 80 training
  rows falls back to an unresidualized model with a warning instead of raising:
  too little data for cross-fitting must not break a default.
- Flow matching always uses the raw-time feature layout and ignores the
  score-only knobs, warning when they are set away from their defaults.
- `max_bin` is exposed as a first-class histogram-resolution knob.

### Notes

The published Treeffuser recipe remains reachable as a special case of the
configuration space (noise prediction, uniform `t`, raw time feature, no
residualization, Euler SDE), and is the baseline reported in the paper.
