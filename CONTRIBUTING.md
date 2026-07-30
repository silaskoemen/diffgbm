# Contributing

Contributions are welcome and appreciated.

## Bug reports

Please file issues at <https://github.com/silaskoemen/diffgbm/issues> and include:

- Your operating system, Python version, and `diffgbm` version.
- A minimal script that reproduces the problem.
- What you expected to happen, and what happened instead.

## Feature requests

Open an issue describing the use case and how you would expect the API to look.
Keep the scope as narrow as possible — this makes a proposal much easier to
implement and review.

## Development setup

The project uses [pixi](https://pixi.sh) for environment management. Every
command that needs the project environment must run through `pixi run`.

```bash
git clone https://github.com/silaskoemen/diffgbm.git
cd diffgbm
pixi run setup
pixi run test
```

Run the test suite and the linters before opening a pull request:

```bash
pixi run test
pixi run lint
```

If you change `pixi.toml`, regenerate the lockfile with `pixi lock` so that the
lockfile stays consistent with the package metadata.

## Benchmarks

`benchmarks/` holds the research harness used for the paper. Sweeps are driven
by YAML configs under `benchmarks/configs/`:

```bash
pixi run -e bench bench-smoke
```

See `benchmarks/README.md` for the full protocol.

## Pull requests

- Add tests for new behavior.
- Keep the public API documented in docstrings.
- Update `CHANGELOG.md` under an `Unreleased` heading.
