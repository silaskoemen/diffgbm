
# Run tests
test:
	pixi run -e test test

lint:
	pixi run -e lint lint

bench-smoke:
	pixi run -e bench bench-smoke

#creates a virtual environment and installs the required packages
make-dev:
	pixi install
	pixi run -e lint pre-commit-install
