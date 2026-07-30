from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    # Single source of truth is the version in pyproject.toml, read back from the
    # installed distribution metadata so the two can never drift apart.
    __version__ = _version("diffgbm")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

from diffgbm._conformal import ConformalQuantileCalibrator
from diffgbm.diffgbm import DiffGBM
from diffgbm.samples import Samples

__all__ = ["ConformalQuantileCalibrator", "DiffGBM", "Samples"]
