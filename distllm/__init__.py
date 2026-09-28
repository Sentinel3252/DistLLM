"""Token-correct distributed training with explicit communication boundaries."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("distllm")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
