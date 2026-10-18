"""agentmemeval: evaluation framework for AI memory systems and tool-calling agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentmemeval")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
