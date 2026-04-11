"""DeepMT 顶层 CLI 包"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("deepmt")
except Exception:
    __version__ = "0.1.0"

from deepmt.client import DeepMT, TestResult

__all__ = ["DeepMT", "TestResult", "__version__"]
