from __future__ import annotations

"""tinygrad ONNX frontend package.

Initially this re-exports the implementation that still lives in
`extra/onnx.py` so that external imports can begin using the stable
`tinygrad.frontend.onnx` path.  Follow-up commits will migrate the code
fully into this package and delete the extra module.
"""

# NOTE: we intentionally do *not* import everything with `import *` at the
# top-level because that would leak a lot of symbols into the public
# namespace before mypy/pylint can analyse them.  Instead we lazily forward
# attribute access to the backing module.  This keeps the diff minimal
# while giving us full flexibility to replace the implementation later.

from importlib import import_module as _imp
from types import ModuleType as _ModuleType
from typing import Any as _Any

_impl: _ModuleType | None = None


def _load_impl() -> _ModuleType:
    global _impl  # noqa: PLW0603
    if _impl is None:
        _impl = _imp("extra.onnx")
    return _impl


# Public API -----------------------------------------------------------------
# These names are historically provided by extra.onnx and are needed by the
# existing codebase/tests.  We expose them via __getattr__ so they remain
# live-forwarded to whatever implementation we eventually settle on.

def __getattr__(name: str) -> _Any:  # noqa: D401,E501  (simple getter)
    mod = _load_impl()
    try:
        return getattr(mod, name)
    except AttributeError as err:
        raise AttributeError(f"module 'tinygrad.frontend.onnx' has no attribute {name!r}") from err


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(dir(_load_impl())))


# Public re-exports from the new internal modules. These names shadow the
# ones from the legacy implementation (if present) so that call-sites start
# picking up the new package layout immediately.

from ._parser import (
    OnnxValue,  # noqa: F401
    OnnxNode,  # noqa: F401
    dtype_parse,  # noqa: F401
    attribute_parse,  # noqa: F401
    buffer_parse,  # noqa: F401
    type_parse,  # noqa: F401
)
from ._ops import get_onnx_ops, onnx_ops  # noqa: F401
from ._runner import OnnxRunner  # noqa: F401
from ._loader import load  # noqa: F401

__all__: list[str] = [
    "OnnxRunner",
    "OnnxValue",
    "OnnxNode",
    "dtype_parse",
    "attribute_parse",
    "buffer_parse",
    "type_parse",
    "get_onnx_ops",
    "onnx_ops",
    "load",
] 