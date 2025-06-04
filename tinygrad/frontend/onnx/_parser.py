from __future__ import annotations

"""Temporary parser shim.

During the refactor we still rely on the implementation that lives in
`extra/onnx.py`.  This module simply re-exports the pieces that the new
runner will depend on so that importing from the new package keeps
working.  In later commits the code will be rewritten to eliminate the
runtime dependency on the external *onnx* wheel.
"""

from importlib import import_module as _imp
from types import ModuleType as _ModuleType

_extra: _ModuleType = _imp("extra.onnx")

# ---------------------------------------------------------------------------
# Public symbols re-exported from the old module
# ---------------------------------------------------------------------------

dtype_parse = _extra.dtype_parse  # type: ignore[attr-defined]
attribute_parse = _extra.attribute_parse  # type: ignore[attr-defined]
buffer_parse = _extra.buffer_parse  # type: ignore[attr-defined]
type_parse = _extra.type_parse  # type: ignore[attr-defined]

OnnxValue = _extra.OnnxValue  # type: ignore[attr-defined]
OnnxNode = _extra.OnnxNode  # type: ignore[attr-defined]

__all__ = [
    "dtype_parse",
    "attribute_parse",
    "buffer_parse",
    "type_parse",
    "OnnxValue",
    "OnnxNode",
] 