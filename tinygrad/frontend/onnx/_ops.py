from __future__ import annotations

"""Temporary ONNX ops shim that forwards to the existing implementation in
`extra.onnx`.  This lets the new package layout stabilise before we migrate
and slim the operator definitions.
"""

from importlib import import_module as _imp
from types import ModuleType as _ModuleType

_extra: _ModuleType = _imp("extra.onnx")

get_onnx_ops = _extra.get_onnx_ops  # type: ignore[attr-defined]
onnx_ops = _extra.onnx_ops  # type: ignore[attr-defined]

__all__ = ["get_onnx_ops", "onnx_ops"] 