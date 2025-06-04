from __future__ import annotations

"""Wrapper around the legacy `extra.onnx.OnnxRunner`.

Keeps public surface identical while allowing us to eventually swap in a
new implementation.
"""

from importlib import import_module as _imp
from types import ModuleType as _ModuleType

_extra: _ModuleType = _imp("extra.onnx")

# Re-export the existing class for now -----------------------------------------------------------

LegacyOnnxRunner = _extra.OnnxRunner  # type: ignore[attr-defined]


class OnnxRunner(LegacyOnnxRunner):
    """Placeholder subclass.

    For the moment we inherit directly from the original implementation to
    ensure perfect behavioural parity.  When the protobuf-free parser and
    slimmed operator set are ready we can replace this with a fresh class
    that depends only on the local `_parser` and `_ops` modules.
    """

    # No overrides yet – this is purely a type alias / forwarder.

    pass


__all__ = ["OnnxRunner"] 