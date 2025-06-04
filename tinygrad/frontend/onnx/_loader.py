from __future__ import annotations

"""ONNX model loader.

Placeholder implementation that still relies on the external *onnx* wheel.
A subsequent commit will parse the protobuf directly so that tinygrad can
run without any third-party ONNX dependency.
"""

from pathlib import Path
from typing import Any, Union

__all__ = ["load"]

PathLike = Union[str, Path]


def _read_bytes(src: PathLike | bytes | bytearray) -> bytes:
    if isinstance(src, (str, Path)):
        p = Path(src).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        return p.read_bytes()
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    raise TypeError("path_or_bytes must be str, Path, bytes or bytearray")


def load(path_or_bytes: PathLike | bytes | bytearray) -> Any:  # noqa: ANN401
    """Return a ModelProto-like object.

    For now we simply call ``onnx.load_model_from_string`` if the *onnx*
    package is available.  When the pure-Python parser lands this function
    will return an instance of :class:`tinygrad.frontend.onnx._schema.ModelProto`.
    """

    blob = _read_bytes(path_or_bytes)

    try:
        import onnx  # pylint: disable=import-error
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise ImportError(
            "Built-in parser not ready and the 'onnx' package is not installed."
        ) from exc

    return onnx.load_model_from_string(blob) 