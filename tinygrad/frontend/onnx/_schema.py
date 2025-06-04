from __future__ import annotations

"""Minimal ONNX protobuf schema (work-in-progress).

Only the enum constants and fields that *tinygrad* currently touches are
represented.  The aim is to break the import dependency on the external
*onnx* wheel while keeping behavioural parity.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Sequence, Tuple

__all__ = [
    # Enum classes
    "TensorProto",
    "AttributeProto",
    "TypeProto",
    "ModelProto",
]

# ---------------------------------------------------------------------------
# Enums – replicated values from onnx.proto
# ---------------------------------------------------------------------------


class TensorProto(IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22


class AttributeProto(IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 11
    TYPE_PROTO = 13
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12
    TYPE_PROTOS = 14

# ---------------------------------------------------------------------------
# Tensor & attribute containers
# ---------------------------------------------------------------------------


@dataclass
class TensorProtoData:
    """A heavily trimmed representation of ONNX *TensorProto*."""

    dims: Tuple[int, ...] = field(default_factory=tuple)
    data_type: int = TensorProto.UNDEFINED
    raw_data: bytes | None = None

    # typed repeated fields (only those we actually read)
    float_data: List[float] = field(default_factory=list)
    int32_data: List[int] = field(default_factory=list)
    int64_data: List[int] = field(default_factory=list)
    double_data: List[float] = field(default_factory=list)
    uint64_data: List[int] = field(default_factory=list)
    string_data: List[bytes] = field(default_factory=list)


@dataclass
class Attribute:
    name: str = ""
    type: int = AttributeProto.UNDEFINED
    f: float | None = None
    i: int | None = None
    s: bytes | None = None
    t: TensorProtoData | None = None
    floats: Sequence[float] = field(default_factory=list)
    ints: Sequence[int] = field(default_factory=list)
    strings: Sequence[bytes] = field(default_factory=list)


@dataclass
class ValueInfo:
    name: str = ""
    # Only tensor_type matters for tinygrad currently
    elem_type: int = TensorProto.UNDEFINED
    shape: Tuple[int | str, ...] = field(default_factory=tuple)


@dataclass
class NodeProto:
    op_type: str = ""
    domain: str = ""
    input: Tuple[str, ...] = field(default_factory=tuple)
    output: Tuple[str, ...] = field(default_factory=tuple)
    attribute: Tuple[Attribute, ...] = field(default_factory=tuple)


@dataclass
class GraphProto:
    node: Tuple[NodeProto, ...] = field(default_factory=tuple)
    initializer: Tuple[TensorProtoData, ...] = field(default_factory=tuple)
    input: Tuple[ValueInfo, ...] = field(default_factory=tuple)
    output: Tuple[ValueInfo, ...] = field(default_factory=tuple)


@dataclass
class ModelProto:
    graph: GraphProto = field(default_factory=GraphProto)
    opset_import: Tuple[Any, ...] = field(default_factory=tuple)  # version as int only 

# ---------------------------------------------------------------------------
# Additional minimal structures
# ---------------------------------------------------------------------------

@dataclass
class TensorShape:
    dim: Tuple[int | str, ...] = field(default_factory=tuple)


@dataclass
class TensorType:
    elem_type: int = TensorProto.UNDEFINED
    shape: TensorShape = field(default_factory=TensorShape)


@dataclass
class TypeProto:
    """Subset of ONNX *TypeProto* we care about (only tensor)."""

    tensor_type: TensorType | None = None
    sequence_type: "TypeProto" | None = None
    optional_type: "TypeProto" | None = None
    # map_type, sparse_tensor_type, opaque_type not used 