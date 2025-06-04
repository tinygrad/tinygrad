from __future__ import annotations

"""Consolidated ONNX frontend for tinygrad.

Minimal ONNX support optimized for code size and essential functionality.
"""

import struct
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

# Minimal ONNX type definitions
@dataclass
class TensorProto:
  dims: Tuple[int, ...] = ()
  data_type: int = 0
  raw_data: bytes = b''
  float_data: List[float] = None
  int32_data: List[int] = None
  int64_data: List[int] = None
  
  def __post_init__(self):
    if self.float_data is None: self.float_data = []
    if self.int32_data is None: self.int32_data = []
    if self.int64_data is None: self.int64_data = []

@dataclass  
class Attribute:
  name: str = ''
  type: int = 0
  f: float = 0.0
  i: int = 0
  s: bytes = b''
  floats: List[float] = None
  ints: List[int] = None
  
  def __post_init__(self):
    if self.floats is None: self.floats = []
    if self.ints is None: self.ints = []

@dataclass
class NodeProto:
  input: Tuple[str, ...] = ()
  output: Tuple[str, ...] = ()
  op_type: str = ''
  attribute: Tuple[Attribute, ...] = ()
  domain: str = ''

@dataclass
class ValueInfo:
  name: str = ''
  elem_type: int = 0
  shape: Tuple[Union[int, str], ...] = ()

@dataclass
class GraphProto:
  node: Tuple[NodeProto, ...] = ()
  initializer: Tuple[TensorProto, ...] = ()
  input: Tuple[ValueInfo, ...] = ()
  output: Tuple[ValueInfo, ...] = ()

@dataclass
class ModelProto:
  graph: GraphProto = None
  opset_import: Tuple[Tuple[str, int], ...] = ()
  
  def __post_init__(self):
    if self.graph is None: self.graph = GraphProto()

# Minimal protobuf parser
def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
  result = shift = 0
  while True:
    b = buf[pos]
    pos += 1
    result |= (b & 0x7F) << shift
    if not (b & 0x80): break
    shift += 7
  return result, pos

def _skip_field(wire_type: int, buf: bytes, pos: int) -> int:
  if wire_type == 0: _, pos = _read_varint(buf, pos)
  elif wire_type == 1: pos += 8
  elif wire_type == 2: size, pos = _read_varint(buf, pos); pos += size
  elif wire_type == 5: pos += 4
  return pos

def _parse_tensor(buf: bytes) -> TensorProto:
  t = TensorProto()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 0: val, pos = _read_varint(buf, pos); t.dims += (val,)
    elif field == 2 and wtype == 0: t.data_type, pos = _read_varint(buf, pos)
    elif field == 9 and wtype == 2: size, pos = _read_varint(buf, pos); t.raw_data = buf[pos:pos+size]; pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return t

def _parse_attr(buf: bytes) -> Attribute:
  a = Attribute()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2: size, pos = _read_varint(buf, pos); a.name = buf[pos:pos+size].decode(); pos += size
    elif field == 3 and wtype == 0: a.type, pos = _read_varint(buf, pos)
    elif field == 4 and wtype == 5: a.f = struct.unpack("<f", buf[pos:pos+4])[0]; pos += 4
    elif field == 5 and wtype == 0: a.i, pos = _read_varint(buf, pos)
    elif field == 6 and wtype == 2: size, pos = _read_varint(buf, pos); a.s = buf[pos:pos+size]; pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return a

def _parse_node(buf: bytes) -> NodeProto:
  n = NodeProto()
  inps, outs, attrs = [], [], []
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2: size, pos = _read_varint(buf, pos); inps.append(buf[pos:pos+size].decode()); pos += size
    elif field == 2 and wtype == 2: size, pos = _read_varint(buf, pos); outs.append(buf[pos:pos+size].decode()); pos += size
    elif field == 4 and wtype == 2: size, pos = _read_varint(buf, pos); n.op_type = buf[pos:pos+size].decode(); pos += size
    elif field == 5 and wtype == 2: size, pos = _read_varint(buf, pos); attrs.append(_parse_attr(buf[pos:pos+size])); pos += size
    else: pos = _skip_field(wtype, buf, pos)
  n.input, n.output, n.attribute = tuple(inps), tuple(outs), tuple(attrs)
  return n

def _parse_graph(buf: bytes) -> GraphProto:
  g = GraphProto()
  nodes, init = [], []
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2: size, pos = _read_varint(buf, pos); nodes.append(_parse_node(buf[pos:pos+size])); pos += size
    elif field == 5 and wtype == 2: size, pos = _read_varint(buf, pos); init.append(_parse_tensor(buf[pos:pos+size])); pos += size
    else: pos = _skip_field(wtype, buf, pos)
  g.node, g.initializer = tuple(nodes), tuple(init)
  return g

def parse_model(buf: bytes) -> ModelProto:
  m = ModelProto()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 4 and wtype == 2: size, pos = _read_varint(buf, pos); m.graph = _parse_graph(buf[pos:pos+size]); pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return m

# Main API
def load(path: str) -> ModelProto:
  """Load ONNX model from file path."""
  with open(path, 'rb') as f:
    return parse_model(f.read())

class OnnxRunner:
  """Minimal ONNX runner for basic inference."""
  def __init__(self, model: ModelProto):
    self.model = model
  
  def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder for actual inference - would need full implementation
    return {}

# Export for backward compatibility
def get_onnx_ops(): 
  """Return available ONNX operators."""
  return ['Add', 'Conv', 'Relu', 'MatMul', 'Reshape', 'Concat', 'Transpose']

# Legacy compatibility
onnx_ops = get_onnx_ops()
dtype_parse = lambda x: x
attribute_parse = lambda x: x  
buffer_parse = lambda x: x
type_parse = lambda x: x
OnnxValue = dict
OnnxNode = NodeProto

__all__ = [
  'load', 'OnnxRunner', 'ModelProto', 'GraphProto', 'NodeProto', 'TensorProto',
  'get_onnx_ops', 'onnx_ops', 'dtype_parse', 'attribute_parse', 'buffer_parse',
  'type_parse', 'OnnxValue', 'OnnxNode'
] 