from __future__ import annotations

"""Minimal protobuf wire-format reader for ONNX models.

This *does not* try to be a full Protobuf implementation—just enough to
recover the fields used by the tinygrad ONNX runtime:

• ModelProto → GraphProto → NodeProto / TensorProto / ValueInfoProto
• AttributeProto (limited to FLOAT | INT | STRING | TENSOR | FLOATS | INTS | STRINGS)

Unsupported/unknown fields are skipped with generic heuristics so that
future opsets don't break the parser outright.
"""

import struct
from typing import Tuple, List

from ._schema import (
  Attribute,
  AttributeProto,
  GraphProto,
  ModelProto,
  NodeProto,
  TensorProtoData,
  ValueInfo,
  TensorProto,
)

# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
  """Return (value, new_pos)."""
  result = 0
  shift = 0
  while True:
      b = buf[pos]
      pos += 1
      result |= (b & 0x7F) << shift
      if not (b & 0x80):
      break
      shift += 7
  return result, pos


def _skip_field(wire_type: int, buf: bytes, pos: int) -> int:
  if wire_type == 0:  # varint
      _, pos = _read_varint(buf, pos)
      return pos
  if wire_type == 1:  # 64-bit
      return pos + 8
  if wire_type == 2:  # len-delimited
      size, pos = _read_varint(buf, pos)
      return pos + size
  if wire_type == 5:  # 32-bit
      return pos + 4
  raise ValueError(f"unsupported wire_type {wire_type}")


# ---------------------------------------------------------------------------
# Parsers (bottom-up order)
# ---------------------------------------------------------------------------


def _parse_tensorproto(buf: bytes) -> TensorProtoData:
  t = TensorProtoData()
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 0:  # dims (repeated int64 varint)
      val, pos = _read_varint(buf, pos)
      t.dims += (val,)
      elif field == 2 and wtype == 0:  # data_type (int32)
      val, pos = _read_varint(buf, pos)
      t.data_type = val
      elif field == 9 and wtype == 2:  # raw_data bytes
      size, pos = _read_varint(buf, pos)
      t.raw_data = buf[pos : pos + size]
      pos += size
      elif field == 4 and wtype == 2:  # float_data (packed repeated 32-bit)
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      t.float_data.extend(struct.unpack("<%sf" % (size // 4), sub))
      pos += size
      elif field == 5 and wtype == 2:  # int32_data packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      t.int32_data.extend(struct.unpack("<%si" % (size // 4), sub))
      pos += size
      elif field == 6 and wtype == 2:  # string_data packed (len-delimited strings)
      size, pos = _read_varint(buf, pos)
      limit = pos + size
      while pos < limit:
          slen, pos = _read_varint(buf, pos)
          t.string_data.append(buf[pos : pos + slen])
          pos += slen
      elif field == 7 and wtype == 2:  # int64_data packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      t.int64_data.extend(struct.unpack("<%sq" % (size // 8), sub))
      pos += size
      elif field == 10 and wtype == 2:  # double_data packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      t.double_data.extend(struct.unpack("<%sd" % (size // 8), sub))
      pos += size
      elif field == 11 and wtype == 2:  # uint64_data packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      t.uint64_data.extend(struct.unpack("<%sQ" % (size // 8), sub))
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return t


def _parse_attribute(buf: bytes) -> Attribute:
  a = Attribute()
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 2:  # name
      size, pos = _read_varint(buf, pos)
      a.name = buf[pos : pos + size].decode()
      pos += size
      elif field == 3 and wtype == 0:  # type (enum)
      val, pos = _read_varint(buf, pos)
      a.type = val
      elif field == 4 and wtype == 5:  # float32
      a.f = struct.unpack("<f", buf[pos : pos + 4])[0]
      pos += 4
      elif field == 5 and wtype == 0:  # int64 varint
      a.i, pos = _read_varint(buf, pos)
      elif field == 6 and wtype == 2:  # string
      size, pos = _read_varint(buf, pos)
      a.s = buf[pos : pos + size]
      pos += size
      elif field == 7 and wtype == 2:  # TensorProto
      size, pos = _read_varint(buf, pos)
      a.t = _parse_tensorproto(buf[pos : pos + size])
      pos += size
      elif field == 11 and wtype == 2:  # floats packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      a.floats = list(struct.unpack("<%sf" % (size // 4), sub))
      pos += size
      elif field == 12 and wtype == 2:  # ints packed
      size, pos = _read_varint(buf, pos)
      sub = buf[pos : pos + size]
      a.ints = list(struct.unpack("<%sq" % (size // 8), sub))
      pos += size
      elif field == 13 and wtype == 2:  # strings packed
      size, pos = _read_varint(buf, pos)
      limit = pos + size
      ss: List[bytes] = []
      while pos < limit:
          sl, pos = _read_varint(buf, pos)
          ss.append(buf[pos : pos + sl])
          pos += sl
      a.strings = ss
      else:
      pos = _skip_field(wtype, buf, pos)
  return a


def _parse_valueinfo(buf: bytes) -> ValueInfo:
  vi = ValueInfo()
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 2:  # name
      size, pos = _read_varint(buf, pos)
      vi.name = buf[pos : pos + size].decode()
      pos += size
      elif field == 2 and wtype == 2:  # type (TypeProto)
      size, pos = _read_varint(buf, pos)
      # We only care about elem_type and shape dims.
      vi = _handle_typeproto(buf[pos : pos + size], vi)
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return vi


def _handle_typeproto(buf: bytes, vi: ValueInfo) -> ValueInfo:
  # Parse TypeProto focusing on elem_type and TensorShapeProto.
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 0:  # tensor_type.elem_type (varint)
      val, pos = _read_varint(buf, pos)
      vi.elem_type = val
      elif field == 1 and wtype == 2:  # nested message tensor_type
      size, pos = _read_varint(buf, pos)
      vi = _parse_tensortype(buf[pos : pos + size], vi)
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return vi


def _parse_tensortype(buf: bytes, vi: ValueInfo) -> ValueInfo:
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 0:  # elem_type
      vi.elem_type, pos = _read_varint(buf, pos)
      elif field == 2 and wtype == 2:  # shape
      size, pos = _read_varint(buf, pos)
      vi.shape = _parse_tensorshape(buf[pos : pos + size])
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return vi


def _parse_tensorshape(buf: bytes) -> Tuple[int | str, ...]:
  dims: List[int | str] = []
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 2:  # dim message
      size, pos = _read_varint(buf, pos)
      dim_bytes = buf[pos : pos + size]
      dim_val = _parse_dim(dim_bytes)
      dims.append(dim_val)
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return tuple(dims)


def _parse_dim(buf: bytes) -> int | str:
  pos = 0
  while pos < len(buf):
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 0:  # dim_value
      val, pos = _read_varint(buf, pos)
      return val
      elif field == 2 and wtype == 2:  # dim_param
      size, pos = _read_varint(buf, pos)
      return buf[pos : pos + size].decode()
      else:
      pos = _skip_field(wtype, buf, pos)
  return -1


def _parse_node(buf: bytes) -> NodeProto:
  n = NodeProto()
  pos = 0
  end = len(buf)
  inps: List[str] = []
  outs: List[str] = []
  attrs: List[Attribute] = []
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 2:  # input string
      size, pos = _read_varint(buf, pos)
      inps.append(buf[pos : pos + size].decode())
      pos += size
      elif field == 2 and wtype == 2:  # output string
      size, pos = _read_varint(buf, pos)
      outs.append(buf[pos : pos + size].decode())
      pos += size
      elif field == 4 and wtype == 2:  # op_type
      size, pos = _read_varint(buf, pos)
      n.op_type = buf[pos : pos + size].decode()
      pos += size
      elif field == 5 and wtype == 2:  # attribute
      size, pos = _read_varint(buf, pos)
      attrs.append(_parse_attribute(buf[pos : pos + size]))
      pos += size
      elif field == 7 and wtype == 2:  # domain
      size, pos = _read_varint(buf, pos)
      n.domain = buf[pos : pos + size].decode()
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  n.input = tuple(inps)
  n.output = tuple(outs)
  n.attribute = tuple(attrs)
  return n


def _parse_graph(buf: bytes) -> GraphProto:
  g = GraphProto()
  pos = 0
  end = len(buf)
  nodes: List[NodeProto] = []
  init: List[TensorProtoData] = []
  inputs: List[ValueInfo] = []
  outputs: List[ValueInfo] = []
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 1 and wtype == 2:  # node
      size, pos = _read_varint(buf, pos)
      nodes.append(_parse_node(buf[pos : pos + size]))
      pos += size
      elif field == 5 and wtype == 2:  # initializer
      size, pos = _read_varint(buf, pos)
      init.append(_parse_tensorproto(buf[pos : pos + size]))
      pos += size
      elif field == 11 and wtype == 2:  # input ValueInfo
      size, pos = _read_varint(buf, pos)
      inputs.append(_parse_valueinfo(buf[pos : pos + size]))
      pos += size
      elif field == 12 and wtype == 2:  # output ValueInfo
      size, pos = _read_varint(buf, pos)
      outputs.append(_parse_valueinfo(buf[pos : pos + size]))
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  g.node = tuple(nodes)
  g.initializer = tuple(init)
  g.input = tuple(inputs)
  g.output = tuple(outputs)
  return g


def parse_model(buf: bytes) -> ModelProto:
  m = ModelProto()
  pos = 0
  end = len(buf)
  while pos < end:
      tag, pos = _read_varint(buf, pos)
      field = tag >> 3
      wtype = tag & 0x07
      if field == 4 and wtype == 2:  # graph
      size, pos = _read_varint(buf, pos)
      m.graph = _parse_graph(buf[pos : pos + size])
      pos += size
      elif field == 8 and wtype == 2:  # opset_import (OperatorSetIdProto)
      size, pos = _read_varint(buf, pos)
      # OperatorSetIdProto is simple: field 1 domain (string), 2 version (int64)
      dom = ""
      ver = 0
      subpos = pos
      limit = pos + size
      while subpos < limit:
          stag, subpos = _read_varint(buf, subpos)
          sf = stag >> 3
          stype = stag & 0x07
          if sf == 1 and stype == 2:
        sz, subpos = _read_varint(buf, subpos)
        dom = buf[subpos : subpos + sz].decode()
        subpos += sz
          elif sf == 2 and stype == 0:
        ver, subpos = _read_varint(buf, subpos)
          else:
        subpos = _skip_field(stype, buf, subpos)
      m.opset_import += ((dom, ver),)
      pos += size
      else:
      pos = _skip_field(wtype, buf, pos)
  return m
