from typing import Any, Sequence, cast, Literal, Callable
import dataclasses, functools, io, math, types, os, pathlib, struct, enum
from io import BufferedReader
from tinygrad.nn.state import TensorIO
from tinygrad.tensor import Tensor, _broadcast_shape, ReductionStr
from tinygrad.helpers import getenv, DEBUG, all_same, prod, flatten, make_tuple, argsort, get_single_element
from tinygrad.dtype import DType, ConstType, dtypes, ImageDType
from tinygrad.device import is_dtype_supported, Device

debug = int(getenv("DEBUGONNX", "0"))
limit = int(getenv("ONNXLIMIT", "-1"))

# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

# https://github.com/protocolbuffers/protobuf/blob/main/python/google/protobuf/internal/wire_format.py#L24-L29
class WireType(enum.Enum):
  WIRETYPE_VARINT = 0; WIRETYPE_FIXED64 = 1; WIRETYPE_LENGTH_DELIMITED = 2; WIRETYPE_START_GROUP = 3; WIRETYPE_END_GROUP = 4; WIRETYPE_FIXED32 = 5 # noqa: E702
class PBType(enum.Enum):
  FLOAT = 1; INT = 2; STRING = 3; FLOATS = 4; INTS = 5; STRINGS = 6; BYTES = 7; SUB = 8 # noqa: E702

PB_INFOS = {
  "OperatorSetIdProto": {1: ("domain", PBType.STRING), 2: ("version", PBType.INT)},
  "StringStringEntryProto": {1: ("key", PBType.STRING), 2: ("value", PBType.STRING)},
  "TensorProto": {1: ("dims", PBType.INT, True), 2: ("data_type", PBType.INT), 4: ("float_data", PBType.FLOATS),
    5: ("int32_data", PBType.INTS), 7: ("int64_data", PBType.INTS), 8: ("name", PBType.STRING), 9: ("raw_data", PBType.BYTES),
    13: ("external_data", PBType.SUB, True, "StringStringEntryProto"), 14: ("data_location", PBType.INT)},
  "TensorShapeProtoDimension": {1: ("dim_value", PBType.INT), 2: ("dim_param", PBType.STRING)},
  "TensorShapeProto": {1: ("dim", PBType.SUB, True, "TensorShapeProtoDimension")},
  "ModelProto": {1: ("ir_version", PBType.INT), 5: ("model_version", PBType.INT),
    2: ("producer_name", PBType.STRING), 3: ("producer_version", PBType.STRING), 4: ("domain", PBType.STRING), 6: ("doc_string", PBType.STRING),
    7: ("graph", PBType.SUB, False, ("GraphProto", lambda: {"node": [], "initializer": [], "input": [], "output": [], "value_info": []})),
    8: ("opset_import",PBType.SUB, True, "OperatorSetIdProto")},
  "GraphProto": {2: ("name", PBType.STRING), 10: ("doc_string", PBType.STRING),
    1: ("node", PBType.SUB, True, ("NodeProto", lambda: {"input": [], "output": [], "attribute": [], "domain": None})),
    5: ("initializer", PBType.SUB, True, ("TensorProto", lambda: {"dims": [], "float_data": [], "int32_data": [], "string_data": [],
                                                                  "int64_data": [], "double_data": [], "uint64_data": []})),
    11: ("input", PBType.SUB, True, "ValueInfoProto"), 12: ("output", PBType.SUB, True, "ValueInfoProto")},
  "NodeProto": {1: ("input", PBType.STRING, True), 2: ("output", PBType.STRING, True), 3: ("name", PBType.STRING),
    4: ("op_type", PBType.STRING), 5: ("attribute", PBType.SUB, True, ("AttributeProto", lambda: {"floats": [], "ints": [], "strings": []})),
    6: ("doc_string", PBType.STRING), 7: ("domain", PBType.STRING),},
  "AttributeProto": {1: ("name", PBType.STRING), 2: ("f", PBType.FLOAT), 3: ("i", PBType.INT), 4: ("s", PBType.BYTES),
    5:("t", PBType.SUB, False, ("TensorProto", lambda: {"dims": [], "float_data": [], "int32_data": [], "string_data": [], "int64_data": [],
                                                        "double_data": [], "uint64_data": []})),
    7: ("floats", PBType.FLOAT, True), 8: ("ints", PBType.INT, True), 9: ("strings", PBType.BYTES, True),
    20: ("type", PBType.INT)},
  "ValueInfoProto": {1: ("name", PBType.STRING), 2: ("type", PBType.SUB, False, "TypeProto"), 3: ("doc_string", PBType.STRING)},
  "TypeProto": {1: ("tensor_type", PBType.SUB, False, "TypeProtoTensor"), 4: ("sequence_type", PBType.SUB, False, "TypeProtoSequence"),
    9: ("optional_type", PBType.SUB, False, "TypeProtoOptional"), 6: ("denotation", PBType.STRING)},
  "TypeProtoSequence": {1: ("elem_type", PBType.SUB, False, "TypeProto")},
  "TypeProtoOptional": {1: ("elem_type", PBType.SUB, False, "TypeProto")},
  "TypeProtoTensor": {1: ("elem_type", PBType.INT), 2: ("shape", PBType.SUB, False, ("TensorShapeProto", lambda: {"dim": []}))},
}

# TensorProto.DataType
class TensorDataType(enum.Enum):
  UNDEFINED = 0; FLOAT = 1; UINT8 = 2; INT8 = 3; UINT16 = 4; INT16 = 5; INT32 = 6; INT64 = 7 # noqa: E702
  STRING = 8; BOOL = 9; FLOAT16 = 10; DOUBLE = 11; UINT32 = 12; UINT64 = 13; COMPLEX64 = 14; COMPLEX128 = 15; BFLOAT16 = 16 # noqa: E702

_DTYPE_ENUM_TO_TINYGRAD_DTYPE_MAP: dict[int, DType] = {
  TensorDataType.FLOAT: dtypes.float32, TensorDataType.UINT8: dtypes.uint8, TensorDataType.INT8: dtypes.int8,
  TensorDataType.UINT16: dtypes.uint16, TensorDataType.INT16: dtypes.int16, TensorDataType.INT32: dtypes.int32,
  TensorDataType.INT64: dtypes.int64, TensorDataType.BOOL: dtypes.bool, TensorDataType.FLOAT16: dtypes.float32,
  TensorDataType.DOUBLE: dtypes.double, TensorDataType.UINT32: dtypes.uint32, TensorDataType.UINT64: dtypes.uint64,
  TensorDataType.BFLOAT16: dtypes.bfloat16
}
# AttributeProto.AttributeType
class AttributeType(enum.Enum):
  UNDEFINED = 0; FLOAT = 1; INT = 2; STRING = 3; TENSOR = 4; GRAPH = 5; FLOATS = 6; INTS = 7; STRINGS = 8; TENSORS = 9; GRAPHS = 10; # noqa: E702
  SPARSE_TENSOR = 11; SPARSE_TENSORS = 12; TYPE_PROTO = 13; TYPE_PROTOS = 14 # noqa: E702

_ATTR_TYPE_ENUM_TO_HANDLER_MAP: dict[str, Callable[[dict], Any]] = {
  AttributeType.FLOAT: lambda a: float(a['f']),
  AttributeType.INT: lambda a: int(a['i']),
  AttributeType.STRING: lambda a: a["s"].data().tobytes().decode("utf8") if isinstance(a["s"], Tensor) else a["s"].decode("utf8"),
  AttributeType.TENSOR: lambda a: buffer_parse(a['t']),
  AttributeType.FLOATS: lambda a: tuple(float(x) for x in a.get('floats', [])),
  AttributeType.INTS: lambda a: tuple(int(x) for x in a.get('ints', [])),
  AttributeType.STRINGS: lambda a: tuple(s.data().tobytes().decode('utf-8') for s in a.get('strings', []))
}

def gen_result(obj: dict, key_name, val, repeated: bool):
  if repeated: obj.setdefault(key_name, []).append(val)
  else: obj[key_name] = val

def decode_varint(reader: BufferedReader) -> int:
  result = 0
  shift = 0
  while True:
    data = reader.read(1)
    if data == b"": raise EOFError("decode_varint EOF")
    result |= (data[0] & 0x7F) << shift
    if not (data[0] & 0x80): return result
    shift += 7
    if shift >= 70: raise ValueError("Varint too long")

def skip_field_value(reader: BufferedReader, wire_type):
  if wire_type == WireType.WIRETYPE_VARINT: decode_varint(reader)
  elif wire_type == WireType.WIRETYPE_FIXED64: reader.seek(8, os.SEEK_CUR)
  elif wire_type == WireType.WIRETYPE_FIXED32: reader.seek(4, os.SEEK_CUR)
  elif wire_type == WireType.WIRETYPE_LENGTH_DELIMITED: reader.seek(decode_varint(reader), os.SEEK_CUR)
  else: raise ValueError(f"Unknown wire type: {wire_type}")

class OnnxParser:
  def __init__(self, model: pathlib.Path, load_external_data: bool=True):
    self.file_path = model
    self.model = Tensor(model)
    self.load_external_data = load_external_data

    attr_func_dict = {PBType.BYTES: self._handle_bytes, PBType.SUB: self._handle_sub_message, PBType.FLOATS: self._handle_packed_floats,
      PBType.INT: self._handle_int64, PBType.INTS: self._handle_packed_int64s, PBType.STRING: self._handle_string, PBType.FLOAT: self._handle_float}

    registered_handles = {}
    for pb_name in PB_INFOS:
      res = {}
      for fid, config in PB_INFOS[pb_name].items():
        parser_fn, repeated = None, False
        if len(config) == 2: name, attr = config
        elif len(config) == 3: name, attr, repeated = config
        elif len(config) == 4: name, attr, repeated, parser_fn = config
        handler_fn = attr_func_dict[attr]
        def _wrapper_handler(obj, reader, wt, h=handler_fn, n=name, p=parser_fn, r=repeated): return h(obj, n, reader, wt, parser_func=p, repeated=r)
        _wrapper_handler._debug_info = f"{fid}, {name} => {handler_fn}"
        res[fid] = _wrapper_handler
      registered_handles[pb_name] = res
    self.registered_handles = registered_handles

  def parse(self):
    reader = BufferedReader(TensorIO(self.model))
    return self._parse_message(reader, "ModelProto", lambda: {"opset_import": [], "domain": None, "graph": None})

  def _parse_message(self, reader, message_field_handlers_name, initial_obj_factory=lambda: {}):
    message_field_handlers = self.registered_handles[message_field_handlers_name]
    obj = initial_obj_factory()
    while True:
      try:
        tag_val = decode_varint(reader)
        field_number = tag_val >> 3
        wire_type = WireType(tag_val & 0x07)
        if handler := message_field_handlers.get(field_number):
          if debug >= 2: print(handler._debug_info)
          handler(obj, reader, wire_type)
        else: skip_field_value(reader, wire_type)
      except EOFError: break
    if message_field_handlers_name == "TensorProto" and self.load_external_data and obj.get("data_location", 0) == 1:
      self._parse_external_data(obj)
    return obj

  def _handle_delimited(self, reader:BufferedReader, use_tensor=False) -> bytes | Tensor:
    str_len = decode_varint(reader)
    if not use_tensor: return reader.read(str_len)
    res = reader.raw._tensor[reader.tell():(reader.tell()+str_len)]
    reader.seek(str_len, os.SEEK_CUR)
    return res

  def _handle_string(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for string field '{key_name}'")
    val = self._handle_delimited(reader)
    val = val.decode("utf-8")
    gen_result(obj, key_name, val, repeated)

  def _handle_bytes(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for bytes field '{key_name}'")
    val = self._handle_delimited(reader, use_tensor=True)
    gen_result(obj, key_name, val, repeated)

  def _handle_int64(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_VARINT: raise ValueError(f"Expected varint for int64 field '{key_name}'")
    val = decode_varint(reader)
    val = val - 2**64 if val & (1 << 63) else val
    gen_result(obj, key_name, val, repeated)

  def _handle_float(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_FIXED32: raise ValueError(f"Expected fixed32 for float field '{key_name}'")
    val = get_single_element(struct.unpack("<f", reader.read(4)))
    gen_result(obj, key_name, val, repeated)

  def _handle_packed_int64s(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed int64s expected length_delimited")
    total_bytes_len = decode_varint(reader)
    old_pos = reader.tell()
    values = []
    while reader.tell() < total_bytes_len + old_pos:
      val = decode_varint(reader) # need copy here because packed ints are varint
      values.append(val - 2**64 if val & (1 << 63) else val)
    obj[key_name] = Tensor(values, dtype=dtypes.int64)

  def _handle_packed_floats(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed floats expected length_delimited")
    value = self._handle_delimited(reader, use_tensor=True)
    obj[key_name] = value.bitcast(dtypes.float32)

  def _handle_sub_message(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WireType.WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for sub-message field '{key_name}'")
    value = self._handle_delimited(reader, use_tensor=True)
    if isinstance(parser_func, str): sub_obj = self._parse_message(BufferedReader(TensorIO(value)), parser_func)
    elif isinstance(parser_func, tuple): sub_obj = self._parse_message(BufferedReader(TensorIO(value)), parser_func[0], parser_func[1])
    else: sub_obj = parser_func(BufferedReader(TensorIO(value)))
    gen_result(obj, key_name, sub_obj, repeated)

  def _parse_external_data(self, obj):
    if "external_data" not in obj: raise ValueError("no external_data")
    if self.file_path.suffix != ".onnx": raise ValueError("External data requires the input model path to have `.onnx` as extension")
    location = None
    length = None
    offset = 0
    for kv in obj["external_data"]:
      if kv["key"] == "location": location = kv["value"]
      if kv["key"] == "offset": offset = int(kv["value"])
      if kv["key"] == "length": length = int(kv["value"])
    if location is None: raise ValueError("no location in external_data")
    ext_path = self.file_path.parent.joinpath(location)
    if not ext_path.exists(): raise RuntimeError("External data must be placed in the same folder as the onnx model")
    ext_tensor = Tensor(ext_path)
    obj["raw_data"] = ext_tensor[offset:offset+length] if length is not None else ext_tensor[offset:]
    obj["data_location"] = 0

# ***** onnx protobuf parsing ******
import numpy as np
from onnx import helper

def buffer_parse(onnx_tensor: dict) -> Tensor:
  if onnx_tensor["string_data"]: raise NotImplementedError("Parsing for buffer with string data is not implemented.")
  dtype, shape = dtype_parse(onnx_tensor["data_type"]), tuple(onnx_tensor["dims"])
  data = None
  if len(onnx_tensor["float_data"]): data = onnx_tensor["float_data"]
  elif len(onnx_tensor["int32_data"]): data = onnx_tensor["int32_data"]
  elif len(onnx_tensor["int64_data"]): data = onnx_tensor["int64_data"]
  elif len(onnx_tensor["double_data"]): data = onnx_tensor["double_data"]
  elif len(onnx_tensor["uint64_data"]): data = onnx_tensor["uint64_data"]
  if isinstance(data, Tensor):
    if len(data) == 1: return Tensor(data.tolist()[0], dtype=dtype).reshape(shape)
    return data.cast(dtype).reshape(shape).to(Device.DEFAULT)
  if isinstance(onnx_tensor.get("raw_data"), Tensor):
    if TensorDataType(onnx_tensor["data_type"]) is TensorDataType.FLOAT16:
      np_buffer = np.frombuffer(onnx_tensor["raw_data"].data().tobytes(),
                                dtype=helper.tensor_dtype_to_np_dtype(onnx_tensor["data_type"])).copy().reshape(shape)
      if np_buffer.size == 1: return Tensor(np_buffer.item(), dtype=dtype).reshape(shape)
      return Tensor(np_buffer, dtype=dtype)
    ret = onnx_tensor["raw_data"].bitcast(dtype).reshape(shape).to(Device.DEFAULT)
    if shape == (): ret = Tensor(ret.item(), dtype=dtype).reshape(shape)
    return ret
  return Tensor(None)

def dtype_parse(onnx_dtype: int) -> DType:
  dtype = _DTYPE_ENUM_TO_TINYGRAD_DTYPE_MAP.get(TensorDataType(onnx_dtype))
  if dtype is None: raise NotImplementedError(f"onnx dtype {onnx_dtype} is not in the supported tinygrad mapping.")
  return dtype if is_dtype_supported(dtype) else dtypes.float32

def attribute_parse(onnx_attribute: dict):
  attr_type = onnx_attribute['type']
  handler = _ATTR_TYPE_ENUM_TO_HANDLER_MAP.get(AttributeType(attr_type))
  if not handler: raise NotImplementedError(f"handler for attribute type {attr_type} is not implemented.")
  return handler(onnx_attribute)

def type_parse(onnx_type: dict):
  elem_type_dict = onnx_type
  if 'map_type' in elem_type_dict or 'sparse_tensor_type' in elem_type_dict or 'opaque_type' in elem_type_dict:
    raise NotImplementedError("Parsing for map_type, sparse_tensor_type, and opaque_type is not implemented.")
  if is_optional := 'optional_type' in elem_type_dict: elem_type_dict = elem_type_dict['optional_type']['elem_type']
  if is_sequence := 'sequence_type' in elem_type_dict: elem_type_dict = elem_type_dict['sequence_type']['elem_type']
  if 'tensor_type' in elem_type_dict:
    tensor_type_info = elem_type_dict['tensor_type']
    shape_dims = tensor_type_info.get('shape', {}).get('dim', [])
    parsed_shape = tuple(d.get('dim_param') or d.get('dim_value') for d in shape_dims)
    dtype = dtype_parse(tensor_type_info['elem_type'])
    return OnnxValue(parsed_shape, dtype, is_optional, is_sequence)
  raise RuntimeError(f"TypeProto dictionary was not parsed properly: {onnx_type=}")

def model_parse(onnx_model: dict):
  opset_version = int(onnx_model['opset_import'][0]['version'])
  graph = onnx_model['graph']
  initializers, nodes, inputs, outputs = [graph.get(key, []) for key in ('initializer', 'node', 'input', 'output')]
  is_training = any(n.get("domain") in {"ai.onnx.training", "ai.onnx.preview.training"} for n in nodes)
  values = {"": None, **{x['name']:buffer_parse(x) for x in initializers}}
  inputs = {inp['name']: type_parse(inp['type']) for inp in inputs if inp['name'] not in values}
  outputs = tuple(out['name'] for out in outputs)
  nodes_list = []
  for i, node_dict in enumerate(nodes):
    attributes = {attr_dict['name']: attribute_parse(attr_dict) for attr_dict in node_dict.get('attribute', [])}
    node_inputs = tuple(node_dict.get('input', []))
    node_outputs = tuple(node_dict.get('output', []))
    nodes_list.append(OnnxNode(num=i, op=node_dict['op_type'], inputs=node_inputs, outputs=node_outputs, opts=attributes))
  nodes = tuple(nodes_list)
  return is_training, values, inputs, outputs, nodes, opset_version

# ***** onnx spec *****
@dataclasses.dataclass(frozen=True)
class OnnxValue:
  shape: tuple[str|int, ...]
  dtype: DType
  is_optional: bool
  is_sequence: bool

@dataclasses.dataclass(frozen=True)
class OnnxNode:
  num: int
  op: str
  inputs: tuple[str, ...]
  outputs: tuple[str, ...]
  opts: dict[str, Any]

# ***** python const *****
required_input_python_consts: dict[str, tuple[int, ...]] = {
  "Tile": (1,), "Range": (0,1,2), "Expand": (1,), "Reshape": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Trilu": (1,), "ConstantOfShape": (0,),
  "CumSum": (1,), "TopK": (1,), "Pad": (1,2,3), "MaxUnpool": (2,), "Dropout": (1,2), "CenterCropPad": (1,), "OneHot": (1,), "Compress": (1,),
  "ImageDecoder": (0,), "AffineGrid": (1,), "Resize": (1,2,3), "Upsample": (1,), "Split": (1,), "Slice": (1,2,3,4),
  **{"Reduce"+r: (1,) for r in ("Max", "Min", "Sum", "Mean", "SumSquare", "Prod", "L1", "L2", "LogSum", "LogSumExp")},
  **{optim: (1,) for optim in ("Adam", "Adagrad", "Momentum")}
}

cache_misses = 0
@functools.cache
def _cached_to_python_const(t:Tensor):
  if t.dtype is dtypes.uint8: return t.data().tobytes()
  if 0 in t.shape: return []
  return t.tolist()

# Tensor -> python value cache for parameters
def to_python_const(t:Any, op:str, idx:int) -> list[ConstType]|ConstType|bytes:
  if idx not in required_input_python_consts.get(op, ()) or not isinstance(t, Tensor): return t
  global cache_misses
  ret = _cached_to_python_const(t)
  if (info := _cached_to_python_const.cache_info()).misses > cache_misses and DEBUG >= 3:
    print(f"Cache miss for {t}")
    cache_misses = info.misses
  return ret

# ***** runner ******
class OnnxRunner:
  """
  `OnnxRunner` executes an ONNX model using Tinygrad as backend.

  Args:
    model: The ONNX model, provided as a file path (a string or path-like object).
  """
  def __init__(self, model:str | os.PathLike):
    model = OnnxParser(pathlib.Path(model)).parse()
    self.is_training, self.graph_values, self.graph_inputs, self.graph_outputs, self.graph_nodes, self.opset_version = model_parse(model)
    self.old_training = Tensor.training
    Tensor.training = True if self.is_training else False
    self.variable_dims: dict[str, int] = {}
    self.onnx_ops = onnx_ops

  def _parse_input(self, name: str, value: Any, spec: OnnxValue):
    if spec.is_optional and value is None: return None
    # TODO: need true float16 for dtype checking
    if spec.is_sequence:
      if not isinstance(value, Sequence): raise RuntimeError(f"{name} received {value}, expected a sequence type")
      sequence = [Tensor(v, dtype=spec.dtype, requires_grad=self.is_training) if not isinstance(v, Tensor) else v for v in value]
      if not all_same(tuple(t.shape for t in sequence)): raise RuntimeError(f"Shapes for {name} sequence must be homogeneous")
      return sequence
    tensor = Tensor(value, dtype=spec.dtype, requires_grad=self.is_training) if not isinstance(value, Tensor) else value
    for dim, (onnx_dim, user_dim_input) in enumerate(zip(spec.shape, tensor.shape, strict=True)):
      if isinstance(onnx_dim, str):
        onnx_dim = self.variable_dims[onnx_dim] if onnx_dim in self.variable_dims else self.variable_dims.setdefault(onnx_dim, int(user_dim_input))
      if user_dim_input != onnx_dim: raise RuntimeError(f"{name} has mismatch on {dim=}. Expected {onnx_dim}, received {user_dim_input}.")
    return tensor

  def _dispatch_op(self, op, inps, opts):
    if op in self.onnx_ops:
      fxn = self.onnx_ops[op]
      if isinstance(fxn, dict):
        for k in sorted(fxn.keys()):
          if k <= self.opset_version:
            real_fxn = fxn[k]
      else: real_fxn = fxn
      return real_fxn(*inps, **opts)
    raise NotImplementedError(f"{op=} not supported")

  def get_empty_input_data(self, device:str|None=None) -> dict[str, Tensor]:
    return {name:Tensor.empty(*spec.shape, device=device, dtype=spec.dtype) for name, spec in self.graph_inputs.items()}

  def __call__(self, inputs:dict[str, Any], debug=debug):
    for name, input_spec in self.graph_inputs.items():
      if name not in inputs: raise RuntimeError(f"Please provide input data for {name}")
      self.graph_values[name] = self._parse_input(name, inputs[name], input_spec)

    for node in self.graph_nodes:
      inps = [to_python_const(self.graph_values[name], node.op, i) for i,name in enumerate(node.inputs)]
      opts = node.opts

      # provide additional opts
      if node.op == "Split" and 'num_outputs' not in opts: opts['num_outputs'] = len(node.outputs)
      if node.op == "Gradient": opts['intermediate_tensors'] = self.graph_values

      if debug >= 1: print(f"{node.num}: op '{node.op}' opt {opts}")
      if debug >= 2 and node.inputs: print("\tinputs:\n" + "\n".join(f"\t\t{x} - {i!r}" for x,i in zip(node.inputs, inps)))
      ret = self._dispatch_op(node.op, inps, opts)
      ret = ret if isinstance(ret, tuple) else (ret,)
      if debug >= 2: print("\toutputs:\n" + "\n".join(f"\t\t{x} - {o!r}" for x,o in zip(node.outputs, ret)))

      self.graph_values.update(dict(zip(node.outputs, ret[:len(node.outputs)], strict=True)))

      if node.num == limit:
        Tensor.training = self.old_training
        return {name:self.graph_values[name] for name in node.outputs}
    Tensor.training = self.old_training
    return {name:self.graph_values[name] for name in self.graph_outputs}

####################
##### ONNX OPS #####
####################
def get_onnx_ops():
  # ***** helper functions *****
  def _axes(axes, noop_with_empty_axes): return axes or ([] if noop_with_empty_axes else None)

  # (padding_top, padding_left, ..., padding_bottom, padding_right, ...) -> (padding_left, padding_right, padding_top, padding_bottom, ...)
  def _onnx_pads_to_tiny_pads(pads): return tuple(flatten(reversed(list(zip(pads, pads[len(pads)//2:])))))

  AUTO_PAD_OPTIONS = Literal["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]
  # (padding_height, padding_width) -> (padding_top, padding_left, padding_bottom, padding_right)
  def _auto_pad(pads, auto_pad: AUTO_PAD_OPTIONS):
    if auto_pad == "SAME_UPPER": return [pads[i]//2 for i in range(len(pads))] + [pads[i]-pads[i]//2 for i in range(len(pads))]
    return [pads[i]-pads[i]//2 for i in range(len(pads))] + [pads[i]//2 for i in range(len(pads))]

  def _resolve_pool_pads(x:Tensor, p_, k_, d_, s_, auto_pad:AUTO_PAD_OPTIONS):
    if auto_pad == "VALID": return [0]*(len(k_)*2)
    i_, (s_,d_,p_) = x.shape[-len(k_):], (make_tuple(x, len(k_)*2) for x in (s_, d_, p_))
    if auto_pad == "NOTSET": return _onnx_pads_to_tiny_pads(p_ if len(p_)==len(k_)*2 else p_*2)
    o_ = [((i - (1 if auto_pad in ("SAME_UPPER", "SAME_LOWER") else k)) // s + 1) for i,k,s in zip(i_, k_, s_)]
    return _onnx_pads_to_tiny_pads(_auto_pad([(o-1)*s+k-i for o,i,k,s in zip(o_, i_, k_, s_)], auto_pad))

  def _clamp_cast(x:Tensor, dtype:DType): return x.clamp(dtypes.min(dtype), dtypes.max(dtype)).cast(dtype)

  def _prepare_quantize(x:Tensor, scale:Tensor, zero_point:Tensor|int, axis=1, block_size=0):
    if axis < 0: axis += x.ndim
    # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_quantize_linear.py#L31
    def reshape(val:Tensor):
      if val.numel() == 1: return val
      if block_size == 0: return val.reshape([val.shape[0] if dim == axis else 1 for dim in range(x.ndim)])
      return val.repeat_interleave(block_size, axis)
    return (reshape(scale), reshape(zero_point) if isinstance(zero_point, Tensor) else zero_point)

  def _op_integer(op, inputs:list[Tensor], zero_points:list[Tensor], **opts):
    adjusted_inputs = [inp.int() - zp for inp, zp in zip(inputs, zero_points)]
    return op(*adjusted_inputs, **opts)

  def _qlinearop_quantized(op, inputs:list[Tensor], zero_points:list[Tensor], scales:list[Tensor], out_scale:Tensor, out_zero_point:Tensor, **opts):
    # op execution is done in quantized int
    out = _op_integer(op, inputs, zero_points, **opts)
    assert dtypes.is_int(out.dtype), "quantized op should've done math in int"
    out_quantized = (out * prod(scales) / out_scale).round() + out_zero_point
    return _clamp_cast(out_quantized, out_zero_point.dtype)

  def _qlinearop_float(op, inputs:list[Tensor], zero_points:list[Tensor], scales:list[Tensor], out_scale:Tensor, out_zero_point:Tensor, **opts):
    # op execution is done in float32
    dequantized_inputs = [(inp.int() - zp) * scale for inp, zp, scale in zip(inputs, zero_points, scales)]
    out = op(*dequantized_inputs, **opts)
    assert dtypes.is_float(out.dtype), "op should've done math in float"
    out_quantized = (out / out_scale).round() + out_zero_point
    return _clamp_cast(out_quantized, out_zero_point.dtype)

  def _onnx_training(input_group_size):
    def __decorator(func):
      def ___wrapper(R:Tensor, T:int, *inputs:Tensor, **kwargs):
        R = R.detach()
        groups = len(inputs) // input_group_size
        ret = [func(R, T, *inps, **kwargs) for inps in (inputs[i::groups] for i in range(groups))]
        return tuple(flatten(zip(*ret)))
      return ___wrapper
    return __decorator

  # ***** Property/Graph Ops *****
  def Identity(x:Tensor): return x
  def Constant(sparse_value:Tensor|None=None, value:Tensor|None=None, value_float:float|None=None, value_floats:list[float]|None=None,
              value_int:int|None=None, value_ints:list[int]|None=None, value_string:str|None=None, value_strings:list[str]|None=None):
    if value is not None: return value
    if value_float is not None: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
    if value_floats is not None: return Tensor(list(value_floats), dtype=dtypes.float32, requires_grad=False)
    if value_int is not None: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
    if value_ints is not None: return Tensor(list(value_ints), dtype=dtypes.int64, requires_grad=False)
    if value_string is not None or value_strings is not None and sparse_value is not None:
      raise NotImplementedError('Constant OP not implemented for value_string, value_strings and sparse_value')

  def Range(start:float|int, limit:float|int, delta:float|int): return Tensor.arange(start=start, stop=limit, step=delta)

  def ImageDecoder(encoded_stream:bytes, pixel_format="RGB"):
    try: import PIL.Image
    except ImportError as e: raise ImportError("Pillow must be installed for the ImageDecoder operator") from e
    img = PIL.Image.open(io.BytesIO(encoded_stream))
    if pixel_format == "BGR": return Tensor(img.tobytes(), dtype=dtypes.uint8).reshape(*img.size, 3).flip(-1)
    if pixel_format == "RGB": return Tensor(img.tobytes(), dtype=dtypes.uint8).reshape(*img.size, 3)
    if pixel_format == "Grayscale": return Tensor(img.convert("L").tobytes(), dtype=dtypes.uint8).reshape(*img.size, 1)
    raise ValueError(f"pixel_format={pixel_format!r} is not supported.")

  def EyeLike(x:Tensor, dtype:int|None=None, k:int=0):
    ret = Tensor.eye(cast(int, min(x.shape)), dtype=dtype_parse(dtype) if dtype is not None else x.dtype)
    return ret if x.size(0) == x.size(1) else ret.pad(tuple(None if d == ret.size(0) else (k, d-ret.shape[0]-k) for d in x.shape))

  def OptionalHasElement(x:Tensor|None=None): return Tensor(x is not None and x.numel() > 0)
  def OptionalGetElement(x:Tensor|None=None): return x if x is not None else Tensor([])
  def ConstantOfShape(shape:list[int], value:Tensor|None=None):
    if value is None: value = Tensor(0, dtype=dtypes.float32)
    if shape == [0]: return Tensor([], dtype=value.dtype)
    return value.expand(shape)

  def Size(data:Tensor): return data.numel()
  def Shape(data:Tensor, end:int|None=None, start:int=0): return Tensor(data.shape[start:end], dtype=dtypes.int64)

  # ***** Unary Ops (math) *****
  def Not(x:Tensor): return x.logical_not()
  def Clip(x: Tensor, min:Tensor|None=None, max:Tensor|None=None): return x if min is None and max is None else x.clip(min, max)
  def IsInf(x:Tensor, detect_negative:int=1, detect_positive:int=1): return x.isinf(bool(detect_positive), bool(detect_negative))

  # ***** Unary Ops (activation) *****
  def Softmax_1(x:Tensor, axis:int=1): return x.softmax(axis)
  def Softmax_13(x:Tensor, axis:int=-1): return x.softmax(axis)
  Softmax = {1:Softmax_1, 13:Softmax_13}
  def HardSigmoid(x:Tensor, alpha:float=0.2, beta:float=0.5): return (alpha*x + beta).clip(0, 1)
  def Gelu(x:Tensor, approximate:str|None=None): return x.gelu() if approximate == "tanh" else 0.5 * x * (1 + (x/math.sqrt(2)).erf())
  def BiasGelu(x: Tensor, bias: Tensor, approximate: str | None = None) -> Tensor: return Gelu(x + bias, approximate)
  def FastGelu(x:Tensor, bias:Tensor|None=None): return (x + bias).gelu() if bias is not None else x.gelu() # this is tanh approximated
  def PRelu(X:Tensor, slope:Tensor): return (X > 0).where(X, X * slope)
  def LeakyRelu(X:Tensor, alpha:float=0.01): return X.leaky_relu(alpha)
  def ThresholdedRelu(X:Tensor, alpha:float=1.0): return (X > alpha).where(X, 0)
  def LogSoftmax(x: Tensor, axis:int=-1): return x.log_softmax(axis)
  def Binarizer(x:Tensor, threshold:float=0.0): return (x > threshold).float()

  # ***** Unary Ops (broadcasted) *****
  def Add(x:Tensor,y:Tensor, broadcast=None, axis=None): return x + y if x.dtype == dtypes.float or isinstance(x.dtype, ImageDType) else (x + y).cast(x.dtype)
  def Sub(x:Tensor|int,y:Tensor): return x - y # some test has input as int
  def Div(x:Tensor,y:Tensor): return x.div(y, rounding_mode='trunc' if dtypes.is_int(x.dtype) else None)
  def Less(x:Tensor,y:Tensor): return x < y
  def LessOrEqual(x:Tensor,y:Tensor): return x <= y
  def Greater(x:Tensor,y:Tensor): return x > y
  def GreaterOrEqual(x:Tensor,y:Tensor): return x >= y
  def Equal(x:Tensor,y:Tensor): return x == y
  def And(x:Tensor,y:Tensor): return (x==y).where(x, False)
  def Or(x:Tensor,y:Tensor): return (x==y).where(x, True)
  def Xor(x:Tensor,y:Tensor): return x.bool().bitwise_xor(y.bool())
  def BitwiseAnd(x:Tensor,y:Tensor): return x & y
  def BitwiseOr(x:Tensor,y:Tensor): return x | y
  def BitwiseXor(x:Tensor,y:Tensor): return x ^ y
  def BitwiseNot(x:Tensor): return ~x
  def Mod(x:Tensor,y:Tensor,fmod=0):
    if fmod: return x - x.div(y, rounding_mode="trunc") * y
    return x % y

  # ***** Casting Ops *****
  # TODO: saturate
  def Cast(x:Tensor, to:int, saturate:int=1): return x.cast(dtype_parse(to))
  def CastLike(x:Tensor, target_type:Tensor, saturate:int=1): return x.cast(target_type.dtype)

  # ***** Reduce Ops *****
  def Max(*data_0:Tensor): return functools.reduce(Tensor.maximum, data_0)
  def Min(*data_0:Tensor): return functools.reduce(Tensor.minimum, data_0)
  def Sum(*data_0:Tensor): return functools.reduce(Tensor.add, data_0)
  def Mean(*data_0:Tensor): return Sum(*data_0) / len(data_0)
  def ReduceMax(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
  def ReduceMin(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
  def ReduceSum(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
  def ReduceMean(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
  def ReduceSumSquare(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return ReduceSum(data.square(), axes, keepdims, noop_with_empty_axes)
  def ReduceProd(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return data.prod(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
  def ReduceL1(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return ReduceSum(data.abs(), axes, keepdims, noop_with_empty_axes)
  def ReduceL2(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return ReduceSumSquare(data, axes, keepdims, noop_with_empty_axes).sqrt()
  def ReduceLogSum(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return ReduceSum(data, axes, keepdims, noop_with_empty_axes).log()
  def ReduceLogSumExp(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    return ReduceSum(data.exp(), axes, keepdims, noop_with_empty_axes).log()
  def ArgMax(x:Tensor, axis:int=0, keepdims:int=1, select_last_index:int=0):
    if select_last_index: return ((x.shape[axis]-1) - x.flip(axis).argmax(axis, keepdim=keepdims)).cast(dtypes.int64)
    return x.argmax(axis, keepdim=keepdims).cast(dtypes.int64)
  def ArgMin(x, axis:int=0, keepdims:int=1, select_last_index:int=0):
    return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

  # ***** Movement Ops *****
  def Reshape(data:Tensor, shape:list[int], allowzero:int=0):
    return data.reshape([x if x != 0 else (0 if allowzero else data.shape[i]) for i,x in enumerate(shape)])
  def Flatten(x:Tensor, axis:int=1): return x.reshape(prod(x.shape[0:axis]), -1)
  def Expand(x:Tensor, shape:list[int]): return x.expand(_broadcast_shape(x.shape, tuple(shape)))
  def Shrink(x:Tensor, bias:float=0.0, lambd:float=0.5): return (x < -lambd)*(x+bias) + (x > lambd)*(x-bias)
  def Transpose(x:Tensor, perm:list[int]|None=None): return x.permute(order=perm or list(range(x.ndim)[::-1]))

  def Squeeze(data:Tensor, axes:list[int]|None=None):
    return data.squeeze() if axes is None else functools.reduce(lambda d, dim: d.squeeze(dim), sorted(axes, reverse=True), data)
  def Unsqueeze(data:Tensor, axes:list[int]): return functools.reduce(lambda d, dim: d.unsqueeze(dim), sorted(axes), data)

  def Tile(x:Tensor, repeats:list[int]): return x.repeat(repeats)
  def Concat(*xs:Tensor, axis:int): return Tensor.cat(*xs, dim=axis)
  def Slice(data:Tensor, starts:list[int], ends:list[int], axes:list[int]|None=None, steps:list[int]|None=None):
    axes = axes or list(range(data.ndim))
    steps = steps or [1]*data.ndim
    slices = [slice(0,x,1) for x in data.shape]
    for i, axis in enumerate(axes): slices[axis] = slice(starts[i], ends[i], steps[i])
    return data[tuple(slices)]

  def Split(data:Tensor, split:list[int]|None=None, num_outputs:int=0, axis:int=0):
    sz = data.shape[axis]
    if split is None: split = [sz // num_outputs + (1 if i < sz % num_outputs else 0) for i in range(num_outputs)]
    return data.split(split, axis)

  def Pad(x:Tensor, pads:list[int], constant_value:ConstType|None=None, axes:list[int]|None=None,
          mode:Literal["constant", "reflect", "edge", "wrap"]="constant", value=0):
    value = constant_value or value
    axes = axes or list(range(x.ndim))
    real_pads = [0] * (x.ndim*2)
    for i,axis in enumerate(axes): real_pads[axis%x.ndim], real_pads[axis%x.ndim+x.ndim] = pads[i], pads[i+len(axes)]
    return x.pad(padding=_onnx_pads_to_tiny_pads(real_pads), mode={"edge":"replicate", "wrap":"circular"}.get(mode, mode), value=value)

  def CenterCropPad(t:Tensor, shape:list[int], axes:list[int]|None=None):
    shrink_arg:list[None|tuple[int,int]] = [None] * t.ndim
    pad_arg:list[None|tuple[int,int]] = [None] * t.ndim
    for s, x in zip(shape, axes or range(t.ndim)):
      tx = t.shape[x]
      if s < tx: shrink_arg[x] = (tx//2 - (s+1)//2, tx//2 + s//2)
      elif s > tx: pad_arg[x] = ((s-tx)//2, (s-tx+1)//2)
    return t.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

  # ***** Processing Ops *****
  def AveragePool(X: Tensor, kernel_shape:list[int], auto_pad:AUTO_PAD_OPTIONS="NOTSET", ceil_mode:int=0, count_include_pad:int=0,
                  dilations:list[int]|int=1, pads:list[int]|int=0, strides:list[int]|int=1):
    return X.avg_pool2d(kernel_shape, strides, dilations, _resolve_pool_pads(X, pads, kernel_shape, dilations, strides, auto_pad),
                        ceil_mode=ceil_mode, count_include_pad=count_include_pad)

  def MaxPool(X: Tensor, kernel_shape:list[int], auto_pad:AUTO_PAD_OPTIONS="NOTSET", ceil_mode:int=0, dilations:list[int]|int=1, pads:list[int]|int=0,
              storage_order:int=0, strides:list[int]|int=1):
    pads = _resolve_pool_pads(X, pads, kernel_shape, dilations, strides, auto_pad)
    ret, idx = X.max_pool2d(kernel_shape, strides, dilations, pads, ceil_mode=ceil_mode, return_indices=True)
    return ret, idx.transpose(-2, -1).cast(dtypes.int64) if storage_order else idx.cast(dtypes.int64)

  def Conv(X: Tensor, W: Tensor, B:Tensor|None=None, auto_pad:AUTO_PAD_OPTIONS="NOTSET", dilations:list[int]|int=1, group:int=1,
          kernel_shape:list[int]|None=None, pads:list[int]|int=0, strides:list[int]|int=1):
    return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations,
                    padding=_resolve_pool_pads(X, pads, kernel_shape or W.shape[2:], dilations, strides, auto_pad))

  def ConvTranspose(X: Tensor, W: Tensor, B:Tensor|None=None, auto_pad:AUTO_PAD_OPTIONS="NOTSET", dilations:list[int]|int=1, group:int=1,
                    kernel_shape:list[int]|None=None, pads:list[int]|None=None, output_shape:list[int]|None=None, output_padding:list[int]|int=0,
                    strides:list[int]|int=1):
    input_shape, kernel_shape = X.shape[2:], (kernel_shape or W.shape[2:])
    strides, dilations, output_padding = (make_tuple(x, len(input_shape)) for x in (strides, dilations, output_padding))
    if output_shape is not None: # we pad according to output_shape
      pads = _auto_pad([s*(i-1) + op + ((k-1)*d+1) - os for s,i,op,k,d,os in
                        zip(strides, input_shape, output_padding, kernel_shape, dilations, output_shape)], auto_pad)
    if pads is None: # we generate pads
      output_shape = output_shape or [X.shape[i+2] * strides[i] for i in range(len(strides))]
      pads = [strides[i]*(input_shape[i]-1) + output_padding[i] + ((kernel_shape[i]-1)*dilations[i]+1)-output_shape[i] for i in range(len(input_shape))]
      pads = _auto_pad(pads, auto_pad) if auto_pad != "NOTSET" else [0] * len(input_shape) * 2
    pads = _onnx_pads_to_tiny_pads(pads)
    return X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=pads, output_padding=output_padding)

  def MaxUnpool(xT: Tensor, xI: Tensor, outshape: list[int]|None=None, kernel_shape:list[int]=None, pads:list[int]|int=0, strides:list[int]|int=1):
    return Tensor.max_unpool2d(xT, xI, kernel_shape, strides, 1, pads, outshape if outshape is None else tuple(outshape))

  def GlobalAveragePool(X:Tensor): return X.mean(axis=tuple(range(2, X.ndim)), keepdim=True)
  def GlobalMaxPool(X:Tensor): return X.max(axis=tuple(range(2, X.ndim)), keepdim=True)

  def Gemm(A:Tensor, B:Tensor, C:Tensor|None=None, alpha:float=1.0, beta:float=1.0, transA:int=0, transB:int=0, broadcast=0):
    ret = alpha * (A.transpose(transA) @ B.transpose(transB))
    if C is not None: ret = ret + beta * (C if broadcast == 0 else C.reshape([-1 if i < len(C.shape) else 1 for i in range(ret.ndim)][::-1]))
    return ret

  def Einsum(*Inputs:list[Tensor], equation:str): return Tensor.einsum(equation, *Inputs)

  def CumSum(X:Tensor, axis:int|list, exclusive:int=0, reverse:int=0):
    axis = X._resolve_dim(axis[0] if isinstance(axis, list) else axis)
    if reverse: X = X.flip(axis)
    if exclusive: X = X.pad(tuple((1,0) if i == axis else None for i in range(X.ndim)))\
                        .shrink(tuple((0,X.shape[axis]) if i == axis else None for i in range(X.ndim)))
    return X.cumsum(axis).flip(axis) if reverse else X.cumsum(axis)

  def Trilu(x:Tensor, k:int=0, upper:int=1): return x.triu(k) if upper else x.tril(k)

  def Resize(X:Tensor, roi:list[float]|None=None, scales:list[float]|None=None, sizes:list[int]|None=None, antialias:int=0,
            axes:list[int]|None=None, coordinate_transformation_mode:str='half_pixel', cubic_coeff_a:float=-0.75, exclude_outside:int=0,
            extrapolation_value:float=0.0, keep_aspect_ratio_policy:str='stretch', mode:str='nearest', nearest_mode:str='round_prefer_floor'):
    def _apply_nearest_mode(index: Tensor, input_dim, mode: str):
      if mode == "round_prefer_floor": index = (index - 0.5).ceil()
      elif mode == "round_prefer_ceil": index = (index + 0.5).floor()
      elif mode in ["floor", "ceil"]: index = getattr(index, mode)()
      else: raise ValueError(f"invalid {nearest_mode=}")
      return index.cast(dtypes.int32).clip(0, input_dim-1)
    def _apply_transformation(index: Tensor, input_dim, scale_dim, mode):
      # TODO: needs more testing, not confident in this
      # NOTE: their reference implementation differ from the implementation in their reference docs
      # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py
      # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
      output_dim = scale_dim * input_dim
      if mode == "half_pixel": index = (index + 0.5) / scale_dim - 0.5
      elif mode == "align_corners": index = index * (input_dim - 1) / (output_dim - 1) if output_dim != 1 else Tensor([0])
      elif mode == "asymmetric": index = index / scale_dim
      elif mode == "pytorch_half_pixel": index = (index + 0.5) / scale_dim - 0.5 if output_dim != 1 else Tensor([-0.5])
      elif mode == "half_pixel_symmetric": index = input_dim / 2 * (1 - int(output_dim) / output_dim) + (index + 0.5) / scale_dim - 0.5
      else: raise NotImplementedError(f"invalid {coordinate_transformation_mode=}")
      return index.clip(0, input_dim-1)

    scales, sizes = (None if scales is None else scales[2-(X.ndim-len(scales)):]), (None if sizes is None else sizes[2-(X.ndim-len(sizes)):])
    # we pre permute the axes and permute back after resize
    axes, input_shape, = (axes or list(range(X.ndim))), cast(tuple[int, ...], X.shape[2:]),
    perm = [a for a in range(len(X.shape)) if a not in axes] + list(axes)
    X = X.permute(*perm)

    if sizes is not None:
      if keep_aspect_ratio_policy in ["not_larger", "not_smaller"]:
        scale_fxn = min if keep_aspect_ratio_policy == "not_larger" else max
        scales = [scale_fxn([sizes[i] / input_shape[i] for i in range(len(input_shape)) if i+2 in axes])] * 2
        sizes = [int((scales[0] * input_shape[i]) + 0.5) if i+2 in axes else input_shape[i] for i in range(X.ndim-2)]
      else:
        scales = [size / input_shape for size, input_shape in zip(sizes, input_shape)]
    else:
      sizes = [int(sc*sh) for sc, sh in zip(scales, input_shape)]

    # NOTE: this transformation makes it so that we can't just call Tensor.interpolate
    # in Tensor.interpolate, we use indexes without any transformation
    indexes = []
    for shape, size, scale in zip(input_shape, sizes, scales):
      indexes.append(_apply_transformation(Tensor.arange(size), shape, scale, coordinate_transformation_mode))

    if mode == "nearest":
      indexes = [_apply_nearest_mode(index, shape, nearest_mode) for (index, shape) in zip(indexes, input_shape)]
      X = X[(..., *Tensor.meshgrid(*indexes))]
    if mode == "linear":
      expand = list(X.shape)
      for i in range(-len(sizes), 0):
        reshape, index = [1] * X.ndim, indexes[i]
        reshape[i] = expand[i] = sizes[i]
        low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor().int(), index.ceil().int(), index - index.floor())]
        X = X.gather(i, low).lerp(X.gather(i, high), perc)
    if mode == "cubic": raise NotImplementedError("cubic interpolation is not implemented")
    return X.permute(*argsort(perm)) if perm else X
  def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)  # deprecated

  def TopK(X:Tensor, K:int|list[int], axis:int=-1, largest:int=1, sorted:int=1):
    val, idx = X.topk(K if isinstance(K, int) else K[0], axis, largest, sorted)
    return val, idx.cast(dtypes.int64)

  # ***** Neural Network Ops *****
  def BatchNormalization(X:Tensor, scale:Tensor, B:Tensor, input_mean:Tensor, input_var:Tensor, epsilon:float=1e-05, momentum:float=0.9,
                        training_mode:int=0, spatial=1, is_test=0):
    if training_mode:
      x_detached = X.detach()
      current_mean = x_detached.mean(axis=(0,2,3))
      y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
      current_var = (y*y).mean(axis=(0,2,3))
      current_invstd = current_var.add(epsilon).rsqrt()

      running_mean = input_mean * momentum + current_mean * (1 - momentum)
      running_var = input_var * momentum + current_var * (1 - momentum)

      return X.batchnorm(scale, B, current_mean, current_invstd), running_mean, running_var
    return X.batchnorm(scale, B, input_mean, (input_var + epsilon).rsqrt())
  def GroupNormalization(x:Tensor, scale:Tensor, bias:Tensor, num_groups:int, epsilon:float=1e-05):
    x = x.reshape(x.shape[0], num_groups, -1).layernorm(eps=epsilon).reshape(x.shape)
    return x * scale.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2))
  def InstanceNormalization(x:Tensor, scale:Tensor, bias:Tensor, epsilon:float=1e-05):
    return GroupNormalization(x, scale, bias, num_groups=x.shape[1], epsilon=epsilon)
  def LayerNormalization(x:Tensor, scale:Tensor, bias:Tensor, axis:int=-1, epsilon:float=1e-05, stash_type:int=1):
    assert stash_type == 1, "only float32 is supported"
    axes = tuple(i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    mean = x.mean(axis=axes, keepdim=True)
    return x.layernorm(axes, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).square().mean(axis=axes, keepdim=True).add(epsilon).rsqrt()
  def SkipLayerNormalization(x:Tensor, skip:Tensor, gamma:Tensor, beta:Tensor|None=None, bias:Tensor|None=None, epsilon:float=1e-12):
    x = x + skip
    if bias is not None: x = x + bias
    ret = x.layernorm(eps=epsilon) * gamma
    if beta is not None: ret = ret + beta
    return ret, None, None, x
  def EmbedLayerNormalization(input_ids: Tensor, segment_ids:Tensor, word_embedding:Tensor, position_embedding:Tensor,
                              segment_embedding:Tensor, gamma=None, beta=None, mask:Tensor|None=None,
                              position_ids:Tensor|None=None, epsilon=1e-12, mask_index_type=0):
    # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
    assert (segment_ids is None) is (segment_embedding is None)
    assert mask is None and not mask_index_type, "functionality not supported yet"  # TODO
    input_shape = input_ids.shape
    seq_length = input_shape[1]
    compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
    vocab_size, max_position_embeddings = word_embedding.shape[0], position_embedding.shape[0]
    type_vocab_size  = (segment_embedding.shape[0] if compute_seg_emb else None)

    def embedding(x:Tensor, vocab_size, weight:Tensor) -> Tensor:
      return x.unsqueeze(-1).expand(*x.shape, vocab_size)._one_hot_along_dim(vocab_size) @ weight

    # bert embedding layer
    if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
    wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
    pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
    seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

    embedding_sum = wrd_embedding_res + pos_embedding_res
    if seg_embedding_res is not None: embedding_sum = embedding_sum + seg_embedding_res
    out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
    return out, None, embedding_sum
  def MeanVarianceNormalization(x:Tensor, axis:list[int]=[0,2,3]):
    return (x - x.mean(axis, keepdim=True)) / (x.std(axis, keepdim=True, correction=0) + 1e-9)

  def OneHot(indices:Tensor, depth:float|int|list, values:Tensor, axis:int=-1):
    # Scalar or Rank 1 tensor containing exactly one element
    depth = int(depth[0] if isinstance(depth, list) else depth)
    indices = indices.int()
    indices = (indices < 0).where(indices+depth, indices)
    return indices.unsqueeze(axis)._one_hot_along_dim(depth, dim=axis).where(values[1], values[0])

  def DepthToSpace(X:Tensor, blocksize:int, mode:str="DCR"):
    return X.rearrange("b (c h1 w1) h w -> b c (h h1) (w w1)" if mode=="CRD" else "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=blocksize, w1=blocksize)
  def SpaceToDepth(X:Tensor, blocksize:int):
    return X.rearrange("b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=blocksize, w1=blocksize)

  # Reimplemented here because you need legacy RNG for passing ONNX tests.
  def Dropout_7(data:Tensor, ratio:float=0.5, training_mode:bool=False, seed:int|None=None):
    if not training_mode: return data, Tensor.ones(data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
    mask = Tensor(np.random.RandomState(seed).random(cast(tuple[int,...], data.shape)) >= ratio, requires_grad=False, device=data.device)
    return data * mask * (1/(1.0 - ratio)), mask
  # 6 with 'is_test' needed for https://github.com/MTlab/onnx2caffe/raw/refs/heads/master/model/MobileNetV2.onnx
  def Dropout_6(data:Tensor, ratio:float=0.5, is_test=0): return Dropout_7(data, ratio, training_mode=not is_test)
  Dropout = {6:Dropout_6, 7:Dropout_7}

  def LRN(x:Tensor, size:int, alpha:float=1e-4, beta:float=0.75, bias:float=1.0):
    pooled_x = (x**2).rearrange('b c h w -> b 1 c (h w)').pad((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1)
    return x / (pooled_x.reshape(x.shape) * alpha + bias).pow(beta)

  def NegativeLogLikelihoodLoss(x:Tensor, target:Tensor, weight:Tensor|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean"):
    return x.nll_loss(target, weight, ignore_index, reduction)
  def SoftmaxCrossEntropyLoss(scores:Tensor, labels:Tensor, weights:Tensor|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean"):
    log_probs = scores.log_softmax(1)
    return log_probs.nll_loss(labels, weights, ignore_index, reduction), log_probs

  def AffineGrid(theta:Tensor, size:list[int], align_corners:int=0):
    N, _, *spatial_dims = size
    def generate_grid(steps):
      return Tensor.linspace(-1, 1, steps, device=theta.device) if align_corners else Tensor.linspace(-1+1/steps, 1-1/steps, steps, device=theta.device)
    grids = Tensor.meshgrid(*(generate_grid(d) for d in spatial_dims))
    base_grid = Tensor.stack(*reversed(grids), Tensor.ones_like(grids[0], device=theta.device), dim=-1)
    base_grid = base_grid.reshape(1, prod(spatial_dims), len(grids)+1).expand(N, -1, -1)
    return (base_grid @ theta.transpose(1, 2)).reshape(N, *spatial_dims, -1)

  def Attention(x:Tensor, weights:Tensor, bias:Tensor|None=None, mask_index:Tensor|None=None, past:Tensor|None=None, attention_bias:Tensor|None=None,
                past_sequence_length:Tensor|None=None,  do_rotary:int=0, mask_filter_value:float=-10000.0, num_heads:int|None=None,
                past_present_share_buffer:int|None=None, qkv_hidden_sizes:list[int]|None=None, rotary_embedding_dim:int|None=None,
                scale:float|None=None, unidirectional:int=0):
    assert not do_rotary and not attention_bias, "TODO"
    if qkv_hidden_sizes is None: qkv_hidden_sizes = [weights.shape[1] // 3] * 3
    qkv = x.linear(weights, bias)
    q, k, v = qkv.split(qkv_hidden_sizes, dim=2)

    batch_size, seq_len, _ = x.shape
    q_head_size, k_head_size, v_head_size = (sz // num_heads for sz in qkv_hidden_sizes)
    q, k, v = (x.reshape(batch_size, seq_len, num_heads, hsz).transpose(1, 2) for x, hsz in zip((q, k, v), (q_head_size, k_head_size, v_head_size)))

    present = None
    if past is not None:
      k, v = past[0].cat(k, dim=2), past[1].cat(v, dim=2)
      present = k.stack(v)

    if scale is None: scale = 1.0 / math.sqrt(q_head_size)
    attn_scores = q @ k.transpose(-1, -2) * scale

    if mask_index is not None:
      assert 4 >= mask_index.ndim >= 1, f"{mask_index.ndim=}"
      if mask_index.ndim != 1: mask = mask_index.bool()
      else:
        if mask_index.shape[0] == batch_size:
          mask = Tensor.arange(attn_scores.shape[-1], requires_grad=False, device=mask_index.device).unsqueeze(0) < mask_index.unsqueeze(1)
        elif mask_index.shape[0] == 2*batch_size:
          end_positions = mask_index[:batch_size]
          start_positions = mask_index[batch_size:]
          arange = Tensor.arange(seq_len).unsqueeze(0)
          mask = (arange < end_positions.unsqueeze(1)) & (arange >= start_positions.unsqueeze(1))
        else: raise NotImplementedError("mask_index with shape (3 * batch_size + 2) is not implemented")
      while mask.ndim < 4: mask = mask.unsqueeze(1)
      attn_scores = mask.where(attn_scores, mask_filter_value)

    if unidirectional:
      causal_mask = Tensor.ones((seq_len, seq_len), dtype=dtypes.bool).tril()
      attn_scores = causal_mask.where(attn_scores, mask_filter_value)

    output = attn_scores.softmax(-1) @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    return output, present

  # ***** Indexing Ops *****
  def ArrayFeatureExtractor(x:Tensor, indices:Tensor): return x[..., indices]

  def Gather(x:Tensor, indices:Tensor, axis:int=0):
    if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
      x_sh = list(x.shape)
      ret_shape = x_sh[:axis] + list(indices.shape) + x_sh[axis+1:]
      if indices.ndim > 1: indices = indices.flatten()
      indices = [_cached_to_python_const(indices)] if indices.shape == () else _cached_to_python_const(indices)
      indices = [x_sh[axis]+x if x<0 else x for x in indices]
      args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(x_sh)] for i in indices] # type: ignore
      return x.shrink(arg=tuple(args[0])).cat(*[x.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
    # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
    return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]
  def Scatter(*args, **kwargs): return ScatterElements(*args, **kwargs) # deprecated

  def GatherND(x:Tensor, indices:Tensor, batch_dims:int=0):
    if batch_dims == 0: return x[tuple(i.squeeze(-1) for i in indices.split(1, -1))]
    x_shape, i_shape = x.shape, indices.shape
    b = math.prod(x.shape[dim] for dim in range(batch_dims))
    # NOTE: each batched dim of both input and indices are equal
    x = x.reshape(b, *x.shape[batch_dims:])
    indices = indices.reshape(b, *indices.shape[batch_dims:])
    b_idx = Tensor.arange(b, device=x.device).reshape(b, *(1,)*(indices.ndim - 2)).expand(*indices.shape[:-1])
    ret = x[(b_idx,) + tuple(i.squeeze(-1) for i in indices.split(1, -1))]
    return ret.reshape(*x_shape[:batch_dims], *i_shape[batch_dims:-1], *ret.shape[indices.ndim-1:])
  def ScatterND(x:Tensor, indices:Tensor, updates:Tensor, reduction:Literal["none", "add", "mul"]='none'):
    assert updates.shape == indices.shape[:-1] + x.shape[cast(int, indices.shape[-1]):]
    x = x.contiguous()
    for index, u in zip(indices.split(1, 0), updates.split(1, 0)):
      i = tuple(idx.squeeze(-1) for idx in index.squeeze(0).split(1, -1))
      u = u.squeeze(0)
      if reduction == "none": x[i] = u
      elif reduction == "add": x[i] += u
      elif reduction == "mul": x[i] *= u
      else: raise NotImplementedError("reduction doesn't support max or min")
    return x

  def ScatterElements(x: Tensor, indices: Tensor, updates: Tensor, axis=0, reduction:Literal["none", "add", "mul", "min", "max"]="none"):
    indices = (indices < 0).where(x.shape[axis], 0) + indices
    if reduction == "none": return x.scatter(axis, indices, updates)
    return x.scatter_reduce(axis, indices, updates, {"add": "sum", "mul": "prod", "min": "amin", "max": "amax"}.get(reduction))
  def GatherElements(x:Tensor, indices:Tensor, axis:int):
    indices = (indices < 0).where(x.shape[axis], 0) + indices
    return x.gather(axis, indices)

  def Compress(inp:Tensor, condition:list[bool], axis:int|None=None):
    if axis is None:
      inp = inp.flatten()
      axis = 0
    if axis < 0: axis += inp.ndim
    con = Tensor([i for i,cond in enumerate(condition) if cond]) # compress in python
    return inp[tuple(con if i == axis else slice(None) for i in range(inp.ndim))]

  # ***** Quantization Ops *****
  def QuantizeLinear(x:Tensor, y_scale:Tensor, y_zero_point:Tensor|int=0, axis:int=1, block_size:int=0, output_dtype:int=0, saturate=1):
    out_dtype = y_zero_point.dtype if isinstance(y_zero_point, Tensor) else dtype_parse(output_dtype) if output_dtype else dtypes.uint8
    y_scale, y_zero_point = _prepare_quantize(x, y_scale, y_zero_point, axis, block_size)
    if out_dtype == dtypes.uchar:
      # this appears to work in practice, at least for uchar out_dtype. it folds with the quantize stuff
      ret = _clamp_cast((x / y_scale + 0.4999999 + y_zero_point).int(), out_dtype)
    else:
      ret = _clamp_cast(((x / y_scale).round() + y_zero_point), out_dtype)
    return ret.contiguous()

  def DynamicQuantizeLinear(x: Tensor):
    # only support uint8
    qmin, qmax = dtypes.min(dtypes.uint8), dtypes.max(dtypes.uint8)
    scale = (x.max().maximum(0) + ((-x).max()).maximum(0)) / (qmax - qmin)
    zero_point = _clamp_cast((qmin - x.min() / scale).round(), dtypes.uint8)
    y = _clamp_cast((x / scale).round() + zero_point, dtypes.uint8)
    return y, scale, zero_point

  def DequantizeLinear(x:Tensor, x_scale:Tensor, x_zero_point:Tensor|int=0, axis:int=1, block_size:int=0):
    x_scale, x_zero_point = _prepare_quantize(x, x_scale, x_zero_point, axis, block_size)
    return ((x.int() - x_zero_point) * x_scale).cast(x_scale.dtype)

  def QLinearConv(x:Tensor, x_scale:Tensor, x_zero_point:Tensor|int, w:Tensor, w_scale:Tensor, w_zero_point:Tensor|int, y_scale:Tensor,
                  y_zero_point: Tensor|int, B:Tensor|None=None, **opts):
    return _qlinearop_quantized(Conv, [x,w], [x_zero_point,w_zero_point], [x_scale,w_scale], y_scale, y_zero_point, **{"B":B, **opts})

  def QLinearMatMul(a:Tensor, a_scale:Tensor, a_zero_point:Tensor|int, b:Tensor, b_scale:Tensor, b_zero_point:Tensor|int, y_scale:Tensor,
                    y_zero_point:Tensor|int) -> Tensor:
    return _qlinearop_quantized(Tensor.matmul, [a,b], [a_zero_point,b_zero_point], [a_scale,b_scale], y_scale, y_zero_point)

  def QLinearAdd(a:Tensor, a_scale:Tensor, a_zero_point:Tensor, b:Tensor, b_scale:Tensor, b_zero_point:Tensor, c_scale:Tensor, c_zero_point:Tensor):
    return _qlinearop_float(Tensor.add, [a,b], [a_zero_point,b_zero_point], [a_scale,b_scale], c_scale, c_zero_point)

  def QLinearMul(a:Tensor, a_scale:Tensor, a_zero_point:Tensor, b:Tensor, b_scale:Tensor, b_zero_point:Tensor, c_scale:Tensor, c_zero_point:Tensor):
    return _qlinearop_quantized(Tensor.mul, [a,b], [a_zero_point,b_zero_point], [a_scale,b_scale], c_scale, c_zero_point)

  def QLinearGlobalAveragePool(X:Tensor, x_scale:Tensor, x_zero_point:Tensor, y_scale:Tensor, y_zero_point:Tensor, channels_last:int):
    assert channels_last == 0, "TODO NHWC"
    return _qlinearop_float(GlobalAveragePool, [X], [x_zero_point], [x_scale], y_scale, y_zero_point)

  def ConvInteger(x: Tensor, w: Tensor, x_zero_point: Tensor | int = 0, w_zero_point: Tensor | int = 0, B: Tensor | None = None, **opts) -> Tensor:
    return _op_integer(Conv, [x,w], [x_zero_point,w_zero_point], **{"B":B, **opts})

  def MatMulInteger(A: Tensor, B: Tensor, a_zero_point: Tensor | int = 0, b_zero_point: Tensor | int = 0) -> Tensor:
    return _op_integer(Tensor.matmul, [A,B], [a_zero_point,b_zero_point])

  # ***** Training Ops *****
  # NOTE: onnx training ops actually don't need the state for optim, all the ops work in a functional way, but we still can reuse optim.py code
  @_onnx_training(3)
  def Adagrad(R:Tensor, T:int, *inputs:Tensor, decay_factor:float=0.0, epsilon:float=0.0, norm_coefficient:float=0.0):
    X, G, H = (i.detach() for i in inputs)
    grad = norm_coefficient * X + G
    H.assign(H + grad.square())
    up = grad / (H.sqrt() + epsilon)
    r = R / (1 + T * decay_factor)
    X.assign(X.detach() - r * up)
    return [X, H]

  @_onnx_training(4)
  def Adam(R:Tensor, T:int, *inputs:Tensor, alpha:float=0.9, beta:float=0.999, epsilon:float=0.0, norm_coefficient:float=0.0,
          norm_coefficient_post:float=0.0):
    from tinygrad.nn.optim import Adam as TinyAdam
    X, G, V, H = inputs
    G, V, H = G.detach(), V.detach(), H.detach()
    X.grad = norm_coefficient * X.detach() + G
    opt = TinyAdam([X], b1=alpha, b2=beta, eps=epsilon)
    opt.m, opt.v, opt.lr = [V], [H], R
    # need no-op for m_hat and v_hat if T == 0
    if T == 0: opt.b1_t, opt.b2_t = opt.b1_t.zeros_like(), opt.b2_t.zeros_like()
    else:
      # `T-1` since it's applied again at the start of `_step`
      opt.b1_t = Tensor([alpha**(T-1)], dtype=dtypes.float32, device=X.device, requires_grad=False)
      opt.b2_t = Tensor([beta**(T-1)], dtype=dtypes.float32, device=X.device, requires_grad=False)
    opt.step()
    X = (1 - norm_coefficient_post) * X
    return [X, V, H]

  @_onnx_training(3)
  def Momentum(R:Tensor, T:int, *inputs:Tensor, alpha:float, beta:float, mode:str, norm_coefficient:float):
    X, G, V = (i.detach() for i in inputs)
    grad = norm_coefficient * X + G
    # NOTE: this beta_adjusted term makes it so we can't use SGD for nesterov
    beta_adjusted = beta if T > 0 else 1
    V.assign(alpha * V + grad * beta_adjusted)
    X.assign(X - R * (V if mode == "standard" else (grad + alpha * V)))
    return [X, V]

  def Gradient(*inputs:Tensor, y:str, intermediate_tensors:dict[str, Tensor], **_):
    intermediate_tensors[y].backward()
    return tuple([t.grad for t in inputs])

  return {
    # Tensor ops
    **{op: getattr(Tensor, op.lower()) for op in ("Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan",
    "Asin", "Acos", "Atan", "Relu", "Sigmoid", "MatMul", "Floor", "Ceil", "IsNaN", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh",
    "Tanh", "Softsign", "Asinh", "Acosh", "Atanh",  "Elu", "Celu", "Selu", "Round", "Erf")},
    # Implemented ops
    **{name:obj for name,obj in locals().items() if isinstance(obj, types.FunctionType) and not name.startswith("_") and name[0].isupper()},
    # Version ops
    **{name:obj for name,obj in locals().items() if isinstance(obj, dict)},
  }

onnx_ops = get_onnx_ops()
