# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

import os, struct
from io import BytesIO
from types import SimpleNamespace
from tinygrad.nn.state import TensorIO, accept_filename
from tinygrad.tensor import Tensor
# Protobuf Wire Types
WIRETYPE_VARINT = 0; WIRETYPE_FIXED64 = 1; WIRETYPE_LENGTH_DELIMITED = 2; WIRETYPE_START_GROUP = 3; WIRETYPE_END_GROUP = 4; WIRETYPE_FIXED32 = 5 # noqa: E702

# TensorProto.DataType
class TensorDataType:
  UNDEFINED = 0; FLOAT = 1; UINT8 = 2; INT8 = 3; UINT16 = 4; INT16 = 5; INT32 = 6; INT64 = 7 # noqa: E702
  STRING = 8; BOOL = 9; FLOAT16 = 10; DOUBLE = 11; UINT32 = 12; UINT64 = 13; COMPLEX64 = 14; COMPLEX128 = 15; BFLOAT16 = 16 # noqa: E702

# AttributeProto.AttributeType
class AttributeType:
  UNDEFINED = 0; FLOAT = 1; INT = 2; STRING = 3; TENSOR = 4; GRAPH = 5; SPARSE_TENSOR = 11; TYPE_PROTO = 13; FLOATS = 6; INTS = 7 # noqa: E702
  STRINGS = 8; TENSORS = 9; GRAPHS = 10; SPARSE_TENSORS = 12; TYPE_PROTOS = 14 # noqa: E702

def decode_varint(reader) -> int:
  result = 0
  shift = 0
  while True:
    data = reader.read(1)
    if data == b'': raise EOFError("end")
    byte = data[0]
    result |= (byte & 0x7F) << shift
    if not (byte & 0x80): return result
    shift += 7
    if shift >= 64: raise ValueError("Varint too long")

def unsigned_to_signed_64(uval):
  if uval & (1 << 63): return uval - (2**64)
  return uval

def skip_field_value(reader, wire_type):
  if wire_type == WIRETYPE_VARINT: decode_varint(reader)
  elif wire_type == WIRETYPE_FIXED64: reader.seek(os.SEEK_CUR, 8)
  elif wire_type == WIRETYPE_FIXED32: reader.seek(os.SEEK_CUR, 4)
  elif wire_type == WIRETYPE_LENGTH_DELIMITED: reader.seek(os.SEEK_CUR, decode_varint(reader))
  elif wire_type == WIRETYPE_START_GROUP or wire_type == WIRETYPE_END_GROUP: raise NotImplementedError("Groups are deprecated")
  else: raise ValueError(f"Unknown wire type: {wire_type}")


def gen_result(obj: dict, key_name, val, repeated: bool):
  if repeated: obj.setdefault(key_name, []).append(val)
  else: obj[key_name] = val

def dict_to_namespace(d):
  if isinstance(d, dict): return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
  elif isinstance(d, list): return [dict_to_namespace(i) for i in d]
  else: return d

@accept_filename
def onnx_load(tensor: Tensor):
  reader = TensorIO(tensor)
  parser = OnnxParser()
  onnx_model = parser.parse_model_proto_from_buffer(reader)
  model = dict_to_namespace(onnx_model)
  return model

class OnnxParser:
  def _parse_message(self, reader, message_field_handlers, initial_obj_factory=lambda: {}, debug=False):
    obj = initial_obj_factory()
    while True:
      try:
        tag_val = decode_varint(reader)
        field_number = tag_val >> 3
        wire_type = tag_val & 0x07
        if debug: print(f"DEBUG _parse_message: {tag_val=}, {field_number=}, {wire_type=}")
        if handler := message_field_handlers.get(field_number):
          if debug: print(f"DEBUG _parse_message call handler: {handler._debug_info}")
          handler(obj, reader, wire_type)
        else: skip_field_value(reader, wire_type)
      except EOFError: break
    return obj

  def _handle_int64_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_VARINT: raise ValueError(f"Expected varint for int64 field '{key_name}'")
    val = decode_varint(reader)
    signed_val = unsigned_to_signed_64(val)
    gen_result(obj, key_name, signed_val, repeated)

  def _handle_int32_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    self._handle_int64_field(obj, key_name, reader, wire_type, repeated)

  def _handle_float_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_FIXED32: raise ValueError(f"Expected fixed32 for float field '{key_name}'")
    val, = struct.unpack("<f", reader.read(4))
    gen_result(obj, key_name, val, repeated)

  def gen_handlers(self, tpl):
    res = {}
    for handler_fn, fields in tpl.items():
      for config in fields:
        parser_fn, repeated = None, False
        if len(config) == 2: fid, name = config
        elif len(config) == 3: fid, name, repeated = config
        elif len(config) == 4: fid, name, repeated, parser_fn = config
        def _wrapper_handler(obj, reader, wt, h=handler_fn, n=name, p=parser_fn, r=repeated): return h(obj, n, reader, wt, parser_func=p, repeated=r)
        _wrapper_handler._debug_info = f"{fid}, {name} => {handler_fn}"
        res[fid] = _wrapper_handler
    return res

  # WIRETYPE_LENGTH_DELIMITED
  def _handle_delimited(self, reader):
    str_len = decode_varint(reader)
    return reader.read(str_len)

  def _handle_string_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for string field '{key_name}'")
    value = self._handle_delimited(reader)
    value = value.decode('utf-8')
    gen_result(obj, key_name, value, repeated)

  def _handle_bytes_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for bytes field '{key_name}'")
    value = self._handle_delimited(reader)
    gen_result(obj, key_name, value, repeated)

  def _handle_packed_repeated_floats(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed floats expected length_delimited")
    value = self._handle_delimited(reader)
    if len(value) % 4 != 0: raise ValueError("Packed float data length not multiple of 4")
    values = list(struct.unpack(f"<{len(value) // 4}f", value))
    obj.setdefault(key_name, []).extend(values)

  def _handle_packed_repeated_int64s(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed int64s expected length_delimited")
    total_bytes_len = decode_varint(reader)
    old_pos = reader.tell()
    values = []
    while reader.tell() < total_bytes_len + old_pos:
      val = decode_varint(reader)
      values.append(unsigned_to_signed_64(val))
    obj.setdefault(key_name, []).extend(values)

  def _handle_packed_repeated_int32s(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    return self._handle_packed_repeated_int64s(obj, key_name, reader, wire_type)

  def _handle_sub_message_field(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for sub-message field '{key_name}'")
    value = self._handle_delimited(reader)
    parsed_sub_obj = parser_func(BytesIO(value))
    gen_result(obj, key_name, parsed_sub_obj, repeated)

  # OperatorSetIdProto
  def parse_opset_id_proto(self, reader): return self._parse_message(reader, self.gen_handlers({
    self._handle_string_field: ((1, 'domain'),), self._handle_int64_field: ((2, 'version'),)}))

  # StringStringEntryProto
  def parse_string_string_entry_proto(self, reader):
    return self._parse_message(reader, self.gen_handlers({ self._handle_string_field: ((1, 'key'), (2, 'value'))}))

  # TensorProto: Tensors, A serialized tensor value.
  def parse_tensor_proto(self, reader):
    handlers = self.gen_handlers({ self._handle_int64_field: ((1, 'dims', True),), self._handle_int32_field: ((2, 'data_type'),),
      self._handle_packed_repeated_floats: ((4, 'float_data'),), self._handle_packed_repeated_int32s: ((5, 'int32_data'),),
      self._handle_bytes_field: ((6, 'string_data', True), (9, 'raw_data')),
      self._handle_packed_repeated_int64s: ((7, 'int64_data'),), self._handle_string_field: ((8, 'name'),)})
    obj = self._parse_message(reader, handlers,
      lambda: {'dims': [], 'float_data': [], 'int32_data': [], 'string_data':[], 'int64_data':[], 'double_data':[], 'uint64_data':[]})
    return obj

  # TensorShapeProto.Dimension
  def parse_tensor_shape_proto_dimension(self, reader):
    return self._parse_message(reader, self.gen_handlers({
      self._handle_int64_field: ((1, 'dim_value'),), self._handle_string_field: ((2, 'dim_param'), (3, 'denotation'))}))

  # TensorShapeProto
  def parse_tensor_shape_proto(self, reader):
    return self._parse_message(reader, self.gen_handlers({
      self._handle_sub_message_field: ((1, 'dim', True, self.parse_tensor_shape_proto_dimension),)}), lambda: {'dim': []})

  # TypeProto.Tensor
  def parse_type_proto_tensor(self, reader): return self._parse_message(reader, self.gen_handlers({
    self._handle_int32_field: ((1, 'elem_type'),), self._handle_sub_message_field: ((2, 'shape', False, self.parse_tensor_shape_proto),)}))

  # TypeProto.Optional
  def parse_type_proto_optional(self, reader): return self._parse_message(reader, self.gen_handlers({
    self._handle_sub_message_field: ((1, 'elem_type', False, self.parse_type_proto),)}))

  # TypeProto.Sequence
  def parse_type_proto_sequence(self, reader): return self._parse_message(reader, self.gen_handlers({
    self._handle_sub_message_field: ((1, 'elem_type', False, self.parse_type_proto),)}))

  # TypeProto: Types, The standard ONNX data types.
  def parse_type_proto(self, reader):
    return self._parse_message(reader, self.gen_handlers({
      self._handle_sub_message_field: ((1, 'tensor_type', False, self.parse_type_proto_tensor),
                                       (4, 'sequence_type', False, self.parse_type_proto_sequence),
                                       (9, 'optional_type', False, self.parse_type_proto_optional)),
      self._handle_string_field: ((6, 'denotation'),)}))

  # ValueInfoProto
  def parse_value_info_proto(self, reader):
    handlers = self.gen_handlers({
      self._handle_sub_message_field: ((2, 'type', False, self.parse_type_proto), (4, 'metadata_props', True, self.parse_string_string_entry_proto)),
      self._handle_string_field: ((1, 'name'), (3, 'doc_string'))})
    return self._parse_message(reader, handlers, lambda: {'metadata_props': []})

  def interpret_tensor_raw_data(self, tensor_obj):
    if 'raw_data' not in tensor_obj or 'data_type' not in tensor_obj: return
    raw_bytes = tensor_obj['raw_data']
    data_type = tensor_obj['data_type']
    dims = tensor_obj.get('dims', [])
    num_elements = 1
    for d in dims: num_elements *= d
    if not dims and not raw_bytes: return
    if num_elements == 0 and raw_bytes and not dims: num_elements = 1
    decoded_data = []
    if data_type == TensorDataType.FLOAT:
      if len(raw_bytes) != num_elements * 4: raise ValueError(f"FLOAT raw data size mismatch: expected {num_elements*4}, got {len(raw_bytes)}")
      decoded_data = list(struct.unpack(f"<{num_elements}f", raw_bytes))
    elif data_type == TensorDataType.INT64:
      if len(raw_bytes) != num_elements * 8: raise ValueError(f"INT64 raw data size mismatch: expected {num_elements*8}, got {len(raw_bytes)}")
      decoded_data = list(struct.unpack(f"<{num_elements}q", raw_bytes))
    else:
      tensor_obj['_warning'] = f"Raw data interpretation for data_type {data_type} not fully implemented."
      decoded_data = "SKIPPED_RAW_DATA_INTERPRETATION"
    tensor_obj['decoded_data'] = decoded_data

  # AttributeProto
  def parse_attribute_proto(self, reader):
    handlers = self.gen_handlers({
      self._handle_string_field: ((1, "name"), (13, "doc_string"), (21, "ref_attr_name")), self._handle_int32_field: ((20, "type"),),
      self._handle_int64_field: ((3, "i"), (8, "ints", True)), self._handle_float_field: ((2, "f"), (7, "floats", True)),
      self._handle_bytes_field: ((4, "s"), (9, "strings", True)),
      self._handle_sub_message_field: ((5, "t", False,  self.parse_tensor_proto), (6, "g", False,  self.parse_graph_proto),
                                       (10, "tensors", True,  self.parse_tensor_proto),(11, "graphs", True,  self.parse_graph_proto),)})
    obj = self._parse_message(reader, handlers, lambda: {'floats': [], 'ints': [], 'strings': [], 'tensors': [], 'graphs': []})
    if 't' in obj and obj['t']: self.interpret_tensor_raw_data(obj['t'])
    if 'tensors' in obj:
      for tensor in obj['tensors']:
        self.interpret_tensor_raw_data(tensor)
    return obj

  # NodeProto
  def parse_node_proto(self, reader):
    handlers = self.gen_handlers({
      self._handle_sub_message_field: ((5, "attribute", True,  self.parse_attribute_proto),),
      self._handle_string_field: ((1, "input", True), (2, "output", True), (3, "name"), (4, "op_type"), (6, "doc_string"), (7, "domain"))})
    return self._parse_message(reader, handlers, lambda: {'input': [], 'output': [], 'attribute': [], 'domain': None})

  # GraphProto
  def parse_graph_proto(self, reader):
    handlers = self.gen_handlers({
      self._handle_string_field: ((2, "name"), (10, "doc_string")),
      self._handle_sub_message_field: ((13, "value_info", True, self.parse_value_info_proto),
        (1, "node", True,  self.parse_node_proto), (5, "initializer", True, self.parse_tensor_proto),
        (11, "input", True, self.parse_value_info_proto), (12, "output", True, self.parse_value_info_proto))})
    obj = self._parse_message(reader, handlers, lambda: {'node': [], 'initializer': [], 'input':[], 'output':[], 'value_info':[]})
    for tensor in obj['initializer']: self.interpret_tensor_raw_data(tensor)
    return obj

  # ModelProto
  def _model_proto_handlers(self):
    return self.gen_handlers({ self._handle_int64_field: ((1, "ir_version"), (5, "model_version")),
      self._handle_string_field: ((2, "producer_name"), (3, "producer_version"), (4, "domain"), (6, "doc_string")),
      self._handle_sub_message_field: ((8, "opset_import", True,  self.parse_opset_id_proto), (7, "graph", False, self.parse_graph_proto),
                                       (14, "metadata_props", True, self.parse_string_string_entry_proto))})

  def parse_model_proto_from_buffer(self, reader):
    parsed_model = self._parse_message(reader, self._model_proto_handlers(), lambda: {'opset_import': [], 'metadata_props': [], 'domain': None})
    return parsed_model
