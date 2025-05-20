import struct

# Protobuf Wire Types
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5

# TensorProto.DataType
class TensorDataType:
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

# AttributeProto.AttributeType
class AttributeType:
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

def decode_varint(data, offset):
  result = 0
  shift = 0
  current_offset = offset
  while True:
    if current_offset >= len(data): raise EOFError("Buffer too short for varint")
    byte = data[current_offset]
    current_offset += 1
    result |= (byte & 0x7F) << shift
    if not (byte & 0x80): return result, current_offset
    shift += 7
    if shift >= 64: raise ValueError("Varint too long")

def skip_field_value(data, offset, wire_type):
  new_offset = offset
  if wire_type == WIRETYPE_VARINT: _, new_offset = decode_varint(data, new_offset)
  elif wire_type == WIRETYPE_FIXED64: new_offset += 8
  elif wire_type == WIRETYPE_FIXED32: new_offset += 4
  elif wire_type == WIRETYPE_LENGTH_DELIMITED:
    length, after_len_offset = decode_varint(data, new_offset)
    new_offset = after_len_offset + length
  elif wire_type == WIRETYPE_START_GROUP or wire_type == WIRETYPE_END_GROUP: raise NotImplementedError("Groups are deprecated")
  else: raise ValueError(f"Unknown wire type: {wire_type} at offset {offset-1}")
  if new_offset > len(data): raise EOFError("Buffer short while skipping field")
  return new_offset

def onnx_load(model_path):
  parser = OnnxParser()
  with open(model_path, "rb") as f:
    onnx_model = parser.parse_model_proto_from_bytes(f.read())
  return onnx_model

class OnnxParser:
  def _parse_message(self, data, offset, message_field_handlers, initial_obj_factory=lambda: {}):
    obj = initial_obj_factory()
    current_offset = offset
    end_offset = len(data)
    while current_offset < end_offset:
      tag_val, after_tag_offset = decode_varint(data, current_offset)
      field_number = tag_val >> 3
      wire_type = tag_val & 0x07
      if handler := message_field_handlers.get(field_number): current_offset = handler(obj, data, after_tag_offset, wire_type)
      else: current_offset = skip_field_value(data, after_tag_offset, wire_type)
    return obj, current_offset

  def _handle_int64_field(self, obj, key_name, data, offset, wire_type, is_repeated=False):
    if wire_type != WIRETYPE_VARINT: raise ValueError(f"Expected varint for int64 field '{key_name}'")
    val, new_offset = decode_varint(data, offset)
    if is_repeated: obj.setdefault(key_name, []).append(val)
    else: obj[key_name] = val
    return new_offset

  def _handle_int32_field(self, obj, key_name, data, offset, wire_type, is_repeated=False):
    return self._handle_int64_field(obj, key_name, data, offset, wire_type, is_repeated)

  def _handle_float_field(self, obj, key_name, data, offset, wire_type, is_repeated=False):
    if wire_type != WIRETYPE_FIXED32: raise ValueError(f"Expected fixed32 for float field '{key_name}'")
    if offset + 4 > len(data): raise EOFError("Buffer too short for float")
    val, = struct.unpack("<f", data[offset:offset+4])
    if is_repeated: obj.setdefault(key_name, []).append(val)
    else: obj[key_name] = val
    return offset + 4

  # WIRETYPE_LENGTH_DELIMITED
  def _handle_delimited(self, data, offset):
    str_len, after_len_offset = decode_varint(data, offset)
    new_offset = after_len_offset
    if new_offset + str_len > len(data): raise EOFError("Buffer too short")
    value = data[new_offset : new_offset + str_len]
    return value, new_offset + str_len
  def _handle_string_field(self, obj, key_name, data, offset, wire_type, is_repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for string field '{key_name}'")
    value, off = self._handle_delimited(data, offset)
    value = value.decode('utf-8')
    if is_repeated: obj.setdefault(key_name, []).append(value)
    else: obj[key_name] = value
    return off
  def _handle_bytes_field(self, obj, key_name, data, offset, wire_type, is_repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for bytes field '{key_name}'")
    value, off = self._handle_delimited(data, offset)
    if is_repeated: obj.setdefault(key_name, []).append(value)
    else: obj[key_name] = value
    return off
  def _handle_packed_repeated_floats(self, obj, key_name, data, offset, wire_type):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed floats expected length_delimited")
    value, off = self._handle_delimited(data, offset)
    if len(value) % 4 != 0: raise ValueError("Packed float data length not multiple of 4")
    values = list(struct.unpack(f"<{len(value) // 4}f", value))
    obj.setdefault(key_name, []).extend(values)
    return off

  def _handle_packed_repeated_int64s(self, obj, key_name, data, offset, wire_type):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed int64s expected length_delimited")
    total_bytes_len, after_len_offset = decode_varint(data, offset)
    new_offset = after_len_offset
    packed_data_end = new_offset + total_bytes_len
    values = []
    current_packed_offset = new_offset
    while current_packed_offset < packed_data_end:
      val, current_packed_offset = decode_varint(data, current_packed_offset)
      values.append(val)
    obj.setdefault(key_name, []).extend(values)
    return packed_data_end

  def _handle_packed_repeated_int32s(self, obj, key_name, data, offset, wire_type):
    return self._handle_packed_repeated_int64s(obj, key_name, data, offset, wire_type)

  def _handle_sub_message_field(self, obj, key_name, data, offset, wire_type, parser_func, is_repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for sub-message field '{key_name}'")
    value, off = self._handle_delimited(data, offset)
    parsed_sub_obj, _ = parser_func(value)
    if is_repeated: obj.setdefault(key_name, []).append(parsed_sub_obj)
    else: obj[key_name] = parsed_sub_obj
    return off

  # OperatorSetIdProto
  def _opset_id_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_string_field(obj, 'domain', data, off, wt),
      2: lambda obj, data, off, wt: self._handle_int64_field(obj, 'version', data, off, wt),
    }
  # StringStringEntryProto
  def _string_string_entry_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_string_field(obj, 'key', data, off, wt),
      2: lambda obj, data, off, wt: self._handle_string_field(obj, 'value', data, off, wt),
    }
  def parse_opset_id_proto(self, data_bytes, offset=0): return self._parse_message(data_bytes, offset, self._opset_id_proto_handlers())
  def parse_string_string_entry_proto(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._string_string_entry_proto_handlers())
  # message TensorProto: Tensors, A serialized tensor value.
  def _tensor_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_int64_field(obj, 'dims', data, off, wt, is_repeated=True),
      2: lambda obj, data, off, wt: self._handle_int32_field(obj, 'data_type', data, off, wt), # TensorDataType enum
      4: lambda obj, data, off, wt: self._handle_packed_repeated_floats(obj, 'float_data',data,off,wt),
      5: lambda obj, data, off, wt: self._handle_packed_repeated_int32s(obj, 'int32_data', data, off, wt),
      6: lambda obj, data, off, wt: self._handle_bytes_field(obj, 'string_data',data,off,wt,is_repeated=True),
      7: lambda obj, data, off, wt: self._handle_packed_repeated_int64s(obj, 'int64_data', data, off, wt),
      8: lambda obj, data, off, wt: self._handle_string_field(obj, 'name', data, off, wt),
      9: lambda obj, data, off, wt: self._handle_bytes_field(obj, 'raw_data', data, off, wt),
    }
  def parse_tensor_proto(self, data_bytes, offset=0):
    obj, final_offset = self._parse_message(data_bytes, offset, self._tensor_proto_handlers(),
      lambda: {'dims': [], 'float_data': [], 'int32_data': [], 'string_data':[], 'int64_data':[], 'double_data':[], 'uint64_data':[]})
    return obj, final_offset

  # TensorShapeProto.Dimension
  def _tensor_shape_proto_dimension_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_int64_field(obj, 'dim_value', data, off, wt), # oneof value
      2: lambda obj, data, off, wt: self._handle_string_field(obj, 'dim_param', data, off, wt), # oneof value
      3: lambda obj, data, off, wt: self._handle_string_field(obj, 'denotation', data, off, wt),
    }
  def parse_tensor_shape_proto_dimension(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._tensor_shape_proto_dimension_handlers())
  # TensorShapeProto
  def _tensor_shape_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'dim', data, off, wt, self.parse_tensor_shape_proto_dimension,
                                                                   is_repeated=True),
    }
  def parse_tensor_shape_proto(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._tensor_shape_proto_handlers(), lambda: {'dim': []})

  # TypeProto.Tensor
  def _type_proto_tensor_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_int32_field(obj, 'elem_type', data, off, wt), # TensorDataType enum
      2: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'shape', data, off, wt, self.parse_tensor_shape_proto),
    }
  def parse_type_proto_tensor(self, data_bytes, offset=0): return self._parse_message(data_bytes, offset, self._type_proto_tensor_handlers())

  # TypeProto: Types, The standard ONNX data types.
  def _type_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'tensor_type', data, off, wt, self.parse_type_proto_tensor), # oneof value
      6: lambda obj, data, off, wt: self._handle_string_field(obj, 'denotation', data, off, wt),
    }
  def parse_type_proto(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._type_proto_handlers())
  # ValueInfoProto
  def _value_info_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_string_field(obj, 'name', data, off, wt),
      2: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'type', data, off, wt, self.parse_type_proto),
      3: lambda obj, data, off, wt: self._handle_string_field(obj, 'doc_string', data, off, wt),
      4: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'metadata_props', data, off, wt,
                                                                   self.parse_string_string_entry_proto, is_repeated=True),
    }
  def parse_value_info_proto(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._value_info_proto_handlers(), lambda: {'metadata_props': []})

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
  def _attribute_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_string_field(obj, 'name', data, off, wt),
      21:lambda obj, data, off, wt: self._handle_string_field(obj, 'ref_attr_name', data, off, wt),
      13:lambda obj, data, off, wt: self._handle_string_field(obj, 'doc_string', data, off, wt),
      20:lambda obj, data, off, wt: self._handle_int32_field(obj, 'type', data, off, wt), # AttributeType enum
      2: lambda obj, data, off, wt: self._handle_float_field(obj, 'f', data, off, wt),
      3: lambda obj, data, off, wt: self._handle_int64_field(obj, 'i', data, off, wt),
      4: lambda obj, data, off, wt: self._handle_bytes_field(obj, 's', data, off, wt),
      5: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 't', data, off, wt, self.parse_tensor_proto),
      6: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'g', data, off, wt, self.parse_graph_proto),
      7: lambda obj, data, off, wt: self._handle_float_field(obj, 'floats', data, off, wt, is_repeated=True),
      8: lambda obj, data, off, wt: self._handle_int64_field(obj, 'ints', data, off, wt, is_repeated=True),
      9: lambda obj, data, off, wt: self._handle_bytes_field(obj, 'strings', data, off, wt, is_repeated=True),
      10:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'tensors', data, off, wt, self.parse_tensor_proto, is_repeated=True),
      11:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'graphs', data, off, wt, self.parse_graph_proto, is_repeated=True),
    }
  def parse_attribute_proto(self, data_bytes, offset=0):
    obj, final_offset = self._parse_message(data_bytes, offset, self._attribute_proto_handlers(), lambda: {
      'floats': [], 'ints': [], 'strings': [], 'tensors': [], 'graphs': []})
    if 't' in obj and obj['t']: self.interpret_tensor_raw_data(obj['t'])
    if 'tensors' in obj:
      for tensor in obj['tensors']:
        self.interpret_tensor_raw_data(tensor)
    return obj, final_offset

  # NodeProto
  def _node_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_string_field(obj, 'input', data, off, wt, is_repeated=True),
      2: lambda obj, data, off, wt: self._handle_string_field(obj, 'output', data, off, wt, is_repeated=True),
      3: lambda obj, data, off, wt: self._handle_string_field(obj, 'name', data, off, wt),
      4: lambda obj, data, off, wt: self._handle_string_field(obj, 'op_type', data, off, wt),
      7: lambda obj, data, off, wt: self._handle_string_field(obj, 'domain', data, off, wt),
      5: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'attribute', data, off, wt, self.parse_attribute_proto, is_repeated=True),
      6: lambda obj, data, off, wt: self._handle_string_field(obj, 'doc_string', data, off, wt),
    }

  def parse_node_proto(self, data_bytes, offset=0):
    return self._parse_message(data_bytes, offset, self._node_proto_handlers(), lambda: {'input': [], 'output': [], 'attribute': []})
  # GraphProto
  def _graph_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'node', data, off, wt, self.parse_node_proto, is_repeated=True),
      2: lambda obj, data, off, wt: self._handle_string_field(obj, 'name', data, off, wt),
      5: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'initializer', data, off, wt, self.parse_tensor_proto, is_repeated=True),
      10:lambda obj, data, off, wt: self._handle_string_field(obj, 'doc_string', data, off, wt),
      11:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'input', data, off, wt, self.parse_value_info_proto, is_repeated=True),
      12:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'output', data, off, wt, self.parse_value_info_proto, is_repeated=True),
      13:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'value_info', data, off, wt, self.parse_value_info_proto, is_repeated=True),
    }

  # GraphProto
  def parse_graph_proto(self, data_bytes, offset=0):
    obj, final_offset = self._parse_message(data_bytes, offset, self._graph_proto_handlers(), lambda: {'node': [], 'initializer': [], 'input':[],
                                                                                                       'output':[], 'value_info':[]})
    for tensor in obj['initializer']: self.interpret_tensor_raw_data(tensor)
    return obj, final_offset

  # ModelProto
  def _model_proto_handlers(self):
    return {
      1: lambda obj, data, off, wt: self._handle_int64_field(obj, 'ir_version', data, off, wt),
      8: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'opset_import', data, off, wt, self.parse_opset_id_proto, is_repeated=True),
      2: lambda obj, data, off, wt: self._handle_string_field(obj, 'producer_name', data, off, wt),
      3: lambda obj, data, off, wt: self._handle_string_field(obj, 'producer_version', data, off, wt),
      4: lambda obj, data, off, wt: self._handle_string_field(obj, 'domain', data, off, wt),
      5: lambda obj, data, off, wt: self._handle_int64_field(obj, 'model_version', data, off, wt),
      6: lambda obj, data, off, wt: self._handle_string_field(obj, 'doc_string', data, off, wt),
      7: lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'graph', data, off, wt, self.parse_graph_proto),
      14:lambda obj, data, off, wt: self._handle_sub_message_field(obj, 'metadata_props', data, off, wt,
                                                                   self.parse_string_string_entry_proto, is_repeated=True),
    }
  def parse_model_proto_from_bytes(self, data_bytes):
    parsed_model, _ = self._parse_message(data_bytes, 0, self._model_proto_handlers(), lambda: {'opset_import': [], 'metadata_props': []})
    return parsed_model