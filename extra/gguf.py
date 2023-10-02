import enum, os, struct, io
from typing import Any, Dict, Tuple

from tinygrad.helpers import dtypes, prod
from tinygrad.ops import Device
from tinygrad.tensor import Tensor

assert Device.DEFAULT == "METAL", "only metal is tested for now. gpu has a known bug in mac, careful."

# --- Compatibility junk
class GgmlDType(enum.Enum): F32 = 0; F16 = 1; Q4_0 = 2; Q4_1 = 3; Q5_0 = 6; Q5_1 = 7; Q8_0 = 8; Q8_1 = 9; Q2K = 10; Q3K = 11; Q4K = 12; Q5K = 13; Q6K = 14; Q8K = 15
class GGUFValueType(enum.IntEnum): UINT8 = 0; INT8 = 1; UINT16 = 2; INT16 = 3; UINT32 = 4; INT32 = 5; FLOAT32 = 6; BOOL = 7; STRING = 8; ARRAY = 9; UINT64 = 10; INT64 = 11; FLOAT64 = 12;
def read_string(reader: io.BufferedReader) -> str:
  str_len = struct.unpack("<Q", reader.read(8))[0]
  string_bytes = reader.read(str_len)
  while string_bytes and string_bytes[-1] == 0:
    string_bytes = string_bytes[:-1]  # Remove null terminators
  return string_bytes.decode("utf-8", "ignore")
def read_value(reader: io.BufferedReader, value_type: GGUFValueType) -> Any:
  if value_type == GGUFValueType.UINT8: return struct.unpack("<B", reader.read(1))[0]
  elif value_type == GGUFValueType.INT8: return struct.unpack("<b", reader.read(1))[0]
  elif value_type == GGUFValueType.UINT16: return struct.unpack("<H", reader.read(2))[0]
  elif value_type == GGUFValueType.INT16: return struct.unpack("<h", reader.read(2))[0]
  elif value_type == GGUFValueType.UINT32: return struct.unpack("<I", reader.read(4))[0]
  elif value_type == GGUFValueType.INT32: return struct.unpack("<i", reader.read(4))[0]
  elif value_type == GGUFValueType.FLOAT32: return struct.unpack("<f", reader.read(4))[0]
  elif value_type == GGUFValueType.UINT64: return struct.unpack("<Q", reader.read(8))[0]
  elif value_type == GGUFValueType.INT64: return struct.unpack("<q", reader.read(8))[0]
  elif value_type == GGUFValueType.FLOAT64: return struct.unpack("<d", reader.read(8))[0]
  elif value_type == GGUFValueType.BOOL: return struct.unpack("<?", reader.read(1))[0]
  elif value_type == GGUFValueType.STRING: return read_string(reader)
  elif value_type == GGUFValueType.ARRAY:
    array_type = GGUFValueType(struct.unpack("<I", reader.read(4))[0])
    array_len = struct.unpack("<Q", reader.read(8))[0]
    values = []
    for _ in range(array_len): values.append(read_value(reader, array_type))
    return values
ggml_to_tinygrad_dtype = { GgmlDType.F32: dtypes.float32, GgmlDType.F16: dtypes.float16, GgmlDType.Q4_0: dtypes.uint8, GgmlDType.Q4_1: dtypes.uint8, GgmlDType.Q5_0: dtypes.uint8, GgmlDType.Q5_1: dtypes.uint8, GgmlDType.Q8_0: dtypes.uint8, GgmlDType.Q8_1: dtypes.uint8, GgmlDType.Q2K: dtypes.uint8, GgmlDType.Q3K: dtypes.uint8, GgmlDType.Q4K: dtypes.uint8, GgmlDType.Q5K: dtypes.uint8, GgmlDType.Q6K: dtypes.uint8, GgmlDType.Q8K: dtypes.uint8 }
QK_K = 256; QK4_0 = 32; QS_SIZE = QK4_0 // 2
ggml_sizes = { GgmlDType.Q4_0: (18, 32), GgmlDType.F32: (4, 1), GgmlDType.F16: (2, 1), GgmlDType.Q6K: (3 * QK_K / 4 + QK_K / 16 + 2, QK_K),}
# --- startof integration logic

def read_model_params(reader: io.BufferedReader) -> Tuple[str, Any]:
  return read_string(reader), read_value(reader, GGUFValueType(read_value(reader, GGUFValueType.UINT32)))

def read_tensor_info(reader: io.BufferedReader) -> Tuple[str, Tuple, GgmlDType, int]: # Doesn't give the actual tensor data, just the information on how to load it
  name = read_string(reader)
  shape = [read_value(reader, GGUFValueType.UINT64) for _ in range(read_value(reader, GGUFValueType.UINT32))][::-1]
  ggml_dtype = GgmlDType(read_value(reader, GGUFValueType.UINT32))
  offset = read_value(reader, GGUFValueType.UINT64)
  return name, tuple(shape), ggml_dtype, offset

def read_data_offset(reader: io.BufferedReader, params: Dict) -> int: # This is where gguf actually stores the tensor data
  alignment = params.get("alignment", 32)
  return (reader.tell() + alignment - 1) // alignment * alignment

# Vectorized version of https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
def dequantize_q4_0_tensor(t: Tensor, n_blocks, shape):
  qs_size = 16; d_size = 2; block_size = d_size + qs_size
  d = t.reshape(n_blocks*2, block_size//2).cast(dtypes.half).to(Device.DEFAULT)[:, :d_size-1][:n_blocks]
  qs = t.cast(dtypes.uint8).reshape(n_blocks, block_size).to(Device.DEFAULT)[:, d_size:]
  x0 = qs - (qs / 16).floor() * 16 - 8 # qs & 0x0F - 8 (can I implement these bitshifts as methods on Tensor?)
  x1 = (qs / 16).floor() - 8 # (qs >> 4) - 8
  return Tensor.stack([x0 * d, x1 * d]).transpose(0, 1).reshape(shape)

def dequantize_q6k_tensor(t: Tensor, n_blocks, shape):
  qk_k = 256; ql_size = qk_k // 2; qh_size = qk_k // 4; scales_size = qk_k // 16; d_size =2; block_size = ql_size + qh_size + scales_size + d_size
  offset = 0
  ql = t.to(Device.DEFAULT).reshape(n_blocks, block_size)[:, offset:ql_size]
  offset += ql_size
  qh = t.to(Device.DEFAULT).reshape(n_blocks, block_size)[:, offset:offset+qh_size]
  offset += qh_size
  sc = t.cast(dtypes.int8).to(Device.DEFAULT).reshape(n_blocks, block_size)[:, offset:offset+scales_size]
  offset += scales_size
  d = t.cast(dtypes.half).to(Device.DEFAULT)[offset//2::offset//2+1][:n_blocks].reshape(-1, 1)

  qlb = ql[:, ql_size//2:]; qhb = qh[:, qh_size//2:]; scb = sc[:, scales_size//2:]

  # --- q1
  qlx = Tensor.stack([ql[:, :32], qlb[:, :32]]).transpose(0, 1)
  qhx = Tensor.stack([qh[:, :32], qhb[:, :32]]).transpose(0, 1)
  ql1 = qlx - (qlx / 16).floor() * 16
  qh1 = (qhx - (qhx / 4).floor() * 4) * 16
  q1 = ql1.cast(dtypes.int8) + qh1.cast(dtypes.int8) - 32

  # --- q2
  qlx = Tensor.stack([ql[:, 32:32*2], qlb[:, 32:32*2]]).transpose(0, 1)
  qhx = Tensor.stack([qh[:, :32], qhb[:, :32]]).transpose(0, 1)
  ql2 = qlx - (qlx / 16).floor() * 16
  qh2 = (((qhx / 4).floor()) - ((qhx / 16).floor()) * 4) * 16
  q2 = ql2.cast(dtypes.int8) + qh2.cast(dtypes.int8) - 32

  # --- q3
  qlx = Tensor.stack([ql[:, :32], qlb[:, :32]]).transpose(0, 1)
  qhx = Tensor.stack([qh[:, :32], qhb[:, :32]]).transpose(0, 1)
  ql3 = (qlx / 16).floor()
  qh3 = ((qhx / 16).floor() - (((qhx / 16).floor() / 4).floor()) * 4) * 16
  q3 = ql3.cast(dtypes.int8) + qh3.cast(dtypes.int8) - 32

  # --- q4
  qlx = Tensor.stack([ql[:, 32:32*2], qlb[:, 32:32*2]]).transpose(0, 1)
  qhx = Tensor.stack([qh[:, :32], qhb[:, :32]]).transpose(0, 1)
  ql4 = (qlx / 16).floor()
  qh4 = (((qhx / 16).floor() / 4).floor()) * 16
  q4 = ql4.cast(dtypes.int8) + qh4.cast(dtypes.int8) - 32

  ys = []
  for i, q in enumerate([q1, q2, q3, q4]):
    scx = Tensor.stack([sc[:, i*2:i*2+2], scb[:, i*2:i*2+2]]).transpose(0, 1)
    y = d[:, :, None] * q
    half = y.shape[-1] // 2
    ys.append(Tensor.cat(y[:, :, :half] * scx[:, :, :1], y[:, :, half:] * scx[:, :, 1:], dim=2))

  return Tensor.cat(*ys, dim=2).reshape(shape)

def tinygrad_tensor_from_gguf(disk_tensor: Tensor, name: str, shape: Tuple, ggml_dtype: GgmlDType, offset: int, data_offset: int) -> Tensor:
  itemsize, block_size = ggml_sizes[ggml_dtype]
  size_in_bytes = int(prod(shape) * itemsize // block_size)
  init_offset = data_offset + offset
  n_blocks = int(size_in_bytes // itemsize)
  fxn_for_dtype = {
      GgmlDType.F32: lambda ret: ret[:n_blocks].cast(dtypes.float32),
      GgmlDType.Q4_0: lambda ret: dequantize_q4_0_tensor(ret, n_blocks, shape),
      GgmlDType.Q6K: lambda ret: dequantize_q6k_tensor(ret, n_blocks, shape),
  }
  return fxn_for_dtype[ggml_dtype](disk_tensor[init_offset:init_offset+size_in_bytes])

def gguf_load(fn:str):
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
  with open(fn, "rb") as f:
    magic = read_value(f, GGUFValueType.UINT32)
    version = read_value(f, GGUFValueType.UINT32); assert version == 2, f"unsupported version {version}"
    tensor_count = read_value(f, GGUFValueType.UINT64)
    param_count = read_value(f, GGUFValueType.UINT64)
    params = {k: v for k, v in [read_model_params(f) for _ in range(param_count)]}
    tensor_info = [read_tensor_info(f) for _ in range(tensor_count)]
    data_offset = read_data_offset(f, params)
    return {info[0]: tinygrad_tensor_from_gguf(t, *info, data_offset) for info in tensor_info}, params
