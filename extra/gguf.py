import enum, os, struct, io
import numpy as np
from typing import Any, Dict, List, Tuple
from tinygrad.helpers import dtypes, prod
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
import sys

Device.DEFAULT = "GPU"
INDEX = int(sys.argv[1])

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
  d = t.cast(dtypes.float16).to("GPU").reshape(n_blocks, block_size).reshape(-1, 9)[:, :d_size-1]
  qs = t.cast(dtypes.uint8).to("GPU").reshape(n_blocks, block_size)[:, d_size:]
  print(d.numpy()[INDEX], qs.numpy()[INDEX])
  """
  x0 = qs - (qs / 16).floor() * 16 - 8 # qs & 0x0F - 8
  x1 = (qs / 16).floor() - 8 # (qs >> 4) - 8
  ret =  Tensor.stack([x0 * d, x1 * d]).transpose(0, 1).reshape(shape)
  return ret
  """

def tinygrad_tensor_from_gguf(disk_tensor: Tensor, name: str, shape: Tuple, ggml_dtype: GgmlDType, offset: int, data_offset: int) -> Tensor:
  itemsize, block_size = ggml_sizes[ggml_dtype]
  size_in_bytes = int(prod(shape) * itemsize // block_size)
  init_offset = data_offset + offset
  n_blocks = size_in_bytes // itemsize
  fxn_for_dtype = {
      GgmlDType.F32: lambda ret: ret[:n_blocks].cast(dtypes.float32),
      GgmlDType.Q4_0: lambda ret: dequantize_q4_0_tensor(ret, n_blocks, shape),
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
    # return [tinygrad_tensor_from_gguf(t, *info, data_offset) for info in tensor_info], params TODO uncomment once dequantize is fast
    tinygrad_tensor_from_gguf(t, *tensor_info[0], data_offset)
    """
    np_tensor = tensor.numpy()
    flat = np_tensor.flatten()
    idxs = np.argwhere(np.isnan(flat))
    print(flat[INDEX])
    """
