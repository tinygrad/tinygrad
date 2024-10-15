from functools import partial
from math import prod
import struct
from typing import Callable
from tinygrad.dtype import DType, dtypes
from tinygrad.tensor import Tensor

class GGUFConverters:
  def conv_bc(dtype: DType, t: Tensor, n: int): return t[:dtype.itemsize * n].bitcast(dtype)
  def dequantize_q8_0(t: Tensor, n: int):
    blocks = t[:(n//32)*34].reshape((-1, 34))
    return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].cast(dtypes.int8)
  def dequantize_q4_0(t: Tensor, n: int):
    blocks = t[:(n//32)*18].reshape((-1, 18))
    delta = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    return (GGUFConverters._q_to_uint8(blocks[:,2:], 4).cast(dtypes.int8) - 8) * delta
  def dequantize_q4_1(t: Tensor, n: int):
    blocks = t[:(n//32)*20].reshape((-1, 20))
    d, m = tuple(blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
    return GGUFConverters._q_to_uint8(blocks[:,4:], 4).cast(dtypes.int8) * d + m
  def dequantize_q6_K(t: Tensor, n: int):
    blocks = t[:(n//256)*210].reshape((-1, 210))
    xl: Tensor = GGUFConverters._q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4)
    xh: Tensor = GGUFConverters._q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
    x = xl.bitwise_or(xh).bitcast(dtypes.int8) - 32
    scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((blocks.shape[0], 16, 16)).reshape((blocks.shape[0], 2, 128))
    d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256)).reshape((-1, 2, 128))
    return d * x * scales
  def _q_to_uint8(t: Tensor, b: int):
    nels = 8 // b
    shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(nels) ]), 0xff >> (8 - b)
    return t.unsqueeze(-1).expand((*t.shape,nels)).div(shift_tensor, upcast=False).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)
  type_map: dict[int, Callable[[Tensor, int], Tensor]] = { 0: partial(conv_bc, dtypes.float32), 1: partial(conv_bc, dtypes.float16),
    16: partial(conv_bc, dtypes.int8), 17: partial(conv_bc, dtypes.int16), 18: partial(conv_bc, dtypes.int32), 2: dequantize_q4_0,
    3: dequantize_q4_1, 14: dequantize_q6_K, 8: dequantize_q8_0 }

def load_gguf(tensor: Tensor) -> tuple[dict, dict[str, Tensor]]:
  if tensor.dtype != dtypes.uint8: raise ValueError("GGUF tensor must have dtype uint8!")
  cursor_pos, read_buffer, rb_start, kv_data, tensor_data = 0, memoryview(bytes()), 0, {}, {}
  def read_bytes(n: int):
    nonlocal cursor_pos, read_buffer, rb_start
    if rb_start + len(read_buffer) < cursor_pos + n: rb_start, read_buffer = cursor_pos, tensor[cursor_pos:(cursor_pos+max(n, 1000_000))].data()
    return read_buffer[cursor_pos-rb_start:(cursor_pos:=cursor_pos+n)-rb_start]
  def read_unpack(fmt: str, n: int): return struct.unpack(fmt, read_bytes(n))[0]
  def read_string(): return str(read_bytes(read_uint64()), "utf-8")
  def read_array():
    reader, n = readers[read_int32()], read_uint64()
    return [ reader() for _ in range(n) ]

  readers: dict[int, Callable] = { 8: read_string, 9: read_array } | { k: partial(read_unpack, *args) for k, args in { 0: ("<c", 1), 1: ("<b", 1),
    2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8) }.items() }
  read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

  if read_bytes(4) != b"GGUF": raise ValueError("Invalid GGUF file.")
  version, n_tensors, n_kv = read_int32(), read_int64(), read_int64()
  if version not in [2, 3]: raise NotImplementedError("Only GGUF v3 and v2 are supported!")
  for _ in range(n_kv):
    k, t = read_string(), read_int32()
    kv_data[k] = readers[t]()

  tensor_infos = [ (read_string(), tuple(read_uint64() for _ in range(read_uint32())), read_int32(), read_uint64()) for _ in range(n_tensors) ]
  alignment = kv_data.get("general.alignment", 32)
  data_start = cursor_pos = cursor_pos + (alignment - cursor_pos % alignment if cursor_pos % alignment != 0 else 0)

  for name, shape, ttype, offset in tensor_infos:
    tensor_data[name] = GGUFConverters.type_map[ttype](tensor[data_start + offset:], prod(shape)).reshape(*reversed(shape))
  return kv_data, tensor_data
