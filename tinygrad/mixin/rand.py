from __future__ import annotations
from typing import Self
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import ceildiv, prod
from tinygrad.mixin import OpMixin


class RandMixin(OpMixin):
  @staticmethod
  def _threefry_random_bits(key, counts0, counts1):
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = x.threefry((key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    return (x & 0xffffffff).cast(dtypes.uint32).cat(((x >> 32) & 0xffffffff).cast(dtypes.uint32))

  @classmethod
  def random_bits(cls, key:Self, counter:Self, num:int) -> Self:
    low, high = counter[0:1], counter[1:2]
    bits = []
    for i in range(0, num, dtypes.uint32.max):
      chunk_num = min(num - i, dtypes.uint32.max)
      c_low = low + (i & 0xffffffff)
      c_high = high + (i >> 32) + (c_low < low).cast(dtypes.uint32)
      new_key = cls._threefry_random_bits(key, c_low, c_high)
      counts0 = cls.arange(ceildiv(chunk_num, 2), dtype=dtypes.uint32)
      counts1 = counts0 + ceildiv(chunk_num, 2)
      bits.append(cls._threefry_random_bits(new_key, counts0, counts1)[:chunk_num])
    return bits[0].cat(*bits[1:])

  @staticmethod
  def _bits_to_rand(bits, shape:tuple[int, ...], dtype:DType):
    _, nmant = dtypes.finfo(dtype)
    uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
    uint_bits = bits.bitcast(uint_dtype)
    float_one_bits = uint_bits.const_like(1).cast(dtype).bitcast(uint_dtype)
    return uint_bits.rshift(dtype.bitsize - nmant).bitwise_or(float_one_bits).bitcast(dtype)[:prod(shape)].sub(1).reshape(shape)
