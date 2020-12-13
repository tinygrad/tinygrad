from .tensor import Tensor, Function, register
from functools import lru_cache
import struct

@lru_cache
def compile_wrapper(ane, dat):
  return ane.compile(dat)

def roundup(x, v):
  return x + (v-x)%v

def fill(dat, addrs, type, val, base=0x4000):
  x = struct.pack(type, val)
  for a in addrs:
    dat[base+a:base+a+len(x)] = x
  return dat

@lru_cache
def compile_relu(ane, sz):
  dat = list(open("ane/ops/relu.hwx", "rb").read())
  # TODO: make this all nice and once
  # number of relus
  dat = fill(dat, [0x128, 0x13C], "H", sz)
  # number of engines? (max 0x100)
  dat = fill(dat, [0x1ec, 0x1f0, 0x1f4, 0x1f8], "I", max(0x100, roundup(sz*2, 0x10)))
  # strides?
  dat = fill(dat, [0x260, 0x264, 0x268], "I", roundup(sz*2, 0x40))
  return compile_wrapper(ane, bytes(dat))

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ret = ctx.ane.tensor(input.shape)
    ctx.ane.run(compile_relu(ctx.ane, input.sz), input, ret)
    return ret
register('relu', ReLU, device=Tensor.ANE)

