# RISK architecture is going to change everything
# implement on S7t-VG6

import numpy as np
from .tensor import Function

# 16x32 * 32x32 -> 16x32 matmul = 32768 FLOPS @ 1 GHz = 32 TOPS
# 16x32 (aka 512 element) ALU
# 512 wide permute
# 512 wide load/store (1 cycle to SRAM)
# all in elements, aka TF32 (19 bits)

# targets:
#   matmul input
#   matmul weights(0-16)
#   matmul weights(16-32)
#   ALU
#   permute

# 1024x1024x4x19 bits = 10MB
# fully strided
# load512 <target>, <address>, <stride x (16)>, <stride y (32)>

# 4 slots
# <input> <weight> <output> <empty>
# <empty> <output> <input> <weight>
# <weight> <input> <empty> <output>

sram = np.zeros((1024*1024*4), dtype=np.float32)
regfile = {}

from enum import Enum
class Target(Enum):
  MATMUL_INPUT = 0
  MATMUL_WEIGHTS_LO = 1
  MATMUL_WEIGHTS_HI = 2
  ALU = 3
for t in Target:
  regfile[t] = np.zeros((16, 32))
  

def risk_instruction_matmul(x, w):
  assert x.shape == (16,32)
  assert w.shape == (32,32)
  return x@w

def risk_instruction_load512(target, address, stride_x, stride_y):
  d = regfile[target]
  for x in range(0, 16):
    for y in range(0, 32):
      d[x, y] = sram[address] 
      address += stride_y
    address += stride_x

def risk_instruction_store512(target, address, stride_x, stride_y):
  d = regfile[target]
  for x in range(0, 16):
    for y in range(0, 32):
      sram[address] = d[x, y]
      address += stride_y
    address += stride_x

def risk_matmul(x, w):
  pass


class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    if type(ctx.stride) == int:
      ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_ = x.shape[0], x.shape[1]
    oy,ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups
    
    if H == 1 and W == 1 and ctx.groups == 1 and ctx.stride == (1,1):
      d1 = x.shape[0] * x.shape[2] * x.shape[3]
      assert x.shape[1] == w.shape[1]
      d2 = x.shape[1]
      d3 = w.shape[0]
      print("1x1 CONV FAST == %d x %d x %d matmul" % (d1, d2, d3))

      x11 = x.reshape(x.shape[1], x.shape[2]*x.shape[3]).T
      w11 = w.reshape(w.shape[0], w.shape[1]).T
      print(x11.shape, w11.shape)

      ret = x11 @ w11

      return ret.T.reshape(1, w.shape[0], x.shape[2], x.shape[3])

    gx = x.reshape(bs,ctx.groups,cin,x.shape[2],x.shape[3])
    tx = np.lib.stride_tricks.as_strided(gx,
      shape=(bs, ctx.groups, cin, oy, ox, H, W),
      strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
      writeable=False,
    )
    tw = w.reshape(ctx.groups, rcout, cin, H, W)
    ctx.save_for_backward(tx, tw, x.shape)

    ret = np.zeros((bs,ctx.groups,oy,ox,rcout),dtype=x.dtype)
    for g in range(ctx.groups):
      #ijYXyx,kjyx -> iYXk ->ikYX
      ret[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
    ret = np.moveaxis(ret,4,2).reshape(bs, cout, oy, ox)

    print(x.shape, w.shape, "->", ret.shape)
    return ret


  def backward(ctx, grad_output):
    pass

