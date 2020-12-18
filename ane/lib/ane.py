#!/usr/bin/env python3
import os
from ctypes import *
import numpy as np
import faulthandler
faulthandler.enable()

libane = cdll.LoadLibrary(os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "libane.dylib"))

libane.ANE_Compile.argtypes = [c_char_p, c_int]
libane.ANE_Compile.restype = c_void_p

libane.ANE_TensorCreate.restype = c_void_p

libane.ANE_TensorData.argtypes = [c_void_p]
libane.ANE_TensorData.restype = POINTER(c_uint16)

libane.ANE_Run.argtypes = [c_void_p]*3
libane.ANE_Run.restype = c_int

ANE_Struct = [
# section @ 0x2C len 0xF4

# section @ 0x128 len 0x3C (conv)
  ("u16", 0x128, "InputWidth"),
  ("u16", 0x12A, "InputHeight"),
  ("u16", 0x12C, "InputDepth"),

  ("u32", 0x130, "InputOutputType"),   # (OutputType * 0x10) | InputType

  ("u32", 0x134, "InputChannels"),
  ("u32", 0x138, "OutputChannels"),

  ("u16", 0x13C, "OutputWidth"),
  ("u16", 0x13E, "OutputHeight"),
  ("u16", 0x140, "OutputDepth"),

  ("u16", 0x144, "KernelSize"),        # 0xa000 | (KernelHeight * 0x20) | KernelWidth
  ("u16", 0x146, "Padding"),           # 0x5000 | (PadTop * 0x40) | (PadLeft * 2)

  ("u16", 0x14C, "BatchSize"),

# section @ 0x16C len 0x6C (input)
  # reloc 0x16c-0x174 = image
  ("u32", 0x178, "InputRowStride"),
  ("u32", 0x17C, "InputPlaneStride"),
  ("u32", 0x180, "InputDepthStride"),
  ("u32", 0x184, "InputBatchStride"),

  ("u8",  0x1A7, "InputInterleave"),

# section @ 0x1E0 len 0x44

# section @ 0x22c len 0xC (scaling)
  ("u16", 0x230, "BiasScalar"),
  ("u16", 0x232, "ScaleScalar"),

# section @ 0x240 len 0x10

# section @ 0x258 len 0x18
  ("u32", 0x260, "OutputRowStride"),
  ("u32", 0x264, "OutputPlaneStride"),
  ("u32", 0x268, "OutputDepthStride"),
  ("u32", 0x26C, "OutputBatchStride"),

  ("u8",  0x273, "OutputInterleave")]


class ANETensor:
  def __init__(self, *shape):
    self.shape = shape
    self.dtype = np.float16
    self.sz = int(np.prod(shape))
    assert(self.sz <= 0x4000)
    self.tt = libane.ANE_TensorCreate(self.sz, 1)
    assert(self.tt is not None)

  def data(self):
    data = libane.ANE_TensorData(self.tt)
    assert(data is not None)
    #print(hex(addressof(data.contents)))
    buf = np.ctypeslib.as_array(data, shape=(self.sz,))
    ret = np.frombuffer(buf, dtype=self.dtype)
    #print(ret.data)
    return ret

class ANE:
  def __init__(self):
    libane.ANE_Open()

  def compile(self, dat):
    ret = libane.ANE_Compile(create_string_buffer(dat), len(dat))
    assert(ret is not None)
    return ret

  def run(self, prog, tin, tout):
    libane.ANE_Run(prog, tin.tt, tout.tt)

  def tensor(self, shape):
    return ANETensor(shape)

if __name__ == "__main__":
  ane = ANE()

  tin = ANETensor(16)
  tout = ANETensor(16)

  tind = tin.data()
  toutd = tout.data()

  tind[0:4] = [-1,1,-2,2]
  print("** before **")
  print(tind)
  print(toutd)

  comp = ane.compile(open("../ops/relu.hwx", "rb").read())
  ret = ane.run(comp, tin, tout)
  print("** after **")
  print(tind)
  print(toutd)
