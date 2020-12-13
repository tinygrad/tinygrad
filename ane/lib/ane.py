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

class ANETensor:
  def __init__(self, *shape):
    self.shape = shape
    self.dtype = np.float16
    self.sz = int(np.prod(shape))
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
  print(tind)
  print(toutd)

  comp = ane.compile(open("../2_compile/model.hwx", "rb").read())
  ret = ane.run(comp, tin, tout)

  print(tind)
  print(toutd)

