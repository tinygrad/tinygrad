import ctypes
from tinygrad.device import Compiled, MallocAllocator
from tinygrad.runtime.ops_hip import HIPCompiler

hip = ctypes.CDLL("/usr/local/lib/libremu.so")

class EmulatedHIPProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
  def __call__(self, *args, global_size, local_size, vals=(), wait=False):
    args = (*args, *vals)
    hip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class EmulatedHIPDevice(Compiled):
  def __init__(self, device=""):
    super().__init__(device, MallocAllocator, HIPCompiler("gfx1100"), EmulatedHIPProgram)
