import ctypes, functools
from tinygrad.helpers import from_mv, init_c_var
from tinygrad.device import Compiled, LRUAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.runtime.ops_hip import compile_hip

hip = ctypes.CDLL("/usr/local/lib/libremu.so")

class EmulatedHIPProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
  def __call__(self, *args, global_size, local_size, vals=(), wait=False):
    assert len(vals) == 0
    hip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class EmulatedHIPAllocator(LRUAllocator):
  def __init__(self): super().__init__()
  def _alloc(self, size:int): return init_c_var(ctypes.c_void_p(None), lambda x: hip.hipMalloc(ctypes.byref(x), size))
  def copyin(self, dest, src): hip.hipMemcpy(dest, from_mv(src), len(src), 1)
  def copyout(self, dest, src): hip.hipMemcpy(from_mv(dest), src, len(dest), 2)

class EmulatedHIPDevice(Compiled):
  def __init__(self, device=""):
    self.arch = "gfx1100"
    super().__init__(EmulatedHIPAllocator(), LinearizerOptions("HIP"), HIPRenderer,
                     functools.partial(compile_hip,arch=self.arch), f"compile_hip_{self.arch}", functools.partial(EmulatedHIPProgram))
