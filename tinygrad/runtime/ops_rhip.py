import ctypes
from tinygrad.device import Compiled, Compiler, CompilerOptions, MallocAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.driver.hip_comgr import compile_hip

class RHIPCompiler(Compiler):
  compiler_opts = CompilerOptions("HSA", has_tensor_cores=True, shared_max=65536)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

rhip = ctypes.CDLL("/Users/qazal/code/tinygrad/remu/target/release/libremu.dylib")
class RHIPProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
  def __call__(self, *args, global_size, local_size, vals=(), wait=False):
    args = (*args, *vals)
    rhip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None,
                              len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class RHIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(device, MallocAllocator, RHIPCompiler("gfx1100"), RHIPProgram)
