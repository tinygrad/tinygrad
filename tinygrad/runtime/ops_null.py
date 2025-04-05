from tinygrad.device import Compiled, Compiler, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner

class NullRenderer(Renderer):
  def render(self, uops:list) -> str: return ""

class NullProgram:
  def __init__(self, name:str, lib:bytes): pass
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    return 1e-4

class NullAllocator(Allocator):
  dev = None
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src): pass
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): pass

class NullGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

class NullDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NullAllocator(), NullRenderer(), Compiler(), NullProgram, NullGraph)
