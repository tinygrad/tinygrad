import functools, inspect
from tinygrad.device import Compiled, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.renderer.cstyle import Renderer, CStyleLanguage, AMDHIPRenderer
from tinygrad.uop.ops import Ops
from tinygrad.helpers import cpu_profile, EMULATE, NULL_CC, NULL_ALLOW_COPYOUT
from tinygrad.renderer import cstyle, nir, ptx, llvmir, wgsl

class NullRenderer(CStyleLanguage):
  device = "NULL"
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}

class NullProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs): self.device, self.name = device, name
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    with cpu_profile(self.name, self.device): return 1e-3

class NullAllocator(Allocator['NullDevice']):
  def _alloc(self, size, options): pass
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src):
    if not NULL_ALLOW_COPYOUT: raise RuntimeError("no copyout on NULL")
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    with cpu_profile(f"{src_dev.device} -> {dest_dev.device}", self.dev.device): pass
  def _offset(self, buf, offset:int, size:int): pass

class NullGraph(MultiGraphRunner):
  def __call__(self, input_buffers, var_vals, wait=False) -> float|None: return 1e-1

class NullDevice(Compiled):
  def __init__(self, device:str):
    renderer:type[Renderer]
    match str(EMULATE.value):
      case "AMD": renderer, self.arch = AMDHIPRenderer, "gfx1100"
      case "AMD_RDNA4": renderer, self.arch = AMDHIPRenderer, "gfx1201"
      case "AMD_CDNA4": renderer, self.arch = AMDHIPRenderer, "gfx950"
      case "": renderer = NullRenderer
      case _: raise RuntimeError(f"can't EMULATE device: {EMULATE.value}")
    def _name(ren): return ren.device + f":{shortname}" if (shortname:=ren.__name__.upper().removesuffix('RENDERER').removeprefix(ren.device)) else ""
    renderers = {'': renderer, **{_name(renderer):renderer for mod in (cstyle, nir, ptx, llvmir, wgsl) for renderer in mod.__dict__.values()
                                  if inspect.isclass(renderer) and issubclass(renderer, Renderer) and renderer.device}}
    super().__init__(device, NullAllocator(self), renderers, functools.partial(NullProgram, device), NullGraph, ctrl_var=NULL_CC)
