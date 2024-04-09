from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Dict, Tuple, ClassVar, NamedTuple
import importlib, inspect, functools, pathlib, time, ctypes, os
from tinygrad.helpers import ansilen, prod, getenv, colored, all_int, to_function_name, from_mv, flat_mv, diskcache_get, diskcache_put
from tinygrad.helpers import DEBUG, CACHECOLLECTING, BEAM, NOOPT, GlobalCounters
from tinygrad.shape.symbolic import Variable, sym_infer, sint
from tinygrad.ops import LazyOp, get_lazyop_info
from tinygrad.buffer import Buffer, BufferOptions
from tinygrad.codegen.uops import UOpGraph

if TYPE_CHECKING:
  from tinygrad.codegen.linearizer import Linearizer

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._devices: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]  # noqa: E501
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "")   # noqa: E501
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT
  def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    if DEBUG >= 1: print(f"opening device {ix} from pid:{os.getpid()}")
    x = ix.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._devices, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "HSA", "CUDA", "GPU", "CLANG", "LLVM"]:
      try:
        if self[device]: return device
      except Exception: pass
    raise RuntimeError("no usable devices")
Device = _Device()

# **************** base Runner + helpers ****************

class JITRunner:
  def __init__(self):
    self.op_estimate:sint = 0
    self.mem_estimate:sint = 0
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    var_vals = var_vals if var_vals is not None else {}
    from tinygrad.engine.jit import CacheCollector
    et = self(rawbufs, var_vals)
    if CACHECOLLECTING: CacheCollector.add(self, rawbufs, var_vals)
    return et
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    raise NotImplementedError("override this")

def update_stats(name:str, op_estimate:sint, mem_estimate:sint, var_vals: Optional[Dict[Variable, int]], et: Optional[float], buf_count:int, jit=False, num_kernels=1, lra: Optional[Dict]=None, device:str="", first_run=False):  # noqa: E501
  if var_vals is None: var_vals = {}
  op_estimate = sym_infer(op_estimate, var_vals)
  mem_estimate = sym_infer(mem_estimate, var_vals)
  GlobalCounters.kernel_count += num_kernels
  GlobalCounters.global_ops += op_estimate
  GlobalCounters.global_mem += mem_estimate
  if et is not None: GlobalCounters.time_sum_s += et
  if DEBUG >= 2:
    ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
    print(f"{colored(f'*** {device[:7]:7s} {GlobalCounters.kernel_count:4d}', ('magenta' if num_kernels == 1 else 'CYAN') if jit else ('green' if first_run else None))} {name+' '*(38-ansilen(name))} arg {buf_count:3d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
          (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))  # noqa: E501

# **************** Buffer / Allocator ****************

class BufferCopy(JITRunner):
  def copy(self, dest, src):
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_fd') and src.nbytes >= 4096 and src._buf.ud.fd is not None:
      dest.allocator.copy_from_fd(dest._buf, src._buf.ud.fd, src._buf.offset, src.nbytes)
    elif src.device.startswith("DISK") and hasattr(dest.allocator, 'as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator.copyout(dest.allocator.as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    et = None
    if wait or DEBUG >= 2:
      Device[dest.device].synchronize()
      et = time.perf_counter() - st
    total_sz = dest.size*dest.dtype.itemsize
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    update_stats(colored(name, "yellow"), 0, total_sz, {}, et, 2, jit, device=dest.device)

class BufferXfer(BufferCopy):
  def copy(self, dest, src):
    if hasattr(dest.allocator.device, "track_cross_buffer") and hasattr(src.allocator, "track_cross_device"):
      dest.allocator.device.track_cross_buffer.append(src)
      src.allocator.track_cross_device.add(dest.allocator.device)
    dest.allocator.transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.device, dest_dev=dest.allocator.device)

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    assert not isinstance(size, int) or size > 0, f"alloc size must be positve, getting {size}"
    return self._alloc(size, options if options is not None else BufferOptions())
  def _alloc(self, size:int, options:BufferOptions): raise NotImplementedError("need alloc")
  def free(self, opaque, size:int, options:Optional[BufferOptions]=None):
    self._free(opaque, options if options is not None else BufferOptions())
  def _free(self, opaque, options:BufferOptions): pass  # if opaque is a Python object, you don't need a free
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):  # pylint: disable=abstract-method
  def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):
    if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferOptions): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
MallocAllocator = _MallocAllocator()

# **************** for Compiled Devices ****************

class CompilerOptions(NamedTuple):
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_shared: bool = True
  has_tensor_cores: bool = False
  # NOTE: these two should be in z,y,x(reversed) order for cstyle backends, they are flipped when kernel is rendered
  global_max: Optional[List[int]] = None
  local_max: Optional[List[int]] = None
  shared_max: int = 32768

class Compiler:
  compiler_opts: ClassVar[CompilerOptions]
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def render(self, name:str, uops:UOpGraph) -> str: raise NotImplementedError("need a render function")
  def compile(self, src:str) -> bytes: raise NotImplementedError("need a compile function")
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib

class CompiledASTRunner(JITRunner):
  def __init__(self, name:str, prg:str, dname:str, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None,
               variables:Optional[List[Variable]]=None, op_estimate:sint=0, mem_estimate:sint=0, precompiled:Optional[bytes]=None, outcount:int=1):
    super().__init__()
    if DEBUG >= 4: print(prg)
    if global_size is not None: global_size = global_size + [1]*(3-len(global_size))
    if local_size is not None: local_size = local_size + [1]*(3-len(local_size))
    self.name, self.display_name, self.prg, self.dname, self.global_size, self.local_size, self.first_run = \
      to_function_name(name), name, prg, dname, global_size, local_size, True
    assert self.device.compiler is not None, "compiler is required to make an AST kernel"
    lib:bytes = precompiled if precompiled is not None else self.device.compiler.compile_cached(prg)
    self.lib, self.clprg, self.outcount = lib, self.device.runtime(self.name, lib), outcount
    self.vars: List[Variable] = [] if variables is None else variables
    self.op_estimate, self.mem_estimate = op_estimate, mem_estimate

  @property
  def device(self): return Device[self.dname]

  def __reduce__(self):
    return self.__class__, (self.name, self.prg, self.dname, self.global_size, self.local_size,
                            self.vars, self.op_estimate, self.mem_estimate, self.lib)

  def launch_dims(self, var_vals):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else self.global_size
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else self.local_size
    return global_size, local_size

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    global_size, local_size = self.launch_dims(var_vals)
    if global_size is not None and local_size is None and all_int(self.global_size): # type: ignore[arg-type]
      # TODO: this is copied from get_program
      from tinygrad.features.search import optimize_local_size
      local_size = self.local_size = optimize_local_size(self.clprg, global_size, rawbufs)
      global_size = self.global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    lra = {}
    if global_size: lra['global_size'] = global_size
    if local_size: lra['local_size'] = local_size
    et = self.clprg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.vars), wait=wait or DEBUG>=2)
    if do_update_stats: update_stats(self.display_name, self.op_estimate, self.mem_estimate, var_vals, et, len(rawbufs), jit,
                                     lra=lra, device=self.dname, first_run=self.first_run)
    self.first_run = False
    return et

class MultiDeviceJITGraph(JITRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    raise NotImplementedError("override this")

logkern, logkern_level = open(getenv("LOGKERN", ""), "a") if getenv("LOGKERN", "") else None, getenv("LOGKERN_LEVEL", 1)
class Compiled:
  def __init__(self, device:str, allocator:Allocator, compiler:Optional[Compiler], runtime, graph=None):
    self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler, runtime, graph
  def synchronize(self): pass  # override this in your device

  def to_program(self, k:Linearizer) -> CompiledASTRunner:
    assert self.compiler is not None, "compiler is required to run AST"
    k.linearize()
    info = get_lazyop_info(k.ast[0])
    ops, mem = k.uops.flops_mem()
    run_count = prod((k.global_size if k.global_size else []) + (k.local_size if k.local_size else []))
    # NOTE: we use min here to ignore the indexing FLOPS
    ret = CompiledASTRunner(k.name, self.compiler.render(to_function_name(k.name), k.uops), self.dname, k.global_size, k.local_size,
                            k.uops.vars(), min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count), outcount=len(k.outbufs))
    return ret

  def get_linearizer(self, *ast:LazyOp) -> Linearizer:
    assert self.compiler is not None, "compiler is required to build AST"
    if DEBUG >= 3:
      from tinygrad.features.graph import print_tree
      for op in ast: print_tree(op)
    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(*ast, opts=self.compiler.compiler_opts)
    k.required_optimizations()
    if not NOOPT:
      if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
      if BEAM >= 1:
        from tinygrad.features.search import beam_search, time_linearizer, bufs_from_lin
        kb, k_opt = Linearizer(*ast, opts=self.compiler.compiler_opts), k
        kb.required_optimizations()
        rawbufs = bufs_from_lin(kb, allocate=False)
        k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
        if getenv("BEAM_COMPARE", 1):
          # TODO: move the HC/TC/BEAM compare to beam_search so it can be optionally cached which choice is better
          lins = [(f"beam{BEAM.value}", k), (("tc" if used_tensor_cores else "hc"), k_opt)]
          if used_tensor_cores:
            lins.append(("hc", Linearizer(*ast, opts=self.compiler.compiler_opts)))
            lins[-1][1].hand_coded_optimizations()
          timed = sorted([(nm, tk, time_linearizer(tk, rawbufs, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
          if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
          k = timed[0][1]
          if logkern is not None and logkern_level > 1: logkern.writelines([f"{(lin.ast, lin.applied_opts)}\n" for (_,lin,_) in timed[1:]])
    # TODO: check the correctness inline once compare_linearizer is in core
    if logkern is not None: logkern.writelines([f"{(k.ast, k.applied_opts)}\n"])
    if DEBUG >= 4: print((k.ast, k.applied_opts)) # print here to show final applied_opts for all kernels instead of just in beam_search
    return k

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, *ast:LazyOp) -> CompiledASTRunner: return self.to_program(self.get_linearizer(*ast))
