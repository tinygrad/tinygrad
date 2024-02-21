from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Dict, Tuple, ClassVar
import importlib, inspect, functools, pathlib, time, ctypes
from tinygrad.dtype import DType, ImageDType
from tinygrad.helpers import ansilen, DEBUG, getenv, colored, BEAM, NOOPT, all_int, to_function_name, from_mv, flat_mv, diskcache_get, diskcache_put
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable, sym_infer, sint
from tinygrad.ops import LazyOp, get_lazyop_info, GlobalCounters
from dataclasses import dataclass

if TYPE_CHECKING:
  from tinygrad.codegen.linearizer import Linearizer
  from tinygrad.codegen.kernel import LinearizerOptions

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
    x = ix.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._devices, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "HIP", "CUDA", "GPU", "LLVM", "CLANG"]:
      try:
        if self[device]: return device
      except Exception: pass
    raise RuntimeError("no usable devices")
Device = _Device()

# **************** base Runner + helpers ****************

class JITRunner:
  def __init__(self): self.op_estimate, self.mem_estimate = 0, 0
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    var_vals = var_vals if var_vals is not None else {}
    from tinygrad.features.jit import CacheCollector
    et = self(rawbufs, var_vals)
    CacheCollector.add(self, rawbufs, var_vals)
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

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  host: bool = False
  signal: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.d, self.options = device, size, dtype, Device[device], options
    self.allocator = self.d.allocator
    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, options)
    # TODO: mem_used for all devices
    if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf'): return # happens when __init__ has raised exception
    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
    self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self): return f"<buf device:{self.device} size:{self.size} dtype:{self.dtype}" + (">" if self.options is None else f"{self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.size*self.dtype.itemsize)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyout(mv, self._buf)
    return mv

class BufferCopy(JITRunner):
  def copy(self, dest, src): dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    et = None
    if wait or DEBUG >= 2:
      dest.d.synchronize()
      et = time.perf_counter() - st
    total_sz = dest.size*dest.dtype.itemsize
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    update_stats(colored(name, "yellow"), 0, total_sz, {}, et, 2, jit, device=dest.device)

class BufferRead(BufferCopy):
  def copy(self, dest, src):
    if hasattr(dest.allocator, 'copy_from_fd') and src.device.startswith("DISK") and src.nbytes >= 4096 and src._buf.ud.fd is not None:
      dest.allocator.copy_from_fd(dest._buf, src._buf.ud.fd, src._buf.offset, src.nbytes)
    elif hasattr(dest.allocator, 'as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator.copyout(dest.allocator.as_buffer(dest._buf), src._buf)
    else: super().copy(dest, src)

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
    return self._alloc_with_options(size, options) if options is not None else self._alloc(size)
  def _alloc(self, size:int): raise NotImplementedError("need alloc")
  def _alloc_with_options(self, size:int, options:BufferOptions): return self._alloc(size)  # TODO: override this if you support options
  def free(self, opaque, size:int, options:Optional[BufferOptions]=None): self._free(opaque)
  def _free(self, opaque): pass  # if opaque is a Python object, you don't need a free
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
    for opaques in self.cache.values():
      for opaque in opaques: self._free(opaque)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):
    if getenv("LRU", 1) and (options is None or not options.signal): self.cache[(size, options)].append(opaque)
    else: self._free(opaque)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
MallocAllocator = _MallocAllocator()

# **************** for Compiled Devices ****************

class Compiler:
  linearizer_opts: ClassVar[LinearizerOptions]
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def render(self, name:str, uops) -> str: raise NotImplementedError("need a render function")
  def compile(self, src:str) -> bytes: raise NotImplementedError("need a compile function")
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib

class CompiledASTRunner(JITRunner):
  def __init__(self, ast:Optional[LazyOp], name:str, prg:str, device:Compiled, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, precompiled:Optional[bytes]=None):  # noqa: E501
    super().__init__()
    if DEBUG >= 4: print(prg)
    if global_size is not None: global_size = global_size + [1]*(3-len(global_size))
    if local_size is not None: local_size = local_size + [1]*(3-len(local_size))
    self.name, self.display_name, self.prg, self.device, self.global_size, self.local_size, self.first_run = \
      to_function_name(name), name, prg, device, global_size, local_size, True
    assert self.device.compiler is not None, "compiler is reuired to make an AST kernel"
    lib:bytes = precompiled if precompiled is not None else self.device.compiler.compile_cached(prg)
    self.lib, self.clprg = lib, self.device.runtime(self.name, lib)
    self.vars: List[Variable] = []
    if ast:
      info = get_lazyop_info(ast)
      self.op_estimate, self.mem_estimate = info.flops, info.mem_estimate
      self.vars = ast.vars()
      assert all(v._val is None for v in self.vars), f"ASTRunner contains bound Variable {self.vars}"

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
                                     lra=lra, device=self.device.dname, first_run=self.first_run)
    self.first_run = False
    return et

class Compiled:
  def __init__(self, device:str, allocator:Allocator, compiler:Optional[Compiler], runtime, graph=None):
    self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler, runtime, graph
  def synchronize(self): pass  # override this in your device

  def to_program(self, k:Linearizer) -> CompiledASTRunner:
    assert self.compiler is not None, "compiler is required to run AST"
    k.linearize()
    ret = CompiledASTRunner(k.ast, k.name, self.compiler.render(to_function_name(k.name), k.uops), self, k.global_size, k.local_size)
    from tinygrad.codegen.uops import uops_flops_mem
    run_count = prod((k.global_size if k.global_size else []) + (k.local_size if k.local_size else []))
    ops, mem = uops_flops_mem(k.uops, {x.expr:x for x in ret.vars})
    # NOTE: we use min here to ignore the indexing FLOPS
    ret.op_estimate = min(ret.op_estimate, ops * run_count)
    ret.mem_estimate = min(ret.mem_estimate, mem * run_count)
    return ret

  def get_linearizer(self, ast:LazyOp) -> Linearizer:
    assert self.compiler is not None, "compiler is required to build AST"
    if DEBUG >= 3:
      from tinygrad.features.graph import print_tree
      print_tree(ast)
    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(ast, self.compiler.linearizer_opts)
    k.required_optimizations()
    if not NOOPT:
      if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
      if BEAM >= 1:
        lins = [(("tc" if used_tensor_cores else "hc"), k)]
        if used_tensor_cores:
          lins.append(("hc", Linearizer(ast, self.compiler.linearizer_opts)))
          lins[-1][1].hand_coded_optimizations()
        kb = Linearizer(ast, self.compiler.linearizer_opts)
        kb.required_optimizations()
        from tinygrad.features.search import beam_search, time_linearizer, bufs_from_lin
        test_rawbuffers = bufs_from_lin(kb)    # allocate scratch buffers for optimization
        lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
        timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
        if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
        k = timed[0][1]
    return k

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp) -> CompiledASTRunner: return self.to_program(self.get_linearizer(ast))
