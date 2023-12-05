from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING, Union, Any, List, Optional, Dict, Callable
import importlib, inspect, functools, pathlib, time, re, ctypes
from tinygrad.helpers import ansilen, DEBUG, getenv, GlobalCounters, colored, BEAM, NOOPT, all_int, to_function_name, DType, from_mv, dtypes, flat_mv, ImageDType, round_up
from tinygrad.shape.symbolic import Variable, sym_infer, sint
from tinygrad.ops import LazyOp, TernaryOps, get_lazyop_info, ReduceOps, BufferOps, BinaryOps, UnaryOps, Op

if TYPE_CHECKING:
  from tinygrad.codegen.linearizer import Linearizer
  from tinygrad.codegen.kernel import LinearizerOptions

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  def __getitem__(self, ix:str) -> Union[Interpreted, Compiled]: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Union[Interpreted, Compiled]:
    x = ix.split(":")[0].upper()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._buffers][0]
    if isinstance(ret, type): ret = ret(ix)
    return ret
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** base Runner + helpers ****************

class JITRunner:
  def __init__(self):
    self.op_estimate, self.mem_estimate = 0, 0
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    var_vals = var_vals if var_vals is not None else {}
    from tinygrad.jit import CacheCollector
    et = self(rawbufs, var_vals)
    CacheCollector.add(self, rawbufs, var_vals)
    return et
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    raise NotImplementedError("override this")

def update_stats(name:str, op_estimate:sint, mem_estimate:sint, var_vals: Optional[Dict[Variable, int]], et: Optional[float], buf_count, jit=False, num_kernels=1, lra: Optional[Dict]=None):
  if var_vals is None: var_vals = {}
  op_estimate, mem_estimate = sym_infer(op_estimate, var_vals), sym_infer(mem_estimate, var_vals)
  GlobalCounters.kernel_count += num_kernels
  GlobalCounters.global_ops += op_estimate
  GlobalCounters.global_mem += mem_estimate
  if et is not None: GlobalCounters.time_sum_s += et
  if DEBUG >= 2:
    print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', ('magenta' if num_kernels == 1 else 'CYAN') if jit else None)} {name+' '*(37-ansilen(name))} arg {buf_count:3d} sz {str(lra.get('global_size', '') if lra else ''):18s} {str(lra.get('local_size', '') if lra else ''):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
          (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))

# **************** Buffer / Allocator ****************

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None):
    assert isinstance(dtype, DType)
    self.device, self.size, self.dtype = device, size, dtype
    self.allocator = Device[self.device].allocator
    # TODO: image hack shouldn't be here. where should it be?
    if isinstance(dtype, ImageDType) and hasattr(self.allocator, "_cast_image"):
      assert opaque is None
      row_pitch_items = round_up(dtype.shape[1], 256) * 4
      self.size = row_pitch_items * dtype.shape[0]  # adjust the size to include the image padding
      self._real_buf = self.allocator.alloc(self.size * dtype.itemsize)
      self._buf = self.allocator._cast_image(self._real_buf, dtype, row_pitch_items * dtype.itemsize)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(size * dtype.itemsize)
    # TODO: mem_used for all devices
    if self.device == Device.DEFAULT: GlobalCounters.mem_used += self.size * self.dtype.itemsize
  def __del__(self):
    if self.device == Device.DEFAULT: GlobalCounters.mem_used -= self.size * self.dtype.itemsize
    if isinstance(self.dtype, ImageDType):
      self.allocator._free(self._buf)
      self.allocator.free(self._real_buf, self.size * self.dtype.itemsize)
    else:
      self.allocator.free(self._buf, self.size * self.dtype.itemsize)
  def __repr__(self): return f"<buf device:{self.device} size:{self.size} dtype:{self.dtype}>"
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.size*self.dtype.itemsize, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyin(self._buf, mv)
    return self
  @staticmethod
  def fromCPU(device:str, x:np.ndarray): return Buffer(device, x.size, dtypes.from_np(x.dtype)).copyin(x.data)
  def toCPU(self) -> np.ndarray:
    # zero copy with as_buffer
    if hasattr(self.allocator, 'as_buffer'): return np.frombuffer(self.allocator.as_buffer(self._buf), dtype=np.dtype(self.dtype.np, metadata={"backing": self._buf}))  # type: ignore
    ret = np.empty(self.size, self.dtype.np)
    if self.size > 0: self.allocator.copyout(flat_mv(ret.data), self._buf)
    return ret

class _BufferCopy(JITRunner):
  # TODO: make wait work
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    dest, src = rawbufs
    assert dest.size == src.size and dest.dtype == src.dtype, "buffer copy size/dtype mismatch"
    if DEBUG >= 2: print(f"***      copy {dest.device} <- {src.device} size {dest.size:<16d} dtype {dest.dtype}")
    if hasattr(dest.allocator, 'transfer') and type(dest.allocator) is type(src.allocator):
      # fast path, used on HIP between GPUs
      # NOTE: it's important we use the dest device here to ensure the transfer is ready
      dest.allocator.transfer(dest._buf, src._buf, dest.size*dest.dtype.itemsize)
      return
    if getenv("FROM_BUFFER") and hasattr(dest.allocator, 'from_buffer') and hasattr(dest.allocator, 'transfer') and hasattr(src.allocator, 'as_buffer'):
      # fast path, used on Metal in OS X Sonoma
      # NOTE: this is *only* faster if the pages from disk are already loaded into memory
      fb = dest.allocator.from_buffer(src.allocator.as_buffer(src._buf))
      if fb:
        dest.allocator.transfer(dest._buf, fb, dest.size*dest.dtype.itemsize)
        return
    if hasattr(dest.allocator, 'as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator.copyout(dest.allocator.as_buffer(dest._buf), src._buf)
    elif hasattr(src.allocator, 'as_buffer'):
      dest.allocator.copyin(dest._buf, src.allocator.as_buffer(src._buf))
    else:
      # slow path, allocates a CPU buffer
      dest.copyin(src.toCPU().data)
BufferCopy = _BufferCopy()

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  def alloc(self, size:int):
    assert size > 0, f"alloc size must be positve, getting {size}"
    return self._alloc(size)
  def _alloc(self, size:int): raise NotImplementedError("need alloc")
  def free(self, opaque, size:int): self._free(opaque) # if you are returning a Python object, you don't need a free
  def _free(self, opaque): pass
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):  # pylint: disable=abstract-method
  def __init__(self): self.cache: Dict[int, Any] = defaultdict(list)
  def alloc(self, size:int):
    if len(c := self.cache[size]): return c.pop()
    try:
      return super().alloc(size)
    except MemoryError:
      self.free_cache()
      return super().alloc(size)
  def free_cache(self):
    for opaques in self.cache.values():
      for opaque in opaques: self._free(opaque)
      opaques.clear()
  def free(self, opaque:Any, size:int):
    if getenv("LRU", 1): self.cache[size].append(opaque)
    else: self._free(opaque)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
MallocAllocator = _MallocAllocator()

# **************** for Interpreted Devices ****************

class InterpretedASTRunner(JITRunner):
  def __init__(self, ast:LazyOp, fxn:Callable):
    super().__init__()
    self.fxn = fxn
    info = get_lazyop_info(ast)
    self.op_estimate, self.mem_estimate = info.flops, info.mem_estimate

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> float:
    st = time.perf_counter()
    rawbufs[0]._buf = self.fxn([x._buf for x in rawbufs], var_vals)
    et = time.perf_counter() - st
    update_stats(f"<interpreted {rawbufs[0].size}>", self.op_estimate, self.mem_estimate, var_vals, et, len(rawbufs), jit)
    return et

class Interpreted:
  def __init__(self, allocator: Allocator, fxn_for_op:Dict[Op, Callable]):
    self.allocator, self.fxn_for_op = allocator, fxn_for_op
    self.synchronize, self.codegen, self.graph = lambda: None, None, None

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp) -> InterpretedASTRunner: return _get_interpreted_fxn(self.fxn_for_op, ast)

def _get_interpreted_fxn(fxn_for_op:Dict[Op, Callable], ast:LazyOp) -> InterpretedASTRunner:
  if DEBUG >= 3:
    from tinygrad.graph import print_tree
    print_tree(ast)
  tglob: Dict[str, Any] = {"Variable": Variable}

  @functools.lru_cache(None)
  def gstr(x:Any, nm=None) -> str:
    if ('Variable' in (str_arg := repr(x)) or 'NumNode' in str_arg):
      str_arg = re.sub(r'Variable\(.*?\)', lambda m: f'var_vals[{str(m.group(0))}]', str_arg)
      # TODO: (Variable - Variable) might create NumNode. can we remove it?
      return re.sub(r'NumNode\((.*?)\)', r'\1', str_arg)
    ret = str(nm).replace(".", "_") if nm else f"m{len(tglob):04d}"
    tglob[ret] = x
    return ret

  lines: List[str] = []
  @functools.lru_cache(None)
  def _interpret_ast(ast:LazyOp) -> str:
    # TODO: shortcutted store won't work with strides
    if ast.op == BufferOps.STORE: return _interpret_ast(ast.src[0])
    if TernaryOps.MULACC in fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)

    if ast.op in BufferOps:
      if ast.op == ast.op == BufferOps.CONST: tmp = f"{gstr(fxn_for_op[ast.op], ast.op)}({gstr(ast.arg.val)}, {gstr(ast.arg.dtype)})"
      else: tmp = f"{gstr(fxn_for_op[UnaryOps.CAST], UnaryOps.CAST)}(inputs[{ast.arg.idx}], ({gstr(ast.arg.dtype)}, True))"
      for mop,arg in ast.arg.st.to_movement_ops(): tmp = f"{gstr(fxn_for_op[mop], mop)}({tmp}, {gstr(arg)})"
    else:
      tmp = f"{gstr(fxn_for_op[ast.op], ast.op)}({', '.join([_interpret_ast(src) for src in ast.src] + ([gstr(ast.arg)] if ast.arg else []))})"

    ret = f"a{len(lines)}"
    lines.append(f"  {ret} = {tmp}")
    return ret

  ret = _interpret_ast(ast)
  src = '\n'.join(['def run(inputs, var_vals):'] + lines + [f"  return {ret}"])
  if DEBUG >= 4: print(functools.reduce(lambda x,y: (x.replace(y[0], str(y[1])) if y[0][0:2] == "m0" else x), tglob.items(), src))
  exec(compile(src, "<ast>", "exec"), tglob) # pylint: disable=exec-used
  return InterpretedASTRunner(ast, tglob['run'])

# **************** for Compiled Devices ****************

class CompiledASTRunner(JITRunner):
  def __init__(self, ast:Optional[LazyOp], name:str, prg:str, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, runtime_args:Optional[dict]=None):
    super().__init__()
    if DEBUG >= 4: print(prg)
    if global_size is not None: global_size = global_size + [1]*(3-len(global_size))
    if local_size is not None: local_size = local_size + [1]*(3-len(local_size))
    self.name, self.display_name, self.prg, self.global_size, self.local_size, self.runtime_args = \
      to_function_name(name), name, prg, global_size, local_size, runtime_args if runtime_args is not None else {}
    self.vars: List[Variable] = []
    if ast:
      info = get_lazyop_info(ast)
      self.op_estimate, self.mem_estimate = info.flops, info.mem_estimate
      from tinygrad.lazy import vars_from_ast
      self.vars = vars_from_ast(ast)
      assert all(v._val is None for v in self.vars), f"ASTRunner contains bound Variable {self.vars}"

  def build(self, compiler, runtime):
    self.lib = compiler.__wrapped__(self.prg) if getenv("DISABLE_COMPILER_CACHE") else compiler(self.prg)
    self.clprg = runtime(self.name, self.lib)
    return self

  def launch_dims(self, var_vals):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else self.global_size
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else self.local_size
    return global_size, local_size

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    global_size, local_size = self.launch_dims(var_vals)
    if global_size is not None and local_size is None and all_int(self.global_size): # type: ignore[arg-type]
      # TODO: this is copied from get_program
      from tinygrad.features.search import optimize_local_size
      local_size = self.local_size = optimize_local_size(self.clprg, global_size, rawbufs)
      global_size = self.global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    lra = self.runtime_args.copy()
    if global_size: lra['global_size'] = global_size
    if local_size and 'local_size' not in lra: lra['local_size'] = local_size
    et = self.clprg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.vars), wait=wait or DEBUG>=2)
    update_stats(self.display_name, self.op_estimate, self.mem_estimate, var_vals, et, len(rawbufs), jit, lra=lra)
    return et

class Compiled:
  def __init__(self, allocator:Allocator, linearizer_opts:LinearizerOptions, renderer, compiler, runtime, graph=None):
    self.allocator, self.linearizer_opts, self.renderer, self.compiler, self.runtime, self.graph = allocator, linearizer_opts, renderer, compiler, runtime, graph
  def synchronize(self): pass  # override this in your device

  def to_program(self, k:Linearizer) -> CompiledASTRunner:
    k.linearize()
    src, runtime_args = self.renderer(to_function_name(k.name), k.uops)
    return CompiledASTRunner(k.ast, k.name, src, k.global_size, k.local_size, runtime_args).build(self.compiler, self.runtime)

  def get_linearizer(self, ast:LazyOp) -> Linearizer:
    if DEBUG >= 3:
      from tinygrad.graph import print_tree
      print_tree(ast)
    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(ast, self.linearizer_opts)
    if not NOOPT:
      if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
      if BEAM >= 1:
        lins = [(("tc" if used_tensor_cores else "hc"), k)]
        if used_tensor_cores:
          lins.append(("hc", Linearizer(ast, self.linearizer_opts)))
          lins[-1][1].hand_coded_optimizations()
        kb = Linearizer(ast, self.linearizer_opts)
        from tinygrad.features.search import beam_search, time_linearizer, bufs_from_lin
        # TODO: this shouldn't use Device.DEFAULT, it should get the device from the LinearizerOptions
        test_rawbuffers = bufs_from_lin(kb)    # allocate scratch buffers for optimization
        lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
        timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
        if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
        k = timed[0][1]
    return k

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp) -> CompiledASTRunner: return self.to_program(self.get_linearizer(ast))
