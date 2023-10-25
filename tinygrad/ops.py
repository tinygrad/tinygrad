from __future__ import annotations
import time, importlib, inspect, functools, pathlib, itertools, random, math, collections
import numpy as np
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, cast, Mapping
from tinygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored, BEAM
from tinygrad.runtime.lib import RawBuffer
from tinygrad.shape.symbolic import Variable, sym_infer
from dataclasses import dataclass

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): MEM = auto(); CONST = auto() # noqa: E702
# Ops below this line are not allowed in ASTs
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

if TYPE_CHECKING:
  from tinygrad.lazy import LazyBuffer
  from tinygrad.shape.shapetracker import ShapeTracker

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Any
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

@dataclass(frozen=True)
class LazyOp:
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any = None
  @property
  def buffers(self):
    buffers: Tuple[Union[LazyOp, LazyBuffer], ...] = ()
    try:  # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      for x in self.src: buffers += x.buffers
    except AttributeError: buffers = ()
    return buffers

  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  def map_buffers(self, real_srcs: Mapping[Any, Union[LazyBuffer, LazyOp]]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) if y not in real_srcs else real_srcs[y] for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    assert self.op in BinaryOps or self.op in UnaryOps or self.op in TernaryOps
    srcs = [z.replace_with_movement_ops(ops) for z in self.src]
    return srcs[0].e(self.op, *srcs[1:], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]:
    x = x.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None)
    if device_from_env: return device_from_env
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** for Interpreted Buffers ****************

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], to_underlying=lambda x: x._buf, from_underlying=None):
    self.buffer, self.fxn_for_op, self.to_underlying = buffer, fxn_for_op, to_underlying
    self.from_underlying = buffer if from_underlying is None else from_underlying
    self.synchronize = lambda: None
    self.codegen = None

  def exec_ast(self, ast:LazyOp, output=None, inputs=None, var_vals=None, context=None, **kwargs):
    if ast.op in BufferOps and ast.op not in self.fxn_for_op:
      if ast.op == BufferOps.MEM:
        assert inputs[ast.arg.idx-1].dtype == ast.arg.dtype, "dtype mismatch"
        buf = self.to_underlying(inputs[ast.arg.idx-1].realized)
      elif ast.op == BufferOps.CONST:
        buf = self.to_underlying(self.buffer.fromCPU(np.array(ast.arg.val, dtype=ast.arg.dtype.np)))
      for mop,arg in ast.arg.st.to_movement_ops(): buf = self.fxn_for_op[mop](buf, arg)
      return self.from_underlying(buf)
    if TernaryOps.MULACC in self.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [self.exec_ast(cast(LazyOp, x), inputs=inputs, context=context, **kwargs) for x in ast.src]
    if DEBUG >= 3: st = time.perf_counter()
    ret = self.from_underlying(self.fxn_for_op[ast.op](*([self.to_underlying(x) for x in srcs] + ([ast.arg] if ast.arg is not None else []))))
    if output is not None and ret.dtype != output.dtype and UnaryOps.CAST in self.fxn_for_op: ret = self.from_underlying(self.fxn_for_op[UnaryOps.CAST](self.to_underlying(ret), (output.dtype, False))) # Do manual casting of ret if it does not match the required output dtype.
    if DEBUG >= 5 or (self.buffer != FlopCounter and DEBUG >= 3): print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB {(time.perf_counter()-st)*1e3:7.2f} ms op: {ast.op:20s} out({ret.dtype.name}): {str(ret._buf.shape) if hasattr(ret._buf, 'shape') else str(len(ret._buf)):30s} in({len(srcs)}):", list(set(x._buf.shape if hasattr(x._buf, 'shape') else len(x._buf) for x in srcs)), ast.arg if ast.arg is not None else "")
    if not created_context: context[ast] = ret
    if output is not None and output.output_buffer is not None:
      # TODO: does this check have any meaning anymore?
      # It fails on things like batchnorm initted with zeros
      #assert output.output_buffer.size == ret.size, f"size mismatch, {output.output_buffer.size} != {ret.size}"
      assert output.output_buffer.dtype == ret.dtype
      output.output_buffer._buf = ret._buf
      return output.output_buffer
    return ret

class FlopCounter:
  def __init__(self, tup:Tuple[Tuple[int, ...], DType, int]): self.shape, self.dtype, self.flops, self._buf = *tup, self
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
shape_fxn_for_op: Dict[Op, Callable] = {
  BufferOps.MEM: lambda arg: (arg.st.shape, arg.dtype, 0), BufferOps.CONST: lambda arg: (arg.st.shape, arg.dtype, 0),
  UnaryOps.CAST: lambda self,arg: (self.shape, arg[0], self.consume_flops()),   # cast uses no flops
  **{op:lambda self: (self.shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in UnaryOps if op != UnaryOps.CAST},
  **{op:lambda self,y: (self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: (new_shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  TernaryOps.WHERE: lambda self,y,z: (self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape))}
InterpretedFlopCounter = Interpreted(FlopCounter, shape_fxn_for_op, lambda x: x)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return InterpretedFlopCounter.exec_ast(ast)

# **************** for Compiled Buffers ****************

class BasicBatchExecutor:
  def __init__(self, jit_cache:List[Tuple[Any, Any, Any]]): pass
  def exec(self, jit_cache: List[Tuple[Any, Any, Any]], updatable_entries):
    for prg, pargs, variables in jit_cache: prg(pargs, variables, jit=True)
  def recalc_stat(self, jit_cache: List[Tuple[Any, Any, Any]]):
    for prg, _, variables in jit_cache:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += sym_infer(prg.op_estimate, variables)
      GlobalCounters.global_mem += prg.mem_estimate

class GraphBatchExecutor(BasicBatchExecutor):
  def __init__(self, jit_cache:List[Tuple[Any, Any, Any]]): self.graphs: List[Any] = []
  def split_into_graphs(self, jit_cache:List[Tuple[Any, Any, Any]]):
    # Splitting the JIT cache into batches to enable parallel execution (cpu+gpu). Batch sizes follow a logarithmic pattern: 4, 4, 8, 16, 32, and so on.
    # This helps push tasks to the GPU while the CPU updates the next graph.
    for i in range(2, max(math.ceil(math.log2(len(jit_cache))), 2) + 1):
      self.create_graph(jit_cache[(1<<(i-1) if i > 2 else 0):(1<<i)])

  def exec(self, jit_cache: List[Tuple[Any, Any, Any]], updatable_entries):
    if not self.graphs: return super().exec(jit_cache, updatable_entries) # No graph is created, switch to basic executor.
    update_keys_per_batch = collections.defaultdict(list)
    for j in updatable_entries.keys(): update_keys_per_batch[max(0, math.ceil(math.log2(j+1))-2)].append(j)
    for instid in range(len(self.graphs)):
      for jcid in update_keys_per_batch[instid]: self.update_node(instid, jcid, jit_cache[jcid][0], jit_cache[jcid][1], jit_cache[jcid][2], updated_args=updatable_entries[jcid])
      self.exec_instance(instid)
    super().recalc_stat(jit_cache)

  def create_graph(self, jit_cache: List[Tuple[Any, Any, Any]]): raise NotImplementedError("must be implemented")
  def update_node(self, instid, jcid, prg, pargs, variables, updated_args=None): raise NotImplementedError("must be implemented")
  def exec_instance(self, instid): raise NotImplementedError("must be implemented")

class ASTRunner:
  def __init__(self, name, prg, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0, display_name:Optional[str]=None, runtime_args:Optional[dict]=None):
    if DEBUG >= 4 and (runtime_args is None or 'binary' not in runtime_args or not runtime_args['binary']): print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.op_estimate, self.mem_estimate, self.display_name, self.runtime_args = name, prg, global_size, local_size, op_estimate, mem_estimate, display_name, runtime_args if runtime_args is not None else {}

  def optimize_local_size(self, global_size, rawbufs) -> List[int]:
    assert self.global_size is not None, "needs a global size to optimize local size"
    MAX_WORKGROUP = self.clprg.max_work_group_size() if hasattr(self.clprg, 'max_work_group_size') else 1024
    local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
    local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
    def try_exec(local_size):
      try:
        return self.clprg([g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size, *rawbufs, wait=True)
      except Exception:
        return float('inf')
    return min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])[1]

  def build(self, runtime, batch_exec=BasicBatchExecutor):
    self.clprg, self.batch_exec = runtime(self.name, self.prg, **self.runtime_args), batch_exec
    return self

  def exec(self, rawbufs, var_vals:Optional[Dict[Variable, int]]=None, force_wait=False, optimizing=False) -> Optional[float]:
    from tinygrad.jit import CacheCollector
    if not optimizing: CacheCollector.add(self, rawbufs, var_vals if var_vals is not None else {})
    return self(rawbufs, var_vals, force_wait=force_wait)

  def launch_dims(self, var_vals):
    global_size = ([sym_infer(sz, var_vals) for sz in self.global_size] + [1]*(3-len(self.global_size))) if self.global_size is not None else self.global_size
    local_size = ([sym_infer(sz, var_vals) for sz in self.local_size] + [1]*(3-len(self.local_size))) if self.local_size is not None else self.local_size
    return global_size, local_size

  def __call__(self, rawbufs:List[RawBuffer], var_vals:Optional[Dict[Variable, int]]=None, jit=False, force_wait=False) -> Optional[float]:
    if var_vals is None: var_vals = {}
    global_size, local_size = self.launch_dims(var_vals)
    if global_size is not None and local_size is None:
      local_size = self.local_size = self.optimize_local_size(global_size, rawbufs)
      global_size = self.global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    if et := self.clprg(global_size, local_size, *rawbufs, *var_vals.values(), wait=force_wait or DEBUG>=2): GlobalCounters.time_sum_s += et
    op_estimate = sym_infer(self.op_estimate, var_vals)
    if DEBUG >= 2:
      print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(37-ansilen(self.display_name))) if self.display_name is not None else self.name:33s} arg {len(rawbufs):3d} sz {str(global_size):18s} {str(local_size):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {self.mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += self.mem_estimate
    return et

class Compiled:
  def __init__(self, buffer: Type[RawBuffer], linearizer_opts, renderer, runtime, synchronize=lambda: None, batch_exec=BasicBatchExecutor):
    self.buffer, self.linearizer_opts, self.renderer, self.runtime, self.synchronize, self.batch_exec = buffer, linearizer_opts, renderer, runtime, synchronize, batch_exec
    self.method_cache: Dict[LazyOp, ASTRunner] = {}

  def to_program(self, k):
    k.linearize()
    src, runtime_args = self.renderer(k.function_name, k.uops)
    return ASTRunner(k.function_name, src, k.global_size, k.local_size,
                     op_estimate=k.info.flops, mem_estimate=k.mem_estimate,
                     display_name=k.display_name, runtime_args=runtime_args).build(self.runtime, self.batch_exec)

  def exec_ast(self, ast:LazyOp, output, inputs, var_vals, **kwargs):
    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
    output.realized = output.output_buffer
    if output.realized:
      for i,a in enumerate(inputs):
        # TODO: if this is contiguous it's fine
        if a.realized == output.realized:
          if any(not x.arg.st.contiguous for x in ast.get_lazyops() if x.op == BufferOps.MEM and x.arg.idx == i+1):
            output.realized = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if not output.realized:
      output.realized = self.buffer(prod((s if isinstance(s, int) else s.max for s in output.shape)), output.dtype, **kwargs)
    else:
      from tinygrad.jit import CacheCollector
      CacheCollector._mark_output_buffer(output.output_buffer)

    # all the rawbuffers
    rawbuffers = [output.realized] + [x.realized for x in inputs]

    # extract real vars used in ast
    from tinygrad.lazy import vars_from_ast
    ast_vars = vars_from_ast(ast)
    assert all(v.val is None for v in ast_vars), f"ast contains bound Variable {ast_vars}"

    # compilation time
    def get_program():
      from tinygrad.codegen.linearizer import Linearizer
      k = Linearizer(ast, self.linearizer_opts)
      assert k.info.dtype == output.dtype, f"linearizer must match dtype. linearizer wants {k.info.dtype} but buffer is {output.dtype}"
      if not getenv("NOOPT"):
        if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
        if BEAM and not vars_from_ast(ast):
          kb = Linearizer(ast, self.linearizer_opts)
          kb.required_optimizations()
          kb.dont_use_locals = bool(getenv("NOLOCALS"))
          from tinygrad.features.search import beam_search, time_linearizer
          lins = [(f"beam{BEAM.value}", beam_search(kb, rawbuffers, BEAM.value)), (("tc" if used_tensor_cores else "hc"), k)]
          if used_tensor_cores:
            lins.append(("hc", Linearizer(ast, self.linearizer_opts)))
            lins[-1][1].hand_coded_optimizations()
          timed = sorted([(nm, tk, time_linearizer(tk, rawbuffers, allow_test_size=False, disable_cache=True)) for nm, tk in lins], key=lambda x: x[2])
          if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
          k = timed[0][1]
      return self.to_program(k)

    if getenv("ENABLE_METHOD_CACHE", 1):
      if ast not in self.method_cache: self.method_cache[ast] = get_program()
      prg = self.method_cache[ast]
    else:
      prg = get_program()

    if prg.name == getenv("PRINT_PRG", ''): print(prg.prg)

    prg.exec(rawbuffers, var_vals={k:var_vals[k] for k in ast_vars})
    return output.realized
