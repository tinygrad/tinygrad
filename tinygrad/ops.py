from __future__ import annotations
import functools, itertools, operator, random, time
from enum import Enum, auto
from typing import Union, Type, NamedTuple, Tuple, Any, List, Optional, Dict, Callable, ClassVar
from tinygrad.helpers import prod, DEBUG, getenv, GlobalCounters, DType, colored
from tinygrad.shape.shapetracker import MovementOps
from tinygrad.runtime.lib import RawBuffer, RawConst

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
class UnaryOps(Enum): NOOP = auto(); EXP = auto(); LOG = auto(); CAST = auto(); SIN = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class FusedOps(Enum): MULACC = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); FROMCPU = auto(); CONTIGUOUS = auto(); TOCPU = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[FusedOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs. on second thought, multiple outputs will have multiple LazyOps.

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])
def map_buffers(real_srcs:Dict[Any, Any], x:Any) -> LazyOp:
  if len(real_srcs) and x in real_srcs: return map_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple((map_buffers(real_srcs, y) if isinstance(y, LazyOp) else real_srcs[y]) for y in x.src), x.arg)

# **************** for Interpreted Buffers ****************

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], from_lazybuffer=lambda x: x.realized, to_underlying=lambda x: x._buf, from_underlying=None):
    self.buffer = buffer
    self.fxn_for_op = fxn_for_op
    self.from_lazybuffer = from_lazybuffer
    self.from_underlying = buffer if from_underlying is None else from_underlying
    self.to_underlying = to_underlying
    self.synchronize = lambda: None
    self.codegen = None

  def exec_ast(self, ast:LazyOp, output=None, context=None, **kwargs):
    if FusedOps.MULACC in self.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, ast.src[0].src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [self.exec_ast(x, context=context, **kwargs) if isinstance(x, LazyOp) else self.from_lazybuffer(x) for x in ast.src]
    if DEBUG >= 3: st = time.perf_counter()
    ret = self.from_underlying(self.fxn_for_op[ast.op](*([self.to_underlying(x) for x in srcs] + ([ast.arg] if ast.arg is not None else []))))
    if DEBUG >= 3: print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB {(time.perf_counter()-st)*1e3:7.2f} ms op: {ast.op:20s} out({ret.dtype.name}): {str(ret._buf.shape) if hasattr(ret._buf, 'shape') else str(len(ret._buf)):30s} in({len(srcs)}):", list(set(x._buf.shape if hasattr(x._buf, 'shape') else len(x._buf) for x in srcs)), ast.arg if ast.arg is not None else "")
    if not created_context: context[ast] = ret
    if output is not None and output.output_buffer is not None:
      assert output.output_buffer.size == ret.size, output.output_buffer.dtype == ret.dtype
      output.output_buffer._buf = ret._buf
      return output.output_buffer
    else:
      return ret

class FlopCounter:
  def __init__(self, tup:Tuple[Tuple[int, ...], DType, int]): self.shape, self.dtype, self.flops, self._buf = *tup, self
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
from tinygrad.shape.shapetracker import ShapeTracker
shape_fxn_for_op: Dict[Op, Callable] = {
  UnaryOps.CAST: lambda self,dtype: (self.shape, dtype, self.consume_flops()),   # cast uses no flops
  **{op:lambda self: (self.shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in UnaryOps if op != UnaryOps.CAST},
  **{op:lambda self,y: (self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: (new_shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  **{op:functools.partial(lambda mop,self,arg: (ShapeTracker(self.shape).movement_op(mop, arg).shape, self.dtype, self.consume_flops()), op) for op in MovementOps}}
InterpretedFlopCounter = Interpreted(FlopCounter, shape_fxn_for_op, lambda x: FlopCounter((x.shape, x.dtype, 0)), lambda x: x)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return InterpretedFlopCounter.exec_ast(ast)

# **************** for Compiled Buffers ****************

class ASTRunner:
  def __init__(self, name, prg, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0, display_name:Optional[str]=None, runtime_args={}):
    if DEBUG >= 4 and 'binary' not in runtime_args: print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.op_estimate, self.mem_estimate, self.display_name, self.runtime_args = name, prg, global_size, local_size, op_estimate, mem_estimate, display_name, runtime_args

  def build(self, runtime):
    self.clprg = runtime(self.name, self.prg, **self.runtime_args)
    return self

  def exec(self, bufs) -> Optional[float]:
    rawbufs = [x.realized for x in bufs if x.realized is not None and not isinstance(x.realized, RawConst)]
    if GlobalCounters.cache is not None: GlobalCounters.cache.append((self, rawbufs))
    return self(rawbufs)

  def __call__(self, rawbufs:List[RawBuffer], jit=False, force_wait=False) -> Optional[float]:
    if getenv("OPTLOCAL") and self.global_size is not None and self.local_size is None: self.local_size = self.optimize_local_size(rawbufs, allow_cache=(getenv("OPTLOCAL") >= 2))
    if et := self.clprg(self.global_size, self.local_size, *rawbufs, wait=force_wait or DEBUG>=1): GlobalCounters.time_sum_s += et
    if DEBUG >= 2:
      print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(29-len(self.name))) if self.display_name is not None else self.name:26s} arg {len(rawbufs):3d} sz {str(self.global_size):18s} {str(self.local_size):12s} OPs {int(self.op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({self.op_estimate/(et*1e9):8.2f} GFLOPS, {self.mem_estimate/(et*1e9):7.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += self.op_estimate
    GlobalCounters.global_mem += self.mem_estimate
    if getenv("EARLY_STOPPING") and GlobalCounters.kernel_count == getenv("EARLY_STOPPING"): exit(0)
    return et

  def timeit(self, rawbufs:List[RawBuffer], local_override=None) -> float:
    try: return self.clprg(self.global_size, local_override if local_override is not None else self.local_size, *rawbufs, wait=True)
    except Exception: return float('inf')

  optlocal_cache: ClassVar[Any] = None
  def optimize_local_size(self, rawbufs:List[RawBuffer], preserve_output=False, allow_cache=False) -> List[int]:
    assert self.global_size is not None, "needs a global size to optimize local size"
    if allow_cache:
      import dbm, pickle
      if ASTRunner.optlocal_cache is None: ASTRunner.optlocal_cache = dbm.open('/tmp/optlocal.db', 'c')
      if self.prg not in ASTRunner.optlocal_cache: ASTRunner.optlocal_cache[self.prg] = pickle.dumps(self.optimize_local_size(rawbufs, preserve_output, allow_cache=False)) # pylint: disable=unsupported-membership-test,unsupported-assignment-operation
      return pickle.loads(ASTRunner.optlocal_cache[self.prg])
    if preserve_output or any(x == rawbufs[0] for x in rawbufs[1:]):  # this is an assignment, replace the output buffer
      output_replacement = type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype)
      rawbufs = [output_replacement if x == rawbufs[0] else x for x in rawbufs]
    MAX_WORKGROUP = self.clprg.max_work_group_size() if hasattr(self.clprg, 'max_work_group_size') else 1024
    local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in self.global_size]
    local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
    return min([(self.timeit(rawbufs, local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])[1]

class Compiled:
  def __init__(self, buffer: Type[RawBuffer], codegen, runtime, synchronize=lambda: None):
    self.buffer, self.codegen, self.runtime, self.synchronize = buffer, codegen, runtime, synchronize
    self.method_cache: Dict[str, ASTRunner] = {}

  def exec_ast(self, ast:LazyOp, output, **kwargs):
    # all movementops do nothing in a Compiled buffer!
    if ast.op in MovementOps and not isinstance(ast.src[0], LazyOp) and ast.src[0].realized is not None: return ast.src[0].realized

    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
    output.realized = output.output_buffer
    if output.realized is not None:
      if isinstance(output.realized, RawConst): output.realized = None  # can't assign to RawConst
      for a in get_buffers(ast):
        if a.realized == output.realized and not a.st.contiguous:
          output.realized = None
          break

    # we don't have an output buffer, we have to create it
    if output.realized is None:
      output.realized = self.buffer(prod(output.shape), output.dtype, **kwargs)

    # compilation time
    k = self.codegen(ast, output)

    # this is the default now
    if hasattr(k, 'key') and getenv("ENABLE_METHOD_CACHE", 1):
      if k.key not in self.method_cache: self.method_cache[k.key] = k.codegen().build(self.runtime)
      elif DEBUG >= 5: print(f"method cache hit : {k.key}")
      prg = self.method_cache[k.key]
    else:
      prg = k.codegen().build(self.runtime)

    if prg.name == getenv("PRINT_PRG", ''): print(prg.prg)

    prg.exec(k.bufs)
    return output.realized
