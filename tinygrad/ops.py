from __future__ import annotations
import functools, time
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, cast
from tinygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored
from tinygrad.shape.shapetracker import MovementOps
from tinygrad.runtime.lib import RawBuffer, RawConst
if TYPE_CHECKING:
  from tinygrad.lazy import LazyBuffer

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); RECIP = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class FusedOps(Enum): MULACC = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[FusedOps]]

class LazyOp:
  # TODO: add dest to support multiple outputs. on second thought, multiple outputs will have multiple LazyOps.
  __slots__ = "op", "src", "arg", "buffers", "__weakref__"
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any
  buffers: Tuple[LazyBuffer, ...]

  def __init__(self, op: Op, src: Tuple[Union[LazyOp, LazyBuffer], ...], arg: Any = None):
    self.op = op
    self.src = src
    self.arg = arg
    try:
      self.buffers = tuple([y for x in src for y in x.buffers])
    except AttributeError:
      # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      pass

  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  def __eq__(self, __value: object) -> bool:
    if __value.__class__ is not LazyOp: return False
    __value = cast(LazyOp, __value)
    return self.op == __value.op and self.src == __value.src and self.arg == __value.arg
  def __hash__(self) -> int: return hash((self.op, self.src, self.arg))
  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  # Any == Union[LazyBuffer, DeviceBuffer]
  def map_buffers(self, real_srcs: Dict[Any, Any]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    from tinygrad.lazy import elementwise_op
    assert self.op in BinaryOps or self.op in UnaryOps
    return elementwise_op(self.op, *[z.replace_with_movement_ops(ops) for z in self.src], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError
  @property
  def shape(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def optype(self): raise NotImplementedError
  def realize(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

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
    if FusedOps.MULACC in self.fxn_for_op and ast.op == ReduceOps.SUM and ast.src[0].__class__ is LazyOp and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, cast(LazyOp, ast.src[0]).src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [self.exec_ast(cast(LazyOp, x), context=context, **kwargs) if x.__class__ is LazyOp else self.from_lazybuffer(x) for x in ast.src]
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
  def __init__(self, name, prg, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0, display_name:Optional[str]=None, runtime_args:Optional[dict]=None):
    if DEBUG >= 4 and (runtime_args is None or 'binary' not in runtime_args): print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.op_estimate, self.mem_estimate, self.display_name, self.runtime_args = name, prg, global_size, local_size, op_estimate, mem_estimate, display_name, runtime_args if runtime_args is not None else {}

  def build(self, runtime):
    self.clprg = runtime(self.name, self.prg, **self.runtime_args)
    return self

  def exec(self, bufs) -> Optional[float]:
    rawbufs = [x.realized for x in bufs if x.realized is not None and x.realized.__class__ is not RawConst]
    if GlobalCounters.cache is not None: GlobalCounters.cache.append((self, rawbufs))
    return self(rawbufs)

  def __call__(self, rawbufs:List[RawBuffer], jit=False, force_wait=False) -> Optional[float]:
    if et := self.clprg((self.global_size + [1]*(3-len(self.global_size))) if self.global_size is not None else None,
                        (self.local_size + [1]*(3-len(self.local_size))) if self.local_size is not None else None,
                        *rawbufs, wait=force_wait or DEBUG>=1): GlobalCounters.time_sum_s += et
    if DEBUG >= 2:
      print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(29-ansilen(self.display_name))) if self.display_name is not None else self.name:26s} arg {len(rawbufs):3d} sz {str(self.global_size):18s} {str(self.local_size):12s} OPs {int(self.op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({self.op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {self.mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += self.op_estimate
    GlobalCounters.global_mem += self.mem_estimate
    if getenv("EARLY_STOPPING") and GlobalCounters.kernel_count == getenv("EARLY_STOPPING"): exit(0)
    return et

class Compiled:
  def __init__(self, buffer: Type[RawBuffer], codegen, runtime, synchronize=lambda: None):
    self.buffer, self.codegen, self.runtime, self.synchronize = buffer, codegen, runtime, synchronize
    self.method_cache: Dict[str, ASTRunner] = {}

  def exec_ast(self, ast:LazyOp, output, **kwargs):
    # all movementops do nothing in a Compiled buffer!
    if ast.op in MovementOps and ast.src[0].__class__ is not LazyOp and ast.src[0].realized: return ast.src[0].realized

    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
    output.realized = output.output_buffer
    if output.realized:
      if output.realized.__class__ is RawConst: output.realized = None  # can't assign to RawConst
      for a in ast.buffers:
        if a.realized == output.realized and not a.st.contiguous:
          output.realized = None
          break

    # we don't have an output buffer, we have to create it
    if not output.realized:
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
