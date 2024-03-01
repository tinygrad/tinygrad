from __future__ import annotations
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Dict, Callable, ClassVar
import functools, hashlib
from enum import Enum, auto
from tinygrad.helpers import prod, dedup
from tinygrad.dtype import dtypes, DType
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum): EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum):
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPEQ = auto(); XOR = auto() # noqa: E702
class TernaryOps(Enum): WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); SYNC = auto(); WAIT = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.lazy import LazyBuffer

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Union[int, float]
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

@dataclass(frozen=True, eq=False)
class LazyOp:
  op: Op
  src: Tuple[LazyOp, ...] = ()
  arg: Any = None
  def cached_compare(self, x, context):
    if id(self) == id(x): return True
    if self.op != x.op or self.arg != x.arg or len(self.src) != len(x.src): return False
    if (key := (id(self), id(x))) in context: return context[key]
    ret = context[key] = all(a.cached_compare(b, context) for a,b in zip(self.src, x.src))
    return ret
  def __eq__(self, x): return self.cached_compare(x, context={})
  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(functools.reduce(lambda x,y: x+y, [s.key for s in self.src], str((self.op, self.arg)).encode())).digest()
  @functools.cached_property
  def hash(self): return hash((self.op, self.src, self.arg))
  def __hash__(self): return self.hash
  @functools.cached_property
  def lazyops(self) -> List[LazyOp]: return dedup([self] + [item for x in self.src for item in x.lazyops])
  def vars(self) -> List[Variable]:
    return sorted(set.union(*[x.arg.st.vars() for x in self.lazyops if x.op in BufferOps], set()), key=lambda x: str(x.expr))

# **************** independent FlopCounter ****************

@dataclass
class FlopCounter:
  shape: Tuple[int, ...]
  dtype: DType
  flops: int
  mem: Dict[int, int]
  @property
  def mem_estimate(self): return sum(self.mem.values())
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret

InterpretedFlopCounter: Dict[Op, Callable] = {
  BufferOps.LOAD: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {arg.idx: arg.dtype.itemsize*arg.st.real_size()}),
  BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {}),
  BufferOps.STORE: lambda self,arg: FlopCounter(arg.st.shape, arg.dtype, self.consume_flops(), {**self.mem, arg.idx: arg.dtype.itemsize*arg.st.real_size()}),  # noqa: E501
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, arg[0], self.consume_flops(), self.mem),   # cast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op != UnaryOps.CAST},  # noqa: E501
  **{op:lambda self,y,op=op: FlopCounter(self.shape,  dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else self.dtype, self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},  # noqa: E501
  **{op:lambda self,axis: FlopCounter(tuple(1 if i in axis else s for i,s in enumerate(self.shape)), self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},  # noqa: E501
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})}  # noqa: E501

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter:
  @functools.lru_cache(None) # NOTE: this cache needs to be recreated for new ASTs
  def run_ast(ast): return InterpretedFlopCounter[ast.op](*([run_ast(x) for x in ast.src]+([ast.arg] if ast.arg is not None else [])))
  return run_ast(ast)

# **************** global state Counters ****************

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0
