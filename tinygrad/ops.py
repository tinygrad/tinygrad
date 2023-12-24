from __future__ import annotations
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Dict, Callable
import functools
from enum import Enum
from tinygrad.helpers import dtypes, prod, DType, dedup
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV is on the chopping block
UnaryOps = Enum("UnaryOps", ['EXP2', 'LOG2', 'CAST', 'SIN', 'SQRT', 'RECIP', 'NEG'])
BinaryOps = Enum("BinaryOps", ['ADD', 'SUB', 'MUL', 'DIV', 'MAX', 'MOD', 'CMPLT', 'XOR'])
TernaryOps = Enum("TernaryOps", ['MULACC', 'WHERE'])
ReduceOps = Enum("ReduceOps", ['SUM', 'MAX'])
BufferOps = Enum("BufferOps", ['LOAD', 'CONST', 'STORE'])
# Ops below this line are not allowed in ASTs
MovementOps = Enum('MovementOps', ['RESHAPE', 'PERMUTE', 'EXPAND', 'PAD', 'SHRINK', 'STRIDE', 'AS_STRIDED'])
LoadOps = Enum('LoadOps', ['EMPTY', 'CONST', 'COPY', 'CONTIGUOUS', 'CUSTOM'])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

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
  def hash(self): return hash((self.op, self.src, self.arg))
  def __hash__(self): return self.hash
  @functools.cached_property
  def lazyops(self) -> List[LazyOp]: return dedup([self] + [item for x in self.src for item in x.lazyops])

def vars_from_ast(ast:LazyOp) -> List[Variable]:
  return sorted(set.union(*[x.arg.st.vars() for x in ast.lazyops if x.op in BufferOps], set()), key=lambda x: str(x.expr))

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
  BufferOps.LOAD: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {arg.idx: arg.dtype.itemsize*arg.st.size()}),
  BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {}),
  BufferOps.STORE: lambda self,arg: FlopCounter(arg.st.shape, arg.dtype, self.consume_flops(), {**self.mem, arg.idx: arg.dtype.itemsize*arg.st.size()}),  # noqa: E501
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, arg[0], self.consume_flops(), self.mem),   # cast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op != UnaryOps.CAST},  # noqa: E501
  **{op:lambda self,y,op=op: FlopCounter(self.shape,  dtypes.bool if op == BinaryOps.CMPLT else self.dtype, self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},  # noqa: E501
  **{op:lambda self,new_shape: FlopCounter(new_shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})}  # noqa: E501

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter:
  @functools.lru_cache(None) # NOTE: this cache needs to be recreated for new ASTs
  def run_ast(ast): return InterpretedFlopCounter[ast.op](*([run_ast(x) for x in ast.src]+([ast.arg] if ast.arg is not None else [])))
  return run_ast(ast)
