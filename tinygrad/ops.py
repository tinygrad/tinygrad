from __future__ import annotations
from typing import Any, List, Optional, Set, Union, Tuple, Dict, Callable, cast, TYPE_CHECKING, TypeVar, DefaultDict
import sys, time, functools, itertools, math, operator, ctypes, struct, hashlib
from enum import auto, IntEnum, Enum
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes, DType
from tinygrad.helpers import pretty_print, prod, getenv, all_same
from tinygrad.shape.symbolic import Variable, sint
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(FastEnum):
  """A -> A (elementwise)"""
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto() # noqa: E702
class BinaryOps(FastEnum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto() # noqa: E702
class TernaryOps(FastEnum):
  """A + A + A -> A (elementwise)"""
  WHERE = auto(); MULACC = auto() # noqa: E702
class ReduceOps(FastEnum):
  """A -> B (reduce)"""
  SUM = auto(); PROD = auto(); MAX = auto() # noqa: E702
class MetaOps(FastEnum):
  EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto(); VIEW = auto() # noqa: E702
Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps]

T = TypeVar("T")
class MathTrait:
  # required to implement
  def alu(self:T, arg:Union[UnaryOps, BinaryOps, TernaryOps], *src) -> T: raise NotImplementedError
  def const_like(self, b:ConstType|Variable|Tuple[ConstType]): raise NotImplementedError

  # great functions you get!
  def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
  def __neg__(self):
    dtype = getattr(self, 'dtype', None)
    assert dtype is not None, "MathTraits __neg__ requires a dtype"
    return self.ne(True) if dtype.scalar() == dtypes.bool else self*(-1)
  def __add__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
  def __radd__(self, x): return self.ufix(x).alu(BinaryOps.ADD, self)
  def __sub__(self, x): return self.alu(BinaryOps.ADD, self.ufix(-x))
  def __rsub__(self, x): return self.ufix(x).alu(BinaryOps.ADD, -self)
  def __mul__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x))
  def __rmul__(self, x): return self.ufix(x).alu(BinaryOps.MUL, self)
  def __floordiv__(self, x): return self.alu(BinaryOps.IDIV, self.ufix(x))
  def __truediv__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x).alu(UnaryOps.RECIP))
  def __mod__(self, x): return self.alu(BinaryOps.MOD, self.ufix(x))
  def __xor__(self, x): return self.alu(BinaryOps.XOR, self.ufix(x))
  def __and__(self, x): return self.alu(BinaryOps.AND, self.ufix(x))
  def __or__(self, x): return self.alu(BinaryOps.OR, self.ufix(x))
  def ne(self, x): return self.alu(BinaryOps.CMPNE, self.ufix(x))
  def eq(self, x): return self.ne(x).ne(True)
  def lt(self, x): return self.alu(BinaryOps.CMPLT, self.ufix(x))
  def gt(self, x): return self.ufix(x).alu(BinaryOps.CMPLT, self)
  def ge(self, x): return (-self).lt(-x+1)
  def max(self, x): return self.alu(BinaryOps.MAX, self.ufix(x))
  def min(self, x): return -(-self).max(-x)
  def where(self, x, y): return self.alu(TernaryOps.WHERE, x, y)
  def threefry(self, seed): return self.alu(BinaryOps.THREEFRY, seed)
  def recip(self): return self.alu(UnaryOps.RECIP)
  def sqrt(self): return self.alu(UnaryOps.SQRT)
  def sin(self): return self.alu(UnaryOps.SIN)
  def log2(self): return self.alu(UnaryOps.LOG2)
  def exp2(self): return self.alu(UnaryOps.EXP2)

# do not preserve f(0) = 0
UNSAFE_PAD_OPS = {UnaryOps.RECIP, UnaryOps.LOG2, UnaryOps.EXP2, BinaryOps.IDIV}

REDUCE_ALU: Dict[ReduceOps, BinaryOps] = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.PROD:BinaryOps.MUL, ReduceOps.MAX:BinaryOps.MAX}

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:BinaryOps, dt:DType): return dtypes.as_const({BinaryOps.ADD:0, BinaryOps.MUL:1, BinaryOps.MAX:dtypes.min(dt)}[op], dt)

# the order of these UOps controls the order of the toposort
class UOps(FastEnum):
  # uops that aren't rendered
  SINK = auto()
  """
  Holds `UOps.STORE`. SINK defines the AST for a Kernel.

  - **`dtype`**: `dtypes.void`
  - **`src`**: `Tuple[UOp, ...]`, Only global STOREs are allowed.
  - **`arg`**: `Optional[KernelInfo]`

  NOTE: `ScheduleItem` ASTs do not have the `KernelInfo` arg, `Kernel` inserts this to the SINK later.
  """
  EXT = auto()
  """
  Holds a single MetaOp. EXT UOps do not need a Kernel.

  - **`dtype`**: Output DType
  - **`src`**: `Tuple[]`
  - **`arg`**: (`MetaOps.CUSTOM | MetaOps.COPY | MetaOps.EMPTY | MetaOps.VIEW`, LazyBuffer arg)
  """
  EXPAND = auto()
  CONTRACT = auto()
  SHAPETRACKER = auto()
  """
  Defines the ShapeTracker for a buffer UOp `UOps.LOAD`, `UOps.STORE` or `UOps.VALID`.

  - **`dtype`**: `dtypes.void`
  - **`src`**: `Tuple[]`
  - **`arg`**: `ShapeTracker`
  """
  SWIZZLE = auto()
  """
  Swizzle inserts a movement op between a UOp and its children. Because movement ops (reshape, expand, shrink, permute, pad) are not allowed in an AST,
  the scheduler rewrites SWIZZLE by pushing its ShapeTracker through reduceops or elementwise ops to the edges of the graph.

  This movement op can push up to the LOADs and/or down to the STOREs.

  Example:
  ```python
  a = Tensor.empty(32, 32)
  first_reduce = a.sum()
  output = (a + first_reduce).sum()
  ```
  `first_reduce` must broadcast to `(32, 32)` before ADD. We UOp this as:

  ```
  UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
    UOp(UOps.SWIZZLE, dtypes.int, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=(
      UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (0, 1)), src=(
        UOp(UOps.LOAD, dtypes.int, arg=None, src=(
          x3:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
          UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),
    UOp(UOps.LOAD, dtypes.int, arg=None, src=(
       x3,
      UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),))
  ```

  The scheduler rewrites this by pushing the expand in SWIZZLE through the reduce, to the LOAD:

  ```diff
  UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
  -   UOp(UOps.SWIZZLE, dtypes.int, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=(
  -     UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (0, 1)), src=(
  -       UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  -         x3:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
  -         UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),
  +   UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (2, 3)), src=(
  +     UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  +       x2:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
  +       UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32, 32, 32), strides=(0, 0, 32, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
    UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  -      x3,
  -     UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),))
  +      x2,
  +     UOp(UOps.SHAPETRACKER, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),))

  ```

  NOTE: Pushing a SWIZZLE through a reduce changes the axis.

  NOTE: Pushing a SWIZZLE changes the output shape of that UOp. We have to reshape every other adjacent node. eg. reshape of the second LOAD to `(32, 32, 1, 1)` above.

  - **`dtype`**: Output DType
  - **`src`**: `Tuple[UOp]`, a single UOp to swizzle.
  - **`arg`**: ShapeTracker
  """ # noqa E501
  DEFINE_GLOBAL = auto()
  DEFINE_VAR = auto()
  DEFINE_LOCAL = auto()
  DEFINE_ACC = auto()
  VCONST = auto()
  CONST = auto()
  """
  Defines a single scalar constant value.

  - **`dtype`**: The scalar DType of the value.

  - **`src`**: `Tuple[]`

  - **`arg`**: The value.
  """
  VALID = auto()
  """
  This is the first argument in a masked CONST.

  - **`dtype`**: `dtypes.bool`
  - **`src`**:
    `Tuple[UOp]`
      - UOps.SHAPETRACKER
  - **`arg`**: `None`

  A masked CONST is defined as `valid.where(value, 0)`.
  """
  SPECIAL = auto()
  NOOP = auto()
  GEP = auto()
  # math ops
  CAST = auto()
  """
  - **`dtype`**: The casted scalar DType
  - **`src`**: `Tuple[UOp]`
  - **`arg`**: `None`
  """
  BITCAST = auto()
  """
  - **`dtype`**: The bitcasted scalar DType
  - **`src`**: `Tuple[UOp]`
  - **`arg`**: `None`
  """
  VECTORIZE = auto()
  """
  - **`dtype`**: The upcasted vector DType
  - **`src`**: `Tuple[UOp, ...]`
  - **`arg`**: `None`

  NOTE: Length of sources must match `dtype.count`
  """
  ALU = auto()
  """
  - **`dtype`**: Output DType
  - **`src`**: `Tuple[UOp] | Tuple[UOp, UOp] | Tuple[UOp, UOp, UOp]`
  - **`arg`**: `UnaryOps | BinaryOps | TernaryOps`
  """
  REDUCE = auto()
  REDUCE_AXIS = auto()
  """
  - **`dtype`**: Output DType
  - **`src`**: Input to reduce `Tuple[UOp]`
  - **`arg`**: `(BinaryOps.ADD | BinaryOps.MUL | BinaryOps.MAX, Tuple[int, ...])`
  """
  WMMA = auto()
  # memory/assignment ops
  LOAD = auto()
  """
  - **`dtype`**: Output DType
  - **`src`**:

    The scheduler and Kernel create LOADs with a SHAPETRACKER uop in src.

    - Normal LOAD: `Tuple[UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL`.
      - SHAPETRACKER UOp.

    - Local LOAD: `Tuple[UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_LOCAL`.
      - SHAPETRACKER UOp.
      - Local UOps.STORE to the same local buffer. We will barrier this later.

    The Lowerer replaces the SHAPETRACKER with an indexing uop and gates the LOAD if needed.

    - Normal LOAD: `Tuple[UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL`.
      - Indexing UOp, can only return `dtypes.int32`.
    - Gated LOAD: `Tuple[UOp, UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL`.
      - Indexing UOp, can only return `dtypes.int32`.
      - Gate UOp, can only return `dtypes.bool`.
      - Value if gate is `False`, can only be a `UOps.CONST` with arg 0, 0.0 or `False`.
    - Barriered LOAD: `Tuple[UOp, UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_LOCAL`.
      - Indexing UOp, can only return `dtypes.int32`.
      - Gate UOp, can only return `dtypes.bool`.
      - Barrier UOp `UOps.BARRIER`.
  - **`arg`**: `None`
  """
  STORE = auto()
  """
  - **`dtype`**: `dtypes.void`
  - **`src`**:

    Similar to LOAD, the scheduler and Kernel create STOREs with a SHAPETRACKER uop in src:

    - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
    - SHAPETRACKER UOp.
    - Value to store.

    The Lowerer replaces the SHAPETRACKER with an indexing uop and gates the STORE if needed.

    - Normal STORE: `Tuple[UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
      - Indexing UOp, can only return `dtypes.int32`.
      - Value to store.
    - Gated STORE: `Tuple[UOp, UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
      - Indexing UOp, can only return `dtypes.int32`.
      - Value to store.
      - Gate UOp, can only return `dtypes.bool`. We rewrite this to an IF block in the end.
  - **`arg`**: `None`
  """
  ASSIGN = auto()
  # control flow ops
  BARRIER = auto()
  """
  Inserts a warp sync between local stores and local loads.

  - **`dtype`**: `dtypes.void`
  - **`src`**: `Tuple[UOp, ...]`, Only local STOREs are allowed.
  - **`arg`**: `None`
  """
  IF = auto()
  """
  Gates a single STORE to global memory. The IF block could also contain additional UOps the STORE depends on.

  - **`dtype`**: `dtypes.void`
  - **`src`**:
    `Tuple[UOp, UOp]`
      - Gate UOp, can only return `dtypes.bool`
      - The second UOp starts the gate block; All of its children are gated until the final STORE.
  - **`arg`**: `None`

  For example, a local reduce must only run on one thread.

  The STORE's IF gate:
  ```
  UOp(UOps.IF, src=(
    UOp(UOps.ALU, dtypes.bool, (...), BinaryOps.CMPNE),
    UOp(UOps.BARRIER, dtypes.void, (...))))
  ```
  The kernel:
  ```
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidx0!=1) {
    int acc1 = 0;
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      int val1 = temp1[ridx1];
      acc1 = (acc1+val1);
    }
    data0[0] = acc1;
  }
  ```
  """
  RANGE = auto()
  # ops that are not graph nodes
  ENDRANGE = auto()
  ENDIF = auto()

BUFFER_UOPS = {UOps.LOAD, UOps.STORE, UOps.CONST}
COMMUTATIVE = {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR}
END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.ASSIGN, UOps.ENDRANGE)}

class UOp(MathTrait):
  __slots__ = ["op", "dtype", "src", "arg"]
  def __init__(self, op: UOps, dtype:DType=dtypes.void, src: Tuple[UOp,...]=tuple(), arg:Any=None):
    # TODO: instant check rules here make debugging easier
    #if op is UOps.ALU and arg is BinaryOps.CMPNE: assert dtype.scalar() == dtypes.bool
    #if op is UOps.VECTORIZE and dtype != dtypes.void: assert len(src) == dtype.count, f"{len(src)} invalid for {dtype}"
    #if op is UOps.ALU and arg not in (BinaryOps.CMPNE, BinaryOps.CMPLT, TernaryOps.WHERE): assert all_same([dtype] + [x.dtype for x in src])
    #if op is UOps.CAST: assert dtype.count == src[0].dtype.count, f"cast can't change vectorization {src[0].dtype} --> {dtype}"
    self.op, self.dtype, self.src, self.arg = op, dtype, src, arg
  def replace(self, op: Optional[UOps]=None, dtype:Optional[DType]=None, src: Optional[Tuple[UOp,...]]=None, arg:Any=None):
    return UOp(op or self.op, dtype or self.dtype, self.src if src is None else src, self.arg if arg is None else arg)
  @functools.cached_property
  def st(self) -> Optional[ShapeTracker]:
    from tinygrad.shape.shapetracker import ShapeTracker
    if self.op in {UOps.DEFINE_LOCAL, UOps.DEFINE_GLOBAL}: return None
    if self.op in BUFFER_UOPS: return self.st_arg
    if self.op in {UOps.SHAPETRACKER, UOps.SWIZZLE}: return self.arg
    src_sts = [x.st for x in self.src if x.st is not None]
    assert all_same([x.shape for x in src_sts]), f"UOp parents must have the same shape {self} {[x.shape for x in src_sts]}"
    return ShapeTracker.from_shape(src_sts[0].reduce(self.arg[1])) if self.op is UOps.REDUCE_AXIS else src_sts[0]
  @functools.cached_property
  def cmp_tuple(self) -> Tuple[int, Any, Optional[DType], Tuple[UOp, ...]]:
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    if self.op is UOps.DEFINE_VAR: arg = self.arg[0]
    elif self.op is UOps.ALU: arg = self.arg.value
    else: arg = self.arg
    return (self.op.value, arg, self.dtype, self.src)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self):
    return f'({", ".join(map(str, self.arg))})' if self.op is UOps.REDUCE_AXIS else repr(self.arg) if isinstance(self.arg, Variable) else self.arg
  # *** uop syntactic sugar
  @property
  def st_loc(self) -> int: return 0 if self.op is UOps.CONST else 1
  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in BUFFER_UOPS, f"st_arg called on {self.op}"
    ret = self.src[self.st_loc]
    assert ret.op is UOps.SHAPETRACKER, f"st_arg trying to return {ret}"
    return ret.arg
  def sink(self, *srcs): return UOp(UOps.SINK, dtypes.void, (self,)+srcs)
  def swizzle(self, st:ShapeTracker): return UOp(UOps.SWIZZLE, self.dtype, (self,), st)
  def const_like(self, b:ConstType|Variable|Tuple[ConstType]): return type(self).const(self.dtype, b)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(UOps.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType): return type(self)(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return type(self)(UOps.BITCAST, dtype, (self,))
  def gep(self, i:Union[Tuple[int, ...], int]):
    if isinstance(i, int): i = (i,)
    if self.dtype == dtypes.void or (i == tuple(range(len(i))) and self.dtype.count == len(i)): return self
    assert len(i) >= 1 and all(x < self.dtype.count for x in i), f"bad GEP on {self.dtype}, {i}"
    return UOp(UOps.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  @classmethod
  def load(cls, *src:UOp, dtype:DType): return cls(UOps.LOAD, dtype, src)
  @classmethod
  def store(cls, *src:UOp): return cls(UOps.STORE, dtypes.void, src)
  def alu(self, arg, *src:UOp):
    out_dtype = (self, *src)[-1].dtype
    if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} and out_dtype is not None:
      out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return type(self)(UOps.ALU, out_dtype, (self,)+src, arg)
  @classmethod
  @functools.lru_cache(None)
  def const(cls, dtype:DType, b:Tuple[ConstType, ...]|ConstType|Variable): return cls._const(dtype, b)
  @classmethod
  def _const(cls, dtype:DType, b:Tuple[ConstType, ...]|ConstType|Variable):
    # TODO: fix dtype of b.max after Variable is just an UOp
    if isinstance(b, Variable): return cls.define_var(b.expr, dtype, b.min, cast(int, b.max))
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    return cls(UOps.VCONST if isinstance(b, tuple) else UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b) # type: ignore
  @staticmethod
  def define_var(name:str, dtype:DType, min_val:ConstType, max_val:ConstType):
    return UOp(UOps.DEFINE_VAR, dtype, arg=(name, UOp.const(dtype, min_val), UOp.const(dtype, max_val)))
  @staticmethod
  def range(dtype:DType, start:ConstType, end:ConstType, idx:int):
    return UOp(UOps.RANGE, dtype=dtype, src=(UOp.const(dtype, start), UOp.const(dtype, end)), arg=(idx,))
  def reduce(self, op, *rng): return UOp(UOps.REDUCE, self.dtype, (self,) + rng, op)
  @functools.cached_property
  def parents(self) -> Dict[UOp, None]: return {**{x:None for x in self.src}, **{k:None for x in self.src for k in x.parents.keys()}}
  @property  # parents with self
  def sparents(self) -> Dict[UOp, None]: return {**self.parents, self:None}
  @functools.cached_property
  def full_shape(self) -> Tuple[sint, ...]:
    if self.op is UOps.SHAPETRACKER: return self.arg.shape
    # NOTE: UOps.DEFINE_GLOBAL and UOps.DEFINE_LOCAL don't have shape
    return tuple(max(x) for x in zip(*[x.full_shape for x in self.src if x.op not in {UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL}]))
  def vars(self) -> Set[UOp]: return set([x for x in self.sparents if x.op is UOps.DEFINE_VAR])
  def variables(self) -> List[Variable]:
    st_vars: List[Set[Variable]] = [x.st_arg.vars() for x in self.sparents if x.op in BUFFER_UOPS]
    return sorted(set.union(*st_vars, [Variable(x.arg[0], x.arg[1].arg, x.arg[2].arg) for x in self.vars()]), key=lambda v: v.expr)
  def const_factor(self) -> int:
    """largest known int that divides self"""
    if self.op is UOps.CONST: return self.arg
    if self.op is UOps.VCONST: return functools.reduce(math.gcd, self.arg)
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
      if self.arg is BinaryOps.MUL: return self.src[0].arg if self.src[0].op is UOps.CONST else self.src[1].arg if self.src[1].op is UOps.CONST else 1
    return 1
  def divides(self, v) -> Optional[UOp]:
    if v==1: return self
    if self.op is UOps.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
      if self.arg is BinaryOps.MUL:
        if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
        if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> ConstType: return self._min_max[0]
  @property
  def vmax(self) -> ConstType: return self._min_max[1]
  @functools.cached_property
  def _min_max(self) -> Tuple[ConstType, ConstType]:
    # NOTE: returned UOp is assumed to be CONST
    if self.op is UOps.DEFINE_VAR and self.arg:
      return self.arg[1].arg, self.arg[2].arg if self.arg[2].op is UOps.CONST else dtypes.max(self.dtype)
    if self.op is UOps.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    if self.op is UOps.EXPAND: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    # TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if self.op is UOps.SPECIAL: return 0, self.arg[1]-1 if isinstance(self.arg[1], int) else dtypes.max(self.dtype)
    if self.op is UOps.CONST: return self.arg, self.arg
    if self.op is UOps.VCONST: return (min(self.arg), max(self.arg))
    if self.op is UOps.ALU and self.dtype.count == 1:
      s0,s1 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(2)]
      if self.arg is BinaryOps.ADD: return s0.vmin+s1.vmin, s0.vmax+s1.vmax
      if self.arg is BinaryOps.MUL:
        # both are non-positive
        if (s0.vmax <= 0 and s1.vmax <= 0): return s0.vmax*s1.vmax, s0.vmin*s1.vmin
        # at lease one is non-negative
        if (s0.vmin >= 0 or s1.vmin >= 0):
          Lmin, Lmax = (s0.vmin, s0.vmax) if s1.vmin >= 0 else (s0.vmax, s0.vmin)
          Rmin, Rmax = (s1.vmin, s1.vmax) if s0.vmin >= 0 else (s1.vmax, s1.vmin)
          return Lmin*Rmin, Lmax*Rmax
      if self.arg is BinaryOps.MOD and s1.vmin > 0: return 0, s1.vmax-1
      if self.arg is BinaryOps.IDIV and s1.op is UOps.CONST:
        if s1.arg > 0: return s0.vmin//s1.arg, s0.vmax//s1.arg
        if s1.arg < 0: return -(s0.vmax//-s1.arg), -(s0.vmin//-s1.arg)
      if self.arg is BinaryOps.MAX: return max(s0.vmin, s1.vmin), max(s0.vmax, s1.vmax)
      if self.arg is BinaryOps.CMPLT: return (s0.vmax<s1.vmin, s0.vmin<s1.vmax)
    return dtypes.min(self.dtype), dtypes.max(self.dtype)

@dataclass(frozen=True)
class KernelInfo:
  local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)
  upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to EXPAND)
  dont_use_locals: bool = False # don't use local indexing

# ***** ops in python *****

def hook_overflow(dv, fxn):
  def wfxn(*args):
    try: return fxn(*args)
    except OverflowError: return dv
  return wfxn

python_alu: Dict[Op, Callable]  = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, UnaryOps.EXP2: hook_overflow(math.inf, lambda x: 2**x),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  UnaryOps.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan,
  BinaryOps.SHR: operator.rshift, BinaryOps.SHL: operator.lshift, BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add,
  BinaryOps.XOR: operator.xor, BinaryOps.MAX: max, BinaryOps.CMPNE: operator.ne, BinaryOps.CMPLT: operator.lt,
  BinaryOps.OR: operator.or_, BinaryOps.AND: operator.and_,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], BinaryOps.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else x*math.inf,
  TernaryOps.MULACC: lambda x,y,z: (x*y)+z, TernaryOps.WHERE: lambda x,y,z: y if x else z}

def truncate_fp16(x):
  try: return struct.unpack("@e", struct.pack("@e", float(x)))[0]
  except OverflowError: return math.copysign(math.inf, x)

truncate: Dict[DType, Callable] = {dtypes.bool: bool,
  # TODO: bfloat16
  dtypes.float16: truncate_fp16, dtypes.float32: lambda x: ctypes.c_float(x).value, dtypes.float64: lambda x: ctypes.c_double(x).value,
  dtypes.uint8: lambda x: ctypes.c_uint8(x).value, dtypes.uint16: lambda x: ctypes.c_uint16(x).value,
  dtypes.uint32: lambda x: ctypes.c_uint32(x).value, dtypes.uint64: lambda x: ctypes.c_uint64(x).value,
  dtypes.int8: lambda x: ctypes.c_int8(x).value, dtypes.int16: lambda x: ctypes.c_int16(x).value, dtypes.int32: lambda x: ctypes.c_int32(x).value \
      if isinstance(x,int) else x, dtypes.int64: lambda x: ctypes.c_int64(x).value}

def exec_alu(op:Op, dtype:DType, operands):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  return truncate.get(dtype, lambda x: x)(python_alu[op](*operands))

def uop_alu_resolve(u:UOp) -> sint:
  if u.op is UOps.CONST: return u.arg
  if u.op is UOps.DEFINE_VAR: return Variable(u.arg[0], u.arg[1].arg, u.arg[2].arg)
  if u.op is UOps.ALU: return exec_alu(u.arg, u.dtype, tuple(map(uop_alu_resolve, u.src)))
  raise RuntimeError(f"ALU resolve fail @ {u.op}")

# ***** uop type spec *****

def type_verify(uops):
  for u in uops:
    uop, arg, src, dtype = u.op, u.arg, u.src, u.dtype
    if uop is UOps.DEFINE_LOCAL: assert isinstance(dtype, PtrDType), f"invalid dtype for local buffer {dtype}"
    if uop is UOps.DEFINE_GLOBAL: assert isinstance(dtype, (PtrDType, ImageDType)), f"invalid dtype for global buffer {dtype}"
    if isinstance(dtype, ImageDType): assert uop is UOps.DEFINE_GLOBAL, f"{uop} can't be image"
    if uop is UOps.SHAPETRACKER: assert len(src) == 0, f"SHAPETRACKER must only define a ShapeTracker arg {uop}"
    if uop is UOps.REDUCE_AXIS: assert isinstance(arg, tuple) and len(arg) == 2 and arg[0] in BinaryOps, f"invalid arg for REDUCE_AXIS {arg}"
    if uop in {UOps.CONST, UOps.DEFINE_ACC}:
      if uop is UOps.CONST:
        assert dtype is not None and dtype == dtype.scalar(), f"consts must be scalar, got {dtype}"
        # TODO: intermediate CONST of Variable is DEFINE_VAR
        assert (isinstance(arg, Variable) and u.src) or (type(arg) is type(dtypes.as_const(arg, dtype))), f"type of {arg=} does not match {dtype}"
      if uop is UOps.DEFINE_ACC: assert dtype != dtypes.void and src[0].dtype == dtype, f"dtype mismatch {src[0].dtype=} != {dtype=}"
    if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype != dtypes.void # type is the output type, not an arg
    if uop is UOps.CAST: assert dtype.count == 1 and len(src) == 1
    if uop is UOps.VECTORIZE:
      assert dtype.count > 1 and len(src) == dtype.count, f"dtype vectorization mismatch {dtype.count=} != {len(src)=}"
      assert all(dtype == x.dtype.vec(len(src)) for x in src), f"{dtype=} must be {src[0].dtype.vec(len(src))}"
    if uop is UOps.LOAD and len(src) > 3 and src[3].op is UOps.ALU: assert src[3].dtype == dtypes.bool and src[2].dtype == dtype
    if uop is UOps.GEP: assert dtype == src[0].dtype.scalar(), f"GEP of {src[0].dtype=} should be {src[0].dtype.scalar()} != {dtype}"
    if uop is UOps.IF: assert dtype == dtypes.void and len(src) == 2 and src[0].dtype == dtypes.bool
    if uop is UOps.VALID: assert dtype == dtypes.bool and len(src) == 1 and src[0].op is UOps.SHAPETRACKER and arg is None
    if uop is UOps.STORE:
      assert dtype == dtypes.void, f"{uop} dtype must be void, got {dtype}"
      if len(src) == 4: assert src[3].dtype == dtypes.bool or src[3].op is UOps.IF, f"bad gate {src[3]}"
    if uop is UOps.ALU:
      if arg in UnaryOps: assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in {BinaryOps.CMPLT, BinaryOps.CMPNE}:
        bd = dtypes.bool.vec(dtype.count) if dtype.count != 1 else dtypes.bool
        assert dtype == bd, f"{arg} output dtype mismatch {dtype=} != {bd=}"
        assert src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg is BinaryOps.IDIV:
        assert dtypes.is_int(src[0].dtype) and dtypes.is_int(src[1].dtype), f"input dtype is not int {src[0].dtype=}, {src[1].dtype=}"
        assert dtypes.is_int(dtype), f"output dtype is not int {dtype=}"
      elif arg in {BinaryOps.SHL, BinaryOps.SHR}:
        # the distance to shift isn't typechecked
        assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in BinaryOps: assert dtype == src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg == TernaryOps.WHERE:
        bd = dtypes.bool.vec(dtype.count) if dtype.count != 1 else dtypes.bool
        assert src[0].dtype == bd, f"{arg} selector dtype mismatch {src[0].dtype=} != {bd}"
        assert dtype == src[1].dtype == src[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {src[1].dtype=} != {src[2].dtype=}"

# ***** uop helpers *****

def print_uops(uops:List[UOp]):
  for i,u in enumerate(uops):
    formatted_parents = [uops.index(x) if x.op is not UOps.CONST else f"{x.arg}" for x in u.src]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):25s} " f"{str(formatted_parents):32s} {u.arg}")

def flops_mem(uops:List[UOp], ignore_indexing=False) -> Tuple[sint, sint]:
  flops: sint = 0
  mem: sint = 0
  mults: sint = 1
  mult_stack: List[sint] = []
  dont_count: Set[UOp] = set()
  if ignore_indexing:
    for u in uops:
      if u.op is UOps.LOAD:
        dont_count = dont_count.union(u.src[1].sparents)
        if len(u.src) > 3: dont_count = dont_count.union(u.src[2].sparents)
      elif u.op is UOps.STORE:
        dont_count = dont_count.union(u.src[1].sparents)
        if len(u.src) > 3: dont_count = dont_count.union(u.src[3].sparents)
      elif u.op is UOps.IF:
        dont_count = dont_count.union(u.src[0].sparents)
  for u in uops:
    if u.op is UOps.RANGE:
      mult_stack.append(mults)
      mults *= uop_alu_resolve(u.src[1] - u.src[0])
    elif u.op is UOps.ENDRANGE:
      mults = mult_stack.pop(-1)
    elif u.op is UOps.SPECIAL:
      mults *= u.arg[1] # NOTE: we don't push to the mult_stack here, you can't end these
    elif u.op is UOps.LOAD:
      mem += u.dtype.itemsize * mults
    elif u.op is UOps.STORE:
      mem += u.src[2].dtype.itemsize * mults
    elif u.op is UOps.ALU and u not in dont_count:
      flops += (mults * (2 if u.arg == TernaryOps.MULACC else 1)) * u.dtype.count
    elif u.op is UOps.WMMA and u not in dont_count:
      flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
  return flops, mem

# ***** pattern matcher *****

def get_location() -> Tuple[str, int]:
  frm = sys._getframe(1)
  # no matchers in ops.py, find the real frame
  while (frm.f_code.co_filename.split('/')[-1] in {"ops.py", '<string>'}) and frm.f_back is not None: frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> List[str]:
  with open(fn) as f: return f.readlines()

class UPat(MathTrait):
  __slots__ = ["op", "dtype", "arg", "name", "src"]
  def __init__(self, op:Optional[Union[UOps, Tuple[UOps, ...]]]=None, dtype:Optional[Union[DType, Tuple[DType, ...]]]=None,
               src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None, arg:Any=None,
               name:Optional[str]=None, allow_any_len:bool=False, location=None,
               custom_early_reject:Optional[Set[Tuple[UOps, Any]]]=None):
    self.op: Optional[Tuple[UOps, ...]] = (op,) if isinstance(op, UOps) else op
    self.dtype: Optional[Tuple[DType, ...]] = (dtype,) if isinstance(dtype, DType) else dtype
    self.arg, self.name = arg, name
    self.src: Any = None

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [src]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.allowed_len: int = 0 if allow_any_len or isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = set((pp.op[0], pp.arg) for pp in upat_match if pp.op is not None and len(pp.op) == 1)

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None, vec=True):
    return UPat((UOps.CONST, UOps.VCONST) if vec else UOps.CONST, dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def const(dtype:Optional[DType], b:ConstType|Variable): return UPat(UOps.CONST, dtype=dtype, arg=b)

  # copied from UOp
  def cast(self, dtype=None): return type(self)(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return type(self)(UOps.BITCAST, dtype, (self,))
  def gep(self, i:int): return type(self)(UOps.GEP, None, (self,), (i,))
  @classmethod
  def load(cls, *src:UPat, dtype:Optional[DType]=None): return cls(UOps.LOAD, dtype, src)
  @classmethod
  def store(cls, *src:UPat): return cls(UOps.STORE, dtypes.void, src)

  def const_like(self, b:ConstType|Variable|Tuple[ConstType]): return type(self).const(self.dtype, b)
  def alu(self, arg, *src:UPat):
    asrc = (self,)+src
    return type(self)(UOps.ALU, None if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else asrc[-1].dtype,
                      list(asrc) if arg in COMMUTATIVE else asrc, arg)

  def printable(self:UPat) -> str:
    try:
      return lines(self.location[0])[self.location[1]-1].strip()
    except FileNotFoundError:
      return "<missing>"
  def __repr__(self):
    def rep(x):
      form = "UPat(%s, %s, name=%s, dtype=%s, allow_any_len=%s, src=%s)"
      return form % (None if x.op is None else ('(%s)'%', '.join(map(str, x.op))), x.arg, repr(x.name),
        set(x.dtype) if x.dtype else None, x.allowed_len == 0, "[%s]" if x.src and len(x.src)>1 else "(%s)")
    return pretty_print(self, rep, srcfn=lambda x:None if x.src is None else [next(x.src[0])] if isinstance(x.src[0], itertools.repeat) else x.src[0])

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
  if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \
     (pat.dtype is not None and uop.dtype not in pat.dtype) or \
     (pat.arg is not None and pat.arg != uop.arg) or \
     (pat.op is not None and uop.op not in pat.op) or \
     (pat.allowed_len != 0 and len(uop.src) != pat.allowed_len): return []
  if pat.src is None: return [store]
  res: List[Dict[str, UOp]] = []
  for vp in pat.src:
    stores, new_stores = [store.copy()], []
    for uu, vv in zip(uop.src, vp):
      for s in stores: new_stores.extend(_match(uu, vv, s))
      stores, new_stores = new_stores, []
    res.extend(stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable, Set]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn, p.early_reject))

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ler = set([v for u in uop.src for v in ((u.op, u.arg), (u.op, None))])
    for p,fxn,early_reject in self.pdict[(uop.op, uop.arg)] + ([] if uop.arg is None else self.pdict[(uop.op, None)]):
      if not early_reject.issubset(ler): continue
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = getenv("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
match_stats:Dict[UPat, List[Union[int, float]]] = dict()
@dataclass(frozen=True)
class TrackedRewriteContext:
  loc: str                                  # location that called graph_rewrite
  sink: UOp                                 # the sink passed into the rewrite
  rewrites: List[Tuple[UOp, UOp, str]]      # all rewrites of sparents. (before, after, UPat printable)
contexts: List[TrackedRewriteContext] = []
class TrackedPattenMatcher(PatternMatcher):
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    super().__init__(patterns)
    for p,_ in self.patterns:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ret = None
    ler = set([v for u in uop.src for v in ((u.op, u.arg), (u.op, None))])
    for p,fxn,early_reject in self.pdict[(uop.op, uop.arg)] + ([] if uop.arg is None else self.pdict[(uop.op, None)]):
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None:
        match_stats[p][0] += 1
        match_stats[p][2] += (et:=time.perf_counter()-st)
        match_stats[p][3] += et
        if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", p.printable())
        if TRACK_MATCH_STATS >= 2: contexts[-1].rewrites.append((uop, ret, p.printable()))
        return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPattenMatcher  # type: ignore
  import atexit, pickle
  @atexit.register
  def print_match_stats():
    ret = [0,0,0.0,0.0]
    if getenv("PRINT_MATCH_STATS", 1):
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {v[2]*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {ret[2]*1000.:9.2f} ms -- TOTAL")
    if TRACK_MATCH_STATS >= 2:
      with open("/tmp/rewrites.pkl", "wb") as f:
        #print(f"rewrote {len(contexts)} graphs and applied {sum(len(x.rewrites) for x in contexts)} rules, saved to /tmp/rewrites.pkl")
        pickle.dump(contexts, f)
    if getenv("VIZ"):
      import viz.serve
      viz.serve.main()

# *** simple graph rewrite engine ***

class RewriteContext:
  def __init__(self, pm):
    self.pm: PatternMatcher = pm
    self.nodes: Dict[Tuple, UOp] = {}
    self.replace: Dict[UOp, UOp] = {}
  def rewrite(self, n:UOp) -> UOp:
    if rn := self.replace.get(n): return rn
    replace_source = (n.op, n.dtype, new_src:=tuple(map(self.rewrite, n.src)), n.arg)
    if found := self.nodes.get(replace_source): self.replace[n] = found
    else:
      x = UOp(*replace_source) if new_src != n.src else n
      self.nodes[replace_source] = self.replace[n] = found = self.rewrite(new_x) if (new_x := self.pm.rewrite(x)) else x
    return found
def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  if TRACK_MATCH_STATS >= 2: contexts.append(TrackedRewriteContext(f"{(l:=get_location())[0].split('/')[-1]}:{l[1]}", sink, []))
  return RewriteContext(pm).rewrite(sink)
