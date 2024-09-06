from __future__ import annotations
from typing import Any, DefaultDict, List, Optional, Set, Union, Tuple, Dict, Callable, cast, TYPE_CHECKING, Sequence, TypeVar
import sys, time, math, operator, ctypes, struct, functools, hashlib, itertools
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes, DType
from tinygrad.helpers import pretty_print, prod, getenv, all_same
from tinygrad.shape.symbolic import Variable, sint
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker

# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum):
  """A -> A (elementwise)"""
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto() # noqa: E702
class BinaryOps(Enum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto() # noqa: E702
class TernaryOps(Enum):
  """A + A + A -> A (elementwise)"""
  WHERE = auto(); MULACC = auto() # noqa: E702
class ReduceOps(Enum):
  """A -> B (reduce)"""
  SUM = auto(); PROD = auto(); MAX = auto() # noqa: E702
class MetaOps(Enum):
  EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto(); VIEW = auto() # noqa: E702
Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps]

T = TypeVar("T")
class MathTrait:
  # required to implement
  def alu(self:T, arg:Union[UnaryOps, BinaryOps, TernaryOps], *src) -> T: raise NotImplementedError
  def const_like(self, b:ConstType|Variable): raise NotImplementedError

  # great functions you get!
  def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
  def __neg__(self): return self.ne(True) if getattr(self, 'dtype', None) == dtypes.bool else self*(-1)
  def __add__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
  def __radd__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
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
  def eq(self, x): return -self.ne(x)
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
class UOps(Enum):
  # uops that aren't rendered
  SINK = auto()
  """
  Holds `UOps.STORE`. SINK defines the AST for a Kernel.

  - **`dtype`**: `None`
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
  Defines the ShapeTracker for a buffer UOp `UOps.LOAD`, `UOps.STORE` or `UOps.CONST`.

  - **`dtype`**: `None`
  - **`src`**: `Tuple[]`
  - **`arg`**: `ShapeTracker`
  """
  SWIZZLE = auto()
  """
  Swizzle inserts a movement op between a UOp and its children. Because movement ops (reshape, expand, shrink, permute, pad) are not allowed in an AST,
  the scheduler rewrites SWIZZLE by pushing its ShapeTracker through reduceops or elementwise ops to the edges of the graph.

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
          UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),
    UOp(UOps.LOAD, dtypes.int, arg=None, src=(
       x3,
      UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),))
  ```

  The scheduler rewrites this by pushing the expand in SWIZZLE through the reduce, to the LOAD:

  ```diff
  UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
  -   UOp(UOps.SWIZZLE, dtypes.int, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=(
  -     UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (0, 1)), src=(
  -       UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  -         x3:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
  -         UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),
  +   UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (2, 3)), src=(
  +     UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  +       x2:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
  +       UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32, 32, 32), strides=(0, 0, 32, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
    UOp(UOps.LOAD, dtypes.int, arg=None, src=(
  -      x3,
  -     UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),))
  +      x2,
  +     UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),))

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
  CONST = auto()
  """
  Defines a single scalar constant value.

  - **`dtype`**: The scalar DType of the value.

  - **`src`**:
    The scheduler creates a CONST with a single SHAPETRACKER UOp src: `Tuple[UOp]`.

    The Lowerer replaces the SHAPETRACKER with an empty src.
    It uses the ShapeTracker valid to create a `WHERE` UOp mask with sources: `(The actual CONST UOp, CONST 0, 0.0 or False)`

  - **`arg`**: The value.
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
  - **`dtype`**: `None`
  - **`src`**:

    Similar to LOAD, the scheduler and Kernel create STOREs with a SHAPETRACKER uop in src:

    - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
    - SHAPETRACKER UOp.
    - Value to store.

    The Lowerer replaces the SHAPETRACKER with an indexing uop and gates the STORE if needed.

    - Normal STORE: `Tuple[UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
      - Indexing Op, can only return `dtypes.int32`.
      - Value to store.
    - Gated STORE: `Tuple[UOp, UOp, UOp, UOp]`
      - Buffer UOp `UOps.DEFINE_GLOBAL` or `UOps.DEFINE_LOCAL`.
      - Indexing UOp, can only return `dtypes.int32`.
      - Value to store.
      - Gate UOp, can only return `dtypes.bool`. We rewrite this to an IF block in the end.
  - **`arg`**: `None`
  """
  PHI = auto()
  # control flow ops
  BARRIER = auto()
  """
  Inserts a warp sync between local stores and local loads.

  - **`dtype`**: `None`
  - **`src`**: `Tuple[UOp, ...]`, Only local STOREs are allowed.
  - **`arg`**: `None`
  """
  IF = auto()
  """
  Gates a single STORE to global memory. The IF block could also contain additional UOps the STORE depends on.

  - **`dtype`**: `None`
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
    UOp(UOps.BARRIER, None, (...))))
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

END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}

@dataclass(frozen=True, eq=False)
class UOp(MathTrait):
  op: UOps
  dtype: Optional[DType] = None
  src: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def __hash__(self): return id(self)
  @functools.cached_property
  def cmp_tuple(self) -> Tuple[int, Any, Optional[DType], Tuple[UOp, ...]]:
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    return (self.op.value, (self.arg if self.op is not UOps.DEFINE_VAR else self.arg.expr) if self.op is not UOps.ALU else \
            self.arg.value, self.dtype, self.src)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self):
    return f'({", ".join(map(str, self.arg))})' if self.op is UOps.REDUCE_AXIS else repr(self.arg) if isinstance(self.arg, Variable) else self.arg
  # *** uop syntactic sugar
  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in BUFFER_UOPS, f"st_arg called on {self.op}"
    ret = self.src[0 if self.op is UOps.CONST else 1]
    assert ret.op is UOps.SHAPETRACKER, f"st_arg trying to return {ret}"
    return ret.arg
  def sink(self, *srcs): return UOp(UOps.SINK, None, (self,)+srcs)
  def cast(self, dtype=None): return type(self)(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return type(self)(UOps.BITCAST, dtype, (self,))
  def gep(self, i:int): return type(self)(UOps.GEP, self.dtype.scalar() if self.dtype is not None else None, (self,), i)
  def const_like(self, b:ConstType|Variable): return type(self).const(self.dtype, b)
  def sconst_like(self, b:ConstType|Variable): return type(self).const(self.dtype.scalar() if self.dtype is not None else None, b)
  @classmethod
  @functools.lru_cache(None)
  def const(cls, dtype:Optional[DType], b:ConstType|Variable): return cls._const(dtype, b)
  @classmethod
  def _const(cls, dtype:Optional[DType], b:ConstType|Variable):
    # TODO: fix dtype of b.max after Variable is just an UOp
    if isinstance(b, Variable): return cls(UOps.DEFINE_VAR, dtype, (cls.const(dtypes.int, b.min), cls.const(dtypes.int, cast(int,b.max))), b)
    if dtype is not None and dtype != (sdtype := dtype.scalar()):
      return cls(UOps.VECTORIZE, dtype, src=tuple(cls(UOps.CONST, sdtype, arg=dtypes.as_const(b, sdtype)) for _ in range(dtype.count)))
    return cls(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b)
  def alu(self, arg, *src:UOp):
    return type(self)(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else (self, *src)[-1].dtype, (self,)+src, arg)
  @classmethod
  def load(cls, *src:UOp, dtype:Optional[DType]=None): return cls(UOps.LOAD, dtype, src)
  @classmethod
  def store(cls, *src:UOp): return cls(UOps.STORE, None, src)
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
    return sorted(set.union(*st_vars, set([x.arg for x in self.sparents if x.op is UOps.DEFINE_VAR])), key=lambda v: v.expr)
  def const_factor(self) -> int:
    """largest known int that divides self"""
    if self.op is UOps.CONST: return self.arg
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return math.gcd(self.src[0].const_factor(), self.src[0].const_factor())
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
  def vmin(self) -> UOp:
    return x if (x:=self._min_max[0]) is not None and not math.isnan(x.arg) else self.sconst_like(dtypes.min(cast(DType, self.dtype)))
  @property
  def vmax(self) -> UOp:
    return x if (x:=self._min_max[1]) is not None and not math.isnan(x.arg) else self.sconst_like(dtypes.max(cast(DType, self.dtype)))
  @functools.cached_property
  def _min_max(self) -> Tuple[Optional[UOp], Optional[UOp]]:
    # NOTE: returned UOp is assumed to be CONST
    if self.op is UOps.DEFINE_VAR and self.src: return self.src[0], self.src[1] if isinstance(self.src[1].arg, int) else None
    if self.op is UOps.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    # TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if self.op is UOps.SPECIAL: return self.const_like(0), self.const_like(self.arg[1]-1) if isinstance(self.arg[1], int) else None
    if self.op is UOps.CONST: return self, self
    if self.op is UOps.ALU and cast(DType, self.dtype).count == 1:
      s0,s1 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(2)]
      if self.arg is BinaryOps.ADD: return self.sconst_like(s0.vmin.arg+s1.vmin.arg), self.sconst_like(s0.vmax.arg+s1.vmax.arg)
      if self.arg is BinaryOps.MUL and (s0.vmin.arg >= 0 or s1.vmin.arg >= 0):
        # handle at lease one is non-negative
        Lmin, Lmax = (s0.vmin.arg, s0.vmax.arg) if s1.vmin.arg >= 0 else (s0.vmax.arg, s0.vmin.arg)
        Rmin, Rmax = (s1.vmin.arg, s1.vmax.arg) if s0.vmin.arg >= 0 else (s1.vmax.arg, s1.vmin.arg)
        assert math.isnan(Lmax*Rmax) or math.isnan(Lmin*Rmin) or Lmax*Rmax >= Lmin*Rmin, f"{Lmax=}, {Lmin=}, {Rmax=}, {Rmin=}"
        return self.sconst_like(Lmin*Rmin), self.sconst_like(Lmax*Rmax)
      if self.arg is BinaryOps.MOD and s1.vmin.arg > 0: return self.sconst_like(0), self.sconst_like(s1.vmax.arg-1)
      if self.arg is BinaryOps.IDIV and s1.op is UOps.CONST:
        if s1.arg > 0: return self.sconst_like(s0.vmin.arg//s1.arg), self.sconst_like(s0.vmax.arg//s1.arg)
        if s1.arg < 0: return self.sconst_like(-(s0.vmax.arg//-s1.arg)), self.sconst_like(-(s0.vmin.arg//-s1.arg))
      if self.arg is BinaryOps.MAX: return self.sconst_like(max(s0.vmin.arg, s1.vmin.arg)), self.sconst_like(max(s0.vmax.arg, s1.vmax.arg))
      if self.arg is BinaryOps.CMPLT: return (UOp.const(dtypes.bool, s0.vmax.arg<s1.vmin.arg), UOp.const(dtypes.bool, s0.vmin.arg<s1.vmax.arg))
    return None, None

@dataclass(frozen=True)
class KernelInfo:
  local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)
  upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to EXPAND)
  dont_use_locals: bool = False # don't use local indexing

# ***** pattern matcher *****

def get_location() -> Tuple[str, int]:
  frm = sys._getframe(1)
  # no matchers in ops.py, find the real frame
  while (frm.f_code.co_filename.endswith("/ops.py") or frm.f_code.co_filename == '<string>') and frm.f_back is not None: frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> List[str]: return open(fn).readlines()

@dataclass(frozen=True, repr=False)  # reuse repr from UOp
class NOp(UOp):
  name: Optional[str] = None
  src: Tuple[NOp, ...] = tuple()
  allow_any_len: bool = False
  location: Tuple[str, int] = field(default_factory=get_location)

  def commutative(self) -> bool:
    return (self.op is UOps.ALU and \
      self.arg in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR})

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.NOOP, dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.CONST, dtype=dtype, name=name)

  # this is needed so NOp has a different cache
  @classmethod
  @functools.lru_cache(None)
  def const(cls, dtype:Optional[DType], b:ConstType|Variable): return cls._const(dtype, b)

  @functools.cached_property
  def upat(self:NOp) -> UPat:
    return UPat(name=self.name, dtype=self.dtype, location=self.location) if self.op is UOps.NOOP else \
      UPat(self.op, self.arg, (list if self.commutative() else tuple)([src.upat for src in self.src]) or None, self.name,
           self.dtype, self.allow_any_len, location=self.location)

class UPat:
  def __init__(self, op:Optional[Union[UOps, Set[UOps]]]=None, arg:Any=None, src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None,
               name:Optional[str]=None, dtype:Optional[Union[DType, Set[DType]]]=None, allow_any_len:bool=False, location=None,
               custom_early_reject:Optional[Set[Tuple[UOps, Any]]]=None):
    self.op: Optional[Tuple[UOps, ...]] = None if op is None else (tuple(op) if isinstance(op, set) else (op,))
    self.dtype: Optional[Tuple[DType, ...]] = None if dtype is None else (tuple(dtype) if isinstance(dtype, set) else (dtype,))
    self.arg, self.name = arg, name
    self.in_src = src
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
      upat_match = [self.in_src] if isinstance(self.in_src, UPat) else ([] if self.in_src is None else self.src[0])
      self.early_reject = set((pp.op[0], pp.arg) for pp in upat_match if pp.op is not None and len(pp.op) == 1)

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
    new_stores = [store.copy()]
    for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]
    res.extend(new_stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:Sequence[Tuple[Union[UPat, NOp], Callable]]):
    self.patterns = [(p.upat if isinstance(p, NOp) else p, fxn) for p,fxn in patterns]
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable, Set]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn, p.early_reject))

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ler = set([(u.op, u.arg) for u in uop.src] + [(u.op, None) for u in uop.src])
    for p,fxn,early_reject in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      if not early_reject.issubset(ler): continue
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = getenv("TRACK_MATCH_STATS", 0)
match_stats:Dict[UPat, List[Union[int, float]]] = dict()
class TrackedPattenMatcher(PatternMatcher):
  def __init__(self, patterns:List[Tuple[Union[UPat, NOp], Callable]]):
    super().__init__(patterns)
    for p,_ in self.patterns:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ret = None
    ler = set([(u.op, u.arg) for u in uop.src] + [(u.op, None) for u in uop.src])
    for p,fxn,early_reject in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None:
        match_stats[p][0] += 1
        match_stats[p][2] += (et:=time.perf_counter()-st)
        match_stats[p][3] += et
        if TRACK_MATCH_STATS >= 2: print(f"{et*1e6:7.2f} us -- ", p.printable())
        return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPattenMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    ret = [0,0,0.0,0.0]
    for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]):
      loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
      print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {v[2]*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
      ret = [x+y for x,y in zip(ret, v)]
    print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {ret[2]*1000.:9.2f} ms -- TOTAL")

# *** simple graph rewrite engine ***

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if rn := replace.get(n): return rn
    replace_source = (n.op, n.dtype, new_src:=tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else:
      x = UOp(*replace_source) if new_src != n.src else n
      nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x)) else x
    return found
  return __inner_rewrite(sink)

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
  try:
    x = float(x)
    struct.pack("@e", x)
    return x
  except OverflowError: return math.copysign(math.inf, x)

truncate: Dict[DType, Callable] = {dtypes.bool: bool,
  # TODO: bfloat16
  dtypes.float16: truncate_fp16, dtypes.float32: lambda x: ctypes.c_float(x).value, dtypes.float64: lambda x: ctypes.c_double(x).value,
  dtypes.uint8: lambda x: ctypes.c_uint8(x).value, dtypes.uint16: lambda x: ctypes.c_uint16(x).value,
  dtypes.uint32: lambda x: ctypes.c_uint32(x).value, dtypes.uint64: lambda x: ctypes.c_uint64(x).value,
  dtypes.int8: lambda x: ctypes.c_int8(x).value, dtypes.int16: lambda x: ctypes.c_int16(x).value, dtypes.int32: lambda x: ctypes.c_int32(x).value \
      if isinstance(x,int) else x, dtypes.int64: lambda x: ctypes.c_int64(x).value}

def exec_alu(op:Op, dtype:DType, operands): return truncate.get(dtype, lambda x: x)(python_alu[op](*operands))

def uop_alu_resolve(u:UOp) -> sint:
  if u.op in {UOps.CONST, UOps.DEFINE_VAR}: return u.arg
  if u.op is UOps.ALU: return exec_alu(u.arg, cast(DType,u.dtype), tuple(map(uop_alu_resolve, u.src)))
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
      if uop is UOps.DEFINE_ACC: assert dtype is not None and src[0].dtype == dtype, f"dtype mismatch {src[0].dtype=} != {dtype=}"
    if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype is not None # type is the output type, not an arg
    if uop is UOps.CAST: assert dtype.count == 1 and len(src) == 1
    if uop is UOps.VECTORIZE:
      assert dtype.count > 1 and len(src) == dtype.count, f"dtype vectorization mismatch {dtype.count=} != {len(src)=}"
      assert all(dtype == x.dtype.vec(len(src)) for x in src), f"{dtype=} must be {src[0].dtype.vec(len(src))}"
    if uop is UOps.LOAD and len(src) > 3 and src[3].op is UOps.ALU: assert src[3].dtype == dtypes.bool and src[2].dtype == dtype
    if uop is UOps.GEP: assert dtype == src[0].dtype.scalar(), f"GEP of {src[0].dtype=} should be {src[0].dtype.scalar()} != {dtype}"
    if uop is UOps.IF: assert dtype is None and len(src) == 2 and src[0].dtype == dtypes.bool
    if uop is UOps.STORE:
      assert dtype is None, f"{uop} dtype must be None, got {dtype}"
      if len(src) == 4: assert src[3].dtype == dtypes.bool, f"gate dtype mismatch {src[3].dtype} != {dtypes.bool}"
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
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str(formatted_parents):32s} {u.arg}")

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
      assert u.dtype is not None
      mem += u.dtype.itemsize * mults
    elif u.op is UOps.STORE:
      assert u.src[2].dtype is not None
      mem += u.src[2].dtype.itemsize * mults
    elif u.op is UOps.ALU and u not in dont_count:
      assert u.dtype is not None
      flops += (mults * (2 if u.arg == TernaryOps.MULACC else 1)) * u.dtype.count
    elif u.op is UOps.WMMA and u not in dont_count:
      assert u.arg[1] is not None
      flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
  return flops, mem
