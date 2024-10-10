from __future__ import annotations
from typing import Any, List, Optional, Set, Union, Tuple, Dict, Callable, cast, TYPE_CHECKING, TypeVar
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle
from enum import auto, IntEnum, Enum
from dataclasses import dataclass, field
from weakref import WeakValueDictionary
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes, DType, truncate
from tinygrad.helpers import ContextVar, prod, getenv, all_same
if TYPE_CHECKING:
  from tinygrad.shape.symbolic import Variable, sint
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
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(FastEnum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto(); SUB = auto() # noqa: E702
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
  def logical_not(self): return self.ne(True)
  def __neg__(self):
    dtype = getattr(self, 'dtype', None)
    assert dtype is not None, "MathTraits __neg__ requires a dtype"
    return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
  def __add__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
  def __radd__(self, x): return self.ufix(x).alu(BinaryOps.ADD, self)
  def __sub__(self, x): return self.alu(BinaryOps.ADD, self.ufix(-x))
  def __rsub__(self, x): return self.ufix(x).alu(BinaryOps.ADD, -self)
  def __mul__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x))
  def __rmul__(self, x): return self.ufix(x).alu(BinaryOps.MUL, self)
  def __floordiv__(self, x): return self.alu(BinaryOps.IDIV, self.ufix(x))
  def __rfloordiv__(self, x): return self.ufix(x).alu(BinaryOps.IDIV, self)
  def __truediv__(self, x): return self.alu(BinaryOps.MUL, self.ufix(x).alu(UnaryOps.RECIP))
  def __rtruediv__(self, x): return self.ufix(x).alu(BinaryOps.MUL, self.alu(UnaryOps.RECIP))
  def __mod__(self, x): return self.alu(BinaryOps.MOD, self.ufix(x))
  def __rmod__(self, x): return self.ufix(x).alu(BinaryOps.MOD, self)
  def __xor__(self, x): return self.alu(BinaryOps.XOR, self.ufix(x))
  def __and__(self, x): return self.alu(BinaryOps.AND, self.ufix(x))
  def __rand__(self, x): return self.ufix(x).alu(BinaryOps.AND, self)
  def __or__(self, x): return self.alu(BinaryOps.OR, self.ufix(x))
  def ne(self, x): return self.alu(BinaryOps.CMPNE, self.ufix(x))
  def eq(self, x): return self.ne(x).logical_not()
  def lt(self, x): return self.alu(BinaryOps.CMPLT, self.ufix(x))
  def gt(self, x): return self.ufix(x).alu(BinaryOps.CMPLT, self)
  def ge(self, x): return self.lt(x).logical_not()
  def le(self, x): return self.gt(x).logical_not()
  # NOTE: __eq__ can't be overridden, and means the same thing as is
  def __ne__(self, x): return self.ne(x)
  def __lt__(self, x): return self.lt(x)
  def __gt__(self, x): return self.gt(x)
  def __ge__(self, x): return self.ge(x)
  def __le__(self, x): return self.le(x)
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

  # metaops
  CUSTOM = auto()
  COPY = auto()
  EMPTY = auto()
  BUFFER_VIEW = auto()

  EXPAND = auto()
  CONTRACT = auto()
  VIEW = auto()
  DEFINE_GLOBAL = auto()
  BUFFER = auto()
  DEFINE_VAR = auto()
  DEFINE_LOCAL = auto()
  DEFINE_ACC = auto()
  VCONST = auto()
  CONST = auto()
  VALID = auto()
  SPECIAL = auto()
  NOOP = auto()
  REDUCE = auto()
  REDUCE_AXIS = auto()

  # helper ops
  GEP = auto()
  VECTORIZE = auto()
  CAST = auto()
  BITCAST = auto()

  # loads before math
  LOAD = auto()

  # math ops
  ALU = auto()
  WMMA = auto()

  # assignment ops
  STORE = auto()
  ASSIGN = auto()
  BIND = auto()

  # control flow ops
  BARRIER = auto()
  IF = auto()
  RANGE = auto()

  # ops that are not graph nodes
  ENDRANGE = auto()
  ENDIF = auto()

BUFFER_UOPS = {UOps.LOAD, UOps.STORE, UOps.VALID}
COMMUTATIVE = {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR}
END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.ASSIGN, UOps.ENDRANGE)}

# With True as the default, this matches the old symbolic behavior
# python3 -c 'from tinygrad.shape.symbolic import Variable; print(bool(Variable("a", 1, 10) < 4))' -> True
def resolve(x, default:bool=True):
  if not isinstance(x, UOp): return bool(x)
  assert x.dtype is dtypes.bool, "UOp in resolve must be bool"
  # NOTE: generating the text for the exception is expensive, so we do this
  return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default
def smax(lst): return max(lst, key=lambda x: x if isinstance(x, int) else x.vmax)
def ssimplify(uop): return uop.ssimplify() if isinstance(uop, UOp) else uop

# used for UOp and UPat
def pretty_print(x:Any, rep:Callable, srcfn=lambda x: x.src, cache=None, d=0)->str:
  def dfs(x:Any, cache:dict):
    for s in srcfn(x) or []:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d} x{cx[0]}"
  cx[2], srcs = True, ('None' if srcfn(x) is None else ''.join(f'\n{pretty_print(s, rep, srcfn, cache, d+2)},' for s in srcfn(x)))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{rep(x)}" % srcs

ucache:WeakValueDictionary[Tuple, UOp] = WeakValueDictionary()
class UOp(MathTrait):
  def __reduce__(self): return UOp, (self.op, self.dtype, self.src, self.arg)
  def __new__(cls, op:UOps, dtype:DType=dtypes.void, src:Tuple[UOp,...]=tuple(), arg:Any=None):
    if (ret:=ucache.get(key:=(op, dtype, src, arg), None)) is not None: return ret
    ucache[key] = ret = super().__new__(cls)
    return ret

  __slots__ = ["op", "dtype", "src", "arg"]
  def __init__(self, op: UOps, dtype:DType=dtypes.void, src: Tuple[UOp,...]=tuple(), arg:Any=None):
    # TODO: instant check rules here make debugging easier
    #if op is UOps.ALU and arg is BinaryOps.CMPNE: assert dtype.scalar() == dtypes.bool
    #if op is UOps.VECTORIZE and dtype != dtypes.void: assert len(src) == dtype.count, f"{len(src)} invalid for {dtype}"
    #if op is UOps.ALU and arg not in (BinaryOps.CMPNE, BinaryOps.CMPLT, TernaryOps.WHERE): assert all_same([dtype] + [x.dtype for x in src])
    #if op is UOps.CAST: assert dtype.count == src[0].dtype.count, f"cast can't change vectorization {src[0].dtype} --> {dtype}"
    self.op, self.dtype, self.src, self.arg = op, dtype, src, arg
  def replace(self, **kwargs) -> UOp:
    for k in kwargs: assert k in self.__slots__, f"unkown replace arg, expected one of {self.__slots__}, got {k}"
    new_args = (kwargs.get("op", self.op), kwargs.get("dtype", self.dtype), kwargs.get("src", self.src), kwargs.get("arg", self.arg))
    if (self.op, self.dtype, self.src, self.arg) == new_args: return self
    return UOp(*new_args)
  @property
  def has_st(self) -> bool: return self.op not in {UOps.DEFINE_LOCAL, UOps.DEFINE_GLOBAL, UOps.BUFFER, UOps.CONST, UOps.DEFINE_VAR}
  @functools.cached_property
  def st(self) -> Optional[ShapeTracker]:
    if not self.has_st: return None
    if self.op in BUFFER_UOPS: return self.st_arg
    if self.op is UOps.VIEW: return self.arg
    src_sts = [x.st for x in self.src if x.st is not None]
    assert all_same([x.shape for x in src_sts]), f"UOp parents must have the same shape {self} {[x.shape for x in src_sts]}"
    from tinygrad.shape.shapetracker import ShapeTracker
    return ShapeTracker.from_shape(src_sts[0].reduce(self.axis_arg)) if self.op is UOps.REDUCE_AXIS else src_sts[0]
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is UOps.REDUCE_AXIS else self.arg
  # *** uop evaluation ***
  def simplify(self): return graph_rewrite(self, symbolic)
  def ssimplify(self) -> Union[UOp, ConstType]: return ret.arg if (ret:=self.simplify()).op is UOps.CONST else ret
  def _eval(self, dtype, expected_type) -> ConstType:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    simple_self = self.simplify()
    vmin, vmax = simple_self._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert type(vmin) is expected_type, f"vmin is wrong dtype {vmin} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return self._eval(dtypes.floats, float)
  # *** uop syntactic sugar
  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in BUFFER_UOPS, f"st_arg called on {self.op}"
    ret = self.src[0 if self.op is UOps.VALID else 1]
    assert ret.op is UOps.VIEW, f"st_arg trying to return {ret}"
    return ret.arg
  @property
  def axis_arg(self) -> Tuple[int, ...]:
    assert self.op in {UOps.REDUCE_AXIS, UOps.WMMA}, f"axis_arg called on {self.op}"
    ret = self.arg[1] if self.op is UOps.REDUCE_AXIS else self.arg[7]
    assert isinstance(ret, tuple) and all(isinstance(x, int) for x in ret), f"axis_arg trying to return {ret}"
    return ret
  def sink(self, *srcs:UOp): return UOp(UOps.SINK, dtypes.void, (self,)+srcs)
  def view(self, st:ShapeTracker): return UOp(UOps.VIEW, self.dtype, (self,), st)
  def const_like(self, b:ConstType|Variable|Tuple[ConstType, ...]): return UOp.const(self.dtype, b)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(UOps.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType): return UOp(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return UOp(UOps.BITCAST, dtype, (self,))
  def gep(self, i:Union[Tuple[int, ...], int]):
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is UOps.VECTORIZE: return self.src[i]
      if self.op is UOps.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
      if self.op is UOps.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    if self.dtype == dtypes.void or (i == tuple(range(len(i))) and self.dtype.count == len(i)): return self
    assert len(i) >= 1 and all(x < self.dtype.count for x in i), f"bad GEP on {self.dtype}, {i}"
    return UOp(UOps.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  @staticmethod
  def load(*src:UOp, dtype:DType): return UOp(UOps.LOAD, dtype, src)
  @staticmethod
  def store(*src:UOp): return UOp(UOps.STORE, dtypes.void, src)
  def alu(self, arg, *src:UOp):
    out_dtype = (self, *src)[-1].dtype
    if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} and out_dtype is not None:
      out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(UOps.ALU, out_dtype, (self,)+src, arg)
  @staticmethod
  def const(dtype:DType, b:Tuple[ConstType, ...]|ConstType|Variable): return UOp._const(dtype, b)
  @staticmethod
  def _const(dtype:DType, b:Tuple[ConstType, ...]|ConstType|Variable):
    # TODO: fix dtype of b.max after Variable is just an UOp
    #if isinstance(b, Variable): return UOp.define_var(b.expr, dtype, b.min, cast(int, b.max))
    if isinstance(b, UOp): return b.unbind()[0] if b.op is UOps.BIND else b
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    return UOp(UOps.VCONST if isinstance(b, tuple) else UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b) # type: ignore
  @staticmethod
  def define_var(name:str, dtype:DType, min_val:ConstType, max_val:ConstType): return UOp(UOps.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  def unbind(self) -> Tuple[Variable, int]:
    assert self.op is UOps.BIND and self.src[0].op is UOps.DEFINE_VAR and self.src[1].op is UOps.CONST, f"can't unbind {self}"
    from tinygrad.shape.symbolic import Variable
    return cast(Variable, self.src[0]), self.src[1].arg
  @property
  def val(self) -> int: return self.unbind()[1]
  # TODO: this is context rewrite
  def substitute(self, dvars:Dict[UOp, UOp]):
    if self in dvars: return dvars[self]
    return self.replace(src=tuple(x.substitute(dvars) for x in self.src))
  @staticmethod
  def range(dtype:DType, start:ConstType, end:ConstType, idx:int):
    return UOp(UOps.RANGE, dtype=dtype, src=(UOp.const(dtype, start), UOp.const(dtype, end)), arg=(idx,))
  def reduce(self, op:BinaryOps, *rng:UOp): return UOp(UOps.REDUCE, self.dtype, (self,) + rng, op)
  @functools.cached_property
  def parents(self) -> Dict[UOp, None]: return {**{x:None for x in self.src}, **{k:None for x in self.src for k in x.parents}}
  @property  # parents with self
  def sparents(self) -> Dict[UOp, None]: return {**self.parents, self:None}
  @functools.cached_property
  def full_shape(self) -> Tuple[sint, ...]:
    return self.arg.shape if self.op is UOps.VIEW else tuple(smax(x) for x in zip(*[x.full_shape for x in self.src if x.has_st]))
  def vars(self) -> Set[UOp]:
    bound_vars = set([x for x in self.sparents if x.op is UOps.BIND and x.src[0].op is UOps.DEFINE_VAR])
    bound_var_base = set(x.src[0] for x in bound_vars)
    all_vars = set([x for x in self.sparents if x.op is UOps.DEFINE_VAR])
    return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
  def variables(self) -> List[Variable]:
    st_vars: List[Set[Variable]] = [x.st_arg.vars() for x in self.sparents if x.op in BUFFER_UOPS]
    from tinygrad.shape.symbolic import Variable
    return sorted(set.union(*st_vars, [x.unbind()[0] if not isinstance(x, Variable) else x for x in self.vars()]), key=lambda v: v.arg)
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
    if self.op is UOps.VCONST: return self.const_like(tuple(x//v for x in self.arg)) if all(x%v == 0 for x in self.arg) else None
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
    if self.op is UOps.DEFINE_VAR and self.arg: return self.arg[1], self.arg[2]
    if self.op is UOps.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    if self.op is UOps.BIND: return self.src[0].vmin, self.src[0].vmax  # ignore the bound value
    if self.op is UOps.EXPAND: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    # TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if self.op is UOps.SPECIAL: return 0, self.arg[1]-1 if isinstance(self.arg[1], int) else dtypes.max(self.dtype)
    if self.op is UOps.CONST: return self.arg, self.arg
    if self.op is UOps.VCONST: return (min(self.arg), max(self.arg))
    if self.op is UOps.ALU and self.dtype.count == 1:
      s0,s1,s2 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(3)]
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
      if self.arg is BinaryOps.CMPNE:
        always_ne = (s0.vmax < s1.vmin) or (s1.vmax < s0.vmin)
        sometimes_ne = not (s0.vmin == s0.vmax == s1.vmin == s1.vmax)
        return (always_ne, sometimes_ne)
      # float has NAN issue and we use explicit NAN in transcendental
      if self.arg is TernaryOps.WHERE and dtypes.is_int(s1.dtype): return min(s1.vmin, s2.vmin), max(s1.vmax, s2.vmax)
      if self.dtype is dtypes.bool:
        if self.arg is BinaryOps.OR: return s0.vmin or s1.vmin, s0.vmax or s1.vmax
        if self.arg is BinaryOps.AND: return s0.vmin and s1.vmin, s0.vmax and s1.vmax
    return dtypes.min(self.dtype), dtypes.max(self.dtype)
  def render(self, simplify=True) -> str:
    ret = graph_rewrite(self.simplify() if simplify else self, renderer)
    return ret.arg if ret.op is UOps.NOOP else str(ret)

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
  UnaryOps.NEG: operator.neg, BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub,
  BinaryOps.SHR: operator.rshift, BinaryOps.SHL: operator.lshift, BinaryOps.MUL: operator.mul,
  BinaryOps.XOR: operator.xor, BinaryOps.MAX: max, BinaryOps.CMPNE: operator.ne, BinaryOps.CMPLT: operator.lt,
  BinaryOps.OR: operator.or_, BinaryOps.AND: operator.and_,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], BinaryOps.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else x*math.inf,
  TernaryOps.MULACC: lambda x,y,z: (x*y)+z, TernaryOps.WHERE: lambda x,y,z: y if x else z}

def exec_alu(op:Op, dtype:DType, operands):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  return truncate.get(dtype, lambda x: x)(python_alu[op](*operands))

def uop_alu_resolve(u:UOp) -> sint:
  if u.op is UOps.CONST: return u.arg
  if u.op is UOps.DEFINE_VAR: return u
  #if u.op is UOps.DEFINE_VAR: return Variable(u.arg[0], u.arg[1], u.arg[2])
  if u.op is UOps.ALU: return exec_alu(u.arg, u.dtype, tuple(map(uop_alu_resolve, u.src)))
  raise RuntimeError(f"ALU resolve fail @ {u.op}")

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
  # find the real frame in the file that has the UPat, TODO: is there a better way to do this?
  while frm.f_back is not None and frm.f_back.f_code.co_filename.split("/")[-1] in {"ops.py", "uopgraph.py", "schedule.py", "lowerer.py"}:
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> List[str]:
  with open(fn) as f: return f.readlines()

class UPat(MathTrait):
  __slots__ = ["op", "dtype", "arg", "name", "src", "_any"]
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

    self.allowed_len: int = -1 if allow_any_len or isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = set((pp.op[0], pp.arg) for pp in upat_match if pp.op is not None and len(pp.op) == 1)

  @staticmethod
  def any(*src): return UPatAny(src=src)

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None, vec=True):
    return UPat((UOps.CONST, UOps.VCONST) if vec else UOps.CONST, dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def const(dtype:Optional[DType], b:ConstType): return UPat(UOps.CONST, dtype=dtype, arg=b)

  # copied from UOp
  def cast(self, dtype=None): return UPat(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return UPat(UOps.BITCAST, dtype, (self,))
  def gep(self, i:int): return UPat(UOps.GEP, None, (self,), (i,))
  @staticmethod
  def load(*src:UPat, dtype:Optional[DType]=None): return UPat(UOps.LOAD, dtype, src)
  @staticmethod
  def store(*src:UPat): return UPat(UOps.STORE, dtypes.void, src)

  def const_like(self, b:ConstType|Variable|Tuple[ConstType]): return UPat.const(self.dtype, b)
  def alu(self, arg, *src:UPat):
    asrc = (self,)+src
    return UPat(UOps.ALU, None if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else asrc[-1].dtype, list(asrc) if arg in COMMUTATIVE else asrc, arg)

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

  def match(self:UPat, uop:UOp, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
    if (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
      (self.dtype is not None and uop.dtype not in self.dtype and uop.dtype.scalar() not in self.dtype) or \
      (self.arg is not None and self.arg != uop.arg) or \
      (self.op is not None and uop.op not in self.op) or \
      (self.allowed_len != -1 and len(uop.src) != self.allowed_len): return []
    if self.src is None: return [store]
    res: List[Dict[str, UOp]] = []
    for vp in self.src:
      stores, new_stores = [store.copy()], []
      for uu, vv in zip(uop.src, vp):
        for s in stores: new_stores.extend(vv.match(uu, s))
        stores, new_stores = new_stores, []
      res.extend(stores)
    return res

class UPatAny(UPat):
  def match(self:UPat, uop:UOp, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
    for x in self.src[0]:
      if (match:=x.match(uop, store.copy())): return match
    return []

def deconstruct_function(fxn:Callable) -> Tuple:
  new_globals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
  for co in fxn.__code__.co_consts:
    if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
  # NOTE: optional round trip through pickle!
  assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
  ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
  return pickle.loads(pickle.dumps(ret)) if getenv("TEST_PICKLE") else ret

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    self.patterns = patterns
    # NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
    self.pdict: Dict[Tuple[UOps, Any], List[Tuple[UPat, Callable, Set]]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      tuple_fxn = fxn if isinstance(fxn, tuple) else deconstruct_function(fxn)
      tuple_fxn[1]['__builtins__'] = __builtins__  # NOTE: Python 3.8 requires this for "all" and "len" and friends
      real_fxn = types.FunctionType(*tuple_fxn)
      for uop in p.op: self.pdict.setdefault((uop, p.arg), []).append((p, real_fxn, p.early_reject))

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    ler = set([v for u in uop.src for v in ((u.op, u.arg), (u.op, None))])
    for p,fxn,early_reject in self.pdict.get((uop.op, uop.arg), []) + ([] if uop.arg is None else self.pdict.get((uop.op, None), [])):
      if not early_reject.issubset(ler): continue
      if (matches := p.match(uop, {})) and (ret:=(fxn(ctx, **matches[0]) if ctx is not None else fxn(**matches[0]))) is not None: return ret
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
match_stats:Dict[UPat, List[Union[int, float]]] = dict()
@dataclass(frozen=True)
class TrackedRewriteContext:
  loc: Tuple[str, int]                                                    # location that called graph_rewrite
  sink: UOp                                                               # the sink passed into the rewrite
  rewrites: List[Tuple[UOp, UOp, UPat]] = field(default_factory=list)     # all rewrites of sparents. (before, after, UPat)

rewrite_stack: List[Tuple[Any, List[TrackedRewriteContext]]] = []
contexts: List[Tuple[Any, List[TrackedRewriteContext]]] = []
def track_rewrites(func):
  def __wrapper(self, *args, **kwargs):
    if TRACK_MATCH_STATS >= 2: rewrite_stack.append((self, []))
    try: ret = func(self, *args, **kwargs)
    finally: # NOTE: save everything in the stack
      if TRACK_MATCH_STATS >= 2: contexts.append(rewrite_stack.pop())
    return ret
  return __wrapper

class TrackedPatternMatcher(PatternMatcher):
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    super().__init__(patterns)
    for p,_ in self.patterns:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

  def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    ret = None
    ler = set([v for u in uop.src for v in ((u.op, u.arg), (u.op, None))])
    for p,fxn,early_reject in self.pdict.get((uop.op, uop.arg), []) + ([] if uop.arg is None else self.pdict.get((uop.op, None), [])):
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      if (matches := p.match(uop, {})) and (ret:=(fxn(ctx, **matches[0]) if ctx is not None else fxn(**matches[0]))) is not None:
        match_stats[p][0] += 1
        match_stats[p][2] += (et:=time.perf_counter()-st)
        match_stats[p][3] += et
        if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", p.printable())
        if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0 and isinstance(ret, UOp): rewrite_stack[-1][1][-1].rewrites.append((uop, ret, p))
        return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open("/tmp/rewrites.pkl", "wb") as f:
        print(f"rewrote {len(contexts)} graphs and applied {sum(len(r.rewrites) for _,x in contexts for r in x)} rules, saved to /tmp/rewrites.pkl")
        pickle.dump(contexts, f)
    if getenv("VIZ"):
      os.environ["VIZ"] = "0"
      os.execv(sys.executable, [sys.executable] + [os.path.join(os.path.dirname(__file__), ".", "viz", "serve.py")])
    if getenv("PRINT_MATCH_STATS", 1):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {v[2]*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {ret[2]*1000.:9.2f} ms -- TOTAL")

# *** simple graph rewrite engine ***

class RewriteContext:
  def __init__(self, pm, ctx):
    self.pm: PatternMatcher = pm
    self.ctx = ctx
    self.replace: Dict[UOp, UOp] = {}
  def rewrite(self, n:UOp) -> UOp:
    if (rn := self.replace.get(n)) is not None: return rn
    new_src = tuple(map(self.rewrite, n.src))
    new_n = self.pm.rewrite(n, self.ctx) if new_src == n.src else UOp(n.op, n.dtype, new_src, n.arg)
    self.replace[n] = ret = n if new_n is None else self.rewrite(new_n)
    return ret

def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None) -> UOp:
  if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0:
    rewrite_stack[-1][1].append(TrackedRewriteContext(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink))
  return RewriteContext(pm, ctx).rewrite(sink)

# ***** uop type spec *****

# this is the matcher for the final rendered UOps
# matcher functions returns True or False (or None to not match)
spec = PatternMatcher([
  (UPat(UOps.DEFINE_GLOBAL, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and not x.dtype.local),
  (UPat(UOps.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.local),
  (UPat(UOps.DEFINE_ACC, src=(UPat.var("c"),), name="x", allow_any_len=True),
   lambda x,c: all(y.op is UOps.RANGE for y in x.src[1:]) and c.dtype == x.dtype),
  (UPat(UOps.DEFINE_VAR, src=(), name="x"), lambda x: isinstance(x.arg[1], int) and isinstance(x.arg[2], int)),

  (UPat(UOps.RANGE, src=(UPat(name="x"), UPat(name="y")), name="rng"), lambda rng,x,y: rng.dtype == x.dtype == y.dtype),
  (UPat(UOps.SPECIAL, src=()), lambda: True),

  # no pyint allowed here!
  (UPat(UOps.ALU, dtype=dtypes.pyint), lambda: False),

  # TODO: confirm the args of both of these are shapetrackers
  (UPat(UOps.VIEW, src=()), lambda: True),
  (UPat(UOps.VIEW, src=(UPat(),)), lambda: True),

  (UPat(UOps.VALID, dtypes.bool, (UPat(UOps.VIEW),)), lambda: True),
  (UPat(UOps.CONST, name="x"), lambda x: x.dtype == x.dtype.scalar() and (type(x.arg) is type(dtypes.as_const(x.arg, x.dtype)))),

  # early LOAD has a <buf, shapetracker, store?>
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(UOps.VIEW))), lambda: True),
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(UOps.VIEW), UPat(UOps.STORE))), lambda: True),

  # LOAD takes a <buf, idx, alt?, gate?, barrier?>
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat())), lambda: True),
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(), UPat((UOps.IF, UOps.BARRIER)))), lambda: True),
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(), UPat(name="alt"), UPat(dtype=dtypes.bool)), name="ld"),
   lambda ld,alt: ld.dtype == alt.dtype),

  # STORE takes a <buf, idx, val, gate?>
  (UPat(UOps.STORE, dtype=dtypes.void, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(), UPat())), lambda: True),
  (UPat(UOps.STORE, dtype=dtypes.void, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(), UPat(), UPat(dtype=dtypes.bool))), lambda: True),
  (UPat(UOps.STORE, dtype=dtypes.void, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)), UPat(), UPat(), UPat(UOps.IF))), lambda: True),

  # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(UOps.ALU, name="w", src=(UPat(dtype=dtypes.bool), UPat(name="x"), UPat(name="y")), arg=TernaryOps.WHERE),
   lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat(UOps.ALU, dtype=dtypes.bool, src=(UPat(name="x"), UPat(name="y")), arg=BinaryOps.CMPLT), lambda x,y: x.dtype == y.dtype),
  (UPat(UOps.ALU, dtype=dtypes.bool, src=(UPat(name="x"), UPat(name="y")), arg=BinaryOps.CMPNE), lambda x,y: x.dtype == y.dtype),
  # and SHL/SHR, the shift distance is an int
  (UPat(UOps.ALU, src=(UPat(name="x"), UPat()), name="alu", arg=BinaryOps.SHL), lambda alu,x: alu.dtype == x.dtype),
  (UPat(UOps.ALU, src=(UPat(name="x"), UPat()), name="alu", arg=BinaryOps.SHR), lambda alu,x: alu.dtype == x.dtype),
  (UPat(UOps.ALU, arg=BinaryOps.IDIV, name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(UOps.ALU, name="x"), lambda x: all(x.dtype == y.dtype for y in x.src)),

  (UPat(UOps.ASSIGN, src=(UPat((UOps.DEFINE_ACC, UOps.DEFINE_GLOBAL)), UPat())), lambda: True),
  (UPat(UOps.ENDRANGE, dtype=dtypes.void, src=(UPat(UOps.RANGE),)), lambda: True),

  # all WMMA has 3 args, <x, w, acc>
  (UPat(UOps.WMMA, src=(UPat(), UPat(), UPat())), lambda: True),
  (UPat(UOps.CONTRACT, name="x"), lambda x: x.dtype.count == prod(y[1] for y in x.arg)),
  (UPat(UOps.EXPAND, name="x"), lambda x: x.src[0].dtype.count == prod(y[1] for y in x.arg)),

  # if has a <gate, barrier?>
  (UPat(UOps.IF, dtype=dtypes.void, src=(UPat(),)), lambda: True),
  (UPat(UOps.IF, dtype=dtypes.void, src=(UPat(), UPat(UOps.BARRIER))), lambda: True),
  (UPat(UOps.ENDIF, dtype=dtypes.void, src=(UPat(UOps.IF),)), lambda: True),

  (UPat(UOps.REDUCE_AXIS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in BinaryOps),
  (UPat(UOps.GEP, src=(UPat(name="src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),
  (UPat(UOps.VECTORIZE, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.count and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat((UOps.BITCAST, UOps.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None and x.dtype.count == 1),
  (UPat(UOps.BARRIER, dtypes.void, src=UPat(UOps.STORE, src=(UPat(UOps.DEFINE_LOCAL),), allow_any_len=True)), lambda: True),

  # NOTE: for testing, we let sinks be anything
  #(UPat(UOps.SINK, src=UPat(UOps.STORE)), lambda: True),
  (UPat(UOps.SINK, dtypes.void), lambda: True),
  (UPat(UOps.NOOP), lambda: True),

  # PTX LOAD/STORE
  (UPat((UOps.LOAD, UOps.STORE), src=(UPat(dtype=dtypes.int64),), allow_any_len=True), lambda: True),
  (UPat(UOps.BARRIER, dtypes.void, src=UPat(UOps.STORE, src=(UPat(dtype=dtypes.int64),), allow_any_len=True)), lambda: True),
])

def type_verify(uops:List[UOp]):
  for i,u in enumerate(uops):
    chk = cast(bool, spec.rewrite(u))
    assert chk is True, f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}"

# *** most of symbolic lives here now ***

def _get_chain(x:UOp, sep:BinaryOps):
  if x.op is UOps.ALU and x.arg is sep:
    for s in x.src: yield from _get_chain(s, sep)
  else: yield x

def mod_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x % c, None means no change

  # simple cancel mod case
  if 0 < c and 0 <= x.vmin and (quotient:=x.vmin//c) == x.vmax//c: return x-quotient*c

  remainder, something_changed = [], False
  for u in _get_chain(x, BinaryOps.ADD):
    if (factor:=u.const_factor())%c != factor:
      divides = u.divides(factor)*(factor%c)
      assert divides is not None
      remainder.append(divides)
      something_changed = True
    elif u.op is UOps.ALU and u.arg is BinaryOps.MOD and (s1:=u.src[1]).op is UOps.CONST and s1.arg%c == 0:
      remainder.append(u.src[0])
      something_changed = True
    else: remainder.append(u)
  if not something_changed: return None
  return functools.reduce(operator.add, remainder)%c if remainder else x.const_like(0)

def div_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x // c, None means no change

  # simple cancel div case
  if 0 <= x.vmin and x.vmax < c: return x.const_like(0)

  quotient, remainder, rem_const, something_changed, gcd, divisor = [], [], 0, False, c, 1
  for u in _get_chain(x, BinaryOps.ADD):
    if u.op is UOps.CONST:
      # add all const together first
      if rem_const != 0: something_changed = True
      rem_const += u.arg
    elif (factor:=u.const_factor())%c == 0:
      if factor:
        divides = u.divides(c)
        assert divides is not None
        quotient.append(divides)
      something_changed = True
    else:
      # divisor is the smallest common divisor of all MULs
      if u.op is UOps.ALU and u.arg is BinaryOps.MUL and factor > 1 and c % factor == 0 and (divisor == 1 or divisor > factor): divisor = factor
      remainder.append(u)
      gcd = math.gcd(gcd, factor)

  # handle the const
  if rem_const%c != rem_const:
    something_changed = True
    quotient.append(x.const_like(rem_const//c))
    rem_const = rem_const%c
  if rem_const != 0: remainder.append(x.const_like(rem_const))

  # x // c -> quotient + (remainder // div) // (c // div)
  div = gcd if gcd > 1 else divisor

  if not something_changed: return newx//(c//div) if 1 < div < c and (newx:=div_folding(x, div)) is not None else None
  rem:Optional[UOp] = functools.reduce(operator.add, remainder) if remainder else None
  quo:Optional[UOp] = functools.reduce(operator.add, quotient) if quotient else None
  if quo is None: return x.const_like(0) if rem is None else cast(UOp, div_folding(rem, div))//(c//div)
  return quo if rem is None else cast(UOp, div_folding(rem, div))//(c//div)+quo

def lt_folding(x:UOp, c:int) -> Optional[UOp]:
  return cast(UOp, x.divides(g)).lt(c//g) if ((g:=math.gcd(x.const_factor(), c)) > 1) else None

def fold_unrolled_divs(divs:UOp):
  # div pattern in unrolled arange
  # example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  add_chain, denominator, seen_const, ans = list(_get_chain(divs, BinaryOps.ADD)), None, [], None
  for u in add_chain:
    if not (u.op is UOps.ALU and u.arg is BinaryOps.IDIV and u.src[1].op is UOps.CONST): return None
    if denominator is None: denominator = u.src[1].arg
    if denominator != u.src[1].arg: return None
    # assumed CONST is the last of an ADD
    if (s0:=u.src[0]).op is UOps.ALU and s0.arg is BinaryOps.ADD and s0.src[1].op is UOps.CONST and s0.src[1].op is UOps.CONST:
      seen_const.append(s0.src[1].arg)
      s0 = s0.src[0]
    else: seen_const.append(0)
    if ans is None: ans = s0
    if ans is not s0: return None
  if denominator is None: return None
  # the first (denominator-len(seen_const)) terms may have been folded to 0 already
  for i in range(denominator-len(seen_const)):
    if ans is not None and 0 <= ans.vmin and ans.vmax + i < denominator: seen_const.append(i)
  return ans if ans is not None and sorted(seen_const)==list(range(denominator)) else None

def is_irreducible(u:UOp): return u.op in (UOps.DEFINE_VAR, UOps.SPECIAL, UOps.RANGE)

def canonicalize_simplex(X:UOp) -> Optional[UOp]:
  # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  # returns x0 + x1 + ... in such case, or None if not
  changed, ret = False, []
  for u in _get_chain(X, BinaryOps.ADD):
    # assumed the const is the last src of MUL
    if u.op is UOps.ALU and u.arg is BinaryOps.MUL and u.src[1].op is UOps.CONST and u.src[1].arg > 0:
      changed = True
      u = u.src[0]
    if not (is_irreducible(u) and u.vmin >= 0): return None
    ret.append(u)
  return functools.reduce(operator.add, ret) if changed else None

symbolic = PatternMatcher([
  # bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat.var('x', dtype=dtypes.bool) * UPat.var('y'), lambda x,y: x&y),
  (UPat.var('x', dtype=dtypes.bool) + UPat.var('y'), lambda x,y: x|y),
  (UPat.var('x', dtype=dtypes.bool).max(UPat.var('y')), lambda x,y: x|y),
  # ** self folding **
  (UPat.var("x") + 0, lambda x: x),    # x+0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x") // -1, lambda x: -x), # x//-1 -> -x
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)), # x/x -> 1
  ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), lambda x,x2: x), # (x*x2)/x2 -> x
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c", vec=False), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c", vec=False), lambda x,c: c if c.arg else x),
  (UPat.var("x").max(UPat.var("x")), lambda x: x),
  ((UPat.var("x") & UPat.var("x")), lambda x: x),
  ((UPat.var("x") | UPat.var("x")), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  # group like
  ((UPat.var("x") + UPat.var("y")) + UPat.var("x") * UPat.cvar("c"), lambda x,y,c: (x+x*c)+y),
  # ** combine terms **
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1)), # (x*c0)+(x*c1) -> x*(c0+c1)
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), lambda x,c: x*(c+1)), # (x+x*c)-> x*(c+1)
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  ((UPat.var("x") / UPat.var("x2")) / UPat.var("x3"), lambda x,x2,x3: x/(x2*x3)), # (x/x2)/x3 -> x/(x2*x3)
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: UOp.const(dtypes.bool.vec(x.dtype.count), False)), # x < x -> False
  (UPat.var("x", dtype=dtypes.ints) != UPat.var("x"), lambda x: UOp.const(dtypes.bool.vec(x.dtype.count), False)), # x != x -> False (only ints)
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat((UOps.VCONST, UOps.CONST))),
   lambda root: root.const_like(exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ALU min==max -> CONST (slow!)
  (UPat(UOps.ALU, name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # max folding
  (UPat.max(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else y if x.vmax <= y.vmin else None),
  # ** two stage ALU folding **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.cvar("c2"), lambda x,c1,c2: x+(c1+c2)),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.cvar("c2"), lambda x,c1,c2: x*(c1*c2)),
  ((UPat.var("x") & UPat.cvar("c1")) & UPat.cvar("c2"), lambda x,c1,c2: x&(c1&c2)),
  ((UPat.var("x") | UPat.cvar("c1")) | UPat.cvar("c2"), lambda x,c1,c2: x|(c1|c2)),
  ((UPat.cvar("c0") + UPat.var("x")) < UPat.cvar("c1"), lambda x,c0,c1: x<(c1-c0)),  # c0 + x < c1 -> x < c1 - c0
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), lambda x,c1,c2: x//(c1*c2)), # (x//c1)//c2 -> x//(c1*c2)
  # mod folding
  (UPat.var("x")%UPat.cvar("c")+(UPat.var("x")//UPat.cvar("c"))*UPat.cvar("c"), lambda x,c: x), # (x%c)+(x//c)*c = x
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.ints)).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: x.lt(math.ceil(c1.arg/c0.arg)) if c0.arg > 0 and c1.arg > 0 else None),
  # c0*x<c1 for negative int c0 and non-positive c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x", dtype=dtypes.ints)).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: (-x).lt(-(math.floor(-c1.arg/-c0.arg))) if c0.arg < 0 and c0.arg != -1 and c1.arg <= 0 else None),
  # x//c0<c1 for positive int c0
  ((UPat.var("x", dtype=dtypes.ints)//UPat.cvar("c0", vec=False)).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: x.lt(c1.arg*c0.arg) if c0.arg > 0 else None),
  # mul add lt
  (((UPat.cvar("c0", vec=False)*UPat.var("x"))+UPat.var("x2")).lt(UPat.cvar("c1", vec=False)),
   lambda x,x2,c0,c1: x.lt(c1//c0) if c1.arg % c0.arg == 0 and c0.arg > x2.vmax and x2.vmin >= 0 else None),
  # ** move add consts to end (NOTE: this is still happening before constant folding) **
  (UPat(UOps.ALU, arg=BinaryOps.ADD, src=(UPat.cvar("c1"), UPat.var("x"))), lambda c1,x: x+c1 if x.op not in (UOps.CONST, UOps.VCONST) else None),
  (UPat(UOps.ALU, arg=BinaryOps.ADD, src=(UPat.var("x"), UPat.cvar("c1"))) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
  # ** move mul consts to end (NOTE: this is still happening before constant folding) **
  (UPat(UOps.ALU, arg=BinaryOps.MUL, src=(UPat.cvar("c1"), UPat.var("x"))), lambda c1,x: x*c1 if x.op not in (UOps.CONST, UOps.VCONST) else None),
  (UPat(UOps.ALU, arg=BinaryOps.MUL, src=(UPat.var("x"), UPat.cvar("c1"))) * UPat.var("y"), lambda x,c1,y: (x*y)*c1),
  # *** rules from symbolic ***
  # unrolled arange div folding
  (UPat(UOps.ALU, name="divs", src=[UPat(), UPat(UOps.ALU, arg=BinaryOps.IDIV)], arg=BinaryOps.ADD), fold_unrolled_divs),
  # generic lt folding
  (UPat.var("x", dtypes.sints).lt(UPat.cvar("c", vec=False)), lambda x,c: lt_folding(x, c.arg) if 0 < c.arg else None),
  # canonicalize a simplex with positive coefficients > 0
  # not x < 1 -> X > 0
  (UPat.var("x", dtypes.ints).lt(1).ne(True), lambda x: newx.lt(1).ne(True) if (newx:=canonicalize_simplex(x)) is not None else None),
  # ** div **
  # # div folding
  (UPat.var("x", dtypes.sints) // UPat.cvar("c", vec=False), lambda x,c: newx if 0 < c.arg and (newx:=div_folding(x,c.arg)) is not None else None),
  # ** mod **
  # mod folding
  (UPat.var("x") % UPat.cvar("c", vec=False), lambda x,c: newx if 0 < c.arg and (newx:=mod_folding(x,c.arg)) is not None else None),
])

symbolic_flat = symbolic+PatternMatcher([
  # ** combine terms (opinionated) **
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x", dtypes.ints) + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c),
])

# for debug
renderer = PatternMatcher([
  (UPat(UOps.DEFINE_VAR, name="x"), lambda x: UOp(UOps.NOOP, arg=x.arg[0])),
  (UPat(UOps.CONST, name="x"), lambda x: UOp(UOps.NOOP, arg=str(x.arg))),
  (UPat(UOps.BIND, src=UPat(UOps.NOOP), name="x"), lambda x: x.src[0]),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.ADD, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}+{x.src[1].arg})")),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.MUL, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}*{x.src[1].arg})")),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.IDIV, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}//{x.src[1].arg})")),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.MOD, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}%{x.src[1].arg})")),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.CMPLT, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}<{x.src[1].arg})")),
  (UPat(UOps.ALU, src=UPat(UOps.NOOP), arg=BinaryOps.CMPNE, name="x"), lambda x: UOp(UOps.NOOP, arg=f"({x.src[0].arg}!={x.src[1].arg})")),
])
