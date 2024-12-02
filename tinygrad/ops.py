from __future__ import annotations
from typing import Any, List, Optional, Set, Union, Tuple, Dict, Callable, cast, TYPE_CHECKING, Type, DefaultDict
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect
from enum import auto, IntEnum, Enum
from dataclasses import dataclass, field
from collections import defaultdict
from weakref import WeakValueDictionary
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes, DType, truncate
from tinygrad.helpers import ContextVar, prod, getenv, all_same, Context, partition, temp, unwrap, T
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

class SimpleMathTrait:
  # required to implement
  def alu(self:T, arg:Ops, *src) -> T: raise NotImplementedError
  def const_like(self:T, b:ConstLike) -> T: raise NotImplementedError

  # great functions you get!
  def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
  def _binop(self, op, x, reverse): return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  def logical_not(self): return self.ne(True)
  def neg(self):
    dtype: Optional[DType] = getattr(self, 'dtype', None)
    assert dtype is not None, "MathTraits __neg__ requires a dtype"
    return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
  def add(self, x, reverse=False): return self._binop(Ops.ADD, x, reverse)
  def mul(self, x, reverse=False): return self._binop(Ops.MUL, x, reverse)
  def bitwise_and(self, x, reverse=False): return self._binop(Ops.AND, x, reverse)
  def bitwise_or(self, x, reverse=False): return self._binop(Ops.OR, x, reverse)
  def xor(self, x, reverse=False): return self._binop(Ops.XOR, x, reverse)
  def idiv(self, x, reverse=False): return self._binop(Ops.IDIV, x, reverse)
  def sub(self, x, reverse=False): return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))
  def div(self, x, reverse=False): return (self.ufix(x)*self.alu(Ops.RECIP)) if reverse else (self*self.ufix(x).alu(Ops.RECIP))

  def __neg__(self): return self.neg()

  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(x)
  def __mul__(self, x): return self.mul(x)
  def __truediv__(self, x): return self.div(x)
  def __floordiv__(self, x): return self.idiv(x)
  def __and__(self, x): return self.bitwise_and(x)
  def __or__(self, x): return self.bitwise_or(x)
  def __xor__(self, x): return self.xor(x)

  def __radd__(self, x): return self.add(x, True)
  def __rsub__(self, x): return self.sub(x, True)
  def __rmul__(self, x): return self.mul(x, True)
  def __rtruediv__(self, x): return self.div(x, True)
  def __rfloordiv__(self, x): return self.idiv(x, True)
  def __rand__(self, x): return self.bitwise_and(x, True)
  def __ror__(self, x): return self.bitwise_or(x, True)
  def __rxor__(self, x): return self.xor(x, True)

  def lt(self, x): return self.alu(Ops.CMPLT, self.ufix(x))
  def gt(self, x): return self.ufix(x).alu(Ops.CMPLT, self)
  def ne(self, x): return self.alu(Ops.CMPNE, self.ufix(x))
  def ge(self, x): return self.lt(x).logical_not()
  def le(self, x): return self.gt(x).logical_not()
  def eq(self, x): return self.ne(x).logical_not()

  def __lt__(self, x): return self.lt(x)
  def __gt__(self, x): return self.gt(x)
  def __ne__(self, x): return self.ne(x)
  def __ge__(self, x): return self.ge(x)
  def __le__(self, x): return self.le(x)
  # NOTE: __eq__ isn't overridden, and means the same thing as is by default

class MathTrait(SimpleMathTrait):
  # TODO: move to Tensor when new backward is done
  def lshift(self, x, reverse=False): return self._binop(Ops.SHL, x, reverse)
  def rshift(self, x, reverse=False): return self._binop(Ops.SHR, x, reverse)
  def __lshift__(self, x): return self.lshift(x)
  def __rshift__(self, x): return self.rshift(x)
  def __rlshift__(self, x): return self.lshift(x, True)
  def __rrshift__(self, x): return self.rshift(x, True)

  # not in Tensor
  def __mod__(self, x): return self.alu(Ops.MOD, self.ufix(x))
  def __rmod__(self, x): return self.ufix(x).alu(Ops.MOD, self)

  def maximum(self, x): return self.alu(Ops.MAX, self.ufix(x))
  def minimum(self, x): return -(-self).maximum(-x)
  def where(self, x, y): return self.alu(Ops.WHERE, x, x.ufix(y))
  def threefry(self, seed): return self.alu(Ops.THREEFRY, seed)
  def reciprocal(self): return self.alu(Ops.RECIP)
  def sqrt(self): return self.alu(Ops.SQRT)
  def sin(self): return self.alu(Ops.SIN)
  def log2(self): return self.alu(Ops.LOG2)
  def exp2(self): return self.alu(Ops.EXP2)

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  # uops that aren't rendered
  SINK = auto()
  CONTIGUOUS = auto()
  PRELOAD = auto()

  # MetaOps
  COPY = auto()
  EMPTY = auto()
  BUFFER_VIEW = auto()

  # blocks in linearizer
  BLOCK = auto(); BLOCKSTART = auto(); BLOCKFORK = auto(); BLOCKEND = auto()  # noqa: E702

  EXPAND = auto()
  CONTRACT = auto()
  VIEW = auto()
  DEFINE_GLOBAL = auto()
  BUFFER = auto()
  DEFINE_VAR = auto()
  DEFINE_LOCAL = auto()
  DEFINE_ACC = auto()
  VALID = auto()
  SPECIAL = auto()
  NOOP = auto()

  # reduce
  REDUCE_AXIS = auto()

  # helper ops
  GEP = auto()
  VECTORIZE = auto()

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702

  # loads before math
  LOAD = auto()

  # math ops
  WMMA = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto(); SUB = auto(); FDIV = auto() # noqa: E702

  # TernaryOps
  WHERE = auto(); MULACC = auto() # noqa: E702

  # assignment ops
  STORE = auto()
  ASSIGN = auto()
  BIND = auto()

  # late INDEX
  INDEX = auto()

  # control flow ops
  BARRIER = auto()
  RANGE = auto()
  IF = auto()

  # ops that are not graph nodes
  ENDRANGE = auto()
  ENDIF = auto()

  # consts last!
  VCONST = auto()
  CONST = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY,
            Ops.SUB, Ops.FDIV}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}

  # meta ops
  Meta = {Ops.COPY, Ops.EMPTY, Ops.BUFFER_VIEW}
  Buffer = {Ops.LOAD, Ops.PRELOAD, Ops.STORE, Ops.VALID}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV}

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:Ops, dt:DType) -> ConstType: return dtypes.as_const({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dtypes.min(dt)}[op], dt)

def can_pad(u:UOp, edges:Dict[UOp, UOp], visisted:Set[UOp]) -> bool:
  if u.op in GroupOp.UnsafePad: return False
  if (len(u.src) == 2 and u.src[0] in edges) or u in visisted: return True
  visisted.add(u)
  return all(can_pad(x.base, edges, visisted) for x in u.src)

END_FOR_UOP = {Ops.IF:(Ops.STORE, Ops.ENDIF), Ops.RANGE:(Ops.ASSIGN, Ops.ENDRANGE)}

# With True as the default, this matches the old symbolic behavior
def resolve(x, default:bool=True):
  if not isinstance(x, UOp): return bool(x)
  assert x.dtype is dtypes.bool, "UOp in resolve must be bool"
  # NOTE: generating the text for the exception is expensive, so we do this
  return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default

# smax/smin are replacements for max/min that preserve symbolic
def _suop(lst, uop_fxn, python_fxn):
  max_uop, max_num = partition(lst, lambda x: isinstance(x, UOp))
  if len(max_uop): return functools.reduce(uop_fxn, (max_uop + [python_fxn(max_num)]) if len(max_num) else max_uop).ssimplify()
  return python_fxn(max_num)
def smax(*lst): return _suop(lst[0] if isinstance(lst[0], (tuple, list)) else lst, UOp.maximum, max)
def smin(*lst): return _suop(lst[0] if isinstance(lst[0], (tuple, list)) else lst, UOp.minimum, min)

def ssimplify(uop): return uop.ssimplify() if isinstance(uop, UOp) else uop
def sym_infer(uop: Union[UOp, int], var_vals: Dict[UOp, int]) -> int: return uop.sym_infer(var_vals) if isinstance(uop, UOp) else uop

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

class UOpMetaClass(type):
  ucache:WeakValueDictionary[Tuple, UOp] = WeakValueDictionary()
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:Tuple[UOp,...]=tuple(), arg:Any=None):
    if (ret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg), None)) is not None: return ret
    UOpMetaClass.ucache[key] = ret = super().__call__(op, dtype, src, arg)
    return ret

class UOp(MathTrait, metaclass=UOpMetaClass):
  __slots__ = ["op", "dtype", "src", "arg"]
  def __init__(self, op:Ops, dtype:DType=dtypes.void, src: Tuple[UOp,...]=tuple(), arg:Any=None):
    # TODO: instant check rules here make debugging easier
    self.op, self.dtype, self.src, self.arg = op, dtype, src, arg
  def __reduce__(self): return UOp, (self.op, self.dtype, self.src, self.arg)
  def replace(self, **kwargs) -> UOp:
    for k in kwargs: assert k in self.__slots__, f"unkown replace arg, expected one of {self.__slots__}, got {k}"
    new_args = (kwargs.get("op", self.op), kwargs.get("dtype", self.dtype), kwargs.get("src", self.src), kwargs.get("arg", self.arg))
    if (self.op, self.dtype, self.src, self.arg) == new_args: return self
    return UOp(*new_args)
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is Ops.REDUCE_AXIS else self.arg

  @functools.cached_property
  def toposort(self) -> Dict[UOp, None]:
    nodes: Dict[UOp, None] = {}
    # NOTE: this is a lot faster than the comprehension in parents
    for parent in self.src: nodes.update(parent.toposort)
    nodes[self] = None
    return nodes

  @functools.cached_property
  def tuplize(self:UOp) -> Tuple[int, Any, Optional[DType], Tuple]:
    return (self.op.value, self.arg, self.dtype, tuple(x.tuplize for x in self.src))

  # *** uop shape stuff ***

  @property
  def has_st(self) -> bool: return self.op not in {Ops.DEFINE_LOCAL, Ops.DEFINE_GLOBAL, Ops.BUFFER, Ops.CONST, Ops.DEFINE_VAR}
  @functools.cached_property
  def st(self) -> Optional[ShapeTracker]:
    if self.op is Ops.VIEW: return self.arg
    # buffer ops can have a non contiguous shapetracker
    if self.op in GroupOp.Buffer and len(src_sts:=[unwrap(x.st) for x in self.src if x.op is Ops.VIEW]) != 0: return src_sts[0]
    if len(src_sts:=[x.st for x in self.src if x.st is not None]) == 0: return None
    assert all_same([x.shape for x in src_sts]), f"UOp parents must have the same shape {self} {[x.shape for x in src_sts]}"
    # all other ops have a contiguous shapetracker
    from tinygrad.shape.shapetracker import ShapeTracker
    return ShapeTracker.from_shape(src_sts[0].reduce(self.axis_arg) if self.op is Ops.REDUCE_AXIS else src_sts[0].shape)
  @functools.cached_property
  def full_shape(self) -> Tuple[sint, ...]:
    return self.shape if self.op is Ops.VIEW else tuple(smax(x) for x in zip(*[x.full_shape for x in self.src if x.has_st]))
  @property
  def shape(self) -> Tuple[sint, ...]: return unwrap(self.st).shape
  @property
  def size(self) -> int: return self.arg[1][1] if self.op is Ops.BUFFER else unwrap(self.st).size

  # *** uop evaluation ***

  def simplify(self):
    with Context(TRACK_MATCH_STATS=0):
      return graph_rewrite(self, symbolic)
  def ssimplify(self) -> Union[UOp, ConstType]: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
  def _eval(self, dtype, expected_type:Type[T]) -> T:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    vmin, vmax = (simple_self:=self.simplify())._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return self._eval(dtypes.floats, float)
  def substitute(self, dvars:Dict[UOp, UOp]):
    with Context(TRACK_MATCH_STATS=0):
      return graph_rewrite(self, _substitute, dvars, bottom_up=True)

  # *** uop syntactic sugar ***

  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in GroupOp.Buffer, f"st_arg called on {self.op}"
    ret = self.src[0 if self.op is Ops.VALID else 1]
    assert ret.op is Ops.VIEW, f"st_arg trying to return {ret}"
    return ret.arg
  @property
  def axis_arg(self) -> Tuple[int, ...]:
    assert self.op in {Ops.REDUCE_AXIS, Ops.WMMA}, f"axis_arg called on {self.op}"
    ret = self.arg[1] if self.op is Ops.REDUCE_AXIS else self.arg[7]
    assert isinstance(ret, tuple) and all(isinstance(x, int) for x in ret), f"axis_arg trying to return {ret}"
    return ret
  def sink(self, *srcs:UOp): return UOp(Ops.SINK, dtypes.void, (self,)+srcs)
  def index(self, idx:UOp, valid:Optional[UOp]=None): return UOp(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def const_like(self, b:ConstLike): return UOp.const(self.dtype, b)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(Ops.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType): return UOp(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return UOp(Ops.BITCAST, dtype, (self,))
  def gep(self, i:Union[Tuple[int, ...], int]):
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is Ops.VECTORIZE: return self.src[i]
      if self.op is Ops.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
      if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    if (self.dtype.vcount == len(i) and i == tuple(range(len(i)))) or self.dtype == dtypes.void: return self
    return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  def load(self, *src:UOp, **kwargs): return UOp(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UOp, **kwargs): return UOp(Ops.STORE, dtypes.void, (self,)+src, **kwargs)
  def alu(self, arg, *src:UOp):
    out_dtype = (self, *src)[-1].dtype
    if arg in {Ops.CMPLT, Ops.CMPNE}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(arg, out_dtype, (self,)+src)
  @staticmethod
  def const(dtype:DType, b:ConstLike):
    if isinstance(b, UOp): return b.unbind()[0] if b.op is Ops.BIND else b
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    return UOp(Ops.VCONST if isinstance(b, tuple) else Ops.CONST, dtype, arg=dtypes.as_const(b, dtype))
  @staticmethod
  def range(dtype:DType, start:ConstType|UOp, end:ConstType|UOp, idx:int):
    return UOp(Ops.RANGE, dtype=dtype, src=(UOp.const(dtype, start) if not isinstance(start, UOp) else start,
                                             UOp.const(dtype, end) if not isinstance(end, UOp) else end), arg=(idx, False))
  def r(self, op:Ops, axis:Tuple[int, ...]): return UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis))
  def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self,x))
  def contiguous(self): return UOp(Ops.CONTIGUOUS, self.dtype, (self,))
  @property
  def is_contiguous_base(self): return self.op is Ops.CONTIGUOUS and not (self.src[0].base.op is Ops.VIEW and len(self.src[0].base.src) == 2)

  # *** from LazyBuffer ***

  @staticmethod
  def const_with_shape(dtype:DType, val:ConstLike, shape:Tuple[sint,...]) -> UOp:
    from tinygrad.shape.shapetracker import ShapeTracker
    return UOp(Ops.VALID, dtypes.bool, (ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape).to_uop(),)).where(UOp.const(dtype, val), 0)

  # *** uop movement ops ***

  @property
  def base(self) -> UOp: return self.src[0] if self.op is Ops.VIEW and len(self.src) == 1 and self.src[0].op is not Ops.BUFFER else self
  def view(self, new_st:ShapeTracker) -> UOp:
    assert self.st is not None and self.base.st is not None, f"must have shape {self}"
    if self.st.size == 0 or (new_st.views[-1].mask is not None and any((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return UOp.const_with_shape(self.dtype, 0, new_st.shape)
    if new_st.contiguous and self.base.st.shape == new_st.shape: return self.base
    return UOp(Ops.VIEW, self.dtype, (self.base,), new_st)
  def reshape(self, arg:Tuple[sint, ...]): return self.view(unwrap(self.st).reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self.view(unwrap(self.st).pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self.view(unwrap(self.st).expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self.view(unwrap(self.st).permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self.view(unwrap(self.st).shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self.view(unwrap(self.st).stride(arg))

  # *** uop Buffer stuff ***

  buffer_num = itertools.count(0)
  @staticmethod
  def new_buffer(device:str, size:int, dtype:DType) -> UOp: return UOp(Ops.BUFFER, dtype.ptr(), (), (next(UOp.buffer_num), (device, size, dtype)))
  @functools.cached_property
  def device(self) -> str: return self.arg[1][0] if self.op is Ops.BUFFER else self.src[0].device
  @property
  def buf_uop(self) -> UOp:
    if self.op is Ops.BUFFER: return self
    assert self.op in {*GroupOp.Buffer, Ops.ASSIGN, Ops.VIEW} and self.src[0].op is Ops.BUFFER, f"buf_uop called on {self.op}"
    return self.src[0]

  # *** uop Variable stuff ***

  @staticmethod
  def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.int):
    assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  @property
  def expr(self):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    return self.arg[0]
  def bind(self, val:int):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    assert self.arg[1] <= val and val <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    return UOp(Ops.BIND, self.dtype, (self, self.const_like(val)))
  def unbind(self) -> Tuple[Variable, int]:
    assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    return self.src[0], self.src[1].arg
  @property
  def val(self) -> int: return self.unbind()[1]
  def vars(self) -> Set[UOp]:
    bound_vars = set([x for x in self.toposort if x.op is Ops.BIND and x.src[0].op is Ops.DEFINE_VAR])
    bound_var_base = set(x.src[0] for x in bound_vars)
    all_vars = set([x for x in self.toposort if x.op is Ops.DEFINE_VAR])
    return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
  def variables(self) -> List[Variable]:
    st_vars: List[Set[Variable]] = [x.st_arg.vars() for x in self.toposort if x.op in GroupOp.Buffer]
    return sorted(set.union(*st_vars, [x.unbind()[0] if x.op is not Ops.DEFINE_VAR else x for x in self.vars()]), key=lambda v: v.arg)

  # *** uop symbolic stuff ***

  def const_factor(self) -> int:
    """largest known int that divides self"""
    if self.op is Ops.CONST: return self.arg
    if self.op is Ops.VCONST: return math.gcd(*self.arg)
    if self.op is Ops.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
    if self.op is Ops.MUL: return self.src[0].arg if self.src[0].op is Ops.CONST else self.src[1].arg if self.src[1].op is Ops.CONST else 1
    return 1
  def divides(self, v) -> Optional[UOp]:
    if v==1: return self
    if self.op is Ops.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    if self.op is Ops.VCONST: return self.const_like(tuple(x//v for x in self.arg)) if all(x%v == 0 for x in self.arg) else None
    if self.op is Ops.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
    if self.op is Ops.MUL:
      if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
      if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> ConstType: return self._min_max[0]
  @property
  def vmax(self) -> ConstType: return self._min_max[1]
  @functools.cached_property
  def _min_max(self) -> Tuple[ConstType, ConstType]:
    if self.op in GroupOp.Binary and not dtypes.is_float(self.dtype):
      (s0_vmin, s0_vmax), (s1_vmin, s1_vmax) = self.src[0]._min_max, self.src[1]._min_max
      if self.op is Ops.ADD: return s0_vmin+s1_vmin, s0_vmax+s1_vmax
      if self.op is Ops.MUL: return min(vals:=(s0_vmin*s1_vmin, s0_vmin*s1_vmax, s0_vmax*s1_vmin, s0_vmax*s1_vmax)), max(vals)
      if self.op is Ops.MOD and s1_vmin > 0: return 0, s1_vmax-1
      if self.op is Ops.IDIV and s1_vmin == s1_vmax:  # min/max are equal in a CONST
        if s1_vmin > 0: return s0_vmin//s1_vmin, s0_vmax//s1_vmin
        if s1_vmin < 0 and s0_vmin >= 0: return -(s0_vmax//-s1_vmin), -(s0_vmin//-s1_vmin)
      if self.op is Ops.MAX: return max(s0_vmin, s1_vmin), max(s0_vmax, s1_vmax)
      if self.op is Ops.CMPLT: return (s0_vmax<s1_vmin, s0_vmin<s1_vmax)
      if self.op is Ops.CMPNE: return ((s0_vmax < s1_vmin) or (s1_vmax < s0_vmin), not (s0_vmin == s0_vmax == s1_vmin == s1_vmax))
      if self.dtype == dtypes.bool:
        if self.op is Ops.OR: return s0_vmin or s1_vmin, s0_vmax or s1_vmax
        if self.op is Ops.AND: return s0_vmin and s1_vmin, s0_vmax and s1_vmax
    # float has NAN issue and we use explicit NAN in transcendental
    if self.op is Ops.WHERE and dtypes.is_int(self.dtype): return min(self.src[1].vmin, self.src[2].vmin), max(self.src[1].vmax, self.src[2].vmax)
    # NOTE: returned UOp is assumed to be CONST
    if self.op is Ops.DEFINE_VAR and self.arg: return self.arg[1], self.arg[2]
    if self.op is Ops.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    if self.op is Ops.BIND: return self.src[0]._min_max # ignore the bound value
    if self.op in {Ops.EXPAND, Ops.VECTORIZE}: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    # TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if self.op is Ops.SPECIAL: return 0, self.arg[1]-1 if isinstance(self.arg[1], int) else dtypes.max(self.dtype)
    if self.op is Ops.CONST: return self.arg, self.arg
    if self.op is Ops.VCONST: return (min(self.arg), max(self.arg))
    return dtypes.min(self.dtype), dtypes.max(self.dtype)

  @functools.cached_property
  def _sym_fxn(self):
    sself = self.simplify()
    varnames = tuple(x.arg[0] for x in sself.toposort if x.op is Ops.DEFINE_VAR)
    # TODO: sanitize varnames, or don't use naked eval while staying fast
    return eval("lambda "+','.join(varnames)+": "+sself.render()), varnames  # pylint: disable=eval-used

  def sym_infer(self, var_vals:Dict[UOp, int]):
    fxn, varnames = self._sym_fxn
    return fxn(**{k.arg[0]:v for k,v in var_vals.items() if k.arg[0] in varnames})

  def render(self, simplify=True) -> str:
    ret = graph_rewrite(self.simplify() if simplify else self, renderer)
    return ret.arg if ret.op is Ops.NOOP else str(ret)

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

python_alu: Dict[Ops, Callable]  = {
  Ops.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, Ops.EXP2: hook_overflow(math.inf, lambda x: 2**x),
  Ops.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, Ops.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  Ops.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan,
  Ops.NEG: operator.neg, Ops.ADD: operator.add, Ops.SUB: operator.sub, Ops.MUL: operator.mul,
  Ops.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], Ops.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else x*math.inf,
  Ops.MAX: max, Ops.CMPNE: operator.ne, Ops.CMPLT: operator.lt, Ops.XOR: operator.xor,
  Ops.OR: operator.or_, Ops.AND: operator.and_, Ops.SHR: operator.rshift, Ops.SHL: operator.lshift,
  Ops.MULACC: lambda x,y,z: (x*y)+z, Ops.WHERE: lambda x,y,z: y if x else z}

def exec_alu(op:Ops, dtype:DType, operands, truncate_output=True):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  alu = python_alu[op](*operands)
  return truncate.get(dtype, lambda x: x)(alu) if truncate_output else alu

# ***** uop helpers *****

def print_uops(uops:List[UOp]):
  for i,u in enumerate(uops):
    formatted_parents = [(uops.index(x) if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):30s} " f"{str(formatted_parents):32s} {u.arg}")

def flops_mem(uops:List[UOp], ignore_indexing=False) -> Tuple[sint, sint]:
  flops: sint = 0
  mem: sint = 0
  mults: sint = 1
  mult_stack: List[sint] = []
  dont_count: Set[UOp] = set()
  if ignore_indexing:
    for u in uops:
      if u.op in {Ops.LOAD, Ops.STORE}:
        dont_count = dont_count.union(u.src[0].toposort)
        if len(u.src) > 2: dont_count = dont_count.union(u.src[2].toposort)
      elif u.op is Ops.IF:
        dont_count = dont_count.union(u.src[0].toposort)
  for u in uops:
    if u.op is Ops.RANGE:
      mult_stack.append(mults)
      mults *= (u.src[1] - u.src[0]).ssimplify()
    elif u.op is Ops.ENDRANGE:
      mults = mult_stack.pop(-1)
    elif u.op is Ops.SPECIAL:
      mults *= u.arg[1] # NOTE: we don't push to the mult_stack here, you can't end these
    elif u.op is Ops.LOAD:
      mem += u.dtype.itemsize * mults
    elif u.op is Ops.STORE:
      mem += u.src[1].dtype.itemsize * mults
    elif u.op in GroupOp.ALU and u not in dont_count:
      flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
    elif u.op is Ops.WMMA and u not in dont_count:
      flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
  return flops, mem

# ***** pattern matcher *****

def get_location() -> Tuple[str, int]:
  frm = sys._getframe(1)
  # find the real frame in the file that has the UPat, TODO: is there a better way to do this?
  while frm.f_back is not None and pathlib.Path(frm.f_back.f_code.co_filename).name in {"ops.py", "uopgraph.py", "schedule.py",
                                                                                        "lowerer.py", "cstyle.py", "linearize.py"}:
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> List[str]:
  with open(fn) as f: return f.readlines()

class UPat(MathTrait):
  __slots__ = ["op", "dtype", "arg", "name", "src"]
  def __init__(self, op:Optional[Union[Ops, Tuple[Ops, ...], Set[Ops]]]=None, dtype:Optional[Union[DType, Tuple[DType, ...]]]=None,
               src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None, arg:Any=None,
               name:Optional[str]=None, allow_any_len:bool=False, location=None, custom_early_reject:Optional[Set[Ops]]=None):
    assert op is None or isinstance(op, Ops) or isinstance(op, tuple) or isinstance(op, set), "op must be Ops or tuple of Ops"
    self.op: Optional[Tuple[Ops, ...]] = (op,) if isinstance(op, Ops) else (tuple(op) if isinstance(op, set) else op)
    self.dtype: Optional[Tuple[DType, ...]] = (dtype,) if isinstance(dtype, DType) else dtype
    self.arg, self.name, self._in_src, self.custom_early_reject = arg, name, src, custom_early_reject
    self.src: Any = None
    assert self.name != "ctx", "UPat can't be named ctx"

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
      self.early_reject = {pp.op[0] for pp in upat_match if pp.op is not None and len(pp.op) == 1}

  def named(self, name:str): return UPat(self.op, self.dtype, self._in_src, self.arg, name, self.allowed_len == -1, self.custom_early_reject)

  @staticmethod
  def any(*src): return UPatAny(src=src)

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[Union[DType, Tuple[DType, ...]]]=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None, vec=True):
    return UPat((Ops.CONST, Ops.VCONST) if vec else Ops.CONST, dtype=dtype, name=name)
  @staticmethod
  def const(dtype:Optional[Union[DType, Tuple[DType, ...]]], b:ConstType): return UPat(Ops.CONST, dtype=dtype, arg=b)

  # copied from UOp
  def index(self, idx:UPat, valid:Optional[UPat]=None): return UPat(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def view(self, st=None, **kwargs): return UPat(Ops.VIEW, self.dtype, (self,), st, **kwargs)
  def cast(self, dtype=None): return UPat(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return UPat(Ops.BITCAST, dtype, (self,))
  def gep(self, i:int): return UPat(Ops.GEP, None, (self,), (i,))
  def load(self, *src:UPat, **kwargs): return UPat(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UPat, **kwargs): return UPat(Ops.STORE, dtypes.void, (self,)+src, **kwargs)
  def assign(self, x:UPat): return UPat(Ops.ASSIGN, self.dtype, (self,x))

  def const_like(self, b:ConstLike): return UPat.const(self.dtype, cast(ConstType, b))
  def alu(self, op:Ops, *src:UPat):
    asrc = (self,)+src
    return UPat(op, dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else asrc[-1].dtype, list(asrc) if op in GroupOp.Commutative else asrc)

  def printable(self:UPat) -> str:
    try: return lines(self.location[0])[self.location[1]-1].strip()
    except FileNotFoundError: return "<missing>"

  def __repr__(self):
    def rep(x):
      form = "UPat(%s, %s, name=%s, dtype=%s, allow_any_len=%s, src=%s)"
      return form % (None if x.op is None else ('(%s)'%', '.join(map(str, x.op))), x.arg, repr(x.name),
        set(x.dtype) if x.dtype else None, x.allowed_len == 0, "[%s]" if x.src and len(x.src)>1 else "(%s)")
    return pretty_print(self, rep, srcfn=lambda x:None if x.src is None else [next(x.src[0])] if isinstance(x.src[0], itertools.repeat) else x.src[0])

  def match(self:UPat, uop:UOp, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
    if (self.op is not None and uop.op not in self.op) or \
       (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
       (self.dtype is not None and uop.dtype not in self.dtype and uop.dtype.scalar() not in self.dtype) or \
       (self.arg is not None and self.arg != uop.arg) or \
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
    ret = []
    for x in self.src[0]:
      if (match:=x.match(uop, store.copy())): ret.extend(match)
    return ret

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
    self.pdict: Dict[Ops, List[Tuple[UPat, Callable, Set, bool]]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      tuple_fxn = fxn if isinstance(fxn, tuple) else deconstruct_function(fxn)
      real_fxn = types.FunctionType(*tuple_fxn)
      for uop in p.op: self.pdict.setdefault(uop, []).append((p, real_fxn, p.early_reject, 'ctx' in inspect.signature(real_fxn).parameters))

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    ler = {u.op for u in uop.src}
    for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
      if not early_reject.issubset(ler): continue
      for match in p.match(uop, {}):
        if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None: return ret
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
match_stats:Dict[UPat, List[Union[int, float]]] = dict()
@dataclass(frozen=True)
class TrackedRewriteContext:
  loc: Tuple[str, int]                                                                              # location that called graph_rewrite
  sink: UOp                                                                                         # the sink passed into the rewrite
  bottom_up: bool
  matches: List[Tuple[UOp, Optional[UOp], Optional[UPat], float]] = field(default_factory=list)     # all matches of sparents

rewrite_stack: List[Tuple[Any, List[TrackedRewriteContext]]] = []
contexts: List[Tuple[Any, List[TrackedRewriteContext]]] = []
_rewrite_cnt: Dict[str, int] = {}
def track_rewrites(named=False):
  def _decorator(func):
    def __wrapper(self, *args, **kwargs):
      if TRACK_MATCH_STATS >= 2:
        if named: _rewrite_cnt[func.__name__] = _rewrite_cnt.setdefault(func.__name__, 0)+1
        rewrite_stack.append((f"{(n:=func.__name__)}_{_rewrite_cnt[n]}" if named else self, []))
      try: ret = func(self, *args, **kwargs)
      finally: # NOTE: save everything in the stack
        if TRACK_MATCH_STATS >= 2: contexts.append(rewrite_stack.pop())
      return ret
    return __wrapper
  return _decorator

class TrackedPatternMatcher(PatternMatcher):
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    super().__init__(patterns)
    for p,_ in self.patterns:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

  def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    ret = None
    ler = {u.op for u in uop.src}
    for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      for match in p.match(uop, {}):
        if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None:
          match_stats[p][0] += 1
          match_stats[p][3] += (et:=time.perf_counter()-st)
          if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", p.printable())
          if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0 and isinstance(ret, UOp): rewrite_stack[-1][1][-1].matches.append((uop, ret, p, et))
          return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0: rewrite_stack[-1][1][-1].matches.append((uop, ret, None, 0))
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open(fn:=temp("rewrites.pkl"), "wb") as f:
        print(f"rewrote {len(contexts)} graphs and matched {sum(len(r.matches) for _,x in contexts for r in x)} times, saved to {fn}")
        pickle.dump(contexts, f)
    if getenv("VIZ"):
      os.environ["VIZ"] = "0"
      os.execv(sys.executable, [sys.executable] + [os.path.join(os.path.dirname(__file__), ".", "viz", "serve.py"), temp("rewrites.pkl")])
    if getenv("PRINT_MATCH_STATS", 1):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")

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
  def bottom_up_rewrite(self, n:UOp) -> UOp:
    if (rn := self.replace.get(n)) is not None: return rn
    new_n: UOp|None = n
    while new_n is not None: last_n, new_n = new_n, self.pm.rewrite(new_n, self.ctx)
    new_src = tuple(map(self.bottom_up_rewrite, last_n.src))
    self.replace[n] = ret = last_n if new_src == last_n.src else self.bottom_up_rewrite(UOp(last_n.op, last_n.dtype, new_src, last_n.arg))
    return ret

def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False) -> UOp:
  if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0:
    rewrite_stack[-1][1].append(TrackedRewriteContext(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink, bottom_up))
  return RewriteContext(pm, ctx).bottom_up_rewrite(sink) if bottom_up else RewriteContext(pm, ctx).rewrite(sink)

# ***** uop type spec *****

# this is the matcher for the final rendered UOps
# matcher functions returns True or False (or None to not match)
spec = PatternMatcher([
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and not x.dtype.local),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.local),
  (UPat(Ops.DEFINE_ACC, src=(UPat.var("c"),), name="x", allow_any_len=True),
   lambda x,c: all(y.op is Ops.RANGE for y in x.src[1:]) and c.dtype == x.dtype),
  (UPat(Ops.DEFINE_VAR, src=(), name="x"), lambda x: isinstance(x.arg[1], int) and isinstance(x.arg[2], int)),

  (UPat(Ops.RANGE, src=(UPat(name="x"), UPat(name="y")), name="rng"), lambda rng,x,y: rng.dtype == x.dtype == y.dtype and isinstance(rng.arg, int)),
  (UPat(Ops.SPECIAL, src=()), lambda: True),

  # TODO: confirm the args of both of these are shapetrackers
  (UPat(Ops.VIEW, dtypes.void, src=()), lambda: True),
  (UPat(Ops.VIEW, src=(UPat.var("src"),), name="x"), lambda x,src: src.op is not Ops.STORE and x.dtype == src.dtype),

  (UPat(Ops.VALID, dtypes.bool, (UPat(Ops.VIEW),)), lambda: True),
  (UPat(Ops.CONST, name="x"), lambda x: x.dtype == x.dtype.scalar() and (type(x.arg) is type(dtypes.as_const(x.arg, x.dtype)))),

  # early LOAD has a <buf, shapetracker, store?>
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW))), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW), UPat(Ops.STORE))), lambda: True),

  # early STORE has a <buf, shapetracker, val>
  (UPat(Ops.STORE, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW), UPat())), lambda: True),

  # **** new style load/store ****

  # INDEX is used in new style load/store
  (UPat(Ops.INDEX, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat())), lambda: True),

  # LOAD takes a <bufidx, alt?, gate?, barrier?>
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)),)), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat((Ops.IF, Ops.BARRIER)))), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(name="alt"), UPat(dtype=dtypes.bool)), name="ld"), lambda ld,alt: ld.dtype == alt.dtype),

  # STORE takes a <bufidx, val, gate?>
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat())), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(dtype=dtypes.bool))), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(Ops.IF))), lambda: True),

  # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat(name="x"), UPat(name="y"))), lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat((Ops.CMPLT, Ops.CMPNE), dtype=dtypes.bool, src=(UPat(name="x"), UPat(name="y"))), lambda x,y: x.dtype == y.dtype),
  # and SHL/SHR, the shift distance can be an int
  (UPat((Ops.SHL, Ops.SHR), src=(UPat(name="x"), UPat(name="y")), name="a"), lambda a,x,y: a.dtype == x.dtype and y.dtype in (x.dtype, dtypes.uint)),
  (UPat(Ops.IDIV, name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(x.dtype == y.dtype for y in x.src)),

  (UPat(Ops.ASSIGN, src=(UPat((Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL)), UPat())), lambda: True),
  (UPat(Ops.ENDRANGE, dtype=dtypes.void, src=(UPat(Ops.RANGE),)), lambda: True),

  # all WMMA has 3 args, <x, w, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat())), lambda: True),
  (UPat(Ops.CONTRACT, name="x"), lambda x: x.dtype.count == prod(y[1] for y in x.arg)),
  (UPat(Ops.EXPAND, name="x"), lambda x: x.src[0].dtype.count == prod(y[1] for y in x.arg)),

  # if has a <gate, barrier?>
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(),)), lambda: True),
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(), UPat(Ops.BARRIER))), lambda: True),
  (UPat(Ops.ENDIF, dtype=dtypes.void, src=(UPat(Ops.IF),)), lambda: True),

  (UPat(Ops.REDUCE_AXIS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in {Ops.ADD, Ops.MUL, Ops.MAX}),
  (UPat(Ops.GEP, src=(UPat(name="src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),
  (UPat(Ops.VECTORIZE, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.count and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None),
  (UPat(Ops.BARRIER, dtypes.void, src=UPat(Ops.STORE, allow_any_len=True)), lambda: True), # NOTE: all pointers must be local

  # NOTE: for testing, we let sinks be anything
  #(UPat(UOps.SINK, src=UPat(UOps.STORE)), lambda: True),
  (UPat(Ops.SINK, dtypes.void), lambda: True),
  (UPat(Ops.NOOP), lambda: True),

  # PTX LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(dtype=dtypes.int64),), allow_any_len=True), lambda: True),
  (UPat(Ops.BARRIER, dtypes.void, src=UPat(Ops.STORE, src=(UPat(dtype=dtypes.int64),), allow_any_len=True)), lambda: True),
])

def type_verify(uops:List[UOp]):
  for i,u in enumerate(uops):
    if not spec.rewrite(u):
      print_uops(uops)
      raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}")

# *** uop helpers ***

def cast_float_to_bf16(x: UOp) -> UOp:
  assert x.dtype == dtypes.float, "cast float -> bf16 must start with float"
  x = x.bitcast(dtypes.uint)
  x = (-x & 0x7f800000).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).where((x | 0x10000), x))
  return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

# *** most of symbolic lives here now ***

def split_uop(x:UOp, sep:Ops):
  if x.op is sep:
    for s in x.src: yield from split_uop(s, sep)
  else: yield x

def mod_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x % c, None means no change

  # simple cancel mod case
  if 0 < c and 0 <= x.vmin and (quotient:=x.vmin//c) == x.vmax//c: return x-quotient*c

  terms, rem_const, something_changed, offset, gcd = [], 0, False, 0, c
  for u in split_uop(x, Ops.ADD):
    factor = u.const_factor()
    e: UOp = u.divides(factor)
    if (new_factor:=factor%c) != factor: something_changed = True
    elif u.op is Ops.MOD and (s1:=u.src[1]).op is Ops.CONST and s1.arg%c == 0:
      e = u.src[0]
      something_changed = True
    offset += new_factor * e.vmin
    if u.op is Ops.CONST: rem_const += new_factor
    else:
      gcd = math.gcd(factor, gcd)
      terms.append((new_factor, e))

  match terms:  # cases like (x[4-5] + 3) % 4 -> -3*x[4-5]+15
    case [(f, e)] if e.vmax-e.vmin == 1: return ((offset+f)%c - offset%c)*(e - e.vmin) + offset%c

  # cases like (3+3x[0-3])%4 -> 3-x[0-3]
  lbound = ubound = offset = offset % c
  for (f, e) in terms:
    if f > c//2:
      if (lbound := lbound + (f-c)*(e.vmax-e.vmin)) < 0: break
    elif (ubound := ubound + f*(e.vmax-e.vmin)) >= c: break
  else: # we have found factors such that vmin/vmax of the final expression is between 0 and c, we can remove the mod
    return functools.reduce(lambda r, t: r + min(t[0], t[0]-c, key=abs)*(t[1]-t[1].vmin), terms, x.const_like(offset))

  if not something_changed and gcd==1: return None
  return gcd*(functools.reduce(lambda r, t: r + t[0]//gcd * t[1], terms, x.const_like(rem_const//gcd)) % (c//gcd)) + rem_const%gcd

def div_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x // c, None means no change

  # simple cancel div case
  if 0 <= x.vmin and x.vmax < c: return x.const_like(0)

  quotient, remainder, rem_const, something_changed, gcd, divisor = [], [], 0, False, c, 1
  for u in split_uop(x, Ops.ADD):
    if u.op is Ops.CONST:
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
      if u.op is Ops.MUL and factor > 1 and c % factor == 0 and (divisor == 1 or divisor > factor): divisor = factor
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
  p, np = partition(split_uop(x, Ops.ADD), lambda u: u.const_factor() == 1)
  if np and (d:=math.gcd(*[u.const_factor() for u in np], c)) > 1 and 0 <= sum(u.vmin for u in p) and sum(u.vmax for u in p) < d:
    return cast(UOp, functools.reduce(operator.add, np).divides(d)).lt(c//d)
  return None

def fold_unrolled_divs(divs:UOp):
  # div pattern in unrolled arange
  # example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  add_chain, denominator, seen_const, ans = list(split_uop(divs, Ops.ADD)), None, [], None
  for u in add_chain:
    if not (u.op is Ops.IDIV and u.src[1].op is Ops.CONST): return None
    if denominator is None: denominator = u.src[1].arg
    if denominator != u.src[1].arg: return None
    # assumed CONST is the last of an ADD
    if (s0:=u.src[0]).op is Ops.ADD and s0.src[1].op is Ops.CONST and s0.src[1].op is Ops.CONST:
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

def canonicalize_simplex(X:UOp) -> Optional[UOp]:
  # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  # returns x0 + x1 + ... in such case, or None if not
  changed, ret = False, []
  for u in split_uop(X, Ops.ADD):
    # assumed the const is the last src of MUL
    if u.op is Ops.MUL and u.src[1].op is Ops.CONST and u.src[1].arg > 0:
      changed = True
      u = u.src[0]
    if not (u.op in GroupOp.Irreducible and u.vmin >= 0): return None
    ret.append(u)
  return functools.reduce(operator.add, ret) if changed else None

def is_increasing(f:UOp) -> bool:
  # is f a monotonically increasing function regards its input
  if f.op in GroupOp.Irreducible: return True
  if f.op is Ops.ADD: return is_increasing(f.src[0]) and is_increasing(f.src[1])
  if f.op in (Ops.MUL, Ops.IDIV) and f.src[1].op is Ops.CONST and f.src[1].arg >= 0: return is_increasing(f.src[0])
  return False  # False if not sure

def parse_valid(valid:UOp) -> Tuple[UOp, bool, int]:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  # (X < c).ne(True) -> X >= c
  if valid.op is Ops.CMPNE and valid.src[1].op is Ops.CONST and valid.src[1].arg == 1 and \
    (s0:=valid.src[0]).op is Ops.CMPLT and s0.src[1].op is Ops.CONST: return s0.src[0], False, s0.src[1].arg
  # X < c -> X <= c-1
  if valid.op is Ops.CMPLT and valid.src[1].op is Ops.CONST: return valid.src[0], True, valid.src[1].arg-1
  raise ValueError(f"not able to parse {valid=}")

def uop_given_valid(valid:UOp, uop:UOp) -> Optional[UOp]:
  # return None if valid is always False, otherwise the simplified uop (might be the same as input)

  # first, parse valid into {expr: (lower_bound, upper_bound)}
  bounds:DefaultDict[UOp, List[Optional[ConstType]]] = defaultdict(lambda: [None, None])
  for stmt in split_uop(valid, Ops.AND):
    try: expr, is_upper, c = parse_valid(stmt)
    except ValueError: return uop  # give up if we cannot parse the valid
    bounds[expr][int(is_upper)] = c

  # simplify uop given that valid is True
  for expr,v in bounds.items():
    # some expr has lower bound > upper bound -> valid is an empty set and we return None
    if v[0] is not None and v[1] is not None and v[0] > v[1]: return None

    # every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
    candidates = []
    if expr.op is Ops.ADD and v[0] == 1 and all(u.op in GroupOp.Irreducible for u in split_uop(expr, Ops.ADD)):
      # if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
      candidates.append([(Xi, UOp.variable("fake", 1, Xi.vmax, Xi.dtype)) for Xi in split_uop(expr, Ops.ADD)])
    # try checking the whole clause
    if expr in uop.toposort:
      candidates.append([(expr, UOp.variable("fake", expr.vmin if v[0] is None else v[0], expr.vmax if v[1] is None else v[1], expr.dtype))])

    for candidate in candidates:
      # if every branch in candidate gives the same simplified uop, we can rewrite the uop
      newuops = [uop.substitute({X:newX}).simplify().substitute({newX:X}).simplify() for X,newX in candidate]
      if uop.op is Ops.VECTORIZE and len(uop.src) == 2:
        if all_same([uops.src[0] for uops in newuops]): uop = uop.replace(src=(newuops[0].src[0], uop.src[1]))
        if all_same([uops.src[1] for uops in newuops]): uop = uop.replace(src=(uop.src[0], newuops[0].src[1]))
      elif all_same(newuops): uop = newuops[0]

  return uop

def _valid_priority(v: UOp, valids:List[UOp]):
  # we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
  try: return sum(-1 if parse_valid(v)[0] in other.toposort else 0 for other in valids)
  except ValueError: return 0

def simplify_valid(valid:UOp) -> Optional[UOp]:
  ret:List[UOp] = []
  something_changed = False
  valids = list(split_uop(valid, Ops.AND))
  for stmt in sorted(valids, key=lambda v: _valid_priority(v, valids)):
    ret.append(newstmt if ret and (newstmt:=uop_given_valid(functools.reduce(operator.and_, ret), stmt)) is not None else stmt)
    if ret[-1] is not stmt: something_changed = True
  return functools.reduce(operator.and_, ret) if something_changed else None

def max_var_const(x:UOp, c1:UOp, c2:UOp):
  if x.vmin >= 0: return x*c1 if c1.arg >= c2.arg else x*c2
  if x.vmax <= 0: return x*c2 if c1.arg >= c2.arg else x*c1

def sint_to_uop(x:sint) -> UOp: return UOp.const(dtypes.int, x) if isinstance(x, int) else x

symbolic_simple = PatternMatcher([
  # ** self folding **
  (UPat.var("x") + 0, lambda x: x),    # x+0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x") // -1, lambda x: -x), # x//-1 -> -x
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)), # x/x -> 1
  ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), lambda x,x2: x), # (x*x2)/x2 -> x
  ((UPat.var() % UPat.var("y")).named("base") % UPat.var("y"), lambda base,y: base),  # (x%y)%y = -> x%y (rewritten with base for speed)
  (UPat.var("x")%UPat.cvar("c")+(UPat.var("x")//UPat.cvar("c"))*UPat.cvar("c"), lambda x,c: x), # (x%c)+(x//c)*c = x
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c", vec=False), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c", vec=False), lambda x,c: c if c.arg else x),
  (UPat.var("x").maximum(UPat.var("x")), lambda x: x),
  ((UPat.var("x") & UPat.var("x")), lambda x: x),
  ((UPat.var("x") | UPat.var("x")), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).logical_not().logical_not(), lambda x: x),
  (UPat.var("x", dtype=dtypes.bool).where(UPat.const(dtypes.bool, True), UPat.const(dtypes.bool, False)), lambda x: x),
  # ** zero folding **
  (UPat.var("x") < UPat.var("x"), lambda x: UOp.const(dtypes.bool.vec(x.dtype.count), False)), # x < x -> False
  (UPat.var("x", dtype=dtypes.ints) != UPat.var("x", dtype=dtypes.ints),
   lambda x: UOp.const(dtypes.bool.vec(x.dtype.count), False)), # x != x -> False (only ints)
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # ** constant folding **
  (UPat(GroupOp.ALU, name="a", src=UPat((Ops.VCONST, Ops.CONST))), lambda a: a.const_like(exec_alu(a.op, a.dtype, [x.arg for x in a.src], False))),
  # bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool), lambda x,y: x&y),
  (UPat.var('x', dtype=dtypes.bool) + UPat.var('y', dtype=dtypes.bool), lambda x,y: x|y),
  (UPat.var('x', dtype=dtypes.bool).maximum(UPat.var('y', dtype=dtypes.bool)), lambda x,y: x|y),
  # *** cast ***
  (UPat(Ops.CAST, name="root", src=UPat.cvar("c")), lambda root, c: root.const_like(c.arg)),
  (UPat(Ops.CAST, name="root"), lambda root: root.src[0] if root.dtype == root.src[0].dtype else None),
])

symbolic = symbolic_simple+PatternMatcher([
  # ** COMMUTATIVE flipping **
  (UPat(GroupOp.Commutative, name='x'), lambda x: x.replace(src=x.src[::-1]) if x.src[1].tuplize < x.src[0].tuplize else None),
  # group like
  ((UPat.var("x") + UPat.var("y")) + UPat.var("x") * UPat.cvar("c"), lambda x,y,c: (x+x*c)+y),
  # ** boolean algebra **
  (UPat.var("x") | (UPat.var("x") & UPat.var()), lambda x: x), # x|(x&y) -> x
  # ** combine terms **
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1)), # (x*c0)+(x*c1) -> x*(c0+c1)
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), lambda x,c: x*(c+1)), # (x+x*c)-> x*(c+1)
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  ((UPat.var("x") / UPat.var("x2")) / UPat.var("x3"), lambda x,x2,x3: x/(x2*x3)), # (x/x2)/x3 -> x/(x2*x3)
  (-1 * (UPat.var("x") + UPat.cvar("c")), lambda x,c: (-x)+(-c)),  # -(x+c) -> -x + -c
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # alu of two where with same conds can combine, only do if true branch or false branch is const
  (UPat(GroupOp.Binary, name="alu", src=(UPat.var("c").where(UPat.var("t"), UPat.var("f")), UPat.var("c").where(UPat.var("tt"), UPat.var("ff")))), \
   lambda alu,c,t,tt,f,ff: c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) if t.op == tt.op == Ops.CONST or f.op == ff.op == Ops.CONST else None),
  # ALU min==max -> CONST (slow!)
  (UPat(GroupOp.ALU, name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # max folding
  (UPat.maximum(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else y if x.vmax <= y.vmin else None),
  # TODO: why does this rule break beautiful_mnist?
  #((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
  ((UPat.var("x")*UPat.cvar("c1")).maximum(UPat.var("x")*UPat.cvar("c2")), max_var_const),
  # ** two stage ALU folding **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.cvar("c2"), lambda x,c1,c2: x+(c1+c2)),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.cvar("c2"), lambda x,c1,c2: x*(c1*c2)),
  ((UPat.var("x") & UPat.cvar("c1")) & UPat.cvar("c2"), lambda x,c1,c2: x&(c1&c2)),
  ((UPat.var("x") | UPat.cvar("c1")) | UPat.cvar("c2"), lambda x,c1,c2: x|(c1|c2)),
  ((UPat.cvar("c0") + UPat.var("x")) < UPat.cvar("c1"), lambda x,c0,c1: x<(c1-c0)),  # c0 + x < c1 -> x < c1 - c0
  ((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), lambda x,c1,c2: x//(c1*c2)), # (x//c1)//c2 -> x//(c1*c2)
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
  # ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
  (UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c1"))) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
  (UPat(Ops.MUL, src=(UPat.var("x"), UPat.cvar("c1"))) * UPat.var("y"), lambda x,c1,y: (x*y)*c1),
  # *** rules from symbolic ***
  # unrolled arange div folding
  (UPat(Ops.ADD, name="divs", src=[UPat(), UPat(Ops.IDIV)]), fold_unrolled_divs),
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

_substitute = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.IDIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg[0])),
  (UPat(Ops.RANGE, name="x"), lambda x: UOp(Ops.NOOP, arg=f"ridx{x.arg[0]}")),
  (UPat(Ops.CONST, name="x"), lambda x: UOp(Ops.NOOP, arg=str(x.arg))),
  (UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: x.src[0]),
  (UPat(Ops.NEG, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(-{x.src[0].arg})")),
  (UPat(Ops.MAX, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"max({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.MULACC, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}*{x.src[1].arg}+{x.src[2].arg})")),
  (UPat(Ops.WHERE, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[1].arg} if {x.src[0].arg} else {x.src[2].arg})")),
  (UPat(GroupOp.ALU, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}{syms[x.op]}{x.src[1].arg})")),
])

# *** what was symbolic.py ***

sint = Union[int, UOp]
Variable = UOp

ConstLike = Union[ConstType, Variable, Tuple[ConstType, ...]]

# *** uop swizzling ***

merge_views = PatternMatcher([(UPat(Ops.VIEW, name="s0").view(name="s1"), lambda s0,s1: s0.replace(arg=s0.st+s1.st))])

# push VIEW to loads
view_left = merge_views+PatternMatcher([
  # VIEW before elementwise ops
  (UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN}, name="e").view(name="v"),
   lambda e,v: e.replace(src=tuple(s if not s.has_st else s.view(v.st) if s is s.base else s.base.view(s.st+v.st) for s in e.src))),
  # early merge VIEW buffer ops
  (UPat(GroupOp.Buffer, name="b").view(name="v"), lambda b,v: b.replace(src=tuple((s.st+v.st).to_uop() if s.op is Ops.VIEW else s for s in b.src))),
])
