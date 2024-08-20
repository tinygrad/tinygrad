from __future__ import annotations
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Set, Union, Tuple, Dict, Callable, cast, TYPE_CHECKING
import math, operator, ctypes, struct, functools, hashlib, itertools
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes, DType
from tinygrad.helpers import merge_dicts, pretty_print, prod
from tinygrad.shape.symbolic import Variable, sint
if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum):
  """A -> A (elementwise)"""
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto(); RECIP = auto() # noqa: E702
class BinaryOps(Enum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto() # noqa: E702
class TernaryOps(Enum):
  """A + A + A -> A (elementwise)"""
  WHERE = auto(); MULACC = auto() # noqa: E702
class ReduceOps(Enum):
  """A -> B (reduce)"""
  SUM = auto(); MAX = auto(); WMMA = auto() # noqa: E702
class MetaOps(Enum):
  EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto(); VIEW = auto() # noqa: E702
Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps]

# do not preserve f(0) = 0
UNSAFE_PAD_OPS = {UnaryOps.RECIP, UnaryOps.LOG2, UnaryOps.EXP2, BinaryOps.IDIV}

@dataclass(frozen=True)
class KernelInfo:
  local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)
  upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to EXPAND)
  dont_use_locals: bool = False # don't use local indexing

# **************** ops in python ****************

def hook_overflow(dv, fxn):
  def wfxn(*args):
    try: return fxn(*args)
    except OverflowError: return dv
  return wfxn

python_alu: Dict[Op, Callable]  = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, UnaryOps.EXP2: hook_overflow(math.inf, lambda x: 2**x),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  UnaryOps.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan, UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
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

# the order of these UOps controls the order of the toposort
class UOps(Enum):
  # ops that aren't rendered
  SINK = auto(); EXT = auto(); EXPAND = auto(); CONTRACT = auto(); SHAPETRACKER = auto()  # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  CONST = auto(); SPECIAL = auto() # noqa: E702
  NOOP = auto(); GEP = auto() # noqa: E702
  # math ops
  CAST = auto(); BITCAST = auto(); VECTORIZE = auto() # noqa: E702
  ALU = auto(); REDUCE = auto(); REDUCE_AXIS = auto(); WMMA = auto() # noqa: E702
  # memory/assignment ops
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702
  # control flow ops
  BARRIER = auto(); IF = auto(); RANGE = auto() # noqa: E702
  # these two are not graph nodes
  ENDRANGE = auto(); ENDIF = auto() # noqa: E702

BUFFER_UOPS = {UOps.LOAD, UOps.STORE, UOps.CONST}

END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}

@dataclass(frozen=True, eq=False)
class UOp:
  op: UOps
  dtype: Optional[DType] = None
  src: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def commutative(self) -> bool:
    return (self.op is UOps.ALU and \
      self.arg in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR})
  @functools.cached_property
  def cmp_tuple(self):
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    return (self.op.value, (self.arg if self.op is not UOps.DEFINE_VAR else self.arg.expr) if self.op is not UOps.ALU else \
            self.arg.value, self.dtype, self.src)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(functools.reduce(lambda x,y: x+y, [s.key for s in self.src], str((self.op, self.dtype, self.arg)).encode())).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is UOps.REDUCE_AXIS else self.arg
  # *** uop syntactic sugar
  @property
  def st_arg(self) -> ShapeTracker:
    assert self.op in BUFFER_UOPS, f"st_arg called on {self.op}"
    ret = self.src[0 if self.op is UOps.CONST else 1]
    assert ret.op is UOps.SHAPETRACKER, f"st_arg trying to return {ret}"
    return ret.arg
  def ufix(self, x): return self.const(x) if not isinstance(x, UOp) else x
  def cast(self, dtype=None): return type(self)(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return type(self)(UOps.BITCAST, dtype, (self,))
  def gep(self, i:int): return type(self)(UOps.GEP, self.dtype.scalar() if self.dtype is not None else None, (self,), i)
  def __neg__(self): return self.alu(UnaryOps.NEG)
  def __add__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
  def __radd__(self, x): return self.alu(BinaryOps.ADD, self.ufix(x))
  def __sub__(self, x): return self.alu(BinaryOps.ADD, self.ufix(-x))
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
  def ge(self, x): return (-self).lt(-x+1)
  def max(self, x): return self.alu(BinaryOps.MAX, x)
  def min(self, x): return -(-self).max(-x)
  def where(self, x, y): return self.alu(TernaryOps.WHERE, x, y)
  def recip(self): return self.alu(UnaryOps.RECIP)
  def const(self:Union[UOp, DType, None], b:ConstType|Variable): return UOp._const(self.dtype if isinstance(self, UOp) else self, b)
  def sconst(self:Union[UOp, DType, None], b:ConstType|Variable):
    return UOp._const(cast(DType, self.dtype if isinstance(self, UOp) else self).scalar() if self is not None else self, b)
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def _const(dtype:Optional[DType], b:ConstType|Variable):
    # TODO: fix dtype of b.max after Variable is just an UOp
    if isinstance(b, Variable): return UOp(UOps.DEFINE_VAR, dtype, (UOp.const(dtypes.int, b.min), UOp.const(dtypes.int, cast(int,b.max))), b)
    if dtype is not None and dtype != (sdtype := dtype.scalar()):
      return UOp(UOps.VECTORIZE, dtype, src=tuple(UOp(UOps.CONST, sdtype, arg=dtypes.as_const(b, sdtype)) for _ in range(dtype.count)))
    return UOp(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b)
  def alu(self, arg, *src:UOp):
    return type(self)(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else (self, *src)[-1].dtype, (self,)+src, arg)
  @staticmethod
  def load(*src:UOp, dtype:Optional[DType]=None, **kwargs): return type(src[0])(UOps.LOAD, dtype, tuple(src)+tuple(kwargs.values()))
  @staticmethod
  def store(*src:UOp, **kwargs): return type((src:=(*src, *kwargs.values()))[0])(UOps.STORE, None, src)
  @functools.cached_property
  def parents(self) -> Dict[UOp, None]: return merge_dicts([{x:None for x in self.src}]+[x.parents for x in self.src])
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
    if self.op is UOps.CONST: return self.const(self.arg//v) if self.arg%v == 0 else None
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
      if self.arg is BinaryOps.MUL:
        if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
        if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> UOp: return x if (x:=self._min_max[0]) is not None and not math.isnan(x.arg) else self.sconst(dtypes.min(cast(DType, self.dtype)))
  @property
  def vmax(self) -> UOp: return x if (x:=self._min_max[1]) is not None and not math.isnan(x.arg) else self.sconst(dtypes.max(cast(DType, self.dtype)))
  @functools.cached_property
  def _min_max(self) -> Tuple[Optional[UOp], Optional[UOp]]:
    # NOTE: returned UOp is assumed to be CONST
    if self.op is UOps.DEFINE_VAR and self.src: return self.src[0], self.src[1] if isinstance(self.src[1].arg, int) else None
    if self.op is UOps.RANGE: return self.src[0].vmin, (self.src[1]-1).vmax
    # TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if self.op is UOps.SPECIAL: return self.const(0), self.const(self.arg[1]-1) if isinstance(self.arg[1], int) else None
    if self.op is UOps.CONST: return self, self
    if self.op is UOps.ALU and cast(DType, self.dtype).count == 1:
      s0,s1 = [cast(UOp, self.src[i] if i < len(self.src) else None) for i in range(2)]
      if self.arg is UnaryOps.NEG and self.dtype != dtypes.bool and not dtypes.is_unsigned(cast(DType, self.dtype)):
        return self.sconst(-s0.vmax.arg), self.sconst(-s0.vmin.arg)
      if self.arg is BinaryOps.ADD: return self.sconst(s0.vmin.arg+s1.vmin.arg), self.sconst(s0.vmax.arg+s1.vmax.arg)
      if self.arg is BinaryOps.MUL and (s0.vmin.arg >= 0 or s1.vmin.arg >= 0):
        # handle at lease one is non-negative
        Lmin, Lmax = (s0.vmin.arg, s0.vmax.arg) if s1.vmin.arg >= 0 else (s0.vmax.arg, s0.vmin.arg)
        Rmin, Rmax = (s1.vmin.arg, s1.vmax.arg) if s0.vmin.arg >= 0 else (s1.vmax.arg, s1.vmin.arg)
        assert math.isnan(Lmax*Rmax) or math.isnan(Lmin*Rmin) or Lmax*Rmax >= Lmin*Rmin, f"{Lmax=}, {Lmin=}, {Rmax=}, {Rmin=}"
        return self.sconst(Lmin*Rmin), self.sconst(Lmax*Rmax)
      if self.arg is BinaryOps.MOD and s1.vmin.arg > 0: return self.sconst(0), self.sconst(s1.vmax.arg-1)
      if self.arg is BinaryOps.IDIV and s1.op is UOps.CONST:
        if s1.arg > 0: return self.sconst(s0.vmin.arg//s1.arg), self.sconst(s0.vmax.arg//s1.arg)
        if s1.arg < 0: return self.sconst(-(s0.vmax.arg//-s1.arg)), self.sconst(-(s0.vmin.arg//-s1.arg))
      if self.arg is BinaryOps.MAX: return self.sconst(max(s0.vmin.arg, s1.vmin.arg)), self.sconst(max(s0.vmax.arg, s1.vmax.arg))
      if self.arg is BinaryOps.CMPLT: return (UOp.sconst(dtypes.bool, True), UOp.sconst(dtypes.bool, True)) if s0.vmax.arg < s1.vmin.arg else \
        (UOp.sconst(dtypes.bool, False), UOp.sconst(dtypes.bool, False)) if s0.vmin.arg >= s1.vmax.arg else (None, None)
    return None, None

@dataclass(frozen=True, repr=False)  # reuse repr from UOp
class NOp(UOp):
  name:Optional[str] = None
  src:Tuple[NOp, ...] = tuple()
  allow_any_len:bool = False
  @staticmethod
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.NOOP, dtype=dtype, name=name)
  @staticmethod
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.CONST, dtype=dtype, name=name)
  def const(self:Union[UOp, DType, None], b:ConstType|Variable): return NOp((x:=UOp.const(self, b)).op, x.dtype, x.src, x.arg)

  def compile(self: NOp, name:Optional[str]=None) -> UPat:
    return UPat(name=self.name, dtype=self.dtype) if self.op is UOps.NOOP else UPat(self.op, self.arg, (list if self.commutative()
      else tuple)(src.compile() for src in self.src) or None, self.name or name, self.dtype, self.allow_any_len)

class UPat:
  def __init__(self, op:Optional[Union[UOps, Set[UOps]]]=None, arg:Any=None, src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None,
               name:Optional[str]=None, dtype:Optional[Union[DType, Set[DType]]]=None, allow_any_len:bool=False):
    self.op: Optional[Tuple[UOps, ...]] = None if op is None else (tuple(op) if isinstance(op, set) else (op,))
    self.dtype: Optional[Tuple[DType, ...]] = None if dtype is None else (tuple(dtype) if isinstance(dtype, set) else (dtype,))
    self.arg, self.name = arg, name
    self.src: Any = None
    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src))
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.allowed_len: int = 0 if allow_any_len or isinstance(src, UPat) or src is None else len(src)

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
     (pat.op is not None and uop.op not in pat.op): return []
  if pat.src is None: return [store]
  res: List[Dict[str, UOp]] = []
  for vp in pat.src:
    if pat.allowed_len != 0 and len(uop.src) != pat.allowed_len: return []
    new_stores = [store.copy()]
    for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]
    res.extend(new_stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Union[UPat, NOp], Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      if isinstance(p, NOp): p = p.compile()
      assert p.op is not None
      for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn))

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if rn := replace.get(n): return rn
    replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else: nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x
    return found
  return __inner_rewrite(sink)

def type_verify(uops):
  for u in uops:
    uop, arg, src, dtype = u.op, u.arg, u.src, u.dtype
    if uop is UOps.DEFINE_LOCAL: assert isinstance(dtype, PtrDType), f"invalid dtype for local buffer {dtype}"
    if uop is UOps.DEFINE_GLOBAL: assert isinstance(dtype, (PtrDType, ImageDType)), f"invalid dtype for global buffer {dtype}"
    if isinstance(dtype, ImageDType): assert uop is UOps.DEFINE_GLOBAL, f"{uop} can't be image"
    if uop is UOps.SHAPETRACKER: assert len(src) == 0, f"SHAPETRACKER must only define a ShapeTracker arg {uop}"
    if uop is UOps.REDUCE_AXIS: assert isinstance(arg, tuple) and len(arg) == 2 and arg[0] in ReduceOps, f"invalid arg for REDUCE_AXIS {arg}"
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

def uop_alu_resolve(u:UOp) -> sint:
  if u.op in {UOps.CONST, UOps.DEFINE_VAR}: return u.arg
  if u.op is UOps.ALU: return exec_alu(u.arg, cast(DType,u.dtype), tuple(map(uop_alu_resolve, u.src)))
  raise RuntimeError(f"ALU resolve fail @ {u.op}")

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
      flops += 2 * prod(u.arg[1]) // 32 * mults
  return flops, mem
