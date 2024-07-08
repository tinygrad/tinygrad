from __future__ import annotations
from typing import Iterator, Optional, Tuple, Any, Dict, List, DefaultDict, Set, Callable, Union, cast, TypeVar
import functools, itertools, heapq, math
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from tinygrad.dtype import ConstType, dtypes, DType
from tinygrad.shape.symbolic import sint, Variable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu
from tinygrad.helpers import prod, DEBUG, getenv

# the order of these UOps controls the order of the toposort
class UOps(Enum):
  # ops that aren't rendered
  SINK = auto(); VAR = auto() # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  CONST = auto(); SPECIAL = auto() # noqa: E702
  NOOP = auto(); UNMUL = auto(); GEP = auto() # noqa: E702
  # math ops
  CAST = auto(); BITCAST = auto(); VECTORIZE = auto() # noqa: E702
  ALU = auto(); WMMA = auto() # noqa: E702
  # memory/assignment ops
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702
  # control flow ops
  BARRIER = auto(); IF = auto(); RANGE = auto() # noqa: E702
  # these two are not graph nodes
  ENDRANGE = auto(); ENDIF = auto() # noqa: E702

def ufix(dtype: Optional[DType], x): return UOp.const(dtype, x) if not isinstance(x, UOp) else x
@dataclass(frozen=True, eq=False)
class UOp:
  op: UOps
  dtype: Optional[DType] = None
  src: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def commutative(self) -> bool:
    return self.op is UOps.ALU and self.arg in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR}
  @functools.cached_property
  def cmp_tuple(self):
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    return (self.op.value, (self.arg if self.op is not UOps.DEFINE_VAR else self.arg.expr) if self.op is not UOps.ALU else \
            self.arg.value, self.dtype, self.src)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  def __repr__(self):
    return f"{str(self.op):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.op for x in self.src]):32s} {self.arg}"
  def cast(self, dtype=None): return UOp(UOps.CAST, dtype, (self,))
  def name(self, name:Optional[str]): return UOp(UOps.VAR, src=(self,), arg=name)
  def __neg__(self): return UOp.alu(UnaryOps.NEG, self)
  def __add__(self, x): return UOp.alu(BinaryOps.ADD, self, ufix(self.dtype, x))
  def __radd__(self, x): return UOp.alu(BinaryOps.ADD, ufix(self.dtype, x), self)
  def __sub__(self, x): return UOp.alu(BinaryOps.ADD, self, -ufix(self.dtype, x))
  def __mul__(self, x): return UOp.alu(BinaryOps.MUL, self, ufix(self.dtype, x))
  def __rmul__(self, x): return UOp.alu(BinaryOps.MUL, ufix(self.dtype, x), self)
  def __floordiv__(self, x): return UOp.alu(BinaryOps.IDIV, self, ufix(self.dtype, x))
  def __truediv__(self, x): return UOp.alu(BinaryOps.MUL, self, UOp.alu(UnaryOps.RECIP, ufix(self.dtype, x)))
  def __mod__(self, x): return UOp.alu(BinaryOps.MOD, self, ufix(self.dtype, x))
  def ne(self, x): return UOp.alu(BinaryOps.CMPNE, self, ufix(self.dtype, x))
  def eq(self, x): return -self.ne(x)
  def lt(self, x): return UOp.alu(BinaryOps.CMPLT, self, ufix(self.dtype, x))
  def ge(self, x): return -self.lt(x)
  def max(self, x): return UOp.alu(BinaryOps.MAX, self, x)
  def min(self, x): return -UOp.alu(BinaryOps.MAX, -self, -x)
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def const(dtype:Optional[DType], b:ConstType|Variable):
    if isinstance(b, Variable): return UOp(UOps.DEFINE_VAR, dtype, (), b)
    return UOp(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b)
  @staticmethod
  def alu(arg, *src:UOp): return UOp(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else src[-1].dtype, src, arg)
  @staticmethod
  def load(*src:UOp, dtype:Optional[DType]=None, **kwargs): return UOp(UOps.LOAD, dtype, tuple(src)+tuple(kwargs.values()))
  @staticmethod
  def store(*src:UOp, **kwargs): return UOp(UOps.STORE, None, tuple(src)+tuple(kwargs.values()))
  @staticmethod
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return UOp(UOps.VAR, dtype=dtype, arg=name)
  @staticmethod
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return UOp(UOps.CONST, dtype=dtype).name(name)
  @functools.cached_property
  def parents(self) -> Set[UOp]: return set.union(set(self.src), *[x.parents for x in self.src])
  @property  # parents with self
  def sparents(self) -> Set[UOp]: return set([self]).union(self.parents)
  def vars(self) -> Set[UOp]: return set([x for x in set.union(set([self]), self.parents) if x.op is UOps.DEFINE_VAR])
  def divides(self, v):
    if self.op is UOps.CONST:
      return self.arg%v == 0
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return all(x.divides(v) for x in self.src)
      if self.arg is BinaryOps.MUL: return any(x.divides(v) for x in self.src)
    return False # generic false if we aren't sure

def type_verify(uops):
  for u in uops:
    uop, arg, src, dtype = u.op, u.arg, u.src, u.dtype
    if uop in {UOps.CONST, UOps.DEFINE_ACC}:
      if uop is UOps.DEFINE_ACC:
        assert dtype is not None and src[0].dtype == dtype.scalar(), f"type of {src[0].dtype=} must be a scalar {dtype.scalar()}"
        arg = src[0].arg
      assert dtype is not None and type(arg) is type(dtypes.as_const(arg, dtype)), f"type of {arg=} does not match {dtype}"
    if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype is not None # type is the output type, not an arg
    if uop is UOps.CAST: assert dtype.count == 1 and len(src) == dtype.count
    if uop is UOps.VECTORIZE:
      assert dtype.count > 1 and len(src) == dtype.count, f"dtype vectorization mismatch {dtype.count=} != {len(src)=}"
      assert dtype == src[0].dtype.vec(len(src)), f"{dtype=} must be {src[0].dtype.vec(len(src))}"
    if uop is UOps.LOAD and len(src) > 3 and src[2].op is UOps.ALU: assert src[2].dtype == dtypes.bool and src[3].dtype == dtype
    if uop is UOps.STORE:
      assert dtype is None, f"{uop} dtype must be None, got {dtype}"
      if len(src) == 4: assert src[3].dtype == dtypes.bool, f"gate dtype mismatch {src[3].dtype} != {dtypes.bool}"
    if uop is UOps.ALU:
      if arg in UnaryOps:
        assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in {BinaryOps.CMPLT, BinaryOps.CMPNE}:
        assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
        assert src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg is BinaryOps.IDIV:
        assert dtypes.is_int(src[0].dtype) and dtypes.is_int(src[1].dtype), \
            f"input dtype mismatch {dtypes.int} != {src[0].dtype=} != {src[1].dtype=}"
        assert dtypes.is_int(dtype), f"{arg} output dtype mismatch {dtype=} != {dtypes.int}"
      elif arg in {BinaryOps.SHL, BinaryOps.SHR}:
        # the distance to shift isn't typechecked
        assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in BinaryOps:
        assert dtype == src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg == TernaryOps.WHERE:
        assert src[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {src[0].dtype=} != {dtypes.bool}"
        assert dtype == src[1].dtype == src[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {src[1].dtype=} != {src[2].dtype=}"

def uop_alu_resolve(u:UOp) -> sint:
  if u.op is UOps.CONST: return u.arg
  if u.op is UOps.DEFINE_VAR: return u.arg
  if u.op is UOps.SPECIAL: return u.arg[2]-1
  if u.op is UOps.ALU and u.arg is BinaryOps.MUL: return uop_alu_resolve(u.src[0]) * uop_alu_resolve(u.src[1])
  if u.op is UOps.ALU and u.arg is BinaryOps.SHL: return uop_alu_resolve(u.src[0]) * (2**cast(int, uop_alu_resolve(u.src[1])))
  if u.op is UOps.ALU and u.arg is BinaryOps.ADD: return uop_alu_resolve(u.src[0]) + uop_alu_resolve(u.src[1])
  raise RuntimeError(f"ALU resolve fail @ {u.op}")

# *** simplification logic ***

@dataclass(frozen=True)
class UPat:
  op: Optional[Union[UOps, Set[UOps]]] = None
  arg: Any = None
  src: Optional[Union[Tuple[UPat, ...], List[UPat], UPat]] = None
  name: Optional[str] = None
  dtype: Optional[Union[DType, Set[DType]]] = None
  allow_len: Set[int] = field(default_factory=set)
  allow_any_len: bool = False

  @staticmethod
  def compile(u: UOp, name:Optional[str]=None) -> UPat:
    if u.op is UOps.VAR: return UPat(name=name or u.arg, dtype=u.dtype) if len(u.src) == 0 else UPat.compile(u.src[0], name or u.arg)
    return UPat(u.op, u.arg, (list if u.commutative() else tuple)([UPat.compile(src) for src in u.src]) if u.src != () else None,
                name, u.dtype, allow_any_len=(isinstance(name, str) and 'allow_any_len' in name))

T = TypeVar("T")
def __unmatch(m1:Union[T, Set[T]], m2:T) -> bool: return m2 not in m1 if isinstance(m1, set) else m2 != m1

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> bool:
  if pat.name is not None and store.setdefault(pat.name, uop) is not uop: return False
  if pat.arg is not None and __unmatch(pat.arg, uop.arg): return False
  if pat.dtype is not None and uop.dtype is not None and __unmatch(pat.dtype, uop.dtype): return False
  if pat.op is not None and __unmatch(pat.op, uop.op): return False
  if pat.src is None: return True
  # only one if it's a tuple
  # try all permutations if it's a list
  # repeat if it's a UPat
  for vp in itertools.permutations(pat.src) if isinstance(pat.src,list) else ([pat.src] if isinstance(pat.src,tuple) else [(pat.src,)*len(uop.src)]):
    if len(uop.src) != len(vp) and (len(uop.src) not in pat.allow_len) and not pat.allow_any_len: return False
    new_store = store.copy()
    if all(_match(uu, vv, new_store) for uu, vv in zip(uop.src, vp)):
      store.update(new_store)
      return True
  return False

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Union[UPat, UOp], Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      if isinstance(p, UOp): p = UPat.compile(p)
      assert p.op is not None
      if isinstance(p.op, set):
        for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn))
      else:
        self.pdict[(p.op, p.arg)].append((p, fxn))

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      store: Dict[str, UOp] = {}
      if _match(uop, p, store) and (ret:=fxn(**store)) is not None: return ret  # NOTE: if it returns None, we keep trying to match
    return None

def sum_collapse(phi_input, loop, val1, val2):
  for v1,v2 in [(val1, val2), (val2, val1)]:
    if loop not in v1.parents:
      loop_range = loop.src[1]-loop.src[0]
      ret = v1*loop_range.cast(v1.dtype)
      return UOp(UOps.PHI, phi_input.dtype, (phi_input, v2))+ret
  return None

def loop_collapse(loop_start, loop_end, compval, idx, mval, multconst, rng):
  if getenv("DISABLE_LOOP_COLLAPSE") or not rng.arg[2]: return None   # must be a reduce
  if mval.arg >= 0 or loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  comprange = UOp.min(loop_end, UOp.max(UOp.alu(BinaryOps.IDIV, idx-compval-mval, mval) + (loop_end-loop_start), loop_start))
  return UOp(UOps.UNMUL, multconst.dtype, (comprange.cast(multconst.dtype) * multconst, loop_end-loop_start))

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # arange loop folding (early)
  (UPat(UOps.ALU, TernaryOps.WHERE, src=(UPat(UOps.ALU, BinaryOps.CMPLT, src=(
    UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(name="idx"), UPat(UOps.ALU, BinaryOps.MUL, src=[UPat(UOps.CONST, name="mval"),
      UPat(UOps.RANGE, name="rng", src=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
      UPat(UOps.CONST, name="compval"))), UPat(UOps.CONST, name="multconst"), UPat(UOps.CONST, 0))), loop_collapse),
  (UPat(UOps.ALU, TernaryOps.WHERE, src=(UPat(UOps.ALU, BinaryOps.CMPLT, src=(
    UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(name="idx"), UPat(UOps.ALU, UnaryOps.NEG, src=[
      UPat(UOps.RANGE, name="rng", src=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
      UPat(UOps.CONST, name="compval"))), UPat(UOps.CONST, name="multconst"), UPat(UOps.CONST, 0))),
      lambda **kwargs: loop_collapse(mval=UOp.const(dtypes.int, -1), **kwargs)),
  # sum collapse to mul (with possible GEP)
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, name="phi_input", src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  (UPat(UOps.PHI, src=(UPat(UOps.GEP, name="phi_input", src=(UPat(UOps.DEFINE_ACC, src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),)),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  # deal with UNMUL
  (UPat(UOps.ALU, BinaryOps.MUL, [UPat(UOps.CONST, name="c1"), UPat(UOps.UNMUL, src=[UPat(UOps.CONST, name="c2"), UPat(name="v")])]),
   lambda c1,c2,v: v if c1.arg == c2.arg else None),
  (UOp(UOps.UNMUL, src=(UOp.const(None, 0).name('zero'), UOp.var())), lambda zero: zero),
  (UOp(UOps.UNMUL).name('unmul').cast().name('root'), lambda root,unmul: UOp(UOps.UNMUL, root.dtype, (unmul.src[0].cast(root.dtype), unmul.src[1]))),
  # max on special can go away (TODO: special should be variable, same thing applies)
  (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')), lambda c,s: c if (s.arg[2]-1) <= c.arg else None),
  # const rules
  (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="c"),)), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.CAST, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.VECTORIZE, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi on a DEFINE_ACC without loops or a CONST is a noop. this is for correctness, not just speed
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, name="acc"), UPat(name="acc"))), lambda acc: UOp.cast(acc.src[0], acc.dtype)),
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, src=(UPat(UOps.CONST),)), UPat(name="x"))), lambda x: x),
  (UPat(UOps.PHI, src=(UPat(UOps.CONST), UPat(name="x"))), lambda x: x),
  # a DEFINE_ACC without inputs is a const + GEP on a const is the const
  (UPat(UOps.DEFINE_ACC, name="root", src=(UPat(UOps.CONST),)), lambda root: UOp.cast(root.src[0], root.dtype)),
  (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="x"),)), lambda root,x: UOp.const(root.dtype, x.arg)),
  # max -2147483648
  (UOp.max(UOp.var('x'), UOp.const(dtypes.int, -2147483648)), lambda x: x),
  # bool < False is always false, True < bool is always false
  (UOp.var().lt(UOp.const(dtypes.bool, False)), lambda: UOp.const(dtypes.bool, False)),
  (UOp.const(dtypes.bool, True).lt(UOp.var()), lambda: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UOp.alu(TernaryOps.WHERE, UOp.var(), UOp.var("val"), UOp.var("val")), lambda val: val),
  (UOp.alu(TernaryOps.WHERE, UOp.cvar('gate'), UOp.var('c0'), UOp.var('c1')), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat(UOps.CONST)), lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  (-(-UOp.var('x')), lambda x: x),    # -(-x) -> x
  (UOp.var('x') + 0, lambda x: x),    # x+0 -> x
  (UOp.var('x') - 0, lambda x: x),    # x-0 -> x
  (UOp.var('x') * 1, lambda x: x),    # x*1 -> x
  (UOp.var('x') * -1, lambda x: -x),  # x*-1 -> -x
  (UOp.var('x') // UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x//x -> 1
  (UOp.var('x') // 1, lambda x: x),   # x//1 -> x
  (UOp.var('x') // -1, lambda x: -x), # x//-1 -> -x
  (UOp.var('x') / UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x/x -> 1
  (UOp.var('x') / UOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
  (UOp.var('x', dtype=dtypes.bool).max(UOp.const(dtypes.bool, False)), lambda x: x),  # max(x, False) -> x
  # ** zero folding **
  #x*0 -> 0 or 0*x -> 0
  #if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UOp.var('x') * 0, lambda x: UOp.const(x.dtype, float('nan') if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  (UOp.var('x') - UOp.var('x'), lambda x: UOp.const(x.dtype, 0)),   # x-x -> 0
  # ** load/store folding **
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.load(UOp.var("buf"), UOp.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/sub folding **
  ((UOp.var('x') + UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((UOp.var('x') - UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c2.arg, -c1.arg]))),
  # *** rules from symbolic ***
  # two stage mul, (x*c1)*c2 = x*(c1*c2)
  ((UOp.var("x") * UOp.cvar("c1")) * UOp.cvar("c2"), lambda x,c1,c2: x*UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # x%1 -> 0
  (UOp.var("x") % UOp.const(None, 1), lambda x: UOp.const(x.dtype, 0)),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (UOp.var("x") * UOp.cvar("c0") + UOp.var("x") * UOp.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)+(y*c0) -> (x+y)*c0
  #((UOp.var("x") * UOp.cvar("c0")) + (UOp.var("y") * UOp.cvar("c0")), lambda x,y,c0: c0*(x+y)),
  # (x*c0)//c0 -> x
  ((UOp.var("x") * UOp.cvar("c0")) // UOp.cvar("c0"), lambda x,c0: x if c0.arg != 0 else None),
  # (x*x2)/x2 -> x
  ((UOp.var("x") * UOp.var("x2")) / UOp.var("x2"), lambda x,x2: x),
  # (x//c0)//c1 -> x//(c0*c1)
  ((UOp.var("x") // UOp.cvar("c0")) // UOp.cvar("c1"), lambda x,c0,c1: x//UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # (x/x1)/x2 -> x/(x1*x2)
  ((UOp.var("x") / UOp.var("x2")) / UOp.var("x3"), lambda x,x2,x3: x/(x2*x3)),
  # c0 + x < c1 -> x < c1 - c0
  ((UOp.cvar("c0") + UOp.var("x")).lt(UOp.cvar("c1")),
    lambda x,c0,c1: UOp.lt(x, UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, -c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (UOp.var("x") + UOp.var("x") * UOp.cvar("c0"), lambda x,c0: x*UOp.const(x.dtype, c0.arg+1)),
  # x!=0 -> (bool)x
  (UOp.var("x").ne(0), lambda x: x.cast(dtypes.bool)),
  # bool != 1 -> not bool
  (UOp.var("x", dtype=dtypes.bool).ne(1), lambda x: -x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.alu(TernaryOps.WHERE, UOp.var("gate"), UOp.var("alt"), UOp.load(UOp.var("buf"), UOp.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # store float4/float2 directly (remove VECTORIZE/GEP)
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.VECTORIZE, src=tuple(
    UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(4)))), UOp.store),
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.VECTORIZE, src=tuple(
    UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(2)))), UOp.store),
  # VECTORIZE-PHI-GEP -> PHI-VECTORIZE
  (UPat(UOps.VECTORIZE, name="root", src=tuple(
    UPat(UOps.PHI, src=(UPat(UOps.GEP, i, src=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(4))),
    lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1, v2, v3))))),
  (UPat(UOps.VECTORIZE, name="root", src=tuple(
    UPat(UOps.PHI, src=(UPat(UOps.GEP, i, src=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(2))),
    lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1))))),
  # NEG/CMPLT -> CMPLT
  (UOp.lt(-UOp.var('x'), UOp.cvar('c', dtypes.int)), lambda c,x: UOp.lt(UOp.const(c.dtype, -c.arg), x)),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (UPat(UOps.CAST, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (UPat(UOps.VECTORIZE, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  # fold gated LOAD/STORE
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var")), lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var"), UOp.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var")), lambda var: var),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var"), UOp.var()), lambda var: var),
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.var("val"), UOp.const(dtypes.bool, True)), UOp.store),
  (UOp.store(UOp.var(), UOp.var(), UOp.var(), UOp.const(dtypes.bool, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None)
])

# *** uop graph ***

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):
  if u in children: return
  children[u] = []
  for x in u.src:
    get_children_dfs(x, children, in_degree)
    children[x].append(u)
  in_degree[u] = len(u.src)

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if n in replace: return replace[n]
    replace_source = (n.op, n.dtype, (lambda x: tuple(sorted(x) if n.commutative() else x))(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else: nodes[replace_source] = replace[n] = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x
    return replace[n]
  return __inner_rewrite(sink)

class UOpGraph:
  def __init__(self, sinks:List[UOp]):
    self.sinks: List[UOp] = sinks
    # used by linearizer
    self._uops: Optional[List[UOp]] = None

  def __iter__(self) -> Iterator[UOp]: return iter(self.uops)
  def __getitem__(self, index) -> UOp: return self.uops[index]

  def vars(self) -> List[Variable]: return sorted([x.arg for x in self.uops if x.op is UOps.DEFINE_VAR], key=lambda v: v.expr)
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.op is UOps.DEFINE_GLOBAL]

  @property
  def uops(self):
    if self._uops is None: self.linearize()
    return self._uops

  def graph(self):
    from tinygrad.engine.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for i,u in enumerate(self):
      print(f"{i:4d} {str(u.op):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str([self.uops.index(x) for x in u.src]):32s} {u.arg}")

  def linearize(self, extra_pm:Optional[PatternMatcher]=None, do_type_verify=True):
    # NOTE: relinearizering should be okay
    #assert self._uops is None, "already linearized"

    # fixup gated stores with an IF block to save extra local loads
    @functools.lru_cache(None)
    def _dfs(u:UOp, gate:UOp) -> UOp:
      if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER:
        if_uop = UOp(UOps.IF, None, (gate, u.src[-1]))
        return UOp(u.op, u.dtype, u.src[:-1]+(if_uop,), u.arg)
      if (replace_source:=tuple(_dfs(x, gate) for x in u.src)) != u.src: return UOp(u.op, u.dtype, replace_source, u.arg)
      return u
    for i, s in enumerate(self.sinks[:]):
      if s.op is UOps.STORE and len(s.src) == 4 and (rw:=_dfs(s, s.src[3])) != s: self.sinks[i] = UOp(rw.op, rw.dtype, rw.src[:3], rw.arg)
    sink = UOp(UOps.SINK, None, tuple(self.sinks))

    # dedup all nodes and do graph rewrite
    sink = graph_rewrite(sink, constant_folder)
    if extra_pm: sink = graph_rewrite(sink, PatternMatcher(constant_folder.patterns+extra_pm.patterns))

    # filter nodes that don't link to a sink
    # BFS toposort
    children: Dict[UOp, List[UOp]] = {}
    in_degree: Dict[UOp, int] = {}
    get_children_dfs(sink, children, in_degree)

    @functools.lru_cache(None)
    def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
      if x.op is UOps.SINK: return set()
      return set.union(set((x,)) if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

    # scope children impact the toposort and END* insertion
    end_for_uop = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}
    scope_children = {p:get_recursive_children(p, end_for_uop[p.op][0]) for p in reversed(in_degree) if p.op in end_for_uop}

    queue:List[Tuple[int, UOp]] = []
    def push(u:UOp):
      priority = 0
      # prefer uops that are loop children
      for l, ss in scope_children.items():
        if l.op is UOps.RANGE and u in ss: priority -= l.arg[0]*1000 + l.arg[1]
      heapq.heappush(queue, (priority, u))

    for u in children:
      if in_degree[u] == 0: push(u)

    if getenv("FUZZ_UOPS", 0):
      from test.external.fuzz_uops import fuzz_uops
      self.fuzz_paths = fuzz_uops(children, in_degree.copy(), scope_children)

    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(p,x)
      if x.op is UOps.DEFINE_ACC and len(x.src) > 1:
        idx = min([self._uops.index(l) for l in x.src if l.op is UOps.RANGE])
        self._uops.insert(idx, x)
      else:
        self._uops.append(x)
      # for u, ss in scope_children.items():
      #   if x in ss:
      #     ss.remove(x)
      #     if len(ss) == 0: self._uops.append(UOp(end_for_uop[u.op][1], None, (u,)))
      for u in children[x]:
        in_degree[u] -= 1
        if in_degree[u] == 0: push(u)

    for u in (self._uops):
      if u.op in end_for_uop: self._uops.insert(max([self._uops.index(l) for l in scope_children[u]])+1, UOp(end_for_uop[u.op][1], None, (u,)))

    assert self._uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
    self._uops = self._uops[:-1]

    if do_type_verify: type_verify(self.uops)

  # *** checker functions ***

  def flops_mem(self, ignore_indexing=False) -> Tuple[sint, sint]:
    flops: sint = 0
    mem: sint = 0
    mults: sint = 1
    mult_stack = []
    dont_count: Set[UOp] = set()
    if ignore_indexing:
      for u in self.uops:
        if u.op is UOps.LOAD:
          dont_count = dont_count.union(u.src[1].sparents)
          if len(u.src) > 3: dont_count = dont_count.union(u.src[2].sparents)
        elif u.op is UOps.STORE:
          dont_count = dont_count.union(u.src[1].sparents)
          if len(u.src) > 3: dont_count = dont_count.union(u.src[3].sparents)
    for u in self.uops:
      if u.op is UOps.RANGE:
        mult_stack.append(mults)
        mults *= uop_alu_resolve(u.src[1])
      elif u.op is UOps.ENDRANGE:
        mults = mult_stack.pop(-1)
      elif u.op is UOps.LOAD:
        assert u.dtype is not None
        mem += u.dtype.itemsize * mults
      elif u.op is UOps.STORE:
        assert u.src[2].dtype is not None
        mem += u.src[2].dtype.itemsize * mults
      elif u.op is UOps.ALU and u not in dont_count:
        flops += mults * (2 if u.arg == TernaryOps.MULACC else 1)
      elif u.op is UOps.WMMA and u not in dont_count:
        assert u.arg[1] is not None
        flops += 2 * prod(u.arg[1]) // 32 * mults
    return flops, mem
