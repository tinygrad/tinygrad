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
  SINK = auto()
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  CONST = auto(); SPECIAL = auto() # noqa: E702
  NOOP = auto(); UNMUL = auto(); GEP = auto() # noqa: E702
  # math ops
  CAST = auto(); BITCAST = auto() # noqa: E702
  ALU = auto(); WMMA = auto() # noqa: E702
  # memory/assignment ops
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702
  # control flow ops
  BARRIER = auto(); IF = auto(); RANGE = auto() # noqa: E702
  # these two are not graph nodes
  ENDRANGE = auto(); ENDIF = auto() # noqa: E702

def ufix(dtype, x): return UOp.const(dtype, x) if not isinstance(x, UOp) else x
@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType] = None
  vin: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def tuple(self): return (self.uop, self.dtype, self.vin, self.arg)
  @functools.cached_property
  def cmp_tuple(self):
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    return (self.uop.value, (self.arg if self.uop is not UOps.DEFINE_VAR else self.arg.expr) if self.uop is not UOps.ALU else \
            (type(self.uop), self.uop.value), self.dtype, self.vin)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"
  def cast(self, dtype): return UOp(UOps.CAST, dtype, (self,))
  def __neg__(self): return UOp.alu(UnaryOps.NEG, self)
  def __add__(self, x): return UOp.alu(BinaryOps.ADD, self, ufix(self.dtype, x))
  def __radd__(self, x): return UOp.alu(BinaryOps.ADD, ufix(self.dtype, x), self)
  def __sub__(self, x): return UOp.alu(BinaryOps.SUB, self, ufix(self.dtype, x))
  def __mul__(self, x): return UOp.alu(BinaryOps.MUL, self, ufix(self.dtype, x))
  def __rmul__(self, x): return UOp.alu(BinaryOps.MUL, ufix(self.dtype, x), self)
  def __floordiv__(self, x): return UOp.alu(BinaryOps.IDIV, self, ufix(self.dtype, x))
  def __mod__(self, x): return UOp.alu(BinaryOps.MOD, self, ufix(self.dtype, x))
  @staticmethod
  def max(x, y): return UOp.alu(BinaryOps.MAX, x, y)
  @staticmethod
  def min(x, y): return -UOp.alu(BinaryOps.MAX, -x, -y)
  @staticmethod
  def const(dtype:DType, b:ConstType|Variable):
    if isinstance(b, Variable): return UOp(UOps.DEFINE_VAR, dtype, (), b)
    return UOp(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype))
  @staticmethod
  def alu(arg, *vin:UOp): return UOp(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else vin[-1].dtype, vin, arg)
  @functools.cached_property
  def parents(self) -> Set[UOp]: return set.union(set(self.vin), *[x.parents for x in self.vin])
  def vars(self) -> Set[UOp]: return set([x for x in set.union(set([self]), self.parents) if x.uop is UOps.DEFINE_VAR])

def uop_alu_resolve(u:UOp) -> sint:
  if u.uop is UOps.CONST: return u.arg
  if u.uop is UOps.DEFINE_VAR: return u.arg
  if u.uop is UOps.SPECIAL: return u.arg[2]-1
  if u.uop is UOps.ALU and u.arg is BinaryOps.MUL: return uop_alu_resolve(u.vin[0]) * uop_alu_resolve(u.vin[1])
  if u.uop is UOps.ALU and u.arg is BinaryOps.SHL: return uop_alu_resolve(u.vin[0]) * (2**cast(int, uop_alu_resolve(u.vin[1])))
  if u.uop is UOps.ALU and u.arg is BinaryOps.ADD: return uop_alu_resolve(u.vin[0]) + uop_alu_resolve(u.vin[1])
  raise RuntimeError(f"ALU resolve fail @ {u.uop}")

# *** simplification logic ***

def upatfix(x: Any, dtype: UPatDType) -> UPat: return x if isinstance(x, UPat) else UPat.const(x, dtype)
@dataclass(frozen=True)
class UPat:
  uop: Optional[Union[UOps, Set[UOps]]] = None
  arg: Any = None
  vin: UPatVin = None
  name: Optional[str] = None
  dtype: UPatDType = None
  allow_len: Set[int] = field(default_factory=set)

  @staticmethod
  def var(name: str) -> UPat: return UPat(name=name)
  @staticmethod
  def cvar(name: str) -> UPat: return UPat(UOps.CONST, name=name)
  @staticmethod
  def const(val: Optional[ConstType] = None, dtype: UPatDType = None, name: Optional[str] = None): return UPat(UOps.CONST, val, None, name, dtype)
  @staticmethod
  def alu(op = None, vin: UPatVin = None, name:Optional[str]=None): return UPat(UOps.ALU, op, vin, name)
  @staticmethod
  def load(*vin: UPat, name: Optional[str] = None, dtype: UPatDType = None): return UPat(UOps.LOAD, vin=tuple(vin), name=name, dtype=dtype)
  @staticmethod
  def store(*vin: UPat, name: Optional[str] = None, dtype: UPatDType = None): return UPat(UOps.STORE, vin=tuple(vin), name=name, dtype=dtype)
  @staticmethod
  def where(gate: UPat, a: UPat, b: UPat, name: Optional[str] = None): return UPat.alu(TernaryOps.WHERE, (gate, a, b), name)

  def nm(self, name: Optional[str]) -> UPat: return UPat(self.uop, self.arg, self.vin, name, self.dtype, self.allow_len)
  def recip(self, name:Optional[str]=None): return UPat.alu(UnaryOps.RECIP, self, name=name)
  def __neg__(self): return UPat.alu(UnaryOps.NEG, self)
  def __add__(self, x): return UPat.alu(BinaryOps.ADD, [self, upatfix(x, self.dtype)])
  def __radd__(self, x): return UPat.alu(BinaryOps.ADD, [upatfix(x, self.dtype), self])
  def __sub__(self, x): return UPat.alu(BinaryOps.SUB, (self, upatfix(x, self.dtype)))
  def __rsub__(self, x): return UPat.alu(BinaryOps.SUB, (upatfix(x, self.dtype), self))
  def __mul__(self, x): return UPat.alu(BinaryOps.MUL, [self, upatfix(x, self.dtype)])
  def __rmul__(self, x): return UPat.alu(BinaryOps.MUL, [upatfix(x, self.dtype), self])
  def __floordiv__(self, x): return UPat.alu(BinaryOps.IDIV, (self, upatfix(x, self.dtype)))
  def __rfloordiv__(self, x): return UPat.alu(BinaryOps.IDIV, (upatfix(x, self.dtype), self))
  def __truediv__(self, x): return self * upatfix(x, self.dtype).recip()
  def __rtruediv__(self, x): return self.recip() * upatfix(x, self.dtype)
  def __lt__(self, x): return UPat.alu(BinaryOps.CMPLT, (self, upatfix(x, self.dtype)))
  def __ne__(self, x): return UPat.alu(BinaryOps.CMPNE, [self, upatfix(x, self.dtype)])
  def __mod__(self, x): return UPat.alu(BinaryOps.MOD, (self, upatfix(x, self.dtype)))
  def __rmod__(self, x): return UPat.alu(BinaryOps.MOD, (upatfix(x, self.dtype), self))
UPatVin = Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]
UPatDType = Optional[Union[DType, Set[DType]]]

T = TypeVar("T")
def __unmatch(m1:Union[T, Set[T]], m2:T) -> bool:
  if isinstance(m1, set):
    if m2 not in m1: return True
  elif m2 != m1: return True
  return False

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> bool:
  if pat.name in store and store[pat.name] is not uop: return False
  if pat.name is not None: store[pat.name] = uop
  if pat.arg is not None and __unmatch(pat.arg, uop.arg): return False
  if pat.dtype is not None and uop.dtype is not None and __unmatch(pat.dtype, uop.dtype): return False
  if pat.uop is not None and __unmatch(pat.uop, uop.uop): return False
  if pat.vin is None: return True
  # only one if it's a tuple
  # try all permutations if it's a list
  # repeat if it's a UPat
  for vp in itertools.permutations(pat.vin) if isinstance(pat.vin,list) else ([pat.vin] if isinstance(pat.vin,tuple) else [(pat.vin,)*len(uop.vin)]):
    if len(uop.vin) != len(vp) and (len(uop.vin) not in pat.allow_len): return False
    new_store = store.copy()
    if all(_match(uu, vv, new_store) for uu, vv in zip(uop.vin, vp)):
      store.update(new_store)
      return True
  return False

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.uop is not None
      if isinstance(p.uop, set):
        for uop in p.uop: self.pdict[(uop, p.arg)].append((p, fxn))
      else:
        self.pdict[(p.uop, p.arg)].append((p, fxn))

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.uop, uop.arg)], self.pdict[(uop.uop, None)]):
      store: Dict[str, UOp] = {}
      if _match(uop, p, store): return fxn(**store)
    return None

def sum_collapse(phi_input, loop, val1, val2):
  for v1,v2 in [(val1, val2), (val2, val1)]:
    if loop not in v1.parents:
      loop_range = loop.vin[1]-loop.vin[0]
      ret = v1*loop_range.cast(v1.dtype)
      return UOp(UOps.PHI, phi_input.dtype, (phi_input, v2))+ret
  return None

def loop_collapse(loop_start, loop_end, compval, idx, mval, multconst):
  if mval.arg >= 0 or loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  comprange = UOp.min(loop_end, UOp.max(UOp.alu(BinaryOps.IDIV, idx-compval-mval, mval) + (loop_end-loop_start), loop_start))
  return UOp(UOps.UNMUL, multconst.dtype, (comprange.cast(multconst.dtype) * multconst, loop_end-loop_start))

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # arange loop folding (early)
  (UPat(UOps.ALU, TernaryOps.WHERE, vin=(UPat(UOps.ALU, BinaryOps.CMPLT, vin=(
    UPat(UOps.ALU, BinaryOps.ADD, vin=[UPat(name="idx"), UPat(UOps.ALU, BinaryOps.MUL,
      vin=[UPat(UOps.CONST, name="mval"), UPat(UOps.RANGE, vin=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
      UPat(UOps.CONST, name="compval"))), UPat(UOps.CONST, name="multconst"), UPat(UOps.CONST, 0))), loop_collapse),
  # sum collapse to mul (with possible GEP)
  (UPat(UOps.PHI, vin=(UPat(UOps.DEFINE_ACC, name="phi_input", vin=(UPat(UOps.RANGE, name="loop"),)), UPat.var("val1") + UPat.var("val2"))),
                                                                                                                                        sum_collapse),
  (UPat(UOps.PHI, vin=(UPat(UOps.GEP, name="phi_input", vin=(UPat(UOps.DEFINE_ACC, vin=(UPat(UOps.RANGE, name="loop"),)),)),
                       UPat.var("val1") + UPat.var("val2"))), sum_collapse),
  # deal with UNMUL
  (UPat.cvar("c1") * UPat(UOps.UNMUL, vin=[UPat.cvar("c2"), UPat.var("v")]), lambda c1,c2,v: v if c1.arg == c2.arg else None),
  (UPat(UOps.UNMUL, vin=(UPat.const(0, name="zero"), UPat())), lambda zero: zero),
  (UPat(UOps.CAST, name="root", vin=(UPat(UOps.UNMUL, name="unmul"),)),
    lambda root,unmul: UOp(UOps.UNMUL, root.dtype, (unmul.vin[0].cast(root.dtype), unmul.vin[1]))),
  # max on special can go away (TODO: special should be variable, same thing applies)
  (UPat.alu(BinaryOps.MAX, [UPat.cvar("c"), UPat(UOps.SPECIAL, name="s")]), lambda c,s: c if (s.arg[2]-1) <= c.arg else None),
  # const rules
  (UPat(UOps.GEP, name="root", vin=(UPat.cvar("c"),)), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.CAST, name="root", vin=UPat.cvar("c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi on a DEFINE_ACC without loops or a CONST is a noop. this is for correctness, not just speed
  (UPat(UOps.PHI, vin=(UPat(UOps.DEFINE_ACC, name="acc"), UPat.var("acc"))), lambda acc: UOp.const(acc.dtype, acc.arg[0])),
  (UPat(UOps.PHI, vin=(UPat(UOps.DEFINE_ACC, vin=tuple()), UPat.var("x"))), lambda x: x),
  (UPat(UOps.PHI, vin=(UPat(UOps.CONST), UPat.var("x"))), lambda x: x),
  # a DEFINE_ACC without inputs is a const + GEP on a const is the const
  (UPat(UOps.DEFINE_ACC, name="root", vin=tuple()), lambda root: UOp.const(root.dtype, root.arg[0])),
  (UPat(UOps.GEP, name="root", vin=(UPat.cvar("x"),)), lambda root,x: UOp.const(root.dtype, x.arg)),
  # max -2147483648
  (UPat.alu(BinaryOps.MAX, [UPat.var("x"), UPat.const(-2147483648, dtypes.int)]), lambda x: x),
  # -(-x) -> x
  (-(-UPat.var('x')), lambda x: x),
  # x+-y -> x-y
  (UPat.var('x')+(-UPat.var('y')), lambda x, y: x-y),
  # -1*x -> -x
  (-1*UPat.var('x'), lambda x: -x),
  # bool < False is always false, True < bool is always false
  (UPat(dtype=dtypes.bool) < UPat.const(False), lambda: UOp.const(dtypes.bool, False)),
  (UPat.const(True) < UPat(dtype=dtypes.bool), lambda: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.where(UPat(), UPat.var("x"), UPat.var("x")), lambda x: x),
  (UPat.where(UPat.cvar('gate'), UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat.alu(vin=UPat.const(), name='root'), lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.vin]))),
  # ** self folding **
  (UPat.var('x')+0, lambda x: x),
  (UPat.var('x')-0, lambda x: x),
  (UPat.var('x')*1, lambda x: x),
  (UPat.var('x')//1, lambda x: x),
  (UPat.var('x')//-1, lambda x: -x),
  # ** zero folding **
  #x*0 -> 0 or 0*x -> 0
  #if x is nan it should render the nan value.
  (UPat.var('x') * UPat.const(0, name='c'), lambda x,c: x if isinstance(x.arg, float) and math.isnan(x.arg) else c),
  (UPat.var('x') - UPat.var('x'), lambda x: UOp.const(x.dtype, 0)),
  # ** load/store folding **
  (UPat.store(UPat.var('buf'), UPat.var('idx'), UPat.load(UPat.var('buf'), UPat.var('idx'))), lambda buf, idx: UOp(UOps.NOOP)),
  # ** two stage add/sub folding **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.cvar("c2"), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((UPat.var("x") - UPat.cvar("c1")) + UPat.cvar("c2"), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.SUB, x.dtype, [c2.arg, c1.arg]))),
  # *** rules from symbolic ***
  # two stage mul, (x*c1)*c2 = x*(c1*c2)
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.cvar("c2"), lambda x,c1,c2: x*UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # x%1 -> 0
  (UPat.var("x") % UPat.const(1), lambda x: UOp.const(x.dtype, 0)),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)/c0 -> x
  ((UPat.var("x") * UPat.cvar("c0")) // UPat.cvar("c0"), lambda x,c0: x if c0.arg != 0 else None),
  # (x/c0)/c1 -> x/(c0*c1)
  ((UPat.var("x") // UPat.cvar("c0")) // UPat.cvar("c1"), lambda x,c0,c1: x//UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # c0 + x < c1 -> x < c1 - c0
  (UPat.cvar("c0") + UPat.var("x") < UPat.cvar("c1"),
   lambda x,c0,c1: UOp.alu(BinaryOps.CMPLT, x, UOp.const(x.dtype, exec_alu(BinaryOps.SUB, x.dtype, [c1.arg, c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c0"), lambda x,c0: x*UOp.const(x.dtype, c0.arg+1)),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.where(UPat.var("gate"), UPat.var("alt"), UPat.load(UPat.var("buf"), UPat.var("idx")))),
    lambda buf, idx, gate, alt: UOp(UOps.STORE, None, (buf, idx, alt, gate))),
  # store float4/float2 directly (remove CAST/GEP) (those two probably can be unified)
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat(UOps.CAST, vin=tuple(UPat(UOps.GEP, i, vin=(UPat.var("val"),)) for i in range(4)))),
   lambda buf,idx,val: UOp(UOps.STORE, None, (buf, idx, val))),
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat(UOps.CAST, vin=tuple(UPat(UOps.GEP, i, vin=(UPat.var("val"),)) for i in range(2)))),
   lambda buf,idx,val: UOp(UOps.STORE, None, (buf, idx, val))),
  # CAST-PHI-GEP -> PHI-CAST
  (UPat(UOps.CAST, name="root", vin=tuple(UPat(UOps.PHI, vin=(UPat(UOps.GEP, i, vin=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(4))),
    lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.CAST, val.dtype, (v0, v1, v2, v3))))),
  (UPat(UOps.CAST, name="root", vin=tuple(UPat(UOps.PHI, vin=(UPat(UOps.GEP, i, vin=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(2))),
    lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.CAST, val.dtype, (v0, v1))))),
  # NEG/CMPLT -> CMPLT
  (UPat(UOps.ALU, BinaryOps.CMPLT, (UPat(UOps.ALU, UnaryOps.NEG, (UPat(name="x"),)), UPat(UOps.CONST, name="c", dtype=dtypes.int))),
    lambda c,x: UOp(UOps.ALU, dtypes.bool, (UOp.const(c.dtype, -c.arg), x), BinaryOps.CMPLT)),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (UPat(UOps.CAST, name="root"), lambda root: root.vin[0] if str(root.dtype) == str(root.vin[0].dtype) else None),
])

# *** uop graph ***

class UOpGraph:
  def __init__(self, sinks:List[UOp]):
    self.sinks: List[UOp] = sinks
    # used by linearizer
    self._uops: Optional[List[UOp]] = None

  def __iter__(self) -> Iterator[UOp]: return iter(self.uops)
  def __getitem__(self, index) -> UOp: return self.uops[index]

  def vars(self) -> List[Variable]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_VAR]
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_GLOBAL]

  @property
  def uops(self):
    if self._uops is None: self.linearize()
    return self._uops

  def graph(self):
    from tinygrad.engine.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for i,u in enumerate(self):
      print(f"{i:4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str([self.uops.index(x) for x in u.vin]):32s} {u.arg}")

  def graph_rewrite(self, sink, pm):
    # recursive rewrite
    changed = getenv("UOPS_REWRITE", 1)
    run_cnt = 0
    while changed:
      changed = 0
      @functools.lru_cache
      def rewrite(u:UOp) -> UOp:
        nonlocal changed
        recurse_cnt = 0
        up = u
        # locally recursively rewrite
        while (rewritten := pm.rewrite(up)):
          assert recurse_cnt < 100, f"recursive_rewrite looped {up} <--> {rewritten}"
          up = rewritten
          recurse_cnt += 1
        changed += recurse_cnt
        # NOTE: this changes UOp, so we have to delete caches
        up.vin = tuple(rewrite(x) for x in up.vin)
        if 'parents' in up.__dict__: delattr(up, 'parents')
        if 'cmp_tuple' in up.__dict__: delattr(up, 'cmp_tuple')
        # replace with cached nodes
        return self.nodes.setdefault(up.tuple(), up)
      sink = rewrite(sink)
      run_cnt += 1
      assert run_cnt < 100, "exceeded 100 rewrite loops!"
    return sink

  def graph_dedup(self, sink):
    # add nodes to graph in reverse BFS order
    # dedup all nodes
    # TODO: i feel like this BFS is written in a few places, possible to library it?
    unprocessed_nodes = [sink]
    early_in_degree: DefaultDict[UOp, int] = defaultdict(int)
    children: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    all_nodes: Dict[UOp, None] = dict()
    while len(unprocessed_nodes):
      n = unprocessed_nodes.pop(0)
      if n in all_nodes: continue
      all_nodes[n] = None
      for x in n.vin:
        early_in_degree[n] += 1
        children[x].append(n)
      unprocessed_nodes += list(n.vin)
    early_queue = [x for x in all_nodes if early_in_degree[x] == 0]
    replace_nodes: Dict[UOp, UOp] = {}
    while len(early_queue):
      n = early_queue.pop(0)
      if n in replace_nodes: continue
      key = (n.uop, n.dtype, tuple(replace_nodes.get(x, x) for x in n.vin), n.arg)
      if found:=self.nodes.get(key): replace_nodes[n] = found
      else: replace_nodes[n] = self.nodes[key] = UOp(*key)
      for x in children[n]:
        early_in_degree[x] -= 1
        if early_in_degree[x] == 0:
          early_queue.append(x)
    return replace_nodes.get(sink, sink)

  def linearize(self, extra_pm:Optional[PatternMatcher]=None, type_verify=True):
    # NOTE: relinearizering should be okay
    #assert self._uops is None, "already linearized"
    self.nodes: Dict[Tuple, UOp] = {}

    # dedup all nodes in graph
    sink = self.graph_dedup(UOp(UOps.SINK, None, tuple(self.sinks)))

    # do graph rewrite
    sink = self.graph_rewrite(sink, constant_folder)
    if extra_pm: sink = self.graph_rewrite(sink, PatternMatcher(constant_folder.patterns+extra_pm.patterns))

    # filter nodes that don't link to a sink
    # BFS toposort
    graph: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    in_degree: DefaultDict[UOp, int] = defaultdict(int)
    loops = []
    ifs = []
    nodes: Dict[UOp, None] = {}
    def add_parents(u:UOp):
      if u in nodes: return
      nodes[u] = None
      for x in u.vin:
        add_parents(x)
        in_degree[u] += 1
        graph[x].append(u)
      if u.uop is UOps.RANGE: loops.append(u)
      if u.uop is UOps.IF: ifs.append(u)
    sink = UOp(UOps.SINK, None, tuple(x for x in sink.vin if x.uop is not UOps.NOOP))
    add_parents(sink)

    @functools.lru_cache(None)
    def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
      if x.uop is UOps.SINK: return set()
      return set.union(set((x,)) if include_self else set(), *([get_recursive_children(u, end, True) for u in graph[x] if x.uop is not end]))
    # scope children impact the toposort and END* insertion
    end_for_uop = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}
    scope_children = {p:get_recursive_children(p, end_for_uop[p.uop][0]) for p in (loops+ifs)[::-1]}

    queue: List = []
    def push(u):
      priority = 0
      # prefer uops that are loop children
      for l, ss in scope_children.items():
        if l.uop is UOps.RANGE and u in ss: priority -= l.arg[0]*1000 + l.arg[1]
      heapq.heappush(queue, (priority, u))

    for u in nodes:
      if in_degree[u] == 0: push(u)

    if getenv("FUZZ_UOPS", 0):
      from test.external.fuzz_uops import fuzz_uops
      self.fuzz_paths = fuzz_uops(graph, in_degree.copy(), scope_children)

    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(p,x)
      if x.uop is UOps.DEFINE_ACC and len(x.vin):
        idx = min([self._uops.index(l) for l in x.vin])
        self._uops.insert(idx, x)
      else:
        self._uops.append(x)
      for u, ss in scope_children.items():
        if x in ss:
          ss.remove(x)
          if len(ss) == 0: self._uops.append(UOp(end_for_uop[u.uop][1], None, (u,)))
      for u in graph[x]:
        in_degree[u] -= 1
        if in_degree[u] == 0: push(u)

    assert self._uops[-1].uop is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
    self._uops = self._uops[:-1]

    if type_verify: self.type_verify()

  # *** checker functions ***

  def flops_mem(self) -> Tuple[sint, sint]:
    flops: sint = 0
    mem: sint = 0
    mults: sint = 1
    mult_stack = []
    for u in self.uops:
      if u.uop is UOps.RANGE:
        mult_stack.append(mults)
        mults *= uop_alu_resolve(u.vin[1])
      elif u.uop is UOps.ENDRANGE:
        mults = mult_stack.pop(-1)
      elif u.uop is UOps.ALU:
        flops += mults * (2 if u.arg == TernaryOps.MULACC else 1)
      elif u.uop is UOps.LOAD:
        assert u.dtype is not None
        mem += u.dtype.itemsize * mults
      elif u.uop is UOps.STORE:
        assert u.vin[2].dtype is not None
        mem += u.vin[2].dtype.itemsize * mults
      elif u.uop is UOps.WMMA:
        assert u.arg[1] is not None
        flops += 2 * prod(u.arg[1]) // 32 * mults
    return flops, mem

  def type_verify(self):
    for u in self.uops:
      uop, arg, vin, dtype = u.uop, u.arg, u.vin, u.dtype
      if uop in {UOps.CONST, UOps.DEFINE_ACC}:
        if uop is UOps.DEFINE_ACC: arg = arg[0]
        assert dtype is not None and type(arg) is type(dtypes.as_const(arg, dtype)), f"type of {arg=} does not match {dtype}"
      if uop in {UOps.CAST, UOps.BITCAST}: assert arg is None   # type is the output type, not an arg
      if uop is UOps.ALU:
        if arg in UnaryOps:
          assert dtype == vin[0].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=}"
        elif arg in (BinaryOps.CMPLT, BinaryOps.CMPNE):
          assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
          assert vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg is BinaryOps.IDIV:
          assert dtypes.is_int(vin[0].dtype) and dtypes.is_int(vin[1].dtype), \
              f"input dtype mismatch {dtypes.int} != {vin[0].dtype=} != {vin[1].dtype=}"
          assert dtypes.is_int(dtype), f"{arg} output dtype mismatch {dtype=} != {dtypes.int}"
        elif arg in BinaryOps:
          assert dtype == vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg == TernaryOps.WHERE:
          assert vin[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {vin[0].dtype=} != {dtypes.bool}"
          assert dtype == vin[1].dtype == vin[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {vin[1].dtype=} != {vin[2].dtype=}"
