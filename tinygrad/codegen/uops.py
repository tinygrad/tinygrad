from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, List, DefaultDict, Set
import functools, itertools, heapq
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType
from tinygrad.shape.symbolic import sint, Variable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu
from tinygrad.helpers import prod, DEBUG

# the order of these UOps controls the order of the toposort
class UOps(Enum):
  # ops that aren't rendered
  SINK = auto()
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  CONST = auto(); SPECIAL = auto() # noqa: E702
  NOOP = auto(); GEP = auto() # noqa: E702
  # math ops
  CAST = auto(); BITCAST = auto() # noqa: E702
  ALU = auto(); WMMA = auto() # noqa: E702
  # memory/assignment ops
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702
  # control flow ops
  BARRIER = auto(); IF = auto(); LOOP = auto() # noqa: E702
  # these two are not graph nodes
  ENDLOOP = auto(); ENDIF = auto() # noqa: E702

@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType] = None
  vin: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def cmp_tuple(self): return (self.uop.value, self.arg if self.uop is not UOps.ALU else (type(self.uop), self.uop.value), self.dtype, self.vin)
  def __lt__(self, x:UOp): return self.cmp_tuple() < x.cmp_tuple()
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"
  @staticmethod
  def const(dtype, val): return UOp(UOps.CONST, dtype, arg=dtypes.as_const(val, dtype))

def uop_alu_resolve(u:UOp) -> sint:
  if u.uop is UOps.CONST: return u.arg
  elif u.uop is UOps.DEFINE_VAR: return u.arg
  elif u.uop is UOps.ALU and u.arg is BinaryOps.MUL: return uop_alu_resolve(u.vin[0]) * uop_alu_resolve(u.vin[1])
  elif u.uop is UOps.ALU and u.arg is BinaryOps.ADD: return uop_alu_resolve(u.vin[0]) + uop_alu_resolve(u.vin[1])
  else: raise RuntimeError(f"ALU resolve fail @ {u.uop}")

# *** simplification logic ***

def _match(uop:UOp, pattern:Dict[str, Any], store:Dict[str, UOp]) -> bool:
  for k,v in pattern.items():
    if k == "__name__":
      if v in store and store[v] != uop: return False
      store[v] = uop
    elif k == "vin":
      # only one if it's a tuple
      # try all permutations if it's a list
      # repeat if it's a dict
      for vp in itertools.permutations(v) if isinstance(v, list) else ([v] if isinstance(v, tuple) else [(v,)*len(uop.vin)]):
        if len(uop.vin) != len(vp): return False
        new_store = store.copy()
        if all(_match(uu, vv, new_store) for uu, vv in zip(uop.vin, vp)):
          for k,v in new_store.items(): store[k] = v
          return True
      return False
    elif k == "dtype":
      if uop.__getattribute__(k) not in (v if isinstance(v, set) else set([v])): return False
    else:
      if uop.__getattribute__(k) != v: return False
  return True

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Dict[str, Any], Any]]):
    self.patterns = patterns
    self.pdict = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns: self.pdict[(p.get("uop"), p.get("arg", None))].append((p, fxn))

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.uop, uop.arg)], self.pdict[(uop.uop, None)]):
      store: Dict[str, UOp] = {}
      if _match(uop, p, store): return fxn(**store)
    return None

  def recursive_rewrite(self, uop:UOp) -> UOp:
    while rewritten := self.rewrite(uop): uop = rewritten
    return uop

constant_folder = PatternMatcher([
  # const rules
  ({"__name__": "root", "uop": UOps.GEP, "vin": ({"__name__": "c", "uop": UOps.CONST},)}, lambda root, c: UOp.const(root.dtype, c.arg)),
  ({"__name__": "root", "uop": UOps.CAST, "vin": {"__name__": "c", "uop": UOps.CONST}}, lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi on a DEFINE_ACC without loops is a noop. this is for correctness, not just speed
  ({"uop": UOps.PHI, "vin": ({"uop": UOps.DEFINE_ACC, "vin": tuple()}, {"__name__": "x"})}, lambda x: x),
  # a DEFINE_ACC without inputs is a const (broken)
  #({"__name__": "root", "uop": UOps.DEFINE_ACC, "vin": tuple()}, lambda root: UOp.const(root.dtype, root.arg[0])),
  # x+-y -> x-y
  ({"uop": UOps.ALU, "arg": BinaryOps.ADD, "vin": ({"__name__": "x"}, {"__name__": "my", "uop": UOps.ALU, "arg": UnaryOps.NEG})},
    lambda x, my: UOp(UOps.ALU, x.dtype, (x, my.vin[0]), BinaryOps.SUB)),
  # bool < False is always false, True < bool is always false
  ({"uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({}, {"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.bool, "arg": False})}, lambda x: x),
  ({"uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.bool, "arg": True}, {})},
    lambda x: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  ({"uop": UOps.ALU, "arg": TernaryOps.WHERE, "vin": ({}, {"__name__": "val"}, {"__name__": "val"})}, lambda val: val),
  ({"uop": UOps.ALU, "arg": TernaryOps.WHERE, "vin": ({"__name__": "gate", "uop": UOps.CONST}, {"__name__": "c0"}, {"__name__": "c1"})},
    lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  ({"__name__": "root", "uop": UOps.ALU, "vin": {"uop": UOps.CONST}},
    lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.vin]))),
  # ** self folding **
  ({"uop": UOps.ALU, "arg": BinaryOps.ADD, "vin": [{"__name__": "x"}, {"uop": UOps.CONST, "arg": 0}]}, lambda x: x),   # x+0 -> x or 0+x -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [{"__name__": "x"}, {"uop": UOps.CONST, "arg": 1}]}, lambda x: x),   # x*1 -> x or 1*x -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": ({"__name__": "x"}, {"uop": UOps.CONST, "arg": 0})}, lambda x: x),   # x-0 -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.DIV, "vin": ({"__name__": "x"}, {"uop": UOps.CONST, "arg": 1})}, lambda x: x),   # x/1 -> x
  # ** zero folding **
  ({"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [{}, {"__name__": "c", "uop": UOps.CONST, "arg": 0}]}, lambda c: c), # x*0 -> 0 or 0*x -> 0
  ({"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": ({"__name__": "x"}, {"__name__": "x"})}, lambda x: UOp.const(x.dtype, 0)),   # x-x -> 0
  # ** load/store folding **
  ({"uop": UOps.STORE, "vin": ({"__name__": "buf"}, {"__name__": "idx"},
                               {"uop": UOps.LOAD, "vin": ({"__name__": "buf"}, {"__name__": "idx"})})}, lambda buf, idx: UOp(UOps.NOOP)),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  ({"uop": UOps.STORE, "vin": ({"__name__": "buf"}, {"__name__": "idx"}, {"uop": UOps.ALU, "arg": TernaryOps.WHERE,
                       "vin": ({"__name__": "gate"}, {"__name__": "alt"}, {"uop": UOps.LOAD, "vin": ({"__name__": "buf"}, {"__name__": "idx"})})})},
    lambda buf, idx, gate, alt: UOp(UOps.STORE, None, (buf, idx, alt, gate))),
])

# *** uop graph ***

class UOpGraph:
  # TODO: remove start_uops
  def __init__(self, start_uops:Optional[List[UOp]]=None):
    self.nodes: Dict[Tuple, UOp] = {}
    self._uops: Optional[List[UOp]] = start_uops

  def uoptimize(self): pass

  def __iter__(self): return iter(self.uops)

  def vars(self) -> List[Variable]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_VAR]
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_GLOBAL]

  @property
  def uops(self):
    if self._uops is None: self.linearize()
    return self._uops

  def graph(self):
    from tinygrad.features.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for i,u in enumerate(self):
      print(f"{i:4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str([self.uops.index(x) for x in u.vin]):32s} {u.arg}")

  def linearize(self, extra_pm:Optional[PatternMatcher]=None):
    assert self._uops is None, "already linearized"
    pm = PatternMatcher(constant_folder.patterns+extra_pm.patterns) if extra_pm is not None else constant_folder

    # constant fold
    rewrite_map: Dict[UOp, UOp] = {}
    sinks: List[UOp] = []
    for u in self.nodes.values():
      rewrite_map[u] = up = pm.recursive_rewrite(u)
      if up.uop is UOps.STORE: sinks.append(up)
      if up.uop is UOps.SINK: sinks.extend(up.vin)

    # filter nodes that don't link to a sink
    nodes: Dict[UOp, None] = {}
    def add_parents(u:UOp):
      if u in nodes: return
      nodes[u] = None
      u.vin = tuple(rewrite_map[x] for x in u.vin)
      for x in u.vin: add_parents(x)
    add_parents(UOp(UOps.SINK, None, tuple(sinks)))

    # BFS toposort
    graph: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    in_degree: DefaultDict[UOp, int] = defaultdict(int)
    loops = []
    ifs = []
    for u in nodes:
      for x in u.vin:
        in_degree[u] += 1
        graph[x].append(u)
      if u.uop is UOps.LOOP: loops.append(u)
      if u.uop is UOps.IF: ifs.append(u)

    @functools.lru_cache(None)
    def get_recursive_children(x:UOp, include_self=False) -> Set[UOp]:
      if x.uop is UOps.SINK: return set()
      return set.union(set((x,)) if include_self else set(), *([get_recursive_children(u, True) for u in graph[x]] if x.uop is not UOps.PHI else []))
    loops_children = {l:get_recursive_children(l) for l in loops[::-1]}

    queue: List = []
    def push(u):
      priority = 0
      # prefer uops that are loop children
      for ss in loops_children.values():
        if u in ss: priority -= 10
      heapq.heappush(queue, (priority, u))

    for u in nodes:
      if in_degree[u] == 0: push(u)

    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(p,x)
      if x.uop is UOps.DEFINE_ACC and len(x.vin):
        idx = min([self._uops.index(l) for l in x.vin])
        self._uops.insert(idx, x)
      else:
        self._uops.append(x)
      for u, ss in loops_children.items():
        if x in ss:
          ss.remove(x)
          if len(ss) == 0: self._uops.append(UOp(UOps.ENDLOOP, None, (u,)))
      for u in graph[x]:
        in_degree[u] -= 1
        if in_degree[u] == 0: push(u)

    assert self._uops[-1].uop is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
    self._uops = self._uops[:-1]

    # TODO: ifs should be removed and just the store should be gated
    for u in ifs[::-1]: self._uops.append(UOp(UOps.ENDIF, None, (u,)))

    self.type_verify()

  def add(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None,
          cachable=True, insert_before=None, simplify=True) -> UOp:
    if uop is UOps.CONST:
      assert dtype is not None
      arg = dtypes.as_const(arg, dtype) # TODO: this doesn't belong here
    if found:=self.nodes.get(key:=(uop, dtype, vin, arg)): return found
    self.nodes[key] = ret = UOp(*key)
    return ret

  # *** checker functions ***

  def flops_mem(self) -> Tuple[sint, sint]:
    flops: sint = 0
    mem: sint = 0
    mults: sint = 1
    mult_stack = []
    for u in self.uops:
      if u.uop is UOps.LOOP:
        mult_stack.append(mults)
        mults *= uop_alu_resolve(u.vin[1])
      elif u.uop is UOps.ENDLOOP:
        mults = mult_stack.pop(-1)
      elif u.uop is UOps.ALU:
        flops += mults
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
        elif arg in (BinaryOps.CMPLT, BinaryOps.CMPEQ):
          assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
          assert vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg in BinaryOps:
          assert dtype == vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg == TernaryOps.WHERE:
          assert vin[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {vin[0].dtype=} != {dtypes.bool}"
          assert dtype == vin[1].dtype == vin[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {vin[1].dtype=} != {vin[2].dtype=}"
