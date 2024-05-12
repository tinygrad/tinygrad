from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, List, DefaultDict, Set
import functools
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType
from tinygrad.shape.symbolic import sint, Variable

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); IF = auto(); ENDLOOP = auto(); ENDIF = auto(); SPECIAL = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); CONST = auto(); BARRIER = auto(); PHI = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto(); BITCAST = auto(); GEP = auto(); NOOP = auto() # noqa: E702

@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType] = None
  vin: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"
  @staticmethod
  def const(dtype, val): return UOp(UOps.CONST, dtype, arg=dtypes.as_const(val, dtype))

class UOpGraph:
  def __init__(self):
    self.nodes: Dict[Tuple, UOp] = {}
    self._uops: Optional[List[UOp]] = None

  def uoptimize(self): pass
  def flops_mem(self) -> Tuple[sint, sint]: return (0,0)

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

  def linearize(self):
    # filter nodes that don't link to a sink
    nodes: List[UOp] = []
    @functools.lru_cache(None)
    def add_parents(u:UOp):
      nodes.append(u)
      for x in u.vin: add_parents(x)
    for u in self.nodes.values():
      if u.uop is UOps.STORE: add_parents(u)

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
    def get_recursive_children(x:UOp) -> Set[UOp]:
      return set.union(set((x,)), *([get_recursive_children(u) for u in graph[x]] if x.uop is not UOps.PHI else []))
    loops_children = {l:get_recursive_children(l) for l in loops}

    self._uops = []
    queue = deque(u for u in nodes if in_degree[u] == 0)
    open_loops = set()
    while queue:
      x = queue.popleft()
      if x.uop is UOps.LOOP:
        # start loops as late as possible
        # if there's only loops left and we can make some progress with this loop, we run it if it's the earliest loop
        if len(queue) and not (all(x.uop is UOps.LOOP for x in queue) and all(x.arg < u.arg for u in queue)):
          queue.append(x)
          continue
        open_loops.add(x)
      if len(queue) and len(open_loops) and all(x not in loops_children[u] for u in open_loops):
        # if it's not a loop child, wait to render it
        queue.append(x)
        continue
      self._uops.append(x)
      for u, ss in loops_children.items():
        if x in ss:
          assert u in open_loops, "loop child found but loop wasn't open"
          ss.remove(x)
          if len(ss) == 0:
            self._uops.append(UOp(UOps.ENDLOOP, None, (u,)))
            open_loops.remove(u)
      for c in graph[x]:
        in_degree[c] -= 1
        if in_degree[c] == 0: queue.append(c)

    # TODO: ifs should be removed and just the store should be gated
    for u in ifs[::-1]: self._uops.append(UOp(UOps.ENDIF, None, (u,)))

  def add(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None,
          cachable=True, insert_before=None, simplify=True) -> UOp:
    if uop is UOps.CONST:
      assert dtype is not None
      arg = dtypes.as_const(arg, dtype) # TODO: this doesn't belong here
    if found:=self.nodes.get(key:=(uop, dtype, vin, arg)): return found
    self.nodes[key] = ret = UOp(*key)
    return ret
