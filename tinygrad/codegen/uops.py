from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, List, DefaultDict
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

  def print(self):
    for i,u in enumerate(self):
      print(f"{i:4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str([self._uops.index(x) for x in u.vin]):32s} {u.arg}")

  def linearize(self):
    # BFS toposort
    graph: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    in_degree: DefaultDict[UOp, int] = defaultdict(int)
    for u in self.nodes.values():
      for x in u.vin:
        in_degree[u] += 1
        graph[x].append(u)

    self._uops = []
    queue = deque(u for u in self.nodes.values() if in_degree[u] == 0)
    print(len(queue))
    while queue:
      x = queue.popleft()
      print(x)
      self._uops.append(x)
      for c in graph[x]:
        in_degree[c] -= 1
        if in_degree[c] == 0:
          queue.append(c)

  def add(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None,
          cachable=True, insert_before=None, simplify=True) -> UOp:
    if found:=self.nodes.get(key:=(uop, dtype, vin, arg)): return found
    self.nodes[key] = ret = UOp(*key)
    return ret
