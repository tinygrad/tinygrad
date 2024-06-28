from collections import defaultdict, deque
import functools
from typing import DefaultDict, Dict, List, Tuple
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.api import ScheduleItem
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

def get_children_dfs(u:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], in_degree:Dict[LazyBuffer, int], realizes:Dict[LazyBuffer, None]):
  if u in in_degree: return
  in_degree[u] = 0
  if u.realized is not None: return realizes.setdefault(u, None)
  if u.op in LoadOps: realizes[u] = None
  for x in u.srcs:
    get_children_dfs(x, children, in_degree, realizes)
    children[x][u] = None
  in_degree[u] = len(u.srcs)

def schedule_buffer(n:LazyBuffer, realizes:Dict[LazyBuffer, None]) -> ScheduleItem:
  if n.op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY, LoadOps.VIEW}:
    ast = LazyOp(n.op, (), n.arg)
    return ScheduleItem((ast, ), (n.buffer,)+tuple(x.buffer for x in n.srcs))
  inputs: Dict[LazyBuffer, int] = {}
  @functools.lru_cache(None)
  def _dfs_lazyop(x:LazyBuffer, st:ShapeTracker) -> LazyOp:
    if x != x.base: st, x = x.st+st, x.base
    assert x.op is not None
    if x.op is LoadOps.CONST: return LazyOp(BufferOps.CONST, (), ConstBuffer(x.arg, x.dtype, st.simplify()))
    if x in realizes and x != n:
      if x not in inputs: inputs[x] = len(inputs)
      return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs[x]+1, x.dtype, st.simplify()))
    lop = LazyOp(x.op, tuple(_dfs_lazyop(x, st) for x in n.srcs), x.arg)
    if x == n: lop = LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, st.simplify()))
    return lop
  ast = _dfs_lazyop(n, ShapeTracker.from_shape(n.st.shape))
  return ScheduleItem((ast, ), (n.buffer, )+tuple(x.buffer for x in inputs))

def create_schedule(outs:List[LazyBuffer],_) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  realizes = {out:None for out in outs}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  in_degree: Dict[LazyBuffer, int] = {}
  for out in outs: get_children_dfs(out, children, in_degree, realizes)
  schedule: List[ScheduleItem] = []
  queue = deque(x for x, d in in_degree.items() if d == 0)
  while queue:
    n = queue.popleft()
    if n.realized is None and n.op is not LoadOps.CONST:
      schedule.append(schedule_buffer(n, realizes))
      del n.srcs
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  assert all(d == 0 for d in in_degree.values())
  return schedule, {}
