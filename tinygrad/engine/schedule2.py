from collections import defaultdict, deque
import functools
from typing import DefaultDict, Dict, List, Tuple
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.api import ScheduleItem
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

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
    if x.realized is not None or x in realizes and x != n:
      if x not in inputs: inputs[x] = len(inputs)
      return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs[x]+1, x.dtype, st.simplify()))
    lop = LazyOp(x.op, tuple(_dfs_lazyop(src, st) for src in x.srcs), x.arg)
    if x == n: lop = LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, st.simplify()))
    return lop
  ast = _dfs_lazyop(n, ShapeTracker.from_shape(n.st.shape))
  return ScheduleItem((ast, ), (n.buffer, )+tuple(x.buffer for x in inputs))

def create_schedule(outs:List[LazyBuffer],_) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  realizes = {out:None for out in outs}
  @functools.lru_cache(None)
  def _dfs_realizes(u:LazyBuffer):
    if u.base.realized is not None: return
    if u != u.base:
      realizes[u.base] = None
      return _dfs_realizes(u.base)
    if u.op in LoadOps or u.forced_realize: realizes[u] = None
    for x in u.srcs: _dfs_realizes(x)
  for out in outs: _dfs_realizes(out)

  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  #@functools.lru_cache(None)
  def _get_parents_dfs(n:LazyBuffer, child:LazyBuffer, first=False):
    in_degree[n] = 0
    if n.realized is not None: return n
    if n in realizes and not first:
      children[n][child] = None
      in_degree[child] += 1
      return _get_parents_dfs(n, n, first=True)
    for x in n.srcs: _get_parents_dfs(x.base, child)
  for out in outs: _get_parents_dfs(out, out, first=True)

  schedule: List[ScheduleItem] = []
  queue = deque(x for x, d in in_degree.items() if d == 0)
  while queue:
    n = queue.popleft()
    if n.realized is None and n.op is not LoadOps.CONST:
      print(n)
      schedule.append(schedule_buffer(n, realizes))
      del n.srcs
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  assert all(d == 0 for d in in_degree.values())
  return schedule, {}
