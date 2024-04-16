import numpy as np
from typing import DefaultDict, Dict, List, Set, TypeVar
from tinygrad.engine.realize import CustomOp, ExecItem, capturing, lower_schedule_item
from tinygrad.helpers import DEBUG, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  toposorts = find_all_toposorts(graph, in_degree)
  if DEBUG >= 1: print(colored(f"fuzzing {len(toposorts)} schedule permutations", "yellow"))

  # setup ground truth
  ground_truth: Dict[LazyBuffer, memoryview] = {}
  # IMPORTANT: freeze prerealized bufs before ScheduleItem exec
  prerealized: Dict[LazyBuffer, memoryview] = {}
  seed = Tensor._seed
  for key in toposorts[0]:
    for out in (ps:=prescheduled[key]).outputs:
      seen.add(out)
      # freeze assign state before exec
      if out.op is LoadOps.ASSIGN: prerealized[out] = out.buffer.as_buffer()
    for x in ps.inputs:
      if x not in ground_truth and x.device != "NPY": prerealized[x] = x.buffer.as_buffer()
    _exec_si(ps, seed)
    for out in ps.outputs: ground_truth[out] = out.buffer.as_buffer()

  # exec and validate each permutation with clean Buffers
  for i, ts in enumerate(toposorts):
    if i == 0: continue
    if DEBUG >= 1: print(colored(f"testing permutation {i}", "yellow"))
    for key in ts:
      for out in (ps:=prescheduled[key]).outputs:
        if out.op is LoadOps.ASSIGN: out.buffer.copyin(prerealized[out])
        else: out.buffer.copyin(np.zeros((out.size, ), dtype=out.dtype.np).data)
      for x in ps.inputs:
        if x in prerealized and x.op is not LoadOps.ASSIGN: x.buffer.copyin(prerealized[x])
      _exec_si(ps, seed)
      for out in ps.outputs:
        outbuf = np.frombuffer(out.buffer.as_buffer(), out.dtype.np)
        try: np.testing.assert_allclose(outbuf, np.frombuffer(ground_truth[out], out.dtype.np), atol=1e-2, rtol=1e-2)
        except Exception as e:
          print(f"FAILED FOR {out}")
          raise e

def _exec_si(ps, seed):
  si = ScheduleItem(ps.ast, tuple(x.buffer for x in ps.outputs if x.size != 0), tuple(x.buffer for x in ps.inputs if x.size != 0))
  ei = ExecItem(lower_schedule_item(si), list(si.outputs+si.inputs))
  if len(capturing): capturing[0].add(ei)
  if isinstance(ei.prg, CustomOp): Tensor._seed = seed
  ei.run()

T = TypeVar("T")
def find_all_toposorts(graph:DefaultDict[T, List[T]], in_degree:DefaultDict[T, int]) -> List[List[T]]:
  visited: Set[T] = set()
  ret: List[List[T]] = []
  path: List[T] = []

  def recurse_paths(path:List[T]):
    for v, d in in_degree.items():
      if d != 0 or v in visited: continue
      for u in graph[v]: in_degree[u] -= 1
      path.append(v)
      visited.add(v)
      recurse_paths(path)
      if len(ret) >= getenv("FUZZ_SCHEDULE_MAX_PATHS", 10): return
      # backtrack
      for u in graph[v]: in_degree[u] += 1
      path.pop()
      visited.remove(v)
    if len(path) == len(in_degree): ret.append([*path])
  recurse_paths(path)

  if len(ret) == 0: raise RuntimeError("detected cycle in the graph")
  # verify all paths are unique
  assert len(ret) == len(set(map(tuple, ret)))
  return ret
