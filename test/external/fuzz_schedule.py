from collections import defaultdict
import numpy as np
from typing import DefaultDict, Dict, List, Set, TypeVar
from tinygrad.buffer import Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.helpers import getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  toposorts = find_all_toposorts(graph, in_degree)

  fuzz_outputs: DefaultDict[LazyBuffer, List[Buffer]] = defaultdict(list)

  ts = toposorts[0]
  schedule: List[ScheduleItem] = []
  for key in ts:
    ps = prescheduled[key]
    outbufs = tuple(x.buffer for x in ps.outputs if x.size != 0)
    inbufs = tuple(x.buffer for x in ps.inputs if x.size != 0)
    schedule.append(ScheduleItem(ps.ast, outbufs, inbufs))
    for x in ps.outputs: fuzz_outputs[x].append(x.buffer)

  schedule_copy: List[ScheduleItem] = []
  realizes: Dict[LazyBuffer, Buffer] = {}
  for key in ts:
    ps = prescheduled[key]
    outputs: List[Buffer] = []
    for x in ps.outputs:
      if x.op is LoadOps.ASSIGN: 
        rawbuf = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=x.buffer.as_buffer())
      else:
        assert x not in realizes
        rawbuf = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype).allocate()
        rawbuf.copyin(np.zeros((x.buffer.size, ), x.buffer.dtype.np).data)
      realizes[x] = rawbuf
      outputs.append(rawbuf)
      fuzz_outputs[x].append(rawbuf)

    inputs: List[Buffer] = []
    for x in ps.inputs:
      if hasattr(x.buffer, "_buf") and x.op is not LoadOps.ASSIGN:
        if x.device == "NPY": rawbuf = x.buffer
        else: rawbuf = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=x.buffer.as_buffer())
      else: rawbuf = realizes[x]
      inputs.append(rawbuf)
    schedule_copy.append(ScheduleItem(ps.ast, tuple(outputs), tuple(inputs)))

  seed = Tensor._seed
  run_schedule(schedule.copy())
  Tensor.manual_seed(seed)
  run_schedule(schedule_copy.copy())

  for lb, rawbufs in fuzz_outputs.items():
    ground_truth = np.frombuffer(rawbufs[0].as_buffer(), rawbufs[0].dtype.np)
    print(lb)
    for buf in rawbufs[1:]:
      print(ground_truth[:4])
      print(np.frombuffer(buf.as_buffer(), buf.dtype.np)[:4])
      np.testing.assert_allclose(ground_truth, np.frombuffer(buf.as_buffer(), buf.dtype.np), atol=1e-2, rtol=1e-2)

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
      if len(ret) >= getenv("FUZZ_SCHEDULE_MAX_PATHS", 500): return
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
