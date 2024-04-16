import numpy as np
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, TypeVar
from tinygrad.buffer import Buffer
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import CustomOp, lower_schedule, capturing
from tinygrad.helpers import DEBUG, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule, create_schedule
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  toposorts = find_all_toposorts(graph, in_degree)
  if DEBUG >= 1: print(colored(f"fuzzing {len(toposorts)} schedule permutation", "yellow"))

  # setup ground truth
  schedules: List[List[ScheduleItem]] = [[] for _ in range(len(toposorts))]
  outputs: DefaultDict[LazyBuffer, List[Buffer]] = defaultdict(list)
  for key in toposorts[0]:
    for out in (ps:=prescheduled[key]).outputs:
      seen.add(out)
      outputs[out].append(out.buffer)
    schedules[0].append(ScheduleItem(ps.ast, tuple(x.buffer for x in ps.outputs if x.size != 0), tuple(x.buffer for x in ps.inputs if x.size != 0)))

  # create new Buffers for each permutation
  for i, ts in enumerate(toposorts[1:]):
    rawbufs: Dict[LazyBuffer, Buffer] = {}
    for key in ts:
      for out in (ps:=prescheduled[key]).outputs:
        rawbufs[out] = Buffer(out.buffer.device, out.buffer.size, out.buffer.dtype)
        if out.op is LoadOps.ASSIGN: rawbufs[out].ensure_allocated().copyin(out.buffer.as_buffer())
        outputs[out].append(rawbufs[out])

      for x in ps.inputs:
        if x not in rawbufs:
          if x.device == "NPY": rawbufs[x] = x.buffer
          # copy the pre realized input
          else: rawbufs[x] = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=x.buffer.as_buffer())
      schedules[i+1].append(ScheduleItem(ps.ast, tuple(rawbufs[x] for x in ps.outputs if x.size != 0),
                                         tuple(rawbufs[x] for x in ps.inputs if x.size != 0)))

  # run all schedules with the same seed
  seed = Tensor._seed
  for i, schedule in enumerate(schedules):
    if DEBUG >= 1: print(colored(f"testing permutation {i}", "yellow"))
    for ei in lower_schedule(schedule):
      if len(capturing): capturing[0].add(ei)
      if isinstance(ei.prg, CustomOp): Tensor._seed = seed
      ei.run()

  # assert all LazyBuffers realized correctly
  for lb, bufs in outputs.items():
    a = Tensor.empty((bufs[0].size,), dtype=bufs[0].dtype)
    b = Tensor.empty((bufs[0].size,), dtype=bufs[0].dtype)
    ast = assert_allclose_ast(a, b)
    for buf in bufs[1:]:
      prg = Device[Device.DEFAULT].get_runner(*ast)
      ret_buf = Buffer(Device.DEFAULT, 1, dtypes.bool).allocate()
      prg.exec([ret_buf, bufs[0], buf])
      del buf
      assert np.frombuffer(ret_buf.as_buffer(), ret_buf.dtype.np), f"FAILED FOR {lb}"

def assert_allclose_ast(a:Tensor, b:Tensor, atol=1e-2, rtol=1e-2):
  diff = (a - b).abs()
  tol = atol + rtol * b.abs()
  return create_schedule([((diff > tol).sum() == 0).lazydata])[-1].ast

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
      if len(ret) >= getenv("FUZZ_SCHEDULE_MAX_PATHS", 5): return
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
