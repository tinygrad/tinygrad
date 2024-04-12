from collections import defaultdict
import numpy as np
from typing import DefaultDict, List, Set, TypeVar, Dict
from tinygrad.buffer import Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.helpers import DEBUG, GlobalCounters, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.ops import ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  sorts = find_all_sorts(graph, in_degree)
  if DEBUG >= 2: print(colored(f"fuzzing {len(sorts)} toposorts", "yellow"))

  schedules: List[List[ScheduleItem]] = [[] for _ in sorts]
  fuzz_outputs: DefaultDict[LazyBuffer, List[Buffer]] = defaultdict(list)
  for i, s in enumerate(sorts):
    scheduled_bufs: Dict[LazyBuffer, Buffer] = {}
    for key in s:
      for buf in (ps:=prescheduled[key]).outputs: seen.add(buf)
      for x in ps.outputs: scheduled_bufs[x] = Buffer(x.device, x.size, x.dtype) if i > 0 else x.buffer
      inputs = tuple(x.buffer if hasattr(x.buffer, "_buf") else scheduled_bufs[x] for x in ps.inputs if x.size != 0)
      schedules[i].append(ScheduleItem(ps.ast, tuple(scheduled_bufs[x] for x in ps.outputs if x.size != 0), inputs))
      for out in ps.outputs: fuzz_outputs[out].append(scheduled_bufs[out])

  # seed is the same between runs
  seed = Tensor._seed
  for i, schedule in enumerate(schedules):
    Tensor.manual_seed(seed)
    if DEBUG >= 2: print(f"toposort permutation {i}")
    run_schedule(schedule.copy())
    GlobalCounters.reset()

  all_passed = True
  for lb, bufs in fuzz_outputs.items():
    for i, buf in enumerate(bufs):
      ground_truth, output = np.frombuffer(bufs[0].as_buffer(), bufs[0].dtype.np), np.frombuffer(buf.as_buffer(), buf.dtype.np)
      try: np.testing.assert_allclose(output, ground_truth, atol=1e-2, rtol=1e2)
      except AssertionError as e:
        print(f"COMPARE FAILED FOR REALIZE {lb}")
        curr_si = [si for si in schedules[i] if buf in si.outputs][0]
        baseline_si = [si for si in schedules[0] if bufs[0] in si.outputs][0]
        print(f"ground truth realized at {schedules[0].index(baseline_si)}, this realize was at {schedules[i].index(curr_si)}")
        print(e)
        all_passed = False

  if not all_passed: raise Exception("some toposorts failed")
  if DEBUG >= 2: print(colored("all toposorts all_passed", "green"))

T = TypeVar("T")
def find_all_sorts(graph:DefaultDict[T, List[T]], in_degree:DefaultDict[T, int]) -> List[List[T]]:
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
  # backtrack cleanup
  for v in in_degree: in_degree[v] = 0
  return ret
