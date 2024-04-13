from collections import defaultdict
import random, pickle
import numpy as np
from typing import DefaultDict, Dict, List, Set, Tuple, TypeVar
from tinygrad.buffer import Buffer
from tinygrad.device import CompiledRunner
from tinygrad.engine.realize import lower_schedule_item, run_schedule
from tinygrad.helpers import DEBUG, GlobalCounters, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.ops import ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  toposorts = find_all_toposorts(graph, in_degree)
  if DEBUG >= 2: print(colored(f"fuzzing {len(toposorts)} toposorts", "yellow"))

  schedules: List[List[ScheduleItem]] = [[] for _ in toposorts]
  fuzz_outputs: DefaultDict[LazyBuffer, List[Buffer]] = defaultdict(list)
  realize_order: List[List[Tuple[LazyBuffer, ScheduleItem]]] = []
  for i, ts in enumerate(toposorts):
    realizes: List[Tuple[LazyBuffer, ScheduleItem]] = []
    for key in ts:
      for out in (ps:=prescheduled[key]).outputs:
        seen.add(out)
        # the first ts changes the LazyBuffer's .buffer
        fuzz_outputs[out].append(out.buffer if i == 0 else Buffer(out.device, out.size, out.dtype))
      inputs = (x.buffer if hasattr(x.buffer, "_buf") else fuzz_outputs[x][i] for x in ps.inputs if x.size != 0)
      schedules[i].append(si:=ScheduleItem(ps.ast, tuple(fuzz_outputs[x][i] for x in ps.outputs if x.size != 0), tuple(inputs)))
      for x in ps.outputs: realizes.append((x,si))
    realize_order.append(realizes)

  # seed is the same between runs
  seed = Tensor._seed
  for i, schedule in enumerate(schedules):
    Tensor.manual_seed(seed)
    if DEBUG >= 2: print(f"toposort permutation {i}")
    # keep the permutation in memory
    run_schedule(schedule.copy())
    GlobalCounters.reset()

  all_passed = True
  for lb, rawbufs in fuzz_outputs.items():
    for i, buf in enumerate(rawbufs):
      ground_truth, output = (np.frombuffer(buf.as_buffer(), buf.dtype.np) for buf in [rawbufs[0], buf])
      try: np.testing.assert_allclose(output, ground_truth, atol=1e-2, rtol=1e2)
      except AssertionError as e:
        print(f"COMPARE FAILED FOR REALIZE {lb} at permutation {i}")
        print(e)
        all_passed = False

  if not all_passed:
    _graph_fuzz(realize_order)
    raise Exception("some toposorts failed")
  if DEBUG >= 2: print(colored("all toposorts all_passed", "green"))

def _graph_fuzz(realize_order: List[List[Tuple[LazyBuffer, ScheduleItem]]]):
  color: Dict[LazyBuffer, str] = {}
  toposorts = {"nodes": [], "edges": []}
  for sort_idx, realizes in enumerate(realize_order):
    for realize_idx, (r,si) in enumerate(realizes):
      if r not in color: color[r] = _random_hex()
      runner = lower_schedule_item(si)
      code = runner.prg if isinstance(runner, CompiledRunner) else ""
      toposorts["nodes"].append({"id": (node_id:=f"{sort_idx}_{realize_idx}"), "fill": color[r], "lb": f"{r.op} {r.shape} {r.device}", "code": code})
      if realize_idx < len(realizes)-1:
        toposorts["edges"].append({"id": f"{node_id}_{realize_idx+1}", "source": node_id, "target": f"{sort_idx}_{realize_idx+1}"})
  pickle.dump(toposorts, open(fp:="/Users/qazal/toposorts.tiny", "wb"))
  print(f"saved toposorts to {fp}")

def _random_hex(): return f'#{random.randint(0, 0xFFFFFF):06x}'

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
