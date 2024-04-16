import numpy as np
from typing import DefaultDict, Dict, List, Set, TypeVar
from tinygrad.buffer import Buffer
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import CustomOp, ExecItem, capturing, lower_schedule_item
from tinygrad.helpers import DEBUG, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule, create_schedule
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.tensor import Tensor

def fuzz_schedule(outs: List[LazyBuffer]):
  graph, in_degree, prescheduled = _graph_schedule(outs, seen:=set())
  toposorts = find_all_toposorts(graph, in_degree)
  if DEBUG >= 1: print(colored(f"fuzzing {len(toposorts)} schedule permutations", "yellow"))

  # setup ground truth
  ground_truth: Dict[LazyBuffer, memoryview] = {}
  assign_bufs: Dict[LazyBuffer, memoryview] = {}
  seed = Tensor._seed
  for key in toposorts[0]:
    for out in (ps:=prescheduled[key]).outputs:
      seen.add(out)
      # freeze assign state before exec
      if out.op is LoadOps.ASSIGN: assign_bufs[out] = out.buffer.as_buffer()
    si = ScheduleItem(ps.ast, tuple(x.buffer for x in ps.outputs if x.size != 0), tuple(x.buffer for x in ps.inputs if x.size != 0))
    ei = ExecItem(lower_schedule_item(si), list(si.outputs+si.inputs))
    if len(capturing): capturing[0].add(ei)
    if isinstance(ei.prg, CustomOp): Tensor._seed = seed
    ei.run()
    for out in ps.outputs: ground_truth[out] = out.buffer.as_buffer()

  # create new Buffers for each permutation
  for i, ts in enumerate(toposorts[1:]):
    if DEBUG >= 1: print(colored(f"testing permutation {i}", "yellow"))
    rawbufs: Dict[LazyBuffer, Buffer] = {}
    for key in ts:
      for out in (ps:=prescheduled[key]).outputs:
        rawbufs[out] = Buffer(out.buffer.device, out.buffer.size, out.buffer.dtype)
        if out in assign_bufs: rawbufs[out].ensure_allocated().copyin(assign_bufs[out])
      for x in ps.inputs:
        if x not in rawbufs:
          if x.device == "NPY": rawbufs[x] = x.buffer
          # copy the pre realized input
          else: rawbufs[x] = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=x.buffer.as_buffer())
      si = ScheduleItem(ps.ast, tuple(rawbufs[x] for x in ps.outputs if x.size != 0), tuple(rawbufs[x] for x in ps.inputs if x.size != 0))
      ei = ExecItem(lower_schedule_item(si), list(si.outputs+si.inputs))
      if len(capturing): capturing[0].add(ei)
      if isinstance(ei.prg, CustomOp): Tensor._seed = seed
      ei.run()
      a = Tensor.empty((ps.outputs[0].size,), dtype=ps.outputs[0].dtype)
      b = Tensor.empty((ps.outputs[0].size,), dtype=ps.outputs[0].dtype)
      ast = assert_allclose_ast(a, b)
      for out in ps.outputs:
        prg = Device[Device.DEFAULT].get_runner(*ast)
        ret_buf = Buffer(Device.DEFAULT, 1, dtypes.bool).allocate()
        gt = Buffer(Device.DEFAULT, out.size, out.dtype, initial_value=ground_truth[out])
        prg.exec([ret_buf, gt, rawbufs[out]])
        try:
          assert np.frombuffer(ret_buf.as_buffer(), dtypes.bool.np)
        except Exception as e:
          gt = np.frombuffer(ground_truth[out], out.dtype.np)
          outbuf = np.frombuffer(rawbufs[out].as_buffer(), out.dtype.np)
          print(gt, outbuf)
          print(f"FAILED FOR {out}")
          raise e

def assert_allclose_ast(a:Tensor, b:Tensor, atol=1e-2, rtol=1e-2):
  if dtypes.is_float(a.dtype):
    tol = atol + rtol * b.abs()
    diff = (a - b).abs() > tol
  else: diff = a - b
  ret = diff.sum() == 0
  return create_schedule([ret.lazydata])[-1].ast

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
