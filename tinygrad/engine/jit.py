from __future__ import annotations
from typing import TypeVar, Generic, Callable, List, Tuple, Union, Dict, cast, Optional
import functools, itertools, operator
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, GRAPH, BEAM, getenv, all_int, GraphException
from tinygrad.device import Buffer, CompiledRunner, BufferXfer, Compiled, MultiDeviceJITGraph, Device
from tinygrad.dtype import DType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sint
from tinygrad.engine.realize import ExecItem, capturing, _internal_memory_planner
from tinygrad.nn.state import get_parameters
from weakref import WeakKeyDictionary

# TODO: these graph functions probably shouldn't exist here

def get_jit_stats(jit_cache: List[ExecItem]) -> Tuple[sint, int]:
  return functools.reduce(operator.add, [ji.prg.op_estimate for ji in jit_cache if isinstance(ji.prg, CompiledRunner)], 0), \
         functools.reduce(operator.add, [ji.prg.mem_estimate for ji in jit_cache if isinstance(ji.prg, CompiledRunner)], 0)
def get_jc_idxs_with_updatable_launch_dims(jit_cache: List[ExecItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledRunner) and \
          ((ji.prg.global_size and not all_int(ji.prg.global_size)) or (ji.prg.local_size and not all_int(ji.prg.local_size)))]
def get_jc_idxs_with_updatable_var_vals(jit_cache: List[ExecItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledRunner) and ji.prg.vars]
def apply_graph_to_jit(jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]) -> List[ExecItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  max_batch_size = getenv("JIT_BATCH_SIZE", 32)
  graphed_jit_cache: List[ExecItem] = []
  current_batch: List[ExecItem] = []
  current_device: Optional[Compiled] = None

  def flush_batch():
    nonlocal current_batch, current_device, max_batch_size
    try:
      if len(current_batch) <= 1 or current_device is None: raise GraphException("only one kernel doesn't graph")
      graphed_jit_cache.append(ExecItem(current_device.graph(current_batch, input_rawbuffers, var_vals), cast(List[Optional[Buffer]], input_rawbuffers))) # noqa: E501
      max_batch_size *= 2
      if DEBUG >= 2: print(f"\tJIT GRAPHing batch with {len(current_batch)} kernels on device {current_device}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"\tJIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_device}: {e}")
    current_batch = []
    current_device = None

  for ji in jit_cache:
    ji_graph_dev: Optional[Compiled] = None # device on which the ji will be graphed. Not graphed if None.
    if isinstance(ji.prg, CompiledRunner): ji_graph_dev = ji.prg.device
    elif isinstance(ji.prg, BufferXfer) and ji.rawbufs[0] and ji.rawbufs[0].device.split(":", 1)[0] in {"HSA", "CUDA"}:
      ji_graph_dev = Device[ji.rawbufs[0].device]

    can_be_graphed = ji_graph_dev and ji_graph_dev.graph
    can_extend_graph_batch = can_be_graphed and len(current_batch) < max_batch_size and (ji_graph_dev == current_device or
      (isinstance(ji_graph_dev.graph, type) and issubclass(ji_graph_dev.graph, MultiDeviceJITGraph) and type(ji_graph_dev) == type(current_device))) #type:ignore
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()

    if can_be_graphed: current_batch.append(ji)
    else: graphed_jit_cache.append(ji)

    current_device = ji_graph_dev

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: List[ExecItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.rawbufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def add_buffer(self, b:Buffer) -> Buffer:
    if found:=self.buffer_replace.get(b, None): return found
    if b.is_allocated() or b.lb_refcount > 0: return b
    self.buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options)
    return ret

  def add(self, ei:ExecItem):
    self.jit_cache.append(ExecItem(ei.prg, [self.add_buffer(buf) for buf in ei.rawbufs if buf is not None]))

  def reset(self):
    self.jit_cache: List[ExecItem] = []
    self.input_replace: Dict[Tuple[int, int], int] = {}
    self.buffer_replace: WeakKeyDictionary[Buffer, Buffer] = WeakKeyDictionary()
    self.cnt: int = 0

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: List[Tuple[Union[int, str], Tensor]] = \
      [(cast(Union[int, str], k),v) for k,v in itertools.chain(enumerate(args), sorted(kwargs.items())) if v.__class__ is Tensor]
    Tensor.corealize([x[1] for x in input_tensors])
    lbs: List[LazyBuffer] = flatten([v.lazydata.lbs for _,v in input_tensors])
    expected_sts_var_dtype_device = [(*x.st.unbind(), x.dtype, x.device) for x in lbs]
    input_rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
    assert len(set(input_rawbuffers)) == len(input_rawbuffers), "duplicate inputs to JIT"
    var_vals: Dict[Variable, int] = merge_dicts([x[1] for x in expected_sts_var_dtype_device] + \
                                                [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])

    expected_names, expected_lbs = [x[0] for x in input_tensors], [(x[0], tuple(x[1].keys()), x[2], x[3]) for x in expected_sts_var_dtype_device]
    if self.cnt >= 2:
      # jit exec
      assert self.expected_names == expected_names and self.expected_lbs == expected_lbs, "args mismatch in JIT"
      for (j,i),input_idx in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_idx]
      if DEBUG >= 1 and len(self.jit_cache) >= 10: print(f"jit execs {len(self.jit_cache)} kernels")
      for ei in self.jit_cache: ei.run(var_vals, jit=True)
    elif self.cnt == 1:
      # jit capture
      self.expected_names: List[Union[int, str]] = expected_names
      self.expected_lbs: List[Tuple[ShapeTracker, Tuple[Variable, ...], DType, str]] = expected_lbs
      with Context(GRAPH=getenv("JITGRAPH", GRAPH.value), BEAM=getenv("JITBEAM", BEAM.value)):
        capturing.append(self)
        self.ret = self.fxn(*args, **kwargs)
        Tensor.corealize(get_parameters(self.ret))
        capturing.clear()
      del self.buffer_replace
      assert len(self.jit_cache), "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # memory planning (optional)
      assigned = _internal_memory_planner([cast(List[Buffer], x.rawbufs) for x in self.jit_cache], debug_prefix="JIT ")
      self.jit_cache = [ExecItem(ei.prg, [assigned.get(x,x).ensure_allocated() for x in ei.rawbufs if x is not None]) for ei in self.jit_cache]

      # Condense the items into a graph executor.
      if getenv("JIT") != 2: self.jit_cache = apply_graph_to_jit(self.jit_cache, input_rawbuffers, var_vals)

      self.input_replace = get_input_replace(self.jit_cache, input_rawbuffers)
      if DEBUG >= 1 and len(set(self.input_replace.values())) != len(input_rawbuffers): print("WARNING: some input tensors not found")
    elif self.cnt == 0:
      # jit ignore
      self.ret = self.fxn(*args, **kwargs)
      Tensor.corealize(get_parameters(self.ret))

    # clear jit inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None

    self.cnt += 1
    return self.ret
