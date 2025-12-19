from typing import TypeVar, Generic, Callable, cast, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, JIT_BATCH_SIZE, dedup, partition, unwrap
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, Variable, sym_infer, Ops
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.engine.realize import capturing, ViewOp, BufferCopy, BufferXfer, EncDec, CompiledRunner, Runner, Estimates
from tinygrad.engine.memory import _internal_memory_planner
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass
from weakref import WeakKeyDictionary

class GraphException(Exception): pass

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

def apply_graph_to_jit(jit_cache: list[ScheduleItem], input_rawbuffers: list[Buffer], var_vals: dict[str, int],
                       max_batch_size=0) -> list[ScheduleItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  graphed_jit_cache: list[ScheduleItem] = []
  current_batch: list[ScheduleItem] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size
    try:
      if len(current_batch_devs) == 0: raise GraphException("no device for graph")
      if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): raise GraphException("only one kernel doesn't graph")
      graph_runner = current_batch_devs[0].graph(current_batch, input_rawbuffers, var_vals)
      # clear jit inputs to allow their memory to be freed/reused
      for (j,i) in graph_runner.input_replace.keys(): graph_runner.jit_cache[j].bufs[i] = None
      graphed_jit_cache.append(ScheduleItem(UOp(Ops.NOOP), cast(list[Buffer|None], input_rawbuffers), prg=graph_runner))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels on device {current_batch_devs[0]}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"JIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_batch_devs[0]}: {e}")
    current_batch = []
    current_batch_devs = []

  for si in jit_cache:
    match si.prg:
      case CompiledRunner(): si_graph_dev = si.prg.dev
      case BufferXfer(): si_graph_dev = Device[unwrap(si.bufs[0]).device]
      case BufferCopy(): si_graph_dev = next((Device[unwrap(b).device] for b in si.bufs if unwrap(b).device != "CPU"), None)
      case ViewOp(): continue # ViewOps are just ignored
      case _: si_graph_dev = None # Everything else is not graphed and flushes existing graph if it's being constructed

    # Check if this jit item can be graphed at all, so check if a new graph supports the current item.
    can_be_graphed = si_graph_dev is not None and si_graph_dev.graph is not None and \
                     graph_class(si_graph_dev).supports_exec_item([si_graph_dev], si)

    # Check if the current batch can be extended with this item.
    can_share_graph = can_be_graphed and len(current_batch_devs) > 0 and \
                      graph_class(current_batch_devs[0]).supports_exec_item(dedup(current_batch_devs + [si_graph_dev]), si)
    can_extend_graph_batch = can_share_graph and (max_batch_size == 0 or len(current_batch) < max_batch_size)

    # Flush the current batch if any, since it can't be extended or is full.
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()
    (current_batch if can_be_graphed else graphed_jit_cache).append(si)
    current_batch_devs = dedup(current_batch_devs + [si_graph_dev]) if can_be_graphed else []

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: list[ScheduleItem], input_rawbuffers:list[Buffer]) -> dict[tuple[int, int], int]:
  input_replace: dict[tuple[int, int], int] = {}
  for j,si in enumerate(jit_cache):
    for i,a in enumerate(si.bufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace

class GraphRunner(Runner):
  def __init__(self, jit_cache: list[ScheduleItem], input_rawbuffers: list[Buffer], var_vals: dict[str, int]):
    self.jit_cache = jit_cache  # NOTE: this is not used, but you have to keep these objects alive for the Graph
    self.input_replace:dict[tuple[int, int], int] = get_input_replace(jit_cache, input_rawbuffers)
    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    self.vars = sorted(var_vals.keys())
    self.symbolic_dims = dedup([tuple(d) for si in jit_cache if isinstance(si.prg, CompiledRunner) and (d:=si.prg.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for si in jit_cache if isinstance(si.prg, CompiledRunner) and (d:=si.prg.p.global_size) and is_sym_dim(d)])
    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    estimates = Estimates()
    for j,si in enumerate(jit_cache):
      assert si.prg is not None
      estimates += si.prg.estimates
      if isinstance(si.prg, CompiledRunner):
        if si.prg.p.vars: self.var_vals_replace[j] = [(i, self.vars.index(v.expr)) for i, v in enumerate(si.prg.p.vars) if v.expr not in si.fixedvars]

        global_dim_idx, local_dim_idx = find_symbolic_dim(si.prg.p.global_size), find_symbolic_dim(si.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert si.prg.p.global_size is not None and si.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(si.prg.p.global_size), tuple(si.prg.p.local_size))

    # used in MultiGraphRunner. the ints are id() of _bufs
    self.w_dependency_map: dict[int, Any] = {}
    self.r_dependency_map: dict[int, list[Any]] = collections.defaultdict(list)

    assert jit_cache[0].prg is not None
    super().__init__(colored(f"<batched {len(jit_cache)}>", "cyan"), jit_cache[0].prg.device.split(":")[0], estimates.simplify())

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, rawbufs:list[Buffer], write:list[int], new_dependency:Any):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    wait_nodes = []

    for i,rawbuf in enumerate(rawbufs):
      if id(rawbuf.base._buf) in self.w_dependency_map: wait_nodes.append(self.w_dependency_map[id(rawbuf.base._buf)])
      if i in write:
        if id(rawbuf.base._buf) in self.r_dependency_map: wait_nodes.extend(self.r_dependency_map.pop(id(rawbuf.base._buf)))

    for i,rawbuf in enumerate(rawbufs):
      if i in write: self.w_dependency_map[id(rawbuf.base._buf)] = new_dependency
      else: self.r_dependency_map[id(rawbuf.base._buf)].append(new_dependency)

    return list({id(x):x for x in wait_nodes}.values())

  @staticmethod
  def supports_exec_item(devs:list[Compiled], si:ScheduleItem) -> bool: return isinstance(si.prg, CompiledRunner) and len(dedup(devs)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def supports_exec_item(devs:list[Compiled], si:ScheduleItem) -> bool:
    # Devices must be the same type
    return isinstance(si.prg, (CompiledRunner, BufferXfer)) and len(dedup([type(Device[b.device]) for b in si.bufs if b]+[type(d) for d in devs]))==1

def get_out_buffers(si:ScheduleItem) -> list[Buffer]:
  if isinstance(si.prg, CompiledRunner): return [cast(Buffer, si.bufs[out]) for out in si.prg.p.outs if out not in si.prg.p.ins]
  if isinstance(si.prg, (BufferCopy, BufferXfer, EncDec)): return [cast(Buffer, si.bufs[0])]
  return []

def update_depends(depends:set[Buffer|None], jit_cache:list[ScheduleItem]):
  for si in jit_cache:
    if any(b in depends for b in si.bufs): depends.update(get_out_buffers(si))

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: list[ScheduleItem]
  input_replace: dict[tuple[int, int], int]
  extra_view_inputs: list[tuple[int, int, str, int, DType]]
  expected_names: list[int|str]
  expected_st_vars_dtype_device: list[tuple[UOp, tuple[Variable, ...], DType, str]]

  def __reduce__(self):
    # TODO: free_intermediates here? replan_buffers_memory_layout here?
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs,
                            self.expected_names, self.expected_st_vars_dtype_device)

  def __post_init__(self):
    self._jit_cache: list[ScheduleItem] = self.jit_cache
    self._input_replace: dict[tuple[int, int], int] = self.input_replace
    self._first_run = True
    self._clear_inputs()

  def _clear_inputs(self):
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  def free_intermediates(self):
    depends: set[Buffer|None] = set([None])
    update_depends(depends, self.jit_cache)
    for b in depends:
      if b is not None:
        if b.is_allocated(): b.deallocate()
        if b._base is not None and b._base.allocated_views == 0 and b._base.is_allocated(): b._base.deallocate()
    self.__post_init__()   # reset the graph state

  def replan_buffers_memory_layout(self):
    blacklist = [t.uop.buffer for t in get_parameters(self.ret)]
    asgn = _internal_memory_planner([[b for item in self.jit_cache for b in item.bufs if b is not None and b not in blacklist]], ignore_checks=True)
    self.jit_cache = [ScheduleItem(item.ast, [asgn.get(b,b) if b is not None else None for b in item.bufs], item.metadata, item.fixedvars, item.prg)
                      for item in self.jit_cache]
    for old, new in asgn.items():
      if old.is_allocated(): new.ensure_allocated().copyin(old.as_buffer())
    self.__post_init__()

  # jit exec
  def __call__(self, input_buffers:list[Buffer], var_vals:dict[str, int]) -> ReturnType:
    # assign inputs
    for idx, offset, device, size, dtype in self.extra_view_inputs:
      input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())
    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # Condense the items into a graph executor.
    if self._first_run:
      # allocate intermediates if freed
      for si in self.jit_cache:
        for b in si.bufs:
          if b is not None: b.ensure_allocated()
      # create graph if needed
      if JIT < 2:
        self._jit_cache = apply_graph_to_jit(self.jit_cache, input_buffers, var_vals, max_batch_size=JIT_BATCH_SIZE.value)
        self._input_replace = get_input_replace(self._jit_cache, input_buffers)
      self._first_run = False

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    for si in self._jit_cache: si.run(var_vals, jit=True)
    self._clear_inputs()
    return self.ret

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  # TODO: this multi unpack stuff is not well tested.
  lbs: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  input_buffers: list[Buffer] = flatten([rb.bufs if isinstance(rb:=lb.base.realized, MultiBuffer) else [rb]
                                         for lb in lbs if lb.base.realized is not None])
  assert len(set(input_buffers)) == len(input_buffers), "duplicate inputs to JIT"
  st_varval_dtype_device = [(*(lb.substitute({lb.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), lb.dtype, lb.device) for lb in lbs]
  _var_vals = merge_dicts([x[1] for x in st_varval_dtype_device] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  st_vars_dtype_device = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in st_varval_dtype_device]
  return input_buffers, var_vals, names, st_vars_dtype_device

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False, optimize=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune
    self.optimize = optimize

  def add_buffer(self, b:Buffer) -> Buffer:
    if found:=self._buffer_replace.get(b, None): return found
    if b.is_allocated() or b.uop_refcount > 0: return b
    if b._base is not None:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, base=self.add_buffer(b._base), offset=b.offset)
    else:
      self._buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options)
    return ret

  def add(self, si:ScheduleItem):
    self._jit_cache.append(ScheduleItem(si.ast, [self.add_buffer(buf) for buf in si.bufs if buf is not None], si.metadata, si.fixedvars, si.prg))

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> list[ScheduleItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> dict[tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buffers, var_vals, names, st_vars_dtype_device = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(params[0], *params[1:])
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._jit_cache: list[ScheduleItem] = []
      self._buffer_replace: WeakKeyDictionary[Buffer, Buffer] = WeakKeyDictionary()
      # TODO: should we always disable the memory planner here? it must be off for prune
      with Context(BEAM=getenv("JITBEAM", BEAM.value), NO_MEMORY_PLANNER=int(self.prune)):
        capturing.append(self)
        try:
          ret = self.fxn(*args, **kwargs)
          if len(params:=get_parameters(ret)): Tensor.realize(params[0], *params[1:])
        except Exception as e: raise e
        finally: capturing.clear()
      jit_cache = self._jit_cache
      del self._buffer_replace, self._jit_cache
      assert len(jit_cache), "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_buffers)} inputs")

      # track inputs that are views of buffers
      # TODO: eventually expected_buffers should live in ScheduleItem
      extra_view_inputs: list[tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      # prune independent kernels (optional)
      if self.prune:
        depends = set(input_buffers)
        update_depends(depends, jit_cache)
        pruned, onetime = partition(jit_cache, lambda si: any(b in depends for b in get_out_buffers(si)))
        if DEBUG >= 1: print(f"pruned from {len(jit_cache)} -> {len(pruned)} kernels")
        # run the onetime kernels here
        for si in onetime:
          for b in si.bufs: cast(Buffer, b).ensure_allocated()
          si.run(var_vals, jit=True)
        jit_cache = pruned

      # memory planning (optional)
      # Exclude buffers involved in transfer ops to preserve parallelism.
      noopt_buffers = {b for si in jit_cache if isinstance(si.prg, (BufferXfer, BufferCopy, EncDec)) for b in si.bufs}
      assigned = _internal_memory_planner([cast(list[Buffer], item.bufs) for item in jit_cache], noopt_buffers, debug_prefix="JIT ")
      jit_cache = [ScheduleItem(item.ast, [assigned.get(b,b).ensure_allocated() for b in item.bufs if b is not None],
                                item.metadata, item.fixedvars, item.prg) for item in jit_cache]

      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # set this for next run
      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, st_vars_dtype_device)
      if self.optimize: self.captured.replan_buffers_memory_layout()
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      assert self.captured.expected_names == names, f"args mismatch in JIT: {self.captured.expected_names=} != {names}"
      assert self.captured.expected_st_vars_dtype_device == st_vars_dtype_device, \
        f"args mismatch in JIT: {self.captured.expected_st_vars_dtype_device=} != {st_vars_dtype_device=}"
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
