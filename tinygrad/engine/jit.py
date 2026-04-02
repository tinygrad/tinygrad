from typing import TypeVar, Generic, Callable, cast, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, JIT_BATCH_SIZE, dedup, unwrap, pluralize, VIZ
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, sym_infer, Ops, buffers, track_rewrites, graph_rewrite
from tinygrad.engine.realize import ExecItem, capturing, BufferCopy, BufferXfer, EncDec, CompiledRunner, Runner, Estimates
from tinygrad.engine.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.engine.schedule import linear_to_schedule
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def create_graph_call(batch:list[UOp], input_buffers:set[Buffer]) -> UOp:
  def bufs_for(b): return b.buffer.bufs if isinstance(b.buffer, MultiBuffer) else [b.buffer]

  input_list = dedup(b for si in batch for b in si.src[1:] if b.op in (Ops.BUFFER, Ops.BUFFER_VIEW) and not input_buffers.isdisjoint(bufs_for(b)))
  cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp(Ops.LINEAR, src=tuple(batch)), *input_list), arg="graph")
  return cf.call(*input_list, metadata=tuple(m for si in batch for m in si.arg.metadata))

def graph_split_rewrite(linear:UOp, input_buffers:set[Buffer], max_batch_size:int=0) -> UOp:
  new_src: list[UOp] = []
  current_batch: list[UOp] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size, new_src
    if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current_batch)
    else:
      new_src.append(create_graph_call(current_batch, input_buffers))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels")
    current_batch, current_batch_devs = [], []

  for si in linear.src:
    if si.src[0].op is Ops.BUFFER_VIEW: continue

    devs = [Device[x] for x in (si.device if isinstance(si.device, tuple) else (si.device,))]
    graph_t = graph_class(devs[0]) if devs[0].graph is not None else None

    can_graph = graph_t is not None and graph_t.supports_exec_item(devs, si)
    can_extend = can_graph and graph_t is not None and (not current_batch_devs or graph_t.supports_exec_item(current_batch_devs, si)) \
      and (max_batch_size == 0 or len(current_batch) < max_batch_size)
    if not can_extend and current_batch: flush_batch()

    # append this si and update devs
    (current_batch if can_graph else new_src).append(si)
    current_batch_devs = dedup(current_batch_devs + devs) if can_graph else []
  if current_batch: flush_batch()
  return linear.replace(src=tuple(new_src))

def jit_cache_bufs(jit_cache:list[ExecItem]):
  for ei in jit_cache:
    for b in ei.bufs:
      if b is not None: yield b
    if isinstance(ei.prg, GraphRunner): yield from jit_cache_bufs(ei.prg.jit_cache)

@track_rewrites(lambda linear,held_bufs,input_buffers=None,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_buffers:list[Buffer]|None=None) -> list[ExecItem]:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")
  linear = memory_plan_rewrite(linear, held_bufs)
  if JIT < 2: linear = graph_split_rewrite(linear, set(input_buffers or []), max_batch_size=JIT_BATCH_SIZE.value)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View graphed linear")
  return [ei.lower() for ei in linear_to_schedule(linear)]

class GraphException(Exception): pass
class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

def get_input_replace(jit_cache: list[ExecItem], input_buffers:list[Buffer],
                      orig_valid_positions: dict[int, set[int]]|None = None) -> dict[tuple[int, int], int]:
  input_replace: dict[tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_buffers:
        # filter out positions that weren't valid inputs in the original capture (prevents aliasing bugs)
        if orig_valid_positions is not None and i not in orig_valid_positions.get(id(ji), set()): continue
        input_replace[(j,i)] = input_buffers.index(a)
  return input_replace

class GraphRunner(Runner):
  def __init__(self, linear:UOp|None, input_buffers:list[Buffer]|None,
               jit_cache:list[ExecItem]|None=None, input_replace:dict[tuple[int,int],int]|None=None):
    # TODO: captured jit as linear?
    if linear is not None:
      jit_cache = [ei.lower() for ei in linear_to_schedule(linear.src[0])]
      for b in jit_cache_bufs(jit_cache): b.ensure_allocated()
      input_replace = get_input_replace(jit_cache, input_buffers) if input_buffers else {}
    self.jit_cache, self.input_replace = unwrap(jit_cache), input_replace or {}

    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    crs = [(ji, ji.prg) for ji in self.jit_cache if isinstance(ji.prg, CompiledRunner)]
    self.vars = sorted({v.expr for ji,p in crs for v in p.p.vars if v.expr not in ji.fixedvars | p.p.runtimevars})
    self.symbolic_dims = dedup([tuple(d) for _,p in crs if (d:=p.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for _,p in crs if (d:=p.p.global_size) and is_sym_dim(d)])

    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    estimates = Estimates()
    for j,ji in enumerate(self.jit_cache):
      assert ji.prg is not None
      estimates += ji.prg.estimates
      if isinstance(ji.prg, CompiledRunner):
        if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(ji.prg.p.vars) if v.expr not in ji.fixedvars | ji.prg.p.runtimevars]):
          self.var_vals_replace[j] = replace

        global_dim_idx, local_dim_idx = find_symbolic_dim(ji.prg.p.global_size), find_symbolic_dim(ji.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert ji.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(ji.prg.p.global_size), tuple(ji.prg.p.local_size))

    # used in MultiGraphRunner. tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)

    assert self.jit_cache[0].prg is not None
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), self.jit_cache[0].prg.device.split(":")[0], estimates.simplify())

  def __reduce__(self): return self.__class__, (None, None, self.jit_cache, self.input_replace)

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      if i in write:
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for st,en,dep in dmap[key]:
            if st < min(s, en): kept.append((st, min(s, en), dep))
            if max(e, st) < en: kept.append((max(e, st), en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

  @staticmethod
  def _all_devs(batch_devs:list[Compiled], new_call:UOp) -> list[Compiled]:
    return dedup(batch_devs + [Device[x] for b in new_call.src[1:] if b.op is not Ops.BIND
                 for x in (b.device if isinstance(b.device, tuple) else (b.device,))])

  @staticmethod
  def supports_exec_item(batch_devs:list[Compiled], new_call:UOp) -> bool:
    return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM) and len(GraphRunner._all_devs(batch_devs, new_call)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def supports_exec_item(batch_devs:list[Compiled], new_call:UOp) -> bool:
    # Devices must be the same type
    return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM, Ops.COPY) and len(dedup([type(d) for d in GraphRunner._all_devs(batch_devs, new_call)])) == 1

def get_out_buffers_for_ei(ei:ExecItem) -> list[Buffer]:
  if isinstance(ei.prg, CompiledRunner): return [cast(Buffer, ei.bufs[out]) for out in ei.prg.p.outs if out not in ei.prg.p.ins]
  if isinstance(ei.prg, (BufferCopy, BufferXfer, EncDec)): return [cast(Buffer, ei.bufs[0])]
  if isinstance(ei.prg, GraphRunner): return dedup([b for inner in ei.prg.jit_cache for b in get_out_buffers_for_ei(inner)])
  return []

def update_depends(depends:set[Buffer|None], jit_cache:list[ExecItem]):
  for ei in jit_cache:
    if any(b in depends for b in ei.bufs): depends.update(get_out_buffers_for_ei(ei))

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: list[ExecItem]
  input_replace: dict[tuple[int, int], int]
  extra_view_inputs: list[tuple[int, int, str, int, DType]]
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  def __reduce__(self):
    # TODO: free_intermediates here?
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs, self.expected_names, self.expected_input_info)

  def __post_init__(self):
    self._jit_cache: list[ExecItem] = self.jit_cache
    self._input_replace: dict[tuple[int, int], int] = self.input_replace
    self._first_run = True
    self._needs_rebuild = False
    # precompute read-after-write hazard detection
    self._output_to_writer = {b: j for j, ei in enumerate(self.jit_cache) for b in get_out_buffers_for_ei(ei)}
    self._input_to_max_reader: dict[int, int] = {}
    for (j, i), idx in self.input_replace.items():
      # only buffers that were different during capture but alias at jit time (e.g. feeding output back as input) need the copy.
      if self.jit_cache[j].bufs[i] not in get_out_buffers_for_ei(self.jit_cache[j]):
        self._input_to_max_reader[idx] = max(self._input_to_max_reader.get(idx, -1), j)
    self._clear_inputs()

  def _clear_inputs(self):
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  def free_intermediates(self):
    depends: set[Buffer|None] = set([None])
    update_depends(depends, self.jit_cache)
    arenas = {b._base for b in depends if b is not None and b._base is not None}
    to_free = {b for b in depends if b is not None} | {b for b in jit_cache_bufs(self.jit_cache) if b._base in arenas}
    for b in to_free:
      if hasattr(b, '_buf'): b.deallocate()
    for a in arenas:
      if a.allocated_views == 0 and a.is_allocated(): a.deallocate()
    self.__post_init__()
    self._needs_rebuild = True

  # jit exec
  def __call__(self, input_buffers:list[Buffer], var_vals:dict[str, int]) -> ReturnType:
    # assign inputs
    for idx, offset, device, size, dtype in self.extra_view_inputs:
      input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())

    # copy aliased inputs to prevent read-after-write hazard
    for i, ib in enumerate(input_buffers):
      if (writer := self._output_to_writer.get(ib)) is not None and self._input_to_max_reader.get(i, -1) >= writer:
        shadow = Buffer(ib.device, ib.size, ib.dtype).ensure_allocated()
        input_buffers[i] = shadow if ib.device.startswith("NULL") else shadow.copyin(ib.as_memoryview())

    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # allocate intermediates if freed on first run
    if self._first_run:
      for b in jit_cache_bufs(self.jit_cache): b.ensure_allocated()
    if self._needs_rebuild:
      for ei in self.jit_cache:
        if isinstance(ei.prg, GraphRunner): ei.prg = type(ei.prg)(None, None, ei.prg.jit_cache, ei.prg.input_replace)
    self._first_run = self._needs_rebuild = False

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    for ei in self._jit_cache: ei.run(var_vals, jit=True)
    self._clear_inputs()
    return self.ret

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.base.op is Ops.CONST for u in input_uops):
    raise JitError("JIT inputs cannot be const, create a buffer with .contiguous()")
  input_buffers: list[Buffer] = flatten([b.bufs if isinstance(b, MultiBuffer) else [b] for u in input_uops if (b:=u.base.realized) is not None])
  if len(set(input_buffers)) != len(input_buffers): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buffers, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_linear(self, linear:UOp, var_vals:dict[str, int]): self._linears.append(linear)

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> list[ExecItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> dict[tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buffers)} inputs")

      # combine all captured linears into one, memory plan, and convert to ExecItems
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, {k for k,v in buffers.items() if isinstance(v, Buffer) and v in set(input_buffers)})
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        for ei in (si.lower() for si in linear_to_schedule(onetime_linear)):
          for b in ei.bufs: cast(Buffer, b).ensure_allocated()
          ei.run(var_vals, jit=True)

      held_bufs = set(buffers) | {t.uop.buf_uop for t in get_parameters(ret) if t.uop.buf_uop.op is Ops.BUFFER}
      with Context(BEAM=getenv("JITBEAM", BEAM.value)):
        jit_cache = jit_lower(big_linear, held_bufs, input_buffers)

      # track inputs that are views of buffers
      # TODO: eventually expected_buffers should live in ExecItem
      extra_view_inputs: list[tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # exec
      for ei in jit_cache: ei.run(var_vals)

      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, expected_input_info)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
