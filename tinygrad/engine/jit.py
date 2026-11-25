from typing import TypeVar, Generic, Callable, Any, overload
import functools
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType, AddrSpace
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Variable, Ops, rewrite_group, graph_rewrite
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear
from tinygrad.codegen import create_linear
from tinygrad.codegen.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
from dataclasses import dataclass, field

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def merge_extern(slots:dict[UOp, int], x:UOp) -> UOp: return x.as_extern(slots.setdefault(x, len(slots)))
pm_merge_export = PatternMatcher([
  (UPat(Ops.EXTERN, name="e"), lambda ctx,e: merge_extern(ctx[1], ctx[0][e.arg.slot])),
  (UPat(Ops.ALLOC, name="b"), lambda ctx,b: merge_extern(ctx[1], UOp.new_buffer(b.device, b.max_numel(), b.dtype))
   if b.addrspace is AddrSpace.GLOBAL else None),
])

@rewrite_group(lambda linear,externs,held_bufs,num_inputs,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, externs:list[UOp], held_bufs:set[UOp], num_inputs:int) -> tuple[UOp, list[UOp]]:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers, storage no one outside the jit holds is scratch: it's planned and bound at link
  params = [UOp.param(i, x.dtype, x.max_numel(), x.device) for i,x in enumerate(externs[:num_inputs])]
  used = [e for e in linear.toposort() if e.op is Ops.EXTERN]
  kept = [x for e in used if e.arg.slot >= num_inputs and ((x:=externs[e.arg.slot]).is_variable or x in held_bufs)]
  slots = {x: num_inputs+i for i,x in enumerate(kept)}
  def lower(e:UOp) -> UOp:
    if e.arg.slot < num_inputs: return params[e.arg.slot]
    return x.as_extern(slots[x]) if (x:=externs[e.arg.slot]) in slots else UOp.alloc((e.max_numel(),), e.dtype, device=e.device)
  linear = memory_plan_rewrite(linear.substitute({e: lower(e) for e in used}, walk=True))
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value))
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View compiled linear")
  return linear, params + kept

class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  _linear: UOp
  _externs: list[UOp]
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input
  _linears: dict[tuple[UOp, ...], tuple[UOp, dict[UOp, UOp]]] = field(default_factory=dict)  # aliased inputs -> (linked linear with copies, staging)

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self._externs, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _slots(self) -> dict[UOp, int]: return {x:i for i,x in enumerate(self._externs)}

  @functools.cached_property
  def _vars(self) -> list[tuple[str, UOp]]: return [(f"e{i}", x) for i,x in enumerate(self._externs) if x.is_bound_var]

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    # aliased inputs
    if (aliased:=tuple(u for u in input_uops if u in self._slots)) not in self._linears:
      if any(u.on_disk() for u in aliased): raise JitError("cannot make an independent copy of a written DISK input")
      copies = [self._externs[i].store_call(u.as_extern(self._slots[u])) for i,u in enumerate(input_uops) if u in self._slots]
      linear = self._linear.replace(src=compile_linear(UOp(Ops.LINEAR, src=tuple(copies))).src+self._linear.src)
      staging = {u: UOp.new_buffer(u.device, u.max_numel(), u.dtype) for u in aliased}
      self._linears[aliased] = (link_linear(linear, self._externs), staging)

    linear, staging = self._linears[aliased]
    if DEBUG >= 1 and len(linear.src) >= 10: print(f"jit execs {len(linear.src)} calls")
    slot_vals = {e: var_vals.get(x.expr, x.arg.val) for e,x in self._vars}
    run_linear(linear, {**var_vals, **slot_vals}, [*[staging.get(u, u) for u in input_uops], *self._externs[len(input_uops):]], jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self): self._linears.clear()

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return [t.uop.src[0] if t.uop.op is Ops.UNSHARD else t.uop for t in tensors]
  if any(u.is_virtual for u in get_input_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops = get_input_uops()
  # collect buffer UOps (including MultiBuffer)
  input_buf_uops: list[UOp] = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

class _TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_export(self, fn:UOp, externs:tuple[UOp, ...]): self._exports.append((fn, externs))

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  @disable_gc()
  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buf_uops, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
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
      self._exports: list[tuple[UOp, tuple[UOp, ...]]] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._exports): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._exports)} exports with {len(input_buf_uops)} inputs")

      # schedule the captured exports on one list of externs. the prune needs unplanned storage, jit_lower plans the memory of all of them together
      slots = {u:i for i,u in enumerate(input_buf_uops)}
      with Context(NO_MEMORY_PLANNER=1):
        linears = [graph_rewrite(create_linear(fn), pm_merge_export, ctx=(externs, slots), walk=True, name="merge export")
                   for fn,externs in self._exports]
      del self._exports
      big_linear, externs = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in linears]))), list(slots)
      del linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, {u.as_extern(i) for i,u in enumerate(input_buf_uops)})
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        onetime_vals = {f"e{i}": x.arg.val for i,x in enumerate(externs) if x.is_bound_var}
        run_linear(link_linear(compile_linear(onetime_linear), externs), onetime_vals, externs)
        del onetime_linear

      # hold all buffers with real storage reachable from live Tensors (e.g. lazy .grad created during capture) and all buffers with
      # allocated storage in the captured linear (e.g. constants baked in by copies): the memory planner can't suballocate those
      def _buf_or_none(u:UOp) -> Buffer|MultiBuffer|None: return u.arg.buffer if u.op is Ops.BUFFER else None
      held_bufs = {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if _buf_or_none(u) is not None}
      held_bufs |= {u for u in externs if (b:=_buf_or_none(u)) is not None and b.is_allocated()}
      linear, externs = jit_lower(big_linear, externs, held_bufs, len(input_buf_uops))
      # drop the pre-planning graph: it keeps the whole capture-time working set allocated (big_linear) or referenced (held_bufs).
      # the planned linear only uses the arena/held buffers, so the intermediates must be freed before linking and first exec
      del big_linear, held_bufs
      self.captured = CapturedJit(ret, linear, externs, names, expected_input_info)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    if JIT: self.cnt += 1
    return ret

# overload signatures support both @TinyJit and @TinyJit(prune=True) syntax
@overload
def TinyJit(fxn:Callable[..., ReturnType], *, prune:bool=False) -> _TinyJit[ReturnType]: ...
@overload
def TinyJit(fxn:None=None, *, prune:bool=False) -> Callable[[Callable[..., ReturnType]], _TinyJit[ReturnType]]: ...
def TinyJit(fxn=None, **kwargs): return (lambda f: _TinyJit(f, **kwargs)) if fxn is None else _TinyJit(fxn, **kwargs)
