from typing import TypeVar, Generic, Callable, Any, overload, cast
import functools
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc, to_tuple, perf_counter_us, \
  PROFILE, GlobalCounters
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType, AddrSpace
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, Ops, GroupOp, rewrite_group, graph_rewrite, sym_infer
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear, get_call_written_bufs
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
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

def _copy_input(u:UOp) -> UOp:
  if u.on_disk(): raise JitError("cannot make an independent copy of a written DISK input")
  run_linear(UOp(Ops.LINEAR, src=((new:=UOp.new_buffer(u.device, u.max_numel(), u.dtype)).store_call(u),)))
  return new

@rewrite_group(lambda linear,held_bufs,input_uops,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp]) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.max_numel(), u.device) for i,u in enumerate(input_uops)}, walk=True)
  linear = memory_plan_rewrite(linear, held_bufs)
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value), input_uops=input_uops, cache=False)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View compiled linear")
  return linear

class JitError(Exception): pass

def _input_key(u:UOp):
  """Structural key of an input view with the base buffer normalized. Returns None if this input needs full validation.
  Matching keys imply the full validation (substitute base -> NOOP + unbind_all) would also match."""
  if u.op is Ops.BUFFER:
    if u.addrspace is AddrSpace.GLOBAL and (b:=u.arg.buffer) is not None and b.is_allocated(): return (Ops.BUFFER, u.dtype, u.device, u.arg.size)
    return None
  if u.op not in GroupOp.Movement and u.op not in (Ops.STACK, Ops.CONST, Ops.DETACH): return None
  srcs = []
  for s in u.src:
    if (k:=_input_key(s)) is None: return None
    srcs.append(k)
  return (u.op, u.arg, tuple(srcs))

def _match_key(u:UOp, k) -> bool:
  """Check input uop u against a capture-time key without building anything."""
  if k[0] is Ops.BUFFER:
    return u.op is Ops.BUFFER and u.dtype == k[1] and u.device == k[2] and u.arg.size == k[3] \
      and u.addrspace is AddrSpace.GLOBAL and (b:=u.arg.buffer) is not None and b.is_allocated()
  return u.op is k[0] and u.arg == k[1] and len(u.src) == len(k[2]) and all(_match_key(s, sk) for s,sk in zip(u.src, k[2]))

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
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input
  input_keys: tuple|None = None  # structural keys of the capture-time inputs for fast validation

  @functools.cached_property
  def linear(self) -> UOp: return link_linear(self._linear, allow_cache=False) # do not cache jit

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self.expected_names, self.expected_input_info, self.input_keys)

  @functools.cached_property
  def _written_uops(self) -> set[UOp]:
    return {b for call in self.linear.toposort() if call.op is Ops.CALL for b in get_call_written_bufs(call)}

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  @functools.cached_property
  def _simple_positional(self) -> bool:
    """Capture had only plain positional Tensor inputs (no kwargs/containers/non-tensor args)."""
    return len(self.expected_input_info) == len(self.expected_names) and self.expected_names == list(range(len(self.expected_names)))

  @functools.cached_property
  def _buffer_keys(self) -> tuple[tuple[Ops|None, DType, str, int], ...]|None:
    """(optional RESHAPE wrapper, dtype, device, size) tuples when the capture was simple positional and all inputs are
    plain allocated buffers, possibly under a (contiguous) RESHAPE view."""
    if self.input_keys is None or not self._simple_positional: return None
    out = []
    for k in self.input_keys:
      if k[0] is Ops.BUFFER: out.append((None, k[1], k[2], k[3]))
      elif k[0] is Ops.RESHAPE and k[1] is None and len(k[2]) == 2 and k[2][0][0] is Ops.BUFFER: out.append((Ops.RESHAPE, *k[2][0][1:]))
      else: return None
    return tuple(out)

  @functools.cached_property
  def _fast_plan(self) -> list[tuple[UOp, bool, Any, Any, tuple[int, int, int]|None]]|None:
    """Pre-resolved launch plan. Only for linears made of single-lane, non-symbolic kernel calls over plain buffers and
    static host-to-host copies (and only when every kernel runs on a synchronous host device); None means use run_linear.
    Entries: (call, is_copy, spec|(dest, src), (rt, fixed_launch)|None, precomputed GlobalCounters stats)."""
    from tinygrad.runtime.support.hcq2 import HCQInfo
    from tinygrad.engine.realize import get_call_arg_uops, get_runtime, estimate_uop, get_call_kernels
    if not all(type(x[3]) is str for x in self.expected_input_info): return None  # no multi-device inputs
    def static_buf(u:UOp) -> Buffer|None:  # the static Buffer (or view buffer) for a non-PARAM global, if any
      try:
        b = u.buffer
        return b if isinstance(b, Buffer) else None
      except Exception: return None
    plan: list[tuple[UOp, bool, Any, Any, tuple[int, int, int]|None]] = []
    all_host, has_copy = True, False
    for call in self.linear.src:
      if (c:=call.without_after).op is not Ops.CALL or isinstance(c.arg.aux, HCQInfo): return None
      stats = None  # precomputed GlobalCounters increments (kernel_count, global_ops, global_mem), matching track_stats
      try:
        estimates, n = estimate_uop(c), len(get_call_kernels(c))
        stats = (n, n*sym_infer(estimates.ops, {}), n*sym_infer(estimates.mem, {}))
      except Exception: pass
      if c.body.op is Ops.STORE:
        # device copy: only copies between two static host buffers are supported on the fast path
        has_copy = True
        arg_uops = get_call_arg_uops(c)
        if len(arg_uops) != 2: return None
        cp = []
        for u in arg_uops:
          if (b:=static_buf(u)) is None: return None
          cp.append(b)
        dest, src = cp
        if dest.get_storage().host is None or src.get_storage().host is None: return None
        plan.append((c, True, (dest, src), None, stats))
        continue
      if c.body.op is not Ops.PROGRAM: return None
      if len(devs:=to_tuple(c.src[1].device)) != 1 or (fixed:=c.body.arg.fixed_launch) is None: return None
      if devs[0].split(":")[0] not in ("CPU", "NPY"): all_host = False
      arg_uops = get_call_arg_uops(c)
      spec: list[int|Buffer] = []  # per global: input slot for PARAMs, the static Buffer otherwise
      for g in c.body.arg.globals:
        if (u:=arg_uops[g]).op is Ops.PARAM: spec.append(u.arg.slot)
        elif (b:=static_buf(u)) is not None: spec.append(b)
        else: return None
      plan.append((c, False, spec, (get_runtime(devs[0], c.body), fixed), stats))
    if has_copy and not all_host: return None  # host copies can't be ordered against async device kernels
    return plan

  @functools.cached_property
  def _launch_fast(self) -> Callable[[list[UOp]], None]|None:
    """Fully pre-resolved replay closure taking the input buffer uops. None means fall back to __call__."""
    if (plan:=self._fast_plan) is None or self._symbolic_ret: return None
    if any(stats is None for *_, stats in plan): return None
    written = self._written_uops
    def launch(input_buf_uops:list[UOp]):
      concrete = tuple(_copy_input(u) if u in written else u for u in input_buf_uops) if written else input_buf_uops
      for _, is_copy, spec, extra, stats in plan:
        kc, gops, gmem = cast(tuple[int, int, int], stats)
        if is_copy:
          dest, src = spec
          dest.ensure_allocated().host[:] = src.ensure_allocated().host[:]
        else:
          rt, (gs, ls, pvals) = extra
          bufs:list[Buffer] = [cast(Buffer, concrete[s].buffer) if isinstance(s, int) else s for s in spec]
          rt(*[bst.buf if (bst:=b._storage) is not None else b.ensure_allocated().get_buf() for b in bufs],
             global_size=gs, local_size=ls, vals=pvals, wait=False)
        GlobalCounters.kernel_count += kc
        GlobalCounters.global_ops += gops
        GlobalCounters.global_mem += gmem
    return launch

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    concrete = tuple(_copy_input(u) if u in self._written_uops else u for u in input_uops)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    if (plan:=self._fast_plan) is not None and not var_vals:
      if DEBUG < 2 and not PROFILE and all(stats is not None for *_, stats in plan):
        # minimal launch loop: no ExecContext/track_stats overhead, counters updated with precomputed ints
        for _, is_copy, spec, extra, stats in plan:
          kc, gops, gmem = cast(tuple[int, int, int], stats)
          if is_copy:
            dest, src = spec
            dest.ensure_allocated().host[:] = src.ensure_allocated().host[:]
          else:
            rt, (gs, ls, pvals) = extra
            bufs:list[Buffer] = [cast(Buffer, concrete[s].buffer) if isinstance(s, int) else s for s in spec]
            rt(*[bst.buf if (bst:=b._storage) is not None else b.ensure_allocated().get_buf() for b in bufs],
               global_size=gs, local_size=ls, vals=pvals, wait=False)
          GlobalCounters.kernel_count += kc
          GlobalCounters.global_ops += gops
          GlobalCounters.global_mem += gmem
      elif not any(is_copy for _, is_copy, *_ in plan):
        from tinygrad.engine.realize import ExecContext, track_stats
        ctx = ExecContext({}, concrete, True, True, DEBUG>=2)
        for c, _, spec, extra, _ in plan:
          rt, (gs, ls, pvals) = extra
          bufs = [cast(Buffer, concrete[s].buffer) if isinstance(s, int) else s for s in spec]
          st = perf_counter_us()
          et = rt(*[b.ensure_allocated().get_buf() for b in bufs], global_size=gs, local_size=ls, vals=pvals, wait=ctx.wait)
          track_stats(ctx, c, st, [et])
      else: run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    else: run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self):
    for u in self._written_uops:
      if u.op is not Ops.BUFFER or (buf:=u.arg.buffer) is None: continue
      for b in (buf.bufs if isinstance(buf, MultiBuffer) else (buf,)):
        if b.is_allocated(): b.deallocate()
        if (base:=b._base) is not None and base.allocated_views == 0 and base.is_allocated(): base.deallocate()

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return flatten([[t.uop.src[0]] if t.uop.op is Ops.UNSHARD else [t.uop] for t in tensors])
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
  keys = tuple(map(_input_key, input_uops))
  return input_buf_uops, var_vals, names, expected_input_info, keys if None not in keys else None

class _TinyJit(Generic[ReturnType]):
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

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  @disable_gc()
  def __call__(self, *args, **kwargs) -> ReturnType:
    # fast path for jit exec: validate inputs with structural keys instead of the full uop rewrite
    if self.cnt >= 2 and self.captured is not None and (cap:=self.captured).input_keys is not None:
      if not kwargs and (bkeys:=cap._buffer_keys) is not None and (launch:=cap._launch_fast) is not None \
          and DEBUG < 2 and not PROFILE and len(args) == len(bkeys):
        # ultra-fast path: positional plain-buffer inputs, pre-resolved launch
        input_buf_uops = []
        for a, (wrap, dt, dev, sz) in zip(args, bkeys):
          if a.__class__ is not Tensor: break
          if (u:=a.uop).op is Ops.UNSHARD: u = u.src[0]
          if wrap is not None:
            if u.op is not wrap or u.arg is not None: break
            u = u.src[0]
          if u.op is not Ops.BUFFER or u.addrspace is not AddrSpace.GLOBAL or u.dtype != dt or u.device != dev or (arg:=u.arg).size != sz: break
          if (b:=arg.buffer) is None or not b.is_allocated(): break
          input_buf_uops.append(u)
        else:
          if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
          self.cnt += 1
          launch(input_buf_uops)
          return cap.ret
      elif not kwargs and cap._simple_positional and len(args) == len(cap.input_keys):
        # ultra-fast path: all-positional Tensor inputs, validated by structural keys only
        input_uops = []
        for a, k in zip(args, cap.input_keys):
          if a.__class__ is not Tensor: break
          u = a.uop
          if u.op is Ops.UNSHARD: u = u.src[0]
          if not _match_key(u, k): break
          input_uops.append(u)
        else:
          input_buf_uops = [u.base for u in input_uops]
          if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
          self.cnt += 1
          if (launch:=cap._launch_fast) is not None and DEBUG < 2 and not PROFILE:
            launch(input_buf_uops)
            return cap.ret
          return cap(input_buf_uops, {})
      elif not any(isinstance(v, UOp) for v in args+tuple(kwargs.values())):
        input_tensors = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
        names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
        for x in args + tuple(kwargs.values()):
          it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
          tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
        input_uops = flatten([[t.uop.src[0]] if t.uop.op is Ops.UNSHARD else [t.uop] for t in tensors])
        if cap.expected_names == names and len(input_uops) == len(cap.input_keys) \
            and all(k is not None and _match_key(u, k) for u,k in zip(input_uops, cap.input_keys)):
          input_buf_uops = [u.base for u in input_uops]
          if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
          self.cnt += 1
          return cap(input_buf_uops, {})
      # key mismatch doesn't imply invalid inputs (keys are stricter); fall through to full validation
    input_buf_uops, var_vals, names, expected_input_info, input_keys = _prepare_jit_inputs(args, kwargs)
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
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buf_uops)} inputs")

      # combine all captured linears into one, memory plan, and compile
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(onetime_linear, var_vals)
        del onetime_linear

      # hold all buffers with real storage reachable from live Tensors (e.g. lazy .grad created during capture) and all buffers with
      # allocated storage in the captured linear (e.g. constants baked in by copies): the memory planner can't suballocate those
      def _buf_or_none(u:UOp) -> Buffer|MultiBuffer|None: return u.arg.buffer if u.op is Ops.BUFFER else None
      held_bufs = {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if _buf_or_none(u) is not None}
      held_bufs |= {u for u in big_linear.toposort() if (b:=_buf_or_none(u)) is not None and b.is_allocated()}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops)
      # drop the pre-planning graph: it keeps the whole capture-time working set allocated (big_linear) or referenced (held_bufs).
      # the planned linear only uses the arena/held buffers, so the intermediates must be freed before linking and first exec
      del big_linear, held_bufs
      self.captured = CapturedJit(ret, linear, names, expected_input_info, input_keys)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    self.cnt += 1
    return ret

# overload signatures support both @TinyJit and @TinyJit(prune=True) syntax
@overload
def TinyJit(fxn:Callable[..., ReturnType], *, prune:bool=False) -> _TinyJit[ReturnType]: ...
@overload
def TinyJit(fxn:None=None, *, prune:bool=False) -> Callable[[Callable[..., ReturnType]], _TinyJit[ReturnType]]: ...
def TinyJit(fxn=None, **kwargs): return (lambda f: _TinyJit(f, **kwargs)) if fxn is None else _TinyJit(fxn, **kwargs)
