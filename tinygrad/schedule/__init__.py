import time, inspect, itertools
from typing import Iterator
from collections import deque
from dataclasses import dataclass, field, replace
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import GroupOp, UOp, Ops, UOpMetaClass, graph_rewrite, MapCtx, gate_kernel_sink, KernelInfo, ParamArg
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, SCACHE, BASEDIR, partition, dedup, all_int, VIZ
from tinygrad.helpers import diskcache_get, diskcache_put, colored

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in GroupOp.Defines|{Ops.AFTER, Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
  return s

# a buffer state is a define or an AFTER. MSELECT/MSTACK join per-device states
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  assert s.op in GroupOp.Defines|{Ops.AFTER}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build dependency graph: kernels and buffer-state barriers
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, tuple[UOp, tuple[UOp, ...]]] = {}  # superseded state -> (AFTER, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      read_states = {k:[st for s in (k.src[0].src[1:] if k.op is Ops.END else k.src[1:]) for st in _states(s)] for k in kernels}
      # the prior state is the first of its source and the same-buffer states its kernels read that is not superseded yet
      prior_states = [_unwrap_src(u.src[0]), *[st for k in kernels for st in read_states[k] if st.storage_base is u.storage_base]]
      prev_state = next((st for st in prior_states if st not in writes), prior_states[0])
      # writes to one buffer are a chain of states: at most one AFTER supersedes a state
      if prev_state in writes: raise RuntimeError("sibling AFTERs detected in assign graph")
      new_kernels = tuple(k for k in kernels if all(a.storage_base is not u.storage_base for a in children.get(k, [])))
      writes[prev_state] = (u, new_kernels)
      # a buffer state is a node after its prior state, its joins and its kernels, so readers inherit its history
      in_degree.setdefault(u, 0)
      for st in ([prev_state] if prev_state.op is Ops.AFTER else []) + [*after_deps, *kernels]:
        children.setdefault(st, []).append(u)
        in_degree[u] += 1
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        reads += [(u, k, st) for st in read_states[k]]
        # RAW deps: a kernel runs after the states it reads or joins
        for st in read_states[k] + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            children.setdefault(st, []).append(k)
            in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; a kernel already on an earlier state of the same buffer is an ordering dep, not a write
    for u, k, s in reads:
      a, write_kernels = writes.get(s, (u, ()))
      if a is u: continue
      for t in write_kernels:
        if t is not k:
          children.setdefault(k, []).append(t)
          in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is not Ops.AFTER:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if not s.is_bound_var)
        linearized.append(k.replace(src=(k.body, *buf_uops)))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat

@dataclass
class ResolveCtx:
  args: tuple[UOp, ...]
  slots: Iterator[int]

def fresh_alloc(ctx:ResolveCtx, b:UOp) -> UOp:
  # every invocation gets its own storage. ALLOCs stay unbound until link, numbered in schedule order so a schedule always
  # resolves to the same linear
  device = b.device if b.device is not None else next(a.device for a in ctx.args if a.device is not None)
  return UOp.alloc((b.max_numel(),), b.dtype, slot=next(ctx.slots), device=device)

pm_post_sched_cache = PatternMatcher([
  # Resolve positional arguments outside kernel bodies; free Variables have slot -1.
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx.args[x.arg.slot] if x.arg.slot >= 0 else None),
  # give ALLOCs fresh slots for this invocation
  (UPat(Ops.ALLOC, name="b"), fresh_alloc),
])

# the JIT captures linears with bound storage
pm_allocs_to_buffers = PatternMatcher([(UPat(Ops.ALLOC, name="b"), lambda b: UOp.new_buffer(b.device, b.max_numel(), b.dtype))])

def resolve_linear_call(linear_call:UOp, outer_binds:dict[int, UOp]|None=None, slots:Iterator[int]|None=None):
  if slots is None: slots = itertools.count()
  linear = graph_rewrite(linear_call.body, pm_post_sched_cache, ctx=ResolveCtx(linear_call.src[1:], slots), walk=True, name="params to buffers")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{i:x.unbound() if x.is_variable else x for i,x in enumerate(linear_call.src[1:])
              if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.body.op is Ops.LINEAR: return resolve_linear_call(si, binds, slots)
    if si.op is Ops.CALL and si.body.op is Ops.PROGRAM: return si  # compiled parameters already have ABI slots
    subs = {v:binds[v.arg.slot] for s in si.src for v in s.variables() if v.arg.name is None and v.arg.slot in binds}
    return si.replace(src=tuple(s.substitute(subs, name="resolve scalar params") for s in si.src))
  return linear.replace(src=tuple(apply_binds(si) for si in linear.src))

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(call:UOp) -> UOp|None:
  function = call.body
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo) or not call.arg.precompile: return None
  st = time.perf_counter()
  cache_key = function.key
  # SCACHE >= 2 also persists the cache to disk
  sc_ret, disk_hit = schedule_cache.get(cache_key, None) if SCACHE else None, False
  if sc_ret is None and SCACHE >= 2: disk_hit = (sc_ret:=diskcache_get("schedule_cache", {"key": cache_key})) is not None
  if sc_ret is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
    if SCACHE: schedule_cache[cache_key] = linear
    if SCACHE >= 2: diskcache_put("schedule_cache", {"key": cache_key}, linear)
  else:
    # schedule cache hit (memory or disk)
    linear = schedule_cache[cache_key] = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {colored(' cache hit', 'yellow') if disk_hit else (' cache hit' if sc_ret is not None else 'CACHE MISS')} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return call.replace(src=(linear,)+call.src[1:])

pm_schedule = PatternMatcher([
  (UPat(Ops.CALL, name="call"), lower_sink_to_linear),
])

def assert_all_same_devices(ast:UOp):
  devices = dedup([x.device for x in ast.toposort() if x.op is Ops.PARAM and x.device is not None])
  if len(devices) >= 2: raise RuntimeError(f"all buffers must be on the same device: {devices}")

def copy_kernel_to_store(call:UOp, dst:UOp, src:UOp, r:UOp|None=None):
  if dst.device == src.device and not dst.on_disk(): return None
  return call.replace(src=(dst.store(src),) + call.src[1:])

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.is_virtual or src.is_virtual or (dst.device == src.device and not dst.on_disk()): return None
  from tinygrad.codegen.simplify import pm_flatten_range, pm_simplify_ranges
  from tinygrad.schedule.prepare import pm_mops
  from tinygrad.uop.symbolic import sym
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, init_ctx=MapCtx, name="simplify ranges in copy")
  # the simplified body is kept only if it is a copy, any other kernel goes to codegen unchanged
  return pm_copy_kernel.rewrite(call.replace(src=(sink,) + call.src[1:]))

pm_copy_kernel = PatternMatcher([
  # lower copy kernels to bulk STOREs
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.CONST, arg=0))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.CONST, arg=0))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r"))).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),
])+pm_copy_kernel+PatternMatcher([
  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

# **** callify: transform a tensor graph into a CALL UOp such that all state is properly scoped

@dataclass
class CallifyCtx:
  replacements: list[UOp] = field(default_factory=list)
  allocs: dict[UOp, UOp] = field(default_factory=dict)
  views: set[UOp] = field(default_factory=set)
  stores: list[UOp] = field(default_factory=list)

def contiguous_mops_to_view(ctx:CallifyCtx|None, c:UOp, src:UOp):
  """MOPS(BUFFER/EXTERN) → SHRINK when movement ops collapse to a contiguous range."""
  if not all_int(c.shape): return None
  buf = src.base
  while buf.op is Ops.BITCAST: buf = buf.src[0].base
  if buf.op is Ops.UNSHARD:
    if isinstance(c.device, str): return None
    if (unshard := graph_rewrite(src, multi_pm, name="multi_buffer_view")).op is not Ops.UNSHARD: return None
    if (view := contiguous_mops_to_view(ctx, unshard.src[0], unshard.src[0])) is None: return None
    view = view.unshard(unshard.arg, unshard.src[1:])
  elif buf.op in {Ops.BUFFER, Ops.EXTERN} and (cv := src.contiguous_view()) is not None and cv[0].op in {Ops.BUFFER, Ops.EXTERN}:
    buf, offset = cv
    view = buf[offset:offset + src.max_numel() * src.element_size() // buf.element_size()].bitcast(src.dtype)
    if ctx is not None: ctx.views.add(view)
    view = view.reshape(c.shape)
  else: return None
  return c.replace(src=(view,)+c.src[1:]) if c.op in {Ops.COPY, Ops.STORE} else view

def is_store_after(u:UOp) -> bool:
  return u.op is Ops.AFTER and (u.src[0].unsharded_base.op is not Ops.ALLOC or u.src[1].op is Ops.STORE)

def collect_stores(ctx:CallifyCtx, u:UOp):
  if is_store_after(u): ctx.stores.append(u)

# NOTE: scheduling rewrites belong in prepare; only storage/interface normalization belongs here.
pm_callify_ctx_collect = PatternMatcher([
  # fold MOPS+BITCAST over BUFFER into SHRINK when movement ops collapse to contiguous range
  (UPat((Ops.COPY, Ops.STAGE), src=(UPat(GroupOp.Movement|{Ops.BITCAST}, name="src"),), allow_any_len=True, name="c"), contiguous_mops_to_view),
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, name="src"), UPat()), name="c", allow_any_len=True), contiguous_mops_to_view),

  # Collect effects after their sources have been rewritten, without entering call bodies.
  (UPat(Ops.AFTER, name="u"), collect_stores),
])

# ALLOCs get canonical scope-local id slots here so structurally identical calls hash identically for the
# schedule cache (fresh slots are all positive from the global counter; negative slots are already canonical)
def canonicalize_alloc(ctx:CallifyCtx, b:UOp):
  if b.arg.slot >= 0 and b not in ctx.allocs: ctx.allocs[b] = b.replace(arg=replace(b.arg, slot=-1-len(ctx.allocs)))
  return ctx.allocs.get(b)

def canonicalize_call_body(c:UOp):
  return c.replace(src=(graph_rewrite(c.body, pm_canonicalize_alloc, init_ctx=CallifyCtx, bottom_up=True),)+c.src[1:])

pm_canonicalize_alloc = PatternMatcher([
  # NOTE: lambda for late binding, canonicalize_call_body references pm_canonicalize_alloc
  (UPat(Ops.CALL, name="c"), lambda c: canonicalize_call_body(c)),
  (UPat(Ops.ALLOC, name="b"), canonicalize_alloc),
])

# **** export: the minimum processing to make a tensor graph a function with no runtime state
# The exported function references everything from outside through EXTERN(slot); externs[slot] is the map back.
# Storage loses its Buffer, bound Variables lose their value and name (keeping their range), ALLOCs get canonical slots.

@dataclass
class ExportCtx:
  externs: list[UOp] = field(default_factory=list)
  allocs: dict[UOp, UOp] = field(default_factory=dict)

def to_extern(ctx:ExportCtx, x:UOp) -> UOp|None:
  if x.op is Ops.BUFFER and x.addrspace is not AddrSpace.GLOBAL: return None
  if x.op is Ops.PARAM and not x.is_bound_var: return None
  ctx.externs.append(x)
  return x.replace(op=Ops.EXTERN, arg=replace(x.arg, slot=len(ctx.externs)-1, buffer=None, name=None, val=None))

pm_export = PatternMatcher([(UPat((Ops.BUFFER, Ops.PARAM), name="x"), to_extern)])+pm_canonicalize_alloc

def assert_exportable(fn:UOp):
  for u in fn.toposort(enter_calls=True):
    if u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL: raise RuntimeError(f"exported function has a GLOBAL BUFFER {u.arg}")
    if isinstance(u.arg, ParamArg):
      if u.arg.buffer is not None or u.arg.val is not None: raise RuntimeError(f"exported function has runtime state in {u.op} {u.arg}")
      if u.op is Ops.ALLOC and u.arg.slot >= 0: raise RuntimeError(f"exported function has a non canonical ALLOC {u.arg}")

def export(big_sink:UOp) -> tuple[UOp, tuple[UOp, ...]]:
  """export the stores of a tensor graph as a function: EXTERN(slot) in the function stands for externs[slot]"""
  stores = UOp.sink(*(u for u in big_sink.toposort() if is_store_after(u)))
  fn = graph_rewrite(stores, pm_export, ctx=(ctx:=ExportCtx()), bottom_up=True, name="export")
  if SPEC:
    type_verify(fn, spec_tensor)
    assert_exportable(fn)
  return fn, tuple(ctx.externs)

pm_import = PatternMatcher([(UPat(Ops.EXTERN, name="e"), lambda ctx,e: ctx[e.arg.slot])])

def replace_input_buffer(ctx:CallifyCtx, b:UOp):
  ctx.replacements.append(b)
  return b.param_like(len(ctx.replacements)-1)

pm_replace_buf = PatternMatcher([
  # replace EXTERNs (storage and bound Variables) with PARAMs for cache key normalization
  (UPat(Ops.EXTERN, name="b"), replace_input_buffer),
  # replace buffer views (SHRINK/BITCAST) with PARAM (only the views created by contiguous_mops_to_view)
  (UPat((Ops.SHRINK, Ops.BITCAST), name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b in ctx.views else None),
])

def transform_to_call(big_sink:UOp) -> UOp:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Graph")
  if SPEC: type_verify(big_sink, spec_tensor)

  fn, externs = export(big_sink)

  # The tensor replacement map is collected before these rewrites change node identities.
  graph_rewrite(fn, pm_callify_ctx_collect, ctx=(ctx:=CallifyCtx()), name="early transform tensor graph")
  ret = graph_rewrite(UOp.sink(*ctx.stores), pm_replace_buf, ctx=ctx, bottom_up=True, name="replace bufs")
  # map back: the call args are what the EXTERNs stand for
  args = graph_rewrite(UOp.sink(*ctx.replacements), pm_import, ctx=externs, walk=True, name="import externs").src
  ret = ret.call(*args, precompile=True)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret

def create_linear_with_vars(big_sink:UOp, bind:bool=True) -> tuple[UOp, dict[str, int]]:
  big_sink = transform_to_call(big_sink)
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # create copies
  linear = graph_rewrite(linear, pm_copy_from_store, name="lower copy kernels to STORE calls")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals from the bound Variables in the call args
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.is_bound_var:
      nm, val = b.expr, b.arg.val
      if nm not in used_vars: continue
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule with bound storage, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(graph_rewrite(linear, pm_allocs_to_buffers, walk=True, name="allocs to buffers"))
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = {b for b in linear_call.src[1:] if b.op is Ops.BUFFER}
  linear = memory_plan_rewrite(linear, held_bufs)
  # binding late lets the linear be run more than once. bound storage is shared by every piece of the linear, even if run separately
  return graph_rewrite(linear, pm_allocs_to_buffers, walk=True, name="allocs to buffers") if bind else linear, var_vals
