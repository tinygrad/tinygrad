import time, inspect
from collections import deque
from dataclasses import replace
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import GroupOp, UOp, Ops, UOpMetaClass, graph_rewrite, MapCtx, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, SCACHE, BASEDIR, partition, dedup, all_int
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
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:])
        linearized.append(k.replace(src=(k.body, *buf_uops)))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.uop.ops import PatternMatcher, UPat, pm_resolve_params, pm_flatten_linear

def resolve_linear_call(linear_call:UOp, outer_binds:dict[int, UOp]|None=None):
  # a scheduled function is shared by all its calls, so the storage it allocates gets a fresh identity in each call
  linear = linear_call.body
  linear = linear.substitute({b:b.replace(arg=replace(b.arg, slot=next(UOp.unique_num))) for b in linear.toposort() if b.op is Ops.ALLOC}, walk=True)
  # the positional params outside the kernel bodies are the call args, free Variables have slot -1
  linear = graph_rewrite(linear, pm_resolve_params, ctx=linear_call.src[1:], walk=True, name="params to args")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{i:x for i,x in enumerate(linear_call.src[1:]) if x.op in {Ops.PARAM, Ops.EXTERN} and x.addrspace is AddrSpace.ALU}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.body.op is Ops.LINEAR: return resolve_linear_call(si, binds)
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

def copy_kernel_to_store(call:UOp, dst:UOp, src:UOp, n:UOp, r:UOp|None=None):
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

def copy_pat(n:UPat, i:UPat) -> UPat:
  # dst[i] = src[i] for two n-element buffers
  return UPat(Ops.PARAM, src=(n,), name="dst").index(i).store(UPat(Ops.PARAM, src=(n,), name="src").index(i))

# a bulk STORE copies whole buffers, so lower only kernels that copy every i in range(n). a size 1 copy has its range folded to 0
pm_copy_kernel = PatternMatcher([
  (UPat(Ops.CALL, src=(copy_pat(UPat.cvar("n", arg=1), UPat.cvar(arg=0)).sink(),), name="call", allow_any_len=True), copy_kernel_to_store),
  (UPat(Ops.CALL, src=(copy_pat(UPat.var("n"), UPat(Ops.RANGE, src=(UPat.var("n"),), name="r")).end(UPat(Ops.RANGE, name="r")).sink(),),
        name="call", allow_any_len=True), copy_kernel_to_store),
])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),
])+pm_copy_kernel+PatternMatcher([
  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

def contiguous_mops_to_view(ctx:dict[UOp, int]|None, c:UOp, src:UOp):
  """MOPS(BUFFER) → SHRINK when movement ops collapse to a contiguous range. a ctx collects the views, mapped to their slot"""
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
    if ctx is not None: ctx.setdefault(view, len(ctx))
    view = view.reshape(c.shape)
  else: return None
  return c.replace(src=(view,)+c.src[1:]) if c.op in {Ops.COPY, Ops.STORE} else view

# NOTE: scheduling rewrites belong in prepare; only storage/interface normalization belongs here.
pm_contiguous_views = PatternMatcher([
  # fold MOPS+BITCAST over BUFFER into SHRINK when movement ops collapse to contiguous range
  (UPat((Ops.COPY, Ops.STAGE), src=(UPat(GroupOp.Movement|{Ops.BITCAST}, name="src"),), allow_any_len=True, name="c"), contiguous_mops_to_view),
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, name="src"), UPat()), name="c", allow_any_len=True), contiguous_mops_to_view),
])

def create_linear(fn:UOp, plan:bool=True) -> UOp:
  """an exported function (no runtime state) to the LINEAR of calls that runs it, on its EXTERNs, view args and ALLOCs"""
  # a contiguous view of storage is a different buffer: it's an input of the scheduled function, the call arg is the view
  views: dict[UOp, int] = {}
  fn = graph_rewrite(fn, pm_contiguous_views, ctx=views, name="contiguous views")
  fn = fn.substitute({v:v.param_like(i) for v,i in views.items()}, walk=True, name="views to params")
  linear_call = graph_rewrite(fn.call(*views, precompile=True), pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call, storage stays unbound until map back
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # create copies
  linear = graph_rewrite(linear, pm_copy_from_store, name="lower copy kernels to STORE calls")

  # plan the storage the function allocates. the jit plans all its captured calls together instead
  return memory_plan_rewrite(linear) if plan else linear
