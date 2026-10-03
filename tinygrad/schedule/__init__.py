import time, inspect, functools
from typing import cast
from dataclasses import dataclass, field, replace
from tinygrad.dtype import AddrSpace
from collections import deque
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, graph_rewrite, gate_kernel_sink, KernelInfo, CallInfo, GroupOp, resolve, resolve_returned_after
from tinygrad.uop.ops import AxisType
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, SCACHE, BASEDIR, partition, dedup, all_int, VIZ
from tinygrad.helpers import diskcache_get, diskcache_put, colored
from tinygrad.schedule.allreduce import is_allreduce_linear_output

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.ALLOC, Ops.PARAM, Ops.MSELECT, Ops.MSTACK} and \
        not (s.op is Ops.SHRINK and s.tag == ("allreduce",) and s.src[0].op is not Ops.INDEX): s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | ALLOC | PARAM. MSELECT/MSTACK join per-device states
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  if s.op is Ops.SHRINK and s.tag == ("allreduce",): return _states(s.src[0])
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.ALLOC, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _slice_region(s:UOp) -> tuple[UOp, int, int]|None:
  """Return the concrete byte interval accessed through nested hardware slices."""
  offset, size = 0, None
  while True:
    s = _unwrap_src(s)
    if s.op is Ops.AFTER: s = s.src[0]
    elif s.op is Ops.SHRINK and s.tag == ("allreduce",) and s.src[1].op is Ops.CONST and s.src[2].op is Ops.CONST:
      offset += s.src[1].val * s.src[0].dtype.itemsize
      if size is None: size = s.src[2].val * s.dtype.itemsize
      s = s.src[0]
    else: break
  return (s.buf_uop, offset, offset+size) if size is not None else None

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE and not (s.op is Ops.RANGE and s.axis_type is AxisType.DEVICE)]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def _call_buf_uop(s:UOp) -> UOp:
  """Resolve a call argument's storage, preserving a dependency-wrapped hardware slice as the actual view."""
  s = _unwrap_src(s)
  if s.op is Ops.AFTER and s.src[0].op is Ops.SHRINK and s.src[0].tag == ("allreduce",): s = s.src[0]
  if s.op is Ops.SHRINK and s.tag == ("allreduce",): return s.replace(src=(s.src[0].buf_uop,)+s.src[1:])
  return s.buf_uop

@functools.cache
def _call_overwrite_outputs(call:UOp) -> tuple[UOp, ...]:
  if call.body.op is Ops.LINEAR:
    return tuple(x for i,x in enumerate(call.src[1:]) if is_allreduce_linear_output(call.body, i))
  return ()

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, list[tuple[UOp, tuple[UOp, ...]]]] = {}  # superseded state -> (AFTER, new kernels)
    reads: list[tuple[UOp, UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read, access)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      prev_state = _unwrap_src(u.src[0])
      prev_kernels = set(_split_after(prev_state)[0]) if prev_state.op is Ops.AFTER else set()
      writes.setdefault(prev_state, []).append((u, tuple(k for k in kernels if k not in prev_kernels)))
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else tuple(x for x in k.src[1:] if x not in _call_overwrite_outputs(k))
        read_states = [(st, s) for s in kernel_deps for st in _states(s)]
        reads += [(u, k, st, access) for st,access in read_states]
        # RAW deps: a kernel runs after the kernels that produced the states it reads or joins
        for st in [st for st,_ in read_states] + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            for t in _split_after(st)[0]:
              children.setdefault(t, []).append(k)
              in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; join members already present in that prior state are ordering deps, not writes
    for u, k, s, access in reads:
      for a, write_kernels in writes.get(s, []):
        if a is u: continue
        for t in write_kernels:
          call = t.src[0] if t.op is Ops.END else t
          # Disjoint physical intervals do not alias and therefore need no WAR edge.
          write_accesses = _call_overwrite_outputs(call) or call.src[1:2]
          if ((rr:=_slice_region(access)) is not None and write_accesses and
              all((wr:=_slice_region(w)) is not None and (rr[0] is not wr[0] or rr[2] <= wr[1] or wr[2] <= rr[1]) for w in write_accesses)): continue
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      k = rk.src[0] if rk.op is Ops.END else rk
      assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
      buf_uops = tuple(_call_buf_uop(s) for s in k.src[1:] if not s.is_bound_var)
      linearized.append(k.replace(src=(k.body, *buf_uops)))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat, ParamArg

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None:
    device = b.device if b.device is not None else next(a.device for a in ctx[1] if a.device is not None)
    ctx[0][b] = ret = UOp.new_buffer(device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # Resolve positional arguments outside kernel bodies; free Variables have slot -1.
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot] if x.arg.slot >= 0 else None),
  # bind ALLOCs to fresh BUFFERs for this invocation
  (UPat(Ops.ALLOC, name="b"), create_new_buffer),
])

def resolve_linear_call(linear_call:UOp, outer_binds:dict[int, UOp]|None=None):
  linear = graph_rewrite(linear_call.body, pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{i:x.unbound() if x.is_variable else x for i,x in enumerate(linear_call.src[1:])
              if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.body.op is Ops.LINEAR: return resolve_linear_call(si, binds)
    if si.op is Ops.CALL and si.body.op is Ops.PROGRAM: return si  # compiled parameters already have ABI slots
    subs = {v:binds[v.arg.slot] for s in si.src for v in s.variables() if v.arg.slot in binds}
    ret = si.replace(src=tuple(s.substitute(subs, name="resolve scalar params") for s in si.src))
    # Tagged all-reduce views are physical runtime arguments and must retain their offset while dropping the state
    # wrapper. Preserve ordinary AFTER arguments: they carry producer dependencies across nested LINEAR boundaries.
    if ret.op is Ops.CALL:
      def resolve_arg(s:UOp) -> UOp:
        if s.is_bound_var: return s
        u = _unwrap_src(s)
        if ((u.op is Ops.AFTER and u.src[0].op is Ops.SHRINK and u.src[0].tag == ("allreduce",)) or
            (u.op is Ops.SHRINK and u.tag == ("allreduce",))): return _call_buf_uop(s)
        return s
      ret = ret.replace(src=(ret.src[0],)+tuple(resolve_arg(s) for s in ret.src[1:]))
    return ret
  return linear.replace(src=tuple(apply_binds(si) for si in linear.src))

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
schedule_cache_param_maps: dict[bytes, dict[int, int]] = {}
schedule_cache_buffer_maps: dict[bytes, dict[int, ParamArg]] = {}

def remap_paramarg_slots(root:UOp, param_map:dict[int, int], buffer_map:dict[int, int|ParamArg]|None=None,
                         clear_buffer:bool=False) -> UOp:
  """Simultaneously rename direct PARAM/BUFFER slots without fixed-point substitution cycling on permutations."""
  rebuilt:dict[UOp, UOp] = {}
  for x in root.toposort(enter_calls=False):
    src = tuple(rebuilt.get(s, s) for s in x.src)
    mapping = param_map if x.op is Ops.PARAM else buffer_map if x.op is Ops.ALLOC else None
    arg = x.arg
    if mapping is not None and isinstance(arg, ParamArg) and arg.slot in mapping:
      mapped = mapping[arg.slot]
      arg = mapped if isinstance(mapped, ParamArg) else replace(arg, slot=mapped, buffer=None if clear_buffer else arg.buffer)
    rebuilt[x] = x.replace(src=src, arg=arg)
  return rebuilt[root]

def canonicalize_sink_outputs(body:UOp) -> UOp:
  # A bound final output retains STORE, while Callify's nested output placement produces AFTER(dest, STORE).
  # They have the same effects; normalize this interface distinction before forming a schedule-cache key.
  return body.replace(src=tuple(s.src[0].after(s) if s.op is Ops.STORE else s for s in body.src)) if \
    body.op is Ops.SINK and body.arg is None else body

def canonicalize_call_for_schedule_cache(call:UOp) -> UOp|None:
  body = call.body
  arg = replace(call.arg, grad_fxn=None) if isinstance(call.arg, CallInfo) and call.arg.grad_fxn is not None else call.arg
  if body.op not in {Ops.SINK, Ops.LINEAR}: return call.replace(arg=arg) if arg is not call.arg else None
  nodes = body.toposort(enter_calls=False)
  params = [x for x in nodes if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot >= 0]
  param_slots = list(dict.fromkeys(x.arg.slot for x in params))
  if any(slot+1 >= len(call.src) for slot in param_slots): return None
  # Callify's negative ALLOC slots are canonical in its enclosing scope, not in this nested body.
  bufs = [x for x in nodes if x.op is Ops.ALLOC]
  buf_slots = list(dict.fromkeys(x.arg.slot for x in bufs))
  pmap:dict[int, int] = {slot:i for i,slot in enumerate(param_slots)}
  bmap:dict[int, int|ParamArg] = {slot:len(param_slots)+i for i,slot in enumerate(buf_slots)}
  body = canonicalize_sink_outputs(remap_paramarg_slots(body, pmap, bmap, clear_buffer=True))
  return call.replace(src=(body,)+tuple(call.src[1+slot] for slot in param_slots), arg=arg)

pm_schedule_cache_key = PatternMatcher([
  (UPat(Ops.CALL, name="call", allow_any_len=True), canonicalize_call_for_schedule_cache),
])

# ctx is just for DEBUG on inner
def lower_sink_to_linear(call:UOp) -> UOp|None:
  function = call.body
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo) or not call.arg.precompile: return None
  st = time.perf_counter()
  # Gradient callbacks have been consumed before scheduling, and opaque CALL parameter numbering is local to each
  # body. Canonicalize each body together with its arguments, then alpha-rename this enclosing function's inputs.
  canonical = graph_rewrite(function, pm_schedule_cache_key, name="canonicalize schedule cache calls", walk=True)
  nodes = canonical.toposort(enter_calls=False)
  params = [x for x in nodes if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot >= 0]
  bufs = [x for x in nodes if x.op is Ops.ALLOC]
  param_slots, buf_slots = (list(dict.fromkeys(x.arg.slot for x in xs)) for xs in (params, bufs))
  pmap:dict[int, int] = {slot:i for i,slot in enumerate(param_slots)}
  bmap:dict[int, int|ParamArg] = {slot:len(param_slots)+i for i,slot in enumerate(buf_slots)}
  canonical = canonicalize_sink_outputs(remap_paramarg_slots(canonical, pmap, bmap, clear_buffer=True))
  param_map = {pmap[x.arg.slot]:x.arg.slot for x in params}
  buffer_map = {cast(int, bmap[x.arg.slot]):x.arg for x in bufs}
  cache_key = canonical.key
  # SCACHE >= 2 also persists the schedule and its slot mappings to disk.
  sc_ret, disk_hit = schedule_cache.get(cache_key, None) if SCACHE else None, False
  if sc_ret is None and SCACHE >= 2:
    if (cached:=diskcache_get("schedule_cache_canonical", {"key": cache_key})) is not None:
      sc_ret, schedule_cache_param_maps[cache_key], schedule_cache_buffer_maps[cache_key] = cached
      disk_hit = True
  if sc_ret is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
    if SCACHE:
      schedule_cache[cache_key] = linear
      schedule_cache_param_maps[cache_key] = param_map
      schedule_cache_buffer_maps[cache_key] = buffer_map
    if SCACHE >= 2: diskcache_put("schedule_cache_canonical", {"key": cache_key}, (linear, param_map, buffer_map))
  else:
    # schedule cache hit (memory or disk)
    linear = schedule_cache[cache_key] = sc_ret
    old_map = schedule_cache_param_maps[cache_key]
    assert old_map.keys() == param_map.keys(), "canonical schedule cache hit has mismatched parameters"
    remap = {old_slot:param_map[canonical_slot] for canonical_slot,old_slot in old_map.items()}
    old_buffer_map = schedule_cache_buffer_maps[cache_key]
    assert old_buffer_map.keys() == buffer_map.keys(), "canonical schedule cache hit has mismatched buffers"
    buffer_remap = {old_arg.slot:buffer_map[canonical_slot] for canonical_slot,old_arg in old_buffer_map.items()}
    linear = remap_paramarg_slots(linear, remap, buffer_remap)
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
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
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  return call.replace(src=(dst.store(src),) + call.src[1:])

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  # Preserve the canonical flat form for a full identity copy. Movement views can leave an equivalent
  # multidimensional index here, but the flat form is what the COPY recognizer and runtimes consume.
  stores = [x for x in ast.toposort() if x.op is Ops.STORE]
  if len(stores) == 1:
    store, value = stores[0], stores[0].src[1]
    if value.op is Ops.COPY: value = value.src[0]
    if (store.src[0].op is Ops.INDEX and value.op is Ops.INDEX and store.src[0].src[1:] == value.src[1:]
        and store.src[0].src[0].numel() == value.src[0].numel() == dst.numel() == src.numel()):
      out, inp, r = store.src[0].src[0], value.src[0], UOp.range(dst.numel(), 0)
      ast = out.index(r).store(inp.index(r)).end(r).sink()
  from tinygrad.codegen.simplify import pm_flatten_range, pm_simplify_ranges
  from tinygrad.schedule.prepare import pm_mops
  from tinygrad.uop.symbolic import sym
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges in copy")
  return call.replace(src=(sink,) + call.src[1:])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),

  # lower an explicit COPY-valued identity kernel to the upstream bulk STORE representation
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r")).f(Ops.COPY)).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.CONST, arg=0))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.CONST, arg=0))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r"))).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
  # Callify can preserve the destination state as AFTER(dst, STORE(dst, COPY(src))). It is the same bulk transfer.
  (UPat(Ops.CALL, src=(UPat(Ops.AFTER, src=(UPat.var("dst"),
                UPat(Ops.STORE, src=(UPat.var("dst"), UPat(Ops.COPY, src=(UPat.var("src"),)))))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),

  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

# *** callify: transform a tensor graph into a CALL UOp such that all state is properly scoped ***

@dataclass
class CallifyCtx:
  buffer_map: dict[UOp, UOp] = field(default_factory=dict)
  bases: set[UOp] = field(default_factory=set)
  stores: list[UOp] = field(default_factory=list)
  replacements: list[UOp] = field(default_factory=list)
  allocs: dict[UOp, UOp] = field(default_factory=dict)
  views: set[UOp] = field(default_factory=set)
  physical_views: dict[UOp, UOp] = field(default_factory=dict)

# a tag is the tuple of original pre-rewrite UOps a node provides storage for
def tag_uop(x:UOp): return None if x.tag is not None else x.replace(tag=(x,))

def creation_copy_is_realized(u:UOp):
  # Transfers from a creation device already own destination storage and must remain persistent across Callify.
  if u.src[0].on_creation_device(): return tag_uop(u)

add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), creation_copy_is_realized),
  # A full STORE of a creation COPY uses the AFTER destination as that COPY's storage.
  (UPat(Ops.AFTER, src=(UPat(name="dest"),
    UPat(Ops.STORE, src=(UPat(name="dest"), UPat(Ops.COPY, name="c")))), name="a"),
   lambda a,c,dest: a.replace(src=(a.src[0], a.src[1].replace(src=(dest, c.rtag(())))), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat(Ops.AFTER, name="x"), tag_uop),
  # materializations synthesized at function boundaries still need storage outside the nested call
  (UPat(Ops.STAGE, src=(UPat((Ops.COPY, Ops.STAGE, Ops.AFTER, Ops.CAST)),), name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(x) if x in ctx.bases else None),
])

def lift_full_buffer_reshape_after(r:UOp, a:UOp) -> UOp|None:
  if resolve(r.numel() != a.numel(), False) or r.dtype != a.dtype or not a.src[0].has_buffer_identity(after_ok=True): return None
  return r.replace(src=(a.src[0], *r.src[1:])).after(*a.src[1:])

def lift_unshard_after(u:UOp, a:UOp) -> UOp|None:
  if not a.src[0].has_buffer_identity(after_ok=True): return None
  return u.replace(src=(a.src[0], *u.src[1:])).after(*a.src[1:])

lift_full_buffer_after_views = PatternMatcher([
  (UPat(Ops.RESHAPE, src=(UPat(Ops.AFTER, name="a"),), allow_any_len=True, name="r"), lift_full_buffer_reshape_after),
  (UPat(Ops.UNSHARD, src=(UPat(Ops.AFTER, name="a"),), allow_any_len=True, name="u"), lift_unshard_after),
])

def bind_call_storage(x:UOp) -> UOp:
  # Callify runs after output bufferization. Its synthesized caller-owned storage must be bound here,
  # while Tensor.empty and allocations inside function bodies retain upstream's late-binding behavior.
  ret = x.empty_like()
  base = ret.unsharded_base
  return ret.substitute({base:UOp.new_buffer(base.device, base.max_numel(), base.dtype)})

def mint_tagged_storage(x:UOp):
  if x.tag == ("replicate",): return x
  if x.tag is None: return None          # untouched
  # Scheduler annotations are not allocation provenance tags.
  if not all(isinstance(t, UOp) for t in x.tag): return None
  # empty tag from rtag(()): a COPY already handled via buffer_map or merged into a parent AFTER.
  # () is falsy but not None, so it isn't re-tagged like a bare (tag=None) node would be; just strip it here
  if not x.tag: return x.rtag(None)
  # a tagged CONTIGUOUS is consumed by the mint: the buffer stores its source directly
  src = x.src[0] if x.op is Ops.STAGE else x.rtag(None)
  # virtual values and DISK tensors don't get real buffers: keep the (single) annotation, drop the tag
  if x.is_virtual or x.on_disk(): return src.alu(Ops.STAGE) if src.device is not None else src
  # if size is 0, remove the contig
  if 0 in x.shape: return src
  buf = bind_call_storage(x)
  return buf.after(buf.store(src)).replace(tag=x.tag)

def mint_function_materialization(x:UOp) -> UOp|None:
  # These CONTIGUOUS nodes are synthesized after provenance tagging. Give them caller-owned storage without inventing
  # a tensor mapping: they are internal function materializations, not additional Callify outputs.
  buf = bind_call_storage(x)
  return buf.after(buf.store(x.src[0])).replace(tag=x.tag)

pm_mint_function_materializations = PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat((Ops.COPY, Ops.STAGE, Ops.AFTER, Ops.CAST)),), name="x"), mint_function_materialization),
])

# Allocation provenance is local to Callify, while physical allreduce annotations are consumed later by the scheduler.
pm_remove_allocation_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x:
  x.replace(tag=None) if x.tag is not None and x.tag not in {
    ("allreduce",), ("allreduce_accumulate",), ("replicate",), ("linear_stack",)} else None)])

def contiguous_mops_to_view(ctx:CallifyCtx|None, c:UOp, src:UOp):
  """MOPS(BUFFER) → SHRINK when movement ops collapse to a contiguous range."""
  if not all_int(c.shape): return None
  buf = src.base
  while buf.op is Ops.BITCAST: buf = buf.src[0].base
  if buf.op is Ops.UNSHARD:
    if isinstance(c.device, str): return None
    if (unshard := graph_rewrite(src, multi_pm, name="multi_buffer_view")).op is not Ops.UNSHARD: return None
    view = contiguous_mops_to_view(ctx, unshard.src[0], unshard.src[0])
    return None if view is None else view.unshard(unshard.arg, unshard.src[1:])

  if buf.op is not Ops.BUFFER or (cv := src.contiguous_view()) is None or cv[0].op is not Ops.BUFFER: return None
  buf, offset = cv
  view = buf[offset:offset + src.max_numel() * src.element_size() // buf.element_size()].bitcast(src.dtype)
  if ctx is not None: ctx.views.add(view)
  view = view.reshape(c.shape)
  return c.replace(src=(view,)+c.src[1:]) if c.op in {Ops.COPY, Ops.STORE} else view

def _precompiled_output_redirect(s:UOp, t:UOp) -> tuple[UOp, dict[UOp, UOp]]|None:
  # how output s lands in the caller's buffer t, or None if it must be copied into t
  # materialize straight into t
  if s.op is Ops.STAGE:
    placed = t.after(t.store(s.src[0]))
    return placed, {s:placed}
  # rebind output storage to t
  if s.op in {Ops.BUFFER, Ops.ALLOC, Ops.UNSHARD} and s.has_buffer_identity(): return t, {s:t}
  # a full-buffer reshape is the same storage with a different logical shape, so rebind both the view and its base
  if (s.op is Ops.RESHAPE and s.has_buffer_identity() and s.contiguous_view_offset() == 0 and resolve(s.numel() == s.base.numel(), False)
      and s.base.op in {Ops.BUFFER, Ops.ALLOC, Ops.UNSHARD}):
    return t, {s:t, s.base:t.reshape(s.base.shape)}
  # A shard-local full-buffer view can still be expressed as movement over UNSHARD here. Resolve it before deciding
  # whether the function output needs a materializing copy.
  if isinstance(s.device, tuple) and s.axis is not None:
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(s, multi_pm, name="resolve precompiled output sharding")
    local = resolved.src[0] if resolved.op is Ops.UNSHARD else resolved
    physical = local
    while physical.op in GroupOp.Movement|{Ops.UNSHARD, Ops.AFTER}: physical = physical.src[0]
    target_physical = t
    while target_physical.op in GroupOp.Movement|{Ops.UNSHARD, Ops.AFTER}: target_physical = target_physical.src[0]
    if (physical.op in {Ops.BUFFER, Ops.ALLOC} and target_physical.op is Ops.PARAM and resolve(physical.numel() == target_physical.numel(), False)
        and resolve(local.numel() == physical.numel(), False) and local.contiguous_view_offset() == 0):
      return t, {physical:target_physical.reshape(physical.shape)}
  return None

def transform_precompiled_call(c:UOp) -> UOp|None:
  if c.arg is None or not c.arg.precompile or not c.has_unbound_outputs: return None
  assert c.body.op is Ops.SINK, "precompiled call bodies are SINKs of stores into the output PARAMs"
  # the RETURNED srcs are the call outputs (slots are src positions)
  ret_pos = [p for p,a in enumerate(c.src[1:]) if (b:=a.unsharded_base).op is Ops.ALLOC and not b.arg.bind_on_realize]
  srcs = tuple(graph_rewrite(st.src[1], lift_full_buffer_after_views, name="lift full-buffer AFTER views")
               for st in c.body.src if st.op is Ops.STORE)

  # add the outputs to the call
  outs = tuple(bind_call_storage(c.src[1+p]) for p in ret_pos)
  # Output storage is max-sized; symbolic extents inside the body must use its lexical parameters, not caller bindings.
  targets = [o.pad_to(o.max_shape).param_like(p).shrink_to(s.shape) for p,o,s in zip(ret_pos, outs, srcs)]

  # how each stored value lands in its output PARAM target: a CONTIGUOUS materializes straight into the target and
  # a real buffer/UNSHARD rebinds its storage to the target (once per unique value); everything else is copied into it
  placed:dict[UOp, UOp] = {}
  items:list[UOp] = []
  for s, t in zip(srcs, targets):
    deps:list[UOp] = []
    while s.op is Ops.AFTER:
      deps.extend(s.src[1:])
      s = s.src[0]
    redirect = _precompiled_output_redirect(s, t)
    if redirect is not None and all(old not in placed for old in redirect[1]):
      placed.update(redirect[1])
      items.append(s.after(*deps) if deps else s)
    else:
      items.append(t.after(t.store(s.after(*deps))))
  # swap every placed value for its target storage, also inside other stores' AFTER deps
  fxn = UOp.sink(*(x.substitute(placed) for x in items))

  # ALLOC has buffer identity, but an unresolved inline-call output is not materialized yet. Opaque calls need
  # caller-owned storage for those inputs; mint_function_materialization supplies it for the synthesized STAGE.
  rmap = dict(zip(ret_pos, outs))
  new_call = c.replace(src=(fxn, *[rmap.get(i, a if a.has_buffer_identity(after_ok=True) and a.storage_base.op is not Ops.ALLOC else a.contiguous())
                                   for i, a in enumerate(c.src[1:])]))
  rets = tuple(o.after(new_call) for o in outs)

  # if the CALL has symbolic shapes, shrink the max-sized output to the actual symbolic shape
  # NOTE: must use the resolved shapes of the RETURNED placeholders (which substitute PARAMs with external args), not raw body shapes
  rets = tuple(r.shrink_to(rs.shape) for r,rs in zip(rets, (c.src[1+p] for p in ret_pos)))

  # the AFTER outputs resolve against this: stores of each real output into its RETURNED placeholder
  return UOp.sink(*[c.src[1+p].store(v) for p, v in zip(ret_pos, rets)])

# NOTE: adding rules to here is bad. these all need to run before the schedule cache
pm_early_transform_tensor_graph = PatternMatcher([
  # transform precompiled value-producing calls into opaque CALLs (outputs become real buffers)
  (UPat(Ops.CALL, name="c"), transform_precompiled_call),

  # resolve AFTER on RETURNED placeholders (for precompiled calls)
  (UPat(Ops.AFTER, src=(UPat(name="r"), UPat(Ops.SINK, name="t")), allow_any_len=True), resolve_returned_after),

  # fold MOPS+BITCAST over BUFFER into SHRINK when movement ops collapse to contiguous range
  (UPat((Ops.COPY, Ops.STAGE), src=(UPat(GroupOp.Movement|{Ops.BITCAST}, name="src"),), allow_any_len=True, name="c"), contiguous_mops_to_view),
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, name="src"), UPat()), name="c", allow_any_len=True), contiguous_mops_to_view),

  # strip DETACH/CONTIGUOUS_BACKWARD before minting (tags carry over)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"),
   lambda x: x.src[0].replace(tag=(x.src[0].tag or ())+(x.tag or ())) if x.tag else x.src[0]),
  # contiguous of an already-materialized value is a no-op. ALLOC identity alone does not establish materialization.
  (UPat(Ops.STAGE, src=(UPat(Ops.AFTER, name="a"),), name="c"),
   lambda a,c: a.replace(tag=(a.tag or ())+(c.tag or ()))
   if a.src[0].has_buffer_identity() and a.storage_base.op is not Ops.ALLOC else None),
  # mint buffers for tagged values; an untagged CONTIGUOUS flows through to the scheduler, which bufferizes it
  (UPat(GroupOp.All-{Ops.AFTER, Ops.STORE}, name="x"), mint_tagged_storage),
])

# a store's storage keeps the views and drops AFTERs (they only sequence stores)
pm_drop_after = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: a.src[0])])

def replace_input_buffer(ctx:CallifyCtx, b:UOp):
  ctx.replacements.append(b)
  return b.param_like(len(ctx.replacements)-1)

def replace_realized_allreduce_view(ctx:CallifyCtx, b:UOp):
  # Shield the physical view from the bottom-up buffer replacement. The placeholder is numbered in ordinary graph
  # order below, while the runtime argument retains the tagged SHRINK (whose operands are physical offset and size).
  placeholder = b.param_like(-1_000_000-len(ctx.physical_views))
  ctx.physical_views[placeholder] = b
  return placeholder

# ALLOCs get canonical scope-local id slots here so structurally identical calls hash identically for the
# schedule cache (fresh slots are all positive from the global counter; negative slots are already canonical)
def canonicalize_alloc(ctx:CallifyCtx, b:UOp):
  if b.arg.slot >= 0 and b not in ctx.allocs: ctx.allocs[b] = b.replace(arg=replace(b.arg, slot=-1-len(ctx.allocs)))
  return ctx.allocs.get(b)

def canonicalize_call_body(c:UOp):
  return c.replace(src=(graph_rewrite(c.body, pm_canonicalize_alloc, ctx=CallifyCtx(), bottom_up=True),)+c.src[1:])

pm_canonicalize_alloc = PatternMatcher([
  (UPat(Ops.CALL, name="c"), lambda c: canonicalize_call_body(c)),
  (UPat(Ops.ALLOC, name="b"), canonicalize_alloc),
])

pm_replace_buf = pm_canonicalize_alloc+PatternMatcher([
  # Number shielded physical views alongside ordinary buffers so CALL arguments retain graph order.
  (UPat(Ops.PARAM, name="p"), lambda ctx,p: replace_input_buffer(ctx, ctx.physical_views[p]) if p in ctx.physical_views else None),
  # replace BUFFER with PARAM for cache key normalization (ALU addrspace buffers are Variables, they stay)
  (UPat(Ops.BUFFER, name="b"), lambda ctx,b:
   replace_input_buffer(ctx, b) if b.addrspace is AddrSpace.GLOBAL else None),
  # replace buffer views created in this Callify
  (UPat((Ops.SHRINK, Ops.BITCAST), name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b in ctx.views else None),
  # strip the stored value from bound Variables for cache key normalization, so different values hit same cache
  (UPat(Ops.PARAM, name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b.is_bound_var else None),
])

pm_replace_realized_allreduce_views = PatternMatcher([
  (UPat(Ops.SHRINK, tag={("allreduce",)}, name="b"), lambda ctx,b:
   replace_realized_allreduce_view(ctx, b) if b._base_buffer_is_realized() else None),
])

def transform_to_call(big_sink:UOp, buffer_map:dict[UOp, UOp]|None=None) -> UOp:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  if SPEC: type_verify(big_sink, spec_tensor)
  # bases to realize. an AFTER already names the storage its store writes into
  ctx = CallifyCtx(bases={base for x in big_sink.src if (base:=x.base).needs_storage() and base.op is not Ops.AFTER})

  # Preserve original storage identities before rewrites change them.
  big_sink = graph_rewrite(big_sink, add_tags, ctx=ctx, bottom_up=True, name="add tags")

  # Inline calls materialize their unresolved outputs here; precompiled calls bind them in transform_precompiled_call.
  srcs:list[UOp] = []
  for u in big_sink.src:
    if u.op is Ops.AFTER and u.src[0].unsharded_base.op is Ops.ALLOC and u.src[1].op is Ops.CALL:
      call = u.src[1]
      if not (call.arg is not None and call.arg.precompile):
        buf = bind_call_storage(u)
        u = buf.after(buf.store(u.rtag(None))).replace(tag=u.tag)
    srcs.append(u)
  big_sink = big_sink.replace(src=tuple(srcs))

  # here we can break the tensor graph. tags propagate through replaces so we can still find the original UOps
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, ctx=ctx, name="early transform tensor graph")
  big_sink = graph_rewrite(big_sink, pm_mint_function_materializations, name="mint function materializations")

  # collect the stores (never entering call bodies) and map tagged AFTERs to their storage; tags are stripped at the end
  # copies to disk are stores to the disk buffer; bound Variables are call inputs and RETURNEDs are call outputs
  # AFTERs on unbound STORAGE (clones) are collected too: the clone's own buffer is the storage, no fresh copy
  for u in big_sink.toposort(enter_calls=False):
    if (u.op is Ops.COPY and u.on_disk()) or (u.op is Ops.AFTER and not u.is_bound_var and
        (u.src[0].unsharded_base.op is not Ops.ALLOC or u.src[1].op is Ops.STORE)):
      ctx.stores.append(u)
      if u.tag: ctx.buffer_map.update({t:graph_rewrite(u.src[0], pm_drop_after).shrink_to(t.shape) for t in u.tag})
  stores = graph_rewrite(UOp.sink(*ctx.stores), pm_replace_realized_allreduce_views, ctx=ctx,
                         walk=True, name="replace realized allreduce views")
  ret = graph_rewrite(stores, pm_replace_buf+pm_remove_allocation_tags, ctx=ctx,
                      bottom_up=True, name="replace bufs").call(*ctx.replacements, precompile=True)
  assert not any(x in ctx.buffer_map for x in ctx.buffer_map.values())
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  if buffer_map is not None: buffer_map.update(ctx.buffer_map)
  return ret

def is_store_after(u:UOp) -> bool:
  return u.op is Ops.AFTER and (u.src[0].unsharded_base.op is not Ops.ALLOC or u.src[1].op is Ops.STORE)

def create_linear_with_vars(big_sink:UOp, buffer_map:dict[UOp, UOp]|None=None) -> tuple[UOp, dict[str, int]]:
  big_sink = transform_to_call(big_sink, buffer_map)
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

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear)
    return UOp(Ops.LINEAR, src=()), var_vals

  # Caller arguments own their storage even when passed through a physical view. Planning only direct BUFFER arguments
  # redirects eager writes through SHRINK(BUFFER) into a temporary arena; JIT happened to mask this by holding live tensors.
  held_bufs = ({b for x in linear_call.src[1:] for b in _collect_bufs(x)} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
