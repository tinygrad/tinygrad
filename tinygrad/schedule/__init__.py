import time, inspect
from collections import deque
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, track_rewrites, graph_rewrite, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and (s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.SLICE, Ops.BIND} or
                        (s.op is Ops.SLICE and s.src[0].op is Ops.INDEX)): s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | PARAM. MSELECT/MSTACK join per-device states, BIND is not a buffer dependency
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  if s.op is Ops.SLICE: return _states(s.src[0])
  if s.op is Ops.BIND: return []
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _slice_region(s: UOp) -> tuple[UOp, int, int]|None:
  offset, size = 0, None
  while True:
    s = _unwrap_src(s)
    if s.op is Ops.AFTER: s = s.src[0]
    elif s.op is Ops.SLICE and s.src[1].op is Ops.CONST:
      offset += s.src[1].arg * s.src[0].dtype.itemsize
      if size is None: size = s.arg * s.dtype.itemsize
      s = s.src[0]
    else: break
  return (s.buf_uop, offset, offset+size) if size is not None else None

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, list[tuple[UOp, UOp, tuple[UOp, ...]]]] = {}  # buffer -> (AFTER, prior state, new kernels)
    reads: list[tuple[UOp, UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read, access)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      prev_state = _unwrap_src(u.src[0])
      prev_kernels = set(_split_after(prev_state)[0]) if prev_state.op is Ops.AFTER else set()
      writes.setdefault(u.buf_uop, []).append((u, prev_state, tuple(k for k in kernels if k not in prev_kernels)))
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
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
      for a, prev_state, write_kernels in writes.get(s.buf_uop, []):
        if a is u or prev_state is not s: continue
        for t in write_kernels:
          call = t.src[0] if t.op is Ops.END else t
          if (rr:=_slice_region(access)) is not None and len(call.src) > 1 and (wr:=_slice_region(call.src[1])) is not None \
             and (rr[0] is not wr[0] or rr[2] <= wr[1] or wr[2] <= rr[1]): continue
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    # Continue through one precompiled CALL while COPY consumers wait on another queue, then drain every consumer that
    # was ready together. A CALL of a LINEAR is the graph-level stage boundary, independent of model and kernel names.
    def is_copy(parent:UOp) -> bool:
      call = parent.src[0] if parent.op is Ops.END else parent
      return call.op is Ops.CALL and call.src[0].op is Ops.COPY
    copy_parents: dict[UOp, set[UOp]] = {}
    for parent, childs in children.items():
      if is_copy(parent):
        for child in childs: copy_parents.setdefault(child, set()).add(parent)
    copy_children = set(copy_parents)
    def is_precompiled(parent:UOp) -> bool:
      call = parent.src[0] if parent.op is Ops.END else parent
      return call.op is Ops.CALL and call.src[0].op is Ops.LINEAR
    linear_distance: dict[UOp, int|None] = {}
    def distance_to_precompiled(root:UOp) -> int|None:
      stack = [(root, False)]
      while stack:
        node, visited = stack.pop()
        if node in linear_distance: continue
        if is_precompiled(node): linear_distance[node] = 0
        elif visited:
          distances = [d for x in children.get(node, []) if (d:=linear_distance[x]) is not None]
          linear_distance[node] = 1 + min(distances) if len(distances) else None
        else:
          stack.append((node, True))
          stack.extend((x, False) for x in children.get(node, []) if x not in linear_distance)
      return linear_distance[root]
    def reachable_distance(root:UOp) -> int:
      ret = distance_to_precompiled(root)
      assert ret is not None
      return ret
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    continuation: deque[UOp] = deque()
    overlap_group: set[UOp] = set()
    overlapping, drain_group, stop_at_precompiled = False, False, False
    overlap_steps, overlap_limit = 0, 0
    linearized: list[UOp] = []
    while len(queue) or len(continuation):
      ready_group = [x for x in queue if x in overlap_group]
      if drain_group and len(ready_group):
        rk = ready_group[0]
        queue.remove(rk)
      elif overlapping and len(continuation):
        reachable = [x for x in continuation if distance_to_precompiled(x) is not None]
        rk = min(reachable, key=reachable_distance) if stop_at_precompiled and len(reachable) else continuation[0]
        continuation.remove(rk)
      elif len(continuation) and len(queue) and queue[0] in copy_children:
        overlap_group = {x for x in queue if x in copy_children}
        reachable = [x for x in continuation if distance_to_precompiled(x) is not None]
        stop_at_precompiled = len(reachable) != 0
        overlap_limit = len(copy_parents[queue[0]])
        overlap_steps = 0
        overlapping = True
        rk = min(reachable, key=reachable_distance) if stop_at_precompiled else continuation[0]
        continuation.remove(rk)
      elif len(continuation) and (not len(queue) or queue[0] not in copy_children): rk = continuation.popleft()
      else: rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
        linearized.append(k.src[0].call(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0:
          if x not in copy_children and len(queue) and queue[0] in copy_children: continuation.append(x)
          else: queue.append(x)
      if overlapping:
        overlap_steps += 1
        if is_precompiled(rk) or not len(continuation) or (not stop_at_precompiled and overlap_steps >= overlap_limit):
          overlapping, drain_group = False, True
      if drain_group and not any(x in overlap_group for x in queue):
        drain_group = False
        overlap_group.clear()
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat, ParamArg
from tinygrad.dtype import AddrSpace

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # only resolve buffer PARAMs (slot>=0); ALU/shape vars use slot=-1 and must not be swapped for call args
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot] if x.arg.slot >= 0 else None),
  # create new BUFFERs
  (UPat(Ops.BUFFER, src=(UPat(),), name="b"), lambda ctx,b:
   create_new_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
])

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), lambda linear_call:
   graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(function:UOp) -> UOp|None:
  st = time.perf_counter()
  if isinstance(function.arg, KernelInfo): return None
  cache_key = function.key
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(function))
    if SCACHE: schedule_cache[cache_key] = linear
  else:
    # schedule cache hit
    linear = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.SINK, name="function"), lower_sink_to_linear),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.op is Ops.BIND:
      nm = b.src[0].expr
      if nm not in used_vars: continue
      val = b.src[1].arg
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for src in linear_call.src[1:] for b in _collect_bufs(src)} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
