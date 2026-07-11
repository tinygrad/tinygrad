import time, inspect
from collections import deque
from dataclasses import replace
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, track_rewrites, graph_rewrite, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | PARAM. MSELECT/MSTACK join per-device states, BIND is not a buffer dependency
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  if s.op is Ops.BIND: return []
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def _kernel_io(call:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]|None:
  if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK: return None
  out_slots = {x.src[0].src[0].arg.slot for x in call.src[0].toposort()
               if x.op is Ops.STORE and x.src[0].op is Ops.INDEX and x.src[0].src[0].op is Ops.PARAM}
  if len(out_slots) != 1 or min(out_slots) < 0 or max(out_slots) >= len(call.src)-1: return None
  return tuple(x for i,x in enumerate(call.src[1:]) if i in out_slots), tuple(x for i,x in enumerate(call.src[1:]) if i not in out_slots)

def _input_indices(call:UOp, inputs:tuple[UOp, ...]) -> set[tuple[UOp, tuple[UOp, ...]]]:
  return {(call.src[x.src[0].arg.slot+1], x.src[1:]) for x in call.src[0].toposort()
          if x.op is Ops.INDEX and x.src[0].op is Ops.PARAM and call.src[x.src[0].arg.slot+1] in inputs}

def _is_sum_sumsq(a:UOp, b:UOp) -> bool:
  av, bv = a.src[0], b.src[0]
  return (bv.op is Ops.MUL and bv.src == (av, av)) or (av.op is Ops.MUL and av.src == (bv, bv))

def _substitute_uops(ast:UOp, mapping:dict[UOp, UOp]) -> UOp:
  memo: dict[UOp, UOp] = {}
  def rec(x:UOp) -> UOp:
    if x in mapping: return mapping[x]
    if x not in memo:
      src = tuple(rec(s) for s in x.src)
      memo[x] = x if src == x.src else x.replace(src=src)
    return memo[x]
  return rec(ast)

def _remap_params(ast:UOp, remap:dict[int, int]) -> UOp:
  return _substitute_uops(ast, {x:x.replace(arg=replace(x.arg, slot=remap[x.arg.slot])) for x in ast.toposort()
                                if x.op is Ops.PARAM and x.arg.slot >= 0})

def _fuse_adjacent_reductions(linearized:list[UOp]) -> list[UOp]:
  ret: list[UOp] = []
  i = 0
  while i < len(linearized):
    if i+1 == len(linearized):
      ret.append(linearized[i])
      break
    a, b = linearized[i:i+2]
    aio, bio = _kernel_io(a), _kernel_io(b)
    ar = [x for x in a.src[0].toposort() if x.op is Ops.REDUCE] if aio is not None else []
    br = [x for x in b.src[0].toposort() if x.op is Ops.REDUCE] if bio is not None else []
    compatible = aio is not None and bio is not None and len(ar) == len(br) == 1 and ar[0].arg == br[0].arg and \
      ar[0].src[1:] == br[0].src[1:] and _is_sum_sumsq(ar[0], br[0]) and aio[1] == bio[1] and \
      _input_indices(a, aio[1]) == _input_indices(b, bio[1]) and aio[0] != bio[0] and \
      len(a.src[0].src) == len(b.src[0].src) == 1 and a.src[0].src[0].op is Ops.END and b.src[0].src[0].op is Ops.END and \
      a.src[0].src[0].src[1:] == b.src[0].src[0].src[1:]
    if not compatible:
      ret.append(a)
      i += 1
      continue
    assert aio is not None and bio is not None
    args = list(a.src[1:])
    if len(set(args)) != len(args) or any(x in args for x in bio[0]):
      ret.append(a)
      i += 1
      continue
    args.extend(bio[0])
    remap = {slot:args.index(x) for slot,x in enumerate(b.src[1:])}
    bast = _remap_params(b.src[0], remap)
    ret.append(a.src[0].replace(src=a.src[0].src+bast.src).call(*args))
    i += 2
  return ret

def _fuse_dependent_reductions(linearized:list[UOp]) -> list[UOp]:
  ret = list(linearized)
  i = 0
  while i < len(ret):
    a, aio = ret[i], _kernel_io(ret[i])
    if aio is None or len(a.src[0].src) != 1 or a.src[0].src[0].op is not Ops.END:
      i += 1
      continue
    ar = [x for x in a.src[0].toposort() if x.op is Ops.REDUCE]
    if len(ar) != 1 or ar[0].arg != (Ops.MAX, 0) or len(ar[0].src[1:]) != 2 or any(int(x.vmax)+1 != 2 for x in ar[0].src[1:]):
      i += 1
      continue
    aend, astore = a.src[0].src[0], a.src[0].src[0].src[0]
    if astore.op is not Ops.STORE:
      i += 1
      continue
    spatial = tuple(int(x.vmax)+1 for x in aend.src[-2:])
    if len(aend.src[1:]) != 4 or spatial[0] != spatial[1] or spatial[0] & (spatial[0]-1):
      i += 1
      continue
    for j in range(i+1, min(i+6, len(ret))):
      b, bio = ret[j], _kernel_io(ret[j])
      if bio is None or len(b.src[0].src) != 1 or b.src[0].src[0].op is not Ops.END: continue
      br = [x for x in b.src[0].toposort() if x.op is Ops.REDUCE]
      if len(br) != 1 or br[0].arg != (Ops.ADD, 0) or ar[0].src[1:] != br[0].src[1:]: continue
      bend = b.src[0].src[0]
      if aend.src[1:] != bend.src[1:] or aio[0][0] not in bio[1] or set(bio[1]) != set(aio[1]+aio[0]): continue
      args = list(a.src[1:])
      if len(set(args)) != len(args) or bio[0][0] in args: continue
      args.extend(bio[0])
      pout_slot = tuple(b.src[1:]).index(aio[0][0])
      pout_idxs = [x for x in b.src[0].toposort() if x.op is Ops.INDEX and x.src[0].op is Ops.PARAM and x.src[0].arg.slot == pout_slot]
      if not pout_idxs or any(x.src[1:] != astore.src[0].src[1:] for x in pout_idxs): continue
      mapping = {x:astore.src[1] for x in pout_idxs}
      for x in b.src[0].toposort():
        if x.op is Ops.PARAM and x.arg.slot >= 0 and x.arg.slot != pout_slot:
          mapping[x] = x.replace(arg=replace(x.arg, slot=args.index(b.src[x.arg.slot+1])))
      bend = _substitute_uops(bend, mapping)
      ret[i] = a.src[0].replace(src=(aend, bend)).call(*args)
      ret.pop(j)
      break
    i += 1
  return ret

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, list[tuple[UOp, UOp, tuple[UOp, ...]]]] = {}  # buffer -> (AFTER, prior state, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
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
        read_states = [st for s in kernel_deps for st in _states(s)]
        reads += [(u, k, st) for st in read_states]
        # RAW deps: a kernel runs after the kernels that produced the states it reads or joins
        for st in read_states + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            for t in _split_after(st)[0]:
              children.setdefault(t, []).append(k)
              in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; join members already present in that prior state are ordering deps, not writes
    for u, k, s in reads:
      for a, prev_state, write_kernels in writes.get(s.buf_uop, []):
        if a is u or prev_state is not s: continue
        for t in write_kernels:
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
        linearized.append(k.src[0].call(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(_fuse_dependent_reductions(_fuse_adjacent_reductions(linearized))))

from tinygrad.schedule.memory import memory_plan_rewrite
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
  cache_key = function.key if SCACHE else b""
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

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
