import time, inspect, dataclasses
from collections import deque
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, rewrite_group, graph_rewrite, gate_kernel_sink, KernelInfo, ProgramInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, dedup

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
  return s

# unwrap per-device buffer arguments without dropping their ordering dependencies
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _call_access(call:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  body = call.src[0]
  if body.op is Ops.SINK and not body.op_in_backward_slice_with_self(Ops.CALL, Ops.CUSTOM, Ops.CUSTOMI, Ops.INS):
    from tinygrad.codegen import pm_add_loads
    info = ProgramInfo.from_sink(graph_rewrite(body, pm_add_loads))
    ins, outs = info.ins, info.outs
  elif body.op is Ops.PROGRAM and isinstance(body.arg, ProgramInfo): ins, outs = body.arg.ins, body.arg.outs
  elif body.op is Ops.COPY: ins, outs = (1,), (0,)
  elif body.op is Ops.LINEAR:
    ins, outs = (tuple(sorted({b.arg.slot for args in group for a in args if (b:=a.buf_uop).op is Ops.PARAM}))
                 for group in zip(*(_call_access(c) for c in body.src))) if body.src else ((), ())
  else: return call.src[1:], call.src[1:]  # opaque calls conservatively may read and write every argument
  return tuple(call.src[i+1] for i in ins), tuple(call.src[i+1] for i in outs)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    afters = [u for u in sched_sink.toposort(gate_kernel_sink) if u.op is Ops.AFTER]
    kernels = dict.fromkeys(k for u in afters for k in u.src[1:] if k.op in {Ops.CALL, Ops.END})
    dependencies: dict[UOp, set[UOp]] = {}
    writes: dict[UOp, set[UOp]] = {}
    reads: list[tuple[UOp, UOp]] = []
    ancestors: dict[UOp, set[UOp]] = {}
    for k in kernels:
      call = k.src[0] if k.op is Ops.END else k
      states = [st for s in call.src[1:] for st in _states(s)]
      for st in states:
        if st not in ancestors: ancestors[st] = kernels.keys() & st.toposort(enter_calls=False).keys()
      # AFTER supplies ordering dependencies, not evidence that its returned buffer was written.
      dependencies[k] = set().union(*(ancestors[st] for st in states))
      read_args, write_args = _call_access(call)
      reads += [(k, st) for s in read_args for st in _states(s)]
      for s in write_args:
        for st in _states(s): writes.setdefault(st.buf_uop, set()).add(k)
    for u in afters:
      for dep in (s for s in u.src[1:] if s.op is Ops.AFTER):
        for k in (s for s in u.src[1:] if s in kernels):
          dependencies[k].update(kernels.keys() & dep.toposort(enter_calls=False).keys() - {k})
    # A read must precede writes which are not part of the state it requested.
    for k, st in reads:
      for writer in writes.get(st.buf_uop, set()):
        if writer is not k and writer not in ancestors[st]: dependencies[writer].add(k)
    children: dict[UOp, list[UOp]] = {}
    in_degree = {k:len(deps) for k,deps in dependencies.items()}
    for k, deps in dependencies.items():
      for p in deps: children.setdefault(p, []).append(k)

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      k = rk.src[0] if rk.op is Ops.END else rk
      assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
      buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if not s.is_bound_var)
      body = k.src[0]
      # Storage aliases may share a parameter now that their dependencies are in the schedule.
      if body.op is Ops.SINK and len(set(buf_uops)) != len(buf_uops):
        params = {p for p in body.toposort(enter_calls=False) if p.op is Ops.PARAM and p.arg.slot >= 0}
        body = body.substitute({p:q for p in params
                               if (q:=p.replace(arg=dataclasses.replace(p.arg, slot=buf_uops.index(buf_uops[p.arg.slot])))) in params})
      linearized.append(body.call(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.prepare import prepare_rangeify
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
  (UPat(Ops.BUFFER, src=(), name="b"), lambda ctx,b:
   create_new_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
])

def resolve_linear_call(linear_call:UOp, outer_binds:dict[str, UOp]|None=None):
  linear = graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{f"p{i}":x.src[0].replace(op=Ops.PARAM) for i,x in enumerate(linear_call.src[1:]) if x.is_bound_var}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.src[0].op is Ops.LINEAR: return resolve_linear_call(si, binds)
    subs = {v:binds[v.expr] for v in si.variables() if v.expr in binds}
    return si.replace(src=tuple(s.substitute(subs, name="resolve scalar params") for s in si.src))
  return linear.replace(src=tuple(apply_binds(si) for si in linear.src))

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(call:UOp) -> UOp|None:
  function = call.src[0]
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo): return None
  # value calls (with unbound outputs) are inlined positionally during prepare: their bodies are not programs to schedule
  if call.has_unbound_outputs: return None
  st = time.perf_counter()
  cache_key = function.key
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
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
  return call.replace(src=(linear,)+call.src[1:])

pm_schedule = PatternMatcher([
  (UPat(Ops.CALL, name="call"), lower_sink_to_linear),
])

def assert_all_same_devices(ast:UOp):
  devices = dedup([x.device for x in ast.toposort() if x.op is Ops.PARAM and x.device is not None])
  if len(devices) >= 2: raise RuntimeError(f"all buffers must be on the same device: {devices}")

def copy_kernel_to_copy_uop(call:UOp, dst:UOp, src:UOp, r:UOp|None=None):
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  return call.replace(src=(UOp(Ops.COPY, src=(src,), arg=dst.device),) + call.src[1:])

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  from tinygrad.codegen.simplify import pm_flatten_range, pm_simplify_ranges
  from tinygrad.schedule.prepare import pm_mops
  from tinygrad.uop.symbolic import sym
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges in copy")
  return call.replace(src=(sink,) + call.src[1:])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),

  # replace this with a copy if it's a copy
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.CONST, arg=0))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.CONST, arg=0))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_copy_uop),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r"))).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_copy_uop),

  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

@rewrite_group(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # create copies
  linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals from the bound Variables in the call args
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.is_bound_var:
      v, val = b.unbind()
      nm = v.expr
      if nm not in used_vars: continue
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
