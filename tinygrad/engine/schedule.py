import time, inspect
from typing import cast
from collections import deque
from tinygrad.uop.ops import UOp, Ops, KernelInfo, buffers, UOpMetaClass, track_rewrites, graph_rewrite, gate_kernel_sink
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR
from tinygrad.engine.realize import ExecItem

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      k = u.src[1]
      assert k.op in {Ops.CALL, Ops.END}, f"AFTER src[1] should be KERNEL or END, not {k.op}"
      in_degree.setdefault(k, 0)
      if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
      # WAR deps from rangeify are stored in AFTER src[2:]
      kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
      for s in kernel_deps + u.src[2:]:
        match (s := _unwrap_src(s)).op:
          case Ops.AFTER:
            children.setdefault(s.src[1], []).append(k)
            in_degree[k] += 1
          case Ops.MSELECT | Ops.MSTACK:
            for ss in s.src:
              if ss.op is Ops.MSELECT: ss = ss.src[0]
              if ss.op not in {Ops.BUFFER, Ops.PARAM}:
                assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
                children.setdefault(ss.src[1], []).append(k)
                in_degree[k] += 1
          case Ops.BUFFER | Ops.PARAM | Ops.BIND:
            pass  # BUFFER/PARAM is already realized, BIND is a bound variable (not a buffer dependency)
          case _:
            raise RuntimeError(f"input to kernel must be AFTER, BUFFER, PARAM, MSELECT, MSTACK, or BIND, not {s.op}")

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      k = rk.src[0] if rk.op is Ops.END else rk
      assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
      buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
      linearized.append(k.src[0].call(*buf_uops, metadata=k.arg.metadata))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.uop.ops import PatternMatcher, UPat

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # tag=True prevents re-matching after replacement (needed when PARAMs replace with PARAMs in nested callify)
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg].replace(tag=True) if x.tag is None else None),
  # create new BUFFERs for LUNIQUE BUFFERs from rangeify
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), create_new_buffer),
])

schedule_cache: dict[bytes, UOp] = {}

def _resolve_params(linear:UOp, params:tuple[UOp, ...]) -> UOp:
  """Replace PARAMs in a LINEAR with the given params (BUFFERs or outer PARAMs), also handling LUNIQUE BUFFERs."""
  from tinygrad.uop.ops import _remove_all_tags
  linear = graph_rewrite(linear, pm_post_sched_cache, ctx=({}, params), name="params to buffers")
  return graph_rewrite(linear, _remove_all_tags, name="remove tags")

def rewrite_call_to_linear(ctx:list, call:UOp) -> UOp|None:
  """Rewrite rule: CALL(SINK, *params) -> LINEAR(...) with caching. Only matches top-level CALLs from transform_to_call."""
  function = call.src[0]
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo): return None
  # recursively schedule any nested CALLs inside the function (from nested callify)
  inner_start = len(ctx)
  function = graph_rewrite(function, pm_schedule, ctx=ctx, name="schedule nested calls")
  if not SCACHE or (linear:=schedule_cache.get(function.key, None)) is None:
    if SPEC: type_verify(call.replace(src=(function,)+call.src[1:]), tensor_spec)
    linear = create_schedule(get_kernel_graph(function))
    if SCACHE: schedule_cache[function.key] = linear
  # late apply params to buffers (tag=True prevents PARAM->PARAM cycles in nested callify)
  linear = _resolve_params(linear, call.src[1:])
  # resolve remaining PARAMs in inner LINEARs from nested CALLs using this call's params
  for i in range(inner_start, len(ctx)):
    inner_call, inner_linear = ctx[i]
    ctx[i] = (inner_call, _resolve_params(inner_linear, call.src[1:]))
  ctx.append((call, linear))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.CALL, name="call"), rewrite_call_to_linear),
  # strip AFTER(buf, LINEAR) -> buf after scheduling
  (UPat(Ops.AFTER, src=(UPat(name="buf"), UPat(Ops.LINEAR))), lambda ctx,buf: buf),
])

def linear_to_schedule(linear:UOp) -> list[ExecItem]:
  """Convert a LINEAR UOp to a list of ExecItems."""
  schedule: list[ExecItem] = []
  for si in linear.src:
    ast, buf_uops = si.src[0], si.src[1:]
    # create subbuffers if needed
    if ast.op is Ops.BUFFER_VIEW:
      base = buf_uops[1].buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[buf_uops[0]] = base.view(buf_uops[0].arg, ast.dtype, ast.arg[1]*base.dtype.itemsize)
    ubufs = [b.buffer for b in buf_uops]
    metadata = si.arg.metadata
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in ast.variables() if x.expr == '_device_num']
      for j, bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ExecItem(ast, list(bufs), metadata, {dnums[0].expr:j} if len(dnums) else {}))
    else:
      schedule.append(ExecItem(ast, list(ubufs), metadata))
  return schedule

# strip AFTER(buf, LINEAR) -> buf, used by _apply_map_to_tensors to clean up scope tensors after scheduling
@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[1]))}")
def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[list[UOp], list[ExecItem], dict[str, int]]:
  st = time.perf_counter()

  # rewrite CALLs to LINEARs and strip AFTERs
  call_linear_pairs: list[tuple[UOp, UOp]] = []
  graph_rewrite(big_sink, pm_schedule, ctx=call_linear_pairs, name="schedule calls")

  # collect ExecItems from all LINEARs
  schedule: list[ExecItem] = []
  for _, linear in call_linear_pairs:
    schedule.extend(linear_to_schedule(linear))

  # get var_vals from CALL params
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for _, linear in call_linear_pairs for si in linear.src])
  var_vals: dict[str, int] = {}
  for call, _ in call_linear_pairs:
    for b in call.src[1:]:
      if b.op is Ops.BIND:
        nm = b.src[0].expr
        if nm not in used_vars: continue
        val = b.src[1].arg
        assert nm not in var_vals or var_vals[nm] == val, f"bind mismatch on {nm}, {var_vals[nm]} != {val}"
        var_vals[nm] = val

  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(schedule):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))

  return [call for call, _ in call_linear_pairs], schedule, var_vals
