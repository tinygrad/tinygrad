import time
from typing import cast
from dataclasses import dataclass, field, replace
from collections import deque
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, track_rewrites
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite, graph_rewrite_map
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import Metadata, DEBUG, cpu_profile, TracingKey, SPEC, flatten, pluralize

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)

@dataclass(frozen=True)
class PreScheduleItem:
  ast: UOp
  buf_uops: tuple[UOp, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  bound_ranges: tuple[UOp, ...] = ()

# **** schedule linearizer

def create_schedule(sched_sink:UOp) -> list[PreScheduleItem]:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # construct the KERNEL children graph based on assigns
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    for u in sched_sink.toposort():
      if u.op is Ops.RANGE:
        in_degree.setdefault(u, 0)
        continue
      if u.op is not Ops.AFTER or u.src[1].op is Ops.RANGE: continue
      k = u.src[1]
      in_degree.setdefault(k, 0)
      for s in k.src[0].src if k.op is Ops.END else k.src:
        if s.op is Ops.AFTER:
          children.setdefault(s.src[1], []).append(k)
          in_degree[k] += 1
        elif s.op in {Ops.MSELECT, Ops.MSTACK}:
          for ss in s.src:
            if ss.op is Ops.MSELECT: ss = ss.src[0]
            if ss.op is not Ops.BUFFER:
              assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
              children.setdefault(ss.src[1], []).append(k)
              in_degree[k] += 1
        elif s.op in {Ops.BUFFER, Ops.BIND}:
          pass  # a BUFFER is already realized, BINDs are handled in complete_create_schedule_with_vars
        else:
          raise RuntimeError(f"input to kernel must be AFTER or BUFFER, not {s.op}")

  with cpu_profile(TracingKey("linearize to PreScheduleItem")):
    queue: deque[UOp] = deque()
    for k,v in in_degree.items():
      if v == 0: queue.append(k)

    schedule: list[PreScheduleItem|UOp] = []
    while len(queue):
      k = rk = queue.popleft()
      if k.op is Ops.END: k = k.src[0]
      if k.op is Ops.RANGE: schedule.append(k)
      elif k.op is Ops.KERNEL:
        ast = k.arg.ast
        buf_uops = tuple(s.buf_uop for s in k.src if s.op is not Ops.BIND)
        bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE)
        schedule.append(PreScheduleItem(ast, buf_uops, k.arg.metadata, bound_ranges=bound_ranges))
        if rk.op is Ops.END: schedule.append(rk)
      else:
        raise RuntimeError(f"can't schedule {k.op}")
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)

  with cpu_profile(TracingKey("expand ranges")):
    real_schedule: list[PreScheduleItem] = []
    sched_ptr = 0
    in_ranges = {}
    range_ptrs = {}
    while sched_ptr < len(schedule):
      si = schedule[sched_ptr]
      if isinstance(si, UOp):
        if si.op is Ops.RANGE:
          in_ranges[si] = 0
          range_ptrs[si] = sched_ptr + 1
        elif si.op is Ops.END:
          if in_ranges[si.src[1]] < si.src[1].vmax:
            in_ranges[si.src[1]] += 1
            sched_ptr = range_ptrs[si.src[1]]
            continue
      else:
        real_schedule.append(replace(si, fixedvars=si.fixedvars | {s.src[0].arg[0]:in_ranges[s.src[1]] for s in si.bound_ranges}, bound_ranges=()))
      sched_ptr += 1
  return real_schedule

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

def resolve_buf_uop(buf_uop: UOp, ctx: dict[UOp, UOp]) -> UOp:
  """Resolve a LUNIQUE buf_uop to its UNIQUE version."""
  if buf_uop.op is Ops.BUFFER:
    if buf_uop in ctx:
      return ctx[buf_uop]
    # New buffer created during rangeify - create with UNIQUE
    ctx[buf_uop] = ret = UOp.new_buffer(buf_uop.device, buf_uop.arg, buf_uop.dtype)
    return ret
  elif buf_uop.op is Ops.MSELECT:
    resolved_src = resolve_buf_uop(buf_uop.src[0], ctx)
    return resolved_src.mselect(buf_uop.arg)
  elif buf_uop.op is Ops.MSTACK:
    resolved_srcs = tuple(resolve_buf_uop(s, ctx) for s in buf_uop.src)
    return UOp(Ops.MSTACK, buf_uop.dtype, src=resolved_srcs)
  else:
    raise RuntimeError(f"unexpected buf_uop op: {buf_uop.op}")

def finalize_schedule(pre_schedule: list[PreScheduleItem], input_buffers_reverse: dict[UOp, UOp]) -> list[ScheduleItem]:
  """Convert PreScheduleItems to ScheduleItems by replacing LUNIQUE with UNIQUE and resolving Buffers."""
  schedule: list[ScheduleItem] = []
  for psi in pre_schedule:
    # Resolve LUNIQUE buf_uops to UNIQUE
    buf_uops = tuple(resolve_buf_uop(b, input_buffers_reverse) for b in psi.buf_uops)

    # Handle BUFFER_VIEW: create subbuffer
    if psi.ast.op is Ops.BUFFER_VIEW:
      base = buf_uops[1].buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[buf_uops[0]] = base.view(buf_uops[0].arg, psi.ast.dtype, psi.ast.arg[1]*base.dtype.itemsize)

    # Get actual Buffer objects
    ubufs = tuple(b.buffer for b in buf_uops)

    # Handle MultiBuffer: expand to multiple ScheduleItems
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in psi.ast.variables() if x.arg[0] == '_device_num']
      for i, bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ScheduleItem(psi.ast, bufs, psi.metadata, psi.fixedvars | ({dnums[0].expr:i} if len(dnums) else {})))
    else:
      schedule.append(ScheduleItem(psi.ast, cast(tuple[Buffer, ...], ubufs), psi.metadata, psi.fixedvars))

  return schedule

def replace_input_buffer(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None:
    if b.op is Ops.BUFFER:
      ctx[b] = ret = b.replace(src=(UOp(Ops.LUNIQUE, arg=len(ctx)), b.src[1]))
    else:
      # TODO: flip args in CONST
      assert b.op is Ops.CONST
      ctx[b] = ret = b.replace(src=(b.src[0], UOp(Ops.LUNIQUE, arg=len(ctx))))
  return ret

pm_pre_sched_cache = PatternMatcher([
  # replace input buffers
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # remove unique consts
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE), UPat(Ops.UNIQUE)), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), lambda ctx,b: ctx.setdefault(b, b.replace(src=(b.src[0],)))),
])

def replace_input_buffer_back(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None:
    assert b.op is Ops.BUFFER
    # if it's not in the cache, create a new buffer
    ctx[b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer_back),
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE), UPat(Ops.LUNIQUE)), name="b"), replace_input_buffer_back),
  # restore BIND value stripped in pm_pre_sched_cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR),), name="b"), lambda ctx,b: ctx.get(b)),
])

schedule_cache: dict[bytes, tuple[list[PreScheduleItem], UOp, UOp]] = {}
@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[1]))}")
def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[dict[UOp, UOp], list[ScheduleItem], dict[str, int]]:
  # big_sink srcs are all the Tensors
  st = time.perf_counter()

  # replace all UNIQUE buffers with LUNIQUE, strip BIND values for cache key
  input_buffers: dict[UOp, UOp] = {}
  big_sink_cache = graph_rewrite(big_sink, pm_pre_sched_cache, ctx=input_buffers, name="rewrite for sched cache")
  sched_cache_key = big_sink_cache.key

  if (sc_ret:=schedule_cache.get(sched_cache_key, None)) is None:
    # verify Tensors match the spec (on big_sink, we only need to do this if cache misses)
    if SPEC: type_verify(big_sink, tensor_spec)

    # hack to preserve metadata
    graph_rewrite_map(big_sink, pm_pre_sched_cache, ctx={}, name="preserve metadata")

    # tensor map is what we return
    tensor_map: dict[UOp, UOp] = {}

    if any(isinstance(x._device, tuple) for x in big_sink_cache.toposort()):
      tensor_map |= get_multi_map(big_sink_cache)
      big_sink_cache = big_sink_cache.substitute(tensor_map, name="Apply Multi Map")
      big_sink_cache = UOp.sink(*flatten([x.src if x.op is Ops.MULTI else [x] for x in big_sink_cache.src]))

    tensor_map |= get_rangeify_map(big_sink_cache)
    big_sink = big_sink_cache.substitute(tensor_map, name="Apply Kernelize Map")

    # create the schedule with LUNIQUE placeholders
    pre_schedule = create_schedule(big_sink)

    # save in schedule cache: pre_schedule, tensor_map_sink, big_sink
    tensor_map_sink = UOp.sink(*flatten([(k,v) for k,v in tensor_map.items()]))
    schedule_cache[sched_cache_key] = (pre_schedule, tensor_map_sink, big_sink)
  else:
    # schedule cache hit - skip create_schedule
    del big_sink_cache
    pre_schedule, tensor_map_sink, big_sink = sc_ret

  # replace LUNIQUE with UNIQUE in ScheduleItems
  input_buffers_reverse = {v:k for k,v in input_buffers.items()}
  schedule = finalize_schedule(pre_schedule, input_buffers_reverse)
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  # extract var_vals from BINDs that were stripped (only if there are kernels)
  var_vals: dict[str, int] = {}
  if schedule:
    for u in input_buffers:
      if u.op is Ops.BIND:
        var, val = u.unbind()
        assert var.expr not in var_vals or var_vals[var.expr] == val, f"bind mismatch on {var}, {var_vals[var.expr]} != {val}"
        var_vals[var.expr] = val

  # replace LUNIQUE in tensor_map_sink and big_sink for tensor_map
  tm_src = graph_rewrite(tensor_map_sink, pm_post_sched_cache, ctx=input_buffers_reverse, name="unrewrite for tensor map").src
  tensor_map = {tm_src[i]:tm_src[i+1] for i in range(0, len(tm_src), 2)}

  # remove all AFTERs, after scheduling, the tensors are just buffers
  big_sink = graph_rewrite(big_sink, pm_post_sched_cache, ctx=input_buffers_reverse, name="unrewrite for sched cache")
  tensor_map |= {u:u.buf_uop for u in big_sink.toposort() if u.op is Ops.AFTER}

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule):4d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if sc_ret is not None else 'CACHE MISS'} {sched_cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache)} uops in cache")
  return tensor_map, schedule, var_vals
