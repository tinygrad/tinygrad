import time
from typing import cast
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, track_rewrites
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite, graph_rewrite_map
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, flatten, pluralize
from tinygrad.engine.realize import ExecItem

def create_schedule(sched_sink:UOp) -> list[ExecItem]:
  from tinygrad.codegen.late.linearizer import linearize
  with cpu_profile(TracingKey("linearize sched_sink")): ordered = linearize(sched_sink, for_scheduling=True)
  with cpu_profile(TracingKey("linearize to schedule")):
    # internal: (ast, bufs, metadata, fixedvars, bound_ranges) or UOp for RANGE/END
    schedule:list[tuple|UOp] = []
    seen_kernels:set[UOp] = set()
    for u in ordered:
      if u.op is Ops.RANGE: schedule.append(u)
      elif u.op is Ops.AFTER and u.src[1].op is not Ops.RANGE:
        rk = k = u.src[1]
        if k.op is Ops.END: k = k.src[0]
        if k.op is Ops.KERNEL and k not in seen_kernels:
          seen_kernels.add(k)
          ast = k.arg.ast
          if ast.op is Ops.BUFFER_VIEW:
            base = k.src[1].buf_uop.buffer
            assert isinstance(base, Buffer), "base can't be MultiBuffer"
            buffers[k.src[0].buf_uop] = base.view(k.src[0].buf_uop.arg, ast.dtype, ast.arg[1]*base.dtype.itemsize)
          ubufs = [s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND]
          bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE)
          if any(isinstance(x, MultiBuffer) for x in ubufs):
            assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
            dnums = [x for x in ast.variables() if x.arg[0] == '_device_num']
            for i,bufs in enumerate(zip(*[x.bufs for x in cast(list[MultiBuffer], ubufs)])):
              schedule.append((ast, list(bufs), k.arg.metadata, {dnums[0].expr:i} if len(dnums) else {}, bound_ranges))
          else: schedule.append((ast, ubufs, k.arg.metadata, {}, bound_ranges))
          if rk.op is Ops.END: schedule.append(rk)
  with cpu_profile(TracingKey("expand ranges")):
    real_schedule:list[ExecItem] = []
    sched_ptr = 0
    in_ranges:dict[UOp, int] = {}
    range_ptrs:dict[UOp, int] = {}
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
        ast, bufs, metadata, fixedvars, bound_ranges = si
        fixedvars = fixedvars | {s.src[0].arg[0]:in_ranges[s.src[1]] for s in bound_ranges}
        real_schedule.append(ExecItem(ast, bufs, metadata, fixedvars))
      sched_ptr += 1
  return real_schedule

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

def replace_input_buffer(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None:
    if b.op is Ops.BUFFER: ctx[b] = ret = b.replace(src=(UOp(Ops.LUNIQUE, arg=len(ctx)), b.src[1]))
    else:
      assert b.op is Ops.CONST
      ctx[b] = ret = b.replace(src=(b.src[0], UOp(Ops.LUNIQUE, arg=len(ctx))))
  return ret

pm_pre_sched_cache = PatternMatcher([
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE), UPat(Ops.UNIQUE)), name="b"), replace_input_buffer),
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), lambda ctx,b: ctx.setdefault(b, b.replace(src=(b.src[0],)))),
])

def replace_input_buffer_back(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None:
    assert b.op is Ops.BUFFER
    ctx[b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer_back),
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE), UPat(Ops.LUNIQUE)), name="b"), replace_input_buffer_back),
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR),), name="b"), lambda ctx,b: ctx.get(b)),
])

schedule_cache:dict[bytes, tuple[UOp, UOp]] = {}
@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[1]))}")
def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[dict[UOp, UOp], list[ExecItem], dict[str, int]]:
  st = time.perf_counter()
  input_buffers:dict[UOp, UOp] = {}
  big_sink_cache = graph_rewrite(big_sink, pm_pre_sched_cache, ctx=input_buffers, name="rewrite for sched cache")
  sched_cache_key = big_sink_cache.key
  if (sc_ret:=schedule_cache.get(sched_cache_key, None)) is None:
    if SPEC: type_verify(big_sink, tensor_spec)
    graph_rewrite_map(big_sink, pm_pre_sched_cache, ctx={}, name="preserve metadata")
    tensor_map:dict[UOp, UOp] = {}
    if any(isinstance(x._device, tuple) for x in big_sink_cache.toposort()):
      tensor_map |= get_multi_map(big_sink_cache)
      big_sink_cache = big_sink_cache.substitute(tensor_map, name="Apply Multi Map")
      big_sink_cache = UOp.sink(*flatten([x.src if x.op is Ops.MULTI else [x] for x in big_sink_cache.src]))
    tensor_map |= get_rangeify_map(big_sink_cache)
    big_sink = big_sink_cache.substitute(tensor_map, name="Apply Kernelize Map")
    tensor_map_sink = UOp.sink(*flatten([(k,v) for k,v in tensor_map.items()]))
    schedule_cache[sched_cache_key] = (big_sink, tensor_map_sink)
  else:
    del big_sink_cache
    big_sink, tensor_map_sink = sc_ret
  input_buffers_reverse = {v:k for k,v in input_buffers.items()}
  big_sink = graph_rewrite(big_sink, pm_post_sched_cache, ctx=input_buffers_reverse, name="unrewrite for sched cache")
  tm_src = graph_rewrite(tensor_map_sink, pm_post_sched_cache, ctx=input_buffers_reverse, name="unrewrite for tensor map").src
  tensor_map = {tm_src[i]:tm_src[i+1] for i in range(0, len(tm_src), 2)}
  schedule = create_schedule(big_sink)
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)
  var_vals:dict[str, int] = {}
  if schedule:
    for u in input_buffers:
      if u.op is Ops.BIND:
        var, val = u.unbind()
        assert var.expr not in var_vals or var_vals[var.expr] == val, f"bind mismatch on {var}, {var_vals[var.expr]} != {val}"
        var_vals[var.expr] = val
  tensor_map |= {u:u.buf_uop for u in big_sink.toposort() if u.op is Ops.AFTER}
  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule):4d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if sc_ret is not None else 'CACHE MISS'} {sched_cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache)} uops in cache")
  return tensor_map, schedule, var_vals
