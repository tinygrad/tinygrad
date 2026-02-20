import time
from typing import cast
from collections import deque
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, track_rewrites, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, gate_kernel_sink
from tinygrad.uop.ops import _remove_all_tags, GroupOp
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import ImageDType
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, flatten, pluralize, SCACHE, prod
from tinygrad.engine.realize import ExecItem

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

def create_schedule(sched_sink:UOp) -> tuple[list[ExecItem], UOp]:
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
    pre_schedule: list[ExecItem] = []
    buf_uops_list: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      k = rk.src[0] if rk.op is Ops.END else rk
      assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
      buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
      pre_schedule.append(ExecItem(k.src[0], [], k.arg.metadata))
      buf_uops_list.append(UOp.sink(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)

  return pre_schedule, UOp.sink(*buf_uops_list)

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

def replace_input_buffer(ctx:tuple[dict[UOp, UOp], dict[str, int], list[int], list[int]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None:
    # replace BUFFER with PARAM for cache key normalization (same as CALL)
    ctx[0][b] = ret = UOp.param(ctx[2][0], b.dtype, b.shape, b.device)
    ctx[2][0] += 1
  return ret

def replace_input_const(ctx:tuple[dict[UOp, UOp], dict[str, int], list[int], list[int]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None:
    # replace UNIQUE with LUNIQUE for CONST cache key normalization
    ctx[0][b] = ret = b.replace(src=(UOp(Ops.LUNIQUE, arg=ctx[3][0]), b.src[1]))
    ctx[3][0] += 1
  return ret

def strip_bind(ctx:tuple[dict[UOp, UOp], dict[str, int], list[int], list[int]], b:UOp):
  var, val = b.src[0], b.src[1].arg
  assert var.expr not in ctx[1] or ctx[1][var.expr] == val, f"bind mismatch on {var}, {ctx[1][var.expr]} != {val}"
  ctx[1][var.expr] = val
  return ctx[0].setdefault(b, b.replace(src=(b.src[0],)))

pm_pre_sched_cache = PatternMatcher([
  # replace BUFFER with PARAM for cache key normalization
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # replace UNIQUE with LUNIQUE for CONST cache key normalization
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_const),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), strip_bind),
])

def create_new_buffer(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None: ctx[b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # create new BUFFERs for LUNIQUE BUFFERs from rangeify
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), create_new_buffer),
  # restore CONST back to original CONST
  (UPat(Ops.CONST, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), lambda ctx,b: ctx.get(b)),
  # restore PARAM back to original BUFFER
  (UPat(Ops.PARAM, src=(UPat(), UPat(Ops.DEVICE)), name="b"), lambda ctx,b: ctx.get(b)),
  # restore BIND value stripped in pm_pre_sched_cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR),), name="b"), lambda ctx,b: ctx.get(b)),
])

# these are the only uops that can get replaced in the tensor graph
from tinygrad.schedule.rangeify import pm_gate_kernel_sink

def tag_uop(ctx:tuple[list[UOp], set[UOp], dict[UOp, UOp], set[UOp]], x:UOp):
  if x.tag is not None or x in ctx[1]: return None
  if x.tag is None and x.op is Ops.CALL:
    # don't tag anything in a CALL
    for u in x.src[0].toposort(): ctx[1].add(u)
  ctx[0].append(x)
  return x.replace(tag=(len(ctx[0])-1,))

def disk_copy_is_buffer(ctx, u):
  if isinstance(u._device, str) and u._device.startswith("DISK"):
    ctx[2][u] = UOp.new_buffer(u.device, u.shard_size, u.dtype).reshape(u.max_shard_shape)

def apply_after(ctx, u):
  ctx[2][u] = u.src[0]

# CONTIGUOUS and ASSIGN + parents are the only nodes that get updated
add_tags = pm_gate_kernel_sink+PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat({Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx[3] else None),
])

def replace_contig_with_assign(u:UOp):
  # no real contig for DISK tensors, they are left alone
  if isinstance(u._device, str) and u._device.startswith("DISK"): return u.rtag(None)
  dtype = u.dtype
  if isinstance(dtype, ImageDType):
    if prod(dtype.shape) != prod(u.max_shard_shape) or ([x for x in u.max_shard_shape if x != 1] or [1])[-1] % 4 != 0:
      if DEBUG >= 1: print(f"demoting Image {dtype} with shape {u.max_shard_shape}")
      dtype = dtype.base
  buffer = UOp.new_buffer(u.device, u.shard_size, dtype).reshape(u.max_shard_shape)
  if isinstance(u.device, tuple) and u.axis is not None: buffer = buffer.multi(u.axis)
  return buffer.assign(u.src[0]).rtag(u.tag)

def replace_assign_with_contig(u:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.ASSIGN, Ops.BITCAST}: assigned_to = assigned_to.src[0].base
  if assigned_to.op is not Ops.BUFFER:
    return u.src[1].contiguous(tag=u.tag)

pm_replace_contig_with_assign = PatternMatcher([
  # add CONTIGUOUS to tagged UOps
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag is not None else None),
  # replace ASSIGN with CONTIGUOUS
  (UPat(Ops.ASSIGN, name="u"), replace_assign_with_contig),
  # replace CONTIGUOUS with ASSIGNs
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_assign),
])

schedule_cache: dict[bytes, tuple[list[ExecItem], UOp]] = {}
@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[1]))}")
def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[dict[UOp, UOp], list[ExecItem], dict[str, int]]:
  # big_sink srcs are all the Tensors
  st = time.perf_counter()

  # uop list is a list in the original_sink graph and we can map to the tags later
  # here we build buffer map
  uop_list: list[UOp] = []
  buffer_map: dict[UOp, UOp] = {}

  dont_realize = {Ops.CONST, Ops.BUFFER, Ops.BIND, Ops.DEFINE_VAR, Ops.AFTER}
  bases = set([x.multibase for x in big_sink.src if x.base.op not in dont_realize])

  # this rewrite is "read-only", it adds simple things to buffer_map and may sink things on big_sink, bottom_up
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=(uop_list, set(), buffer_map, bases), bottom_up=True, name="number the uops")

  # here we can break the tensor graph
  big_sink = graph_rewrite(big_sink, pm_replace_contig_with_assign, name="replace contig")

  # here we construct the final buffer_map
  for s in big_sink.toposort():
    if s.tag is not None:
      assert s.op is Ops.ASSIGN
      for t in s.tag:
        original_uop = uop_list[t]
        replace_uop = s
        while replace_uop.op is Ops.ASSIGN: replace_uop = replace_uop.src[0]
        buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  big_sink = graph_rewrite(big_sink, _remove_all_tags, name="remove tags")

  #for k,v in buffer_map.items(): print(k.op, v.op)

  # replace BUFFERs with PARAMs, CONSTs UNIQUE with LUNIQUE, strip BIND values for cache key, extract var_vals
  input_buffers: dict[UOp, UOp] = {}
  var_vals: dict[str, int] = {}
  big_sink_cache = graph_rewrite(big_sink, pm_pre_sched_cache, ctx=(input_buffers, var_vals, [0], [0]), name="rewrite for sched cache")
  sched_cache_key = big_sink_cache.key

  if not SCACHE or (sc_ret:=schedule_cache.get(sched_cache_key, None)) is None:
    # verify Tensors match the spec (on big_sink, we only need to do this if cache misses)
    if SPEC: type_verify(big_sink, tensor_spec)

    # hack to preserve metadata
    graph_rewrite_map(big_sink, pm_pre_sched_cache, ctx=({}, {}, [0], [0]), name="preserve metadata")

    # tensor map is what we return
    tensor_map: dict[UOp, UOp] = {}

    if any(isinstance(x._device, tuple) for x in big_sink_cache.toposort()):
      tensor_map |= get_multi_map(big_sink_cache)
      big_sink_cache = big_sink_cache.substitute(tensor_map, name="Apply Multi Map")
      big_sink_cache = UOp.sink(*flatten([x.src if x.op is Ops.MULTI else [x] for x in big_sink_cache.src]))

    tensor_map |= get_rangeify_map(big_sink_cache)
    big_sink = big_sink_cache.substitute(tensor_map, name="Apply Kernelize Map")

    pre_schedule, buf_uops_sink = create_schedule(big_sink)
    if SCACHE: schedule_cache[sched_cache_key] = (pre_schedule, buf_uops_sink)
  else:
    # schedule cache hit
    pre_schedule, buf_uops_sink = sc_ret
  del big_sink_cache

  # replace all the PARAMs/LUNIQUEs back (single graph_rewrite for everything)
  input_buffers_inverse = {v:k for k,v in input_buffers.items()}
  buf_uops_sink = graph_rewrite(buf_uops_sink, pm_post_sched_cache, ctx=input_buffers_inverse, name="unrewrite combined")

  # add bufs to pre_schedule
  schedule: list[ExecItem] = []
  for i, si in enumerate(pre_schedule):
    buf_uops = buf_uops_sink.src[i].src
    # create subbuffers if needed
    if si.ast.op is Ops.BUFFER_VIEW:
      base = buf_uops[1].buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[buf_uops[0]] = base.view(buf_uops[0].arg, si.ast.dtype, si.ast.arg[1]*base.dtype.itemsize)
    ubufs = tuple(b.buffer for b in buf_uops)
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in si.ast.variables() if x.expr == '_device_num']
      for j, bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ExecItem(si.ast, list(bufs), si.metadata, si.fixedvars | ({dnums[0].expr:j} if len(dnums) else {})))
    else:
      # ONE -> ONE
      schedule.append(ExecItem(si.ast, list(ubufs), si.metadata, si.fixedvars))
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule):4d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {sched_cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache)} uops in cache")

  used_vars = set().union(*[{v.expr for v in si.ast.variables()} for si in schedule])
  return buffer_map, schedule, {k:v for k,v in var_vals.items() if k in used_vars}