from __future__ import annotations
import time, pprint
from typing import cast, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, track_rewrites, sym_infer
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite, graph_rewrite_map
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import Metadata, DEBUG, cpu_profile, TracingKey, SPEC, flatten, pluralize, PROFILE, GlobalCounters, ansilen, TRACEMETA
from tinygrad.helpers import time_to_str, unwrap

if TYPE_CHECKING:
  from tinygrad.engine.realize import Runner

# **** ScheduleItem return type

@dataclass
class ScheduleItem:
  ast: UOp
  bufs: list[Buffer|None] = field(default_factory=list)
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  prg: Runner|None = None

  def _lower(self) -> None:
    """Populate self.prg by lowering the AST."""
    if self.prg is not None:
      return
    from tinygrad.engine.realize import si_lowerer
    try:
      runner, new_bufs = cast(tuple, si_lowerer.rewrite(self.ast, self.bufs))
      self.prg = runner
      self.bufs = new_bufs
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {self.ast.op}")
        print("tensor operations:")
        pprint.pprint(self.metadata, indent=2)
      raise e

  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    from tinygrad.engine.realize import CompiledRunner
    from tinygrad.helpers import ProfilePointEvent, cpu_events
    if self.prg is None:
      self._lower()
    assert self.prg is not None
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    bufs = [unwrap(x) for x in self.bufs] if jit else [unwrap(x).ensure_allocated() for x in self.bufs]
    if PROFILE:
      payload = {"metadata":self.metadata, "var_vals":var_vals, "bufs":[b.trace_num for b in bufs], "name":self.prg.display_name}
      payload["outputs"], payload["inputs"] = (self.prg.p.outs, self.prg.p.ins) if isinstance(self.prg, CompiledRunner) else ([0], [1])
      cpu_events.append(ProfilePointEvent(self.prg.device, "exec", len(cpu_events), payload))
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.estimates.ops, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.estimates.mem, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.estimates.lds, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        header_color = 'magenta' if jit else ('green' if self.prg.first_run else None)
        from tinygrad.helpers import colored
        ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
        flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
        flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
        mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
          colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
          f" {self.prg.display_name+' '*(46-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
          ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
          f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}")
      self.prg.first_run = False
    return et

# **** schedule linearizer

def create_schedule(sched_sink:UOp) -> tuple[list[ScheduleItem], UOp]:
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

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque()
    for k,v in in_degree.items():
      if v == 0: queue.append(k)

    schedule: list[tuple|UOp] = []
    while len(queue):
      k = rk = queue.popleft()
      if k.op is Ops.END: k = k.src[0]
      if k.op is Ops.RANGE: schedule.append(k)
      elif k.op is Ops.KERNEL:
        ast = k.arg.ast
        buf_uops = tuple(s.buf_uop for s in k.src if s.op is not Ops.BIND)
        bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE)
        schedule.append((ast, buf_uops, k.arg.metadata, {}, bound_ranges))
        if rk.op is Ops.END: schedule.append(rk)
      else:
        raise RuntimeError(f"can't schedule {k.op}")
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)

  with cpu_profile(TracingKey("expand ranges")):
    pre_schedule: list[ScheduleItem] = []
    buf_uops_list: list[UOp] = []
    sched_ptr = 0
    in_ranges: dict[UOp, int] = {}
    range_ptrs: dict[UOp, int] = {}
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
        ast, buf_uops, metadata, fixedvars, bound_ranges = si
        fixedvars = fixedvars | {s.src[0].arg[0]:in_ranges[s.src[1]] for s in bound_ranges}
        pre_schedule.append(ScheduleItem(ast, [], metadata, fixedvars))
        buf_uops_list.append(UOp.sink(*buf_uops))
      sched_ptr += 1
  return pre_schedule, UOp.sink(*buf_uops_list)

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

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

schedule_cache: dict[bytes, tuple[list[ScheduleItem], UOp]] = {}
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

    pre_schedule, buf_uops_sink = create_schedule(big_sink)

    # save in schedule cache (include AFTERs in tensor_map so we don't need big_sink)
    after_map = [(u, u.buf_uop) for u in big_sink.toposort() if u.op is Ops.AFTER]
    tensor_map_sink = UOp.sink(*flatten([(k,v) for k,v in tensor_map.items()]), *flatten(after_map))
    combined_sink = UOp.sink(tensor_map_sink, buf_uops_sink)
    schedule_cache[sched_cache_key] = (pre_schedule, combined_sink)
  else:
    # schedule cache hit
    del big_sink_cache
    pre_schedule, combined_sink = sc_ret

  # replace all the LUNIQUEs with UNIQUEs (single graph_rewrite for everything)
  input_buffers_reverse = {v:k for k,v in input_buffers.items()}
  combined = graph_rewrite(combined_sink, pm_post_sched_cache, ctx=input_buffers_reverse, name="unrewrite combined")
  tensor_map_sink, buf_uops_sink = combined.src
  tm_src = tensor_map_sink.src
  tensor_map = {tm_src[i]:tm_src[i+1] for i in range(0, len(tm_src), 2)}

  # add bufs to pre_schedule
  schedule: list[ScheduleItem] = []
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
      dnums = [x for x in si.ast.variables() if x.arg[0] == '_device_num']
      for j, bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ScheduleItem(si.ast, list(bufs), si.metadata, si.fixedvars | ({dnums[0].expr:j} if len(dnums) else {})))
    else:
      # ONE -> ONE
      schedule.append(ScheduleItem(si.ast, list(ubufs), si.metadata, si.fixedvars))
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  # extract var_vals from BINDs that were stripped (only if there are kernels)
  var_vals: dict[str, int] = {}
  if schedule:
    for u in input_buffers:
      if u.op is Ops.BIND:
        var, val = u.unbind()
        assert var.expr not in var_vals or var_vals[var.expr] == val, f"bind mismatch on {var}, {var_vals[var.expr]} != {val}"
        var_vals[var.expr] = val

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule):4d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if sc_ret is not None else 'CACHE MISS'} {sched_cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache)} uops in cache")
  return tensor_map, schedule, var_vals
