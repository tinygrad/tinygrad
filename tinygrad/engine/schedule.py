from typing import cast
from dataclasses import dataclass, field, replace
from collections import deque, defaultdict
from tinygrad.uop.ops import UOp, Ops, buffers
from tinygrad.device import Device, Buffer, MultiBuffer
from tinygrad.helpers import Metadata, all_same

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  bound_ranges: tuple[UOp, ...] = ()

# **** schedule linearizer

def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[str, int]]:
  # construct the KERNEL children graph based on assigns
  children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree: dict[UOp, int] = {}
  var_vals: dict[str, int] = {}
  for u in sched_sink.toposort():
    if u.op is not Ops.AFTER: continue  # anything that's not an ASSIGN doesn't write a kernel, so we can skip
    k = u.src[1]
    in_degree.setdefault(k, 0)
    if k.op is Ops.RANGE: continue
    for s in k.src[0].src if k.op is Ops.END else k.src:
      if s.op is Ops.AFTER:
        children[s.src[1]].append(k)
        in_degree[k] += 1
      elif s.op in {Ops.MSELECT, Ops.MSTACK}:
        for ss in s.src:
          if ss.op is Ops.MSELECT: ss = ss.src[0]
          if ss.op is not Ops.BUFFER:
            assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
            children[ss.src[1]].append(k)
            in_degree[k] += 1
      elif s.op is Ops.BUFFER:
        pass  # a BUFFER is already realized, nothing to do here
      elif s.op is Ops.BIND:
        # for RANGE this is in fixedvars
        if s.src[1].op is not Ops.RANGE:
          var, val = s.unbind()
          assert var.expr not in var_vals or var_vals[var.expr] == val, f"bind mismatch on {var}, {var_vals[var.expr]} != {val}"
          var_vals[var.expr] = val
      else:
        raise RuntimeError(f"input to kernel must be AFTER or BUFFER, not {s.op}")

  # linearize KERNEL UOps into ScheduleItems in BFS order

  def _heuristic(k: UOp):
    if k.op is Ops.KERNEL and k.arg.ast.op is Ops.COPY and not all_same([Device[cast(Buffer, s.buf_uop.buffer).device].group_id for s in k.src]):
      return 1000
    return 0

  last_heuristic: int = 0
  queues: defaultdict[int, deque[UOp]] = defaultdict(deque)
  last_queue: deque[UOp] = deque()
  for k,v in in_degree.items():
    if v == 0: queues[_heuristic(k)].append(k)

  schedule: list[ScheduleItem|UOp] = []
  while last_queue or any(queues.values()):
    if not last_queue: last_heuristic, last_queue = min((it for it in queues.items() if it[1]), key=lambda x: abs(x[0]-last_heuristic))
    k = rk = last_queue.popleft()
    if k.op is Ops.END: k = k.src[0]
    if k.op is Ops.RANGE: schedule.append(k)
    elif k.op is Ops.KERNEL:
      ast = k.arg.ast
      # create subbuffers if needed
      if ast.op is Ops.BUFFER_VIEW:
        base = k.src[1].buf_uop.buffer
        assert isinstance(base, Buffer), "base can't be MultiBuffer"
        buffers[k.src[0]] = base.view(k.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
      ubufs = tuple(s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND)
      bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and s.src[1].op is Ops.RANGE)
      if any(isinstance(x, MultiBuffer) for x in ubufs):
        assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
        dnums = [x for x in ast.variables() if x.arg[0] == '_device_num']
        for i,bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
          schedule.append(ScheduleItem(ast, bufs, k.arg.metadata, {dnums[0].expr:i} if len(dnums) else {}, bound_ranges=bound_ranges))
      else:
        # ONE -> ONE
        schedule.append(ScheduleItem(ast, cast(tuple[Buffer, ...], ubufs), k.arg.metadata, bound_ranges=bound_ranges))
      if rk.op is Ops.END: schedule.append(rk)
    else:
      raise RuntimeError(f"can't schedule {k.op}")
    for x in children[k]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queues[_heuristic(x)].append(x)

  # expand the ranges in the schedule
  real_schedule: list[ScheduleItem] = []
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
  return real_schedule, var_vals
