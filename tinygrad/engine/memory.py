from collections import defaultdict
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.ops import Ops
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.allocator import TLSFAllocator

# **************** memory planning ****************

def _internal_memory_planner(buffers:list[list[Buffer]|tuple[Buffer, ...]], noopt_buffers=None, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance = {}, {}
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i

  # Sort buffers operations in timeline order. 2 events: buffer is allocted and buffer is freed.
  buffer_requests = sorted([((first_appearance[buf], True), buf) for buf in first_appearance.keys()] + \
                           [((last_appearance[buf] + 1, False), buf) for buf in first_appearance.keys()], key=lambda x: x[0])

  buffer_replace = {}
  reuse_buffers:dict[tuple[str], list[Buffer]] = defaultdict(list)
  global_planner:dict[tuple[str], tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(256 << 30, block_size=0x100, lv2_cnt=32)))
  for (time, is_open_ev), buf in buffer_requests:
    if can_suballoc:=hasattr(Device[buf.device].allocator, "offset") and not isinstance(buf.dtype, ImageDType):
      if is_open_ev: buffer_replace[buf] = (None, global_planner[buf.device][1].alloc(buf.nbytes))
      else: global_planner[buf.device][1].free(buffer_replace[buf][1])
      global_planner[buf.device][0] = (max(global_planner[buf.device][0], [buffer_replace[buf][1] + buf.nbytes]), global_planner[buf.device][1])
    else:
      key = (buf.device, buf.dtype, buf.options, buf.nbytes)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), 0) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, 0)
      else: reuse_buffers[key].append((buffer_replace[buf][0], None))

  global_buffers = {dev: Buffer(dev, round_up(sz, 0x1000), dtypes.int8) for dev, (sz, _) in global_planner.items()}
  buffer_replace = {dev: (global_buffers[buf.device] if buf is None else buf, offset) for dev, (sz, buf) in global_planner.items()}

  # assign all known root buffers first
  assigned = {buf: Buffer(buf.device, buf.size, buf.dtype, base=base, offset=off) for buf, (base, offset) in buffer_replace.items() if buf != base}

  # and now subbuffers
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers) or buf._base is None: continue
      assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=(pbuf:=assigned.get(buf.base, buf.base)).base, offset=pbuf.offset+buf.offset)

  if DEBUG >= 1 and len(ak:=dedup(x for x in assigned.keys() if x._base is None)) != len(av:=dedup(x for x in assigned.values() if x._base is None)):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av+list(global_buffers.values())])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")

  return assigned

def memory_planner(schedule:list[ScheduleItem]) -> list[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([si.bufs for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not Ops.SINK for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata) for si in schedule]
