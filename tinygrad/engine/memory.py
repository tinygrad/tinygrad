from collections import defaultdict
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up, getenv
from tinygrad.ops import Ops
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.allocator import TLSFAllocator

# **************** memory planning ****************

def _internal_memory_planner(buffers:list[list[Buffer]|tuple[Buffer, ...]], noopt_buffers=None, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance, buf_to_opt = {}, {}, set()
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.base.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i
      buf_to_opt.add(buf)

  # Sort buffers operations in timeline order. 2 events: buffer is allocted and buffer is freed.
  buffer_requests = sorted([((first_appearance[buf], True), buf) for buf in first_appearance.keys()] + \
                           [((last_appearance[buf] + 1, False), buf) for buf in first_appearance.keys()], key=lambda x: x[0])

  buffer_replace = {}
  reuse_buffers:dict[tuple[str], list[Buffer]] = defaultdict(list)
  global_planner:dict[tuple[str], tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(256 << 30, block_size=0x200, lv2_cnt=32)))
  for (time, is_open_ev), buf in buffer_requests:
    # Check if can suballoc.
    if hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType):
      if is_open_ev: buffer_replace[buf] = (None, global_planner[buf.device][1].alloc(buf.nbytes))
      else: global_planner[buf.device][1].free(buffer_replace[buf][1])
      global_planner[buf.device] = (max(global_planner[buf.device][0], buffer_replace[buf][1] + buf.nbytes), global_planner[buf.device][1])
    else:
      key = (buf.device, buf.dtype, buf.options, buf.nbytes)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, None)
      else: reuse_buffers[key].append((buffer_replace[buf][0], None))

  assigned = {}
  global_buffers = {dev: Buffer(dev, round_up(sz, 0x1000), dtypes.int8) for dev, (sz, _) in global_planner.items()}
  buffer_replace = {buf: (base or global_buffers[buf.device], off) for buf, (base, off) in buffer_replace.items()}
  for buf, (base, off) in buffer_replace.items():
    if buf == base: continue
    if off is None: assigned[buf] = base
    else: assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=base, offset=off)

  # and now subbuffers
  for buf in buf_to_opt:
    if buf._base is None: continue
    assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=(pbuf:=assigned.get(buf.base, buf.base)).base, offset=pbuf.offset+buf.offset)

  if getenv("VALIDATE_MEMORY_PLANNER", 0):
    taken_parts = set()
    for i,u in enumerate(buffers):
      for buf in u:
        if buf.is_allocated() or buf.base.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
        cur, base = assigned.get(buf, buf), assigned.get(buf.base, buf.base)
        if buf._base is not None:
          assert cur.base == base.base and cur.offset == buf.offset + base.offset, f"failed: {buf} {cur} {base} {buf.offset} {base.offset}"
        else:
          for part in taken_parts:
            assert buf.base == part[3] or part[0] != cur.base or part[1] + part[2] <= cur.offset or part[1] >= cur.offset + buf.nbytes, f"failed: {buf} {cur} {part}"
          if first_appearance[buf.base] == i: taken_parts.add((cur.base, cur.offset, buf.nbytes, buf.base))
          if last_appearance[buf.base] == i: taken_parts.remove((cur.base, cur.offset, buf.nbytes, buf.base))

  if DEBUG >= 1:
    ak=dedup(x for x in assigned.keys() if x._base is None)
    av=dedup(x for x in assigned.values() if x._base is None)
    print(f"{debug_prefix} memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB ->",
          f"{sum([x.nbytes for x in av+list(global_buffers.values())])/1e6:.2f} MB,", f"{len(ak)} -> {len(av)} bufs")

  return assigned

def memory_planner(schedule:list[ScheduleItem]) -> list[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([si.bufs for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not Ops.SINK for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata) for si in schedule]
