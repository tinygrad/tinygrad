from typing import cast
from collections import defaultdict
from tinygrad.engine.realize import ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.uop.ops import Ops
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.memory import TLSFAllocator

# **************** memory planning ****************

def _internal_memory_planner(buffers:list[list[Buffer]], copy_buffers=None, ignore_checks=False, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance, buf_to_opt = {}, {}, set()
  for i,u in enumerate(buffers):
    for buf in u:
      should_skip = buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0
      if not ignore_checks and should_skip: continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i
      buf_to_opt.add(buf)

  # Sort buffer operations in timeline order. Two events: buffer is allocated or buffer is freed.
  buffer_requests = sorted([((first_appearance[buf], True), buf) for buf in first_appearance.keys()] + \
                           [((last_appearance[buf] + 1, False), buf) for buf in first_appearance.keys()], key=lambda x: x[0])
  total_memory = sum(round_up(buf.nbytes, min_block_size:=0x1000) for buf in first_appearance.keys()) * 2 # *2 for fragmentation (which is about 15%)

  # Try to suballocate from a shared buffer managed by global_planner using TLSFAllocator.
  # Also track buffer replacements for buffers that do not support suballocation.
  # Copy buffers are optimized in a separate lane (keyed by is_copy) to preserve exec/copy parallelism.
  buffer_replace:dict[Buffer, tuple[Buffer|None, int|None]] = {}
  reuse_buffers:dict[tuple, list[Buffer]] = defaultdict(list)
  global_planner:dict[tuple[str,bool], tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=min_block_size, lv2_cnt=32)))
  buf_gkey:dict[Buffer, tuple[str, bool]] = {}
  deferred_copy_frees:dict[tuple[str,bool], list[int]] = defaultdict(list)
  for (_, is_open_ev), buf in buffer_requests:
    is_copy = copy_buffers is not None and buf in copy_buffers
    gk = (buf.device, is_copy)
    # Check if suballocation is possible for the given buffer and device.
    if hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType):
      if is_open_ev:
        # Defer copy dest frees by ~2 copy allocs to avoid copy→compute→copy serialization in graphs.
        if is_copy:
          while len(deferred_copy_frees[gk]) > 8: global_planner[gk][1].free(deferred_copy_frees[gk].pop(0))
        buffer_replace[buf] = (None, global_planner[gk][1].alloc(round_up(buf.nbytes, 0x1000)))
      elif is_copy: deferred_copy_frees[gk].append(cast(int, buffer_replace[buf][1]))
      else: global_planner[gk][1].free(cast(int, buffer_replace[buf][1]))
      global_planner[gk] = (max(global_planner[gk][0], buffer_replace[buf][1] + buf.nbytes), global_planner[gk][1])
      buf_gkey[buf] = gk
    else:
      key = (buf.device, buf.dtype, buf.options, buf.nbytes, is_copy)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, None)
      elif not is_copy: reuse_buffers[key].append(cast(Buffer, buffer_replace[buf][0]))

  # Allocate global buffers based on the memory planner.
  global_buffers = {key: Buffer(key[0], round_up(sz, 0x1000), dtypes.int8) for key, (sz, _) in global_planner.items()}
  buffer_resolve:dict[Buffer, tuple[Buffer, int|None]] = {buf: (base or global_buffers[buf_gkey[buf]], off) for buf,(base,off) in buffer_replace.items()}

  # Assign buffers. First, assign full buffers (not sub-buffers).
  assigned:dict[Buffer, Buffer] = {}
  for buf, (base, off) in buffer_resolve.items():
    if buf != base:
      assigned[buf] = base if off is None else Buffer(buf.device, buf.size, buf.dtype, base=base, offset=off)

  # Now assign sub-buffers.
  for buf in buf_to_opt:
    if buf._base is not None:
      assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=(pbuf:=assigned.get(buf.base, buf.base)).base, offset=pbuf.offset+buf.offset)

  if DEBUG >= 1:
    ak, av = dedup(x for x in assigned.keys() if x._base is None),dedup(x for x in assigned.values() if x._base is None)+list(global_buffers.values())
    omem, nmem = sum([x.nbytes for x in ak])/1e6, sum([x.nbytes for x in av])/1e6
    if omem != nmem: print(f"{debug_prefix}memory reduced from {omem:.2f} MB -> {nmem:.2f} MB,", f"{len(ak)} -> {len(av)} bufs")

  return assigned

def memory_planner(schedule:list[ExecItem]) -> list[ExecItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([[b for b in si.bufs if b is not None] for si in schedule],
                                      copy_buffers={b for si in schedule if si.ast.op is not Ops.SINK for b in si.bufs if b is not None})
  return [ExecItem(si.ast, [assigned.get(x, x) if x is not None else None for x in si.bufs], si.metadata, si.fixedvars) for si in schedule]
