from typing import cast
from collections import defaultdict
from tinygrad.engine.realize import ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite, buffers
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.memory import TLSFAllocator

# **************** memory planning ****************

def memory_plan_rewrite(linear:UOp, external_bufs:set[UOp]=frozenset()) -> tuple[UOp, dict[UOp, UOp]]:
  if NO_MEMORY_PLANNER: return linear, {}
  # phase 1: identify internal buffers & compute lifetimes
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  # skip buffers in BUFFER_VIEW AST operations (view ops expect BUFFER arg format)
  view_bufs:set[UOp] = set()
  for si in linear.src:
    if si.src[0].op is Ops.BUFFER_VIEW:
      for b in si.src[1:]: view_bufs.add(b)
  for i, si in enumerate(linear.src):
    for buf_uop in si.src[1:]:
      if buf_uop.op is not Ops.BUFFER: continue
      if buf_uop in external_bufs: continue  # user-visible buffer, skip
      if buf_uop in view_bufs: continue  # part of view op, skip
      if buf_uop in buffers: continue  # already realized, skip
      if isinstance(buf_uop.dtype, ImageDType): continue
      if not hasattr(Device[buf_uop.device].allocator, "_offset"): continue
      if buf_uop not in first_appearance: first_appearance[buf_uop] = i
      last_appearance[buf_uop] = i
  if not first_appearance: return linear, {}

  # phase 2: TLSF allocation per device
  min_block_size = 0x1000
  buffer_requests = sorted([((first_appearance[b], True), b) for b in first_appearance] +
                           [((last_appearance[b] + 1, False), b) for b in first_appearance], key=lambda x: x[0])
  total_memory = sum(round_up(b.arg * b.dtype.itemsize, min_block_size) for b in first_appearance) * 2
  tlsf_offsets:dict[UOp, int] = {}
  peaks:dict[str, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=min_block_size, lv2_cnt=32)))
  for (_, is_open), buf_uop in buffer_requests:
    dev = buf_uop.device
    nbytes = buf_uop.arg * buf_uop.dtype.itemsize
    if is_open: tlsf_offsets[buf_uop] = peaks[dev][1].alloc(round_up(nbytes, min_block_size))
    else: peaks[dev][1].free(tlsf_offsets[buf_uop])
    peaks[dev] = (max(peaks[dev][0], tlsf_offsets[buf_uop] + nbytes), peaks[dev][1])

  # phase 3: build replace_map with BUFFER_VIEW into shared arenas
  arenas = {dev: UOp.new_buffer(dev, round_up(peak, min_block_size), dtypes.int8) for dev, (peak, _) in peaks.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in tlsf_offsets.items():
    # offset is in bytes, BUFFER_VIEW arg[1] is in elements
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[buf_uop.device],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1:
    omem = sum(round_up(b.arg * b.dtype.itemsize, min_block_size) for b in first_appearance) / 1e6
    nmem = sum(round_up(peak, min_block_size) for peak, _ in peaks.values()) / 1e6
    if omem != nmem: print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  # phase 4: apply via graph_rewrite
  pm_memory_plan = PatternMatcher([(UPat(Ops.BUFFER, name="b"), lambda ctx, b: ctx.get(b))])
  return graph_rewrite(linear, pm_memory_plan, ctx=replace_map, walk=True, name="memory plan"), replace_map

def _internal_memory_planner(buffers:list[list[Buffer]], noopt_buffers=None, ignore_checks=False, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance, buf_to_opt = {}, {}, set()
  for i,u in enumerate(buffers):
    for buf in u:
      should_skip = buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers)
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
  buffer_replace:dict[Buffer, tuple[Buffer|None, int|None]] = {}
  reuse_buffers:dict[tuple, list[Buffer]] = defaultdict(list)
  global_planner:dict[str, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=min_block_size, lv2_cnt=32)))
  for (_, is_open_ev), buf in buffer_requests:
    # Check if suballocation is possible for the given buffer and device.
    if hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType):
      if is_open_ev: buffer_replace[buf] = (None, global_planner[buf.device][1].alloc(round_up(buf.nbytes, 0x1000)))
      else: global_planner[buf.device][1].free(cast(int, buffer_replace[buf][1]))
      global_planner[buf.device] = (max(global_planner[buf.device][0], buffer_replace[buf][1] + buf.nbytes), global_planner[buf.device][1])
    else:
      key = (buf.device, buf.dtype, buf.options, buf.nbytes)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, None)
      else: reuse_buffers[key].append(cast(Buffer, buffer_replace[buf][0]))

  # Allocate global buffers based on the memory planner.
  global_buffers = {dev: Buffer(dev, round_up(sz, 0x1000), dtypes.int8) for dev, (sz, _) in global_planner.items()}
  buffer_resolve:dict[Buffer, tuple[Buffer, int|None]] = {buf: (base or global_buffers[buf.device], off) for buf,(base,off) in buffer_replace.items()}

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

