from typing import cast
from collections import defaultdict
from tinygrad.engine.realize import ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops, buffers
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _get_external_bufs() -> set[UOp]:
  from tinygrad.tensor import all_tensors
  return {n for tref in list(all_tensors) if (t:=tref()) is not None for n in t.uop.toposort() if n.op is Ops.BUFFER}

def _can_plan(b:UOp, external_bufs:set[UOp]) -> bool:
  if b in external_bufs or b in buffers: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  return all(not d.startswith(("DISK", "TINYFS")) and hasattr(Device[d].allocator, "_offset") for d in devs)

def memory_plan_rewrite(linear:UOp) -> UOp:
  if NO_MEMORY_PLANNER: return linear
  external_bufs = _get_external_bufs()

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  copy_bufs: set[UOp] = set()
  for i, si in enumerate(linear.src):
    si_bufs = [b for src in si.src[1:] for b in _collect_bufs(src) if _can_plan(b, external_bufs)]
    for b in si_bufs:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
    if si.src[0].op is Ops.COPY: copy_bufs.update(si_bufs)
  if not first_appearance: return linear

  # separate copy and compute buffers into different lanes to avoid introducing dependencies (copy->compute->copy)
  def _key(b:UOp): return (b.device, 1 if b in copy_bufs else 0)
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}

  # suballocation: build sorted open/close events, then alloc/free in order
  block_size = 128 # ~cache-line
  nbytes = {b: round_up(b.arg * b.dtype.itemsize, block_size) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1 + buf_hold.get(b, 0), False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2

  offsets:dict[UOp, int] = {}
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    if is_open: offsets[buf] = peaks[_key(buf)][1].alloc(nbytes[buf])
    else: peaks[_key(buf)][1].free(offsets[buf])
    peaks[_key(buf)] = (max(peaks[_key(buf)][0], offsets[buf] + buf.arg * buf.dtype.itemsize), peaks[_key(buf)][1])
  arena_sizes = {key: round_up(peak, block_size) for key, (peak, _) in peaks.items()}

  # build replace_map: each buffer becomes a BUFFER_VIEW into a shared per-device-lane arena
  arenas = {key: UOp.new_buffer(key[0], sz, dtypes.int8) for key, sz in arena_sizes.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[_key(buf_uop)],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1 and (omem:=sum(nbytes.values()) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)

# **************** Buffer-level memory planning (used by replan_buffers_memory_layout) ****************

LaneKey = tuple[str, int]

def _internal_memory_planner(buffers:list[list[Buffer]], copies:list[tuple[Buffer, Buffer]]|None=None,
                             ignore_checks=False, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance, buf_to_opt = {}, {}, set()
  for i,u in enumerate(buffers):
    for buf in u:
      if not ignore_checks and (buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i
      buf_to_opt.add(buf)

  # Separate copy and compute buffers into different lanes and defer cross-queue frees to avoid introducing dependencies (copy->compute->copy)
  copy_dsts, copy_srcs = ({dst.base for dst,_ in copies}, {src.base for _,src in copies}) if copies else (set(), set())
  def _key(buf) -> LaneKey: return (buf.device, 1 if buf in copy_dsts or buf in copy_srcs else 0)
  buf_hold = {buf: last_appearance[buf] - first_appearance[buf] + 1 for buf in first_appearance if buf in copy_dsts or buf in copy_srcs}

  # Sort buffer operations in timeline order. Two events: buffer is allocated or buffer is freed.
  buffer_requests = sorted([((first_appearance[buf], True), buf) for buf in first_appearance.keys()] + \
    [((last_appearance[buf] + 1 + buf_hold.get(buf, 0), False), buf) for buf in first_appearance.keys()], key=lambda x: x[0])
  total_memory = sum(round_up(buf.nbytes, BLK:=0x1000) for buf in first_appearance.keys()) * 2 # *2 for fragmentation (which is about 15%)

  # Try to suballocate from a shared buffer managed by global_planner using TLSFAllocator.
  # Also track buffer replacements for buffers that do not support suballocation.
  buffer_replace:dict[Buffer, tuple[Buffer|None, int|None]] = {}
  reuse_buffers:dict[tuple, list[Buffer]] = defaultdict(list)
  global_planner:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=BLK, lv2_cnt=32)))
  for (_, is_open_ev), buf in buffer_requests:
    # Check if suballocation is possible for the given buffer and device.
    if hasattr(Device[buf.device].allocator, "_offset"):
      if is_open_ev: buffer_replace[buf] = (None, global_planner[_key(buf)][1].alloc(round_up(buf.nbytes, BLK)))
      else: global_planner[_key(buf)][1].free(cast(int, buffer_replace[buf][1]))
      global_planner[_key(buf)] = (max(global_planner[_key(buf)][0], buffer_replace[buf][1] + buf.nbytes), global_planner[_key(buf)][1])
    else:
      key = (_key(buf), buf.dtype, buf.options, buf.nbytes)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, None)
      else: reuse_buffers[key].append(cast(Buffer, buffer_replace[buf][0]))

  # Allocate global buffers based on the memory planner.
  global_buffers = {key: Buffer(key[0], round_up(sz, BLK), dtypes.int8) for key, (sz, _) in global_planner.items()}
  buffer_resolve:dict[Buffer, tuple[Buffer, int|None]] = {buf: (base or global_buffers[_key(buf)], off) for buf,(base,off) in buffer_replace.items()}

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
