from collections import defaultdict
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up, to_tuple
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  # CL and WEBGPU do not support views, see explanation in contiguous_view_offset
  return all(not d.startswith(("DISK", "CL", "WEBGPU")) for d in to_tuple(b.device))

LaneKey = tuple[str, int]
LARGE_BUFFER = 1 << 20  # buffers placed before the others (see memory_plan_rewrite)

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None, arenas:dict[LaneKey, UOp]|None=None) -> UOp:
  # arenas: the arena of every lane, shared with other linears that never run at the same time (e.g. the jits of a model): an arena that is big
  # enough is reused, a bigger one replaces it
  if NO_MEMORY_PLANNER: return linear
  if held_bufs is None: held_bufs = set()

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  copy_bufs: set[UOp] = set()
  for i, si in enumerate(linear.src):
    si_bufs = [b for src in si.src[1:] for b in _collect_bufs(src) if _can_plan(b, held_bufs)]
    for b in si_bufs:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
    if si.src[0].op is Ops.STORE: copy_bufs.update(si_bufs)
  if not first_appearance: return linear

  # separate copy and compute buffers into different lanes to avoid introducing dependencies (copy->compute->copy)
  def _key(b:UOp): return (b.device, 1 if b in copy_bufs else 0)
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}

  # suballocation: build sorted open/close events, then alloc/free in order
  block_size = 256
  nbytes = {b: round_up(b.max_numel() * b.dtype.itemsize, block_size) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1 + buf_hold.get(b, 0), False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2

  # the large buffers are placed first, knowing all the lifetimes: the largest first, each at the lowest offset free for its lifetime (an online
  # allocator fragments the arena: its size can be twice the memory live at once). the many small ones are suballocated after them
  start = {b: first_appearance[b] for b in first_appearance}
  end = {b: last_appearance[b] + 1 + buf_hold.get(b, 0) for b in first_appearance}
  offsets:dict[UOp, int] = {}
  large_peak:dict[LaneKey, int] = defaultdict(int)
  placed:dict[LaneKey, list[tuple[int, int, int, int]]] = defaultdict(list)
  for buf in sorted((b for b in first_appearance if nbytes[b] >= LARGE_BUFFER), key=lambda b: (-nbytes[b], start[b])):
    off = 0
    for lo, hi in sorted((o, o+n) for s, e, o, n in placed[_key(buf)] if s < end[buf] and start[buf] < e):
      if off + nbytes[buf] <= lo: break
      off = max(off, hi)
    placed[_key(buf)].append((start[buf], end[buf], off, nbytes[buf]))
    offsets[buf], large_peak[_key(buf)] = off, max(large_peak[_key(buf)], off + nbytes[buf])
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    if buf in offsets and nbytes[buf] >= LARGE_BUFFER: continue
    if is_open: offsets[buf] = peaks[_key(buf)][1].alloc(nbytes[buf]) + large_peak[_key(buf)]
    else: peaks[_key(buf)][1].free(offsets[buf] - large_peak[_key(buf)])
    peaks[_key(buf)] = (max(peaks[_key(buf)][0], offsets[buf] + buf.max_numel() * buf.dtype.itemsize), peaks[_key(buf)][1])
  arena_sizes = {key: round_up(max(peaks[key][0] if key in peaks else 0, large_peak[key]), block_size)
                 for key in set(large_peak) | set(peaks)}

  # build replace_map: each buffer becomes a SHRINK/BITCAST into a shared per-device-lane arena
  if arenas is None: arenas = {}
  for key, sz in arena_sizes.items():
    if key not in arenas or arenas[key].max_numel() < sz: arenas[key] = UOp.new_buffer(key[0], sz, dtypes.int8)
  replace_map = {buf_uop:arenas[_key(buf_uop)][offset:offset+buf_uop.nbytes()].bitcast(buf_uop.dtype) for buf_uop, offset in offsets.items()}

  if DEBUG >= 1 and (omem:=sum(nbytes.values()) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)
