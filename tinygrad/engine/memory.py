from collections import defaultdict
from tinygrad.device import Device
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops, buffers
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  """Recursively collect BUFFER UOps, following through MSELECT/MSTACK for multi-device."""
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp) -> bool:
  return isinstance(b.device, str) and not b.device.startswith(("DISK", "TINYFS")) \
    and not isinstance(b.dtype, ImageDType) and hasattr(Device[b.device].allocator, "_offset")

def memory_plan_rewrite(linear:UOp, external_bufs:set[UOp]=frozenset()) -> UOp:
  if NO_MEMORY_PLANNER: return linear

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  for i, si in enumerate(linear.src):
    for b in [b for src in si.src[1:] for b in _collect_bufs(src) if b not in external_bufs and b not in buffers and _can_plan(b)]:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
  if not first_appearance: return linear

  # suballocation: build sorted open/close events, then alloc/free in order
  block_size = 32
  nbytes = {b: round_up(b.arg * b.dtype.itemsize, block_size) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1, False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2

  offsets:dict[UOp, int] = {}
  peaks:dict[str, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    if is_open: offsets[buf] = peaks[buf.device][1].alloc(nbytes[buf])
    else: peaks[buf.device][1].free(offsets[buf])
    peaks[buf.device] = (max(peaks[buf.device][0], offsets[buf] + buf.arg * buf.dtype.itemsize), peaks[buf.device][1])
  arena_sizes = {dev: round_up(peak, block_size) for dev, (peak, _) in peaks.items()}

  # build replace_map: each buffer becomes a BUFFER_VIEW into a shared per-device arena
  arenas = {dev: UOp.new_buffer(dev, sz, dtypes.int8) for dev, sz in arena_sizes.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[buf_uop.device],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1 and (omem:=sum(nbytes.values()) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)
