from collections import defaultdict
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite, buffers
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.memory import TLSFAllocator

# **************** memory planning ****************

MIN_BLOCK_SIZE = 0x1000

def _memory_plan(allocs:dict) -> tuple[dict, dict[str, int]]:
  """TLSF suballocation given {buf: (first_step, last_step, device, nbytes)}. Returns (offsets {buf: byte_offset}, arena_sizes {dev: bytes})."""
  buffer_requests = sorted([((f, True), b) for b, (f, _, _, _) in allocs.items()] +
                           [((l + 1, False), b) for b, (_, l, _, _) in allocs.items()], key=lambda x: x[0])
  total_memory = sum(round_up(nb, MIN_BLOCK_SIZE) for _, _, _, nb in allocs.values()) * 2

  offsets:dict = {}
  peaks:dict[str, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=MIN_BLOCK_SIZE, lv2_cnt=32)))
  for (_, is_open), buf in buffer_requests:
    _, _, dev, nb = allocs[buf]
    if is_open: offsets[buf] = peaks[dev][1].alloc(round_up(nb, MIN_BLOCK_SIZE))
    else: peaks[dev][1].free(offsets[buf])
    peaks[dev] = (max(peaks[dev][0], offsets[buf] + nb), peaks[dev][1])

  arena_sizes = {dev: round_up(peak, MIN_BLOCK_SIZE) for dev, (peak, _) in peaks.items()}
  return offsets, arena_sizes

# **************** UOp memory plan rewrite ****************

def _collect_bufs(u:UOp) -> list[UOp]:
  """Recursively collect BUFFER UOps, following through MSELECT/MSTACK for multi-device."""
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp) -> bool:
  return isinstance(b.device, str) and not isinstance(b.dtype, ImageDType) and hasattr(Device[b.device].allocator, "_offset")

def memory_plan_rewrite(linear:UOp, external_bufs:set[UOp]=frozenset()) -> tuple[UOp, dict[UOp, UOp]]:
  if NO_MEMORY_PLANNER: return linear, {}

  # buffers we must not touch: externally visible, already realized, or part of a BUFFER_VIEW op
  skip = external_bufs | {b for si in linear.src if si.src[0].op is Ops.BUFFER_VIEW for b in si.src[1:]}

  # compute lifetimes for all plannable internal buffers
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  for i, si in enumerate(linear.src):
    for b in [b for src in si.src[1:] for b in _collect_bufs(src) if b not in skip and b not in buffers and _can_plan(b)]:
      if b not in first_appearance: first_appearance[b] = i
      last_appearance[b] = i
  if not first_appearance: return linear, {}

  # allocate
  allocs = {b: (first_appearance[b], last_appearance[b], b.device, b.arg * b.dtype.itemsize) for b in first_appearance}
  offsets, arena_sizes = _memory_plan(allocs)

  # build replace_map: each buffer becomes a BUFFER_VIEW into a shared per-device arena
  arenas = {dev: UOp.new_buffer(dev, sz, dtypes.int8) for dev, sz in arena_sizes.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[buf_uop.device],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1:
    omem = sum(round_up(nb, MIN_BLOCK_SIZE) for _, _, _, nb in allocs.values()) / 1e6
    nmem = sum(arena_sizes.values()) / 1e6
    if omem != nmem: print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  # apply
  pm_memory_plan = PatternMatcher([(UPat(Ops.BUFFER, name="b"), lambda ctx, b: ctx.get(b))])
  return graph_rewrite(linear, pm_memory_plan, ctx=replace_map, walk=True, name="memory plan"), replace_map

# **************** Buffer memory planner ****************

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

  # split into suballocatable (TLSF) vs full-buffer reuse (images, etc)
  def _can_suballoc(buf:Buffer) -> bool: return hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType)
  suballoc = {b: (first_appearance[b], last_appearance[b], b.device, b.nbytes) for b in first_appearance if _can_suballoc(b)}
  reuse_bufs = [b for b in first_appearance if not _can_suballoc(b)]

  # TLSF suballocation
  offsets, arena_sizes = _memory_plan(suballoc)
  global_buffers = {dev: Buffer(dev, sz, dtypes.int8) for dev, sz in arena_sizes.items()}

  # full-buffer reuse for non-suballocatable buffers
  reuse_pool:dict[tuple, list[Buffer]] = defaultdict(list)
  reuse_requests = sorted([((first_appearance[b], True), b) for b in reuse_bufs] +
                          [((last_appearance[b] + 1, False), b) for b in reuse_bufs], key=lambda x: x[0])
  reuse_map:dict[Buffer, Buffer] = {}
  for (_, is_open), buf in reuse_requests:
    key = (buf.device, buf.dtype, buf.options, buf.nbytes)
    if is_open: reuse_map[buf] = reuse_pool[key].pop() if reuse_pool[key] else buf
    else: reuse_pool[key].append(reuse_map[buf])

  # assign base buffers
  assigned:dict[Buffer, Buffer] = {}
  for buf in first_appearance:
    if buf in offsets:
      assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=global_buffers[buf.device], offset=offsets[buf])
    elif buf in reuse_map and reuse_map[buf] is not buf:
      assigned[buf] = reuse_map[buf]

  # assign sub-buffers
  for buf in buf_to_opt:
    if buf._base is not None:
      assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=(pbuf:=assigned.get(buf.base, buf.base)).base, offset=pbuf.offset+buf.offset)

  if DEBUG >= 1:
    ak, av = dedup(x for x in assigned.keys() if x._base is None),dedup(x for x in assigned.values() if x._base is None)+list(global_buffers.values())
    omem, nmem = sum([x.nbytes for x in ak])/1e6, sum([x.nbytes for x in av])/1e6
    if omem != nmem: print(f"{debug_prefix}memory reduced from {omem:.2f} MB -> {nmem:.2f} MB,", f"{len(ak)} -> {len(av)} bufs")

  return assigned
