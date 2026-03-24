from collections import defaultdict
from tinygrad.device import Device
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up, dedup
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.runtime.support.memory import TLSFAllocator

def _plan_memory(lifetimes:dict, block_size:int=256, lv2_cnt:int=32) -> tuple[dict[UOp, int], dict]:
  nbytes = {b: round_up(lt[0], block_size) for b, lt in lifetimes.items()}

  # sort by (position, is_open) — False < True ensures frees before allocs at same position
  events = sorted([(lt[1], True, b, lt[3]) for b, lt in lifetimes.items()] +
                  [(lt[2], False, b, lt[3]) for b, lt in lifetimes.items()], key=lambda x: (x[0], x[1]))

  offsets:dict[UOp, int] = {}
  peaks:dict = defaultdict(lambda: (0, TLSFAllocator(sum(nbytes.values()) * 2, block_size=block_size, lv2_cnt=lv2_cnt)))
  for _, is_open, buf, key in events:
    if is_open:
      offsets[buf] = peaks[key][1].alloc(nbytes[buf])
      peak, alloc = peaks[key]
      peaks[key] = (max(peak, offsets[buf] + lifetimes[buf][0]), alloc)
    else: peaks[key][1].free(offsets[buf])
  return offsets, {key: round_up(peak, block_size) for key, (peak, _) in peaks.items()}

# ** global memory planner **

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  return all(not d.startswith(("DISK", "TINYFS")) and hasattr(Device[d].allocator, "_offset") for d in devs)

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None) -> UOp:
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
    if si.src[0].op is Ops.COPY: copy_bufs.update(si_bufs)
  if not first_appearance: return linear

  def _key(b:UOp): return (b.device, 1 if b in copy_bufs else 0)
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}

  lifetimes = {b: (b.arg * b.dtype.itemsize, first_appearance[b], last_appearance[b] + 1 + buf_hold.get(b, 0), _key(b)) for b in first_appearance}
  offsets, arena_sizes = _plan_memory(lifetimes, block_size=256)

  arenas = {key: UOp.new_buffer(key[0], sz, dtypes.int8) for key, sz in arena_sizes.items()}
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    assert offset % buf_uop.dtype.itemsize == 0, f"offset {offset} not aligned to {buf_uop.dtype.itemsize}"
    replace_map[buf_uop] = UOp(Ops.BUFFER_VIEW, buf_uop.dtype, (arenas[lifetimes[buf_uop][3]],), (buf_uop.arg, offset // buf_uop.dtype.itemsize))

  if DEBUG >= 1 and (omem:=sum(lifetimes[b][0] for b in lifetimes) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)

# ** local (shared) memory planner **

def local_memory_plan(lin:UOp) -> UOp|None:
  if NO_MEMORY_PLANNER: return None

  # build lifetimes (nbytes, first_chunk, last_chunk+1, key=0(ignored)) per DEFINE_LOCAL
  chunk:int = 0
  lifetimes:dict[UOp, list] = {}
  for u in lin.src:
    if u.op is Ops.BARRIER: chunk += 1
    if u.op in (Ops.STORE, Ops.LOAD) and (dl:=u.buf_uop).op is Ops.DEFINE_LOCAL:
      lifetimes.setdefault(dl, [dl.ptrdtype.nbytes(), chunk, 0, 0])[2] = chunk + 1
  plannable = [u for u in lin.src if u.op is Ops.DEFINE_LOCAL and u in lifetimes]

  offsets, arena_sizes = _plan_memory({dl: lifetimes[dl] for dl in plannable}, block_size=16, lv2_cnt=16)

  if (arena_size:=next(iter(arena_sizes.values()), 0)) >= sum(lt[0] for lt in lifetimes.values()): return None
  if DEBUG >= 1: print(f"local memory reduced from {sum(lt[0] for lt in lifetimes.values())} -> {arena_size} bytes.")

  # create merged DEFINE_LOCAL
  merged = UOp(Ops.DEFINE_LOCAL, dtypes.int8.ptr(size=arena_size, addrspace=AddrSpace.LOCAL), arg=plannable[0].arg)

  # replace DEFINE_LOCAL with merged DEFINE_LOCAL + INDEX
  rmap:dict[UOp, UOp] = {dl: (merged.index(UOp.const(dtypes.int32, offsets[dl]), ptr=True) if offsets[dl] else merged) for dl in plannable}

  # rebuild linear source with new DEFINE_LOCALs + INDEX
  offset_ptrs = dedup(v for v in rmap.values() if v is not merged)
  new_src = (merged, *[x for p in offset_ptrs for x in (p.src[1], p)], *(u for u in lin.src if u not in rmap))

  # return new linear with substitutions
  return UOp(Ops.LINEAR, src=new_src).substitute(rmap, name="local memory plan", walk=True)
