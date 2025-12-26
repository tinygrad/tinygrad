# RDNA3 Register Allocator with liveness-based reuse
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import DType, PtrDType, AddrSpace, dtypes
from tinygrad.helpers import getenv
from extra.assembly.rdna3.autogen import VGPR, SGPR

class RDNARegAlloc:
  """Register allocator for RDNA3 with liveness analysis and register reuse."""
  MAX_SGPR = 100  # RDNA3 limit ~106, reserve some for scratch

  def __init__(self, uops: list[UOp]):
    self.uops = uops
    # Register pools
    self._free_vgprs: list[int] = []
    self._free_vgpr_pairs: list[int] = []
    self._free_vgpr_ranges: list[tuple[int, int]] = []
    self._free_sgprs: list[int] = []
    # Ownership tracking
    self._vgpr_owner: dict[int, UOp] = {}
    self._sgpr_owner: dict[int, UOp] = {}
    self._range_owner: dict[int, UOp] = {}
    self._vgpr_ranges: dict[int, int] = {}  # base -> count
    self._vgpr_pairs: set[int] = set()
    self._sgpr_pairs: set[int] = set()
    # Counters: v[0:2] is local_xyz, s[0:1] kernarg, s[2:4] group id
    self._next_vgpr, self._next_sgpr = 3, 5
    self._max_vgpr, self._max_sgpr = 3, 5
    # Pending deaths scheduled by position
    self._pending_vgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_sgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_range_deaths: dict[int, list[int]] = defaultdict(list)
    # Scratch registers
    self._scratch_vgpr = -1
    self._deferred_store_vgpr = -1
    # Run liveness analysis
    self._last_use, self._aliases, self._effective_death = self._analyze_liveness()

  def _analyze_liveness(self) -> tuple[dict[UOp, int], dict[UOp, UOp], dict[UOp, int]]:
    """Compute last use positions, aliases, and effective death times."""
    last_use: dict[UOp, int] = {}
    aliases: dict[UOp, UOp] = {}
    # Find loop ranges for lifetime extension
    loop_ranges: dict[int, int] = {}
    range_positions: dict[UOp, int] = {}
    for i, u in enumerate(self.uops):
      if u.op is Ops.RANGE: range_positions[u] = i
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        if u.src[1] in range_positions: loop_ranges[range_positions[u.src[1]]] = i
    # First pass: track direct uses and aliases
    for i, u in enumerate(self.uops):
      for src in u.src: last_use[src] = i
      # Track INDEX sources through LOAD/STORE
      if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].op is Ops.INDEX:
        last_use[u.src[0]] = i
        for src in u.src[0].src: last_use[src] = i
      # Track RANGE.src[0] through END
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE and len(u.src[1].src) > 0:
        last_use[u.src[1].src[0]] = i
      # Build alias relationships
      if u.op is Ops.AFTER: aliases[u] = u.src[0]
      # BITCAST is always an alias (just reinterprets bits) - critical for int32<->uint32 in division lowering
      if u.op is Ops.BITCAST: aliases[u] = u.src[0]
      # CAST is an alias only when dtypes match or source is pointer
      if u.op is Ops.CAST and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        aliases[u] = u.src[0]
      if u.op is Ops.GEP and isinstance(u.src[0].dtype, DType) and u.src[0].dtype.count > 1:
        aliases[u] = u.src[0]
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and len(u.src) > 0:
        if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
          if u.op in {Ops.INDEX, Ops.LOAD}: aliases[u] = u.src[0]
      if u.op is Ops.VECTORIZE:
        for src in u.src:
          if src in aliases:
            root = src
            while root in aliases: root = aliases[root]
            if root.op is Ops.DEFINE_REG: continue
          aliases[src] = u
          for src_src in src.src:
            if src_src not in aliases: aliases[src_src] = u
    # Extend lifetimes for values defined outside but used inside loops
    uop_positions = {u: i for i, u in enumerate(self.uops)}
    for uop, use_pos in list(last_use.items()):
      if uop not in uop_positions: continue
      def_pos = uop_positions[uop]
      for range_pos, end_pos in loop_ranges.items():
        if def_pos <= range_pos and range_pos < use_pos <= end_pos:
          last_use[uop] = max(last_use[uop], end_pos)
    # Extend SPECIAL lifetimes to end of kernel
    max_pos = len(self.uops) - 1
    for u in self.uops:
      if u.op is Ops.SPECIAL: last_use[u] = max_pos
    # Compute effective death for alias groups
    def get_root(u: UOp) -> UOp:
      while u in aliases: u = aliases[u]
      return u
    alias_groups: dict[UOp, list[UOp]] = defaultdict(list)
    for u in aliases: alias_groups[get_root(u)].append(u)
    effective_death: dict[UOp, int] = {}
    for root, alias_list in alias_groups.items():
      death = last_use.get(root, -1)
      for alias in alias_list: death = max(death, last_use.get(alias, -1))
      effective_death[root] = death
    return last_use, aliases, effective_death

  def _get_root(self, u: UOp) -> UOp:
    while u in self._aliases: u = self._aliases[u]
    return u

  def _get_death_pos(self, owner: UOp) -> int:
    root = self._get_root(owner)
    return self._effective_death.get(root, self._last_use.get(owner, -1))

  def _schedule_vgpr_death(self, reg: int, owner: UOp):
    death_pos = self._get_death_pos(owner)
    if death_pos >= 0: self._pending_vgpr_deaths[death_pos + 1].append(reg)

  def _schedule_sgpr_death(self, reg: int, owner: UOp):
    death_pos = self._get_death_pos(owner)
    if death_pos >= 0: self._pending_sgpr_deaths[death_pos + 1].append(reg)

  def _schedule_range_death(self, base: int, owner: UOp):
    death_pos = self._get_death_pos(owner)
    if death_pos >= 0: self._pending_range_deaths[death_pos + 1].append(base)

  def cancel_vgpr_death(self, reg: int):
    """Cancel pending death for a VGPR (for register ownership transfer)."""
    for pos in list(self._pending_vgpr_deaths.keys()):
      if reg in self._pending_vgpr_deaths[pos]: self._pending_vgpr_deaths[pos].remove(reg)

  def reschedule_vgpr_death(self, reg: int, new_owner: UOp):
    """Transfer VGPR ownership and reschedule death."""
    self._vgpr_owner[reg] = new_owner
    self.cancel_vgpr_death(reg)
    self._schedule_vgpr_death(reg, new_owner)

  def free_dead_regs(self, pos: int):
    """Free registers scheduled to die at position pos."""
    # Free ranges
    for base in self._pending_range_deaths.get(pos, []):
      if base in self._range_owner:
        del self._range_owner[base]
        count = self._vgpr_ranges.pop(base, 8)
        claimed = [r for r in range(base, base + count) if r in self._vgpr_owner]
        if not claimed:
          self._free_vgpr_ranges.append((base, count))
        else:
          for r in range(base, base + count):
            if r not in self._vgpr_owner: self._free_vgprs.append(r)
    # Free VGPRs
    dead_set = set(self._pending_vgpr_deaths.get(pos, []))
    for reg in self._pending_vgpr_deaths.get(pos, []):
      if reg not in self._vgpr_owner: continue
      del self._vgpr_owner[reg]
      if reg in self._vgpr_pairs:
        base_reg = reg if reg % 2 == 0 else reg - 1
        other = base_reg + 1 if reg == base_reg else base_reg
        if other in dead_set and base_reg not in self._free_vgpr_pairs:
          self._free_vgpr_pairs.append(base_reg)
          self._vgpr_pairs.discard(base_reg)
          self._vgpr_pairs.discard(other)
          if other in self._vgpr_owner: del self._vgpr_owner[other]
      else:
        self._free_vgprs.append(reg)
    # Free SGPRs
    for reg in self._pending_sgpr_deaths.get(pos, []):
      if reg not in self._sgpr_owner or reg in self._sgpr_pairs: continue
      del self._sgpr_owner[reg]
      self._free_sgprs.append(reg)

  def alloc_vgpr(self, owner: UOp) -> VGPR:
    """Allocate a single VGPR."""
    if self._free_vgprs:
      reg = self._free_vgprs.pop()
    elif self._free_vgpr_ranges:
      base, count = self._free_vgpr_ranges.pop()
      reg = base
      if count > 1: self._free_vgpr_ranges.append((base + 1, count - 1))
    else:
      reg = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    self._vgpr_owner[reg] = owner
    self._schedule_vgpr_death(reg, owner)
    return VGPR(reg)

  def alloc_vgpr_pair(self, owner: UOp) -> VGPR:
    """Allocate aligned VGPR pair for 64-bit values."""
    if self._free_vgpr_pairs:
      reg = self._free_vgpr_pairs.pop()
    else:
      if self._next_vgpr % 2 != 0: self._next_vgpr += 1
      reg = self._next_vgpr
      self._next_vgpr += 2
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    self._vgpr_owner[reg] = self._vgpr_owner[reg + 1] = owner
    self._vgpr_pairs.add(reg)
    self._vgpr_pairs.add(reg + 1)
    self._schedule_vgpr_death(reg, owner)
    self._schedule_vgpr_death(reg + 1, owner)
    return VGPR(reg, 2)

  def alloc_vgpr_range(self, owner: UOp, count: int = 8) -> VGPR:
    """Allocate contiguous VGPR range (for WMMA/VECTORIZE)."""
    for i, (base, range_count) in enumerate(self._free_vgpr_ranges):
      if range_count >= count:
        self._free_vgpr_ranges.pop(i)
        if range_count > count: self._free_vgpr_ranges.append((base + count, range_count - count))
        self._range_owner[base] = owner
        self._vgpr_ranges[base] = count
        self._schedule_range_death(base, owner)
        return VGPR(base, count)
    base = self._next_vgpr
    if base % 2 != 0: base = self._next_vgpr = self._next_vgpr + 1
    self._next_vgpr = base + count
    self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    self._range_owner[base] = owner
    self._vgpr_ranges[base] = count
    self._schedule_range_death(base, owner)
    return VGPR(base, count)

  def alloc_sgpr(self, owner: UOp) -> SGPR | None:
    """Allocate single SGPR, returns None if exhausted."""
    if self._free_sgprs:
      reg = self._free_sgprs.pop()
    elif self._next_sgpr < self.MAX_SGPR:
      reg = self._next_sgpr
      self._next_sgpr += 1
      self._max_sgpr = max(self._max_sgpr, self._next_sgpr)
    else:
      return None
    self._sgpr_owner[reg] = owner
    self._schedule_sgpr_death(reg, owner)
    return SGPR(reg)

  def alloc_sgpr_pair(self, owner: UOp) -> SGPR:
    """Allocate aligned SGPR pair for 64-bit buffer addresses."""
    if self._next_sgpr % 2 != 0: self._next_sgpr += 1
    reg = self._next_sgpr
    self._next_sgpr += 2
    self._max_sgpr = max(self._max_sgpr, self._next_sgpr)
    self._sgpr_owner[reg] = self._sgpr_owner[reg + 1] = owner
    self._sgpr_pairs.add(reg)
    self._sgpr_pairs.add(reg + 1)
    return SGPR(reg, 2)

  def get_scratch_vgpr(self, count: int = 1) -> int:
    """Get scratch VGPR base for temporary operations."""
    if self._scratch_vgpr < 0:
      self._scratch_vgpr = self._next_vgpr
      self._next_vgpr += 32  # Reserve for 64-bit division temps
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    return self._scratch_vgpr

  def get_deferred_store_vgpr(self) -> str:
    """Get dedicated VGPR for deferred store address computation."""
    if self._deferred_store_vgpr < 0:
      self._deferred_store_vgpr = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    return f"v{self._deferred_store_vgpr}"

  def extend_lifetime(self, uop: UOp, pos: int):
    """Extend a UOp's last use position (for recomputation patterns)."""
    self._last_use[uop] = pos

  def get_last_use(self, uop: UOp) -> int:
    """Get last use position for a UOp."""
    return self._last_use.get(uop, -1)

  def is_vgpr_owner(self, reg: int) -> bool:
    """Check if register has an owner."""
    return reg in self._vgpr_owner

  def get_vgpr_owner(self, reg: int) -> UOp | None:
    """Get the owner of a VGPR."""
    return self._vgpr_owner.get(reg)

  def free_vgpr(self, reg: int):
    """Immediately free a VGPR (for look-ahead packing)."""
    if reg in self._vgpr_owner:
      del self._vgpr_owner[reg]
      self._free_vgprs.append(reg)

  @property
  def max_vgpr(self) -> int: return self._max_vgpr
  @property
  def max_sgpr(self) -> int: return self._max_sgpr

  @staticmethod
  def needs_vgpr_pair(dtype: DType) -> bool:
    """Check if dtype needs VGPR pair (64-bit)."""
    return dtype in (dtypes.float64, dtypes.long, dtypes.ulong) or (hasattr(dtype, 'itemsize') and dtype.itemsize == 8)
