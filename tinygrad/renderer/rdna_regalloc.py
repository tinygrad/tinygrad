# RDNA3 Register Allocator with liveness-based reuse
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import DType, PtrDType, AddrSpace, dtypes
from tinygrad.helpers import getenv
from extra.assembly.rdna3.autogen import VGPR, SGPR

class RDNARegAlloc:
  """Register allocator for RDNA3 with liveness analysis and register reuse."""
  MAX_VGPR = 256  # RDNA3 has v0-v255
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
    self._peak_vgpr = 3  # Track peak simultaneous usage
    self._peak_info = None  # Info about when peak was hit
    # Pending deaths scheduled by position
    self._pending_vgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_sgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_range_deaths: dict[int, list[int]] = defaultdict(list)
    # Scratch registers
    self._scratch_vgpr = -1
    self._scratch_count = 0
    self._deferred_store_vgpr = -1
    # Loop-local buffer tracking: DEFINE_REG -> (loop_start, loop_end) if buffer is loop-local
    self._loop_local_buffers: dict[UOp, tuple[int, int]] = {}
    # Run liveness analysis
    self._last_use, self._aliases, self._effective_death = self._analyze_liveness()
    # Analyze loop-local buffers after liveness (needs loop_ranges)
    self._analyze_loop_local_buffers()
    # Pre-analyze VECTORIZE needs and reserve high registers for them
    self._vectorize_pool: list[tuple[int, int]] = []  # (base, count) reserved ranges
    self._init_vectorize_pool()
    if getenv("RDNA_POOL_DEBUG", 0) and self._vectorize_pool:
      print(f"[POOL] VECTORIZE pool: {self._vectorize_pool}")

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
      # Track INDEX through LOAD/STORE - offset and condition need to live until the memory op
      # src[0] is the buffer (SGPR), src[1] is the offset (VGPR address), src[2] is optional condition
      if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].op is Ops.INDEX:
        last_use[u.src[0]] = i
        if len(u.src[0].src) > 1: last_use[u.src[0].src[1]] = i  # Extend offset lifetime
        if len(u.src[0].src) > 2: last_use[u.src[0].src[2]] = i  # Extend condition lifetime
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
        # Only alias GEP if it doesn't need a shift (extracting low bits)
        # High-bit extraction (idx % 2 == 1 for 16-bit) needs its own register for shift result
        idx = u.arg[0] if isinstance(u.arg, tuple) else u.arg
        src_dtype = u.src[0].dtype
        needs_shift = False
        if src_dtype.scalar().itemsize == 2: needs_shift = (idx % 2 == 1)  # 16-bit: high half needs shift
        elif src_dtype.scalar().itemsize == 1: needs_shift = (idx % 4 != 0)  # 8-bit: non-first byte needs shift
        if not needs_shift: aliases[u] = u.src[0]
      # NOTE: We intentionally DON'T alias register-space INDEX/LOAD here.
      # Register-space operations reference the accumulator range directly without allocating,
      # so they don't need aliasing for register reuse. More importantly, aliasing them
      # would incorrectly extend the accumulator's lifetime based on CAST uses.
      if u.op is Ops.VECTORIZE:
        # Only alias sources if VECTORIZE might reuse their registers (32-bit types with contiguous layout)
        # For 16-bit types, VECTORIZE packs sources into new registers, so sources should die at VECTORIZE position
        scalar_dtype = u.dtype.scalar()
        if scalar_dtype.itemsize >= 4:  # 32-bit or larger - might reuse source registers
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
    # Extend DEFINE_REG lifetime for register-space LOADs
    # Register-space LOADs return a reference to the accumulator register, not a copy.
    # The accumulator must stay alive until the last use of any LOAD that references it.
    for i, u in enumerate(self.uops):
      if u.op is Ops.LOAD and len(u.src) > 0 and u.src[0].op is Ops.INDEX:
        idx_uop = u.src[0]
        buf_uop = idx_uop.src[0] if len(idx_uop.src) > 0 else None
        # Walk through AFTER chain to find DEFINE_REG
        while buf_uop is not None and buf_uop.op is Ops.AFTER:
          buf_uop = buf_uop.src[0]
        if buf_uop is not None and buf_uop.op is Ops.DEFINE_REG:
          # Check if this is actually a register-space buffer
          if isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.REG:
            # Extend DEFINE_REG's last_use to this LOAD's last use
            load_last_use = last_use.get(u, i)
            last_use[buf_uop] = max(last_use.get(buf_uop, 0), load_last_use)
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

  def _analyze_loop_local_buffers(self):
    """Detect DEFINE_REG buffers that are completely reinitialized inside a loop.
    
    If a buffer is zeroed/initialized at the start of each loop iteration, its registers
    can be freed at the end of each iteration and reallocated, rather than staying live
    for the entire kernel. This is what LLVM does automatically.
    """
    # Find loop ranges
    loop_ranges: dict[int, int] = {}  # range_pos -> end_pos
    range_uops: dict[int, UOp] = {}  # range_pos -> RANGE UOp
    for i, u in enumerate(self.uops):
      if u.op is Ops.RANGE: range_uops[i] = u
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        for rpos, ruop in range_uops.items():
          if ruop is u.src[1]:
            loop_ranges[rpos] = i
            break

    # Find DEFINE_REG buffers
    define_regs: list[tuple[int, UOp]] = []
    for i, u in enumerate(self.uops):
      if u.op is Ops.DEFINE_REG:
        if isinstance(u.dtype, PtrDType) and u.dtype.addrspace == AddrSpace.REG:
          define_regs.append((i, u))

    # For each DEFINE_REG, check if it's loop-local
    for def_pos, def_uop in define_regs:
      buf_size = def_uop.dtype.size if hasattr(def_uop.dtype, 'size') and def_uop.dtype.size > 0 else 0
      if buf_size == 0: continue

      # Find all STOREs to this buffer
      stores: list[tuple[int, bool, int]] = []  # (pos, is_const_zero, offset)
      for i, u in enumerate(self.uops):
        if u.op is Ops.STORE and len(u.src) >= 2:
          idx_uop = u.src[0]
          val_uop = u.src[1]
          if idx_uop.op is Ops.INDEX and len(idx_uop.src) >= 2:
            buf = idx_uop.src[0]
            while buf.op is Ops.AFTER: buf = buf.src[0]
            if buf is def_uop:
              offset_uop = idx_uop.src[1]
              offset = offset_uop.arg if offset_uop.op is Ops.CONST else -1
              is_zero = val_uop.op is Ops.CONST and val_uop.arg == 0
              stores.append((i, is_zero, offset))

      if not stores: continue

      # Check each loop to see if this buffer is completely zeroed at the start
      for range_pos, end_pos in loop_ranges.items():
        if range_pos <= def_pos: continue  # Buffer defined before this loop

        # Find stores inside this loop, right after loop start (initialization region)
        # Allow some slack - init stores should be within first ~50% of loop body before inner loops
        init_region_end = range_pos + (end_pos - range_pos) // 2
        
        # Find the first inner loop (if any) - init must be before it
        inner_loop_start = end_pos
        for other_range_pos in loop_ranges:
          if range_pos < other_range_pos < end_pos:
            inner_loop_start = min(inner_loop_start, other_range_pos)
        init_region_end = min(init_region_end, inner_loop_start)

        # Count zero-init stores in the init region
        init_stores = [(pos, is_zero, off) for pos, is_zero, off in stores 
                       if range_pos < pos < init_region_end]
        zero_init_offsets = set(off for pos, is_zero, off in init_stores if is_zero and off >= 0)

        # Check if ALL buffer elements are zero-initialized
        if len(zero_init_offsets) >= buf_size:
          # This buffer is completely reinitialized at the start of this loop
          self._loop_local_buffers[def_uop] = (range_pos, end_pos)
          if getenv("RDNA_LOOP_LOCAL_DEBUG", 0):
            print(f"[LOOP_LOCAL] DEFINE_REG@{def_pos} ({buf_size} elements) is loop-local to RANGE@{range_pos}-END@{end_pos}")
          break  # Use the innermost containing loop

  def _init_vectorize_pool(self):
    """Pre-analyze VECTORIZE ops and reserve high registers for contiguous allocations.
    This prevents fragmentation from LOADs affecting VECTORIZE range allocation.
    
    NOTE: Currently disabled as it causes register allocation issues when LOADs
    overlap with the reserved pool. The proper fix requires ensuring _next_vgpr
    never exceeds the pool boundary, but this needs more careful implementation.
    """
    # TODO: Re-enable when pool/regular allocation interaction is properly handled
    return

  def _get_root(self, u: UOp) -> UOp:
    while u in self._aliases: u = self._aliases[u]
    return u

  def _get_death_pos(self, owner: UOp) -> int:
    # For loop-local buffers, death is at the loop END, not kernel end
    if owner in self._loop_local_buffers:
      _, end_pos = self._loop_local_buffers[owner]
      return end_pos
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
    elif self._next_vgpr < self.MAX_VGPR:
      reg = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    else:
      # At limit - find any unused register in 0-255
      used = set(self._vgpr_owner.keys())
      for rbase, rcount in self._vgpr_ranges.items():
        used.update(range(rbase, rbase + rcount))
      reg = next((r for r in range(self.MAX_VGPR) if r not in used), self._next_vgpr)
      if reg >= self.MAX_VGPR:
        self._next_vgpr = reg + 1
        self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    # Don't fail immediately - allow temporary overflow, check at finalize
    self._vgpr_owner[reg] = owner
    self._schedule_vgpr_death(reg, owner)
    # Track peak simultaneous usage (owned + ranges)
    current = len(self._vgpr_owner) + sum(self._vgpr_ranges.values())
    if current > self._peak_vgpr:
      self._peak_vgpr = current
      # Count by op type for debugging
      op_counts: dict[str, int] = {}
      load_lifetimes: list[int] = []
      load_details: list[tuple] = []  # (reg, def_pos)
      add_details: list[tuple] = []  # (reg, def_pos, lifetime)
      uop_positions = {u: i for i, u in enumerate(self.uops)}
      for r, o in self._vgpr_owner.items():
        op_name = o.op.name
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
        if o.op is Ops.LOAD and o in uop_positions:
          def_pos = uop_positions[o]
          death_pos = self._get_death_pos(o)
          load_lifetimes.append((def_pos, death_pos - def_pos))
          load_details.append((r, def_pos))
        if o.op is Ops.ADD and o in uop_positions:
          def_pos = uop_positions[o]
          death_pos = self._get_death_pos(o)
          add_details.append((r, def_pos, death_pos - def_pos))
      # Find current position
      cur_pos = len([u for u in self.uops if u in self._vgpr_owner.values() or u in self._range_owner.values()])
      for pi, pu in enumerate(self.uops):
        if pu == owner:
          cur_pos = pi
          break
      self._peak_info = (dict(self._vgpr_ranges), len(self._vgpr_owner), owner.op.name, op_counts, load_lifetimes, cur_pos, load_details, add_details)
    return VGPR(reg)

  def alloc_vgpr_pair(self, owner: UOp) -> VGPR:
    """Allocate aligned VGPR pair for 64-bit values."""
    if self._free_vgpr_pairs:
      reg = self._free_vgpr_pairs.pop()
    else:
      if self._next_vgpr % 2 != 0: self._next_vgpr += 1
      reg = self._next_vgpr
      self._next_vgpr = reg + 2
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    # Don't fail immediately - allow temporary overflow, check at finalize
    self._vgpr_owner[reg] = self._vgpr_owner[reg + 1] = owner
    self._vgpr_pairs.add(reg)
    self._vgpr_pairs.add(reg + 1)
    self._schedule_vgpr_death(reg, owner)
    self._schedule_vgpr_death(reg + 1, owner)
    return VGPR(reg, 2)

  def alloc_vgpr_range(self, owner: UOp, count: int = 8, align: int = 2) -> VGPR:
    """Allocate contiguous VGPR range (for WMMA/VECTORIZE/DEFINE_REG).
    align=2 for WMMA (default), align=1 for DEFINE_REG accumulators."""
    # For VECTORIZE, try to use the reserved pool first
    if owner is not None and owner.op is Ops.VECTORIZE and self._vectorize_pool:
      for i, (pool_base, pool_size) in enumerate(self._vectorize_pool):
        if pool_size >= count and (align <= 1 or pool_base % align == 0):
          # Allocate from the start of the pool
          self._vectorize_pool[i] = (pool_base + count, pool_size - count)
          if self._vectorize_pool[i][1] == 0:
            self._vectorize_pool.pop(i)
          self._range_owner[pool_base] = owner
          self._vgpr_ranges[pool_base] = count
          self._max_vgpr = max(self._max_vgpr, pool_base + count)
          self._schedule_range_death(pool_base, owner)
          return VGPR(pool_base, count)
    # First try existing free ranges
    for i, (base, range_count) in enumerate(self._free_vgpr_ranges):
      if range_count >= count and (align <= 1 or base % align == 0):
        self._free_vgpr_ranges.pop(i)
        if range_count > count: self._free_vgpr_ranges.append((base + count, range_count - count))
        self._range_owner[base] = owner
        self._vgpr_ranges[base] = count
        self._schedule_range_death(base, owner)
        return VGPR(base, count)
    # Try to find contiguous free single VGPRs
    if self._free_vgprs and count <= 16:  # Only for small ranges to avoid expensive search
      sorted_free = sorted(self._free_vgprs)
      for i in range(len(sorted_free) - count + 1):
        base = sorted_free[i]
        if align > 1 and base % align != 0: continue
        # Check if next 'count' registers are contiguous
        if sorted_free[i:i+count] == list(range(base, base + count)):
          # Found contiguous range in free_vgprs - claim them
          for r in range(base, base + count):
            self._free_vgprs.remove(r)
          self._range_owner[base] = owner
          self._vgpr_ranges[base] = count
          self._schedule_range_death(base, owner)
          return VGPR(base, count)
    # Allocate new registers (but not if it would collide with VECTORIZE pool)
    base = self._next_vgpr
    if align > 1 and base % align != 0: base = self._next_vgpr = self._next_vgpr + (align - base % align)
    # Check for collision with VECTORIZE pool
    if self._vectorize_pool:
      pool_start = self._vectorize_pool[0][0]
      if base + count > pool_start:
        # Would collide with pool - this means we've run out of low registers
        # Fall through to allocate anyway (will overflow and fail at finalize)
        pass
    # If this would exceed 256, try harder to find existing free space
    if base + count > self.MAX_VGPR:
      # Look for any contiguous free region in existing allocations
      # Build a set of all currently used registers
      used = set(self._vgpr_owner.keys())
      for rbase, rcount in self._vgpr_ranges.items():
        used.update(range(rbase, rbase + rcount))
      # Find a gap of size 'count' - try aligned first, then unaligned
      found_gap = False
      for try_align in ([align, 1] if align > 1 else [1]):
        for start in range(0, self.MAX_VGPR - count + 1, try_align):
          if all(r not in used for r in range(start, start + count)):
            base = start
            found_gap = True
            break
        if found_gap: break
    self._next_vgpr = max(self._next_vgpr, base + count)
    self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    # Don't fail immediately - allow temporary overflow, check at finalize
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
    """Get scratch VGPR base for temporary operations. Dynamically expands as needed."""
    if self._scratch_vgpr < 0:
      # Find a range of 'count' registers that are not currently owned
      base = self._next_vgpr
      while any(r in self._vgpr_owner for r in range(base, base + count)):
        base += 1
      self._scratch_vgpr = base
      self._scratch_count = count
      self._next_vgpr = max(self._next_vgpr, base + count)
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      if self._next_vgpr > self.MAX_VGPR:
        raise RuntimeError(f"VGPR overflow: scratch VGPRs exceed limit (need {self._next_vgpr}, max {self.MAX_VGPR})")
    elif count > self._scratch_count:
      # Need more scratch VGPRs. Check if we can expand in place or need to relocate.
      expand_start = self._scratch_vgpr + self._scratch_count
      expand_end = self._scratch_vgpr + count
      # Check if expansion range overlaps with any owned registers
      can_expand = all(r not in self._vgpr_owner for r in range(expand_start, expand_end))
      if can_expand and expand_end <= self._next_vgpr:
        # Expansion range is within already-allocated space and not owned - just expand
        self._scratch_count = count
      elif can_expand:
        # Expansion range extends past _next_vgpr but is free - extend
        self._scratch_count = count
        self._next_vgpr = expand_end
        self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      else:
        # Expansion would overlap with owned registers - relocate scratch to end
        self._scratch_vgpr = self._next_vgpr
        self._scratch_count = count
        self._next_vgpr += count
        self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      if self._next_vgpr > self.MAX_VGPR:
        raise RuntimeError(f"VGPR overflow: scratch VGPRs exceed limit (need {self._next_vgpr}, max {self.MAX_VGPR})")
    return self._scratch_vgpr

  def get_deferred_store_vgpr(self) -> str:
    """Get dedicated VGPR for deferred store address computation."""
    if self._deferred_store_vgpr < 0:
      self._deferred_store_vgpr = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      if self._next_vgpr > self.MAX_VGPR:
        raise RuntimeError(f"VGPR overflow: deferred store VGPR exceeds limit (need {self._next_vgpr}, max {self.MAX_VGPR})")
    return f"v{self._deferred_store_vgpr}"

  def extend_lifetime(self, uop: UOp, pos: int):
    """Extend a UOp's last use position (for recomputation patterns)."""
    self._last_use[uop] = pos

  def get_last_use(self, uop: UOp) -> int:
    """Get effective death position for a UOp (considering alias groups)."""
    return self._get_death_pos(uop)

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

  def finalize(self):
    """Check final register counts and raise error if exceeded limits."""
    # Use peak simultaneous usage, not max allocated index
    # Registers may temporarily get high indices but be freed before peak
    if self._peak_vgpr > self.MAX_VGPR:
      # Build summary showing what exceeded the limit
      summary = [f"VGPR overflow: allocated up to v{self._max_vgpr-1}, max v{self.MAX_VGPR-1}"]
      summary.append(f"  Peak simultaneous: {self._peak_vgpr} registers")
      if self._peak_info:
        ranges, owned, last_op, op_counts, load_lifetimes, peak_pos, load_details, add_details = self._peak_info
        summary.append(f"  At peak (pos {peak_pos}): DEFINE_REG={ranges}, owned={owned}, allocating for {last_op}")
        summary.append(f"  Owned by op: {op_counts}")
        if load_lifetimes:
          lifetimes = [l for _, l in load_lifetimes]
          positions = [p for p, _ in load_lifetimes]
          summary.append(f"  LOAD lifetimes: min={min(lifetimes)}, max={max(lifetimes)}, avg={sum(lifetimes)/len(lifetimes):.1f}")
          summary.append(f"  LOAD positions: {min(positions)}-{max(positions)}")
          # Show some load register details
          load_details.sort(key=lambda x: x[1])
          regs = sorted(set(r for r, _ in load_details))
          summary.append(f"  LOAD regs: {min(regs)}-{max(regs)} ({len(regs)} distinct)")
        if add_details:
          lifetimes = [l for _, _, l in add_details]
          positions = [p for _, p, _ in add_details]
          summary.append(f"  ADD lifetimes: min={min(lifetimes)}, max={max(lifetimes)}, avg={sum(lifetimes)/len(lifetimes):.1f}")
          summary.append(f"  ADD positions: {min(positions)}-{max(positions)}")
          # Show long-lived ADDs
          long_adds = [(r, p, l) for r, p, l in add_details if l > 100]
          if long_adds:
            summary.append(f"  Long-lived ADDs (lifetime>100): {len(long_adds)}")
            for r, p, l in sorted(long_adds, key=lambda x: -x[2])[:10]:
              summary.append(f"    v{r}: pos={p}, lifetime={l}")
      if self._scratch_vgpr >= 0: summary.append(f"  Scratch: v{self._scratch_vgpr}")
      raise RuntimeError("\n".join(summary))

  @staticmethod
  def needs_vgpr_pair(dtype: DType) -> bool:
    """Check if dtype needs VGPR pair (64-bit)."""
    return dtype in (dtypes.float64, dtypes.long, dtypes.ulong) or (hasattr(dtype, 'itemsize') and dtype.itemsize == 8)
