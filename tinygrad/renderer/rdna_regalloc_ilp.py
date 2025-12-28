# RDNA3 Register Allocator using OR-Tools CP-SAT
# Enable with RDNA_ILP_REGALLOC=1
# Debug with RDNA_ILP_DEBUG=1
#
# Uses constraint programming with NoOverlap2D for efficient interference handling:
# 1. Sweep-line algorithm for O(n log n) liveness analysis
# 2. Single NoOverlap2D constraint instead of O(n²) pairwise constraints
# 3. Domain restriction for alignment and reserved registers

from collections import defaultdict
from dataclasses import dataclass
from ortools.sat.python import cp_model  # requires: pip install ortools
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import DType, PtrDType, AddrSpace, dtypes
from tinygrad.helpers import getenv
from extra.assembly.rdna3.autogen import VGPR, SGPR

DEBUG_ILP = getenv("RDNA_ILP_DEBUG", 0)

@dataclass(frozen=True)
class TempReg:
  """Synthetic register request for temporaries needed by complex operations."""
  parent: UOp
  index: int
  count: int
  align: int

class RDNARegAllocILP:
  """CP-SAT based register allocator for RDNA3 that minimizes total register usage."""
  MAX_VGPR = 256
  MAX_SGPR = 100

  def __init__(self, uops: list[UOp], reg_element_last_use: dict[tuple[UOp, int], int] | None = None):
    self.uops = uops
    self._reg_element_last_use = reg_element_last_use or {}
    self._last_use, self._aliases, self._effective_death = self._analyze_liveness()
    self._vgpr_assignment: dict[UOp | TempReg, int] = {}
    self._sgpr_assignment: dict[UOp | TempReg, int] = {}
    self._vgpr_sizes: dict[UOp | TempReg, int] = {}
    self._sgpr_sizes: dict[UOp | TempReg, int] = {}
    self._temp_reg_map: dict[tuple[UOp, int], TempReg] = {}
    self._temp_alloc_order: dict[UOp, list[TempReg]] = defaultdict(list)
    self._solve_ilp()
    self._vgpr_owner: dict[int, UOp] = {}
    self._sgpr_owner: dict[int, UOp] = {}
    self._range_owner: dict[int, UOp] = {}
    self._vgpr_ranges: dict[int, int] = {}
    self._vgpr_pairs: set[int] = set()
    self._sgpr_pairs: set[int] = set()
    self._free_vgprs: list[int] = []
    self._free_vgpr_pairs: list[int] = []
    self._free_vgpr_ranges: list[tuple[int, int]] = []
    self._free_sgprs: list[int] = []
    self._pending_vgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_sgpr_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_range_deaths: dict[int, list[int]] = defaultdict(list)
    self._pending_element_deaths: dict[int, list[tuple[int, UOp]]] = defaultdict(list)
    self._scratch_vgpr = -1
    self._deferred_store_vgpr = -1
    self._temp_alloc_idx: dict[UOp, int] = {}
    self._vgpr_allocated: set[UOp] = set()  # track which UOps have had their main register allocated
    self._sgpr_allocated: set[UOp] = set()
    self._max_vgpr = max((base + size for base, size in zip(self._vgpr_assignment.values(), self._vgpr_sizes.values())), default=2)
    self._max_sgpr = max((base + size for base, size in zip(self._sgpr_assignment.values(), self._sgpr_sizes.values())), default=5)
    # Start greedy allocation after ILP-assigned registers
    self._next_vgpr = self._max_vgpr
    self._next_sgpr = self._max_sgpr

  def _analyze_liveness(self) -> tuple[dict[UOp, int], dict[UOp, UOp], dict[UOp, int]]:
    last_use: dict[UOp, int] = {}
    aliases: dict[UOp, UOp] = {}
    loop_ranges: dict[int, int] = {}
    range_positions: dict[UOp, int] = {}
    for i, u in enumerate(self.uops):
      if u.op is Ops.RANGE: range_positions[u] = i
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        if u.src[1] in range_positions: loop_ranges[range_positions[u.src[1]]] = i
    for i, u in enumerate(self.uops):
      for src in u.src: last_use[src] = i
      # Track INDEX through LOAD/STORE - only the offset (src[1]) needs to live until the memory op
      # src[0] is the buffer (SGPR), src[1] is the offset (VGPR address)
      if u.op in {Ops.LOAD, Ops.STORE} and len(u.src) > 0 and u.src[0].op is Ops.INDEX:
        last_use[u.src[0]] = i
        if len(u.src[0].src) > 1: last_use[u.src[0].src[1]] = i  # Only extend offset, not buffer
      # STORE: the value being stored needs to live until the STORE
      # Only extend the immediate value, not its transitive sources (which are consumed when computing the value)
      if u.op is Ops.STORE and len(u.src) > 1:
        last_use[u.src[1]] = max(last_use.get(u.src[1], 0), i)
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE and len(u.src[1].src) > 0:
        last_use[u.src[1].src[0]] = i
      if u.op is Ops.AFTER: aliases[u] = u.src[0]
      if u.op is Ops.BITCAST: aliases[u] = u.src[0]
      if u.op is Ops.CAST:
        # CAST is alias when dtypes match OR source is pointer
        if u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType):
          aliases[u] = u.src[0]
        # CAST from register-space LOAD reuses the accumulator register
        elif (u.src[0].op is Ops.LOAD and len(u.src[0].src) > 0 and u.src[0].src[0].op is Ops.INDEX and
              len(u.src[0].src[0].src) > 0 and isinstance(u.src[0].src[0].src[0].dtype, PtrDType) and
              u.src[0].src[0].src[0].dtype.addrspace == AddrSpace.REG):
          aliases[u] = u.src[0]
      if u.op is Ops.GEP and isinstance(u.src[0].dtype, DType) and u.src[0].dtype.count > 1:
        aliases[u] = u.src[0]
      # NOTE: We intentionally DON'T alias register-space INDEX/LOAD here.
      # Register-space operations reference the accumulator range directly without allocating,
      # so they don't need aliasing for register reuse. More importantly, aliasing them
      # would incorrectly extend the accumulator's lifetime based on CAST uses.

      # NOTE: We do NOT alias scalar ALU ops here. Although the greedy allocator reuses
      # dying source registers, the ILP allocator pre-assigns all registers. The solver
      # will find optimal placement for ALU ops given their short lifetimes.

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
    uop_positions = {u: i for i, u in enumerate(self.uops)}
    if DEBUG_ILP:
      print(f"[ILP] Loop ranges: {loop_ranges}")
    for uop, use_pos in list(last_use.items()):
      if uop not in uop_positions: continue
      def_pos = uop_positions[uop]
      for range_pos, end_pos in loop_ranges.items():
        # If defined before/at loop start and used inside loop, extend to loop end
        if def_pos <= range_pos and range_pos < use_pos <= end_pos:
          if DEBUG_ILP >= 2 and uop.op is Ops.SHL:
            print(f"[ILP] Extending SHL@{def_pos} from {use_pos} to {end_pos}")
          last_use[uop] = max(last_use[uop], end_pos)
        # If defined inside loop and used after loop, ensure it survives past loop end
        # This handles loop-carried values that accumulate and are stored after the loop
        if range_pos < def_pos <= end_pos and use_pos > end_pos:
          last_use[uop] = max(last_use[uop], use_pos)
    max_pos = len(self.uops) - 1
    for u in self.uops:
      if u.op is Ops.SPECIAL: last_use[u] = max_pos
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

  def _get_live_interval(self, u: UOp) -> tuple[int, int]:
    uop_positions = {uop: i for i, uop in enumerate(self.uops)}
    def_pos = uop_positions.get(u, 0)
    root = self._get_root(u)
    death_pos = self._effective_death.get(root, self._last_use.get(u, def_pos))
    return (def_pos, death_pos)

  def _get_reg_requirements(self, u: UOp) -> tuple[str, int, int, list[tuple[int, int]]]:
    if u.op is Ops.DEFINE_GLOBAL: return ('sgpr', 2, 2, [])
    if u.op is Ops.DEFINE_VAR: return ('sgpr', 1, 1, [])
    if u.op is Ops.DEFINE_REG:
      num_regs = u.dtype.size if hasattr(u.dtype, 'size') and u.dtype.size > 0 else 16
      return ('vgpr', num_regs, 1, [])  # align=1 to reduce fragmentation
    if u.op is Ops.DEFINE_LOCAL: return ('none', 0, 1, [])
    if u.op is Ops.CONST:
      # Most CONSTs are inlined in instructions. Only allocate a register for:
      # 1. 64-bit types (can't be inlined)
      # 2. CONSTs used as STORE data operand (must be in VGPR for global_store)
      # The consts_needing_regs set is populated in _solve_ilp before calling this
      val = u.arg
      if u.dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong): return ('vgpr', 2, 2, [])
      if u.dtype == dtypes.float64: return ('vgpr', 2, 2, [])
      # All other CONSTs are assumed inline - only override if in consts_needing_regs set
      return ('none', 0, 1, [])  # Inline constant (may be overridden in _solve_ilp)
    if u.op is Ops.RANGE: return ('vgpr', 1, 1, [])
    if u.op is Ops.SPECIAL: return ('vgpr', 1, 1, [])
    # WMMA writes to the C input (accumulator) in-place, so no new register allocation needed
    if u.op is Ops.WMMA: return ('none', 0, 1, [])
    if u.op is Ops.VECTORIZE:
      count = len(u.src)
      scalar_dtype = u.dtype.scalar()
      # Use align=1 for VECTORIZE to reduce fragmentation (WMMA can handle any alignment)
      if scalar_dtype.itemsize == 2: return ('vgpr', (count + 1) // 2, 1, [(1, 1)] * (count // 2))
      elif scalar_dtype.itemsize == 1: return ('vgpr', (count + 3) // 4, 1, [(1, 1)] * max(0, count - (count + 3) // 4))
      return ('vgpr', count, 1, [])
    if u.op is Ops.LOAD:
      # LOAD from REG buffer is an alias, not a new register allocation
      if len(u.src) > 0 and u.src[0].op is Ops.INDEX and len(u.src[0].src) > 0:
        buf = u.src[0].src[0]
        if isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.REG:
          return ('none', 0, 1, [])  # Alias to the DEFINE_REG buffer
      # Check if conditional LOAD (INDEX has 3+ sources where 3rd is condition)
      # Conditional loads need an extra temp register for clamped_addr
      temps = []
      if len(u.src) > 0 and u.src[0].op is Ops.INDEX and len(u.src[0].src) > 2:
        temps = [(1, 1)]  # Extra temp for clamped_addr
      # Use align=1 for pairs to reduce fragmentation (hardware doesn't require alignment for most ops)
      if self._needs_vgpr_pair(u.dtype): return ('vgpr', 2, 1, temps)
      if hasattr(u.dtype, 'itemsize') and u.dtype.itemsize == 16: return ('vgpr', 4, 1, temps)
      return ('vgpr', 1, 1, temps)
    if u.op is Ops.INDEX:
      # INDEX needs a register if the offset is a constant (will be loaded into VGPR)
      # and it's pointing to global memory (not REG or LOCAL which handle offsets differently)
      if len(u.src) > 1:
        buf, idx = u.src[0], u.src[1]
        # Skip REG and LOCAL address spaces - they don't need VGPRs for constant offsets
        if isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace in (AddrSpace.REG, AddrSpace.LOCAL):
          return ('none', 0, 1, [])
        # For global memory with constant offset, need a VGPR
        if idx.op is Ops.CONST:
          return ('vgpr', 1, 1, [])
      return ('none', 0, 1, [])
    if u.op is Ops.IDIV:
      if u.dtype in (dtypes.int64, dtypes.uint64): return ('vgpr', 2, 2, [(8, 2)])
      elif u.dtype in (dtypes.int32, dtypes.int16, dtypes.int8): return ('vgpr', 1, 1, [(1, 1)] * 8)
      else: return ('vgpr', 1, 1, [(1, 1)] * 4)
    if u.op is Ops.MOD:
      if u.dtype in (dtypes.int32, dtypes.int16, dtypes.int8): return ('vgpr', 1, 1, [(1, 1)] * 5)
      else: return ('vgpr', 1, 1, [(1, 1)] * 6)
    if u.op is Ops.MUL and u.dtype in (dtypes.int64, dtypes.uint64):
      if len(u.src) >= 2:
        a_uop, b_uop = u.src[0], u.src[1]
        a_is_signed_cast = a_uop.op is Ops.CAST and a_uop.src[0].dtype == dtypes.int32
        b_is_const_hibit = b_uop.op is Ops.CONST and isinstance(b_uop.arg, int) and (b_uop.arg & 0x80000000) != 0
        if u.dtype == dtypes.int64 and a_is_signed_cast and b_is_const_hibit:
          return ('vgpr', 2, 2, [(1, 1)])
      return ('vgpr', 2, 2, [])
    if u.op is Ops.CAST:
      # CAST from register-space LOAD reuses the accumulator register (aliased)
      if (u.src[0].op is Ops.LOAD and len(u.src[0].src) > 0 and u.src[0].src[0].op is Ops.INDEX and
          len(u.src[0].src[0].src) > 0 and isinstance(u.src[0].src[0].src[0].dtype, PtrDType) and
          u.src[0].src[0].src[0].dtype.addrspace == AddrSpace.REG):
        return ('none', 0, 1, [])  # Aliased to accumulator
      if self._needs_vgpr_pair(u.dtype): return ('vgpr', 2, 2, [])
      return ('vgpr', 1, 1, [])
    if u.op in {Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR,
                Ops.MAX, Ops.MULACC, Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2,
                Ops.TRUNC, Ops.NEG, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.WHERE}:
      if self._needs_vgpr_pair(u.dtype): return ('vgpr', 2, 2, [])
      return ('vgpr', 1, 1, [])
    if u.op is Ops.GEP:
      src_dtype = u.src[0].dtype if u.src else None
      if src_dtype and hasattr(src_dtype, 'scalar'):
        if src_dtype.scalar().itemsize in (1, 2):
          idx = u.arg[0] if isinstance(u.arg, tuple) else u.arg
          if (src_dtype.scalar().itemsize == 2 and idx % 2 == 1) or \
             (src_dtype.scalar().itemsize == 1 and idx % 4 != 0):
            return ('vgpr', 1, 1, [])
      return ('none', 0, 1, [])
    if u.op is Ops.STORE:
      if len(u.src) > 0 and u.src[0].op is Ops.INDEX and len(u.src[0].src) > 2:
        return ('none', 0, 1, [(1, 1)])
      return ('none', 0, 1, [])
    return ('none', 0, 1, [])

  def _needs_vgpr_pair(self, dtype: DType) -> bool:
    return dtype in (dtypes.float64, dtypes.long, dtypes.ulong, dtypes.int64, dtypes.uint64) or \
           (hasattr(dtype, 'itemsize') and dtype.itemsize == 8)

  def _solve_ilp(self):
    # Pre-compute CONSTs that need registers due to usage context (e.g., STORE data operand)
    consts_needing_regs: set[UOp] = set()
    for u in self.uops:
      # STORE data operand must be in a VGPR, not an inline literal
      if u.op is Ops.STORE and len(u.src) > 1:
        val = u.src[1]
        if val.op is Ops.CONST:
          consts_needing_regs.add(val)

    vgpr_requests: list[tuple[UOp | TempReg, int, int, int, int]] = []
    sgpr_requests: list[tuple[UOp | TempReg, int, int, int, int]] = []
    for i, u in enumerate(self.uops):
      reg_type, num_regs, align, temps = self._get_reg_requirements(u)
      # Override for CONSTs that need registers due to usage
      if u.op is Ops.CONST and u in consts_needing_regs and reg_type == 'none':
        itemsize = u.dtype.itemsize if hasattr(u.dtype, 'itemsize') else 4
        if itemsize == 8:
          reg_type, num_regs, align = 'vgpr', 2, 2
        else:
          reg_type, num_regs, align = 'vgpr', 1, 1
      if reg_type == 'none' and not temps: continue
      def_pos, death_pos = self._get_live_interval(u)
      if DEBUG_ILP >= 2 and u.op is Ops.SHL and death_pos - def_pos > 500:
        root = self._get_root(u)
        # Find what uses this SHL at its last_use position
        last_use_pos = self._last_use.get(u, -1)
        user_at_last = None
        for j, uu in enumerate(self.uops):
          if j == last_use_pos:
            for src in uu.src:
              if src == u: user_at_last = uu.op.name
        print(f"[ILP] Long SHL@{def_pos}: death={death_pos} (lifetime={death_pos-def_pos}), root={root.op.name}@{self.uops.index(root) if root in self.uops else '?'}, last_use={last_use_pos} by {user_at_last}")
      if reg_type == 'vgpr' and num_regs > 0:
        vgpr_requests.append((u, def_pos, death_pos, num_regs, align))
        self._vgpr_sizes[u] = num_regs
      elif reg_type == 'sgpr' and num_regs > 0:
        sgpr_requests.append((u, def_pos, death_pos, num_regs, align))
        self._sgpr_sizes[u] = num_regs
      for temp_idx, (temp_count, temp_align) in enumerate(temps):
        temp_reg = TempReg(parent=u, index=temp_idx, count=temp_count, align=temp_align)
        self._temp_reg_map[(u, temp_idx)] = temp_reg
        self._temp_alloc_order[u].append(temp_reg)
        vgpr_requests.append((temp_reg, i, i, temp_count, temp_align))
        self._vgpr_sizes[temp_reg] = temp_count
    # Reserve v0 for packed workitem IDs (.amdhsa_system_vgpr_workitem_id 2)
    # v1-v2 are free (not used by RDNA3 ABI when using packed workitem IDs)
    # Reserve s0-s4: s[0:1] kernarg ptr, s[2:4] group IDs
    self._vgpr_assignment = self._solve_register_class(vgpr_requests, self.MAX_VGPR, reserved={0})
    self._sgpr_assignment = self._solve_register_class(sgpr_requests, self.MAX_SGPR, reserved={0, 1, 2, 3, 4})

  def _solve_register_class(self, requests: list[tuple[UOp | TempReg, int, int, int, int]], max_regs: int,
                            reserved: set[int]) -> dict[UOp | TempReg, int]:
    if not requests: return {}

    model = cp_model.CpModel()
    n = len(requests)

    reg_vars: list[cp_model.IntVar] = []
    time_intervals: list[cp_model.IntervalVar] = []
    reg_intervals: list[cp_model.IntervalVar] = []

    for i, (item, def_pos, death_pos, num_regs, align) in enumerate(requests):
      # Build valid domain (respects alignment and reserved registers)
      valid_starts = [r for r in range(max_regs - num_regs + 1)
                      if (align <= 1 or r % align == 0)
                      and not any(r + j in reserved for j in range(num_regs))]
      assert valid_starts, f"No valid register assignments for request {i}: {item}"

      # Create register start variable with restricted domain
      reg = model.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_starts), f'reg_{i}')
      reg_vars.append(reg)

      # Time interval (fixed start and size)
      duration = max(1, death_pos - def_pos + 1)
      time_int = model.NewFixedSizeIntervalVar(def_pos, duration, f'time_{i}')
      time_intervals.append(time_int)

      # Register interval (variable start, fixed size)
      reg_end = model.NewIntVar(0, max_regs, f'reg_end_{i}')
      model.Add(reg_end == reg + num_regs)
      reg_int = model.NewIntervalVar(reg, num_regs, reg_end, f'regint_{i}')
      reg_intervals.append(reg_int)

    # Single constraint handles ALL interference
    model.AddNoOverlap2D(time_intervals, reg_intervals)

    # Minimize max register used
    max_reg = model.NewIntVar(0, max_regs, 'max_reg')
    for i, (item, _, _, num_regs, _) in enumerate(requests):
      model.Add(max_reg >= reg_vars[i] + num_regs)
    model.Minimize(max_reg)

    # Solve with timeout (longer for large problems)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0  # Increase timeout for complex problems
    status = solver.Solve(model)

    if DEBUG_ILP:
      # Count alignment requirements
      align_counts = {}
      size_counts = {}
      for item, def_pos, death_pos, num_regs, align in requests:
        align_counts[align] = align_counts.get(align, 0) + 1
        size_counts[num_regs] = size_counts.get(num_regs, 0) + 1
      print(f"[ILP] Solver status: {solver.StatusName(status)} for {n} requests, aligns: {align_counts}, sizes: {size_counts}")

    # If solver fails (timeout, infeasible, etc.), print debug info and raise error
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      # Calculate max live registers at any point using sweep-line
      # Key insight: death_pos is when register is last used. At death_pos:
      # 1. The consumer reads the value
      # 2. The register can be freed immediately after reading
      # 3. The consumer allocates its output registers
      # So deaths should happen BEFORE births at the same position.
      # We achieve this by having deaths at (pos, 0) and births at (pos, 1).
      events = []
      for item, def_pos, death_pos, num_regs, align in requests:
        events.append((def_pos, 1, num_regs))        # birth at def_pos (type=1 for birth)
        events.append((death_pos, 0, -num_regs))     # death AT death_pos (type=0 for death)
      events.sort()  # sorts by (pos, type, delta) - deaths (type=0) before births (type=1)
      live = 0
      max_live = 0
      for pos, typ, delta in events:
        live += delta
        if live > max_live: max_live = live
      # Count by op type
      from collections import Counter
      op_counts = Counter()
      op_lifetimes: dict[str, list[int]] = {}
      for item, def_pos, death_pos, num_regs, _ in requests:
        op_name = item.op.name if isinstance(item, UOp) else f"TempReg({item.parent.op.name})"
        op_counts[op_name] += num_regs
        if op_name not in op_lifetimes: op_lifetimes[op_name] = []
        op_lifetimes[op_name].append(death_pos - def_pos)
      # Show avg lifetimes for high-count ops
      lifetime_info = {k: f"avg={sum(v)/len(v):.1f}, max={max(v)}" for k, v in op_lifetimes.items() if len(v) > 10}
      # Find the position of max live and verify count
      events_sorted = sorted(events)  # already sorted correctly
      live = 0
      peak_pos = 0
      peak_live = 0
      for pos, typ, delta in events_sorted:
        live += delta
        if live > peak_live:
          peak_live = live
          peak_pos = pos
      # Show requests around the peak - count manually
      peak_requests = [(item, def_pos, death_pos, num_regs) for item, def_pos, death_pos, num_regs, _ in requests
                       if def_pos <= peak_pos <= death_pos]
      peak_total = sum(num_regs for _, _, _, num_regs in peak_requests)
      peak_by_op = Counter()
      for item, _, _, num_regs in peak_requests:
        peak_by_op[item.op.name if isinstance(item, UOp) else f"TempReg({item.parent.op.name})"] += num_regs
      raise RuntimeError(f"[ILP] kernel requires {max_live} (sweep) / {peak_total} (manual) VGPRs at position {peak_pos}. "
                         f"Usage: {dict(op_counts)}\nAt peak: {dict(peak_by_op)}\nLifetimes: {lifetime_info}")

    result = {requests[i][0]: solver.Value(reg_vars[i]) for i in range(n)}

    if DEBUG_ILP:
      max_reg_used = solver.Value(max_reg)
      print(f"[ILP] {n} requests -> {max_reg_used} registers (status: {solver.StatusName(status)})")
      if DEBUG_ILP >= 2:
        for i, (item, def_pos, death_pos, num_regs, align) in enumerate(requests):
          reg = solver.Value(reg_vars[i])
          item_str = f"{item.op.name}" if isinstance(item, UOp) else f"TempReg({item.parent.op.name}, {item.index})"
          print(f"  [{def_pos:3d}-{death_pos:3d}] v{reg:3d}-v{reg+num_regs-1:3d} ({num_regs:2d} regs, align={align}) <- {item_str}")

    return result

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

  # === Public interface ===
  def free_dead_regs(self, pos: int):
    """Free registers scheduled to die at position pos."""
    self._current_pos = pos
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
    # First call for this owner: use ILP-assigned register if available
    if owner not in self._vgpr_allocated and owner in self._vgpr_assignment:
      self._vgpr_allocated.add(owner)
      reg = self._vgpr_assignment[owner]
      self._vgpr_owner[reg] = owner
      return VGPR(reg)
    # Subsequent calls or no ILP assignment: try temp registers, then greedy
    if owner in self._temp_alloc_order and self._temp_alloc_order[owner]:
      idx = self._temp_alloc_idx.get(owner, 0)
      if idx < len(self._temp_alloc_order[owner]):
        temp_reg = self._temp_alloc_order[owner][idx]
        self._temp_alloc_idx[owner] = idx + 1
        if temp_reg in self._vgpr_assignment:
          reg = self._vgpr_assignment[temp_reg]
          self._vgpr_owner[reg] = owner
          return VGPR(reg)
    return self._alloc_vgpr_greedy(owner)

  def _alloc_vgpr_greedy(self, owner: UOp) -> VGPR:
    if self._free_vgprs: reg = self._free_vgprs.pop()
    elif self._free_vgpr_ranges:
      base, count = self._free_vgpr_ranges.pop()
      reg = base
      if count > 1: self._free_vgpr_ranges.append((base + 1, count - 1))
    else:
      reg = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    if reg >= self.MAX_VGPR:
      raise RuntimeError(f"VGPR allocation exceeded maximum {self.MAX_VGPR} registers (greedy alloc for {owner.op.name if owner is not None else 'temp'})")
    if DEBUG_ILP >= 3:
      print(f"[ILP GREEDY] v{reg} <- {owner.op.name if owner is not None else 'temp'}")
    self._vgpr_owner[reg] = owner
    if owner is not None:
      self._schedule_vgpr_death(reg, owner)
    return VGPR(reg)

  def alloc_vgpr_pair(self, owner: UOp) -> VGPR:
    # First call for this owner: use ILP-assigned register if available
    if owner not in self._vgpr_allocated and owner in self._vgpr_assignment:
      self._vgpr_allocated.add(owner)
      reg = self._vgpr_assignment[owner]
      self._vgpr_owner[reg] = owner
      self._vgpr_owner[reg + 1] = owner
      self._vgpr_pairs.add(reg)
      self._vgpr_pairs.add(reg + 1)
      return VGPR(reg, 2)
    # Greedy fallback - try free pairs first
    if self._free_vgpr_pairs:
      reg = self._free_vgpr_pairs.pop()
    else:
      if self._next_vgpr % 2 != 0: self._next_vgpr += 1
      reg = self._next_vgpr
      self._next_vgpr += 2
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    if reg + 1 >= self.MAX_VGPR:
      raise RuntimeError(f"VGPR pair allocation exceeded maximum {self.MAX_VGPR} registers (greedy alloc for {owner.op.name if owner is not None else 'temp'})")
    self._vgpr_owner[reg] = owner
    self._vgpr_owner[reg + 1] = owner
    self._vgpr_pairs.add(reg)
    self._vgpr_pairs.add(reg + 1)
    if owner is not None:
      self._schedule_vgpr_death(reg, owner)
      self._schedule_vgpr_death(reg + 1, owner)
    return VGPR(reg, 2)

  def alloc_vgpr_range(self, owner: UOp, count: int = 8, align: int = 2) -> VGPR:
    # First call for this owner: use ILP-assigned register if available
    if owner not in self._vgpr_allocated and owner in self._vgpr_assignment:
      self._vgpr_allocated.add(owner)
      base = self._vgpr_assignment[owner]
      self._range_owner[base] = owner
      self._vgpr_ranges[base] = count
      for i in range(count): self._vgpr_owner[base + i] = owner
      return VGPR(base, count)
    # Greedy fallback - try free ranges first
    for i, (range_base, range_count) in enumerate(self._free_vgpr_ranges):
      if range_count >= count:
        self._free_vgpr_ranges.pop(i)
        if range_count > count: self._free_vgpr_ranges.append((range_base + count, range_count - count))
        self._range_owner[range_base] = owner
        self._vgpr_ranges[range_base] = count
        if owner is not None:
          self._schedule_range_death(range_base, owner)
        return VGPR(range_base, count)
    # Allocate new range
    if self._next_vgpr % 2 != 0: self._next_vgpr += 1
    base = self._next_vgpr
    self._next_vgpr += count
    self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    if base + count > self.MAX_VGPR:
      raise RuntimeError(f"VGPR range allocation exceeded maximum {self.MAX_VGPR} registers (greedy alloc {count} for {owner.op.name if owner is not None else 'temp'})")
    self._range_owner[base] = owner
    self._vgpr_ranges[base] = count
    if owner is not None:
      self._schedule_range_death(base, owner)
    return VGPR(base, count)

  def alloc_sgpr(self, owner: UOp) -> SGPR | None:
    # First call for this owner: use ILP-assigned register if available
    if owner not in self._sgpr_allocated and owner in self._sgpr_assignment:
      self._sgpr_allocated.add(owner)
      reg = self._sgpr_assignment[owner]
      self._sgpr_owner[reg] = owner
      return SGPR(reg)
    # Greedy fallback for subsequent calls
    if self._free_sgprs: reg = self._free_sgprs.pop()
    elif self._next_sgpr < self.MAX_SGPR:
      reg = self._next_sgpr
      self._next_sgpr += 1
      self._max_sgpr = max(self._max_sgpr, self._next_sgpr)
    else: return None
    self._sgpr_owner[reg] = owner
    if owner is not None:
      self._schedule_sgpr_death(reg, owner)
    return SGPR(reg)

  def alloc_sgpr_pair(self, owner: UOp) -> SGPR:
    # First call for this owner: use ILP-assigned register if available
    if owner not in self._sgpr_allocated and owner in self._sgpr_assignment:
      self._sgpr_allocated.add(owner)
      reg = self._sgpr_assignment[owner]
      self._sgpr_owner[reg] = owner
      self._sgpr_owner[reg + 1] = owner
      self._sgpr_pairs.add(reg)
      self._sgpr_pairs.add(reg + 1)
      return SGPR(reg, 2)
    # Greedy fallback for subsequent calls
    if self._next_sgpr % 2 != 0: self._next_sgpr += 1
    reg = self._next_sgpr
    self._next_sgpr += 2
    self._max_sgpr = max(self._max_sgpr, self._next_sgpr)
    self._sgpr_owner[reg] = owner
    self._sgpr_owner[reg + 1] = owner
    self._sgpr_pairs.add(reg)
    self._sgpr_pairs.add(reg + 1)
    # Note: SGPR pairs for buffer addresses typically live for the whole kernel, no death scheduling needed
    return SGPR(reg, 2)

  def get_scratch_vgpr(self, count: int = 1) -> int:
    if self._scratch_vgpr < 0:
      self._scratch_vgpr = self._next_vgpr
      alloc_count = max(count, 32)
      self._next_vgpr += alloc_count
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      if self._scratch_vgpr + alloc_count > self.MAX_VGPR:
        raise RuntimeError(f"Scratch VGPR allocation exceeded maximum {self.MAX_VGPR} registers")
    return self._scratch_vgpr

  def get_deferred_store_vgpr(self) -> str:
    if self._deferred_store_vgpr < 0:
      self._deferred_store_vgpr = self._next_vgpr
      self._next_vgpr += 1
      self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
      if self._deferred_store_vgpr >= self.MAX_VGPR:
        raise RuntimeError(f"Deferred store VGPR allocation exceeded maximum {self.MAX_VGPR} registers")
    return f"v{self._deferred_store_vgpr}"

  def get_temp_vgpr(self) -> VGPR:
    if self._free_vgprs: return VGPR(self._free_vgprs.pop())
    reg = self._next_vgpr
    self._next_vgpr += 1
    self._max_vgpr = max(self._max_vgpr, self._next_vgpr)
    if reg >= self.MAX_VGPR:
      raise RuntimeError(f"Temp VGPR allocation exceeded maximum {self.MAX_VGPR} registers")
    return VGPR(reg)

  def return_temp_vgpr(self, reg: VGPR): self._free_vgprs.append(reg.idx)
  def cancel_vgpr_death(self, reg: int): pass
  def reschedule_vgpr_death(self, reg: int, new_owner: UOp): self._vgpr_owner[reg] = new_owner
  def schedule_v0_free(self, pos: int): pass
  def extend_lifetime(self, uop: UOp, pos: int): pass
  def get_last_use(self, uop: UOp) -> int: return self._last_use.get(uop, -1)
  def is_vgpr_owner(self, reg: int) -> bool: return reg in self._vgpr_owner
  def get_vgpr_owner(self, reg: int) -> UOp | None: return self._vgpr_owner.get(reg)
  def free_vgpr(self, reg: int):
    if reg in self._vgpr_owner:
      del self._vgpr_owner[reg]
      self._free_vgprs.append(reg)

  @property
  def max_vgpr(self) -> int: return self._max_vgpr
  @property
  def max_sgpr(self) -> int: return self._max_sgpr

  def finalize(self):
    """Check final register counts - ILP pre-validates during solve, so this is mostly a no-op."""
    if self._max_vgpr > self.MAX_VGPR:
      raise RuntimeError(f"VGPR overflow: allocated up to v{self._max_vgpr-1}, max v{self.MAX_VGPR-1}")
    if self._max_sgpr > self.MAX_SGPR:
      raise RuntimeError(f"SGPR overflow: allocated up to s{self._max_sgpr-1}, max s{self.MAX_SGPR-1}")

  @staticmethod
  def needs_vgpr_pair(dtype: DType) -> bool:
    return dtype in (dtypes.float64, dtypes.long, dtypes.ulong, dtypes.int64, dtypes.uint64) or \
           (hasattr(dtype, 'itemsize') and dtype.itemsize == 8)
