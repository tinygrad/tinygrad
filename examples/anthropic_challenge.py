from tinygrad import Tensor, dtypes, Context, getenv, UOp, fetch
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, GroupOp
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps

# ************************* implementation of the problem ************************

def myhash(a: Tensor) -> Tensor:
  a = (a + 0x7ED55D16) + (a << 12)
  a = (a ^ 0xC761C23C) ^ (a >> 19)
  a = (a + 0x165667B1) + (a << 5)
  a = (a + 0xD3A2646C) ^ (a << 9)
  a = (a + 0xFD7046C5) + (a << 3)
  a = (a ^ 0xB55A4F09) ^ (a >> 16)
  return a

def select_with_where_tree(values: Tensor, relative_idx: Tensor, is_top: bool = True) -> Tensor:
  n = values.shape[0]
  if n == 1: return values[0].expand(relative_idx.shape)

  mid = n // 2
  left = select_with_where_tree(values[:mid], relative_idx, False)
  right = select_with_where_tree(values[mid:], relative_idx, False)

  shift = n.bit_length() - 2  # top bit position of [0, n-1]
  go_right = (relative_idx >> shift) if is_top else (relative_idx >> shift) & 1
  return go_right.where(right, left)

def tree_traversal(forest: Tensor, val: Tensor, height: int, rounds: int, where_tree_threshold=3) -> Tensor:
  # All walkers start at idx=0
  idx = Tensor.zeros(val.shape, device=val.device, dtype=dtypes.uint32)

  for r in range(rounds):
    level = r % (height + 1)
    level_start = (1 << level) - 1
    level_size = 1 << level

    if level == 0:
      # At root (level 0), all walkers are at idx=0
      # No gather needed, just broadcast the root value
      node_val = forest[0].expand(val.shape)
      idx = idx * 0  # Reset to 0
    elif level <= where_tree_threshold:
      # Small level: use where-tree
      level_values = forest[level_start : level_start + level_size]
      relative_idx = (idx - level_start)
      node_val = select_with_where_tree(level_values, relative_idx)
    else:
      # Large level: use gather
      node_val = forest.gather(0, idx)

    val = myhash(val ^ node_val)
    idx = (idx << 1) + (1 + (val & 1))

    # No wrap check needed! At round 10 (level becomes 0), we reset idx above.

  return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))

# ************************* renderer for VLIW machine *************************

# *** machine spec ***
SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledOp:
  """Wraps UOp for scheduling. Allows same UOp in multiple bundles with different element ranges (for split demotion)."""
  uop: UOp
  engine: str
  elements: tuple[int, int] | None = None  # None = all, (0,4) = lo, (4,8) = hi

def loop_unrolling(sink:UOp):
  rng = [x for x in sink.toposort() if x.op is Ops.RANGE]
  if len(rng) == 0: return None
  print(f"unrolling loop with size {rng[0].vmax+1}")
  unrolled_sinks = [sink.substitute({rng[0]:rng[0].const_like(i)}).src[0] for i in range(rng[0].vmax+1)]
  return UOp.sink(*unrolled_sinks, arg=sink.arg)

global_addrs = []

def make_mulacc(a, b, c):
  if a.dtype.count != 8: return None
  return UOp(Ops.MULACC, a.dtype, (a, b, c))

vliw_prepare = PatternMatcher([
  # loop unrolling (should be a part of tinygrad)
  (UPat(Ops.SINK, name="sink"), loop_unrolling),
  # cast is fake
  (UPat(Ops.CAST, name="c"), lambda c: c.src[0]),
  # rewrites to hardcode the addresses in memory
  (UPat(Ops.DEFINE_GLOBAL, name="dg"), lambda dg: UOp.const(dtypes.uint, global_addrs[dg.arg])),
  # INDEX is just plus
  (UPat(Ops.INDEX, name="i"), lambda i: i.src[0]+i.src[1]),
  # (a * b) + c → MULACC(a, b, c) for vectors only
  (UPat(Ops.ADD, src=[UPat(Ops.MUL, src=(UPat(name="a"), UPat(name="b"))), UPat(name="c")]), make_mulacc),
])+symbolic

def is_broadcast(u: UOp) -> bool: return all(s is u.src[0] for s in u.src)

class VLIWPacker:
  """
  Pack UOps into VLIW bundles using list scheduling.

  Key optimizations:
  1. Linear batch offset - staggers 32 batches to overlap load/compute phases
  2. Opportunistic demotion - when VALU is full (6 ops), demote vector ops to ALU as 8 scalar ops
  """
  # ops that can be demoted from VALU to ALU (have scalar equivalents)
  demotable_ops = {Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.ADD}

  def __init__(self, uops: list[UOp]):
    self.uops = uops
    self.n = len(uops)
    self.uop_to_idx = {u: i for i, u in enumerate(uops)}
    # deps[i] = uops that must complete before uop i can run
    self.deps = self._build_deps()
    # depth[i] = longest chain of deps leading to uop i (for priority)
    self.depth = self._compute_depth()
    self.batch_ids, self.offsets = self._assign_batches()

  def _build_deps(self) -> list[set[int]]:
    """Build dependency graph: deps[i] = set of uop indices that uop i reads from."""
    deps: list[set[int]] = [set() for _ in range(self.n)]
    for i, u in enumerate(self.uops):
      for src in u.src:
        if src in self.uop_to_idx:
          deps[i].add(self.uop_to_idx[src])
    return deps

  def _compute_depth(self) -> list[int]:
    """Compute critical path depth for each uop (used for scheduling priority)."""
    depth = [0] * self.n
    for i in range(self.n):
      if self.deps[i]:
        depth[i] = 1 + max(depth[d] for d in self.deps[i])
    return depth

  def _assign_batches(self) -> tuple[list[int | None], list[int]]:
    """Assign batch IDs and compute stagger offsets."""
    batch_ids: list[int | None] = [None] * self.n
    sink = self.uops[-1]
    assert sink.op is Ops.SINK
    batch_count = len(sink.src)

    if batch_count > 1:
      # visited_by[i] = which batch last visited uop i
      visited_by: list[int | None] = [None] * self.n
      def walk(u: UOp, bi: int):
        idx = self.uop_to_idx.get(u)
        assert idx is not None
        if visited_by[idx] == bi: return  # already visited by this batch
        visited_by[idx] = bi
        if batch_ids[idx] is None: batch_ids[idx] = bi
        elif batch_ids[idx] != bi: batch_ids[idx] = None  # shared across batches
        for s in u.src: walk(s, bi)
      for bi, root in enumerate(sink.src): walk(root, bi)

    # stagger batches evenly across the schedule
    schedule_length = max(self.depth) + 1 if self.depth else 1
    spacing = schedule_length // batch_count if batch_count > 0 else 1
    offsets = [(bi * spacing) % schedule_length for bi in range(batch_count)]
    return batch_ids, offsets

  def _priority(self, i: int) -> int:
    """Scheduling priority: lower = scheduled earlier."""
    bid = self.batch_ids[i]
    # shared uops (bid=None) get offset 0, so they're scheduled first
    return self.depth[i] + (self.offsets[bid] if bid is not None else 0)

  @staticmethod
  def _engine(u: UOp) -> str:
    """Return the primary engine for a UOp."""
    match u.op:
      case Ops.CONST | Ops.LOAD: return "load"
      case Ops.STORE: return "store"
      case Ops.VECTORIZE:
        # broadcast is valu, otherwise alu moves
        return "valu" if is_broadcast(u) else "alu"
      case Ops.MULACC: return "valu" if u.dtype.count == 8 else "alu"
      case Ops.WHERE: return "flow"
      case Ops.SINK: return "flow"
      case Ops.GROUP | Ops.GEP: return "none"  # pseudo-ops, no instruction emitted
      case _ if u.op in GroupOp.Binary: return "valu" if u.dtype.count > 1 else "alu"
      case _: raise NotImplementedError(f"unknown engine for {u.op}")

  @staticmethod
  def _slot_count(u: UOp, engine: str) -> int:
    """How many slots does this UOp use in the given engine?"""
    # one move per unique source
    if u.op is Ops.VECTORIZE and not is_broadcast(u): return len({id(s) for s in u.src})
    if u.op is Ops.MULACC and u.dtype.count == 1: return 2  # scalar MULACC needs MUL + ADD
    if engine == "alu" and u.dtype.count == 8: return 8  # demoted vector op costs 8 slots
    return 1

  def pack(self) -> list[list[ScheduledOp]]:
    bundles: list[list[ScheduledOp]] = []
    scheduled = [False] * self.n
    remaining = self.n
    pending_split: tuple[int, int] | None = None  # (uop_idx, bundle_idx where lo was emitted)

    while remaining > 0:
      ready = [i for i in range(self.n) if not scheduled[i] and all(scheduled[d] for d in self.deps[i])]
      if pending_split is not None:
        ready = [i for i in ready if i != pending_split[0]]
      if not ready and pending_split is None:
        raise RuntimeError("Deadlock in UOp scheduling")

      bundle: list[ScheduledOp] = []
      slots_used = {k: 0 for k in SLOT_LIMITS}
      scheduled_this_bundle: list[int] = []

      # complete any pending split demotion FIRST (hi half)
      if pending_split is not None and SLOT_LIMITS["alu"] - slots_used["alu"] >= 4:
        split_idx, _ = pending_split
        bundle.append(ScheduledOp(self.uops[split_idx], "alu", (4, 8)))
        slots_used["alu"] += 4
        scheduled_this_bundle.append(split_idx)  # NOW mark as scheduled (after both halves done)
        pending_split = None

      # group ready ops by engine and sort by priority
      ready_by_engine: dict[str, list[int]] = {k: [] for k in [*SLOT_LIMITS.keys(), "none"]}
      for i in ready:
        ready_by_engine[self._engine(self.uops[i])].append(i)
      for eng in ready_by_engine:
        ready_by_engine[eng].sort(key=self._priority)

      # pick ops for each engine
      for eng in ["load", "valu", "alu", "store", "flow"]:
        for i in ready_by_engine[eng][:]:
          slots = self._slot_count(self.uops[i], eng)
          if slots_used[eng] + slots <= SLOT_LIMITS[eng]:
            bundle.append(ScheduledOp(self.uops[i], eng, None))
            slots_used[eng] += slots
            scheduled_this_bundle.append(i)
            ready_by_engine[eng].remove(i)

      # handle pseudo-ops (GEP, GROUP) - no instruction emitted
      for i in ready_by_engine["none"][:]:
        bundle.append(ScheduledOp(self.uops[i], "none", None))
        scheduled_this_bundle.append(i)
        ready_by_engine["none"].remove(i)

      # opportunistic demotion: move VALU ops to spare ALU slots
      alu_spare = SLOT_LIMITS["alu"] - slots_used["alu"]
      for i in ready_by_engine["valu"][:]:
        u = self.uops[i]
        if u.op not in self.demotable_ops or u.dtype.count != 8: continue
        if alu_spare >= 8:  # full demotion: all 8 elements as scalar ops
          bundle.append(ScheduledOp(u, "alu", (0, 8)))
          slots_used["alu"] += 8
          alu_spare -= 8
          scheduled_this_bundle.append(i)
          ready_by_engine["valu"].remove(i)
        elif alu_spare >= 4 and pending_split is None:  # split demotion: lo half now, hi half later
          bundle.append(ScheduledOp(u, "alu", (0, 4)))
          slots_used["alu"] += 4
          alu_spare -= 4
          # NOT marked scheduled yet, wait until hi half is done
          ready_by_engine["valu"].remove(i)
          pending_split = (i, len(bundles))
          break

      # mark scheduled
      for i in scheduled_this_bundle:
        scheduled[i] = True
        remaining -= 1

      if bundle:
        bundles.append(bundle)

    return bundles

class RegisterAllocator:
  """Linear scan register allocator over scheduled bundles."""
  def __init__(self, bundles: list[list[ScheduledOp]]):
    self.r: dict[UOp, int] = {}
    self.zero_reg = 0
    self._next_reg = 1
    self._free_scalar: list[int] = []
    self._free_vector: list[int] = []

    last_use = self._compute_last_use(bundles)
    self._assign(bundles, last_use)

  def __getitem__(self, u: UOp) -> int: return self.r[u]
  @property
  def total(self) -> int: return self._next_reg

  def _compute_last_use(self, bundles: list[list[ScheduledOp]]) -> dict[UOp, int]:
    """For each UOp, find the last bundle where it's read. We can free its register after that."""
    last_use: dict[UOp, int] = {}
    for bi, bundle in enumerate(bundles):
      for sop in bundle:
        for src in sop.uop.src:
          # GEP aliases into parent, so track parent's lifetime, not GEP's (GEP has no register to free)
          last_use[src.src[0] if src.op is Ops.GEP else src] = bi
    return last_use

  def _alloc(self, u: UOp):
    if u.dtype.count == 8:
      self.r[u] = self._free_vector.pop() if self._free_vector else self._bump(8)
    else:
      self.r[u] = self._free_scalar.pop() if self._free_scalar else self._bump(1)

  def _bump(self, size: int) -> int:
    reg = self._next_reg
    self._next_reg += size
    return reg

  def _free(self, u: UOp):
    free_list = self._free_vector if u.dtype.count == 8 else self._free_scalar
    if self.r[u] not in free_list: free_list.append(self.r[u])

  def _assign(self, bundles: list[list[ScheduledOp]], last_use: dict[UOp, int]):
    """Linear scan: allocate outputs, free after last use."""
    for bi, bundle in enumerate(bundles):
      for sop in bundle:  # allocate (GEPs need parent first, so separate loop)
        u = sop.uop
        if u.op in {Ops.STORE, Ops.SINK, Ops.GEP} or u in self.r: continue
        self._alloc(u)
      for sop in bundle:  # set GEP aliases, then free
        if sop.uop.op is Ops.GEP: self.r[sop.uop] = self.r[sop.uop.src[0]] + sop.uop.arg[0]
        for src in sop.uop.src:
          if last_use.get(src) == bi and src in self.r: self._free(src)

class VLIWRenderer(Renderer):
  has_local = False  # TODO: this should be the default / cleaned up
  # this says this backend supports MULACC + more. decompositions uses this
  code_for_op: dict = {Ops.MULACC: None, Ops.ADD: "+", Ops.MUL: "*",
                       Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|",
                       Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
  # this matcher runs while still in graph form
  pre_matcher = vliw_prepare

  def render(self, uops:list[UOp]):
    print(f"rendering with {len(uops)} uops")

    packer = VLIWPacker(uops)
    bundles = packer.pack()
    alloc = RegisterAllocator(bundles)
    inst: list[dict[str, list]] = [{"load": [("const", alloc.zero_reg, 0)]}]

    for bundle in bundles:
      ops: dict[str, list] = {}
      def emit(engine, *args): ops.setdefault(engine, []).append(args)

      for sop in bundle:
        u = sop.uop
        assert u.dtype.count in (1,8), "dtype count must be 1 or 8"
        match u.op:
          case Ops.SINK: emit("flow", "halt")
          case Ops.CONST: emit("load", "const", alloc[u], u.arg)
          # GEP is just an alias into parent register, handled by allocator
          case Ops.GEP | Ops.GROUP: pass
          case Ops.VECTORIZE:
            # if all sources are the same, we can broadcast
            if is_broadcast(u): emit("valu", "vbroadcast", alloc[u], alloc[u.src[0]])
            else:
              # gather scalars into contiguous registers (skip if already in place)
              for j, s in enumerate(u.src):
                if alloc[s] != alloc[u]+j: emit("alu", "+", alloc[u]+j, alloc[s], alloc.zero_reg)
          case Ops.LOAD: emit("load", "vload" if u.dtype.count > 1 else "load", alloc[u], alloc[u.src[0]])
          case Ops.STORE: emit("store", "vstore" if u.src[1].dtype.count > 1 else "store", alloc[u.src[0]], alloc[u.src[1]])
          case Ops.MULACC:
            assert u.dtype.count == 8, "MULACC only created for vectors"
            emit("valu", "multiply_add", alloc[u], alloc[u.src[0]], alloc[u.src[1]], alloc[u.src[2]])
          case Ops.WHERE: emit("flow", "vselect", alloc[u], alloc[u.src[0]], alloc[u.src[1]], alloc[u.src[2]])
          case _ if u.op in self.code_for_op:
            if sop.elements is not None:
              # demoted vector op: emit scalar ops for each element
              for i in range(sop.elements[0], sop.elements[1]):
                emit("alu", self.code_for_op[u.op], alloc[u]+i, alloc[u.src[0]]+i, alloc[u.src[1]]+i)
            elif sop.engine == "valu" or u.dtype.count == 1:
              emit(sop.engine, self.code_for_op[u.op], alloc[u], alloc[u.src[0]], alloc[u.src[1]])
            else: raise NotImplementedError(f"unexpected engine {sop.engine} for {u.op}")
          case _: raise NotImplementedError(f"unhandled op {u.op}")

      if ops: inst.append(ops)
    return repr(inst)

# ************************* test and render *************************

import sys, types
PROBLEM_URL = "https://raw.githubusercontent.com/anthropics/original_performance_takehome/refs/heads/main/tests/frozen_problem.py"
sys.modules["problem"] = problem = types.ModuleType("problem")
exec(fetch(PROBLEM_URL).read_text(), problem.__dict__)

if __name__ == "__main__":
  batch_size = getenv("BS", 256)
  height = 10
  rounds = getenv("ROUNDS", 16)

  # build problem
  tree = problem.Tree.generate(height)
  inp = problem.Input.generate(tree, batch_size, rounds)
  mem = problem.build_mem_image(tree, inp)
  global_addrs.extend([mem[6], mem[6], mem[4]])  # output, input, forest

  # *** verify the kernel in tinygrad compared to reference ***

  forest_t = Tensor(tree.values, dtype=dtypes.uint32)
  val_t = Tensor(inp.values, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    # verify on normal tinygrad device
    with Context(PCONTIG=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    problem.reference_kernel(tree, inp)
    assert val_out == inp.values
    print("verification passed")

  # *** render to device ***

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, SPEC=0):
    out = tree_traversal(forest_t, val_t, height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  # *** run on Machine and compare ***

  # NOTE: the scratch size needs to be reduced to 1536 when you have a register allocator
  src = eval(prg.src)
  max_regs = max(t[1] for instr in src for v in instr.values() for t in v if len(t) > 1) + 8
  print(f"{max_regs:5d} regs used" + ("" if max_regs <= 1536 else "       <-- WARNING: TOO MANY REGISTERS, MUST BE <= 1536"))
  machine = problem.Machine(mem, src, problem.DebugInfo(scratch_map={}), n_cores=1, trace=False, scratch_size=max_regs)
  machine.run()
  print(f"ran for {machine.cycle:5d} cycles" + ("" if machine.cycle <= 1363 else "  <-- EVEN CLAUDE GOT 1363"))

  # compare to reference
  ref_mem = mem.copy()
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
  print("compare passed!")
