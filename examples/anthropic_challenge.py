from tinygrad import Context, Tensor, UOp, dtypes, fetch, getenv
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import Ops, PatternMatcher, UPat

# ************************* implementation of the problem ************************

def myhash(a: Tensor) -> Tensor:
  a = (a + 0x7ED55D16) + (a << 12)
  a = (a ^ 0xC761C23C) ^ (a >> 19)
  a = (a + 0x165667B1) + (a << 5)
  a = (a + 0xD3A2646C) ^ (a << 9)
  a = (a + 0xFD7046C5) + (a << 3)
  a = (a ^ 0xB55A4F09) ^ (a >> 16)
  return a

def tree_traversal(forest: Tensor, val: Tensor, height: int, rounds: int) -> Tensor:
  idx = Tensor.zeros(val.shape, device=val.device, dtype=dtypes.uint32)

  for r in range(rounds):
    level = r % (height + 1)

    if level == 0:
      node_val = forest[0].expand(val.shape)
      idx = idx * 0
    elif level == 1:
      bit = idx - 1
      node_val = bit.where(forest[2].expand(val.shape), forest[1].expand(val.shape))
    elif level == 2:
      offset = idx - 3
      bit0 = offset & 1
      bit1 = offset >> 1
      left = bit0.where(forest[4].expand(val.shape), forest[3].expand(val.shape))
      right = bit0.where(forest[6].expand(val.shape), forest[5].expand(val.shape))
      node_val = bit1.where(right, left)
    elif level == 3:
      # Level 3: 8 nodes (indices 7-14), use 3-bit select to avoid gather
      offset = idx - 7
      bit0 = offset & 1
      bit1 = (offset >> 1) & 1
      bit2 = offset >> 2
      # First level: select pairs based on bit0
      v01 = bit0.where(forest[8].expand(val.shape), forest[7].expand(val.shape))
      v23 = bit0.where(forest[10].expand(val.shape), forest[9].expand(val.shape))
      v45 = bit0.where(forest[12].expand(val.shape), forest[11].expand(val.shape))
      v67 = bit0.where(forest[14].expand(val.shape), forest[13].expand(val.shape))
      # Second level: select based on bit1
      v0123 = bit1.where(v23, v01)
      v4567 = bit1.where(v67, v45)
      # Third level: select based on bit2
      node_val = bit2.where(v4567, v0123)
    else:
      node_val = forest.gather(0, idx)

    val = myhash(val ^ node_val)
    idx = (idx << 1) + (1 + (val & 1))

  return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))

# ************************* renderer for VLIW machine *************************

SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

def get_reads_writes(engine: str, slot: tuple) -> tuple[set[int], set[int]]:
  """Returns (reads, writes) sets of scratch addresses for dependency tracking."""
  reads, writes = set(), set()
  op = slot[0]

  if op == "const":
    _, dest, val = slot
    writes.add(dest)
  elif op == "load":
    _, dest, addr = slot
    writes.add(dest)
    reads.add(addr)
  elif op == "vload":
    _, dest, addr = slot
    writes.update(range(dest, dest + 8))
    reads.add(addr)
  elif op == "store":
    _, addr, src = slot
    reads.add(addr)
    reads.add(src)
  elif op == "vstore":
    _, addr, src = slot
    reads.add(addr)
    reads.update(range(src, src + 8))
  elif op == "vbroadcast":
    _, dest, src = slot
    writes.update(range(dest, dest + 8))
    reads.add(src)
  elif op == "multiply_add":
    _, dest, a, b, c = slot
    writes.update(range(dest, dest + 8))
    reads.update(range(a, a + 8))
    reads.update(range(b, b + 8))
    reads.update(range(c, c + 8))
  elif op == "vselect":
    _, dest, cond, a, b = slot
    writes.update(range(dest, dest + 8))
    reads.update(range(cond, cond + 8))
    reads.update(range(a, a + 8))
    reads.update(range(b, b + 8))
  elif op == "halt":
    pass
  elif op in {"+", "*", "^", "&", "|", "<<", ">>", "<"}:
    _, dest, a, b = slot
    if engine == "valu":
      writes.update(range(dest, dest + 8))
      reads.update(range(a, a + 8))
      reads.update(range(b, b + 8))
    else:
      writes.add(dest)
      reads.add(a)
      reads.add(b)
  return reads, writes

# ************************* UOp preparation *************************

def loop_unrolling(sink: UOp):
  rng = [x for x in sink.toposort() if x.op is Ops.RANGE]
  if len(rng) == 0: return None
  print(f"unrolling loop with size {rng[0].vmax+1}")
  unrolled_sinks = [sink.substitute({rng[0]: rng[0].const_like(i)}).src[0] for i in range(rng[0].vmax+1)]
  return UOp.sink(*unrolled_sinks, arg=sink.arg)

global_addrs = []

def _make_mulacc(a, b, c):
  if a.dtype.count != 8: return None
  return UOp(Ops.MULACC, a.dtype, (a, b, c))

vliw_prepare = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), loop_unrolling),
  (UPat(Ops.CAST, name="c"), lambda c: c.src[0]),
  (UPat(Ops.DEFINE_GLOBAL, name="dg"), lambda dg: UOp.const(dtypes.uint, global_addrs[dg.arg])),
  (UPat(Ops.INDEX, name="i"), lambda i: i.src[0]+i.src[1]),
  # (a * b) + c → MULACC(a, b, c) for vectors only
  (UPat(Ops.ADD, src=[UPat(Ops.MUL, src=(UPat(name="a"), UPat(name="b"))), UPat(name="c")]), _make_mulacc),
])

# ************************* engine assignment *************************

def uop_to_engine(u: UOp, code_for_op: dict) -> str:
  """Return the primary engine for a UOp."""
  match u.op:
    case Ops.CONST | Ops.LOAD: return "load"
    case Ops.STORE: return "store"
    case Ops.VECTORIZE:
      return "valu" if all(s is u.src[0] for s in u.src) else "alu"
    case Ops.MULACC:
      return "valu" if u.dtype.count == 8 else "alu"
    case Ops.WHERE: return "flow"
    case Ops.SINK: return "flow"
    case Ops.GROUP | Ops.GEP: return "none"
    case _ if u.op in code_for_op:
      return "valu" if u.dtype.count > 1 else "alu"
    case _: return "unknown"

def uop_slot_count(u: UOp, engine: str) -> int:
  """How many slots does this UOp use in the given engine?"""
  if u.op is Ops.VECTORIZE and not all(s is u.src[0] for s in u.src):
    return len([s for i, s in enumerate(u.src) if i == 0 or s is not u.src[0]])
  if u.op is Ops.MULACC and u.dtype.count == 1:
    return 2  # scalar MULACC needs MUL + ADD
  # Demoted vector op on ALU costs 8 slots (one per element)
  if engine == "alu" and u.dtype.count == 8:
    return 8
  return 1

# ************************* VLIW packer with opportunistic demotion *************************

def pack_uops_vliw(uops: list[UOp], code_for_op: dict) -> tuple[list[tuple[list[UOp], dict[UOp, str]]], dict]:
  """
  Pack UOps into VLIW bundles using list scheduling.

  Key optimizations:
  1. Linear batch offset - staggers 32 batches to overlap load/compute phases
  2. Opportunistic demotion - when VALU is full (6 ops), demote one op to ALU (8 scalar ops)
  """
  n = len(uops)
  uop_to_idx = {u: i for i, u in enumerate(uops)}

  # Build dependency graph
  deps: list[set[int]] = [set() for _ in range(n)]
  for i, u in enumerate(uops):
    for src in u.src:
      if src in uop_to_idx:
        deps[i].add(uop_to_idx[src])

  # Compute depth for each op (for priority scheduling)
  depth = [0] * n
  for i in range(n):
    if deps[i]:
      depth[i] = 1 + max(depth[d] for d in deps[i])

  # Assign batch IDs from the SINK sources (each source is one batch)
  batch_id = [-1] * n
  sink = next((u for u in uops if u.op is Ops.SINK), None)
  batch_count = len(sink.src) if sink is not None else 0
  if sink is not None and batch_count > 1:
    visit = [0] * n
    for bi, root in enumerate(sink.src):
      stack = [root]
      stamp = bi + 1
      while stack:
        u = stack.pop()
        idx = uop_to_idx.get(u)
        if idx is None or visit[idx] == stamp:
          continue
        visit[idx] = stamp
        if batch_id[idx] == -1:
          batch_id[idx] = bi
        elif batch_id[idx] != bi:
          batch_id[idx] = -1  # shared across batches
        for s in u.src:
          stack.append(s)

  # Linear batch offset strategy: stagger batches evenly across the schedule
  horizon = max(depth) + 1 if depth else 1
  spacing = horizon // batch_count if batch_count > 0 else 1
  offsets = [(bi * spacing) % horizon for bi in range(batch_count)]

  def time_tag(i: int) -> int:
    """Priority for scheduling: earlier depth + batch offset = higher priority."""
    bid = batch_id[i]
    return depth[i] + (offsets[bid] if bid >= 0 else 0)

  # Ops that can be demoted from VALU to ALU (have scalar equivalents)
  demotable_ops = {Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.ADD}

  bundles: list[tuple[list[UOp], dict[UOp, str]]] = []
  scheduled = [False] * n
  remaining = n
  cycle = 0

  # Split demotion tracking:
  # - pending_split: (uop_idx, start_bundle_idx) - op currently being split
  # - split_lo: dict[bundle_idx] = uop - bundles that emit first half (elements 0-3)
  # - split_hi: dict[bundle_idx] = uop - bundles that emit second half (elements 4-7)
  pending_split: tuple[int, int] | None = None
  split_lo: dict[int, UOp] = {}
  split_hi: dict[int, UOp] = {}

  while remaining > 0:
    # Find all ready ops (dependencies satisfied, excluding pending split)
    ready = [i for i in range(n) if not scheduled[i] and all(scheduled[d] for d in deps[i])]
    if pending_split is not None:
      ready = [i for i in ready if i != pending_split[0]]

    if not ready and pending_split is None:
      raise RuntimeError("Deadlock in UOp scheduling")

    bundle: list[int] = []
    engine_choices: dict[int, str] = {}
    slots_used = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0}

    # Complete any pending split demotion FIRST
    alu_spare = SLOT_LIMITS["alu"]
    if pending_split is not None:
      split_idx, _ = pending_split
      if alu_spare >= 4:
        bundle.append(split_idx)
        engine_choices[split_idx] = "alu_hi"
        split_hi[len(bundles)] = uops[split_idx]
        slots_used["alu"] += 4
        alu_spare -= 4
        pending_split = None

    # Group ready ops by their engine
    ready_by_engine: dict[str, list[int]] = {"valu": [], "load": [], "alu": [], "store": [], "flow": [], "none": [], "unknown": []}
    for i in ready:
      engine = uop_to_engine(uops[i], code_for_op)
      ready_by_engine[engine].append(i)

    # Sort each engine's ready list by priority
    for eng in ready_by_engine:
      ready_by_engine[eng].sort(key=time_tag)

    def pick(engine: str) -> bool:
      """Try to pick one op from the engine's ready queue."""
      for i in ready_by_engine[engine][:]:
        slots_needed = uop_slot_count(uops[i], engine)
        if slots_used[engine] + slots_needed <= SLOT_LIMITS.get(engine, 0):
          bundle.append(i)
          engine_choices[i] = engine
          slots_used[engine] += slots_needed
          ready_by_engine[engine].remove(i)
          return True
      return False

    # Schedule ops in priority order
    for eng in ["load", "valu", "alu", "store", "flow"]:
      while pick(eng):
        pass

    # Handle "none" and "unknown" engine ops
    for eng in ("none", "unknown"):
      for i in ready_by_engine[eng][:]:
        bundle.append(i)
        engine_choices[i] = eng
        ready_by_engine[eng].remove(i)

    # Recalculate spare ALU slots
    alu_spare = SLOT_LIMITS["alu"] - slots_used["alu"]

    # FULL DEMOTION: when ALU has 8+ spare slots, demote waiting VALU ops
    while alu_spare >= 8 and ready_by_engine["valu"]:
      demoted = False
      for i in ready_by_engine["valu"][:]:
        u = uops[i]
        if u.op in demotable_ops and u.dtype.count == 8:
          bundle.append(i)
          engine_choices[i] = "alu"
          slots_used["alu"] += 8
          alu_spare -= 8
          ready_by_engine["valu"].remove(i)
          demoted = True
          break
      if not demoted:
        break

    # SPLIT DEMOTION: if 4 spare slots and no pending split, start one
    if alu_spare >= 4 and pending_split is None and ready_by_engine["valu"]:
      for i in ready_by_engine["valu"][:]:
        u = uops[i]
        if u.op in demotable_ops and u.dtype.count == 8:
          split_lo[len(bundles)] = u
          slots_used["alu"] += 4
          alu_spare -= 4
          ready_by_engine["valu"].remove(i)
          pending_split = (i, len(bundles))
          break

    # Mark scheduled
    for i in bundle:
      scheduled[i] = True
      remaining -= 1

    if bundle or len(bundles) in split_lo:
      bundles.append(([uops[i] for i in bundle], {uops[i]: engine_choices[i] for i in bundle}))
    cycle += 1

  return bundles, {"split_lo": split_lo, "split_hi": split_hi}

# ************************* VLIW renderer *************************

class VLIWRenderer(Renderer):
  has_local = False
  code_for_op: dict = {Ops.MULACC: None, Ops.ADD: "+", Ops.MUL: "*",
                       Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|",
                       Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
  pre_matcher = vliw_prepare

  def render(self, uops: list[UOp]):
    print(f"rendering with {len(uops)} uops")

    # Pack UOps into bundles
    packed, pack_info = pack_uops_vliw(uops, self.code_for_op)
    bundles = [b for b, _ in packed]
    engine_choices = {}
    for _, choices in packed:
      engine_choices.update(choices)
    split_lo = pack_info.get("split_lo", {})
    split_hi = pack_info.get("split_hi", {})
    print(f"  packed {len(uops)} UOps into {len(bundles)} bundles")

    # Register allocation
    uop_to_bundle = {u: bi for bi, bundle in enumerate(bundles) for u in bundle}

    last_use_bundle: dict[UOp, int] = {}
    gep_source: dict[UOp, UOp] = {}
    for bi, bundle in enumerate(bundles):
      for u in bundle:
        if u.op is Ops.GEP:
          gep_source[u] = u.src[0]
        for src in u.src:
          last_use_bundle[src] = bi
          if src in gep_source:
            last_use_bundle[gep_source[src]] = bi
      # Split demotions read inputs in split_lo bundle
      if bi in split_lo:
        u = split_lo[bi]
        for src in u.src:
          last_use_bundle[src] = max(last_use_bundle.get(src, 0), bi)
          if src in gep_source:
            last_use_bundle[gep_source[src]] = max(last_use_bundle.get(gep_source[src], 0), bi)

    r: dict[UOp, int] = {}
    free_scalar: list[int] = []
    free_vector: list[int] = []

    zero_reg = 0
    reg = 1

    for bi, bundle in enumerate(bundles):
      # Allocate outputs for split_lo ops FIRST
      if bi in split_lo:
        u = split_lo[bi]
        if u not in r and u.dtype.count == 8:
          if free_vector:
            r[u] = free_vector.pop()
          else:
            r[u] = reg
            reg += 8

      # Allocate outputs
      for u in bundle:
        if u.op in {Ops.STORE, Ops.SINK, Ops.GEP}:
          continue
        if u in r:  # Skip if already allocated (split_lo)
          continue
        if u.dtype.count == 8:
          if free_vector:
            r[u] = free_vector.pop()
          else:
            r[u] = reg
            reg += 8
        else:
          if free_scalar:
            r[u] = free_scalar.pop()
          else:
            r[u] = reg
            reg += 1

      # Handle GEPs
      for u in bundle:
        if u.op is Ops.GEP:
          r[u] = r[u.src[0]] + u.arg[0]

      # Free registers
      for u in bundle:
        for src in u.src:
          if last_use_bundle.get(src) == bi and src in r:
            if src.op is Ops.GEP:
              pass
            elif src.dtype.count == 8:
              if r[src] not in free_vector:
                free_vector.append(r[src])
            else:
              if r[src] not in free_scalar:
                free_scalar.append(r[src])

    # Emit instructions
    inst = [{"load": [("const", zero_reg, 0)]}]  # Prelude: load zero

    for bi, bundle in enumerate(bundles):
      bundle_inst: dict[str, list] = {}

      # Emit split_lo (elements 0-3)
      if bi in split_lo:
        u = split_lo[bi]
        for i in range(4):
          bundle_inst.setdefault("alu", []).append((self.code_for_op[u.op], r[u]+i, r[u.src[0]]+i, r[u.src[1]]+i))

      # Emit split_hi (elements 4-7)
      if bi in split_hi:
        u = split_hi[bi]
        for i in range(4, 8):
          bundle_inst.setdefault("alu", []).append((self.code_for_op[u.op], r[u]+i, r[u.src[0]]+i, r[u.src[1]]+i))

      for u in bundle:
        match u.op:
          case Ops.SINK:
            bundle_inst.setdefault("flow", []).append(("halt",))
          case Ops.CONST:
            bundle_inst.setdefault("load", []).append(("const", r[u], u.arg))
          case Ops.GEP | Ops.GROUP:
            pass
          case Ops.VECTORIZE:
            if all(s is u.src[0] for s in u.src):
              bundle_inst.setdefault("valu", []).append(("vbroadcast", r[u], r[u.src[0]]))
            else:
              for j, s in enumerate(u.src):
                if r[s] != r[u]+j:
                  bundle_inst.setdefault("alu", []).append(("+", r[u]+j, r[s], zero_reg))
          case Ops.LOAD:
            op = "vload" if u.dtype.count > 1 else "load"
            bundle_inst.setdefault("load", []).append((op, r[u], r[u.src[0]]))
          case Ops.STORE:
            op = "vstore" if u.src[1].dtype.count > 1 else "store"
            bundle_inst.setdefault("store", []).append((op, r[u.src[0]], r[u.src[1]]))
          case Ops.MULACC:
            if u.dtype.count == 8:
              bundle_inst.setdefault("valu", []).append(("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]))
            else:
              if bundle_inst:
                inst.append(bundle_inst)
                bundle_inst = {}
              inst.append({"alu": [("*", r[u], r[u.src[0]], r[u.src[1]])]})
              bundle_inst.setdefault("alu", []).append(("+", r[u], r[u], r[u.src[2]]))
          case Ops.WHERE:
            bundle_inst.setdefault("flow", []).append(("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]))
          case _ if u.op in self.code_for_op:
            chosen_engine = engine_choices.get(u, "valu" if u.dtype.count > 1 else "alu")
            if chosen_engine == "alu_hi":
              pass  # Already handled by split_hi above
            elif chosen_engine == "alu" and u.dtype.count == 8:
              # Full demoted vector op: emit 8 scalar ops
              for i in range(8):
                bundle_inst.setdefault("alu", []).append((self.code_for_op[u.op], r[u]+i, r[u.src[0]]+i, r[u.src[1]]+i))
            else:
              bundle_inst.setdefault(chosen_engine, []).append((self.code_for_op[u.op], r[u], r[u.src[0]], r[u.src[1]]))
          case _:
            raise NotImplementedError(f"unhandled op {u.op}")

      if bundle_inst:
        inst.append(bundle_inst)

    print(f"  emitted {len(inst)} VLIW instructions")
    print(f"  reg allocator used {reg} addresses")

    # Stats
    slot_usage = {k: 0 for k in SLOT_LIMITS}
    for bundle in inst:
      for engine, slots in bundle.items():
        if engine in slot_usage:
          slot_usage[engine] += len(slots)
    print(f"  slot usage: {slot_usage}")

    return repr(inst)

# ************************* test and verify *************************

import sys
import types

PROBLEM_URL = "https://raw.githubusercontent.com/anthropics/original_performance_takehome/refs/heads/main/tests/frozen_problem.py"
sys.modules["problem"] = problem = types.ModuleType("problem")
exec(fetch(PROBLEM_URL).read_text(), problem.__dict__)

if __name__ == "__main__":
  batch_size = getenv("BS", 256)
  height = 10
  rounds = getenv("ROUNDS", 16)

  tree = problem.Tree.generate(height)
  inp = problem.Input.generate(tree, batch_size, rounds)
  mem = problem.build_mem_image(tree, inp)
  global_addrs.extend([mem[6], mem[6], mem[4]])

  forest_t = Tensor(tree.values, dtype=dtypes.uint32)
  val_t = Tensor(inp.values, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    with Context(PCONTIG=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    problem.reference_kernel(tree, inp)
    assert val_out == inp.values
    print("verification passed")

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, SPEC=0):
    out = tree_traversal(forest_t, val_t, height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  src = eval(prg.src)
  max_regs = max(t[1] for instr in src for v in instr.values() for t in v if len(t) > 1) + 8
  print(f"{max_regs:5d} regs used" + ("" if max_regs <= 1536 else "       <-- WARNING: TOO MANY REGISTERS"))

  machine = problem.Machine(mem, src, problem.DebugInfo(scratch_map={}), n_cores=1, trace=getenv("TRACE", 0), scratch_size=max_regs)
  machine.run()
  print(f"ran for {machine.cycle:5d} cycles" + ("" if machine.cycle <= 1363 else "  <-- TARGET: 1363"))

  ref_mem = mem.copy()
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
  print("compare passed!")
