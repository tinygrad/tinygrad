from tinygrad import Tensor, dtypes, Context, getenv, UOp, fetch
from tinygrad.uop.ops import Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps

# ************************* implementation of problem ************************

def myhash(a):
  a = (a + 0x7ED55D16) + (a << 12)
  a = (a ^ 0xC761C23C) ^ (a >> 19)
  a = (a + 0x165667B1) + (a << 5)
  a = (a + 0xD3A2646C) ^ (a << 9)
  a = (a + 0xFD7046C5) + (a << 3)
  a = (a ^ 0xB55A4F09) ^ (a >> 16)
  return a

def select_with_where_tree(values: Tensor, relative_idx: Tensor) -> Tensor:
  n = values.shape[0]
  if n == 1: return values[0].expand(relative_idx.shape)

  mid = n // 2
  left = select_with_where_tree(values[:mid], relative_idx)
  right = select_with_where_tree(values[mid:], relative_idx - mid)

  go_left = relative_idx < mid
  return go_left.where(left, right)

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

BS = getenv("BS", 256)

def loop_unrolling(sink:UOp):
  rng = [x for x in sink.toposort() if x.op is Ops.RANGE]
  if len(rng) == 0: return None
  unrolled_sinks = [sink.substitute({rng[0]:rng[0].const_like(i)}).src[0] for i in range(rng[0].vmax+1)]
  return UOp.sink(*unrolled_sinks, arg=sink.arg)

global_addrs = [0,0,0]
vliw_prepare = PatternMatcher([
  # loop unrolling (should be a part of tinygrad)
  (UPat(Ops.SINK, name="sink"), loop_unrolling),
  # cast is fake
  (UPat(Ops.CAST, name="c"), lambda c: c.src[0]),
  # rewrites to hardcode the addresses in memory
  (UPat(Ops.DEFINE_GLOBAL, name="dg"), lambda dg: UOp.const(dtypes.uint, global_addrs[dg.arg])),
  # INDEX is just plus
  (UPat(Ops.INDEX, name="i"), lambda i: i.src[0]+i.src[1]),
])+symbolic

class VLIWRenderer(Renderer):
  has_local = False  # TODO: this should be the default / cleaned up
  # this says this backend supports MULACC, SHR, and SHL
  code_for_op: dict = {Ops.MULACC: None, Ops.SHR: None, Ops.SHL: None}
  # this matcher runs while still in graph form
  pre_matcher = vliw_prepare

  def render(self, uops:list[UOp]):
    VLEN, alu_ops = 8, {Ops.ADD: "+", Ops.MUL: "*", Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|", Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
    reg, uop_reg, instrs = 0, {}, []
    def r(u): return uop_reg[u]
    for u in uops:
      if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}: uop_reg[u] = reg; reg += u.dtype.count
      if u.op is Ops.CONST: instrs.append({"load": [("const", r(u), u.arg)]})
      elif u.op is Ops.VECTORIZE:
        if all(s == u.src[0] for s in u.src): instrs.append({"valu": [("vbroadcast", r(u), r(u.src[0]))]})
        else: instrs.extend({"flow": [("add_imm", r(u)+i, r(s), 0)]} for i, s in enumerate(u.src) if r(s) != r(u)+i)
      elif u.op is Ops.GEP: uop_reg[u] = r(u.src[0]) + u.arg[0]  # extract element from vector
      elif u.op is Ops.LOAD: instrs.append({"load": [("vload", r(u), r(u.src[0]))] if u.dtype.count > 1 else [("load", r(u), r(u.src[0]))]})
      elif u.op is Ops.STORE: instrs.append({"store": [("vstore", r(u.src[0]), r(u.src[1]))] if u.src[1].dtype.count > 1 else [("store", r(u.src[0]), r(u.src[1]))]})
      elif u.op is Ops.MULACC: assert u.dtype.count == 8; instrs.append({"valu": [("multiply_add", r(u), r(u.src[0]), r(u.src[1]), r(u.src[2]))]})
      elif u.op is Ops.WHERE: assert u.dtype.count == 8; instrs.append({"flow": [("vselect", r(u), r(u.src[0]), r(u.src[1]), r(u.src[2]))]})
      elif u.op in alu_ops: instrs.append({("valu" if u.dtype.count > 1 else "alu"): [(alu_ops[u.op], r(u), r(u.src[0]), r(u.src[1]))]})
      elif u.op is Ops.SINK: instrs.append({"flow": [("halt",)]})
      else: assert False, f"unhandled op {u.op}"
    return repr(instrs)

# ************************* test and render *************************

import sys, types
PROBLEM_URL = "https://raw.githubusercontent.com/anthropics/original_performance_takehome/refs/heads/main/tests/frozen_problem.py"
sys.modules["problem"] = problem = types.ModuleType("problem")
exec(fetch(PROBLEM_URL).read_text(), problem.__dict__)

if __name__ == "__main__":
  height = 10
  rounds = getenv("ROUNDS", 16)

  # build problem
  tree = problem.Tree.generate(height)
  inp = problem.Input.generate(tree, BS, rounds)
  mem = problem.build_mem_image(tree, inp)
  global_addrs[0] = global_addrs[1] = mem[6]  # input and output
  global_addrs[2] = mem[4]                    # forest

  # *** verify the kernel in tinygrad compared to reference ***

  forest_t = Tensor(tree.values, dtype=dtypes.uint32)
  val_t = Tensor(inp.values, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    with Context(PCONTIG=2, DEVECTORIZE=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    problem.reference_kernel(tree, inp)
    assert val_out == inp.values

  # *** render to device ***

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, CPU_LLVM=1, SPEC=0):
    out = tree_traversal(forest_t.to("CPU"), val_t.to("CPU"), height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  # *** run on Machine and compare ***

  ref_mem = mem.copy()
  machine = problem.Machine(mem, eval(prg.src), problem.DebugInfo(scratch_map={}), n_cores=1, trace=False, scratch_size=100000)
  machine.run()
  print(f"ran for {machine.cycle} cycles")

  # compare to reference
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
