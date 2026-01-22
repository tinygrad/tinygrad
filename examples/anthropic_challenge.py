import random
from tinygrad import Tensor, dtypes, Context, getenv, UOp
from tinygrad.uop.ops import Ops
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps

def myhash(a, m=lambda x: x):
  a = m(m(a + 0x7ED55D16) + m(a << 12))
  a = m(m(a ^ 0xC761C23C) ^ m(a >> 19))
  a = m(m(a + 0x165667B1) + m(a << 5))
  a = m(m(a + 0xD3A2646C) ^ m(a << 9))
  a = m(m(a + 0xFD7046C5) + m(a << 3))
  a = m(m(a ^ 0xB55A4F09) ^ m(a >> 16))
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

def python_reference(forest: list[int], val: list[int], height: int, rounds: int) -> list[int]:
  n_nodes = len(forest)
  idx, val = [0] * len(val), val.copy()
  for _ in range(rounds):
    for i in range(len(idx)):
      val[i] = myhash(val[i] ^ forest[idx[i]], lambda x: x % (2**32))
      idx[i] = 2 * idx[i] + (1 if val[i] % 2 == 0 else 2)
      if idx[i] >= n_nodes: idx[i] = 0
  return val

class VLIWRenderer(Renderer):
  # this says this backend supports MULACC
  code_for_op: dict = {Ops.MULACC: None}

  def render(self, uops:list[UOp]):
    # TODO: implement this
    from tinygrad.uop.ops import print_uops
    print_uops(uops)

if __name__ == "__main__":
  batch_size = getenv("BS", 256)
  height = 10
  rounds = getenv("ROUNDS", 16)

  forest = [random.randint(0, 2**30 - 1) for _ in range(2 ** (height + 1) - 1)]
  val = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]
  forest_t = Tensor(forest, dtype=dtypes.uint32)
  val_t = Tensor(val, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    with Context(PCONTIG=2, DEVECTORIZE=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    assert val_out == python_reference(forest, val, height, rounds)

  # *** render to device ***

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, CPU_LLVM=1):
    out = tree_traversal(forest_t.to("CPU"), val_t.to("CPU"), height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  # TODO: run prg.src in "Machine" from original_performance_takehome
  print(prg.src)
