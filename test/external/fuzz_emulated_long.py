import random
import z3
from tinygrad import dtypes, Device
from tinygrad.uop.validate import uops_to_z3, z3_cdiv
from tinygrad.uop.ops import UOp
from tinygrad.uop import Ops
from tinygrad.uop.decompositions import l2i
random.seed(42)

def split(v): return (v & 0xFFFFFFFF).cast(dtypes.uint), (v >> 32).cast(dtypes.uint)
def combine(lo, hi): return lo.cast(dtypes.ulong) | (hi.cast(dtypes.ulong) << 32)

if __name__ == "__main__":
  for i in range(10_000):
    if i % 1000 == 0:
      print(f"Progress: {i}")
    a = UOp.variable('a0', random.randint(dtypes.ulong.min, 0), random.randint(1, dtypes.ulong.max), dtype=dtypes.ulong)
    b = UOp.variable('b0', random.randint(dtypes.ulong.min, 0), random.randint(1, dtypes.ulong.max), dtype=dtypes.ulong)
    expr = combine(*l2i(Ops.IDIV, dtypes.uint, *split(a), *split(b)))

    solver = z3.Solver()
    z3_expr, x = uops_to_z3(solver, expr, a, b)

    if solver.check(z3_expr != z3_cdiv(a, b)) == z3.sat:
      assert False, f"Failed: {expr.render()} != x//{d} at x={solver.model()}\nx={u}\nd={d}\n{z3_expr=}\n{x/d=}"
