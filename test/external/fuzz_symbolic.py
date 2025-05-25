import random
from z3 import IntVal, Solver, unsat, sat, ArithRef, Ints, If
from tinygrad import Variable, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.helpers import DEBUG
random.seed(42)

# z3 division on integers is floored, but tinygrad idiv is truncated division
ArithRef.__floordiv__ = lambda self, other: If(self<0, (self+(other-1))/other, self/other)

def add_v(expr, rng=None):
  if rng is None: rng = random.randint(0,2)
  return expr + v[rng], rng

def div(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr // rng, rng

def mul(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr * rng, rng

def mod(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr % rng, rng

def add_num(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr + rng, rng

def lt(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr < rng, rng

def ge(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr >= rng, rng

def le(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr <= rng, rng

def gt(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr > rng, rng


if __name__ == "__main__":
  ops = [add_v, mul, add_num, div]
  for i in range(1000):
    solver = Solver()
    if i % 100 == 0:
      print(f"Running test {i}")
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256, 512, 1024]
    v1, v2, v3 = Ints("v1 v2 v3")
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    solver.add(v1 >= 0, v2 >= 0, v3 >= 0)
    solver.add(v1 <= u1.arg[2], v2 <= u2.arg[2], v3 <= u3.arg[2])
    v = [u1,u2,u3]
    tape = [random.choice(ops) for _ in range(random.randint(2, 30))]
    # 10% of the time, add one of lt, le, gt, ge
    if random.random() < 0.1: tape.append(random.choice([lt, le, gt, ge]))
    expr = UOp.const(dtypes.int, 0)
    rngs = []
    for t in tape:
      expr, rng = t(expr)
      if DEBUG >= 1: print(t.__name__, rng)
      rngs.append(rng)
    if DEBUG >=1: print(expr)

    v = [v1,v2,v3]
    z3_expr = IntVal(0)
    for t,r in zip(tape, rngs): z3_expr, _ = t(z3_expr, r)
    simplified_expr = eval(expr.render())
    if solver.check(simplified_expr != z3_expr) == sat:
      m = solver.model()
      unsimplified_expr = eval(expr.render(simplify=False))
      assert solver.check(unsimplified_expr != z3_expr) == unsat, f"UNSIMPLIFIED MISMATCH!, {expr.render(simplify=False)} != {z3_expr}"
      n1, n2, n3 = m[v1], m[v2], m[v3]
      u1_val, u2_val, u3_val = u1.const_like(n1.as_long()), u2.const_like(n2.as_long()), u3.const_like(n3.as_long())
      num = expr.simplify().substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
      rn = expr.substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
      assert False, f"""
mismatched {expr.render()} at v1={m[v1]}; v2={m[v2]}; v3={m[v3]} = {num} != {rn}
Reproduce with:
v1={u1}
v2={u2}
v3={u3}
expr = {expr.render(simplify=False)}
v1_val, v2_val, v3_val = v1.const_like({n1.as_long()}), v2.const_like({n2.as_long()}), v3.const_like({n3.as_long()})
num = expr.simplify().substitute({{v1:v1_val, v2:v2_val, v3:v3_val}}).ssimplify()
rn = expr.substitute({{v1:v1_val, v2:v2_val, v3:v3_val}}).ssimplify()
assert num==rn, f"{{num}} != {{rn}}"
"""

    if DEBUG >= 1: print(f"validated {expr.render()}")
