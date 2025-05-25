import random
import z3
from tinygrad import Variable, dtypes
from tinygrad.uop.ops import UOp, graph_rewrite
from tinygrad.uop.spec import z3_renderer
from tinygrad.helpers import DEBUG, Context
seed = random.randint(0, 100)
print(f"Seed: {seed}")
random.seed(seed)

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
  ops = [add_v, mul, add_num, div, mod]
  for i in range(50000):
    if i % 10 == 0:
      print(f"Running test {i}")
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256]
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    v = [u1,u2,u3]
    tape = [random.choice(ops) for _ in range(random.randint(2, 30))]
    # 10% of the time, add one of lt, le, gt, ge
    if random.random() < 0.1: tape.append(random.choice([lt, le, gt, ge]))
    expr = UOp.const(dtypes.int, 0)
    for t in tape:
      expr, rng = t[0](expr)
      if DEBUG >= 3: print(t[0].__name__, rng)
    if DEBUG >=3: print(expr)
    with Context(CORRECT_DIVMOD_FOLDING=1):
      simplified_expr = expr.simplify()

    solver = z3.Solver()
    solver.set(timeout=1000)  # some expressions take very long verify, but its very unlikely they actually return sat
    z3_sink = graph_rewrite(expr.sink(simplified_expr, u1, u2, u3), z3_renderer, ctx=(solver, {}))
    z3_expr, z3_simplified_expr = z3_sink.src[0].arg, z3_sink.src[1].arg
    check = solver.check(z3_simplified_expr != z3_expr)
    if check == z3.unknown and DEBUG>=1:
      print("Skipped due to timeout:\n" +
            f"v1=Variable(\"{u1.arg[0]}\", {u1.arg[1]}, {u1.arg[2]})\n" +
            f"v2=Variable(\"{u2.arg[0]}\", {u2.arg[1]}, {u2.arg[2]})\n" +
            f"v3=Variable(\"{u3.arg[0]}\", {u3.arg[1]}, {u3.arg[2]})\n" +
            f"expr = {expr.render(simplify=False)}\n")
    elif check == z3.sat:
      m = solver.model()
      v1, v2, v3 = z3_sink.src[2].arg, z3_sink.src[3].arg, z3_sink.src[4].arg
      n1, n2, n3 = m[v1], m[v2], m[v3]
      u1_val, u2_val, u3_val = u1.const_like(n1.as_long()), u2.const_like(n2.as_long()), u3.const_like(n3.as_long())
      with Context(CORRECT_DIVMOD_FOLDING=1):
        num = expr.simplify().substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
        rn = expr.substitute({u1:u1_val, u2:u2_val, u3:u3_val}).ssimplify()
        if num==rn: print("z3 found a mismatch but the expressions are equal!!")
      print("mismatched {expr.render()} at v1={m[v1]}; v2={m[v2]}; v3={m[v3]} = {num} != {rn}\n" +
            "Reproduce with:\n" +
            f"v1=Variable(\"{u1.arg[0]}\", {u1.arg[1]}, {u1.arg[2]})\n" +
            f"v2=Variable(\"{u2.arg[0]}\", {u2.arg[1]}, {u2.arg[2]})\n" +
            f"v3=Variable(\"{u3.arg[0]}\", {u3.arg[1]}, {u3.arg[2]})\n" +
            f"expr = {expr.render(simplify=False)}\n" +
            f"v1_val, v2_val, v3_val = v1.const_like({n1.as_long()}), v2.const_like({n2.as_long()}), v3.const_like({n3.as_long()})\n" +
            "num = expr.simplify().substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()\n" +
            "rn = expr.substitute({v1:v1_val, v2:v2_val, v3:v3_val}).ssimplify()\n" +
            f"assert num==rn, {num} != {rn}\n")

    if DEBUG >= 2: print(f"validated {expr.render()}")
