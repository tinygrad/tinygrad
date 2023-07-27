import itertools
import random
from tinygrad.helpers import DEBUG
from tinygrad.shape.symbolic import Variable
random.seed(random.random())

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

render_real_expr = {
  "add_v": lambda a, b: f"{a}+v{b+1}",
  "div": lambda a, b: f"({a})//{b}",
  "mul": lambda a, b: f"({a})*{b}",
  "mod": lambda a, b: f"({a})%{b}",
  "add_num": lambda a, b: f"{a}{b:+}",
  "lt": lambda a, b: f"(({a})<({b}))",
  "ge": lambda a, b: f"(({a})>=({b}))",
  "le": lambda a, b: f"(({a})<=({b}))",
  "gt": lambda a, b: f"(({a})>({b}))",
}

if __name__ == "__main__":
  ops = [add_v, div, mul, add_num, mod]
  total_tests = 1000
  for test in range(1, total_tests+1):
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256]
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    v = [u1,u2,u3]
    tape = [random.choice(ops) for _ in range(random.randint(2, 30))]
    # 10% of the time, add one of lt, le, gt, ge
    if random.random() < 0.1: tape.append(random.choice([lt, le, gt, ge]))
    expr = u1
    rngs = []
    real_expr = "v1"
    for t in tape:
      expr, rng = t(expr)
      real_expr = render_real_expr[t.__name__](real_expr, rng)
      rngs.append(rng)
    render = expr.render()
    sample_size = min(101, (u1.max*u2.max*u3.max))
    if sample_size < 101: samples = list(itertools.product(range(u1.min, u1.max+1), range(u2.min, u2.max+1), range(u3.min, u3.max+1)))
    else: samples = [ (random.randint(u1.min, u1.max), random.randint(u2.min, u2.max), random.randint(u3.min, u3.max)) for _ in range(sample_size) ]
    for count, (v1, v2, v3) in enumerate(samples):
      v = [v1,v2,v3]
      num = eval(render)
      rn = v1
      for t,r in zip(tape, rngs): rn, _ = t(rn, r)
      assert num == rn, f"""FAILURE expr#{test} iteration#{count+1} (expressions NOT equivalent!)
      * symbolic: {render} [{expr.min}, {expr.max}]
      * real:     {real_expr}
      * for {u1} = {v1}, {u2} = {v2}, {u3} = {v3} => symbolic:{num} != real:{rn}
      * {count} different values of v1, v2 and v3 were successfully tested till now"""
      if DEBUG >= 2: print(f"success expr#{test} iteration#{count+1} for {v1=} {v2=} {v3=} => {num} == {rn}")
    if DEBUG >=1: print(f"SUCCESS expr#{test} (tests:{sample_size}): {render} is equivalent to {real_expr}")
  print(f"{total_tests} tests successful!")
