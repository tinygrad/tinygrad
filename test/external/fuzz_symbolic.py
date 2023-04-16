import random
from tinygrad.shape.symbolic import Variable

def add_v(expr, rng=None):
  if rng is None: rng = random.randint(0,2)
  return expr + v[rng], rng

def div(expr, rng=None):
  if rng is None: rng = random.randint(1,4)
  return expr // rng, rng

def mul(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr * rng, rng

def mod(expr, rng=None):
  if rng is None: rng = random.randint(1,4)
  return expr % rng, rng

def add_num(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr + rng, rng

if __name__ == "__main__":
  ops = [add_v, div, mul, add_num]
  while 1:
    v1 = Variable("v1", 0, 3)
    v2 = Variable("v2", 0, 3)
    v3 = Variable("v3", 0, 3)
    v = [v1,v2,v3]
    tape = [random.choice(ops) for _ in range(5)]
    expr = Variable.num(0)
    rngs = []
    for t in tape:
      expr, rng = t(expr)
      print(t.__name__, rng)
      rngs.append(rng)
    print(expr)
    for v1 in range(4):
      for v2 in range(4):
        for v3 in range(4):
          v = [v1,v2,v3]
          rn = 0
          for t,r in zip(tape, rngs):
            rn, _ = t(rn, r)
          num = eval(expr.render())
          assert num == rn, f"mismatch at {v1} {v2} {v3}, {num} != {rn}"
          #print(v1, v2, v3, num, rn)


