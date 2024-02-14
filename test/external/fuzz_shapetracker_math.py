import random
from typing import Tuple
from tqdm import trange
from tinygrad.helpers import getenv, DEBUG, colored
from tinygrad.shape.shapetracker import ShapeTracker
from test.external.fuzz_shapetracker import shapetracker_ops
from test.external.fuzz_shapetracker import do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad
from test.unit.test_shapetracker_math import st_equal, MultiShapeTracker

def fuzz_plus() -> Tuple[ShapeTracker, ShapeTracker]:
  m = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
  for _ in range(4): random.choice(shapetracker_ops)(m)
  backup = m.sts[0]
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m)
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

# shrink and expand aren't invertible, and stride is only invertible in the flip case
invertible_shapetracker_ops = [do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad]

def fuzz_invert() -> Tuple[ShapeTracker, ShapeTracker]:
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  for _ in range(8): random.choice(invertible_shapetracker_ops)(m)
  inv = m.sts[0].invert(start.shape)
  st_sum = (m.sts[0] + inv) if inv else None
  return start, st_sum

if __name__ == "__main__":
  if seed:=getenv("SEED"): random.seed(seed)
  total = getenv("CNT", 1000)
  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
    same_but_neq = same_but_neq_canon = 0
    for _ in trange(total, desc=f"{fuzz}"):
      st1, st2 = fuzz()
      sts1, sts2 = st1.simplify(), st2.simplify()
      stc1, stc2 = st1.canonicalize(), st2.canonicalize()
      eq = st_equal(st1, st2)
      eqs = sts1 == st2
      eqc = stc1 == stc2
      if eq and not eqc: same_but_neq_canon += 1
      if getenv("CHECK_NEQ") and eq and not eqs:
        print(colored("same but unequal", "yellow"))
        print(st1.simplify())
        print(st2.simplify())
        same_but_neq += 1
      if DEBUG >= 1:
        print(f"EXP: {st1}")
        print(f"GOT: {st2}")
      if DEBUG >= 2:
        print(f"EXP CANON: {st1}")
        print(f"GOT CANON: {st2}")
      if DEBUG >= 1:
        print(colored(f"****{' (symbolic)' if DEBUG>=2 else ''}", "green" if eq else "red"))
      if DEBUG >= 2:
        print(colored("**** (canon)", "green" if eqc else "red"))
      if not (eq or eqc): exit(0)
    if getenv("CHECK_NEQ"):
      print(f"same but unequal {(same_but_neq/total)*100:.2f}%")
      if DEBUG >= 2: print(f"same but unequal canon {(same_but_neq_canon/total)*100:.2f}%")
