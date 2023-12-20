import random
from typing import List
from tqdm import trange
from tinygrad.helpers import getenv, DEBUG, colored, prod
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sym_infer
from test.external.fuzz_shapetracker import shapetracker_ops
from test.external.fuzz_shapetracker import do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad

class MultiShapeTracker:
  def __init__(self, sts:List[ShapeTracker]): self.sts = sts
  @property
  def shape(self): return self.sts[0].shape
  def reshape(self, arg): self.sts = [x.reshape(arg) for x in self.sts]
  def permute(self, arg): self.sts = [x.permute(arg) for x in self.sts]
  def expand(self, arg): self.sts = [x.expand(arg) for x in self.sts]
  def shrink(self, arg): self.sts = [x.shrink(arg) for x in self.sts]
  def stride(self, arg): self.sts = [x.stride(arg) for x in self.sts]
  def pad(self, arg): self.sts = [x.pad(arg) for x in self.sts]

def fuzz_plus():
  m = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
  for _ in range(4): random.choice(shapetracker_ops)(m)
  backup = m.sts[0]
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m)
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

# shrink and expand aren't invertible, and stride is only invertible in the flip case
# do_pad and do_flip were removed because simplify isn't good
invertible_shapetracker_ops = [do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip] #, do_flip]

def fuzz_invert():
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  for _ in range(8): random.choice(invertible_shapetracker_ops)(m)
  inv = m.sts[0].invert(start.shape)
  st_sum = (m.sts[0] + inv) if inv else None
  return start, st_sum

if __name__ == "__main__":
  # random.seed(42)
  total = getenv("CNT", 1000)
  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
    good = 0
    for _ in trange(total):
      st1, st2 = fuzz()
      bad = False
      #if st1 == st2 and False:
        # simplify is good enough to tell they are the same
        #good += 1
      if st1.shape == st2.shape:
        # otherwise we have to check them
        idx = Variable("idx", 0, prod(st1.shape)-1)
        st1_idx, st1_valid = st1.expr_node(idx)
        st2_idx, st2_valid = st2.expr_node(idx)
        for i in range(idx.min, idx.max):
          st1_off = sym_infer(st1_idx, {idx: i})
          st2_off = sym_infer(st2_idx, {idx: i})
          st1_v = sym_infer(st1_valid, {idx: i})
          st2_v = sym_infer(st2_valid, {idx: i})
          if st1_off != st2_off or st1_v != st2_v:
            print(f"MISMATCH {i=}, {st1_off=} != {st2_off=}, {st1_v=} != {st2_v=}")
            bad = True
            break
      else:
        bad = True
      if not bad: good += 1
      if bad or DEBUG == 1:
        print(f"EXP: {st1}")
        print(f"GOT: {st2}")
        print(colored("****", "red" if st1 != st2 else "green"))
    print(f"hit {good}/{total}")
    assert good == total
