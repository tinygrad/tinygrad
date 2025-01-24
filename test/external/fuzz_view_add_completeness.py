# fuzz for false negatives in view add
from test.unit.test_shapetracker_math import MultiShapeTracker
from test.external.fuzz_shapetracker_math import shapetracker_ops
from tinygrad.shape.view import View, unravel
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.helpers import getenv, prod, trange, tqdm
import random, functools

#@functools.lru_cache(maxsize=None)
def brute_merge(v2, v1):
  # special case
  if 0 in v1.shape: return v1
  
  def _unravel(x): return unravel(v1.shape, x)
  def _eval(x):
    idx1, valid1 = v1.to_indexed_uops(_unravel(x))
    idx2, valid2 = v2.to_indexed_uops(unravel(v2.shape, idx1))
    return int(idx2), bool(valid1 and valid2)

  n = len(v1.shape)
  N = prod(v1.shape)
  strides = [0]*n
  
  # find first corner
  for x in range(N+1):
    if _eval(x)[1]: break
  if x == N:
    return View.create(v1.shape, (0,)*n, 0, ((0,0),)*n)
  m = _unravel(x)
  origin = _eval(x)[0]
  #M = [None]*n
  # special case that the mask is on the border of v1.shape on some axis
  M = [v1.shape[j] if m[j]==v1.shape[j]-1 else None for j in range(n)]

  while (x:=x+1) < N:
    i, (idx, valid) = _unravel(x), _eval(x)
    diff = [i[j] != m[j] for j in range(n)]
    ## handle masks
    # if it's valid outside the rectangle of possibility for the mask then impossible to merge
    if any(i[j] < m[j] or i[j] >= (M[j] or N) for j in range(n)):
      if valid:
        return None
    # we are inside the rectangle of possibility, so it may be valid or invalid;
    # check for mask that extends to end of shape
    elif valid:
      for j in range(n):
        if i[j] == v1.shape[j]-1: M[j] = v1.shape[j]
    # if it goes to invalid then it must be at a rectangular point on the new axis
    elif not valid:
      if 1 != sum(diff):
        return None
      # get new mask dim
      axis = diff.index(True)
      assert M[axis] is None
      M[axis] = i[axis]

    ## handle strides
    if valid:
      axis = diff.index(True)
      if sum(diff) == 1 and i[axis]-m[axis] == 1:
        strides[axis] = idx - origin
      if idx != origin + sum( (i[j]-m[j])*strides[j] for j in range(n)):
        return None
  
  return View.create(v1.shape, tuple(strides), origin-sum(m[j]*strides[j] for j in range(n)), tuple(zip(tuple(m), tuple(M))))

if __name__ == "__main__":
  total_adds = 0
  false_adds = 0
  random.seed(getenv("SEED", 42))
  for i in tqdm(range(getenv("I",0), getenv("CNT",1000))):
    # generate adds
    m1 = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
    for _ in range(4): random.choice(shapetracker_ops)(m1)
    m2 = MultiShapeTracker([ShapeTracker.from_shape(m1.sts[0].shape)])
    for _ in range(4): random.choice(shapetracker_ops)(m2)
    last_view = m1.sts[0].views[-1]
    new_views = list(m2.sts[0].views)
    while new_views:
      total_adds += 1
      fast_sum = last_view + new_views[0]
      true_sum = brute_merge(last_view, new_views[0])
      if fast_sum != true_sum:
        if fast_sum is not None:  # this should never happen, means a bug in brute_merge
          import pdb; pdb.set_trace()
          brute_merge(last_view, new_views[0])
        false_adds += 1
        print(f"missed case: {last_view} + {new_views[0]} == {true_sum}")
#        import pdb; pdb.set_trace()
#        last_view + new_views[0]
      if true_sum is not None:
        last_view = true_sum
        del new_views[0]
      else: break
  print(f"Missed adds: {false_adds}/{total_adds} {false_adds/total_adds*100:.2f}%")
