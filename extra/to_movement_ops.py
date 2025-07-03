import itertools
from collections import defaultdict
from typing import List, Tuple, DefaultDict
from extra.optimization.helpers import load_worlds, ast_str_to_ast
from tinygrad.helpers import prod, tqdm
from tinygrad.uop.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker, apply_mop
from tinygrad.tensor import Tensor

def make_scratch_st(st: ShapeTracker) -> ShapeTracker:
  return ShapeTracker.from_shape((get_buffer_size(st.views[0].shape, st.views[0].strides, st.views[0].offset, st.views[0].mask),))

def get_real_view(shape, strides, offset, mask):
  real_shape = tuple(y-x for x,y in mask) if mask else shape
  offset = offset + sum(st * (s-1) for s,st in zip(real_shape, strides) if st<0)
  real_offset = offset + (sum(x*st for (x,_),st in zip(mask, strides)) if mask else 0)
  real_real_shape = [s for s,st in zip(real_shape, strides) if st]
  strides = [abs(st) if isinstance(st,int) else st for st in strides if st]
  return real_real_shape, strides, real_offset

def get_buffer_size(shape, strides, offset, mask):
  real_real_shape, strides, real_offset = get_real_view(shape, strides, offset, mask)
  return real_offset + sum((s-1)*st for s, st in zip(real_real_shape,strides)) + 1

def st_equivalent(st1: ShapeTracker, st2: ShapeTracker):
  if (idxs1:=st1.expr_idxs()) == (idxs2:=st2.expr_idxs()): return True
  idx1, valid1 = idxs1
  idx2, valid2 = idxs2
  # always invalid
  if valid1 == 0 and valid2 == 0: return True

  var1 = idx1.vars() | valid1.vars()
  var2 = idx2.vars() | valid2.vars()
  # Maybe there are cases that vars are different yet the sts are the same?
  if var1 != var2: return False

  # brute force over the vars range
  vs = list(var1)
  for i, ranges in enumerate(itertools.product(*[range(v.min, v.max+1) for v in vs])):
    if i > 1000:
      print("WARNING: did not search all possible combinations")
      break
    var_vals = {k:v for k,v in zip(vs, ranges)}
    r1 = sym_infer(idx1, var_vals) if sym_infer(valid1, var_vals) else 0
    r2 = sym_infer(idx2, var_vals) if sym_infer(valid2, var_vals) else 0
    if r1 != r2: return False

  return True

c: DefaultDict[int,int] = defaultdict(int)
def test_rebuild(st: ShapeTracker):
  rebuilt_st = make_scratch_st(st)
  mops = st.to_movement_ops()
  c[len(mops)] += 1
  for mop_arg in mops: rebuilt_st = apply_mop(rebuilt_st, mop_arg)
  rebuilt_st = rebuilt_st.simplify()
  # why is the "all(x == 0 for x in rebuilt_st.views[-1].strides)" hack needed?
  assert st_equivalent(st, rebuilt_st) or all(x == 0 for x in rebuilt_st.views[-1].strides), f"mismatch {st} {rebuilt_st}"
  last_v1 = st.views[-1]
  last_v2 = rebuilt_st.views[-1]
  assert last_v1.shape == last_v2.shape, f"{last_v1.shape} != {last_v2.shape}"

def test_rebuild_bufferop_st(ast:UOp):
  if ast.op is Ops.SHAPETRACKER:
    test_rebuild(ast.arg)
  for src in ast.src: test_rebuild_bufferop_st(src)

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, True)[:2000]
  for ast_str in tqdm(ast_strs):
    test_rebuild_bufferop_st(ast_str_to_ast(ast_str))

  print(f"avg length of mop = {sum(k*v for k,v in c.items()) / sum(c.values()):.2f}")
