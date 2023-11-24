from tqdm import tqdm
import itertools
from typing import List, Tuple
from extra.optimization.helpers import load_worlds, ast_str_to_ast
from tinygrad.ops import MovementOps, BufferOps, LazyOp
from tinygrad.helpers import prod
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import sym_infer, Node

# ShapeTracker to an equivalent series of MovementOps (https://github.com/tinygrad/tinygrad/pull/2216)
def to_movement_ops(st: ShapeTracker) -> List[Tuple[MovementOps, Tuple]]:
  to_apply:List[Tuple[MovementOps, Tuple]] = []
  for i, v in enumerate(st.views):
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    offset = v.offset + sum(st*(s-1) for s,st in zip(real_shape, v.strides) if st<0)
    real_offset = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    real_real_shape = [s for s,st in zip(real_shape, v.strides) if st]
    strides: List[Node|int] = [abs(st) if isinstance(st,int) else st for st in v.strides if st]
    buffer_size = sum((s-1)*st for s,st in zip(real_real_shape,strides)) + 1
    if i: buffer_size = prod(st.views[i-1].shape) - real_offset
    def sort_by_strides(shape, strides): return sorted(zip(shape, strides), key=lambda k: (k[1],-k[0]), reverse=True), sorted(range(len(strides)), key=lambda k: (strides[k],-real_real_shape[k]), reverse=True)
    ordered_shape_strides, order = sort_by_strides(real_real_shape, strides)
    to_apply.extend([(MovementOps.RESHAPE, (-1,)), (MovementOps.SHRINK, ((real_offset, real_offset+buffer_size),))])
    if strides:
      if (ordered_shape_strides[0][0]*ordered_shape_strides[0][1])-buffer_size>0: to_apply.append((MovementOps.PAD, ((0, (ordered_shape_strides[0][0] * ordered_shape_strides[0][1]) - buffer_size),)))
      for i, shape_stride in enumerate(ordered_shape_strides):
        if i<len(ordered_shape_strides)-1 and shape_stride[1] < ordered_shape_strides[i+1][0]*ordered_shape_strides[i+1][1]:
          remaining_buffer = ordered_shape_strides[i-1][1] if i>0 else buffer_size
          to_apply.append((MovementOps.EXPAND, (shape_stride[0], *(s[0] for s in ordered_shape_strides[:i]), remaining_buffer)))
          to_apply.append((MovementOps.PERMUTE, (*range(1,i+1), 0, i+1)))
          to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i]), shape_stride[0]*remaining_buffer)))
          to_apply.append((MovementOps.PAD, (*((0,0) for _ in range(i)), (0, shape_stride[0]*shape_stride[1]))))
          to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i+1]), remaining_buffer+shape_stride[1])))
          ordered_shape_strides[i] = (ordered_shape_strides[i][0], remaining_buffer+shape_stride[1])
        else:
          to_apply.append((MovementOps.SHRINK, (*((0, s[0]) for s in ordered_shape_strides[:i]), (0, shape_stride[0]*shape_stride[1]))))
          to_apply.append((MovementOps.RESHAPE, (*[s[0] for s in ordered_shape_strides[:i+1]], shape_stride[1])))
      to_apply.extend([(MovementOps.SHRINK, (*[(0, s[0]) for s in ordered_shape_strides], (0,1))), (MovementOps.RESHAPE, tuple(s[0] for s in ordered_shape_strides))])
      if order != list(range(len(order))): to_apply.append((MovementOps.PERMUTE, tuple(order.index(i) for i in range(len(strides)))))
    to_apply.append((MovementOps.RESHAPE, tuple(s if st else 1 for s,st in zip(real_shape, v.strides))))
    if any(i<0 for i in v.strides): to_apply.append((MovementOps.STRIDE, tuple(-1 if st<0 else 1 for st in v.strides)))
    # then, we apply pre expand pads
    if v.mask is not None:
      pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in pre_expand_pads):
        to_apply.append((MovementOps.PAD, pre_expand_pads))
        real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
    # then, we do any expands
    if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
    # lastly, we apply post expand pads
    if v.mask is not None and any(x != (0,0) for x in post_expand_pads): to_apply.append((MovementOps.PAD, post_expand_pads))
  return to_apply

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
      # not happening for now
      break
    var_vals = {k:v for k,v in zip(vs, ranges)}
    r1 = sym_infer(idx1, var_vals) if sym_infer(valid1, var_vals) else 0
    r2 = sym_infer(idx2, var_vals) if sym_infer(valid2, var_vals) else 0
    if r1 != r2: return False

  return True

def test_rebuild(st: ShapeTracker):
  rebuilt_st = ShapeTracker.from_shape((get_buffer_size(st.views[0].shape, st.views[0].strides, st.views[0].offset, st.views[0].mask),))
  for mop, arg in to_movement_ops(st):
    if mop == MovementOps.RESHAPE:
      # shapetracker doesn't allow flattening with -1 but required for MovementOps.RESHAPE
      if arg == (-1,):
        rebuilt_st = rebuilt_st.reshape((prod(rebuilt_st.views[-1].shape),))
      else:
        rebuilt_st = rebuilt_st.reshape(arg)
    elif mop == MovementOps.PERMUTE:
      rebuilt_st = rebuilt_st.permute(arg)
    elif mop == MovementOps.EXPAND:
      if len(arg) != len(rebuilt_st.shape):
        rebuilt_st = rebuilt_st.reshape((1,*rebuilt_st.shape))
      rebuilt_st = rebuilt_st.expand(arg)
    elif mop == MovementOps.PAD:
      rebuilt_st = rebuilt_st.pad(arg)
    elif mop == MovementOps.SHRINK:
      rebuilt_st = rebuilt_st.shrink(arg)
    elif mop == MovementOps.STRIDE:
      rebuilt_st = rebuilt_st.stride(arg)
    else:
      raise Exception("invalid mop")
  rebuilt_st = rebuilt_st.simplify()
  assert st_equivalent(st, rebuilt_st)
  last_v1 = st.views[-1]
  last_v2 = rebuilt_st.views[-1]
  assert last_v1.shape == last_v2.shape, f"{last_v1.shape} != {last_v2.shape}"

def test_interpret_ast(ast:LazyOp):
  if ast.op in BufferOps:
    test_rebuild(ast.arg.st)
  else:
    for src in ast.src: test_interpret_ast(src)


if __name__ == "__main__":
  ast_strs = load_worlds(False, False, True)[:4000]
  for ast_str in tqdm(ast_strs):
    test_interpret_ast(ast_str_to_ast(ast_str))
