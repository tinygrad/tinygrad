import itertools
from enum import Enum, auto
from collections import defaultdict
from typing import List, Tuple, DefaultDict
from tinygrad.helpers import prod, tqdm
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.ops import sym_infer
from tinygrad.tensor import Tensor

class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702

def apply_mop(t: Tensor, mop_arg: Tuple[MovementOps, Tuple]) -> Tensor:
  mop, arg = mop_arg
  if mop == MovementOps.RESHAPE:
    # shapetracker doesn't allow flattening with -1 but required for MovementOps.RESHAPE
    if arg == (-1,): return t.reshape((prod(t.shape),))
    return t.reshape(arg)
  if mop == MovementOps.PERMUTE: return t.permute(arg)
  if mop == MovementOps.EXPAND:
    if len(arg) != len(t.shape): t = t.reshape((1,*t.shape))
    return t.expand(arg)
  if mop == MovementOps.PAD: return t.pad(arg)
  if mop == MovementOps.SHRINK: return t.shrink(arg)
  if mop == MovementOps.STRIDE:
    assert all(x in [-1, 1] for x in arg)
    return t.flip(tuple(i for i,x in enumerate(arg) if x == -1))
  raise ValueError("invalid mop")

def make_scratch_st(st: ShapeTracker) -> ShapeTracker:
  return ShapeTracker.from_shape((get_buffer_size(st.views[0].shape, st.views[0].strides, st.views[0].offset, st.views[0].mask),))

# ShapeTracker to an equivalent series of MovementOps (https://github.com/tinygrad/tinygrad/pull/2216)
def to_movement_ops(t: Tensor) -> List[Tuple[MovementOps, Tuple]]:
  to_apply:List[Tuple[MovementOps, Tuple]] = []
  t = t.uop.base
  shape_stride = list(zip(shape, stride))
  order = sorted(range(len(stride)), key=lambda i: (-stride[i], -shape[i]))
  ordered_pairs = [shape_stride[i] for i in order]
  buffer_size = sum([(shape[i]-1)*stride[i] for i in range(len(shape))]) + 1
  assert storage_offset + buffer_size <= t.shape[0], f"requested buffer size {storage_offset + buffer_size} out of bounds for current buffer size {buffer_size}"
  # t = t.shrink(((storage_offset, storage_offset + buffer_size),))
  to_apply.append((MovementOps.SHRINK, ((storage_offset, storage_offset + buffer_size),)))
  broadcast_idx = -1
  if ordered_pairs[0][0]*ordered_pairs[0][1] > buffer_size: 
    t = t.pad(((0, (ordered_pairs[0][0] * ordered_pairs[0][1]) - buffer_size),))
    to_apply.append((MovementOps.PAD, ((0, (ordered_pairs[0][0] * ordered_pairs[0][1]) - buffer_size),)))
  for i,(sh, st) in enumerate(ordered_pairs):
    if i+1<len(ordered_pairs) and st < ordered_pairs[i+1][0]*ordered_pairs[i+1][1]:
      rem_buffer = ordered_pairs[i-1][1] if i>0 else buffer_size
      # t = t.expand(sh, *(s[0] for s in ordered_pairs[:i]), rem_buffer)
      to_apply.append((MovementOps.EXPAND, (sh, *(s[0] for s in ordered_pairs[:i]), rem_buffer)))
      # expand_shape = (sh, *[s[0] for s in ordered_pairs[:i]], rem_buffer)
      # t = _safe_expand(t, expand_shape)
      # t = t.permute(*(range(1,i+1)), 0, i+1)
      to_apply.append((MovementOps.PERMUTE, (*(range(1,i+1)), 0, i+1)))
      # t = t.reshape(*(s[0] for s in ordered_pairs[:i]), sh*rem_buffer)
      to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_pairs[:i]), sh*rem_buffer)))
      # t = t.pad((*((0,0) for _ in range(i)), (0, sh*st)))
      to_apply.append((MovementOps.PAD, (*((0,0) for _ in range(i)), (0, sh*st))))
      # t = t.reshape(*(s[0] for s in ordered_pairs[:i+1]), rem_buffer+st)
      to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_pairs[:i+1]), rem_buffer+st)))
      ordered_pairs[i] = (ordered_pairs[i][0], rem_buffer+st)
    else:
      # handle 0 strides (broadcast)
      if st == 0:
          broadcast_idx = i
          break # remaining all the other strides will be 0 since we are in reverse sorted order wrt strides
      # t = t.shrink((*((0,s[0]) for s in ordered_pairs[:i]), (0, sh*st + 1)))
      to_apply.append((MovementOps.SHRINK, (*((0,s[0]) for s in ordered_pairs[:i]), (0, sh*st + 1))))
      to_apply.append((MovementOps.RESHAPE, (*[s[0] for s in ordered_pairs[:i+1]], st)))
      # t = t.reshape(*[s[0] for s in ordered_pairs[:i+1]], st)
  # check for broadcasts (0 strides)
  if broadcast_idx != -1:
    # expand along the remaining dims starting from broadcast_idx
    for i in range(broadcast_idx, len(ordered_pairs)):
      # t = t.expand(*(s[0] for s in ordered_pairs[:i+1]))
      to_apply.append((MovementOps.EXPAND, (*(s[0] for s in ordered_pairs[:i+1]))))
      if i != len(ordered_pairs)-1: to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_pairs[:i+1]), 1))) # t = t.reshape(*(s[0] for s in ordered_pairs[:i+1]), 1)
  else:
    # final shrink and reshape
    to_apply.append((MovementOps.SHRINK, (*[(0,s[0]) for s in ordered_pairs], (0,1))))
    # t = t.shrink((*[(0,s[0]) for s in ordered_pairs], (0,1)))
    # t = t.reshape(tuple(s[0] for s in ordered_pairs))
    to_apply.append((MovementOps.RESHAPE, tuple(s[0] for s in ordered_pairs)))
  # permute if required
  if order != list(range(len(order))): to_apply.append((MovementOps.PERMUTE, tuple(order.index(i) for i in range(len(stride))))) # t = t.permute(tuple(order.index(i) for i in range(len(stride))))
  
  # for i, v in enumerate(st.views):
  #   real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
  #   offset = (v.offset or 0) + sum(st*(s-1) for s,st in zip(real_shape, v.strides) if st<0)
  #   real_offset = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
  #   real_real_shape = [s for s,st in zip(real_shape, v.strides) if st]
  #   strides: List[int] = [abs(st) if isinstance(st,int) else st for st in v.strides if st]
  #   buffer_size = sum((s-1)*st for s,st in zip(real_real_shape,strides)) + 1
  #   if i: buffer_size = prod(st.views[i-1].shape) - real_offset if real_shape else 1
  #   def sort_by_strides(shape, strides): return sorted(zip(shape, strides), key=lambda k: (k[1],-k[0]), reverse=True), sorted(range(len(strides)), key=lambda k: (strides[k],-real_real_shape[k]), reverse=True)
  #   ordered_shape_strides, order = sort_by_strides(real_real_shape, strides)
  #   to_apply.extend([(MovementOps.RESHAPE, (-1,)), (MovementOps.SHRINK, ((real_offset, real_offset+buffer_size),))])
  #   if strides:
  #     if (ordered_shape_strides[0][0]*ordered_shape_strides[0][1])-buffer_size>0: to_apply.append((MovementOps.PAD, ((0, (ordered_shape_strides[0][0] * ordered_shape_strides[0][1]) - buffer_size),)))
  #     for i, shape_stride in enumerate(ordered_shape_strides):
  #       if i<len(ordered_shape_strides)-1 and shape_stride[1] < ordered_shape_strides[i+1][0]*ordered_shape_strides[i+1][1]:
  #         remaining_buffer = ordered_shape_strides[i-1][1] if i>0 else buffer_size
  #         to_apply.append((MovementOps.EXPAND, (shape_stride[0], *(s[0] for s in ordered_shape_strides[:i]), remaining_buffer)))
  #         to_apply.append((MovementOps.PERMUTE, (*range(1,i+1), 0, i+1)))
  #         to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i]), shape_stride[0]*remaining_buffer)))
  #         to_apply.append((MovementOps.PAD, (*((0,0) for _ in range(i)), (0, shape_stride[0]*shape_stride[1]))))
  #         to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i+1]), remaining_buffer+shape_stride[1])))
  #         ordered_shape_strides[i] = (ordered_shape_strides[i][0], remaining_buffer+shape_stride[1])
  #       else:
  #         to_apply.append((MovementOps.SHRINK, (*((0, s[0]) for s in ordered_shape_strides[:i]), (0, shape_stride[0]*shape_stride[1]))))
  #         to_apply.append((MovementOps.RESHAPE, (*[s[0] for s in ordered_shape_strides[:i+1]], shape_stride[1])))
  #     to_apply.extend([(MovementOps.SHRINK, (*[(0, s[0]) for s in ordered_shape_strides], (0,1))), (MovementOps.RESHAPE, tuple(s[0] for s in ordered_shape_strides))])
  #     if order != list(range(len(order))): to_apply.append((MovementOps.PERMUTE, tuple(order.index(i) for i in range(len(strides)))))
  #   to_apply.append((MovementOps.RESHAPE, tuple(s if st else 1 for s,st in zip(real_shape, v.strides))))
  #   if any(i<0 for i in v.strides): to_apply.append((MovementOps.STRIDE, tuple(-1 if st<0 else 1 for st in v.strides)))
  #   # then, we apply pre expand pads
  #   if v.mask is not None:
  #     pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
  #     post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
  #     if any(x != (0,0) for x in pre_expand_pads):
  #       to_apply.append((MovementOps.PAD, pre_expand_pads))
  #       real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
  #   # then, we do any expands
  #   if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
  #   # lastly, we apply post expand pads
  #   if v.mask is not None and any(x != (0,0) for x in post_expand_pads): to_apply.append((MovementOps.PAD, post_expand_pads))

  # scratch_st = make_scratch_st(st)
  # ret = []
  # seen = {}  # {shapetracker: list of mops to generate that shapetracker}
  # for mop_arg in to_apply:
  #   scratch_st = apply_mop(scratch_st, mop_arg)
  #   if scratch_st in seen:
  #     ret = seen[scratch_st][:]
  #   else:
  #     if len(ret) and ret[-1][0] == MovementOps.RESHAPE and mop_arg[0] == MovementOps.RESHAPE:
  #       ret[-1] = mop_arg
  #     else:
  #       if mop_arg == (MovementOps.RESHAPE, -1): mop_arg = (MovementOps.RESHAPE, (prod(st.shape),))
  #       ret.append(mop_arg)
  #     seen[scratch_st] = ret[:]
  # return ret

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
    var_vals = {k.expr:v for k,v in zip(vs, ranges)}
    r1 = sym_infer(idx1, var_vals) if sym_infer(valid1, var_vals) else 0
    r2 = sym_infer(idx2, var_vals) if sym_infer(valid2, var_vals) else 0
    if r1 != r2: return False

  return True

c: DefaultDict[int,int] = defaultdict(int)
def test_rebuild(st: ShapeTracker):
  rebuilt_st = make_scratch_st(st)
  mops = to_movement_ops(st)
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
  from extra.optimization.helpers import load_worlds, ast_str_to_ast
  ast_strs = load_worlds(False, False, True)[:2000]
  for ast_str in tqdm(ast_strs):
    test_rebuild_bufferop_st(ast_str_to_ast(ast_str))

  print(f"avg length of mop = {sum(k*v for k,v in c.items()) / sum(c.values()):.2f}")
