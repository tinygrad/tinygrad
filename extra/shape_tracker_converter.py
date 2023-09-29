from math import prod
from itertools import zip_longest
from typing import List, Tuple

from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import strides_for_shape
from tinygrad.ops import MovementOps

def find_permutation(old_shape, new_shape):
  indices_dict = {element: tuple(index for index, value in enumerate(old_shape) if value == element) for element in new_shape}
  return tuple(index for indices in indices_dict.values() for index in indices)

def convert_st_to_movement_ops(st: ShapeTracker, prev_ops: List[Tuple[MovementOps,Tuple[int, ...]]] = [], initial_call:bool = True) -> List[Tuple[MovementOps, Tuple[int,...]]]:
  # Identify ops applied to st from most recent to oldest. Track changes to st by applying reverse operations.
  # TO DO: make this line work
  # if prev_ops == []: st = deepcopy(st)
  # Stop space from polluting. TO DO: remove this line.
  if initial_call: prev_ops = []
  if any(dim==1 for dim in st.views[-1].shape):
    prev_ops.append((MovementOps.RESHAPE, st.views[-1].shape))
    return convert_st_to_movement_ops(st.reshape(tuple(filter(lambda x: x != 1, st.views[-1].shape))), prev_ops, False)
  # identify expands if stride is 0 on a dimension != 1
  strides = st.views[-1].strides
  default_strides = strides_for_shape(st.views[-1].shape)
  if ((reduced_shape := tuple(dim if stride != 0 else 1 for stride,dim in zip(strides, st.shape))) != st.shape):
    if len(prev_ops) == 0 or prev_ops[-1][0] != MovementOps.EXPAND:
      prev_ops.append((MovementOps.EXPAND, st.shape))
      return convert_st_to_movement_ops(st.shrink(tuple((0,dim) for dim in reduced_shape)), prev_ops, initial_call=False)
  if any(i < 0 for i in strides):
    stride_arg = tuple(1 if i>=0 else -1 for i in strides)
    prev_ops.append((MovementOps.STRIDE, stride_arg))
    return convert_st_to_movement_ops(st.stride(stride_arg), prev_ops, False)
   # strides should no longer have values > 0 after this point. Uncomment the following line if ever in doubt.
  assert all(stride > 0 for stride in strides)
  if (offset := st.views[-1].offset) != 0:
    # only pad produces masks.
    if offset < 0 and st.views[-1].mask is not None:
      if len(prev_ops) == 0 or prev_ops[-1][0] != MovementOps.PAD:
        pad_arg = tuple((x[0], y-x[1]) for x,y in zip(st.views[-1].mask, st.views[-1].shape))
        prev_ops.append((MovementOps.PAD,pad_arg))
        return convert_st_to_movement_ops(st.shrink(st.views[-1].mask), prev_ops, False)
    # identify shrinks by prod of strides
    if offset > 0:
      if len(prev_ops) == 0 or prev_ops[-1][0] != MovementOps.SHRINK:
        total = st.views[-1].offset
        shrink_first_arg = []
        for stride in strides:
          if total > stride:
            idx = total // stride
            shrink_first_arg.append(idx)
            total -= idx*stride
          else:
            shrink_first_arg.append(0)
        shrink_arg = tuple((x, x+y) for x,y in zip(shrink_first_arg, st.views[-1].shape))
        # TO DO: first dimension is not well defined in current format.
        expanded_shape = (shrink_arg[0][1],) + tuple(int(strides[i]/strides[i+1]) for i in range(len(strides)-1))
        expand_arg = tuple((x[0],y-x[1]) for x,y in zip(shrink_arg, expanded_shape))
        prev_ops.append((MovementOps.SHRINK, shrink_arg))
        return convert_st_to_movement_ops(st.pad(expand_arg), prev_ops, False)

  if strides != default_strides:
    # only identify negative strides since technically the rest is a combination of other movement ops.
    # identify permutations by strides not being ordered. TO DO: this breaks when a shape includes a dim=1.
    if (sorted_strides := tuple(reversed(sorted(strides)))) != strides:
      if len(prev_ops) == 0 or prev_ops[-1][0] != MovementOps.PERMUTE:
        permutation = find_permutation(sorted_strides, strides)
        prev_ops.append((MovementOps.PERMUTE, permutation))
        inv_permutation = find_permutation(permutation, tuple(i for i in range(len(permutation))))
        return convert_st_to_movement_ops(st.permute(inv_permutation), prev_ops, initial_call=False)

  if len(st.views) >= 2:
    # identify reshapes by multiple views.
    if prod(st.views[-1].shape) == prod(st.views[-2].shape):
      if len(prev_ops) == 0 or prev_ops[-1][0] != MovementOps.RESHAPE:
        prev_ops.append((MovementOps.RESHAPE, st.views[-1].shape))
        prev_st = ShapeTracker(st.views[:-1])
        return convert_st_to_movement_ops(prev_st, prev_ops, initial_call=False)
      
  if reshapes := tuple(filter(lambda x: x[0]==MovementOps.RESHAPE and 1 in x[1], prev_ops)):
    idxs = set(y.index(1) for x,y in reshapes)
    reshape_arg = list(st.views[-1].shape)
    for idx in idxs: reshape_arg.insert(idx, 1)
    prev_ops.append((MovementOps.RESHAPE, tuple(reshape_arg)))
    
  # ops have been identified and appended in reverse order
  return prev_ops[::-1]