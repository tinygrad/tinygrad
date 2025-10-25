import itertools
from enum import Enum, auto
from collections import defaultdict
from typing import List, Tuple, DefaultDict
from tinygrad.helpers import prod, tqdm
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.ops import sym_infer
from tinygrad.tensor import Tensor

class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702

def apply_mop(st: Tensor, mop_arg: Tuple[MovementOps, Tuple]):
  """Apply a movement operation to a Tensor or ShapeTracker."""
  mop, arg = mop_arg
  if mop == MovementOps.RESHAPE:
    if arg == (-1,): return st.reshape((prod(st.shape),))
    return st.reshape(arg)
  if mop == MovementOps.PERMUTE: return st.permute(arg)
  if mop == MovementOps.EXPAND:
    if len(arg) != len(st.shape): st = st.reshape((1,*st.shape))
    return st.expand(arg)
  if mop == MovementOps.PAD: return st.pad(arg)
  if mop == MovementOps.SHRINK: return st.shrink(arg)
  if mop == MovementOps.STRIDE:
    assert all(x in [-1, 1] for x in arg)
    return st.flip(tuple(i for i,x in enumerate(arg) if x == -1))
  if mop == MovementOps.AS_STRIDED:
    # arg = (size, stride, storage_offset)
    size, stride, offset = arg
    real_strides = [abs(s) for s in stride if s != 0]
    if not real_strides: offset, buf_sz = offset, 1
    else: offset, buf_sz = offset + sum(s * (sz-1) for s, sz in zip(stride, size) if s < 0), sum((sz-1) * abs(s) for sz, s in zip(size, stride) if s != 0) + 1
    # Flatten, shrink to buffer, build dims by stride order, handle broadcast/flip
    st = st.reshape((-1,)).shrink(((offset, offset + buf_sz),))
    dims = sorted([(i, sz, abs(s)) for i, (sz, s) in enumerate(zip(size, stride)) if s != 0], key=lambda x: (x[2], -x[1]), reverse=True)
    if dims and (dims[0][1] * dims[0][2] - buf_sz) > 0: st = st.pad(((0, dims[0][1] * dims[0][2] - buf_sz),)); buf_sz = dims[0][1] * dims[0][2]
    # Build dimensions, handling overlapping strides
    built_shapes = []
    for i, (_, sz, s) in enumerate(dims):
      if i < len(dims) - 1 and s < dims[i+1][1] * dims[i+1][2]:
        # Overlapping stride - use expand approach
        # First reshape to add a dimension of size 1, then expand it
        last_dim = st.shape[-1] if i > 0 else buf_sz
        st = st.reshape((1, *built_shapes, last_dim)).expand((sz, *built_shapes, last_dim))
        # Permute to move the new dim to the end, flatten it
        st = st.permute((*range(1, i+1), 0, i+1)).reshape((*built_shapes, sz * last_dim))
        # Pad and reshape to the final size
        st = st.pad((*[(0,0) for _ in range(i)], (0, sz * s))).reshape((*built_shapes, sz, last_dim + s))
        buf_sz = last_dim + s
        built_shapes.append(sz)
      else:
        # Non-overlapping stride
        st = st.shrink((*[(0, st.shape[j]) for j in range(i)], (0, sz * s))).reshape((*st.shape[:i], sz, s))
        buf_sz = s  # Update buf_sz for next iteration
        built_shapes.append(sz)
    st = st.shrink((*[(0, d) for d in st.shape[:-1]], (0, 1))).reshape(st.shape[:-1])
    # Permute to restore original order (only for non-zero stride dims)
    orig_order = [dims[i][0] for i in range(len(dims))]
    if orig_order != sorted(orig_order):
      perm = [orig_order.index(i) for i in sorted(orig_order)]
      st = st.permute(tuple(perm))
    # Add broadcast dimensions (stride 0)
    for i, (sz, s) in enumerate(zip(size, stride)):
      if s == 0: pos = sum(1 for j in range(i) if stride[j] != 0); st = st.reshape((*st.shape[:pos], 1, *st.shape[pos:])).expand((*st.shape[:pos], sz, *st.shape[pos:]))
    if any(s < 0 for s in stride): st = st.flip(tuple(i for i, s in enumerate(stride) if s < 0))
    return st.reshape(size) if st.shape != size else st
  raise ValueError("invalid mop")


c: DefaultDict[int,int] = defaultdict(int)

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
