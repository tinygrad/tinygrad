import random
from tqdm import tqdm
from extra.optimization.helpers import load_worlds
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.ops import LazyOp, MovementOps, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes, prod
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Node, Variable
inf, nan = float('inf'), float('nan')

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

def flatten_view(view: View):
  real_real_shape, strides, real_offset = get_real_view(view.shape, view.strides, view.offset, view.mask)
  def sort_by_strides(shape, strides): return sorted(zip(shape, strides), key=lambda k: (k[1],-k[0]), reverse=True), sorted(range(len(strides)), key=lambda k: (strides[k],-real_real_shape[k]), reverse=True)
  ordered_shape_strides, _ = sort_by_strides(real_real_shape, strides)
  ordered_shape_strides = [list(s) for s in ordered_shape_strides]
  if strides:
    i = 0
    while i < len(ordered_shape_strides):
      if i<len(ordered_shape_strides)-1 and ordered_shape_strides[i][1] == ordered_shape_strides[i+1][0]*ordered_shape_strides[i+1][1]:
        ordered_shape_strides[i+1][0] = ordered_shape_strides[i][0]*ordered_shape_strides[i+1][0]
      else: i += 1
    flat_shape = [shape_stride[0] for shape_stride in ordered_shape_strides]
    flat_strides = [shape_stride[1] for shape_stride in ordered_shape_strides]
    return (flat_shape, flat_strides, real_offset)
  return (real_real_shape, view.strides, real_offset)

def views_equivalent(v1: View, v2: View) -> bool:
  return v1 == v2 or flatten_view(v1) == flatten_view(v2)


def st_equivalent(st: ShapeTracker, st_rebuilt: ShapeTracker):
  views = list(st.views)
  rebuilt_views = list(st_rebuilt.views)
  i = 0
  while i < len(views):
    view, rebuilt_view = views[i], rebuilt_views[i]
    if view == rebuilt_view:
      i += 1
      continue
    elif view.shape == rebuilt_view.shape:
      i += 1
    # hack to skip expands for overlapped strides
    else:
      rebuilt_views.pop(i)
  return True

def test_rebuild(st: ShapeTracker):
  rebuilt_st = ShapeTracker.from_shape((get_buffer_size(st.views[0].shape, st.views[0].strides, st.views[0].offset, st.views[0].mask),))
  for mop, arg in st.to_movement_ops():
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
  if len(st.views) != len(rebuilt_st.views):
    if not set(st.views).issubset(set(rebuilt_st.views)):
      assert st_equivalent(st, rebuilt_st)
  else:
    for v1,v2 in zip(st.views, rebuilt_st.views):
      assert views_equivalent(v1, v2), f"{v1} not equivalent to {v2}"
  last_v1 = st.views[-1]
  last_v2 = rebuilt_st.views[-1]
  assert last_v1.shape == last_v2.shape, f"{last_v1.shape} != {last_v2.shape}"

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, True)
  random.shuffle(ast_strs)
  ast_strs = ast_strs[:2000]
  def interpret_ast(ast):
    if ast.op in BufferOps:
      test_rebuild(ast.arg.st)
    else:
      for src in ast.src: interpret_ast(src)
  for ast_str in tqdm(ast_strs):
    ast = eval(ast_str)
    interpret_ast(ast)
