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
import numpy as np

def get_buffer_size(v: View):
  real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
  offset = v.offset + sum(v.strides[i] * (real_shape[i]-1) for i in range(len(v.strides)) if v.strides[i]<0)
  real_offset = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
  real_real_shape = [s for s,st in zip(real_shape, v.strides) if st]
  strides = [abs(st) for st in v.strides if st and not isinstance(st, Node)]
  return real_offset + sum((s-1)*st for s, st in zip(real_real_shape,strides)) + 1

def flatten_view(view: View):
  buffer_size = get_buffer_size(view)
  real_shape = tuple(y-x for x,y in view.mask) if view.mask else view.shape
  offset = view.offset + sum(view.strides[i] * (real_shape[i]-1) for i in range(len(view.strides)) if view.strides[i]<0)
  real_offset = offset + (sum(x*st for (x,_),st in zip(view.mask, view.strides)) if view.mask else 0)
  real_real_shape = [s for s,st in zip(real_shape, view.strides) if st]
  strides = [abs(st) for st in view.strides if st and not isinstance(st, Node)]
  def sort_by_strides(shape, strides): return sorted(zip(shape, strides), key=lambda k: (k[1],-k[0]), reverse=True), sorted(range(len(strides)), key=lambda k: (strides[k],-real_real_shape[k]), reverse=True)
  ordered_shape_strides, order = sort_by_strides(real_real_shape, strides)
  if strides:
    i = 0
    while i < len(ordered_shape_strides):
      if i<len(ordered_shape_strides)-1 and ordered_shape_strides[i][1] == ordered_shape_strides[i+1][0]*ordered_shape_strides[i+1][1]:
        ordered_shape_strides[i+1][0] = ordered_shape_strides[i][0]*ordered_shape_strides[i+1][0]
      else: i += 1
    flat_shape = [shape_stride[0] for shape_stride in ordered_shape_strides]
    flat_strides = [shape_stride[1] for shape_stride in ordered_shape_strides]
    return (flat_shape, flat_strides, real_offset)
  return (real_shape, view.strides, real_offset)

def equivalent(v1: View, v2: View) -> bool:
  if v1 == v2: return True
  return flatten_view(v1) == flatten_view(v2)

def test_rebuild(st: ShapeTracker):
  rebuilt_st = ShapeTracker.from_shape((get_buffer_size(st.views[0]),))
  for mop, arg in st.to_movement_ops():
    match mop:
      case MovementOps.RESHAPE:
        # shapetracker doesn't allow flattening with -1 but required for MovementOps.RESHAPE
        if arg == (-1,):
          rebuilt_st = rebuilt_st.reshape((prod(rebuilt_st.views[-1].shape),))
        else:
          rebuilt_st = rebuilt_st.reshape(arg)
      case MovementOps.PERMUTE:
        rebuilt_st = rebuilt_st.permute(arg)
      case MovementOps.EXPAND:
        if len(arg) != len(rebuilt_st.shape):
          rebuilt_st = rebuilt_st.reshape((1,*rebuilt_st.shape))
        rebuilt_st = rebuilt_st.expand(arg)
      case MovementOps.PAD:
        rebuilt_st = rebuilt_st.pad(arg)
      case MovementOps.SHRINK:
        rebuilt_st = rebuilt_st.shrink(arg)
      case MovementOps.STRIDE:
        rebuilt_st = rebuilt_st.stride(arg)
      case _:
        raise Exception("invalid mop")

  rebuilt_st = rebuilt_st.simplify()
  for v1,v2 in zip(st.views, rebuilt_st.views):
    assert equivalent(v1, v2), f"{v1} not equivalent to {v2}"
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
