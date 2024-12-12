from typing import cast
from tinygrad.dtype import dtypes, sum_acc_dtype
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops
from tinygrad.helpers import prod, argsort
import math


# NOTE: this is very similar to invert
from typing import Tuple, List
from enum import Enum, auto
from tinygrad.ops import sint
from tinygrad.shape.shapetracker import ShapeTracker
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702

def to_movement_ops(st: ShapeTracker, in_shape) -> List[Tuple[MovementOps, Tuple]]:
  to_apply:List[Tuple[MovementOps, Tuple]] = []

  for i, v in enumerate(st.views):
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    offset = v.offset + sum(st*(s-1) for s,st in zip(real_shape, v.strides) if st<0)
    real_offset = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    real_real_shape = [s for s,st in zip(real_shape, v.strides) if st]

    # deal with strides?
    strides: List[sint] = [abs(st) if isinstance(st,int) else st for st in v.strides if st]
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

    # handle FLIP
    if any(i < 0 for i in v.strides): to_apply.append((MovementOps.STRIDE, tuple(-1 if st<0 else 1 for st in v.strides)))

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

  st = ShapeTracker.from_shape(in_shape)
  ret = []
  for (mop, arg) in to_apply:
    if mop == MovementOps.RESHAPE:
      if arg == (-1,): arg = (prod(st.views[-1].shape),)
      st = st.reshape(arg)
    if mop == MovementOps.PERMUTE: st = st.permute(arg)
    if mop == MovementOps.EXPAND: st = st.expand(arg)
    if mop == MovementOps.PAD: st = st.pad(arg)
    if mop == MovementOps.SHRINK: st = st.shrink(arg)
    if mop == MovementOps.STRIDE: st = st.stride(arg)
    ret.append((mop, arg, st.shape))
  return ret

def view_gradient(ctx:UOp, ret:UOp):
  assert ctx.shape == ret.shape, "grad_output shape must match output shape"

  forward_ops = to_movement_ops(ret.arg, ret.src[0].shape)
  shapes = [ret.src[0].shape] + [x for (_,_,x) in forward_ops]

  for (mop, arg, _), in_shape in reversed(list(zip(forward_ops, shapes[:-1]))):
    if mop == MovementOps.EXPAND:
      if (expanded_axes:=tuple(i for i, (si, so) in enumerate(zip(in_shape, arg)) if si != so)):
        ctx = ctx.cast(sum_acc_dtype(ctx.dtype)).r(Ops.ADD, expanded_axes).cast(ctx.dtype)
    elif mop == MovementOps.RESHAPE: ctx = ctx.reshape(in_shape)
    elif mop == MovementOps.PERMUTE: ctx = ctx.permute(argsort(arg))
    elif mop == MovementOps.PAD: ctx = ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(in_shape, arg)]))
    elif mop == MovementOps.SHRINK: ctx = ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(in_shape, arg)]))
    elif mop == MovementOps.STRIDE: ctx = ctx.stride(arg)
    assert ctx.shape == in_shape, f"shape mismatch {ctx.shape} != {in_shape}"

  return (ctx,)

def reduce_gradient(ctx:UOp, ret:UOp):
  if ret.arg[0] == Ops.ADD: return (ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MAX:
    max_is_1s = ret.src[0].ne(ret.expand(ret.src[0].shape)).ne(ret.src[0].const_like(1).cast(dtypes.bool)).cast(ctx.dtype)
    div = max_is_1s.r(Ops.ADD, ret.arg[1]).expand(ret.src[0].shape)
    return ((max_is_1s/div) * ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MUL: return ((ctx * ret).expand(ret.src[0].shape) / ret.src[0],)

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (UPat(Ops.RECIP, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.MAX, name="ret"), lambda ctx, ret: ((ret.src[0]>ret.src[1]).where(ctx, (ret.src[0]!=ret.src[1]).where(ctx.const_like(0), ctx * 0.5)),
                                                (ret.src[0]<ret.src[1]).where(ctx, (ret.src[0]!=ret.src[1]).where(ctx.const_like(0), ctx * 0.5)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.VIEW, name="ret"), view_gradient),
  (UPat(Ops.REDUCE_AXIS, name="ret"), reduce_gradient),
])

# copied from tensor.py, get relevant toposort of gradients
def _deepwalk(root:UOp, targets:list[UOp]):
  def _walk(node:UOp, visited:set[UOp]):
    visited.add(node)
    if node.op is Ops.DETACH: return
    if any(x in node.toposort for x in targets if x is not node):
      for i in node.src:
        if i not in visited: yield from _walk(i, visited)
      yield node
  return list(_walk(root, set()))

def gradient(root:UOp, targets:list[UOp]) -> list[UOp]:
  grads = {root: root.const_like(1.0)}
  for t0 in reversed(_deepwalk(root, targets)):
    if t0 not in grads: continue
    lgrads: tuple[UOp, ...]|None = cast(tuple[UOp, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}")
    assert len(lgrads) == len(t0.src)
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      if k in grads: grads[k] = grads[k] + v
      else: grads[k] = v
  ret = [grads.get(x, None) for x in targets]
  for i,x in enumerate(ret):
    if x is None: raise RuntimeError(f"{targets[i]}\n\nnot found in\n\n{root}")
  return ret

