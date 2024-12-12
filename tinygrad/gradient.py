from typing import cast
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops
from tinygrad.shape.view import View
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.helpers import argsort, prod
import math

def reduce_gradient(ctx:UOp, ret:UOp):
  if ret.arg[0] == Ops.ADD: return (ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MAX:
    max_is_1s = ret.src[0].ne(ret.expand(ret.src[0].shape)).ne(ret.src[0].const_like(1).cast(dtypes.bool)).cast(ctx.dtype)
    div = max_is_1s.r(Ops.ADD, ret.arg[1]).expand(ret.src[0].shape)
    return ((max_is_1s/div) * ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MUL: return ((ctx * ret).expand(ret.src[0].shape) / ret.src[0],)

# NOTE: this is very similar to invert
def view_gradient(ctx:UOp, ret:UOp):
  assert len(ret.arg.views) == 1, "no multiview support yet"
  v = ret.arg.views[0]
  assert ctx.shape == v.shape, "grad_output shape must match output shape"
  # find all stride 0 and add sum (inverse of expand)
  out = ctx.r(Ops.ADD, tuple(i for i,(s,st) in enumerate(zip(v.shape, v.strides)) if s > 1 and st == 0))

  # invert
  iv = View.create(out.shape)
  if v.mask: iv = iv.shrink(v.mask)
  if 0 in iv.shape: return (ret.src[0].const_like(0),)
  iv = iv.stride(tuple(-1 if x < 0 else 1 for x in v.strides)).permute(argsort(tuple(-x if x > 0 else x for x in v.strides)))
  assert prod(iv.shape) == prod(ret.src[0].shape), f"shape mismatch? {iv.shape} vs {ret.src[0].shape}"
  return (out.view(ShapeTracker((iv,)).reshape(ret.src[0].shape)),)

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

