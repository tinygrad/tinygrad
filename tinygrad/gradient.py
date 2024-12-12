from typing import cast
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops
import math

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
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  # TODO: reduce/movement ops once we have the new lazy stuff
])

# copied from tensor.py, get relevant toposort of gradients
def _deepwalk(root:UOp, targets:list[UOp]):
  def _walk(node:UOp, visited:set[UOp]):
    visited.add(node)
    if any(x in node.toposort for x in targets if x is not node):
      for i in node.src:
        if i not in visited: yield from _walk(i, visited)
      yield node
  return list(_walk(root, set()))

def gradient(root:UOp, targets:list[UOp]) -> list[UOp]:
  # TODO: better error
  if not all(x in root.toposort for x in targets): raise RuntimeError("some gradient targets not found in parents")
  grads = {root: root.const_like(1.0)}
  for t0 in reversed(_deepwalk(root, targets)):
    lgrads: tuple[UOp, ...]|None = cast(tuple[UOp, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}")
    assert len(lgrads) == len(t0.src)
    for k,v in zip(t0.src, lgrads):
      if k in grads: grads[k] = grads[k] + v
      else: grads[k] = v
  return [grads[x] for x in targets]

