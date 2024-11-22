from tinygrad.ops import UOp, PatternMatcher, UPat, Ops
import math

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: ctx.cast(ret.src[0].dtype)),
  (UPat(Ops.RECIP, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: (math.pi/2 - ret.src[0]).sin() * ctx),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: ctx / (ret.src[0] * math.log(2))),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: ret.src[0] * ctx * math.log(2)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: ctx / (ret*2)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  # TODO: reduce/movement ops once we have the new lazy stuff
])

def gradient(root:UOp, t:list[UOp]) -> list[UOp]:
  root.parents
