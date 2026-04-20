from tinygrad.uop.ops import PatternMatcher, UPat, Ops, AxisType, UOp, GroupOp, _align_left, _broadcast_shape
from tinygrad.helpers import all_same

def build_range_map(ctx, sink:UOp):
  for x in sink.toposort():
    if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}:
      ctx[x.arg[0]] = len(ctx)

expander2 = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), build_range_map),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(r.dtype.vec(s:=r.vmax+1), tuple(range(s))) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
])

def broadcast_binary(x:UOp):
  shapes = [u.shape for u in x.src]
  if all_same(shapes): return None
  shaped_aligned = _align_left(*shapes)
  broadcasted = _broadcast_shape(*shapes)
  src_reshaped = [u.reshape(shp).expand(broadcasted) for u,shp in zip(x.src, shaped_aligned)]
  return x.replace(src=tuple(src_reshaped))

expander_broadcast = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary, name="x"), broadcast_binary),
])