from dataclasses import dataclass, field
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, graph_rewrite, KernelInfo, GroupOp
from tinygrad.shape.shapetracker import ShapeTracker

@dataclass
class LowererContext:
  current_range: list[UOp]=field(default_factory=list)
  range_number: int = 0
  kernel_info: KernelInfo = KernelInfo()

def _count_ranges(lst:list[UOp]): return sum([1 for x in lst if x.op is Ops.RANGE])

def add_store_indexing(ctx:LowererContext, store:UOp, buf:UOp, v:UOp):
  if not len(ctx.current_range):
    # create the output range
    ctx.current_range = [UOp.range(dtypes.int, s, i) for i,s in enumerate(v.arg.shape)]
    ctx.range_number = _count_ranges(ctx.current_range)
  idx, valid = v.arg.to_indexed_uops(ctx.current_range)
  return store.replace(src=(buf.index(idx, valid),)+store.src[1:])

def add_reduce_indexing(ctx:LowererContext, red:UOp):
  # NOTE: should never be range 1 by earlier rule
  reduce_range = ctx.current_range[:]
  final_reduce_ranges = []
  for i,axis in enumerate(red.arg[1]):
    final_reduce_ranges.append(UOp.range(dtypes.int, red.src[0].shape[axis], ctx.range_number+i))
    reduce_range[axis] = final_reduce_ranges[-1]
  lc = LowererContext(reduce_range, ctx.range_number+len(final_reduce_ranges))
  from tinygrad.codegen.lowerer2 import pm_lowerer  # TODO: better way to do this?
  ret = graph_rewrite(red.src[0], pm_lowerer, lc, name="subreduce", bottom_up=True)
  ctx.range_number = lc.range_number
  return ret.reduce(*final_reduce_ranges, arg=red.arg[0])

def view_const(ctx:LowererContext, view:UOp, c:UOp):
  if all(x.mask is None for x in view.arg.views): return c
  _, valid = view.arg.to_indexed_uops(ctx.current_range)
  return valid.where(c, c.const_like(0))

def view_buffer(ctx:LowererContext, view:UOp, buf:UOp):
  idx, valid = view.arg.to_indexed_uops(ctx.current_range)
  return buf.index(idx, valid).load()

pm_lowerer = PatternMatcher([
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for sometimes having a view on the store and sometimes not
  (UPat(Ops.STORE, src=(UPat(GroupOp.Ptr, name="buf").view().named("v"), UPat()), name="store"), add_store_indexing),
  (UPat(Ops.REDUCE_AXIS, name="red"), add_reduce_indexing),
  (UPat(Ops.VIEW, src=(UPat.cvar("c"),), name="view"), view_const),
  (UPat(Ops.VIEW, src=(UPat(GroupOp.Ptr, name="buf"),), name="view").load(), view_buffer),
])
