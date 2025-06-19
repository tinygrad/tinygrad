from dataclasses import dataclass, field
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, graph_rewrite
from tinygrad.shape.shapetracker import ShapeTracker

@dataclass
class LowererContext:
  current_range: list[UOp]=field(default_factory=list)
  range_number: int = 0

def _count_ranges(lst:list[UOp]): return sum([1 for x in lst if x.op is Ops.RANGE])

def add_store_indexing(ctx:LowererContext, store:UOp, buf:UOp):
  # this is always contiguous
  st = ShapeTracker.from_shape(store.src[1].shape)
  # create the output range
  ctx.current_range = [UOp.range(dtypes.int, s, i) if s != 1 else UOp.const(dtypes.int, 1) for i,s in enumerate(st.shape)]
  ctx.range_number = _count_ranges(ctx.current_range)
  idx, valid = st.to_indexed_uops(ctx.current_range)
  return store.replace(src=(buf.index(idx, valid),)+store.src[1:])

def add_reduce_indexing(ctx:LowererContext, red:UOp):
  more_shape = red.src[0].shape[len(ctx.current_range):]
  reduce_range = [UOp.range(dtypes.int, s, ctx.range_number+i) if s != 1 else UOp.const(dtypes.int, 1) for i,s in enumerate(more_shape)]
  lc = LowererContext(ctx.current_range+reduce_range, ctx.range_number+_count_ranges(reduce_range))
  from tinygrad.codegen.lowerer2 import pm_lowerer  # TODO: better way to do this?
  ret = graph_rewrite(red.src[0], pm_lowerer, lc, name="subreduce", bottom_up=True)
  ctx.range_number = lc.range_number
  return ret.reduce(*reduce_range, arg=red.arg[0])

def view_const(ctx:LowererContext, view:UOp, c:UOp):
  if all(x.mask is None for x in view.arg.views): return c
  _, valid = view.arg.to_indexed_uops(ctx.current_range)
  return valid.where(c, c.const_like(0))

def view_buffer(ctx:LowererContext, view:UOp, buf:UOp):
  idx, valid = view.arg.to_indexed_uops(ctx.current_range)
  return buf.index(idx, valid).load()

pm_lowerer = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat.any(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat(Ops.DEFINE_GLOBAL, name="buf").view()), UPat()),
        name="store"), add_store_indexing),
  (UPat(Ops.REDUCE_AXIS, name="red"), add_reduce_indexing),
  (UPat(Ops.VIEW, src=(UPat.cvar("c"),), name="view"), view_const),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"),), name="view").load(), view_buffer),
])
