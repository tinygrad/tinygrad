# the job of the lowerer is to do indexing
from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, graph_rewrite
from tinygrad.helpers import prod, flatten

# ***** indexing *****

@dataclass
class IndexContext:
  idxs: list[UOp]
  ranges_used: int = 0
  upcasted: int = 0

def get_index(ast:UOp) -> IndexContext:
  ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
  return IndexContext([], upcasted=ki.upcasted)

# ***** lowering (given index) *****

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  ridxs = ctx.idxs[:]
  reduce_range, reduce_expand = [], []
  for i,axis in enumerate(x.arg[1]):
    s = x.src[0].shape[axis]
    if axis < (len(x.src[0].shape)-ctx.upcasted):
      ridxs[axis] = UOp.range(dtypes.int, s, ctx.ranges_used+i)
      reduce_range.append(ridxs[axis])
    else:
      ridxs[axis] = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((ctx.ranges_used+i,s),))
      reduce_expand.append(ridxs[axis])
  ctx.ranges_used += len(x.arg[1])

  ret = x.src[0]

  subctx = IndexContext(ridxs, ctx.ranges_used, ctx.upcasted)
  # TODO: better way to do this?
  from tinygrad.codegen.lowerer import pm_lowerer # pylint: disable=import-self
  ret = graph_rewrite(ret, pm_lowerer, subctx, name="subreduce", bottom_up=True)
  ctx.ranges_used = subctx.ranges_used

  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))

  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple(reduce_range), x.arg[0])

def lower_load(ctx: IndexContext, x: UOp, buf: UOp):
  idx, valid = x.src[0].st_arg.to_indexed_uops(ctx.idxs)
  barrier = (UOp(Ops.BARRIER, dtypes.void, (x.src[1],)),) if buf.op is Ops.DEFINE_LOCAL else ()
  return UOp(Ops.LOAD, x.dtype, (buf.index(idx, valid),) + barrier)

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  store_shape = x.src[1].shape
  first_upcasted = len(store_shape)-ctx.upcasted
  ctx.idxs = [UOp(Ops.RANGE, dtypes.int, (sint_to_uop(g),), i) for i,g in enumerate(store_shape[:first_upcasted])]
  ctx.idxs += [UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((i,s),)) \
              for i,s in enumerate(store_shape[first_upcasted:], start=first_upcasted) if isinstance(s, int)]
  ctx.ranges_used += len(ctx.idxs)
  idx, valid = x.st_arg.to_indexed_uops(ctx.idxs)
  return UOp(Ops.STORE, dtypes.void, (buf.index(idx, valid), x.src[1]))

def lower_const(ctx:IndexContext, view:UOp, c:UOp):
  if all(x.mask is None for x in view.arg.views): return c
  _, valid = view.arg.to_indexed_uops(ctx.idxs)
  return valid.where(c, c.const_like(0))

pm_lowerer = PatternMatcher([
  # TODO: remove these hacks
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat((Ops.DEFINE_VAR, Ops.CONST), src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),

  # reduce/view_const
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"), lower_const),
  # rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_load),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
])
