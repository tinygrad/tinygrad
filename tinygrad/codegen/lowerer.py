# the job of the lowerer is to do indexing
from dataclasses import dataclass
from typing import cast
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, AxisType, graph_rewrite
from tinygrad.helpers import prod, partition, flatten

# ***** indexing *****

@dataclass
class IndexContext:
  axis_types: list[AxisType]
  idxs: list[UOp]
  start: int = 0
  #ridxs: list[UOp]

def shape_to_idx(s, axis_types, start=0, allow_unroll=False):
  idxs = []
  for i, (s, at) in enumerate(zip(s, axis_types)):
    if at in (AxisType.UPCAST, AxisType.UNROLL) and allow_unroll:
      assert isinstance(s, int), "needs to be int to upcast/unroll"
      idxs.append(UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((i,s),)))
    else:
      # all others are RANGES
      idxs.append(UOp(Ops.RANGE, dtypes.int, (sint_to_uop(s),), start+i))
  return idxs

def get_index(ast:UOp) -> IndexContext:
  axis_types = ast.arg.axis_types if isinstance(ast.arg, KernelInfo) else ()
  if len(ast.full_shape) != len(axis_types): axis_types = (AxisType.LOOP,)*len(ast.full_shape)

  """
  # indexes
  idxs = []
  for i, (s, at) in enumerate(zip(ast.full_shape, axis_types)):
    if at in (AxisType.UPCAST, AxisType.UNROLL):
      assert isinstance(s, int), "needs to be int to upcast/unroll"
      idxs.append(UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((i,s),)))
    else:
      # all others are RANGES
      idxs.append(UOp(Ops.RANGE, dtypes.int, (sint_to_uop(s),), i))

  # late indexes (group for reduce)
  ridxs = idxs[:]
  for i, (s, at) in enumerate(zip(ast.full_shape, axis_types)):
    if at == AxisType.GROUP_REDUCE:
      ridxs[i] = UOp(Ops.RANGE, dtypes.int, (sint_to_uop(s),), 1000+i)
  """

  return IndexContext(axis_types, []) #idxs, ridxs)

# ***** lowering (given index) *****

def lower_reduce_into(ctx: IndexContext, x: UOp):
  print("reduce into")
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)

  # x.src[1] is the big
  axis_arg = [i for i,(s0,s1) in enumerate(zip(x.src[0].shape, x.src[1].shape)) if s0 != s1]
  reduce_idxs = shape_to_idx([s for i,s in enumerate(x.src[1].shape) if i in axis_arg],
                             [s for i,s in enumerate(ctx.axis_types) if i in axis_arg], ctx.start+len(new_idxs))
  new_idxs += reduce_idxs

  idx, valid = x.src[0].arg.to_indexed_uops(new_idxs)
  used_idxs = [x for x in UOp.sink(idx, valid).toposort() if x in new_idxs]

  real_new_idxs = []
  for i in range(len(x.src[0].shape)):
    if new_idxs[i] in used_idxs or new_idxs[i] in reduce_idxs or len(ctx.idxs) <= i: real_new_idxs.append(new_idxs[i])
    else: real_new_idxs.append(ctx.idxs[i])
  non_replaced = [x for x in real_new_idxs if x not in ctx.idxs]

  acc = x.src[0].load(*reduce_idxs).alu(x.arg, x.src[1])

  lc = IndexContext(ctx.axis_types, tuple(real_new_idxs), ctx.start+len(new_idxs))
  from tinygrad.codegen.lowerer import pm_lowerer  # TODO: better way to do this?
  ret = graph_rewrite(acc, pm_lowerer, lc, name="subreduce", bottom_up=True)
  ctx.start = lc.start

  red = x.src[0].src[0].index(idx, valid).store(ret, *used_idxs, *reduce_idxs)
  return x.src[0].load(red)

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  new_idx = shape_to_idx([s for i,s in enumerate(x.src[0].shape) if i in x.axis_arg],
                         [s for i,s in enumerate(ctx.axis_types) if i in x.axis_arg], ctx.start)
  full_new_idx = list(ctx.idxs)
  for i,s in zip(x.axis_arg, new_idx):
    #assert len(full_new_idx) == i, f"len(full_new_idx) = {len(full_new_idx)}, trying to place {i}"
    #full_new_idx.append(s)
    full_new_idx[i] = s

  lc = IndexContext(ctx.axis_types, tuple(full_new_idx), ctx.start+len(new_idx))
  from tinygrad.codegen.lowerer import pm_lowerer  # TODO: better way to do this?
  ret = graph_rewrite(x.src[0], pm_lowerer, lc, name="subreduce", bottom_up=True)
  ctx.start = lc.start

  # NOTE: always using ridxs is fine here
  reduce_range, reduce_expand = partition([full_new_idx[i] for i in x.axis_arg], lambda y: y.op is Ops.RANGE)
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand} for {x.axis_arg}"
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple(reduce_range), x.arg[0])

def lower_load(ctx: IndexContext, x: UOp, buf: UOp):
  idx, valid = x.st_arg.to_indexed_uops(ctx.idxs)
  return UOp(Ops.LOAD, x.dtype, (buf.index(idx, valid),) + x.src[1:])

  #barrier = tuple([y.barrier() if buf.op is Ops.DEFINE_LOCAL else y for y in x.src[1:]])

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  assert x.src[1].shape == x.src[0].shape, f"shape mismatch on store {x.src[1].shape} != {x.src[0].shape}"
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  idx, valid = x.st_arg.to_indexed_uops(new_idxs)
  used_idxs = [x for x in UOp.sink(idx, valid).toposort() if x in new_idxs]
  real_new_idxs = []
  for i in range(len(x.src[0].shape)):
    if new_idxs[i] in used_idxs or len(ctx.idxs) <= i:
      real_new_idxs.append(new_idxs[i])
    else:
      real_new_idxs.append(ctx.idxs[i])
  print("got", len(real_new_idxs), len(used_idxs))
  lc = IndexContext(ctx.axis_types, tuple(real_new_idxs), ctx.start+len(new_idxs))
  from tinygrad.codegen.lowerer import pm_lowerer  # TODO: better way to do this?
  stored = graph_rewrite(x.src[1], pm_lowerer, lc, name="substore", bottom_up=True)
  ctx.start = lc.start
  return buf.index(idx, valid).store(stored, *[x for x in used_idxs if x.op is Ops.RANGE])

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
  (UPat(Ops.REDUCE_INTO, name="x"), lower_reduce_into),
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"), lower_const),
  # rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_load),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
])
