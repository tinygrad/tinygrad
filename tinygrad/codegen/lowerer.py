# the job of the lowerer is to do indexing
from dataclasses import dataclass
from collections import defaultdict
from tinygrad.dtype import dtypes, least_upper_dtype, ConstType
from tinygrad.device import is_dtype_supported
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, AxisType, graph_rewrite, resolve, GroupOp
from tinygrad.uop.symbolic import split_uop, parse_valid

# ***** indexing *****

@dataclass
class IndexContext:
  axis_types: tuple[AxisType, ...]
  idxs: list[UOp]
  start: int = 0

def shape_to_idx(s, axis_types, start=0):
  return [UOp.range(sint_to_uop(s), start+i, at) for i, (s, at) in enumerate(zip(s, axis_types))]

def get_index(ast:UOp) -> IndexContext:
  axis_types = ast.arg.axis_types if isinstance(ast.arg, KernelInfo) else ()
  if len(ast.full_shape) != len(axis_types) and ast.st is not None:
    axis_types = tuple([AxisType.REDUCE if resolve(s != fs) else AxisType.LOOP for s,fs in zip(ast.shape, ast.full_shape)])
  return IndexContext(axis_types, [], 0)

# ***** lowering (given index) *****

def subblock(ctx: IndexContext, full_new_idx: list[UOp], src: UOp):
  lc = IndexContext(ctx.axis_types, full_new_idx, ctx.start+1000)
  ctx.start = lc.start
  return graph_rewrite(src, pm_lowerer, lc, name="subblock", bottom_up=True)

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.axis_arg: full_new_idx[a] = new_idxs[a]
  ret = subblock(ctx, full_new_idx, x.src[0])
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple([full_new_idx[i] for i in x.axis_arg]), x.arg[0])

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  # TODO: reenable after REDUCE_AXIS is fixed
  #assert x.src[1].shape == x.src[0].shape, f"shape mismatch on store {x.src[1].shape} != {x.src[0].shape}"

  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  idx, valid = x.st_arg.to_indexed_uops(new_idxs)
  used_idxs = [x for x in UOp.sink(idx, valid).toposort() if x in new_idxs]
  real_new_idxs = []
  for i in range(len(x.src[0].shape)):
    if new_idxs[i] in used_idxs or len(ctx.idxs) <= i: real_new_idxs.append(new_idxs[i])
    else: real_new_idxs.append(ctx.idxs[i])

  stored = subblock(ctx, real_new_idxs, x.src[1])
  used_ranges = [x for x in used_idxs if x.op is Ops.RANGE]
  return buf.index(idx, valid).store(stored, *used_ranges)

def fixup_wmma(ctx:IndexContext, x:UOp):
  if x.tag is not None: return None
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.arg[-1]: full_new_idx[a] = new_idxs[a]

  srcs = subblock(ctx, full_new_idx, UOp.sink(*x.src)).src

  # NOTE: this assumes these are expanded. which now shouldn't change anything
  new_x_arg_m2 = tuple([tuple([(full_new_idx[a].arg[0], sz) for a,sz in v]) for v in x.arg[-2]])
  new_x_arg_m1 = tuple([full_new_idx[a].arg[0] for a in x.arg[-1]])
  return x.replace(src=srcs, arg=x.arg[:-2]+(new_x_arg_m2, new_x_arg_m1), tag=1)

pm_lowerer = PatternMatcher([
  # TODO: remove these hacks
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat((Ops.DEFINE_VAR, Ops.CONST), src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),

  # consts and loads
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"),
   lambda ctx,view,c: c if all(x.mask is None for x in view.arg.views) else view.arg.to_indexed_uops(ctx.idxs)[1].where(c, c.const_like(0))),
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"),
   lambda ctx,buf,x: UOp(Ops.LOAD, x.dtype, (buf.index(*x.st_arg.to_indexed_uops(ctx.idxs)),)+x.src[1:])),

  # reduce/view_const
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
  (UPat(Ops.WMMA, name="x"), fixup_wmma),

  # axis fixups for WMMA
  (UPat((Ops.CONTRACT, Ops.UNROLL), name="x"),
   lambda ctx,x: x.replace(tag=1, arg=tuple([(ctx.idxs[a].arg[0], sz) for a,sz in x.arg])) if x.tag is None else None),
])

def lower_alu_index_dtype(u: UOp, x:UOp, y:UOp, ctx, cond:UOp|None=None) -> UOp|None:
  if u.overflows(dtypes.int64): raise ValueError("indexing overflows int64")
  # cond is a UOp only if u is a WHERE
  casted_srcs = ((cond,) if cond is not None else ()) + (x.cast(dtypes.int64), y.cast(dtypes.int64))
  # TODO: use the default int dtype and try to promote untill you run out of dtypes
  if u.overflows(dtypes.int32):
    if not is_dtype_supported(dtypes.int64, ctx): raise ValueError(f"index overflows int32 and int64 is not supported on {ctx}")
    return u.replace(dtype=dtypes.int64.vec(u.dtype.count), src=casted_srcs)
  # if any inputs are int64 and this *doesn't* overflow, cast back to int
  if x.dtype == dtypes.int64 or y.dtype == dtypes.int64:
    return u.replace(dtype=dtypes.int64.vec(u.dtype.count), src=casted_srcs).cast(dtypes.int32)
  return u.replace(dtype=dtypes.int32.vec(u.dtype.count))

def fix_src_dtype(u,x,y):
  if x.dtype==y.dtype: return None
  dt = least_upper_dtype(x.dtype, y.dtype)
  return u.replace(src=(x.cast(dt), y.cast(dt)))

pm_lower_index_dtype = PatternMatcher([
  # There are no Unary ops at this point in symbolic, those are introduced in later
  (UPat(GroupOp.Binary, dtype=dtypes.index, name="u", src=(UPat.var("x"), UPat.var("y"))), lower_alu_index_dtype),
  (UPat(GroupOp.Comparison, name="u", src=(UPat.var("x"), UPat.var("y"))), fix_src_dtype),
  (UPat(Ops.WHERE, dtype=dtypes.index, src=(UPat.var("cond"), UPat.var("x"), UPat.var("y")), name="u"), lower_alu_index_dtype),
  (UPat((Ops.CONST, Ops.VCONST), dtype=dtypes.index, name="u"), lambda u: u.replace(dtype=dtypes.int32.vec(u.dtype.count))),
  # TODO: assert that these fit in int32 or promote
  (UPat((Ops.RANGE,), dtype=dtypes.index, src=(UPat.var("end")), name="r"), lambda r,end:
    r.replace(dtype=dtypes.int32.vec(r.dtype.count), src=(end.cast(dtypes.int32),))),
  (UPat(Ops.CAST, dtype=dtypes.index, src=(UPat.var("x", dtypes.ints),), name="u"), lambda u,x: x),
  (UPat(Ops.VECTORIZE, dtype=dtypes.index, name="u"), lambda u: u.replace(
    dtype=(dt:=least_upper_dtype(*[x.dtype for x in u.src])).vec(u.dtype.count), src=tuple(x.cast(dt) for x in u.src)))
])

def lower_index_dtype(ctx:str, buf:UOp, x:UOp, gate:UOp|None=None):
  # ctx is the device string
  bounds:defaultdict[UOp, list[ConstType|None]] = defaultdict(lambda: [None, None])

  def get_min_max(u:UOp) -> tuple[int,int]:
    v0, v1 = bounds[u]
    return (expr.vmin if v0 is None else v0, expr.vmax if v1 is None else v1)
  if gate is not None:
    for stmt in split_uop(gate, Ops.AND):
      try: expr, is_upper, c = parse_valid(stmt)
      except ValueError: continue  # cant parse this valid
      bounds[expr][int(is_upper)] = c
  subs = [UOp.variable(f"fake{i}", *get_min_max(k)) for i,k in enumerate(bounds.keys())]
  subs = [s.replace(dtype=dtypes.int64) if s.overflows(dtypes.int32) else s for s in subs]
  subs_dict = dict(zip(bounds.keys(), subs))
  x = x.substitute(subs_dict)
  x = x.substitute((index_subs:={u: UOp(Ops.NOOP, arg=u) for u in x.toposort() if u.op is Ops.INDEX}))
  x = graph_rewrite(x, pm_lower_index_dtype, ctx=ctx)
  x = x.substitute({v:k for k,v in index_subs.items()})
  x = x.substitute({v:k.cast(v.dtype) for k,v in subs_dict.items()})

  return buf.index(*((x, gate) if gate is not None else (x,)))

pm_lower_index_dtype_with_gate = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x", dtype=dtypes.index), UPat.var("gate"),)), lower_index_dtype),
])
