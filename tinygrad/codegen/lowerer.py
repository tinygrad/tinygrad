# the job of the lowerer is to do indexing
from dataclasses import dataclass
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, AxisType, graph_rewrite

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
  #if len(ast.full_shape) != len(axis_types) and ast.st is not None:
  #  axis_types = tuple([AxisType.REDUCE if resolve(s != fs) else AxisType.LOOP for s,fs in zip(ast.shape, ast.full_shape)])
  return IndexContext(axis_types, [], 0)

# ***** lowering (given index) *****

def subblock(ctx: IndexContext, full_new_idx: list[UOp], src: UOp):
  lc = IndexContext(ctx.axis_types, full_new_idx, ctx.start+1000)
  ctx.start = lc.start
  return graph_rewrite(src, pm_lowerer, lc, name="subblock", bottom_up=True)

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
  (UPat(Ops.WMMA, name="x"), fixup_wmma),

  # axis fixups for WMMA
  (UPat((Ops.CONTRACT, Ops.UNROLL), name="x"),
   lambda ctx,x: x.replace(tag=1, arg=tuple([(ctx.idxs[a].arg[0], sz) for a,sz in x.arg])) if x.tag is None else None),
])
