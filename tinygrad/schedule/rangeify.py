from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp
from tinygrad.helpers import argsort

@dataclass
class RangeifyContext:
  idx: int = 0
  regs: int = 0

def map_contiguous(ctx:RangeifyContext, x:UOp, idx:UOp|None=None):
  if x.tag == 1: return None
  ranges = []
  new_ranges = []
  passthrough_idx = []
  for i,s in enumerate(x.shape):
    if x.arg is not None and i not in x.arg:
      assert idx is not None, "partial contig requires index"
      ranges.append(idx.src[1+i])
      continue
    if idx is not None: passthrough_idx.append(idx.src[1+i])
    if resolve(s!=1):
      ranges.append(UOp.range(dtypes.int, s, ctx.idx))
      new_ranges.append(ranges[-1])
      ctx.idx += 1
    else:
      ranges.append(UOp.const(dtypes.int, 0))
  ret = x.src[0].index(*ranges).contiguous(arg=x.arg, tag=1)
  ret = ret.replace(src=(ret.src[0],)+tuple(new_ranges))
  return ret.index(*passthrough_idx) if idx is not None else ret

def map_reshape(x:UOp, r:UOp):
  acc = 1
  to_sum = []
  for s,src in list(zip(x.shape, x.src[1:]))[::-1]:
    to_sum.append(acc*src)
    acc *= s
  mish = sum(to_sum)
  ret = []
  for s in x.src[0].src[0].shape[::-1]:
    if resolve(s!=1):
      # this MOD should limit any ranges outside s
      ret.append(mish % s)
      mish //= s
    else:
      ret.append(UOp.const(dtypes.int, 0))
  ret = UOp.sink(*ret).simplify().src[::-1] if len(ret) else ()
  return UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret))

def map_pad(x:UOp, r:UOp):
  ret = list(x.src[1:])
  bigwhere = UOp.const(dtypes.bool, True)
  for i,(sh,(s,e)) in enumerate(zip(r.shape, r.arg)):
    if s == 0 and e == 0: continue
    where = UOp.const(dtypes.bool, True)
    if e > 0: where = where & (ret[i] < (sh-e))
    if s > 0: where = where & (ret[i] >= s)
    bigwhere = bigwhere & where
    # this is safe but dumb
    ret[i] = (ret[i] - s).maximum(0).minimum(r.src[0].shape[i]-1)
    # mask the load
    #ret[i] = where.where(ret[i], UOp(Ops.INVALID, dtype=ret[i].dtype))
  # PAD is with 0
  return bigwhere.simplify().where(UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret)), UOp.const(r.dtype, 0))

def map_reduce(ctx:RangeifyContext, idx:UOp, red:UOp):
  rngs = list(idx.src[1:])
  new_ranges = []
  for i,s in enumerate(red.src[0].shape):
    if i in red.arg[1]:
      rngs[i] = UOp.range(dtypes.int, s, ctx.idx)
      ctx.idx += 1
      new_ranges.append(rngs[i])
  return UOp(Ops.REDUCE, red.dtype, src=(red.src[0].index(*rngs),)+tuple(new_ranges), arg=red.arg[0])

pm_rangeify = PatternMatcher([
  # if there's an INDEX it can support partial contig
  (UPat(Ops.INDEX, src=(UPat(Ops.CONTIGUOUS, name="x"),), allow_any_len=True, name="idx"), map_contiguous),
  (UPat(Ops.CONTIGUOUS, name="x"), map_contiguous),

  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="red"),), allow_any_len=True, name="idx"), map_reduce),

  # this is like the definitions of these
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple([x.src[1+p] for p in argsort(x.src[0].arg)]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.SHRINK, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple([a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(x.src[1:], r.arg)]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.FLIP, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple([((s-1)-a) if f else a for a,s,f in zip(x.src[1:], r.shape, r.arg)]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.EXPAND, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+
                   tuple([a.const_like(0) if resolve(x!=y, False) else a for a,x,y in zip(x.src[1:], r.src[0].shape, r.shape)]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE, name="r"),), allow_any_len=True, name="x"), map_reshape),
  (UPat(Ops.INDEX, src=(UPat(Ops.PAD, name="r"),), allow_any_len=True, name="x"), map_pad),

  # move MAP through elementwise ALU
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.STORE})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([UOp(Ops.INDEX, dtype=s.dtype, src=(s,)+x.src[1:]) for s in x.src[0].src]))),

  # CONST can't have axes
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST,name="c"),)), lambda c: c),
])