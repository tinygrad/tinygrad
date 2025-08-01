from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, KernelInfo, GroupOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import argsort

def rangify_store(ctx:list[int], x:UOp):
  if len(x.src) != 2: return None
  ranges = []
  for s in x.shape:
    if resolve(s!=1):
      ranges.append(UOp.range(dtypes.int, s, ctx[0]))
      ctx[0] += 1
    else:
      ranges.append(UOp.const(dtypes.int, 0))
  mm = UOp(Ops.INDEX, dtype=x.src[0].dtype, src=(x.src[0],)+tuple(ranges))
  mm2 = UOp(Ops.INDEX, dtype=x.src[0].dtype, src=(x.src[1],)+tuple(ranges))
  return UOp(Ops.STORE, src=(mm, mm2)+tuple(ranges))

def map_reshape(x:UOp):
  r = x.src[0]
  acc = 1
  to_sum = []
  for s,src in list(zip(x.shape, x.src[1:]))[::-1]:
    to_sum.append(acc*src)
    acc *= s
  mish = sum(to_sum)
  ret = []
  for s in x.src[0].src[0].shape[::-1]:
    if resolve(s!=1):
      ret.append(mish % s)
      mish //= s
    else:
      ret.append(UOp.const(dtypes.int, 0))
  ret = UOp.sink(*ret).simplify().src[::-1] if len(ret) else ()
  return UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret))

def map_expand(x:UOp):
  r = x.src[0]
  inp_shape, exp_shape = x.src[0].src[0].shape, x.src[0].shape
  ret = list(x.src[1:])
  exp_ranges = []
  for i,(x,y) in enumerate(zip(inp_shape, exp_shape)):
    if x != y:
      exp_ranges.append(ret[i])
      ret[i] = UOp.const(dtypes.int, 0)
  return UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret))

def map_permute(x:UOp):
  r = x.src[0]
  ret = x.src[1:]
  # argsort or not?
  perm = argsort(x.src[0].arg)
  ret = tuple([ret[p] for p in perm])
  return UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret))

pm_rangeify = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(arg=KernelInfo()) if x.arg is None else None),
  # TODO: handle MAP on STORE
  (UPat(Ops.STORE, name="x"), rangify_store),
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE),), allow_any_len=True, name="x"), map_permute),
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE),), allow_any_len=True, name="x"), map_reshape),
  (UPat(Ops.INDEX, src=(UPat(Ops.EXPAND),), allow_any_len=True, name="x"), map_expand),
  # TODO: CONST shouldn't have src
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),)), lambda c: c.replace(src=())),
  # move MAP through elementwise ALU
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.LOAD})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([UOp(Ops.INDEX, dtype=s.dtype, src=(s,)+x.src[1:]) for s in x.src[0].src]))),
])