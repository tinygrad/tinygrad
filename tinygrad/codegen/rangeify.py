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


def map_shrink(ctx:list[int], x:UOp):
  r = x.src[0]
  ret = list(x.src[1:])
  for i,(s,(ss,se)) in enumerate(zip(r.src[0].shape, r.arg)):
    assert ss == 0, "add to range?"
    if se-ss != s and False:
      new_ret_i = [ret[i]]
      if ss != 0:
        new_ret_i = [UOp.range(dtypes.int, ss, ctx[0])] + new_ret_i
        ctx[0] += 1
      if se != s:
        new_ret_i = new_ret_i + [UOp.range(dtypes.int, s-se, ctx[0])]
        ctx[0] += 1
      ret[i] = UOp(Ops.CATRANGE, src=tuple(new_ret_i))
  return UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret))

def map_pad(x:UOp):
  r = x.src[0]
  ret = list(x.src[1:])
  bigwhere = UOp.const(dtypes.bool, True)
  for i,(sh,(s,e)) in enumerate(zip(r.shape, r.arg)):
    if s == 0 and e == 0: continue
    where = UOp.const(dtypes.bool, True)
    if e > 0: where = where & (ret[i] < (sh-e))
    if s > 0: where = where & (ret[i] >= s)
    bigwhere = bigwhere & where
    # mask the load
    ret[i] = where.where(ret[i], ret[i].const_like(-100000))
  # PAD is with 0
  return bigwhere.where(UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret)), r.const_like(0))

def map_reduce(ctx:list[int], x:UOp):
  rngs = list(x.src[1:])
  r = x.src[0]
  new_ranges = []
  for i,s in enumerate(r.src[0].shape):
    if i in r.arg[1]:
      assert rngs[i].op == Ops.CONST
      rngs[i] = UOp.range(dtypes.int, s, ctx[0])
      new_ranges.append(rngs[i])
      ctx[0] += 1
  mm = UOp(Ops.INDEX, r.src[0].dtype, src=(r.src[0],)+tuple(rngs))
  return UOp(Ops.REDUCE, r.dtype, src=(mm,)+tuple(new_ranges), arg=r.arg[0])

pm_rangeify = PatternMatcher([
  # TODO: handle MAP on STORE
  (UPat(Ops.STORE, name="x"), rangify_store),
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE),), allow_any_len=True, name="x"), map_permute),
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE),), allow_any_len=True, name="x"), map_reshape),
  (UPat(Ops.INDEX, src=(UPat(Ops.EXPAND),), allow_any_len=True, name="x"), map_expand),
  (UPat(Ops.INDEX, src=(UPat(Ops.SHRINK),), allow_any_len=True, name="x"), map_shrink),
  (UPat(Ops.INDEX, src=(UPat(Ops.PAD),), allow_any_len=True, name="x"), map_pad),
  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS),), allow_any_len=True, name="x"), map_reduce),
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),)), lambda c: c),

  # move MAP through elementwise ALU
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.LOAD})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([UOp(Ops.INDEX, dtype=s.dtype, src=(s,)+x.src[1:]) for s in x.src[0].src]))),
])

def name_the_sink(x:UOp):
  if x.arg is not None: return None
  ranges = sorted([u for u in x.toposort() if u.op is Ops.RANGE], key=lambda y: y.arg)
  return x.replace(arg=KernelInfo(name='k_'+'_'.join([str(u.src[0].arg) for u in ranges])))

pm_name = PatternMatcher([
  (UPat(Ops.SINK, name="x"), name_the_sink),
])