from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, KernelInfo, GroupOp, AxisType, TRACK_MATCH_STATS
from tinygrad.opt.kernel import axis_colors, Opt, OptOps
from dataclasses import dataclass
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import argsort, colored, prod, all_same, getenv

@dataclass
class RangeifyContext:
  idx: int = 0
  regs: int = 0
  opts: tuple[Opt, ...] = ()

def map_store(ctx:RangeifyContext, x:UOp):
  if x.tag == 1: return None
  ranges = []
  for i,s in enumerate(x.shape):
    upcast_amount = prod([o.arg if o.arg != 0 else s for o in ctx.opts if o.axis == i and o.op == OptOps.UPCAST])
    if resolve(s!=1):
      if upcast_amount != 1:
        print(x.shape, upcast_amount)
        assert s%upcast_amount == 0
        rng = UOp.range(dtypes.int, s//upcast_amount, (ctx.idx, AxisType.LOOP)) * upcast_amount
        rng = rng + UOp.range(dtypes.int, upcast_amount, (ctx.idx+1, AxisType.UPCAST))
        ranges.append(rng)
        ctx.idx += 2
      else:
        ranges.append(UOp.range(dtypes.int, s, (ctx.idx, AxisType.LOOP)))
        ctx.idx += 1
    else:
      ranges.append(UOp.const(dtypes.int, 0))
  mm = UOp(Ops.INDEX, dtype=x.src[0].dtype, src=(x.src[0],)+tuple(ranges))
  mm2 = UOp(Ops.INDEX, dtype=x.src[0].dtype, src=(x.src[1],)+tuple(ranges))
  return UOp(Ops.STORE, src=(mm, mm2)+tuple([x for x in UOp.sink(*ranges).toposort() if x.op is Ops.RANGE]), tag=1)

def map_load(ctx:RangeifyContext, idx:UOp, load:UOp):
  out_ranges = idx.src[1:]
  idx_sink = UOp.sink(*out_ranges)
  upcast_ranges = [x for x in idx_sink.toposort() if x.op is Ops.RANGE and x.arg[1] == AxisType.UPCAST]
  if len(upcast_ranges) or True:
    replace_ranges = {}
    reg_size = prod([x.vmax+1 for x in upcast_ranges])
    buf = UOp(Ops.DEFINE_REG, load.dtype.ptr(size=reg_size, addrspace=AddrSpace.REG), arg=(ctx.regs,))
    ctx.regs += 1
    for r in upcast_ranges:
      replace_ranges[r] = UOp.range(dtypes.int, r.vmax+1, (ctx.idx, AxisType.UPCAST))
      ctx.idx += 1
    replace_ranges_v = list(replace_ranges.values())
    out_ranges = idx_sink.substitute(replace_ranges).src
    ret = load.src[0].index(*out_ranges).load()
    ret = buf.index(*upcast_ranges).load(buf.index(*replace_ranges_v).store(ret, *replace_ranges_v, tag=1))
    return ret
  else:
    return UOp(Ops.INDEX, load.src[0].dtype, src=(load.src[0],)+out_ranges).load()

def map_reduce(ctx:RangeifyContext, x:UOp, r:UOp):
  rngs = list(x.src[1:])
  new_ranges = []
  for i,s in enumerate(r.src[0].shape):
    if i in r.arg[1]:
      assert rngs[i].op == Ops.CONST
      rngs[i] = UOp.range(dtypes.int, s, (ctx.idx, AxisType.REDUCE))
      new_ranges.append(rngs[i])
      ctx.idx += 1
  mm = UOp(Ops.INDEX, r.src[0].dtype, src=(r.src[0],)+tuple(rngs))
  return UOp(Ops.REDUCE, r.dtype, src=(mm,)+tuple(new_ranges), arg=r.arg[0])

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

def capture_sink(ctx:RangeifyContext, x: UOp):
  if x.tag == 1:
    late_subs = {}
    for k,v in x.get_children_map().items():
      if k.op is Ops.CHILDREN and all([vi.op is Ops.INDEX for vi in v]):
        idxs = list(zip(*[vi.src[1:] for vi in v]))
        new_idxs = []
        only_new_idxs = []
        save_shape = []
        full_shape = []
        for idx in idxs:
          if all_same(idx):
            new_idxs.append(idx[0])
            save_shape.append(1)
            full_shape.append(idx[0].vmax+1)
          else:
            ll = [z.vmax+1 for z in idx]
            assert all_same(ll), f"mismatch shapes {ll}"
            save_shape.append(ll[0])
            full_shape.append(ll[0])
            new_idxs.append(UOp.range(dtypes.int, ll[0], (ctx.idx, AxisType.LOOP)))
            only_new_idxs.append(new_idxs[-1])
            ctx.idx += 1
        new_idxs = tuple(new_idxs)
        inp = k.src[0]
        print(save_shape, full_shape)
        if len(save_shape):
          buf = UOp(Ops.DEFINE_REG, inp.dtype.ptr(size=prod(save_shape), addrspace=AddrSpace.REG), arg=(ctx.regs,))
          ctx.regs += 1
          buf = buf.reshape(tuple(save_shape)).expand(tuple(full_shape))
          store = UOp(Ops.INDEX, buf.dtype, (buf,)+new_idxs).store(UOp(Ops.INDEX, inp.dtype, (inp,)+new_idxs), *only_new_idxs, tag=1)
          for vi in v:
            late_subs[vi] = UOp(Ops.INDEX, buf.dtype, (buf,)+vi.src[1:]).load(store)
        else:
          print("no replace")
          for vi in v:
            assert new_idxs == vi.src[1:]
            late_subs[vi] = UOp(Ops.INDEX, inp.dtype, (inp,)+new_idxs)
    if not len(late_subs): return None
    return x.substitute(late_subs)
  if x.arg is not None and x.arg.opts_to_apply is not None: ctx.opts = x.arg.opts_to_apply
  replace_children = {}
  for k,v in x.get_children_map().items():
    if k.op not in {Ops.CHILDREN, Ops.DEVICE} and len(v) > 1:
      replace_children[k] = UOp(Ops.CHILDREN, dtype=k.dtype, src=(k.replace(tag=len(v)),))
  if getenv("FUSE") and TRACK_MATCH_STATS > 0: x = x.substitute(replace_children)
  return x.replace(arg=None, tag=1)

pm_rangeify = PatternMatcher([
  (UPat(Ops.SINK, name="x"), capture_sink),

  # TODO: handle INDEX on STORE
  (UPat(Ops.STORE, name="x"), map_store),
  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="r"),), allow_any_len=True, name="x"), map_reduce),

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

  # bring where to the front
  #(UPat(GroupOp.Binary, name="base", src=(UPat.var("c").where(UPat.var("x"), UPat(Ops.INVALID, name="inv")), UPat.var("a"))),
  # lambda c,x,a,base,inv: c.where(UOp(base.op, base.dtype, (x,a)), inv)),
  #(UPat(GroupOp.Binary, name="base", src=(UPat.var("c").where(UPat(Ops.INVALID, name="inv"), UPat.var("x")), UPat.var("a"))),
  # lambda c,x,a,base,inv: c.where(inv, UOp(base.op, base.dtype, (x,a)))),
  #(UPat(GroupOp.Binary, name="base", src=(UPat.var("a"), UPat.var("c").where(UPat.var("x"), UPat(Ops.INVALID, name="inv")))),
  # lambda c,x,a,base,inv: c.where(UOp(base.op, base.dtype, (a,x)), inv)),
  #(UPat(GroupOp.Binary, name="base", src=(UPat.var("a"), UPat.var("c").where(UPat(Ops.INVALID, name="inv"), UPat.var("x")))),
  # lambda c,x,a,base,inv: c.where(inv, UOp(base.op, base.dtype, (a,x)))),

  # move MAP through elementwise ALU
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.STORE})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([UOp(Ops.INDEX, dtype=s.dtype, src=(s,)+x.src[1:]) for s in x.src[0].src]))),

  # map load
  (UPat(Ops.INDEX, src=(UPat(Ops.LOAD, name="load"),), allow_any_len=True, name="idx"), map_load),

  # INDEX without ranges on a DEFINE is just index 0
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Defines),), name="x"), lambda x: x.replace(src=x.src+(UOp.const(dtypes.int, 0),))),

  # CONST can't have axes
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST,name="c"),)), lambda c: c),

  # unbind...but this is too late
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR, name="v"), UPat(Ops.CONST))), lambda v: v),
])

def name_the_sink(x:UOp):
  if x.arg is not None: return None
  ranges = sorted([u for u in x.toposort() if u.op is Ops.RANGE], key=lambda y: y.arg)
  return x.replace(arg=KernelInfo(name='k_'+'_'.join([colored(str(u.src[0].arg), axis_colors[u.arg[1]]) for u in ranges])))

pm_name = PatternMatcher([
  (UPat(Ops.SINK, name="x"), name_the_sink),
])