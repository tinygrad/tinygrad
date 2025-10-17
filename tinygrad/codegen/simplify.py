from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, graph_rewrite, _substitute, range_start, ImageDType
from tinygrad.uop.symbolic import symbolic_flat, sym
from tinygrad.helpers import partition
from tinygrad.dtype import dtypes

def flatten_range(r:UOp):
  off = range_start[r.op]
  rngs = r.src[off:]
  if not len(rngs): return None
  new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
  return r.replace(src=r.src[:off]+tuple(new_rngs))

pm_flatten_range = PatternMatcher([
  # real ranges only
  (UPat((Ops.REDUCE, Ops.STORE), name="r"), flatten_range),
])

def count_divmod(x:UOp): return len([u for u in x.toposort() if u.op in {Ops.IDIV, Ops.MOD}])
def simplify_merge_adjacent(u:UOp) -> UOp|None:
  reduce_ranges = [x.ranges for x in u.backward_slice_with_self if x.op is Ops.REDUCE]
  i = range_start[u.op]
  while i < len(u.src)-1:
    r0, r1 = u.src[i], u.src[i+1]
    # check same type
    if r0.arg[-1] == r1.arg[-1]:
      # check if the ranges to merge are in the same reduces
      if all((r0 in rngs) == (r1 in rngs) for rngs in reduce_ranges):
        s0, s1 = r0.src[0], r1.src[0]
        # do the merge
        new_range = r0.replace(src=(s0*s1,))
        nidx = graph_rewrite(u, _substitute+symbolic_flat+pm_flatten_range, ctx={r0:new_range//s1, r1:new_range%s1},
                             name=f"check_merge_{r0.arg[0]}_{r1.arg[0]}")

        # check if it simplifies
        if count_divmod(nidx) <= count_divmod(u):
          u = nidx
          continue
    i += 1
  return u

pm_simplify_ranges = PatternMatcher([
  (UPat((Ops.STORE, Ops.REDUCE), name="u"), simplify_merge_adjacent),
])

def mark_range_mod(ctx, r:UOp, c:UOp):
  if r not in ctx and r.src[0].op is Ops.CONST and r.src[0].divides(c.arg) is not None: ctx[r] = c

def do_substitute(ctx, x: UOp):
  subs = {}
  for k,v in ctx.items():
    if v is not None:
      subs[k] = k.replace(src=(k.src[0]//v,), arg=k.arg[0:-1]+(0,k.arg[-1]))*v + k.replace(src=(v,), arg=k.arg[0:-1]+(1,k.arg[-1]))
  if not len(subs): return None
  ret = x.substitute(subs).simplify()
  ctx.clear()
  return ret

def dont_sub_ranges_for_image(ctx, x:UOp):
  if isinstance(x.src[0].dtype, ImageDType):
    for s in x.src[1:]: ctx[s] = None

pm_split_ranges = PatternMatcher([
  (UPat(Ops.RANGE, name="r")%UPat.cvar("c"), mark_range_mod),
  (UPat(Ops.STORE, name="x"), dont_sub_ranges_for_image),
  (UPat(Ops.SINK, name="x"), do_substitute),
])

# **** reduce simplification ****

def no_range(u:UOp) -> bool: return not any(x.op is Ops.RANGE for x in u.backward_slice_with_self)

def reduce_unparented(red:UOp):
  if red.arg not in {Ops.ADD, Ops.MAX, Ops.MUL}: return None
  assert all(x.op is Ops.RANGE for x in red.src[1:]), "some reduce srcs aren't ranges"
  reduce_parented, reduce_unparented = partition(red.src[1:], lambda x: x in red.src[0].ranges)
  if len(reduce_unparented) == 0: return None
  ret = red.replace(src=(red.src[0],)+tuple(reduce_parented)) if len(reduce_parented) or red.dtype != red.src[0].dtype else red.src[0]
  if red.arg is Ops.ADD:
    for r in reduce_unparented: ret = ret * r.src[0].cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
  if red.arg is Ops.MUL:
    for r in reduce_unparented: ret = ret ** r.src[0].cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
  return ret

pm_reduce_unparented = PatternMatcher([
  # remove any ranges from a REDUCE that aren't referenced in the reduce source
  (UPat(Ops.REDUCE, name="red"), reduce_unparented),
])

pm_reduce_collapse = pm_reduce_unparented + PatternMatcher([
  # lift x+y out of reduce on lt
  ((UPat.var("x")+UPat.var("y")).or_casted() < UPat.var("c"), lambda x,y,c: (x < (c.cast(y.dtype)-y)) if no_range(y) and no_range(c) else None),
  # lift x*y out of reduce
  ((UPat.var("x")*UPat.var("y")) < UPat.var("c"),
   lambda x,y,c: (x < ((c+y-1) // y)) if no_range(y) and no_range(c) and y.vmin > 0 else None),
  # lift x+y out of reduce on ne
  ((UPat.var("x")+UPat.var("y")).or_casted() != UPat.var("c"), lambda x,y,c: (x != (c.cast(y.dtype)-y)) if no_range(y) and no_range(c) else None),
  # fold the range
  ((UPat(Ops.RANGE, name="r") < UPat.var("cut")).where(0, UPat.cvar("val")).reduce(UPat.var("r"), arg=Ops.ADD),
   lambda r,cut,val: (r.src[0]-cut).maximum(0).minimum(r.src[0]).cast(val.dtype) * val),
  (((UPat.var("r")<UPat.var("lower")).logical_not()&(UPat(Ops.RANGE, name="r")<UPat.var("upper"))).where(UPat.cvar("val"), 0).reduce(UPat.var("r"),
    arg=Ops.ADD), lambda r,lower,upper,val: (upper.minimum(r.src[0])-lower.maximum(0)).maximum(0).minimum(r.src[0]).cast(val.dtype) * val),
  ((UPat(Ops.RANGE, name="r") < UPat.var("cut")).where(UPat.cvar("val"), 0).reduce(UPat.var("r"), arg=Ops.ADD),
   lambda r,cut,val: cut.maximum(0).minimum(r.src[0]).cast(val.dtype) * val),
  # REDUCE on ADD
  ((UPat.var("x")+UPat.var("y")).reduce(arg=Ops.ADD, allow_any_len=True, name="r"),
   lambda x,y,r: x.reduce(*r.src[1:], arg=Ops.ADD) + y.reduce(*r.src[1:],arg=Ops.ADD)),
  # MUL casted bool
  ((UPat.var("x") * UPat.var("gate", dtype=dtypes.bool).cast()), lambda x,gate: gate.where(x, 0)),
  # reduce on gated load becomes can substitute the range and remove the reduce
  ((UPat.var("idx")!=(UPat(Ops.RANGE, name="r").or_casted())).where(0, UPat.var("expr")).reduce(UPat.var("r"), arg=Ops.ADD),
   lambda r,idx,expr: (v:=(idx.cast(r.dtype) >= 0) & (idx.cast(r.dtype) < r.src[0])).where(expr.substitute({r:idx.cast(r.dtype).valid(v)}),0)),
  # AND on WHERE
  ((UPat(Ops.DEFINE_VAR, name="x") & UPat.var("y")).where(UPat.cvar("c"), 0).reduce(arg=Ops.ADD, allow_any_len=True, name="r"),
    lambda x,y,c,r: y.where(c, 0).reduce(*r.src[1:], arg=Ops.ADD)*x.cast(c.dtype)),
])+sym

def reduce_collapse(red:UOp):
  included, not_included = partition(red.backward_slice, lambda x: any(y in x.backward_slice_with_self for y in red.src[1:]))
  if any(x.op in {Ops.STORE, Ops.REDUCE} for x in included): return None
  replaces: dict[UOp, UOp] = {}
  for u in included:
    for s in u.src:
      if s in not_included and s not in replaces and s.op not in {Ops.CONST, Ops.VCONST, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR}:
        replaces[s] = UOp(Ops.DEFINE_VAR, dtype=s.dtype, arg=(f'in{len(replaces)}', s.vmin, s.vmax))
  collapse_fxn = red.substitute(replaces)
  sink = graph_rewrite(collapse_fxn, pm_reduce_collapse, name="reduce_collapse")
  return sink.substitute({v:k for k,v in replaces.items()}) if no_range(sink) else None

pm_reduce_simplify = pm_reduce_unparented + PatternMatcher([
  # remove REDUCE without loads (generic arange opt / indexing). TODO: support multi range
  (UPat(Ops.REDUCE, src=(UPat(), UPat()), name="red"), reduce_collapse),
])
