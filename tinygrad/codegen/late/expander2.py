import itertools
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, AxisType, UOp, GroupOp, _align_left, _broadcast_shape
from tinygrad.dtype import dtypes
from tinygrad.helpers import all_same
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.schedule.rangeify import pm_index_mops

def build_range_map(ctx, sink:UOp):
  for x in sink.toposort():
    if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}:
      ctx[x.arg[0]] = len(ctx)

expander2 = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), build_range_map),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp(Ops.VCONST, r.dtype, arg=tuple(range(r.vmax+1))) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
])+pm_flatten_range

def broadcast_binary(x:UOp):
  shapes = [u.shape for u in x.src]
  print(x.op, shapes)
  if all_same(shapes): return None
  shaped_aligned = _align_left(*shapes)
  broadcasted = _broadcast_shape(*shapes)
  src_reshaped = [u.reshape(shp).expand(broadcasted) for u,shp in zip(x.src, shaped_aligned)]
  return x.replace(src=tuple(src_reshaped))

unbroadcast = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), broadcast_binary),
])

def do_devectorize(b:UOp):
  if b.shape == (): return None
  # broadcasting needs to be already unpacked
  if not all_same([x.shape for x in b.src]): return None
  src = []
  for idx in itertools.product(*[range(x) for x in b.shape]):
    idx_c = [UOp.const(dtypes.weakint, i) for i in idx]
    src.append(b.replace(src=tuple([x.index(*idx_c) for x in b.src])))
  return UOp.cat(*src)

devectorizer2 = pm_index_mops+PatternMatcher([
  # INDEX with one src is a noop
  (UPat(Ops.INDEX, src=(UPat.var("x"),)), lambda x: x),
  # INDEX into VCONST is CONST
  (UPat(Ops.INDEX, src=(UPat(Ops.VCONST, name="a"), UPat.cvar("i", vec=False))),
   lambda a,i: UOp.const(a.dtype, a.arg[i.arg])),
  # INDEX into CAT is src
  (UPat(Ops.INDEX, src=(UPat(Ops.CAT, name="a"), UPat.cvar("i", vec=False))),
   lambda a,i: a.src[i.arg] if a.arg == -1 else None),

  # cat goes through index
  (UPat(Ops.INDEX, src=(UPat.var("a"), UPat(Ops.CAT, name="c"))),
   lambda a,c: UOp.cat(*[a.index(x) for x in c.src])),

  # cat on store is group (TODO: do we need group?)
  (UPat(Ops.CAT, src=UPat(Ops.STORE), name="x"), lambda x: UOp.group(*x.src)),

  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.STORE}, name="b"), do_devectorize),
])
