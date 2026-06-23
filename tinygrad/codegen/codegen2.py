from typing import Any
import itertools, functools
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, AxisType, resolve, graph_rewrite
from tinygrad.dtype import dtypes, AddrSpace, ImageDType, Invalid
from tinygrad.helpers import all_same, flatten, getenv
from tinygrad.uop.ops import _align_left, _broadcast_shape, identity_element
from tinygrad.codegen.late.devectorizer import ReduceContext
from tinygrad.uop.symbolic import pm_clean_up_group_sink
from collections import defaultdict

def maybe_load(u:UOp): return u.load() if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG) else u
pm_move_regs = PatternMatcher([
  # BITCAST?
  (UPat(GroupOp.Elementwise|{Ops.REDUCE}, name="x"), lambda x: x.replace(src=tuple([maybe_load(u) for u in x.src]))),
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(src=(x.src[0], maybe_load(x.src[1]))+x.src[2:])),
])

pm_lower_weakints = PatternMatcher([
  (UPat(GroupOp.All, dtype=dtypes.weakint, name="x"), lambda x: x.replace(dtype=dtypes.int)),
])

def build_range_map(ctx, sink:UOp):
  for x in sink.toposort():
    if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}:
      ctx[x.arg[0]] = len(ctx)

def fix_reduce(ctx, r:UOp):
  range_to_axis = {u:ctx[u.arg[0]] for u in r.ended_ranges if u.arg[0] in ctx if u.arg[1] == AxisType.UNROLL}
  return r.replace(src=tuple([u for u in r.src if u not in range_to_axis]), arg=(r.arg[0], r.arg[1]+tuple(range_to_axis.values())))

expander2 = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), build_range_map),
  (UPat(Ops.REDUCE, name="r"), fix_reduce),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(r.dtype, tuple(range(r.vmax+1))) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
])+pm_flatten_range

def broadcast_binary(x:UOp):
  shapes = [u.shape for u in x.src]
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
  return UOp.vectorize(*src).reshape(b.shape) if b.op is not Ops.STORE else UOp.group(*src)

devectorizer2 = pm_mops+PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.LOAD,Ops.STORE}, name="b"), do_devectorize),
  # const INDEX into STACK is src
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="a"), UPat.cvar("i"))), lambda a,i: a.src[i.arg]),
  # stacked INDEX is many INDEX
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.STACK, name="s"))),
   lambda b,s: UOp.vectorize(*[b.index(u) for u in s.src])),
  # INDEX into RESHAPE moves the RESHAPE
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.RESHAPE, name="s"))),
   lambda b,s: b.index(s.src[0]).reshape(s.shape)),
  # RESHAPE a void is removed (hack for AFTER)
  (UPat(Ops.RESHAPE, dtype=dtypes.void, name="x"), lambda x: x.src[0]),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(UOp.const(dtypes.weakint, 0)) if x.marg == () and x.src[0].shape == (1,) else None),
  # INDEX without src is nothing
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
  # RESHAPE+EXPAND -> STACK
  (UPat(Ops.EXPAND, src=(UPat(Ops.RESHAPE, src=(UPat.var("x"), UPat())), UPat()), name="out"),
   lambda x,out: UOp.vectorize(*([x]*out.max_numel())) if out.shape == (out.max_numel(),) else None),
])

def reduce_ranges_to_acc(ctx:ReduceContext, r:UOp):
  acc = UOp.placeholder_like(r, ctx.acc_num, AddrSpace.REG)
  ctx.acc_num += 1
  topo = r.src[0].toposort()
  ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
  input_ranges = tuple(x for x in topo if x.op is Ops.RANGE and x not in r.src[1:] and x not in ended_ranges)
  acc_init = acc.after(*input_ranges).store(identity_element(r.arg[0], r.dtype.scalar()))
  acc_initted = acc.after(acc_init, *r.src[1:])
  inp = r.src[0].reduce(arg=r.arg) if r.arg[1] else r.src[0]
  acc_out = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:])
  return acc.after(acc_out)

def expand_horizontal_reduce(r:UOp):
  axes = r.arg[1]
  vals = [r.src[0].shrink(tuple((idx[axes.index(i)], idx[axes.index(i)]+1) if i in axes else None for i in range(r.src[0].ndim)))
          for idx in itertools.product(*[range(r.src[0].max_shape[a]) for a in axes])]
  return functools.reduce(lambda x,y: x.alu(r.arg[0], y), vals)

pm_reduce_local = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat(), UPat()), allow_any_len=True, name="r"), reduce_ranges_to_acc),
])+pm_clean_up_group_sink

pm_horizontal_reduce = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), expand_horizontal_reduce),
])

# *** memory coalesing ***

def memory_coalesing(sink:UOp):
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[UOp, UOp, UOp], dict[int, list[UOp]]]  = defaultdict(dict)
  for u in sink.toposort():
    if u.op in {Ops.LOAD, Ops.STORE}:
      assert u.src[0].op is Ops.INDEX
      buf,idx_u = u.src[0].src
      idx: Any = idx_u.get_idx()
      valid: Any = idx_u.src[0] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else None
      if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
      elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
      elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
      elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
      else: root_src, arg = idx, 0
      memory[(u.op, buf, root_src, valid)].setdefault(arg, []).append(u)

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for grp in grouped_offsets:
      offset = (base+grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.weakint, grp[0])
      idx = buf._mop(Ops.SHRINK, arg=[(offset, len(grp))]) if len(grp) > 1 else buf.index(offset)
      if op is Ops.STORE:
        datas = []
        for i,g in enumerate(grp):
          assert len(offsets[g]) == 1
          datas.append(offsets[g][0].src[1])
        data = UOp.vectorize(*datas) if len(datas) > 1 else datas[0]
        store = idx.store(data, valid) if valid is not None else idx.store(data)
        for i,g in enumerate(grp): replacements[offsets[g][0]] = store
      else:
        ld = idx.load(idx.vconst_like(0), valid) if valid is not None else idx.load()
        for i,g in enumerate(grp):
          for oo in offsets[g]:
            replacements[oo] = ld.index(UOp.const(dtypes.int, i)) if len(grp) > 1 else ld

  # apply
  return sink.substitute(replacements, name="memory coalesing")

