from typing import Any
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, RewriteNotReady
from tinygrad.helpers import argsort, prod, all_same, pluralize, getenv

from tinygrad.uop.ops import track_rewrites, graph_rewrite_map, graph_rewrite

# 1. add contiguous where we have to

add_contiguous = PatternMatcher([(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"),
                                  lambda ctx,x: x.replace(tag=1).contiguous() if x in ctx and x.tag is None else None)])
remove_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None)])

# 2. mark all children

@dataclass
class ChildrenContext: children: dict[UOp, list[UOp]]|None = None
def extract_children(ctx:ChildrenContext, x:UOp):
  if ctx.children is not None: return
  # REDUCE_AXIS is fine here, should go to contig only (gate)
  ctx.children = {k:list(v.keys()) for k,v in x.get_children_map().items() if len(v) > 1 and any(x.op is Ops.REDUCE_AXIS for x in k.toposort())}
def mark_children(ctx:ChildrenContext, x:UOp):
  new_srcs = [(UOp(Ops.CHILD, s.dtype, src=(s,), arg=(ctx.children[s].index(x), len(ctx.children[s]))) if s in ctx.children else s) for s in x.src]
  return x.replace(src=tuple(new_srcs))
pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All-{Ops.CHILD}, name="x"), mark_children),
])

# 3. rangeify

@dataclass
class RangeifyContext:
  idx: int = 0
  regs: int = 0
  seen_children: dict[UOp, dict[int, UOp]] = field(default_factory=dict)
  seen_child: dict[UOp, Any] = field(default_factory=dict)

def map_reshape(x:UOp, r:UOp):
  acc = 1
  to_sum = []
  for s,src in list(zip(x.shape, x.src[1:]))[::-1]:
    to_sum.append(acc*src)
    acc *= s
  mish = sum(to_sum)
  ret = []
  for s in r.src[0].shape[::-1]:
    if resolve(s!=1):
      # this MOD should limit any ranges outside s
      ret.append(mish % s)
      mish //= s
    else:
      ret.append(UOp.const(dtypes.int, 0))
  ret = UOp.sink(*ret).simplify().src[::-1] if len(ret) else ()
  return r.src[0].index(*ret, dtype=x.dtype)

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
  # PAD is with 0
  return bigwhere.simplify().where(UOp(Ops.INDEX, r.dtype, src=(r.src[0],)+tuple(ret)), UOp.const(r.dtype, 0))

def map_expand(r:UOp, x:UOp):
  new_rngs = []
  ending_ranges = []
  non_ending_ranges = []
  for a,x,y in zip(x.src[1:], r.src[0].shape, r.shape):
    axis_to_range = [u for u in a.toposort() if u.op is Ops.RANGE]
    if resolve(x!=y, False):
      ending_ranges.extend(axis_to_range)
      new_rngs.append(a.const_like(0))
    else:
      non_ending_ranges.extend(axis_to_range)
      new_rngs.append(a)
  ending_ranges = [x for x in ending_ranges if x not in non_ending_ranges]
  ret = r.src[0]
  ret = UOp(Ops.ENDRANGE, dtype=ret.dtype, src=(ret,)+tuple(ending_ranges)) if len(ending_ranges) else ret
  return ret.index(*new_rngs)

pm_mops = PatternMatcher([
  # this is like the definitions of these
  (UPat(Ops.INDEX, src=(UPat(Ops.SHRINK, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(x.src[1:], r.arg)], dtype=x.dtype)),
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[x.src[1+p] for p in argsort(x.src[0].arg)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.FLIP, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[((s-1)-a) if f else a for a,s,f in zip(x.src[1:], r.shape, r.arg)])),
  # expand needs to end ranges
  (UPat(Ops.INDEX, src=(UPat(Ops.EXPAND, name="r"),), allow_any_len=True, name="x"), map_expand),
  # reshape does a lot of symbolic stuff
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE, name="r"),), allow_any_len=True, name="x"), map_reshape),
  # pad adds min and max
  (UPat(Ops.INDEX, src=(UPat(Ops.PAD, name="r"),), allow_any_len=True, name="x"), map_pad),
])

def map_contiguous(ctx:RangeifyContext, x:UOp, idx:UOp|None=None):
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
  ret = x.src[0].index(*ranges).bufferize(*new_ranges)
  # if there's no open ranges, set arg to None so this uses a DEFINE_GLOBAL
  if len(ret.ranges) == 0: ret = ret.replace(arg=None)
  ret = ret.index(*passthrough_idx) if len(passthrough_idx) else ret
  return ret

def map_reduce(ctx:RangeifyContext, idx:UOp, red:UOp):
  rngs = list(idx.src[1:])
  new_ranges = []
  for i,s in enumerate(red.src[0].shape):
    if i in red.arg[1]:
      rngs[i] = UOp.range(dtypes.int, s, ctx.idx)
      ctx.idx += 1
      new_ranges.append(rngs[i])
  return UOp(Ops.REDUCE, red.dtype, src=(red.src[0].index(*rngs),)+tuple(new_ranges), arg=red.arg[0])

def index_child(ctx:RangeifyContext, c:UOp, x:UOp, idx:UOp):
  if c not in ctx.seen_children: ctx.seen_children[c] = {}
  ctx.seen_children[c][x.arg[0]] = idx
  # wait here until we have seen all the children
  if len(ctx.seen_children[c]) != x.arg[1]: raise RewriteNotReady

  if c not in ctx.seen_child:
    all_rngs = zip(*[ch.src[1:] for ch in ctx.seen_children[c].values()])
    out_rngs = []
    end_ranges = []
    idx_ranges = []
    for i,r in enumerate(all_rngs):
      if all_same(r):
        out_rngs.append(r[0])
      else:
        out_rngs.append(UOp.range(dtypes.int, c.shape[i], ctx.idx))
        ctx.idx += 1
        end_ranges.append(out_rngs[-1])
        idx_ranges.append(i)
    ctx.seen_child[c] = (idx_ranges, end_ranges)
  else:
    out_rngs = list(idx.src[1:])
    idx_ranges, end_ranges = ctx.seen_child[c]
    for i,nr in zip(idx_ranges, end_ranges): out_rngs[i] = nr
  if len(idx_ranges) == 0: return c.index(*out_rngs)
  return c.index(*out_rngs).bufferize(*end_ranges).index(*[idx.src[1+i] for i in idx_ranges])

def indexed_endrange(er:UOp, idx:UOp):
  ended = er.src[1:]
  earliest_ending_axis = min([x.arg for x in ended])
  to_end_axis = []
  for i,a in enumerate(idx.src[1:]):
    if any(x.arg > earliest_ending_axis for x in a.toposort() if x.op is Ops.RANGE):
      to_end_axis.append(i)
  if to_end_axis: return idx.replace(src=(er.src[0].contiguous(arg=tuple(to_end_axis)),)+idx.src[1:])
  return idx.replace(src=(er.src[0],)+idx.src[1:])

pm_rangeify = pm_mops+PatternMatcher([
  # if there are new ended children, tag the SINK
  (UPat(Ops.INDEX, src=(UPat(Ops.CHILD, src=(UPat(name="c"), ), name="x"),), allow_any_len=True, name="idx"), index_child),

  # if there's an INDEX it can support partial contig
  (UPat(Ops.INDEX, src=(UPat(Ops.CONTIGUOUS, src=(UPat(),), name="x"),), allow_any_len=True, name="idx"), map_contiguous),

  # sink contigs to kick it off
  (UPat(Ops.CONTIGUOUS, src=(UPat(),), name="x"), lambda ctx,x: map_contiguous(ctx, x)), #.reshape(x.shape) if x.tag == 2 else None),

  # handle ENDRANGE on movement
  (UPat(Ops.ENDRANGE, src=(UPat(GroupOp.Movement),), allow_any_len=True, name="er"),
   lambda er: er.src[0].replace(src=(UOp(Ops.ENDRANGE, dtype=er.dtype, src=(er.src[0].src[0],)+er.src[1:]),))),
  # handle ENDRANGE on BUFFER
  # and CHILD: python3 test/test_schedule.py TestSchedule.test_cache_reduce_parent
  (UPat(Ops.ENDRANGE, src=(UPat((Ops.BUFFER, Ops.CONST, Ops.CONTIGUOUS, Ops.CHILD)),), allow_any_len=True, name="er"), lambda er: er.src[0]),
  # handle INDEXed ENDRANGE
  (UPat(Ops.INDEX, src=(UPat(Ops.ENDRANGE, src=(UPat(GroupOp.Elementwise.union({Ops.REDUCE_AXIS})),), allow_any_len=True, name="er"),),
        allow_any_len=True, name="idx"), indexed_endrange),

  # move MAP through elementwise ALU / reduce. these are the items with cost
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.STORE, Ops.ASSIGN, Ops.COPY, Ops.DEVICE})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([s.index(*x.src[1:]) for s in x.src[0].src]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="red"),), allow_any_len=True, name="idx"), map_reduce),

  # CONTIGUOUS on ASSIGN is STORE
  # TODO: tag in UPat?
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.ASSIGN, name="a"),), name="c", allow_any_len=True),
   lambda c,a: UOp(Ops.STORE, src=a.src+c.src[1:]) if c.tag == 1 else None),
])

@track_rewrites(name=lambda sink,ret: f"Schedule {pluralize('Kernel',len([u for u in ret[sink].toposort() if u.op is Ops.KERNEL]))}", replay=True)
def get_kernelize_map(sink:UOp) -> dict[UOp, UOp]:
  tensor_map = {sink:sink}
  realize_map = {x.base:None for x in sink.src}
  tensor_map = graph_rewrite_map(tensor_map[sink], add_contiguous, ctx=realize_map, bottom_up=True, input_map=tensor_map, name="add_contiguous")
  tensor_map = graph_rewrite_map(tensor_map[sink], remove_tags, input_map=tensor_map, name="finalize_contiguous")
  tensor_map = graph_rewrite_map(tensor_map[sink], pm_children, ctx=ChildrenContext(), bottom_up=True, input_map=tensor_map, name="children")
  tensor_map = graph_rewrite_map(tensor_map[sink], pm_rangeify, ctx=RangeifyContext(), bottom_up=True, input_map=tensor_map, name="rangeify")
  if getenv("VIZ"): graph_rewrite(tensor_map[sink], PatternMatcher([]), name="View Rangeify Graph")

  from tinygrad.codegen import rewrites_for_linearizer, apply_rewrites
  rsink = apply_rewrites(tensor_map[sink], rewrites_for_linearizer)
  from tinygrad.renderer.cstyle import CStyleLanguage
  src = CStyleLanguage().render(rsink.arg.lst)
  print(src)

  return {sink:sink}
