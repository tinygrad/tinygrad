from typing import Any
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, AxisType, RewriteNotReady
from tinygrad.helpers import argsort, getenv, prod, all_same

@dataclass
class RangeifyContext:
  idx: int = 0
  regs: int = 0
  children: dict[UOp, int]|None = None
  seen_children: dict[UOp, dict[int, UOp]] = field(default_factory=dict)
  seen_child: dict[UOp, Any] = field(default_factory=dict)

  #indexed_child: dict[UOp, list[UOp]] = field(default_factory=dict)
  #ended_child: dict[UOp, int] = field(default_factory=dict)

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
      ranges.append(UOp.range(dtypes.int, s, (ctx.idx, AxisType.LOOP)))
      new_ranges.append(ranges[-1])
      ctx.idx += 1
    else:
      ranges.append(UOp.const(dtypes.int, 0))

  # conv hack. we replace ranges
  if idx is not None and getenv("CONVHACK"):
    sub = {}
    sub[UOp.range(dtypes.int, 3, 4)] = UOp.range(dtypes.int, 3, ctx.idx)
    ctx.idx += 1
    sub[UOp.range(dtypes.int, 3, 5)] = UOp.range(dtypes.int, 3, ctx.idx)
    ctx.idx += 1
    ranges = UOp.sink(*ranges).substitute(sub).src
    passthrough_idx.extend(sub.keys())
    new_ranges.extend(sub.values())

  ret = x.src[0].index(*ranges).contiguous(*new_ranges, arg=x.shape, tag=1)
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
      rngs[i] = UOp.range(dtypes.int, s, (ctx.idx, AxisType.REDUCE))
      ctx.idx += 1
      new_ranges.append(rngs[i])
  return UOp(Ops.REDUCE, red.dtype, src=(red.src[0].index(*rngs),)+tuple(new_ranges), arg=red.arg[0])

def extract_children(ctx:RangeifyContext, x:UOp):
  if ctx.children is not None: return
  ctx.children = {}
  for k,v in x.get_children_map().items():
    if len(v) > 1 and k.op is not Ops.DEVICE: ctx.children[k] = list(v.keys())

def mark_children(ctx:RangeifyContext, x:UOp):
  new_srcs = []
  for s in x.src:
    if s in ctx.children:
      ret = UOp(Ops.CHILDREN, s.dtype, (s,), arg=len(ctx.children[s]))
      ret = UOp(Ops.CHILD, s.dtype, src=(ret,), arg=ctx.children[s].index(x))
      new_srcs.append(ret)
    else:
      new_srcs.append(s)
  return x.replace(src=tuple(new_srcs))

pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All-{Ops.CHILDREN}, name="x"), mark_children),
])


rangeify_fixups = PatternMatcher([
  # all contiguous on SINK
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple([s.contiguous() if s.op is not Ops.CONTIGUOUS else s for s in x.src]))),

  # const
  (UPat(Ops.CONST, name="x"), lambda x:
   x.replace(src=(x.src[0].src[0],)).reshape((1,)*len(x.shape)).expand(x.shape) if len(x.src) and x.src[0].op is Ops.VIEW else None),

  # add contiguous to EXPAND
  #(UPat(Ops.EXPAND, name="x"), lambda x: x.src[0].contiguous().expand(x.arg).replace(tag=1) if x.tag is None else None),
])

"""
def record_child(ctx:RangeifyContext, c:UOp, idx:UOp):
  if c not in ctx.indexed_child: ctx.indexed_child[c] = []
  print("record child", id(idx), len(ctx.indexed_child[c]))
  if idx not in ctx.indexed_child[c]:
    ctx.indexed_child[c].append(idx)
    return idx.replace(src=(c.replace(tag=1),)+idx.src[1:])
  if len(ctx.indexed_child[c]) == c.arg:
    assert idx in ctx.indexed_child[c]
    ec = UOp(Ops.ENDCHILD, dtype=c.dtype, src=(c,))
    if ec not in ctx.ended_child: ctx.ended_child[ec] = 1
    else: ctx.ended_child[ec] += 1
    print("creating endchild", ctx.ended_child[ec])
    if c in ctx.seen_child:
      out_rngs = list(idx.src[1:])
      idx_ranges, end_ranges = ctx.seen_child[c]
      if len(idx_ranges) == 0:
        return c.src[0].index(*out_rngs)
      for i,nr in zip(idx_ranges, end_ranges):
        out_rngs[i] = nr
      return ec.index(*out_rngs).contiguous(*end_ranges, arg=ec.shape, tag=1).index(*[idx.src[1+i] for i in idx_ranges])
    else:
      # heres where we can compute everything about mismatched ranges
      all_rngs = zip(*[x.src[1:] for x in ctx.indexed_child[c]])
      out_rngs = []
      end_ranges = []
      idx_ranges = []
      for i,r in enumerate(all_rngs):
        if all_same(r):
          out_rngs.append(r[0])
        else:
          out_rngs.append(UOp.range(dtypes.int, c.shape[i], (ctx.idx, AxisType.LOOP)))
          ctx.idx += 1
          end_ranges.append(out_rngs[-1])
          idx_ranges.append(i)
      if len(end_ranges) == 0:
        # safe to remove child right away
        #return c.src[0].index(*out_rngs)
        return ec.index(*out_rngs)
      else:
        ctx.seen_child[c] = (idx_ranges, end_ranges)
        return ec.index(*out_rngs).contiguous(*end_ranges, arg=ec.shape, tag=1).index(*[idx.src[1+i] for i in idx_ranges])

def child_check(ctx:RangeifyContext, sink:UOp):
  subs = {}
  for x in ctx.ended_child:
    if ctx.ended_child[x] == x.src[0].arg:
      print("sub")
      subs[x] = x.src[0].src[0]
  ctx.ended_child.clear()
  return sink.substitute(subs)
"""

"""
def visit_child(ctx:RangeifyContext, x:UOp):
  print(f"visit CHILD {x.arg} bottom up")
  if x.src[0] not in ctx.seen_children: ctx.seen_children[x.src[0]] = set()
  ctx.seen_children[x.src[0]].add(x.arg)
  if len(ctx.seen_children[x.src[0]]) != x.src[0].arg: raise RewriteNotReady
  print("READY")

def visit_children(ctx:RangeifyContext, x:UOp):
  if x.tag == 1: return None
  if len(ctx.seen_children[x]) != x.arg:
    print("visit CHILDREN bottom up -- not ready")
    raise RewriteNotReady
  print("visit CHILDREN bottom up -- READY")
  return x.replace(tag=1)
"""

def index_child(ctx:RangeifyContext, c:UOp, x:UOp, idx:UOp):
  print(f"visit CHILD {x.arg} bottom up")
  if c not in ctx.seen_children: ctx.seen_children[c] = {}
  ctx.seen_children[c][x.arg] = idx
  # wait here until we have seen all the children
  if len(ctx.seen_children[c]) != c.arg: raise RewriteNotReady

  if c not in ctx.seen_child:
    all_rngs = zip(*[ch.src[1:] for ch in ctx.seen_children[c].values()])
    out_rngs = []
    end_ranges = []
    idx_ranges = []
    for i,r in enumerate(all_rngs):
      if all_same(r):
        out_rngs.append(r[0])
      else:
        out_rngs.append(UOp.range(dtypes.int, c.shape[i], (ctx.idx, AxisType.LOOP)))
        ctx.idx += 1
        end_ranges.append(out_rngs[-1])
        idx_ranges.append(i)
    ctx.seen_child[c] = (idx_ranges, end_ranges)
  else:
    out_rngs = list(idx.src[1:])
    idx_ranges, end_ranges = ctx.seen_child[c]
    for i,nr in zip(idx_ranges, end_ranges): out_rngs[i] = nr
  if len(idx_ranges) == 0: return c.src[0].index(*out_rngs)
  return c.src[0].index(*out_rngs).contiguous(*end_ranges, arg=c.src[0].shape, tag=1).index(*[idx.src[1+i] for i in idx_ranges])

pm_rangeify = PatternMatcher([
  # if there are new ended children, tag the SINK
  (UPat(Ops.INDEX, src=(UPat(Ops.CHILD, src=(UPat(Ops.CHILDREN, name="c"), ), name="x"),), allow_any_len=True, name="idx"), index_child),
  #(UPat(Ops.SINK, name="sink"), child_check),
  #(UPat(Ops.CHILD, name="x"), visit_child),
  #(UPat(Ops.CHILDREN, name="x"), visit_children),


  # if there's an INDEX it can support partial contig
  (UPat(Ops.INDEX, src=(UPat(Ops.CONTIGUOUS, name="x"),), allow_any_len=True, name="idx"), map_contiguous),
  (UPat(Ops.CONTIGUOUS, name="x"), map_contiguous),

  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="red"),), allow_any_len=True, name="idx"), map_reduce),

  # this is like the definitions of these
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[x.src[1+p] for p in argsort(x.src[0].arg)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.SHRINK, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(x.src[1:], r.arg)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.FLIP, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[((s-1)-a) if f else a for a,s,f in zip(x.src[1:], r.shape, r.arg)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.EXPAND, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[a.const_like(0) if resolve(x!=y, False) else a for a,x,y in zip(x.src[1:], r.src[0].shape, r.shape)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE, name="r"),), allow_any_len=True, name="x"), map_reshape),
  (UPat(Ops.INDEX, src=(UPat(Ops.PAD, name="r"),), allow_any_len=True, name="x"), map_pad),

  # move MAP through elementwise ALU
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.STORE})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([s.index(*x.src[1:]) for s in x.src[0].src]))),

  # CONST can't have axes. remove srcs when we idx
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST,name="c"),)), lambda c: c.replace(src=())),
])

@dataclass
class AddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)

def add_store(ctx:AddBufferContext, x:UOp):
  rngs = x.src[1:]
  shape = tuple([r.vmax+1 for r in rngs])
  buf = UOp(Ops.DEFINE_GLOBAL if prod(shape) > 65536 or ctx.dg == 0 else Ops.DEFINE_LOCAL, dtype=x.dtype.ptr(size=prod(shape)), arg=ctx.dg)
  ctx.map[buf] = (buf.op, ctx.dg)
  ctx.dg += 1
  return buf.reshape(shape).index(*rngs).store(x.src[0], *rngs)

def add_load(ctx:AddBufferContext, x:UOp, b:UOp, idx:UOp):
  if b not in ctx.map:
    ctx.map[b] = (Ops.DEFINE_GLOBAL, ctx.dg)
    ctx.dg += 1
  return UOp(ctx.map[b][0], dtype=x.dtype.ptr(size=b.arg), arg=ctx.map[b][1]).index(idx).load()

def add_load_on_store(ctx:AddBufferContext, x:UOp, st:UOp):
  rngs = x.src[1:]
  shape = tuple([r.vmax+1 for r in rngs])
  return st.src[0].src[0].reshape(shape).index(*rngs).load(st)

from tinygrad.schedule.rangeify import map_reshape

pm_add_buffers = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), add_store),
  (UPat(Ops.INDEX, src=(UPat(Ops.BUFFER, name="b"), UPat(name="idx")), name="x"), add_load),
  (UPat(Ops.INDEX, src=(UPat(Ops.STORE, name="st"),), allow_any_len=True, name="x"), add_load_on_store),
  (UPat(Ops.INDEX, src=(UPat(Ops.RESHAPE, name="r"),), allow_any_len=True, name="x"), map_reshape),
])
