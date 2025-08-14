from typing import Any
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, RewriteNotReady
from tinygrad.helpers import argsort, prod, all_same

rangeify_fixups = PatternMatcher([
  # all contiguous on SINK
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple([s.contiguous() if s.op not in {Ops.CONTIGUOUS, Ops.CONST} else s for s in x.src]))),
  # double contiguous merge
  (UPat(Ops.CONTIGUOUS, name="c2", src=(UPat(Ops.CONTIGUOUS, name="c1"))), lambda c1,c2: c1 if c1.arg is None and c2.arg is None else None),
  # const
  (UPat(Ops.CONST, name="x"), lambda x:
   x.replace(src=(x.src[0].src[0],)).reshape((1,)*len(x.shape)).expand(x.shape) if \
    len(x.src) and x.src[0].op is Ops.VIEW and not any(s == 0 for s in x.shape) else None),
])

@dataclass
class ChildrenContext:
  children: dict[UOp, list[UOp]]|None = None

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

  # hack for one kernel threefry
  #(UPat(Ops.CHILD, src=(UPat(Ops.THREEFRY, name="x"),)), lambda x: x),
])

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
  return r.src[0].index(*ret)

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
  (UPat(Ops.INDEX, src=(UPat(Ops.PERMUTE, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[x.src[1+p] for p in argsort(x.src[0].arg)])),
  (UPat(Ops.INDEX, src=(UPat(Ops.SHRINK, name="r"),), allow_any_len=True, name="x"),
   lambda r,x: r.src[0].index(*[a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(x.src[1:], r.arg)])),
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

  ret = x.src[0].index(*ranges).contiguous(*new_ranges, arg=x.arg, tag=1)
  # if there's no open ranges, set arg to None so this uses a DEFINE_GLOBAL
  if len(ret.ranges) == 0: ret = ret.replace(arg=None)
  ret = ret.index(*passthrough_idx) if len(passthrough_idx) else ret.reshape(x.shape)
  return ret

def map_reduce(ctx:RangeifyContext, idx:UOp, red:UOp):
  # TODO: this should be in the cache
  #print(f"reduce {id(red)}")
  rngs = list(idx.src[1:])
  new_ranges = []
  for i,s in enumerate(red.src[0].shape):
    if i in red.arg[1]:
      rngs[i] = UOp.range(dtypes.int, s, ctx.idx)
      ctx.idx += 1
      new_ranges.append(rngs[i])
  return UOp(Ops.REDUCE, red.dtype, src=(red.src[0].index(*rngs),)+tuple(new_ranges), arg=red.arg[0])

def index_child(ctx:RangeifyContext, c:UOp, x:UOp, idx:UOp):
  #print(f"visit CHILD {x.arg} bottom up")
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
  return c.index(*out_rngs).contiguous(*end_ranges, arg=tuple(idx_ranges), tag=1).index(*[idx.src[1+i] for i in idx_ranges])

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
  (UPat(Ops.INDEX, src=(UPat(Ops.CONTIGUOUS, name="x"),), allow_any_len=True, name="idx"), map_contiguous),
  (UPat(Ops.CONTIGUOUS, name="x"), map_contiguous),

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
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise.union({Ops.STORE, Ops.ASSIGN})),), allow_any_len=True, name="x"),
   lambda x: x.src[0].replace(src=tuple([s.index(*x.src[1:]) for s in x.src[0].src]))),
  (UPat(Ops.INDEX, src=(UPat(Ops.REDUCE_AXIS, name="red"),), allow_any_len=True, name="idx"), map_reduce),


  # CONTIGUOUS on ASSIGN is STORE
  # TODO: tag in UPat?
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.ASSIGN, name="a"),), name="c", allow_any_len=True),
   lambda c,a: UOp(Ops.STORE, src=a.src+c.src[1:]) if c.tag == 1 else None),
])

@dataclass
class AddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)

def add_store(ctx:AddBufferContext, x:UOp):
  assert x.tag == 1
  rngs = x.src[1:]
  shape = tuple([r.vmax+1 for r in rngs])
  assert prod(shape) > 0, f"no zero sized buffers {shape}"
  if x.arg is None or prod(shape) > 65536:
    buf = UOp.new_buffer(x.device, prod(shape), x.dtype)
  else:
    buf = UOp(Ops.DEFINE_LOCAL, dtype=x.dtype.ptr(size=prod(shape), addrspace=AddrSpace.LOCAL), arg=ctx.dg)
  ctx.map[buf] = (buf.op, ctx.dg)
  ctx.dg += 1
  return buf.reshape(shape).index(*rngs, dtype=x.dtype.ptr(size=prod(shape))).store(x.src[0], *rngs)

def add_load(ctx:AddBufferContext, x:UOp, b:UOp, idx:UOp):
  if isinstance(x.dtype, PtrDType): return None
  return x.replace(dtype=x.dtype.ptr(b.size)).load()

def add_load_on_store(ctx:AddBufferContext, x:UOp, st:UOp):
  rngs = x.src[1:]
  shape = tuple([r.vmax+1 for r in rngs])
  return st.src[0].src[0].shrink(((0,prod(shape)),)).reshape(shape).index(*rngs).load(st)

pm_add_buffers = pm_mops+PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), add_store),
  (UPat(Ops.ENDRANGE, name="x"), lambda x: x.src[0]),
  (UPat(Ops.INDEX, src=(UPat(Ops.BUFFER, name="b"), UPat(name="idx")), name="x"), add_load),
  (UPat(Ops.INDEX, src=(UPat(Ops.STORE, name="st"),), allow_any_len=True, name="x"), add_load_on_store),
  (UPat(Ops.BIND, name="b"), lambda b: b.src[0]),
  # HACK: ignore copy
  (UPat(Ops.COPY, name="x"), lambda x: x.src[0]),
  # CONST can't have axes. remove srcs when we idx
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),)), lambda c: c.replace(src=())),
  # HACK: consts shouldn't have srcs by here
  (UPat(Ops.CONST, name="x"), lambda x: x.replace(src=()) if len(x.src) else None),
])
