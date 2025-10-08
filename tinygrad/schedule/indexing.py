from typing import Iterator
import functools, operator, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp
from tinygrad.uop.symbolic import sym
from tinygrad.helpers import argsort, all_same, Context
from tinygrad.uop.ops import graph_rewrite, sint, AxisType

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL

@dataclass
class IndexingContext:
  realize_map: dict[UOp, None] = field(default_factory=dict)
  range_map: dict[UOp, tuple[list[UOp], list[UOp]]] = field(default_factory=dict)
  pads_gate: dict[UOp, UOp] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP):
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.index, 0)

def apply_rangeify(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.BUFFERIZE, Ops.INDEX, Ops.KERNEL}: return None
  if x.op is Ops.ASSIGN and x.src[1].op is Ops.KERNEL: return None
  new_srcs = []
  for s in x.src:
    new_src = s
    if s.op in {Ops.BUFFER, Ops.MSTACK, Ops.MSELECT} or (s.op is Ops.ASSIGN and s.src[1].op is Ops.KERNEL):
      if x in ctx.range_map: new_src = new_src.index(*ctx.range_map[x][0])
    elif s in ctx.realize_map:
      new_src = UOp(Ops.BUFFERIZE, s.dtype, src=(s,)+tuple(ctx.range_map[s][1]), arg=BufferizeOpts(device=s.device), tag=s.tag)
      if x in ctx.range_map: new_src = new_src.index(*ctx.range_map[x][0])
    new_srcs.append(new_src)
  # NOTE: do we need this?
  return x.replace(src=tns) if x.src != (tns:=tuple(new_srcs)) else None

def apply_pad(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  ret = ctx.pads_gate[x].where(x.src[0], UOp.const(x.dtype, 0))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def fix_reduce_axis(ctx:IndexingContext, x:UOp):
  # input ranges
  new_ranges = [r for i,r in enumerate(ctx.range_map[x][0]) if i in x.arg[1]]
  ret = UOp(Ops.REDUCE, x.dtype, src=(x.src[0],)+tuple(new_ranges), arg=x.arg[0], tag=x.tag)
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def remove_movement(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

def fix_assign(ctx:IndexingContext, assign:UOp):
  if assign.src[1].op is Ops.KERNEL: return None
  to_mop = graph_rewrite(assign.src[0], PatternMatcher([(UPat(GroupOp.Movement, name="x"), lambda x: x.replace(tag=()))]))
  ret = assign.replace(src=assign.src+(to_mop,))
  ctx.range_map[ret] = ctx.range_map[assign]
  return ret

pm_apply_rangeify = PatternMatcher([
  # REDUCE_AXIS -> REDUCE
  (UPat(Ops.REDUCE_AXIS, name="x"), fix_reduce_axis),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), apply_pad),
  # add third op to assign
  (UPat(Ops.ASSIGN, src=(UPat(), UPat()), name="assign"), fix_assign),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), apply_rangeify),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement),
  # const/define_var shouldn't have src
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda ctx,c: c.replace(src=()) if c in ctx.range_map else None),
])

def run_rangeify(tsink:UOp, realize_map:dict[UOp, None], debug) -> tuple[UOp, IndexingContext]:
  tsink_base = UOp.sink(*[x.base for x in tsink.src])

  # explicit rangeify
  rctx = IndexingContext()
  consumer_map = tsink_base.get_consumer_map()
  ending_ranges: dict[UOp, bool] = {}
  for x in tsink_base.reverse_toposort(consumer_map):
    if x.op in {Ops.DEVICE, Ops.UNIQUE}: continue
    ending_ranges[x] = any(ending_ranges[u] for u in consumer_map[x])

    # if this element has weight and it's ending a range, we (force) realize it
    if ending_ranges[x] and x.op in GroupOp.Elementwise.union({Ops.REDUCE_AXIS}):
      if x.op_in_backward_slice_with_self(Ops.BUFFER, Ops.REALIZE, Ops.BUFFERIZE, Ops.CONTIGUOUS):
        if x.op_in_backward_slice_with_self(Ops.REDUCE_AXIS):
          realize_map[x] = None

    # if we are realizing, it doesn't matter if we are ending ranges
    if x in realize_map: ending_ranges[x] = False

    # *** these are the ranges on the output ***

    consumer_rngs = [rctx.range_map[c][0] for c in consumer_map[x] if c in rctx.range_map]
    if x in realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      #assert x.op not in GroupOp.Movement
      out_rngs = [rctx.new_range(s) for s in x.shape]
    elif x.op in {Ops.MSTACK, Ops.MSELECT}:
      # treat MSTACK/MSELECT like SINK
      continue
    elif len(consumer_rngs) == 0:
      # if no consumers have ranges and this isn't realized, this doesn't have ranges either.
      continue
    elif len(consumer_rngs) == 1:
      # if this has one consumer, it inherits the ranges from it
      out_rngs = consumer_rngs[0]
    elif len(consumer_rngs) > 1:
      # if this has two consumers, we have to merge the ranges and might create new ones
      all_rngs = list(zip(*consumer_rngs))
      rngs_valids = []
      for valid_rngs in all_rngs:
        local_rngs, valids = zip(*[(r.get_idx(), r.get_valid()) for r in valid_rngs])
        # if a range has a 1 src, it's the same as UOp.const(dtypes.index, 0)
        same_rngs = [x if x.op is not Ops.RANGE or resolve(x.src[0] != 1) else UOp.const(dtypes.index, 0) for x in local_rngs]
        rngs_valids.append((local_rngs, valids, all_same(same_rngs)))

      # TODO: in RANGEIFY > 1 all_all_same isn't required
      all_all_same = all(same_rngs for _,_,same_rngs in rngs_valids)
      out_rngs = []
      for i,(local_rngs,valids,same_rngs) in enumerate(rngs_valids):
        # we compare the ranges without their valids
        if all_all_same:
          # the new valid is the OR of all the children valids
          minimum_valid = functools.reduce(operator.or_, valids, UOp.const(dtypes.bool, False))
          out_rngs.append(minimum_valid.where(local_rngs[0], UOp.invalid()).simplify())
        else:
          out_rngs.append(rctx.new_range(x.shape[i]))

      # we have to realize here if there's new ranges
      if not all_all_same: realize_map[x] = None

    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # rngs is the input ranges
    rngs = out_rngs

    # handle REDUCE
    if x.op is Ops.REDUCE_AXIS:
      rngs = rngs[:]
      for i,s in enumerate(x.src[0].shape):
        if i in x.arg[1]: rngs[i] = rctx.new_range(s, axistype=AxisType.REDUCE)

    # apply movement ops. this is the definition of them
    if x.op is Ops.SHRINK:  rngs = [a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(rngs, x.arg)]
    if x.op is Ops.PERMUTE: rngs = [rngs[p] for p in argsort(x.arg)]
    if x.op is Ops.FLIP:    rngs = [((s-1)-a) if f else a for a,s,f in zip(rngs, x.shape, x.arg)]
    if x.op is Ops.EXPAND:
      rngs = [a if resolve(x==y, False) else a.const_like(0) for a,x,y in zip(rngs, x.src[0].shape, x.shape)]
      ending_ranges[x] = True
    if x.op is Ops.PAD:
      rngs = rngs[:]
      bigwhere = UOp.const(dtypes.bool, True)
      for i,(sh,(s,e)) in enumerate(zip(x.shape, x.arg)):
        if s == 0 and e == 0: continue
        where = UOp.const(dtypes.bool, True)
        if resolve(e > 0): where = where & (rngs[i] < (sh-e))
        if resolve(s > 0): where = where & (rngs[i] >= s)
        bigwhere = bigwhere & where
        with Context(TRACK_MATCH_STATS=0):
          rngs[i] = graph_rewrite(where.where(rngs[i]-s, UOp.invalid()), sym)
      # PAD is replaced with a WHERE in the big graph to inject the 0s at the right place
      rctx.pads_gate[x] = bigwhere.simplify()
    if x.op is Ops.RESHAPE:
      acc = 1
      to_sum = []
      for s,src in list(zip(x.shape, rngs))[::-1]:
        to_sum.append(acc*src)
        acc *= s
      mish = sum(to_sum, start=UOp.const(dtypes.index, 0))
      ret:list[UOp] = []
      for s in x.src[0].shape[::-1]:
        ret.append(mish % s) # NOTE: simplify will turn this to CONST
        mish //= s
      # this simplify is doing a lot of heavy lifting. this is the replacement for the view merger in RESHAPE
      rngs = list(UOp.sink(*ret[::-1]).simplify().src)

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

    if debug:
      print("***" if x in realize_map else "   ", len(consumer_map[x]), f"{str(x.op):20s}",
            UOp.sink().index(*rngs).render(), " -> ", UOp.sink().index(*out_rngs).render())
  rctx.realize_map = realize_map
  tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
  return tsink, rctx
