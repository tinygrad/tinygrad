from typing import Iterator, Sequence
import functools, operator, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp
from tinygrad.uop.symbolic import sym
from tinygrad.helpers import argsort, all_same, Context
from tinygrad.uop.ops import graph_rewrite, sint, AxisType

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.ASSIGN, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_GLOBAL,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD, Ops.KERNEL}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_assign(ctx:dict[UOp, None], a:UOp) -> None:
  if a.src[1].op not in ALWAYS_CONTIGUOUS: ctx[a.src[1]] = None
  # if it's a kernel, we don't realize it
  if a.src[1].op is not Ops.KERNEL: ctx[a] = None

pm_generate_realize_map = PatternMatcher([
  # always realize SINK src
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)),
  # always realize COPY/BUFFER_VIEW/CONTIGUOUS
  (UPat({Ops.COPY, Ops.BUFFER_VIEW, Ops.CONTIGUOUS}, name="tr"), realize),
  # realize srcs of COPY, MSELECT, MSTACK
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_srcs),
  # realize ASSIGN and input to assign (might be optimized out)
  (UPat(Ops.ASSIGN, name="a"), realize_assign),
])

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL

@dataclass
class IndexingContext:
  realize_map: dict[UOp, None] = field(default_factory=dict)
  range_map: dict[UOp, tuple[list[UOp], list[UOp]]] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP):
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.index, 0)

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.BUFFERIZE, Ops.INDEX, Ops.KERNEL}: return None
  if x.op is Ops.ASSIGN and x.src[1].op is Ops.KERNEL: return None
  new_srcs = []
  for s in x.src:
    new_src = s
    if s.op in {Ops.BUFFER, Ops.BUFFER_VIEW, Ops.MSTACK, Ops.MSELECT} or (s.op is Ops.ASSIGN and s.src[1].op is Ops.KERNEL):
      if x in ctx.range_map: new_src = new_src.index(*ctx.range_map[x][0])
    elif s in ctx.realize_map:
      new_src = UOp(Ops.BUFFERIZE, s.dtype, src=(new_src,)+tuple(ctx.range_map[s][1]), arg=BufferizeOpts(device=s.device), tag=s.tag)
      if x in ctx.range_map: new_src = new_src.index(*ctx.range_map[x][0])
    new_srcs.append(new_src)
  # NOTE: do we need this?
  return x.replace(src=tns) if x.src != (tns:=tuple(new_srcs)) else None

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  valid: UOp = functools.reduce(operator.and_, [r.get_valid() for r in ctx.range_map[x][0]], UOp.const(dtypes.bool, True))
  ret = valid.where(x.src[0], UOp.const(x.dtype, 0))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def convert_reduce_axis_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  # input ranges
  new_ranges = [r for i,r in enumerate(ctx.range_map[x][0]) if i in x.arg[1]]
  ret = UOp(Ops.REDUCE, x.dtype, src=(x.src[0],)+tuple(new_ranges), arg=x.arg[0], tag=x.tag)
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

def add_third_op_to_assign_to_track_shape(ctx:IndexingContext, assign:UOp):
  if assign.src[1].op is Ops.KERNEL: return None
  to_mop = graph_rewrite(assign.src[0], PatternMatcher([(UPat(GroupOp.Movement, name="x"), lambda x: x.replace(tag=()))]))
  ret = assign.replace(src=assign.src+(to_mop,))
  ctx.range_map[ret] = ctx.range_map[assign]
  return ret

pm_apply_rangeify = PatternMatcher([
  # REDUCE_AXIS -> REDUCE
  (UPat(Ops.REDUCE_AXIS, name="x"), convert_reduce_axis_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # add third op to assign
  (UPat(Ops.ASSIGN, src=(UPat(), UPat()), name="assign"), add_third_op_to_assign_to_track_shape),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
  # const/define_var shouldn't have src
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"), lambda ctx,c: c.replace(src=()) if c in ctx.range_map else None),
])

def apply_movement_op(x:UOp, rngs:Sequence[UOp]) -> list[UOp]:
  if x.op is Ops.SHRINK:  rngs = [a+ss if resolve(ss != 0) else a for a,(ss,_) in zip(rngs, x.arg)]
  elif x.op is Ops.PERMUTE: rngs = [rngs[p] for p in argsort(x.arg)]
  elif x.op is Ops.FLIP:    rngs = [((s-1)-a) if f else a for a,s,f in zip(rngs, x.shape, x.arg)]
  elif x.op is Ops.EXPAND:
    rngs = [a.const_like(0) if resolve(in_sh!=out_sh) else a for a,in_sh,out_sh in zip(rngs, x.src[0].shape, x.shape)]
  elif x.op is Ops.PAD:
    rngs = list(rngs)
    for i,(sh,(s,e)) in enumerate(zip(x.shape, x.arg)):
      if s == 0 and e == 0: continue
      where = UOp.const(dtypes.bool, True)
      if resolve(e > 0): where = where & (rngs[i] < (sh-e))
      if resolve(s > 0): where = where & (rngs[i] >= s)
      with Context(TRACK_MATCH_STATS=0): rngs[i] = graph_rewrite(where.where(rngs[i]-s, UOp.invalid()), sym)
  elif x.op is Ops.RESHAPE:
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
  else: raise RuntimeError(f"{x.op} is not a MovementOp")
  return rngs

# movement op on INDEX as a PatternMatcher
#pm_mops = PatternMatcher([
#  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), lambda r,idx: r.src[0].index(*apply_movement_op(r, idx.src[1:]))),
#])

def run_rangeify(tsink:UOp, debug:bool=False) -> tuple[UOp, IndexingContext]:
  rctx = IndexingContext()

  # get ops to realize
  graph_rewrite(tsink, pm_generate_realize_map, ctx=rctx.realize_map, name="Input Graph")

  # explicit rangeify
  ending_ranges: dict[UOp, bool] = {}
  for x in tsink.reverse_toposort(consumer_map:=tsink.get_consumer_map()):
    if x.op in {Ops.DEVICE, Ops.UNIQUE}: continue
    ending_ranges[x] = any(ending_ranges[u] for u in consumer_map[x])

    # if this element has weight and it's ending a range, we (force) realize it
    if ending_ranges[x] and x.op in GroupOp.Elementwise.union({Ops.REDUCE_AXIS}):
      if x.op_in_backward_slice_with_self(Ops.BUFFER, Ops.BUFFERIZE, Ops.CONTIGUOUS):
        if x.op_in_backward_slice_with_self(Ops.REDUCE_AXIS):
          rctx.realize_map[x] = None

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = [rctx.range_map[c][0] for c in consumer_map[x] if c in rctx.range_map]
    if x in rctx.realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      out_rngs = [rctx.new_range(s) for s in x.shape]
      # all ranges are ended now
      ending_ranges[x] = False
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
      if not all_all_same: rctx.realize_map[x] = None

    # TODO: some ops don't have shape, enable this after the `.st` property is removed
    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # *** the ranges on the inputs are
    #  1. swizzled for MovementOps
    #  2. newly created for REDUCE_AXIS
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x, rngs)
    if x.op is Ops.EXPAND: ending_ranges[x] = True

    # REDUCE_AXIS creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE_AXIS:
      rngs = rngs[:]
      for i,s in enumerate(x.src[0].shape):
        if i in x.arg[1]: rngs[i] = rctx.new_range(s, axistype=AxisType.REDUCE)

    if debug:
      print("***" if x in rctx.realize_map else "   ", len(consumer_map[x]), f"{str(x.op):20s}",
            UOp.sink().index(*rngs).render(), " -> ", UOp.sink().index(*out_rngs).render())

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

  tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
  return tsink, rctx
