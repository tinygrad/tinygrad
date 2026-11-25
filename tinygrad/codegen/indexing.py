from typing import Iterator
import functools, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, rewrite_group, broadcast_axes
from tinygrad.uop.ops import gate_kernel_sink
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile

@dataclass
class IndexingContext:
  device: str|tuple[str, ...]|None = None
  realize_map: dict[UOp, None] = field(default_factory=dict)
  non_removable: dict[UOp, None] = field(default_factory=dict)
  range_map: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint) -> UOp:
    if isinstance(s, UOp) and s.op is Ops.RANGE: return s
    # if a range has a 1 src, it's the same as UOp.const(0)
    return UOp.range(s, next(self.range_idx), AxisType.WEAK) if resolve(s!=1) else UOp.const(0)


ALWAYS_CONTIGUOUS: set[Ops] = GroupOp.Defines|{Ops.AFTER, Ops.CONST, Ops.MSELECT, Ops.MSTACK, Ops.LOAD, Ops.CALL}

@dataclass(frozen=True)
class BufferizeOpts:
  device: str|tuple[str, ...]|None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  removable: bool = True

def broadcast_rngs(x:UOp, src:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable|{Ops.STORE}: return rngs
  baxes, nleft = broadcast_axes(src.shape, x.shape), len(x.shape)-len(src.shape)
  return tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(rngs) if j >= nleft)

# TODO: srcs contain (real data srcs, something else, ranges) and the boundary is confusing. see range_start
def data_srcs(op:Ops, src:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if op in GroupOp.Defines|{Ops.RANGE, Ops.SPECIAL}: return ()
  if op in GroupOp.Movement|{Ops.INDEX, Ops.STAGE, Ops.REDUCE, Ops.AFTER, Ops.END, Ops.BACKEDGE, Ops.COPY}: return src[:1]
  return src

def _stack_select(r0:UOp, srcs:list[UOp]) -> UOp:
  # MSB-first bit tree
  if len(srcs) == 1: return srcs[0]
  half = 1 << ((len(srcs)-1).bit_length()-1)
  return (r0 % (2*half) < half).where(_stack_select(r0, srcs[:half]), _stack_select(r0, srcs[half:]))

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes, axes_out = UOp.const(0).usum(axes_in), []
  for s in in_shape[::-1]:
    axes_out.append(combined_axes % s)
    combined_axes //= s
  return graph_rewrite(UOp.sink(*axes_out[::-1]), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

# this is the definition of the movement ops
@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.SHRINK:  rngs = tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
    case Ops.EXPAND:  rngs = rngs[len(arg):]
    case Ops.PAD:
      # NOTE: the .where(r-s, i) is not inside the graph_rewrite so that `convert_pad_to_where_to_keep_behavior_local`
      #       wraps the pad with only the newly added valid
      rngs = tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs)
      if not all(r.op is Ops.CONST or (r.op is Ops.RANGE and r.vmax > 0) for r in rngs): sink = sink.simplify()
      sub_array = {r:r.replace(arg=(AxisType.PLACEHOLDER, i)) for i,r in enumerate(sink.ranges)}
      rngs = _apply_reshape(in_shape, arg, sink.substitute(sub_array, walk=True)).substitute({v:k for k,v in sub_array.items()}, walk=True).src
    case _: raise RuntimeError(f"{op} is not a MovementOp")
  return rngs

@rewrite_group(new_ctx=False)
def run_rangeify(tsink:UOp) -> UOp:
  rctx = IndexingContext(device=tsink.device)

  # collect ops to realize and consumer map in a single toposort
  with cpu_profile("consumer map in rangeify", "TINY"):
    tsink_toposort = tsink.toposort(gate_kernel_sink)
    consumer_map: dict[UOp, dict[UOp, None]] = {x:{} for x in tsink_toposort}
    for c in tsink_toposort:
      for x in data_srcs(c.op, c.src):
        if x in consumer_map: consumer_map[x][c] = None
      if c.op is Ops.STORE:
        # always realize; also realize src if there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce
        rctx.realize_map[c] = None
        if len(c.src) == 2 and c.src[0].base in c.src[1].toposort(gate=lambda u, b=c.src[0].base: u is b or u.op is not Ops.AFTER):
          rctx.realize_map[c.src[1]] = None
      elif c.op in (Ops.MSELECT, Ops.MSTACK):
        # realize srcs of MSELECT/MSTACK
        for s in c.src:
          if s.base.op not in ALWAYS_CONTIGUOUS: rctx.realize_map[s] = None
      elif c.op is Ops.CALL and c.body.op in (Ops.SINK, Ops.PROGRAM):
        # realize the inputs of custom kernel calls
        for s in c.src[1:]:
          while s.op is Ops.RESHAPE: s = s.src[0]
          if s.op not in ALWAYS_CONTIGUOUS: rctx.realize_map[s] = rctx.non_removable[s] = None

  # explicit rangeify
  ending_ranges: dict[UOp, bool] = {}
  for x in reversed(tsink_toposort):
    # kernels (internal), AFTER and MSTACK/MSELECT (like SINK) have no ranges of their own
    if x.op in {Ops.CALL, Ops.AFTER, Ops.MSTACK, Ops.MSELECT}: continue

    ending_ranges[x] = any(ending_ranges.get(u, False) for u in consumer_map[x])
    # ranges the consumers iterate that this node broadcasts over
    ended = [rctx.range_map[c][0][i] for c in consumer_map[x] if c in rctx.range_map and c.op in GroupOp.Broadcastable
             for i in broadcast_axes(x.shape, c.shape)]
    broadcast_ending_ranges = any(r.ranges for r in ended)
    # fusion decision: REDUCE before the broadcast
    if x.op is Ops.REDUCE: ending_ranges[x] |= broadcast_ending_ranges

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = [broadcast_rngs(c, x, rctx.range_map[c][0]) for c in consumer_map[x] if c in rctx.range_map]
    if x in rctx.realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      out_rngs = tuple(rctx.new_range(s) for s in x.shape)
      # all ranges are ended now
      ending_ranges[x] = False
    elif len(consumer_rngs) == 0:
      # if no consumers have ranges and this isn't realized, this doesn't have ranges either.
      continue
    elif len(consumer_rngs) == 1:
      # if this has one consumer, it inherits the ranges from it
      out_rngs = consumer_rngs[0]
    elif len(consumer_rngs) > 1:
      # if this has two consumers, we have to merge the ranges and might create new ones
      # we compare the ranges without their valids
      rngs_valids = [tuple(zip(*((r.get_idx(), r.get_valid()) for r in vr))) for vr in zip(*consumer_rngs)]
      if all(all_same(lr) for lr, _ in rngs_valids):
        # the new valid is the OR of all the children valids
        out_rngs = tuple(graph_rewrite(lr[0].valid(UOp.const(False).usum(vs)), symbolic, name="minimum_valid") for lr, vs in rngs_valids)
      else:
        # we have to realize here if there's new ranges
        rctx.realize_map[x] = None
        out_rngs = tuple(rctx.new_range(s) for s in x.shape)

    # if this element is a reduce and there's ended ranges, we might have to end some other ranges
    if ending_ranges[x] and x.op in GroupOp.Elementwise.union({Ops.REDUCE}):
      ending_ranges[x] = False
      if out_rngs:
        rctx.realize_map[x] = None
        out_rngs = tuple(rctx.new_range(x.shape[i]) for i in range(len(out_rngs)))
    ending_ranges[x] |= broadcast_ending_ranges

    # TODO: some ops don't have shape, enable this after the `.st` property is removed
    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # *** the ranges on the inputs are
    #  1. swizzled for MovementOps
    #  2. newly created for REDUCE (tensor graph form with axis)
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges  # pylint: disable=possibly-used-before-assignment

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, rngs)
    # STACK: the leading range selects the src, srcs get the trailing ranges
    if x.op is Ops.STACK: rngs = out_rngs[1:]
    # when the index srcs and output ranges of INDEX address every axis of its src, the index srcs select the leading axes
    if x.op is Ops.INDEX and len(x.src)-1+len(out_rngs) == x.src[0].ndim: rngs = x.src[1:]+out_rngs
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] |= any(r.ranges for r in out_rngs[:len(x.marg)])

    # REDUCE creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE:
      rngs = tuple(rctx.new_range(s) for s in x.src[0].shape[:x.arg[1]]) + out_rngs

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

  # apply rangeify in a single bottom-up pass over tsink_toposort
  replaced, staged = dict[UOp, UOp](), dict[UOp, UOp]()
  for x in tsink_toposort:
    op, in_range_map = x.op, x in rctx.range_map
    if op is Ops.STAGE or (op is Ops.INDEX and (not in_range_map or len(rctx.range_map[x][0]) == len(rctx.range_map[x][1]))):
      rebuilt_srcs = tuple(replaced.get(s, s) for s in x.src)
      replaced[x] = UOp(op, rebuilt_srcs, x.arg) if rebuilt_srcs != x.src else x
      continue
    # shape/bound/index args that are not data src should not be indexed
    skip, data_src_count = int(op is Ops.CALL), len(data_srcs(op, x.src))
    new_srcs = list(x.src[:skip])
    for i in range(skip, len(x.src)):
      s = x.src[i]
      new_src = replaced.get(s, s)
      src_rngs = broadcast_rngs(x, s, rctx.range_map[x][0]) if in_range_map else ()
      if s.op in GroupOp.Defines|{Ops.MSTACK, Ops.MSELECT, Ops.AFTER, Ops.STAGE}:
        if in_range_map and i < data_src_count: new_src = new_src.index(*src_rngs)
      elif s in rctx.realize_map:
        closed_ranges = rctx.range_map[s][1]
        if s.op is Ops.STORE:
          # add the ends if this is a store
          new_src = new_src.end(*[r for r in closed_ranges if r.op is Ops.RANGE])
        else:
          if (stage:=staged.get(s)) is None:
            # if a deviceless value must materialize, place it on the sink device
            staged[s] = stage = UOp(Ops.STAGE, src=(new_src,)+closed_ranges, arg=BufferizeOpts(device=s.device or rctx.device,
                                    removable=s.op not in ALWAYS_CONTIGUOUS and s not in rctx.non_removable))
          new_src = stage.index(*src_rngs) if in_range_map else stage
      new_srcs.append(new_src)
    # REDUCE(op, axis) -> REDUCE(op) with ranges (only non-singleton popped axes contribute loops)
    if op is Ops.REDUCE and x.arg[1] != 0:
      if not in_range_map: raise RuntimeError("REDUCE has no ranges in rangeify, UOp verification failed")
      replaced[x] = UOp(Ops.REDUCE, src=(new_srcs[0],)+tuple(r for r in rctx.range_map[x][0][:x.arg[1]] if r.op is Ops.RANGE)+tuple(new_srcs[1:]),
                        arg=(x.arg[0], 0))
    # PAD -> WHERE
    elif op in (Ops.PAD, Ops.INDEX) and in_range_map:
      replaced[x] = UOp.const(True).uprod([r.get_valid() for r in rctx.range_map[x][0]]).where(new_srcs[0], UOp.const(x.dtype.const(0)))
    # STACK -> WHERE select on the leading range (only data STACKs are in range_map)
    elif op is Ops.STACK and in_range_map: replaced[x] = _stack_select(rctx.range_map[x][1][0], new_srcs)
    # remove movement op
    elif op in GroupOp.Movement and (in_range_map or new_srcs[0].op is Ops.INDEX): replaced[x] = new_srcs[0]
    else: replaced[x] = UOp(op, tuple(new_srcs), x.arg) if tuple(new_srcs) != x.src else x
  return replaced[tsink]
