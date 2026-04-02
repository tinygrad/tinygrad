from typing import Iterator, Sequence
import functools, itertools
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, profile_matches
from tinygrad.uop.ops import consumer_map_from_toposort, gate_kernel_sink
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile, PCONTIG, colored, prod

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.AFTER, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD, Ops.CALL}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_store_after_src(ctx:dict[UOp, None], dest:UOp, src:UOp):
  # don't realize COPY/BUFFER_VIEW when they are the direct source of STORE+AFTER — the target buffer is the output
  if src.op in {Ops.COPY, Ops.BUFFER_VIEW} and src in ctx \
     and not dest.op_in_backward_slice_with_self(Ops.SHRINK, Ops.PERMUTE, Ops.FLIP, Ops.PAD):
    del ctx[src]
  # you don't usually have to do this for assign unless there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce
  if src.contains_in_backward_slice_with_self(dest.base): ctx[src] = None

pm_generate_realize_map = PatternMatcher([
  # always realize
  (UPat({Ops.COPY, Ops.CONTIGUOUS}, name="tr"), realize),
  # realize AFTER of STORE+AFTER
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE)), allow_any_len=True, name="tr"), realize),
  # realize srcs of these
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_srcs),
  # sometimes realize/unrealize src of store+after
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat.var("dest"), UPat.var("src"))))), realize_store_after_src),
])

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  removable: bool = True

@dataclass
class IndexingContext:
  realize_map: dict[UOp, None|list[int]] = field(default_factory=dict)
  range_map: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = field(default_factory=dict)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.LOOP) -> UOp:
    if isinstance(s, UOp) and s.op is Ops.RANGE: return s
    # if a range has a 1 src, it's the same as UOp.const(dtypes.weakint, 0)
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(dtypes.weakint, 0)

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  if x.op in {Ops.BUFFERIZE, Ops.INDEX}: return None
  x_range_map = ctx.range_map.get(x)
  x_input_ranges = x_range_map[0] if x_range_map is not None else None
  new_srcs = []
  for s in x.src:
    new_src = s
    if s.op in {Ops.PARAM, Ops.BUFFER_VIEW, Ops.MSTACK, Ops.MSELECT} or \
       (s.op is Ops.AFTER and not any(c.op in {Ops.STORE, Ops.END} for c in s.src[1:])):
      if x_input_ranges is not None: new_src = new_src.index(*x_input_ranges)
    elif s in ctx.realize_map:
      realized_ranges = ctx.realize_map[s]
      assert isinstance(realized_ranges, list), "realize map must contain range list"
      s_output_ranges = ctx.range_map[s][1]
      closed_ranges = tuple(r for i,r in enumerate(s_output_ranges) if i in realized_ranges)
      if s.op is Ops.STORE:
        # add the ends if this is a store
        new_src = s.end(*[r for r in closed_ranges if r.op is Ops.RANGE])
        del ctx.realize_map[s]
      else:
        # the Bufferize before a COPY is not removable. there should be a better way to do this
        removable = x.op is not Ops.COPY and s.op not in ALWAYS_CONTIGUOUS
        # None in the device assigns it a number later
        opts = BufferizeOpts(device=s.device, removable=removable) if len(ctx.range_map[s][1]) == len(realized_ranges) else \
               BufferizeOpts(device=s.device, addrspace=AddrSpace.LOCAL, removable=removable)
        new_src = UOp(Ops.BUFFERIZE, s.dtype, src=(new_src,)+closed_ranges, arg=opts)
        if x_input_ranges is not None: new_src = new_src.index(*(r for i,r in enumerate(x_input_ranges) if i in realized_ranges))
    new_srcs.append(new_src)
  # NOTE: do we need this?
  return x.replace(src=tns) if x.src != (tns:=tuple(new_srcs)) else None

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  if x not in ctx.range_map: return None
  valid: UOp = UOp.const(dtypes.bool, True).uprod(*[r.get_valid() for r in ctx.range_map[x][0]])
  ret = valid.where(x.src[0], UOp.const(x.dtype, 0))
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def convert_reduce_axis_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  # input ranges
  new_ranges = [r for i,r in enumerate(ctx.range_map[x][0]) if i in x.arg[1]]
  ret = UOp(Ops.REDUCE, x.dtype, src=(x.src[0],)+tuple(new_ranges), arg=x.arg[0])
  ctx.range_map[ret] = ctx.range_map[x]
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]

pm_apply_rangeify = PatternMatcher([
  # REDUCE_AXIS -> REDUCE
  (UPat(Ops.REDUCE_AXIS, name="x"), convert_reduce_axis_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
])

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes = UOp.const(dtypes.weakint, 0).usum(*axes_in)
  axes_out:list[UOp] = []
  for s in in_shape[::-1]:
    axes_out.append(combined_axes % s)
    combined_axes //= s
  # this simplify is doing a lot of heavy lifting. this is the replacement for the reshape view merging code
  return graph_rewrite(UOp.sink(*axes_out[::-1]), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

@functools.cache
def _pad_movement_template(in_shape:tuple[sint, ...], arg:tuple) -> tuple[tuple[UOp, ...], UOp]:
  prngs = tuple(UOp.range(sh+s+e, i, AxisType.PLACEHOLDER) for i,(sh,(s,e)) in enumerate(zip(in_shape, arg)))
  trngs = tuple(r if (s == 0 and e == 0) else graph_rewrite((r >= s) & (r < (sh+s)),
    symbolic+pm_simplify_valid, name="pad").where(r-s, UOp.invalid()) for r,sh,(s,e) in zip(prngs, in_shape, arg))
  return prngs, UOp.sink(*trngs)

@functools.cache
def _reshape_movement_template(in_shape:tuple[sint, ...], out_shape:tuple[sint, ...]) -> tuple[tuple[UOp, ...], UOp]:
  prngs = tuple(UOp.range(s, i, AxisType.PLACEHOLDER) for i,s in enumerate(out_shape))
  return prngs, _apply_reshape(in_shape, out_shape, UOp.sink(*prngs).simplify())

def _substitute_template(template:UOp, mapping:dict[UOp, UOp]) -> tuple[UOp, ...]:
  replaced: dict[UOp, UOp] = {}
  stack: list[tuple[UOp, bool]] = [(template, False)]
  while stack:
    node, visited = stack.pop()
    if node in replaced: continue
    if (mapped:=mapping.get(node)) is not None:
      replaced[node] = mapped
      continue
    if visited:
      new_src = tuple(replaced[s] for s in node.src)
      replaced[node] = node if new_src == node.src else UOp(node.op, node.dtype, new_src, node.arg, node.tag)
    else:
      stack.append((node, True))
      for s in reversed(node.src):
        if s not in replaced: stack.append((s, False))
  return replaced[template].src

def _shape_eq(a:sint, b:sint) -> bool:
  return (isinstance(a, int) and isinstance(b, int) and a == b) or a is b

def _shape_is_one(s:sint) -> bool:
  return isinstance(s, int) and s == 1

def _reshape_insert_remove_ones(in_shape:tuple[sint, ...], out_shape:tuple[sint, ...], rngs:tuple[UOp, ...]) -> tuple[UOp, ...]|None:
  if tuple(s for s in in_shape if not _shape_is_one(s)) != tuple(s for s in out_shape if not _shape_is_one(s)): return None
  out_rngs = iter(r for r,s in zip(rngs, out_shape) if not _shape_is_one(s))
  return tuple(next(out_rngs) if not _shape_is_one(s) else UOp.const(dtypes.weakint, 0) for s in in_shape)

def _reshape_linearize(out_shape:tuple[int, ...], rngs:tuple[UOp, ...]) -> UOp:
  acc: UOp = UOp.const(dtypes.weakint, 0)
  stride = 1
  for s,r in zip(out_shape[::-1], rngs[::-1]):
    acc = r if stride == 1 and acc.arg == 0 else r*stride + acc if stride != 1 else r + acc
    stride *= s
  return acc

def _reshape_unlinearize(in_shape:tuple[int, ...], rng:UOp) -> tuple[UOp, ...]:
  ret: list[UOp] = []
  curr = rng
  for i,s in enumerate(in_shape[::-1]):
    if i == len(in_shape)-1: ret.append(curr)
    else:
      ret.append(curr % s)
      curr //= s
  return tuple(ret[::-1])

@functools.cache
def _reshape_fastpath_template(in_shape:tuple[sint, ...], out_shape:tuple[sint, ...]) -> tuple[tuple[UOp, ...], UOp]|None:
  rngs = tuple(UOp.range(s, i, AxisType.PLACEHOLDER) for i,s in enumerate(out_shape))
  prefix = 0
  while prefix < len(in_shape) and prefix < len(out_shape) and _shape_eq(in_shape[prefix], out_shape[prefix]): prefix += 1
  suffix = 0
  while suffix < len(in_shape)-prefix and suffix < len(out_shape)-prefix and _shape_eq(in_shape[-1-suffix], out_shape[-1-suffix]): suffix += 1

  mid_in = in_shape[prefix:len(in_shape)-suffix if suffix else len(in_shape)]
  mid_out = out_shape[prefix:len(out_shape)-suffix if suffix else len(out_shape)]
  mid_rngs = rngs[prefix:len(rngs)-suffix if suffix else len(rngs)]

  if (fast_mid := _reshape_insert_remove_ones(mid_in, mid_out, mid_rngs)) is not None:
    return rngs, UOp.sink(*(rngs[:prefix] + fast_mid + (rngs[len(rngs)-suffix:] if suffix else ())))
  if len(mid_in) == 1 and isinstance(mid_in[0], int) and all(isinstance(s, int) for s in mid_out) and prod(mid_out) == mid_in[0]:
    return rngs, UOp.sink(*(rngs[:prefix] + (_reshape_linearize(tuple(mid_out), mid_rngs),) + (rngs[len(rngs)-suffix:] if suffix else ())))
  if len(mid_out) == 1 and isinstance(mid_out[0], int) and all(isinstance(s, int) for s in mid_in) and prod(mid_in) == mid_out[0]:
    return rngs, UOp.sink(*(rngs[:prefix] + _reshape_unlinearize(tuple(mid_in), mid_rngs[0]) + (rngs[len(rngs)-suffix:] if suffix else ())))
  if all(isinstance(s, int) for s in mid_in) and all(isinstance(s, int) for s in mid_out) and (mid_prod:=prod(mid_in)) != 0 and mid_prod == prod(mid_out):
    linear = _reshape_linearize(tuple(mid_out), mid_rngs)
    return rngs, UOp.sink(*(rngs[:prefix] + _reshape_unlinearize(tuple(mid_in), linear) + (rngs[len(rngs)-suffix:] if suffix else ())))
  return None

# this is the definition of the movement ops
@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.SHRINK:  rngs = tuple(a if ss == 0 else a+ss for a,(ss,_) in zip(rngs, arg))
    case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
    case Ops.FLIP:    rngs = tuple(((s-1)-a) if f else a for a,s,f in zip(rngs, in_shape, arg))
    case Ops.EXPAND:  rngs = tuple(a if in_sh == out_sh else a.const_like(0) for a,in_sh,out_sh in zip(rngs, in_shape, arg))
    case Ops.PAD:
      prngs, pad_template = _pad_movement_template(in_shape, arg)
      rngs = pad_template.src if all(a is b for a,b in zip(rngs, prngs)) else _substitute_template(pad_template, dict(zip(prngs, rngs)))
    case Ops.RESHAPE:
      if (fast_template:=_reshape_fastpath_template(in_shape, arg)) is not None: prngs, reshape_template = fast_template
      else: prngs, reshape_template = _reshape_movement_template(in_shape, arg)
      rngs = reshape_template.src if all(a is b for a,b in zip(rngs, prngs)) else _substitute_template(reshape_template, dict(zip(prngs, rngs)))
    case _: raise RuntimeError(f"{op} is not a MovementOp")
  return rngs

def merge_ending_ranges(consumers:Sequence[UOp], ending_ranges:dict[UOp, list[UOp]]) -> list[UOp]:
  if len(consumers) == 0: return []
  if len(consumers) == 1: return list(ending_ranges.get(consumers[0], ()))
  merged: list[UOp] = []
  for u in consumers: merged.extend(ending_ranges.get(u, ()))
  return merged

@profile_matches
def run_rangeify(tsink:UOp, debug:bool=False) -> tuple[UOp, IndexingContext]:
  if debug: print("**************************")
  rctx = IndexingContext()

  # get ops to realize
  graph_rewrite(tsink, pm_generate_realize_map, ctx=rctx.realize_map, name="get realize")

  # get the consumer map
  with cpu_profile("consumer map in rangeify", "TINY"):
    consumer_map = consumer_map_from_toposort(tsink_toposort:=tsink.toposort(gate_kernel_sink))

  # explicit rangeify
  ending_ranges: dict[UOp, list[UOp]] = {}
  for x in reversed(tsink_toposort):
    if x.op in {Ops.DEVICE, Ops.UNIQUE}: continue

    # no ranges on kernels, they are internal
    if x.op in {Ops.CALL, Ops.LINEAR}: continue

    # only STORE+AFTER has range
    if x.op is Ops.AFTER and all(s.op is not Ops.STORE for s in x.src[1:]): continue

    # treat MSTACK/MSELECT like SINK
    if x.op in {Ops.MSTACK, Ops.MSELECT}: continue

    if x.dtype.scalar() == dtypes.weakint: continue  # TODO: why do I need this?
    consumers = consumer_map[x]
    ending_ranges[x] = merge_ending_ranges(consumers, ending_ranges)

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = [cr[0] for c in consumers if (cr:=rctx.range_map.get(c)) is not None]
    if x in rctx.realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      out_rngs = tuple(rctx.new_range(s) for s in x.shape)
      # all ranges are ended now
      ending_ranges[x] = []
      # mark all ranges as ended
      assert rctx.realize_map[x] is None
      rctx.realize_map[x] = list(range(len(x.shape)))
    elif len(consumer_rngs) == 0:
      # if no consumers have ranges and this isn't realized, this doesn't have ranges either.
      continue
    elif len(consumer_rngs) == 1:
      # if this has one consumer, it inherits the ranges from it
      out_rngs = consumer_rngs[0]
    elif len(consumer_rngs) > 1:
      first_consumer_rngs = consumer_rngs[0]
      if all(rngs == first_consumer_rngs for rngs in consumer_rngs[1:]):
        out_rngs = first_consumer_rngs
      else:
        # if this has two consumers, we have to merge the ranges and might create new ones
        all_rngs: list[tuple[UOp, ...]] = list(zip(*consumer_rngs))
        rngs_valids = []
        for valid_rngs in all_rngs:
          local_rngs, valids = zip(*[(r.get_idx(), r.get_valid()) for r in valid_rngs])
          rngs_valids.append((local_rngs, valids))

        # TODO: in RANGEIFY > 1 all_all_same isn't required
        all_all_same = all(all_same(local_rngs) for local_rngs,_ in rngs_valids)
        _out_rngs = []
        _realize_axis = []
        for i,(local_rngs,valids) in enumerate(rngs_valids):
          # we compare the ranges without their valids
          if all_all_same or (PCONTIG and all_same(local_rngs)):
            # the new valid is the OR of all the children valids
            minimum_valid = UOp.const(dtypes.bool, False).usum(*valids)
            _out_rngs.append(graph_rewrite(minimum_valid.where(local_rngs[0], UOp.invalid()), symbolic, name="minimum_valid"))
          else:
            _out_rngs.append(rctx.new_range(x.shape[i]))
            _realize_axis.append(i)
        out_rngs = tuple(_out_rngs)

        # we have to (partially) realize here if there's new ranges
        if len(_realize_axis): rctx.realize_map[x] = _realize_axis

    # if this element is a reduce and there's ended ranges, we might have to end some other ranges
    if len(ending_ranges[x]) and x.op in GroupOp.Elementwise.union({Ops.REDUCE_AXIS}):
      _realize_axis = rctx.realize_map.get(x) or []
      for i,r in enumerate(out_rngs):
        if i in _realize_axis: continue
        if not (PCONTIG > 1) or any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
          _realize_axis.append(i)
      ending_ranges[x] = []
      if len(_realize_axis):
        rctx.realize_map[x] = _realize_axis
        out_rngs = tuple([(rctx.new_range(x.shape[i]) if i in _realize_axis else r) for i,r in enumerate(out_rngs)])

    # TODO: some ops don't have shape, enable this after the `.st` property is removed
    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # *** the ranges on the inputs are
    #  1. swizzled for MovementOps
    #  2. newly created for REDUCE_AXIS
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges  # pylint: disable=possibly-used-before-assignment

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, rngs)
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] += list(UOp.sink(*[ro for ri, ro in zip(rngs, out_rngs) if ri is not ro]).ranges.keys())

    # REDUCE_AXIS creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE_AXIS:
      rngs = tuple(rctx.new_range(s, axistype=AxisType.REDUCE) if i in x.arg[1] else r for i,(r,s) in enumerate(zip(rngs, x.src[0].shape)))

    if debug:
      realized_ranges = rctx.realize_map.get(x, None)
      if x.op is Ops.RESHAPE or len(rngs) != len(out_rngs):
        disp = render_ranges(rngs, realized=realized_ranges) + " -> " + render_ranges(out_rngs, realized=realized_ranges)
      else:
        disp = render_ranges(rngs, out_rngs, realized=realized_ranges)
      print("***" if x in rctx.realize_map else "   ",
            f"{len(consumer_map[x]):2d} {str(x.op):20s} {str(x._shape):35s} {len(ending_ranges[x]):2d}", disp)

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

  tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
  return tsink, rctx

def render_ranges(*rngs_list, realized) -> str:
  disp = []
  for i, rs in enumerate(zip(*[[r.render() for r in rngs] for rngs in rngs_list])):
    rng = rs[0] if all_same(rs) else " -> ".join(rs)
    if realized is not None and i in realized: rng = colored(rng, "yellow")
    disp.append("["+rng+"]")
  return ''.join(disp)
