from dataclasses import dataclass, field
import itertools
from tinygrad.dtype import AddrSpace, Invalid
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp, KernelInfo
from tinygrad.uop.ops import graph_rewrite, AxisType, rewrite_group, remove_all_tags, resolve
from tinygrad.helpers import all_int, VIZ, SPEC, Context, panic
from tinygrad.schedule.indexing import BufferizeOpts, apply_movement_op
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen.simplify import pm_reduce_simplify

# *** RANGE creation ***

def rangeify_on_reduce(ctx, inp:UOp, red:UOp, idx:UOp|None=None):
  if red.arg[1] == 0: return None
  if idx is None and len(red.shape) > 0: return None
  # TODO: is AxisType.REDUCE a real thing?
  rngs = [UOp.range(s, next(ctx), AxisType.REDUCE) for s in inp.shape[:red.arg[1]]]
  return inp.index(*rngs, *(idx.src[1:] if idx is not None else ())).reduce(*rngs, arg=(red.arg[0], 0))

def rangeify_on_store(ctx, x:UOp):
  if x.shape == (): return None
  rngs = [UOp.range(s, next(ctx)) for s in x.shape]
  return x.src[0].index(*rngs).store(x.src[1].index(*rngs)).end(*rngs)

def rangeify_on_stage(ctx, x:UOp):
  if x.src[0].shape == (): return None
  # size 1 dims don't get ranges, they are reshaped out and back in
  if all_int(x.shape) and 0 < len(sq := tuple(s for s in x.shape if s != 1)) < len(x.shape):
    return rangeify_on_stage(ctx, x.src[0].reshape(sq).bufferize(arg=x.arg)).reshape(x.shape)
  rngs = [UOp.range(s, next(ctx)) for s in x.shape]
  return x.replace(src=(x.src[0].index(*rngs), *rngs))

pm_range_creation = PatternMatcher([
  # reduce/store are what creates ranges
  (UPat(Ops.REDUCE, src=(UPat.var('inp'),), name="red").index(name="idx", allow_any_len=True), rangeify_on_reduce),
  (UPat(Ops.REDUCE, src=(UPat.var('inp'),), name="red"), rangeify_on_reduce),
  (UPat(Ops.STORE, name="x"), rangeify_on_store),
  (UPat(Ops.STAGE, name="x"), rangeify_on_stage),
])

# *** RANGE migration ***

# movement op on INDEX as a PatternMatcher
def _mop_index(r:UOp, idx:UOp):
  idxs = idx.src[1:]
  if len(idxs) == len(r.shape):
    ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), dtype=idx.dtype, arg=idx.arg)
    if r.op is Ops.PAD:
      # NOTE: neither 0 or ret.const_like(0) is correct here.
      # const_like breaks because it adds casts, and 0 is wrong if ret is a bool
      invalid_value = UOp.const(ret.dtype.const(0))
      # insert invalid_value for PAD with where
      a = UOp.const(True)
      for s in UOp.sink(*ret.src[1:]).simplify().src:
        if s.is_invalid: return invalid_value
        if s.op is Ops.WHERE and s.src[2].op is Ops.CONST and s.src[2].arg == Invalid: a = a & s.src[0]
      ret = a.where(ret, invalid_value)
    return ret
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0] if r.src[0].dtype == idx.dtype else None
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), dtype=idx.dtype, arg=idx.arg)
      return ret if ret.shape == idx.shape else None

# TODO: this should be in _mop_index
def index_on_stack(stack:UOp, idx:UOp):
  srcs = [s.index(*idx.src[2:]) for s in stack.src]
  r0 = idx.src[1]
  ret = srcs[-1]
  for k in range(len(srcs)-2, -1, -1): ret = r0.eq(k).where(srcs[k], ret)
  return ret

pm_range_migration = PatternMatcher([
  # STAGE on shape () is nothing
  (UPat(Ops.STAGE, src=(UPat.var('x'),)), lambda x: x if x.shape == () else None),
  # if INDEX is on STAGE with the same ranges, remove the pair
  (UPat(Ops.STAGE, allow_any_len=True, name="s").index(allow_any_len=True, name="i"),
   lambda s,i: s.src[0] if s.src[1:] == i.src[1:] else None),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(0) if x.marg == () and x.src[0].shape == (1,) else None),
  # handle movement ops on INDEX
  (UPat(GroupOp.Movement, name="r").index(name="idx", allow_any_len=True), _mop_index),
  (UPat(Ops.STACK, name="stack").index(name="idx", allow_any_len=True), index_on_stack),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)),
  # block bitcast that changes shape
  (UPat(Ops.BITCAST, name="b").index(allow_any_len=True),
   lambda b: panic(RuntimeError, "shape changing bitcast not allowed in rangeify") if b.src[0].shape != b.shape else None),
  # pass index through elementwise
  (UPat(GroupOp.Elementwise, name="b").index(name="idx", allow_any_len=True),
   lambda b,idx: b.replace(src=tuple(s.index(*idx.src[1:]) for s in b.src))),
  # INDEX without src is nothing (must be at the bottom)
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
])

# *** split into kernels ***

@dataclass
class SplitCtx:
  call_args:list = field(default_factory=list)
  range_number:int = -1
  addrspace:AddrSpace = AddrSpace.GLOBAL

def _split_graph(ctx:SplitCtx, u:UOp) -> UOp|None:
  if u.tag is not None: return None
  if u.addrspace != ctx.addrspace: return None
  us = u.flatten() if u.addrspace == AddrSpace.GLOBAL else u
  ctx.call_args.append(us)
  return us.param_like(len(ctx.call_args)-1).rtag().reshape(u.shape)

def _renumber_range(ctx:SplitCtx, u:UOp) -> UOp|None:
  if u.tag is not None: return None
  ctx.range_number += 1
  return u.replace(arg=(ctx.range_number, u.arg[-1])).rtag()

pm_split_graph = pm_range_migration+PatternMatcher([
  (UPat((Ops.PARAM, Ops.AFTER, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK), name="u"), _split_graph),
  (UPat(Ops.RANGE, name="u"), _renumber_range),
])

def split_store(x:UOp) -> UOp:
  ret = graph_rewrite(x, pm_split_graph, ctx:=SplitCtx(), name="split kernel", bottom_up=True)
  # TODO: params and args should be able to be in any order
  ctx.addrspace = AddrSpace.ALU
  ret = graph_rewrite(ret, pm_split_graph, ctx, name="split kernel (vars)", bottom_up=True)
  ret = graph_rewrite(ret, remove_all_tags, name="remove split tags", bottom_up=True)
  return ret.sink(arg=KernelInfo()).call(*ctx.call_args)

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

# cleanups

def strip_zero_offset_shrink(x:UOp) -> UOp:
  return x.src[0] if x.op is Ops.SHRINK and all(resolve(start == 0, False) for start,_ in x.marg) else x

def no_indexing_calls(u:UOp):
  new_srcs = []
  for x in u.src:
    if x.op is Ops.INDEX:
      # sometimes if call srcs have children the call will get an INDEX. we remove it here.
      # TODO: we should add safety checks here for contiguous
      new_srcs.append(x.src[0])
    elif x.op is Ops.SHRINK:
      # SHRINK with offset 0 is fine
      new_srcs.append(strip_zero_offset_shrink(x))
    elif x.op is Ops.MSTACK:
      new_srcs.append(x.replace(src=tuple(strip_zero_offset_shrink(s) for s in x.src)))
    else:
      # everything else we pass through
      new_srcs.append(x)
  return u.replace(src=tuple(new_srcs))

pm_no_indexing_calls = PatternMatcher([
  (UPat(Ops.CALL, name="u"), no_indexing_calls),
])

# *** rangeify ***

debug_tag_factor = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.rtag(ctx[0][x] if x not in ctx[1] else 'REAL') if x.tag is None else None),
])

def remove_stage(ctx, x:UOp) -> UOp:
  buf = UOp.new_buffer(x.arg.device, x.max_numel(), x.dtype, num=next(ctx))
  return buf.after(buf.reshape(x.shape).index(*x.src[1:]).store(x.src[0]).end(*x.src[1:])).reshape(x.shape)

pm_remove_stage = PatternMatcher([
  (UPat(Ops.STAGE, name="x"), remove_stage),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(tsink:UOp) -> UOp:
  # add safe STAGEs to never duplicate compute
  # we compute the number of times a buffer is consumed. if > 1, we realize
  realize = {}
  consumes = {tsink:0}
  for u in reversed(tsink.toposort()):
    assert u in consumes, f"{u.op} not in consumes"
    if (u.op in GroupOp.ALU or u.op is Ops.REDUCE) and consumes[u] > 1 and u.device is not None:
      # TODO: rename to stage
      realize[u] = u.rtag(1).bufferize(arg=BufferizeOpts(device=u.device))
      consumes[u] = 1
    if u.op is Ops.STORE: consumes[u] = 1
    if u.op is Ops.EXPAND: consumes[u] *= u.max_numel() // u.src[0].max_numel()
    for i,s in enumerate(u.src):
      if s not in consumes: consumes[s] = 0
      if u.op is not Ops.STORE or i > 0:
        consumes[s] += consumes[u]

  if VIZ:
    with Context(TRACK_MATCH_STATS=0): ctags = graph_rewrite(tsink, debug_tag_factor, ctx=(consumes, realize), bottom_up=True)
    graph_rewrite(ctags, PatternMatcher([]), name="View Consumes")

  # add stages
  tsink = graph_rewrite(tsink.substitute(realize), remove_all_tags, name="untag")

  # simple rangeify
  tsink = graph_rewrite(tsink, pm_range_creation+pm_range_migration, ctx=itertools.count(0), bottom_up=True, name="simple rangeify")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  # ***** MERGING AND SPLITTING (should be totally optional) *****


  # ***** MERGING AND SPLITTING (should be totally optional) *****

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Merged Rangeify")

  next_buffer_num = itertools.count(1000)
  tsink = graph_rewrite(tsink, symbolic+pm_remove_stage, ctx=next_buffer_num, bottom_up=True, name="remove stage")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")
  tsink = graph_rewrite(tsink, pm_no_indexing_calls, name="remove indexing from call args")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    # validate the kernel graph
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph, enter_calls=False)
  return tsink
