from dataclasses import dataclass, field, replace
from typing import cast
import itertools
from tinygrad.dtype import dtypes, AddrSpace, Invalid, to_dtype, strong_dtype
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, KernelInfo, ParamArg, shape_to_shape_arg
from tinygrad.uop.ops import graph_rewrite, sint, AxisType, BottomUpGate, rewrite_group, identity_element
from tinygrad.uop.symbolic import symbolic, pm_fold_cast_const
from tinygrad.uop.movement import mop_cleanup
from tinygrad.helpers import prod, getenv, dedup, all_int, DEBUG, SPLIT_REDUCEOP, DEBUG_RANGEIFY, VIZ, MAX_KERNEL_BUFFERS, SPEC
from tinygrad.helpers import PCONTIG, FLOAT16, OPENPILOT_HACKS, argsort, partition, get_single_element, Context
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.opt import Opt
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, IndexingContext, apply_movement_op
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.allreduce import create_allreduce_function

# *** preparation ***

from tinygrad.helpers import all_same
from tinygrad.uop.ops import _broadcast_shape

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

pm_expand_broadcast = PatternMatcher([
  # expand broadcasts first
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
])

def convert_copy_to_store(ctx, copy:UOp, existing_buf:UOp|None=None):
  input_src = copy.src[0]
  if not input_src.has_buffer_identity(after_ok=True): input_src = input_src.contiguous()
  input_src = input_src.flatten()
  if existing_buf is not None:
    # if the existing buffer is not a full buffer, we can't use it
    if not existing_buf.has_buffer_identity(after_ok=True): return None
    # if there's already a buffer, we just use it
    return existing_buf.flatten().store(input_src)
  # create the output buffer
  buf = UOp(Ops.BUFFER, src=(shape_to_shape_arg(input_src.max_shape),), arg=ParamArg(next(ctx), copy.dtype, device=copy.device))
  # reshape back to input
  return buf.after(buf.store(input_src)).reshape(copy.shape)

def convert_contig_to_store(ctx, copy:UOp):
  input_src = copy.src[0]
  # create the output buffer
  buf = UOp(Ops.BUFFER, src=(shape_to_shape_arg(input_src.max_shape),), arg=ParamArg(next(ctx), copy.dtype, device=copy.device))
  # reshape back to input
  view = buf.shrink_to(input_src.shape)
  return view.after(view.store(input_src))

pm_copy_to_store = PatternMatcher([
  (UPat(name="existing_buf").store(UPat(Ops.COPY, name="copy")), convert_copy_to_store),
  (UPat(Ops.COPY, name="copy"), convert_copy_to_store),
  (UPat(Ops.CONTIGUOUS, name="copy"), convert_contig_to_store),
])

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
      # insert 0 for PAD with where
      # TODO: does this need simplify to ensure the Invalids are at the base?
      a = UOp.const(True)
      for s in ret.src[1:]:
        if s.op is Ops.WHERE and s.src[2].op is Ops.CONST and s.src[2].arg == Invalid: a = a & s.src[0]
      ret = a.where(ret, ret.const_like(0))
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

def walk_mop(u:UOp):
  if u.op in GroupOp.Movement or u.op is Ops.INDEX: return u.src[0]
  assert u.op == Ops.AFTER
  return u

pm_range_migration = PatternMatcher([
  # INDEX without src is nothing
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
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
  # pass index through elementwise
  (UPat(GroupOp.Elementwise, name="b").index(name="idx", allow_any_len=True),
   lambda b,idx: b.replace(src=tuple(s.index(*idx.src[1:]) for s in b.src))),
  # remove movement ops from SINK. TODO: should be generic
  (UPat(Ops.SINK, name="s"), lambda s: s.replace(src=tuple(walk_mop(u) for u in s.src))),
])

# *** split into kernels ***

@dataclass
class SplitCtx:
  call_args:list = field(default_factory=list)
  range_number:int = -1

def _split_graph(ctx:SplitCtx, u:UOp) -> UOp:
  assert len(u.shape) == 1, "rangeify needs to reduce to a single idx"
  ctx.call_args.append(u)
  return u.param_like(len(ctx.call_args)-1)

def _renumber_range(ctx:SplitCtx, u:UOp) -> UOp:
  ctx.range_number += 1
  return u.replace(arg=(ctx.range_number, u.arg[-1]))

pm_split_graph = PatternMatcher([
  (UPat((Ops.PARAM, Ops.AFTER), name="u"), _split_graph),
  (UPat(Ops.RANGE, name="u"), _renumber_range),
])

def split_store(x:UOp) -> UOp:
  ret = graph_rewrite(x, pm_split_graph, ctx:=SplitCtx(), name="split kernel", bottom_up=True, walk=True)
  return ret.sink(arg=KernelInfo()).call(*ctx.call_args)

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

# *** main rangeify ***

debug_tag_factor = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.rtag(ctx[0][x] if x not in ctx[1] else 'REAL') if x.tag is None else None),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(sink:UOp) -> UOp:
  # TODO: multi should just be part of rangeify
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")

  # prepare
  tsink = graph_rewrite(tsink, pm_expand_broadcast, bottom_up=True, name="expand broadcast")
  tsink = graph_rewrite(tsink, pm_copy_to_store, ctx=itertools.count(0), bottom_up=True, name="convert copy to store")

  # add safe STAGEs to never duplicate compute
  # we compute the number of times a buffer is consumed. if > 1, we realize
  realize = {}
  consumes = {tsink:0}
  for u in reversed(tsink.toposort()):
    assert u in consumes
    if (u.op in GroupOp.ALU or u.op is Ops.REDUCE) and consumes[u] > 1:
      realize[u] = None
      consumes[u] = 1
    if u.op is Ops.STORE: consumes[u] = 1
    if u.op is Ops.EXPAND: consumes[u] *= u.max_numel() // u.src[0].max_numel()
    for s in u.src[1:] if u.op is Ops.STORE else u.src:
      if s not in consumes: consumes[s] = 0
      consumes[s] += consumes[u]
  if VIZ:
    with Context(TRACK_MATCH_STATS=0): ctags = graph_rewrite(tsink, debug_tag_factor, ctx=(consumes, realize), bottom_up=True)
    graph_rewrite(ctags, PatternMatcher([]), name="View Consumes")

  # simple rangeify
  tsink = graph_rewrite(tsink, pm_range_creation+pm_range_migration, ctx=itertools.count(0), bottom_up=True, name="simple rangeify")

  # TODO: merging and splitting algorithm

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    # validate the kernel graph
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph, enter_calls=False)
  return tsink

