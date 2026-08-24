from dataclasses import dataclass, field
import itertools
from tinygrad.dtype import AddrSpace, Invalid, to_dtype
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp, KernelInfo, ParamArg, shape_to_shape_arg
from tinygrad.uop.ops import graph_rewrite, AxisType, rewrite_group, identity_element, remove_all_tags, resolve
from tinygrad.helpers import all_int, VIZ, SPEC, Context, panic
from tinygrad.schedule.indexing import BufferizeOpts, apply_movement_op
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.allreduce import create_allreduce_function

def walk_mop(u:UOp):
  if u.op in GroupOp.Movement or u.op in {Ops.INDEX, Ops.UNSHARD}: return walk_mop(u.src[0])
  return u

fix_mselect_mstack = PatternMatcher([
  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),
])

# *** preparation ***

from tinygrad.helpers import all_same
from tinygrad.uop.ops import _broadcast_shape

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

# shape-changing bitcast
def expand_bitcast(bc:UOp) -> UOp|None:
  x = bc.src[0]
  if (ns:=bc.dtype.itemsize) == (os:=x.dtype.itemsize) or (isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS"))): return None
  new_uint, tmp = to_dtype(f"uint{8*ns}"), x.bitcast(to_dtype(f"uint{8*os}"))
  if ns > os:
    tmp = tmp.reshape(x.shape[:-1] + (x.shape[-1]//(rate := ns//os), rate))
    parts = [tmp.shrink((None,)*(len(tmp.shape)-1) + ((i, i+1),)).cast(new_uint)<<8*i*os for i in range(rate)]
    return parts[0].usum(*parts[1:]).squeeze(-1).bitcast(bc.dtype)
  parts = [tmp>>8*i*ns for i in range(os//ns)]
  return parts[0].stack(*parts[1:], dim=-1).flatten(-2).cast(new_uint).bitcast(bc.dtype)

pm_gather_params = PatternMatcher([ (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.append(p) if p.arg.slot >= 0 else None), ])
def resolve_function(c:UOp, allow_param_mismatch=True) -> UOp|None:
  if c.arg.precompile: return None
  params: list[UOp] = []
  graph_rewrite(c.src[0], pm_gather_params, bottom_up=True, ctx=params, name="gather params")
  params = sorted(params, key=lambda x: x.arg.slot)
  args = c.src[1:]

  # NOTE: this isn't really needed. it's okay if there's unused args in the function
  if not allow_param_mismatch:
    if [x.arg.slot for x in params] != list(range(len(params))): raise RuntimeError(f"params not in order: {[x.arg.slot for x in params]}")
    if len(params) != len(args): raise TypeError(f"expected {len(params)} args, got {len(args)}")

  dict_map = {x:args[x.arg.slot] for x in params}
  for i, (p, a) in enumerate(dict_map.items()):
    if p.axis != a.axis: raise TypeError(f"arg {i} axis mismatch: expected {p.axis}, got {a.axis}")
    if p.max_shape != a.max_shape: raise TypeError(f"arg {i} shape mismatch: expected {p.shape}, got {a.shape}")
    if p.dtype != a.dtype: raise TypeError(f"arg {i} dtype mismatch: expected {p.dtype}, got {a.dtype}")
  return c.src[0].substitute(dict_map, walk=True)

def fix_store_hazard(target:UOp, src:UOp):
  if (base:=target.base) not in src.toposort(enter_calls=False): return None
  # PERMUTE and FLIP reorder indices, SHRINK can have overlapping regions when dest is also shrunk
  unsafe = {Ops.PERMUTE, Ops.FLIP} | ({Ops.SHRINK} if target.op_in_backward_slice_with_self(Ops.SHRINK) else set())
  reaches_base: dict[UOp, bool] = {}
  for s in src.toposort(gate=lambda s: s.op is not Ops.CONTIGUOUS):
    reaches_base[s] = s is base or any(reaches_base.get(c) for c in s.src)
    if reaches_base[s] and s.op in unsafe and not (s is target and s.op is Ops.SHRINK): return target.store(src.contiguous())

pm_prepare_graph = PatternMatcher([
  # CALL inputs need buffer identity (and to be flat)
  (UPat(Ops.CALL, name="c"),
   lambda c: c.replace(src=c.src[0:1]+tuple(x.contiguous() if not x.has_buffer_identity(after_ok=True) else x for x in c.src[1:]))),
  # MSTACK inputs need buffer identity
  (UPat(Ops.MSTACK, name="c"),
   lambda c: c.replace(src=tuple(x.contiguous() if not x.has_buffer_identity(after_ok=True) else x for x in c.src))),
  # resolve FUNCTION calls (inline the body)
  (UPat(Ops.FUNCTION, name="c"), resolve_function),
  # resolve allreduce (must be bottom up)
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), create_allreduce_function),
  # resolve TUPLE+GETTUPLE
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),
  # expand broadcasts first
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
  # also expand bitcasts
  (UPat(Ops.BITCAST, name="bc"), expand_bitcast),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)),
  # remove movement ops from SINK/AFTER. TODO: should be generic
  (UPat(Ops.SINK, name="s"), lambda s: s.replace(src=tuple(walk_mop(u) for u in s.src if u.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="s"), lambda s: s.replace(src=(s.src[0],)+tuple(walk_mop(u) for u in s.src[1:] if u.op is not Ops.NOOP))),

  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if 0 in x.shape and 0 not in reduce.shape else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and 0 in x.shape else None),

  # STORE to () is reshaped to (1,)
  (UPat(Ops.STORE, name="s"), lambda s: s.src[0].reshape((1,)).store(s.src[1].reshape((1,))) if s.shape == () else None),
  # fix store hazard (dest is in used in src) by adding contiguous: TestAssign.test_post_flipped_assignment
  (UPat(Ops.STORE, src=(UPat(name="target"), UPat(name="src"))), fix_store_hazard),
])

def convert_copy_to_store(ctx, copy:UOp, existing_buf:UOp|None=None):
  input_src = copy.src[0]
  # if it's a COPY, we need to give the input buffer identity
  if not input_src.has_buffer_identity(after_ok=True) and copy.op is Ops.COPY: input_src = input_src.contiguous()
  input_src = input_src.flatten()
  if existing_buf is not None:
    # if the existing buffer is not a full buffer, we can't use it
    if not existing_buf.has_buffer_identity(after_ok=True): return None
    # if there's already a buffer, we just use it
    return existing_buf.flatten().store(input_src)
  # create the output buffer
  buf = UOp(Ops.BUFFER, src=(shape_to_shape_arg(input_src.max_shape),), arg=ParamArg(next(ctx), copy.dtype, device=copy.device))
  # reshape back to input
  return buf.after(buf.store(input_src)).reshape(copy.max_shape).shrink_to(copy.shape)

pm_copy_to_store = PatternMatcher([
  (UPat(name="existing_buf").store(UPat(Ops.COPY, name="copy")), convert_copy_to_store),
  (UPat((Ops.COPY, Ops.CONTIGUOUS), name="copy"), convert_copy_to_store),
])+fix_mselect_mstack

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

# *** main rangeify ***

debug_tag_factor = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.rtag(ctx[0][x] if x not in ctx[1] else 'REAL') if x.tag is None else None),
])

def remove_stage(ctx, x:UOp) -> UOp:
  buf = UOp.new_buffer(x.arg.device, x.max_numel(), x.dtype, num=next(ctx))
  return buf.after(buf.reshape(x.shape).index(*x.src[1:]).store(x.src[0]).end(*x.src[1:])).reshape(x.shape)

pm_remove_stage = PatternMatcher([
  (UPat(Ops.STAGE, name="x"), remove_stage),
])+fix_mselect_mstack

@rewrite_group(new_ctx=False)
def get_kernel_graph(sink:UOp) -> UOp:
  # TODO: multi should just be part of rangeify
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")

  # prepare
  tsink = graph_rewrite(tsink, pm_prepare_graph, bottom_up=True, name="prepare graph")
  tsink = graph_rewrite(tsink, pm_copy_to_store, ctx=itertools.count(0), bottom_up=True, name="convert copy to store")

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

  # TODO: merging and splitting algorithm

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  tsink = graph_rewrite(tsink, pm_remove_stage, ctx=itertools.count(0), bottom_up=True, name="remove stage")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")
  tsink = graph_rewrite(tsink, pm_no_indexing_calls, name="remove indexing from call args")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    # validate the kernel graph
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph, enter_calls=False)
  return tsink

