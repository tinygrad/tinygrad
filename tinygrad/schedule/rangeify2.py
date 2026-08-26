from dataclasses import dataclass, field
import itertools
from tinygrad.dtype import AddrSpace, Invalid, strong_dtype
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp, KernelInfo
from tinygrad.uop.ops import graph_rewrite, AxisType, rewrite_group, remove_all_tags, resolve, shape_to_shape_arg
from tinygrad.helpers import all_int, prod, VIZ, SPEC, Context, panic
from tinygrad.schedule.indexing import BufferizeOpts, apply_movement_op
from tinygrad.schedule.prepare import has_buffer_view
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen.simplify import pm_reduce_simplify

# *** preparation ***

fix_mselect_mstack = PatternMatcher([
  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),
])

from tinygrad.helpers import all_same
from tinygrad.uop.ops import _broadcast_shape

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

pm_lil_prepare_graph = PatternMatcher([
  # expand broadcasts first
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
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
  call_args:list[UOp] = field(default_factory=list)
  buffers:dict[UOp, int] = field(default_factory=dict)
  range_number:int = -1
  addrspace:AddrSpace = AddrSpace.GLOBAL

def _split_graph(ctx:SplitCtx, u:UOp) -> UOp|None:
  if u.tag is not None: return None
  if u.addrspace != ctx.addrspace: return None
  if u.addrspace == AddrSpace.ALU: return u.param_like(-1).rtag().reshape(u.shape)

  # A kernel takes each underlying buffer state once. In particular, AFTER and its buffer must use the same slot, with AFTER kept as the call
  # argument so its dependencies are preserved.
  key = u.buf_uop if u.op is Ops.AFTER else u
  if (slot:=ctx.buffers.get(key)) is None:
    slot = ctx.buffers[key] = len(ctx.call_args)
    ctx.call_args.append(u)
  elif u.op is Ops.AFTER:
    ctx.call_args[slot] = u

  # Parameters describe the max-sized physical allocation. A symbolic logical shape is a view of that allocation, not part of the PARAM itself.
  param = u.param_like(slot).rtag().replace(src=(shape_to_shape_arg((u.max_numel(),)),))
  return param.reshape(u.max_shape).shrink_to(u.shape)

def _renumber_range(ctx:SplitCtx, u:UOp) -> UOp|None:
  if u.tag is not None: return None
  ctx.range_number += 1
  return u.replace(arg=(ctx.range_number, u.arg[-1])).rtag()

pm_split_graph = pm_range_migration+PatternMatcher([
  (UPat((Ops.PARAM, Ops.AFTER, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK), name="u"), _split_graph),
  (UPat(Ops.RANGE, name="u"), _renumber_range),
])

def _is_fully_invalid_state(x:UOp) -> bool:
  while x.op in GroupOp.Movement|{Ops.INDEX}: x = x.src[0]
  if x.op is not Ops.AFTER or len(x.src) != 2: return False
  end = x.src[1]
  st = end.src[0] if end.op is Ops.END else end
  if st.op is not Ops.STORE or not st.src[1].base.is_invalid: return False
  if end.op is not Ops.END: return st.src[0].max_numel() == x.max_numel()
  covered = 1
  for r in end.src[1:]:
    if r.op is not Ops.RANGE or r.src[0].op is not Ops.CONST or not isinstance(r.src[0].val, int): return False
    covered *= r.src[0].val
  return covered == x.max_numel()

def split_store(x:UOp) -> UOp|None:
  st = x.src[0] if x.op is Ops.END else x
  if st.op is Ops.STORE and st.src[0].is_variable: return None
  if st.op is Ops.STORE and st.src[0] is st.src[1]: return UOp(Ops.NOOP)
  # A directly-invalid value makes this store a no-op. An AFTER carrying an
  # invalid partial store is still a valid buffer state: uncovered elements
  # must continue to read from the previous state.
  if st.op is Ops.STORE and (st.src[1].base.is_invalid or _is_fully_invalid_state(st.src[1])): return UOp(Ops.NOOP)
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
  (UPat(Ops.AFTER, name="u"), lambda u: u.replace(src=tuple(s for s in u.src if s.op is not Ops.NOOP))),
])

# *** main rangeify ***

debug_tag_factor = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.rtag(ctx[0][x] if x not in ctx[1] else 'REAL') if x.tag is None else None),
])

def remove_stage(ctx, x:UOp) -> UOp:
  dtype = strong_dtype(x.dtype)
  buf = UOp.new_buffer(x.arg.device, x.max_numel(), dtype, num=next(ctx))
  val = x.src[0] if x.src[0].dtype == dtype else x.src[0].cast(dtype)
  return buf.after(buf.reshape(x.shape).index(*x.src[1:]).store(val).end(*x.src[1:])).reshape(x.shape)

pm_remove_stage = PatternMatcher([
  (UPat(Ops.STAGE, name="x"), remove_stage),
])+fix_mselect_mstack

def remove_selected_stage(ctx:set[UOp], stage:UOp, idx:UOp) -> UOp|None:
  return stage.src[0] if stage in ctx and stage.src[1:] == idx.src[1:] else None

pm_remove_selected_stage = PatternMatcher([
  (UPat(Ops.STAGE, name="stage").index(name="idx", allow_any_len=True), remove_selected_stage),
])

def inline_stage_index(stage:UOp, idx:UOp) -> UOp:
  replacements, cache = dict(zip(stage.src[1:], idx.src[1:])), {}
  def replace(x:UOp) -> UOp:
    if x in replacements: return replacements[x]
    if x.has_buffer_identity(after_ok=True) or x.op is Ops.STAGE: return x
    if x not in cache: cache[x] = x.replace(src=tuple(replace(s) for s in x.src))
    return cache[x]
  return replace(stage.src[0])

MAX_RECOMPUTE = 8
MAX_SCALAR_RECOMPUTE = 64

def recompute_cost(x:UOp, seen:set[UOp]|None=None) -> int|None:
  if seen is None: seen = set()
  if x in seen or x.op is Ops.STAGE or x.has_buffer_identity(after_ok=True): return 0
  seen.add(x)
  if x.op is Ops.REDUCE: return None
  costs = [recompute_cost(s, seen) for s in (x.src[:1] if x.op is Ops.INDEX else x.src)]
  return None if any(c is None for c in costs) else sum(c for c in costs if c is not None) + (x.op in GroupOp.Elementwise)

def materialize_call_args(c:UOp) -> UOp:
  srcs:list[UOp] = []
  for x in c.src[1:]:
    device = x.device or c.device
    srcs.append(x if x.op is Ops.STAGE or x.is_bound_var or has_buffer_view(x) or x.shape == () or device is None
                else x.bufferize(arg=BufferizeOpts(device=device)))
  return c.replace(src=(c.src[0], *srcs))

def materialize_mselect(m:UOp, x:UOp) -> UOp|None:
  if x.device is None or x.op is Ops.STAGE or (x.op not in GroupOp.ALU and x.has_buffer_identity(after_ok=True)): return None
  return m.replace(src=(x.bufferize(arg=BufferizeOpts(device=x.device)),))

pm_materialize_call_args = PatternMatcher([
  (UPat(Ops.CALL, name="c"), materialize_call_args),
  (UPat(Ops.MSELECT, src=(UPat(name="x"),), name="m"), materialize_mselect),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(sink:UOp) -> UOp:
  tsink = graph_rewrite(sink, pm_lil_prepare_graph, bottom_up=True, name="prepare graph")
  # Calls can only receive buffer states. Materialize lazy constants/computations instead of silently unwrapping them to a nonexistent base buffer.
  tsink = graph_rewrite(tsink, pm_materialize_call_args, name="materialize call args")

  read_cache:dict[UOp, set[UOp]] = {}
  def read_buffers(x:UOp) -> set[UOp]:
    if x not in read_cache:
      read_cache[x] = {x.buf_uop} if x.has_buffer_identity(after_ok=True) else set().union(*(read_buffers(s) for s in x.src))
    return read_cache[x]
  stores = [u for u in tsink.toposort() if u.op is Ops.STORE and not u.src[0].is_variable]
  dests = [u.src[0].buf_uop for u in stores]
  reads = [read_buffers(u.src[1]) for u in stores]
  force_stage:set[UOp] = set()
  param_writes = [i for i,dest in enumerate(dests) if dest.op is Ops.PARAM]
  if len(param_writes) == 2:
    i, j = param_writes
    if stores[j].src[1].op_in_backward_slice_with_self(Ops.REDUCE) and dests[i] is not dests[j] and \
       dests[i] in reads[j] and dests[j] in reads[i]: force_stage.add(stores[j].src[1])

  # add safe STAGEs to never duplicate compute
  # we compute the number of times a buffer is consumed. if > 1, we realize
  realize = {}
  consumes = {tsink:0}
  for u in reversed(tsink.toposort()):
    assert u in consumes, f"{u.op} not in consumes"
    if u in force_stage:
      realize[u] = u.rtag(1).bufferize(arg=BufferizeOpts(device=u.device, removable=False))
      consumes[u] = 1
    elif (u.op in GroupOp.ALU or u.op is Ops.REDUCE) and consumes[u] > 1 and u.device is not None:
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

  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify, name="pre-fusion reduce simplify")

  # remove stage boundaries when this doesn't duplicate expensive compute or nest reductions
  while 1:
    staged:dict[UOp, list[UOp]] = {}
    children:dict[UOp, list[UOp]] = {}
    for u in tsink.toposort():
      for s in u.src: children.setdefault(s, []).append(u)
      if u.op is Ops.INDEX and u.src[0].op is Ops.STAGE: staged.setdefault(u.src[0], []).append(u)

    boundary_cache:dict[UOp, set[UOp]] = {}
    def boundaries(x:UOp) -> set[UOp]:
      if x not in boundary_cache:
        boundary_cache[x] = set().union(*({c} if c.op in {Ops.STAGE, Ops.STORE, Ops.CALL} else boundaries(c)
                                          for c in children.get(x, [])))
      return boundary_cache[x]

    reduce_cache:dict[UOp, bool] = {}
    def feeds_reduce(x:UOp) -> bool:
      if x not in reduce_cache:
        reduce_cache[x] = any(c.op is Ops.REDUCE or (c.op not in {Ops.STAGE, Ops.STORE, Ops.CALL} and feeds_reduce(c))
                              for c in children.get(x, []))
      return reduce_cache[x]

    def boundary_work(boundary:UOp) -> int|None:
      value = boundary.src[1] if boundary.op is Ops.STORE else boundary.src[0]
      if any(r.src[0].op is not Ops.CONST or not isinstance(r.src[0].val, int) for r in value.ranges): return None
      return prod(r.src[0].val for r in value.ranges)

    def recompute_work(idxs:list[UOp]) -> int|None:
      works = [boundary_work(next(iter(bs))) for idx in idxs if len(bs:=boundaries(idx)) == 1]
      return sum(x for x in works if x is not None) if len(works) == len(idxs) and all(x is not None for x in works) else None

    replacements:dict[UOp, UOp] = {}
    range_replacements:dict[UOp, UOp] = {}
    selected_stages:set[UOp] = set()
    for stage,idxs in staged.items():
      if not stage.arg.removable or children.get(stage) != idxs: continue
      inlinable = stage.src[0].op in GroupOp.ALU or stage.src[0].op is Ops.REDUCE
      cost = recompute_cost(stage.src[0])

      # passthrough stages don't duplicate compute when indexing is unchanged
      if not inlinable:
        if len(idxs) == 1 and stage.src[1:] == idxs[0].src[1:]:
          replacements[idxs[0]] = stage.src[0]
          break
        continue

      # duplicate cheap elementwise stages; reductions require identical indexing into one output boundary
      if len(idxs) > 1:
        work = recompute_work(idxs)
        if cost is not None and cost <= MAX_RECOMPUTE and work is not None and work <= stage.max_numel() * len(idxs):
          replacements.update((idx, inline_stage_index(stage, idx)) for idx in idxs)
        elif cost is None and all(idx.src[1:] == idxs[0].src[1:] for idx in idxs) and not any(feeds_reduce(idx) for idx in idxs):
          stage_boundaries = set().union(*(boundaries(idx) for idx in idxs))
          if len(stage_boundaries) == 1 and boundary_work(next(iter(stage_boundaries))) == stage.max_numel():
            range_replacements.update(zip(stage.src[1:], idxs[0].src[1:]))
            selected_stages.add(stage)
        if replacements or range_replacements: break
        continue

      idx = idxs[0]
      stage_boundaries = boundaries(idx)
      work = boundary_work(next(iter(stage_boundaries))) if len(stage_boundaries) == 1 else None
      scalar = stage.max_numel() == 1 and cost is not None and cost <= MAX_SCALAR_RECOMPUTE and all(r.op is Ops.CONST for r in idx.src[1:])
      small = cost is not None and cost <= MAX_SCALAR_RECOMPUTE and stage.max_numel() <= 8 and idx.max_numel() <= 8
      if cost is not None:
        if stage.src[0].op_in_backward_slice_with_self(Ops.THREEFRY) or scalar or small or \
           (cost <= MAX_RECOMPUTE and work is not None and work <= stage.max_numel()):
          replacements[idx] = inline_stage_index(stage, idx)
      elif len(stage_boundaries) == 1 and not feeds_reduce(idx) and work == stage.max_numel():
        if all(r.op is Ops.RANGE for r in idx.src[1:]):
          range_replacements.update(zip(stage.src[1:], idx.src[1:]))
          selected_stages.add(stage)
        else: replacements[idx] = inline_stage_index(stage, idx)
      if replacements or range_replacements: break

    if not replacements and not range_replacements: break
    tsink = tsink.substitute(replacements).substitute(range_replacements)
    if selected_stages:
      selected_stages = {stage.substitute(range_replacements) for stage in selected_stages}
      tsink = graph_rewrite(tsink, pm_remove_selected_stage, ctx=selected_stages, name="remove selected stage")

  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify, name="reduce simplify")

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

