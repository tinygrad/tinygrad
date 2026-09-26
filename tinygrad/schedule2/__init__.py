import itertools
from tinygrad.dtype import Invalid
from tinygrad.uop.ops import UOp, rewrite_group, Ops, PatternMatcher, ParamArg, UPat, graph_rewrite, GroupOp, _broadcast_shape, AxisType
from tinygrad.uop.ops import remove_all_tags, KernelInfo, pm_drop_after
from tinygrad.helpers import pluralize, prod, all_same, panic, all_int, VIZ, Context
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.movement import mop_cleanup

# ************************** CANONICALIZE **************************

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

pm_canonicalize = mop_cleanup+PatternMatcher([
  # expand broadcasts first
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
    lambda r,a: UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)),
])

pm_replace_param = PatternMatcher([(UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx[p.arg.slot]),])
pm_apply_calls = PatternMatcher([(UPat(Ops.CALL, name="c"), lambda c: graph_rewrite(c.body, pm_replace_param, ctx=c.src[1:])),])

# ************************** PREPARE **************************

def copy_to_anon_store(x:UOp, copy:UOp):
  # copies are always cross device: pad to the max shape so the copy reads a whole buffer (SDMA can't do offset copies)
  x = x.pad_to(x.max_shape)
  buf = UOp(Ops.ALLOC, arg=ParamArg(next(UOp.unique_num), copy.dtype, prod(x.max_shape), device=copy.device)).reshape(x.max_shape)
  return buf.after(buf.store(x)).shrink_to(copy.shape)

def stage_to_anon_store(x:UOp, stg:UOp):
  # the buffer created here is inside the call and is not persisted, like the buffers created for copies
  buf = UOp(Ops.ALLOC, arg=ParamArg(next(UOp.unique_num), stg.dtype, prod(x.max_shape), device=x.device)).reshape(x.max_shape)
  view = buf.shrink_to(stg.shape)
  return view.after(view.store(x))

pm_copy_and_stage_to_store = PatternMatcher([
  # a bare COPY is an anonymous store: realize it as a STORE into a fresh call-local buffer on the copy device
  (UPat(Ops.COPY, src=(UPat.var("x"),), name="copy"), copy_to_anon_store),

  # a bare STAGE is an anonymous same-device materialization: realize it as a STORE into a fresh call-local buffer
  (UPat(Ops.STAGE, src=(UPat.var("x"),), name="stg"), stage_to_anon_store),
])

# ************************** RANGEIFY **************************

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
    ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), arg=idx.arg)
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
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), arg=idx.arg)
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

debug_tag_factor = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.rtag(ctx[0][x] if x not in ctx[1] else 'REAL') if x.tag is None else None),
])

# ***** split ****

def gather_param(ctx:dict[UOp, UOp], x:UOp):
  if x in ctx: return ctx[x]
  ret = x.param_like(len(ctx))
  ctx[x] = ret
  return ret

pm_gather = PatternMatcher([
  (UPat((Ops.AFTER, Ops.BUFFER, Ops.ALLOC, Ops.STAGE, Ops.COPY), name="x"), gather_param),
])

def split_kernel(x:UOp):
  nodes: dict[UOp, UOp] = {}
  src = graph_rewrite(UOp.sink(*x.src), pm_gather, ctx=nodes, name="gather", bottom_up=True).src
  body = x.replace(src=src)

  # rangeify (TODO: this should be in codegen?)
  if body.op == Ops.STORE:
    body = graph_rewrite(body, pm_range_creation+pm_range_migration, ctx=itertools.count(0), bottom_up=True, name="simple rangeify")
    body = body.sink(arg=KernelInfo())
  return body.call(*nodes.keys())

pm_split = PatternMatcher([
  (UPat((Ops.STAGE, Ops.STORE, Ops.COPY), name="x"), split_kernel),
])

pm_alloc_to_buffer = PatternMatcher([
  (UPat(Ops.ALLOC, name="x"), lambda x: UOp.new_buffer(x.device, x.max_numel(), x.dtype))
])

@rewrite_group(lambda _,ret: f"Schedule2 {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(sink:UOp) -> tuple[UOp, dict[str, int]]:
  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Tensor Graph")

  # canonicalize
  sink = graph_rewrite(sink, pm_apply_calls+pm_canonicalize, name="canonicalize")

  # add safe STAGEs to never duplicate compute
  # we compute the number of times a buffer is consumed. if > 1, we realize
  realize = {}
  consumes = {sink:0}
  for u in reversed(sink.toposort()):
    assert u in consumes, f"{u.op} not in consumes"
    if (u.op in GroupOp.ALU or u.op is Ops.REDUCE) and consumes[u] > 1 and u.device is not None:
      # TODO: rename to stage
      realize[u] = u.rtag(1).bufferize()
      consumes[u] = 1
    if u.op is Ops.STORE: consumes[u] = 1
    if u.op is Ops.EXPAND: consumes[u] *= u.max_numel() // u.src[0].max_numel()
    for i,s in enumerate(u.src):
      if s not in consumes: consumes[s] = 0
      if u.op is not Ops.STORE or i > 0:
        consumes[s] += consumes[u]

  if VIZ:
    with Context(TRACK_MATCH_STATS=0): ctags = graph_rewrite(sink, debug_tag_factor, ctx=(consumes, realize), bottom_up=True)
    graph_rewrite(ctags, PatternMatcher([]), name="View Consumes")

  # add stages
  sink = graph_rewrite(sink.substitute(realize), remove_all_tags, name="untag")

  # convert to stores (should happen later)
  sink = graph_rewrite(sink, pm_canonicalize+pm_copy_and_stage_to_store, name="copy and stage to store")

  # split into calls
  sink = graph_rewrite(sink, pm_split, name="split kernels", bottom_up=True)

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Rangeify")

  lin = []
  for u in sink.toposort():
    if u.op == Ops.CALL:
      lin.append(u)
  sink = UOp(Ops.LINEAR, src=tuple(lin))
  sink = graph_rewrite(sink, pm_alloc_to_buffer+pm_drop_after, name="Drop After + ALLOC")

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output")
  return sink, {}