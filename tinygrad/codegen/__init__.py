from dataclasses import replace, dataclass
import itertools, functools
from tinygrad.helpers import DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, DEBUG, VIZ, PROFILE, IMAGE, NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC
from tinygrad.helpers import ALLOW_TF32, TracingKey, Context, panic
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, Ops, UPat, track_rewrites, KernelInfo, ProgramInfo, GroupOp
from tinygrad.uop.ops import TRACK_MATCH_STATS
from tinygrad.uop.ops import AxisType
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext
from tinygrad.dtype import dtypes, AddrSpace, Invalid

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, symbolic, pm_simplify_valid, pm_move_where_on_load, pm_clean_up_group_sink, pm_remove_invalid
from tinygrad.uop.movement import mop_cleanup
from tinygrad.codegen.decomp.dtype import pm_dtype_decomps
from tinygrad.codegen.decomp.op import get_late_rewrite_patterns, get_simplifying_rewrite_patterns
from tinygrad.codegen.decomp.transcendental import get_transcendental_patterns
from tinygrad.codegen.late.coalese import indexing_simplify
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.codegen.late.coalese import memory_coalesing, pm_simplify_add_image
from tinygrad.helpers import all_same, flatten, argsort, partition
from tinygrad.uop.ops import _align_left, _broadcast_shape, identity_element
from tinygrad.schedule.rangeify import BufferizeOpts

empty_matcher = PatternMatcher([])

def do_number_param(ctx:list[int], x:UOp):
  if x.arg.slot != -1: return None
  ctx[0] += 1
  return x.replace(arg=replace(x.arg, slot=ctx[0]-1))

pm_number_params = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), do_number_param),
])

pm_no_index = PatternMatcher([
  (UPat(GroupOp.ALU.union({Ops.CONST}), dtype=dtypes.index, name="x"), lambda x: x.replace(dtype=dtypes.int)),
  (UPat(Ops.CAST, dtype=dtypes.index, src=(UPat.var("x"),)), lambda x: x.cast(dtypes.int)),
])

def build_range_map(sink:UOp) -> dict[int, int]:
  ctx: dict[int, int] = {}
  for x in sink.toposort():
    if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}:
      ctx[x.arg[0]] = len(ctx)
  return ctx

def expand_reduce(r:UOp):
  range_srcs = []
  new_axes = []
  for u in r.src[1:]:
    if u.op == Ops.RANGE:
      range_srcs.append(u)
    else:
      for i,s in enumerate(u.shape):
        if s > 1: new_axes.append(i)
  if len(new_axes) == 0: return None
  assert r.arg[1] == 0
  # permute so new_axes come to front, then reduce
  perm = tuple(new_axes) + tuple(i for i in range(len(r.src[0].shape)) if i not in new_axes)
  out_shape = tuple([1 if i in new_axes else s for i,s in enumerate(r.src[0].shape)])
  return r.src[0].permute(perm).reduce(*range_srcs, arg=(r.arg[0], len(new_axes))).reshape(out_shape)

def contract_axis(ctx:dict[int, int], u:UOp, arg):
  permute_tail = [ctx[rn] for rn,_ in arg]
  permute_head = [i for i in range(len(u.shape)) if i not in permute_tail]
  out = u.permute(permute_head+permute_tail)
  return out.reshape(*out.shape[:len(permute_head)], -1)

def unroll_axis(ctx:dict[int, int], u:UOp, arg):
  permute_tail = [ctx[rn] for rn,_ in arg]
  out = u.reshape(*u.shape[:-1], *[nm for _,nm in arg])
  permute_head = [i for i in range(len(out.shape)) if i not in permute_tail]
  return out.permute(argsort(permute_head+permute_tail))

def expand_wmma(ctx:dict[int, int], u:UOp):
  if u.tag != 1: return None
  in0, in1, out0 = u.arg[6]
  wmma = u.replace(src=(contract_axis(ctx, u.src[0], in0), contract_axis(ctx, u.src[1], in1), u.src[2]), tag=None)
  return unroll_axis(ctx, wmma, out0)

expander2 = PatternMatcher([
  (UPat(Ops.REDUCE, name="r"), expand_reduce),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(r.dtype, tuple(range(r.vmax+1))) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
  (UPat(Ops.WMMA, name="u"), expand_wmma),
])+pm_flatten_range+mop_cleanup

def broadcast_binary(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shaped_aligned = _align_left(*shapes)
  broadcasted = _broadcast_shape(*shapes)
  src_reshaped = [u.reshape(shp).expand(broadcasted) for u,shp in zip(x.src, shaped_aligned)]
  return x.replace(src=tuple(src_reshaped))

def broadcast_and_devec_wmma(b:UOp):
  shapes = [u.shape[:-1] for u in b.src]
  if all_same(shapes): return None
  shaped_aligned = _align_left(*shapes)
  broadcasted = _broadcast_shape(*shapes)
  src_reshaped = [u.reshape(shp+(u.shape[-1],)).expand(broadcasted+(u.shape[-1],))
                  for u,shp in zip(b.src, shaped_aligned)]
  src = []
  for idx in itertools.product(*[range(i) for i in b.shape[:-1]]):
    idx_c = [UOp.const(dtypes.index, i) for i in idx]
    src.append(b.replace(src=tuple([x.index(*idx_c) for x in src_reshaped])))
  return UOp.stack(*src).reshape(b.shape)

@functools.cache
def shape_indexes(shape:tuple[int, ...]) -> tuple[tuple[UOp, ...], ...]:
  return tuple(tuple(UOp.const(dtypes.index, i) for i in idx) for idx in itertools.product(*map(range, shape)))

pm_wmma_add = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
   lambda add, wmma: UOp(wmma.op, src=(wmma.src[0], wmma.src[1], wmma.src[2]+add), arg=wmma.arg)),
  # push permute/reshape to the other side of the add
  (UPat(Ops.PERMUTE, src=(UPat(Ops.WMMA, name="wmma"),), name="permute") + UPat.var("add"),
    lambda wmma,permute,add: (wmma + add.permute(argsort(permute.arg))).permute(permute.arg)),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name="wmma"), UPat()), name="reshape"),), name="permute") + UPat.var("add"),
    lambda wmma,reshape,permute,add: (wmma + add.permute(argsort(permute.arg)).reshape(wmma.shape)).reshape(reshape.shape).permute(permute.arg)),
])

unbroadcast = pm_wmma_add+PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), broadcast_binary),
  (UPat(Ops.WMMA, name="b"), broadcast_and_devec_wmma),
])

def do_devectorize(ctx, b:UOp):
  ren = ctx.ren if isinstance(ctx, DevectorizeContext) else (ctx[-1] if isinstance(ctx, tuple) else ctx)
  if b.op in GroupOp.Elementwise and b.dtype in dtypes.floats and ren.supports_float4: return None
  if (shape:=b._shape) is None or shape == (): return None
  # broadcasting needs to be already unpacked
  if any(x._shape != shape for x in b.src): return None
  src = []
  for idx_c in shape_indexes(shape):
    new_src = tuple(index_lane(ctx, x, idx_c) if isinstance(ctx, DevectorizeContext) else
                    (x.src[idx_c[0].arg] if len(idx_c) == 1 and x.op is Ops.STACK else UOp(Ops.INDEX, x.dtype, (x,)+idx_c)) for x in b.src)
    src.append(UOp(b.op, b.dtype, new_src, b.arg, b.tag))
  return UOp(Ops.STACK, b.dtype, tuple(src)).reshape(shape) if b.op is not Ops.STORE else UOp.group(*src)

@dataclass
class DevectorizeContext:
  ren: Renderer
  lanes: dict[tuple[UOp, tuple[UOp, ...]], UOp]
  @property
  def rewrite_cache_key(self): return (type(self.ren), self.ren.target)

devectorize_caches: list[dict[tuple[UOp, tuple[UOp, ...]], UOp]] = []
class scoped_devectorize_cache:
  def __enter__(self): devectorize_caches.append({})
  def __exit__(self, *args): devectorize_caches.pop()

apply_opts_caches: list[dict[tuple[UOp, type[Renderer], object, int], UOp]] = []
class scoped_apply_opts_cache:
  def __enter__(self): apply_opts_caches.append({})
  def __exit__(self, *args): apply_opts_caches.pop()

postopt_codegen_caches: list[dict[tuple[UOp, type[Renderer], object], UOp]] = []
class scoped_postopt_codegen_cache:
  def __enter__(self): postopt_codegen_caches.append({})
  def __exit__(self, *args): postopt_codegen_caches.pop()

def index_lane(ctx:DevectorizeContext, x:UOp, idxs:tuple[UOp, ...]) -> UOp:
  key = (x, idxs)
  if (ret:=ctx.lanes.get(key)) is not None: return ret
  if x.op is Ops.STACK and idxs and idxs[0].op is Ops.CONST:
    ret = index_lane(ctx, x.src[idxs[0].arg], idxs[1:]) if len(idxs) > 1 else x.src[idxs[0].arg]
  elif x.op in GroupOp.Movement and len(idxs) == len(x.shape):
    ret = index_lane(ctx, x.src[0], apply_movement_op(x.op, x.src[0].shape, x.marg, idxs))
  elif x.op is Ops.INDEX:
    ret = index_lane(ctx, x.src[0], x.src[1:]+idxs)
  elif x.op in GroupOp.Elementwise:
    ret = UOp(x.op, x.dtype, tuple(index_lane(ctx, s, idxs) if s._shape else s for s in x.src), x.arg, x.tag)
  else: ret = UOp(Ops.INDEX, x.dtype, (x,)+idxs)
  ctx.lanes[key] = ret
  return ret

def index_elementwise(x:UOp, idx:UOp):
  indexes = idx.src[1:]
  return UOp(x.op, x.dtype, tuple(UOp(Ops.INDEX, s.dtype, (s,)+indexes) if s._shape else s for s in x.src), x.arg, x.tag)

def index_elementwise_lane(ctx:DevectorizeContext, x:UOp, idx:UOp):
  indexes = idx.src[1:]
  return UOp(x.op, x.dtype, tuple(index_lane(ctx, s, indexes) if s._shape else s for s in x.src), x.arg, x.tag)

def do_stack_wmma(u:UOp):
  if all(x.op in (Ops.STACK, Ops.WMMA) for x in u.src): return None
  assert len(u.shape) == 1
  src = []
  for b in u.src:
    if b.op != Ops.STACK:
      src.append(UOp.stack(*[b.index(UOp.const(dtypes.index, i)) for i in range(b.max_numel())]))
    else:
      src.append(b)
  return u.replace(src=tuple(src))

ew_devectorizer = PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise, name="b"), do_devectorize),
  (UPat(GroupOp.Elementwise, name="x").f(Ops.INDEX, allow_any_len=True, name="idx"), index_elementwise),
])

devectorizer2 = mop_cleanup+pm_mops+PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.LOAD,Ops.STORE}, name="b"), do_devectorize),
  (UPat(GroupOp.Elementwise, name="x").f(Ops.INDEX, allow_any_len=True, name="idx"), index_elementwise_lane),
  # INDEX without src is nothing (TODO: this should be in mop_cleanup)
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
  # unpack WMMA
  (UPat(Ops.WMMA, name="u"), do_stack_wmma),
  # stacked INDEX is many INDEX
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.STACK, name="s"))),
   lambda b,s: UOp.stack(*[b.index(u) for u in s.src])),
  # INDEX into RESHAPE moves the RESHAPE
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.RESHAPE, name="s"))),
   lambda b,s: b.index(s.src[0]).reshape(s.shape)),
  # RESHAPE a void is removed (hack for AFTER)
  (UPat(Ops.RESHAPE, dtype=dtypes.void, name="x"), lambda x: x.src[0]),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(UOp.const(dtypes.index, 0)) if x.marg == () and x.src[0].shape == (1,) else None),
  # EXPAND on scalar -> STACK
  (UPat(Ops.EXPAND, src=(UPat.var("x"), UPat()), name="out"),
   lambda x,out: UOp.stack(*([x]*out.max_numel())) if x.shape == () and out.shape == (out.max_numel(),) else None),
  # INDEX on INDEX is INDEX
  (UPat(Ops.INDEX, src=(UPat(Ops.INDEX, name="idx1", allow_any_len=True),), allow_any_len=True, name="idx2"),
   lambda idx1, idx2: idx1.src[0].index(*idx1.src[1:], *idx2.src[1:])),
])

def fix_group_for_reduce(x:UOp):
  reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.arg[1] == AxisType.GROUP_REDUCE)
  if len(reduce_gfr) == 0: return None

  # NOTE: if there's other locals here, we need them in the buffer too
  upstream_locals = [u for u in x.toposort() if u.op is Ops.RANGE and u.arg[1] == AxisType.LOCAL]

  # do only the non grouped reduces early
  ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
  reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
  buf = ret.bufferize(*upstream_locals, *reduce_gfr, arg=BufferizeOpts(reduce_gfr[0].arg[0], AddrSpace.LOCAL)).index(*upstream_locals, *reduce_loop)

  # do the final reduce (if/barrier are added in gpudims step)
  # NOTE: we remove all horizontal reduces here, they remain in the first reduce
  return buf.reduce(*reduce_loop, arg=(x.arg[0], 0))

@dataclass
class ReduceContext:
  acc_num: int = 0

def merge_reduce_ends(sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

def reduce_ranges_to_acc(ctx:ReduceContext, r:UOp):
  acc = UOp.placeholder_like(r, ctx.acc_num, AddrSpace.REG)
  ctx.acc_num += 1
  topo = r.src[0].toposort()
  ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
  input_ranges = tuple(x for x in topo if x.op is Ops.RANGE and x not in r.src[1:] and x not in ended_ranges)
  acc_init = acc.after(*input_ranges).store(identity_element(r.arg[0], r.dtype))
  acc_initted = acc.after(acc_init, *r.src[1:])
  inp = r.src[0].reduce(arg=r.arg) if r.arg[1] else r.src[0]
  acc_out = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:]).rtag("mergeable")
  return acc.after(acc_out)

def expand_horizontal_reduce(r:UOp):
  inp = r.src[0]
  vals = [inp.index(*idx) for idx in itertools.product(*[range(inp.max_shape[a]) for a in range(r.arg[1])])]
  return functools.reduce(lambda x,y: x.alu(r.arg[0], y), vals)

pm_reduce_local = pm_wmma_add+PatternMatcher([
  # fix group for reduce
  (UPat(Ops.REDUCE, name="x"), fix_group_for_reduce),
  # remove reduces
  (UPat(Ops.REDUCE, src=(UPat(), UPat()), allow_any_len=True, name="r"), reduce_ranges_to_acc),
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), expand_horizontal_reduce),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
])+pm_clean_up_group_sink

def maybe_load(u:UOp): return u.load() if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG) else u
pm_add_loads = PatternMatcher([
  # BITCAST?
  (UPat(GroupOp.Elementwise|{Ops.REDUCE,Ops.WMMA,Ops.STACK}, name="x"), lambda x: x.replace(src=tuple([maybe_load(u) for u in x.src]))),
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(src=(x.src[0], maybe_load(x.src[1]))+x.src[2:])),
])

def add_local_buffer(ctx, x:UOp):
  buf = UOp.placeholder(x.max_shape, x.dtype, slot=next(ctx), addrspace=x.arg.addrspace)
  return buf.after(buf.index(*x.src[1:]).store(x.src[0]).end(*x.src[1:]).barrier())

pm_add_local_buffers = PatternMatcher([
  (UPat(Ops.STAGE, name="x"), add_local_buffer),
])+pm_mops

def full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if VIZ: graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))
  if SPEC: type_verify(ast, spec_tensor)

  # preprocess
  sink = graph_rewrite(ast, pm_mops, name="early movement ops", bottom_up=True)

  # first we optimize
  if optimize:
    # collapse loads reduce (indexing by a tensor)
    sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

    # split ranges
    sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges")

    # do postrange optimization, BEAM or hand_coded_optimizations
    opt_key = (sink, type(ren), ren.target, ast.arg.beam)
    if apply_opts_caches and (optimized:=apply_opts_caches[-1].get(opt_key)) is not None: sink = optimized
    else:
      sink = apply_opts(sink, ren, beam=ast.arg.beam)
      if apply_opts_caches: apply_opts_caches[-1][opt_key] = sink

  postopt_key = (sink, type(ren), ren.target)
  use_postopt_cache = bool(postopt_codegen_caches) and not (VIZ or PROFILE or TRACK_MATCH_STATS)
  if use_postopt_cache and (cached_sink:=postopt_codegen_caches[-1].get(postopt_key)) is not None: return cached_sink

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+pm_move_where_on_load+pm_flatten_range, name="postopt symbolic")

  # expand
  sink = graph_rewrite(sink, expander2, ctx=build_range_map(sink), name="expander")

  # remove reduce
  sink = graph_rewrite(sink, mop_cleanup+pm_reduce_local, ctx=ReduceContext(), name="remove reduces")

  # add locals
  sink = graph_rewrite(sink, pm_add_local_buffers, ctx=itertools.count(0), name="add local buffers")

  # add gpu dims (late). this works after devectorize, but it's faster here
  if VIZ or PROFILE or TRACK_MATCH_STATS: sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")
  elif (gpu_sink:=pm_add_gpudims.rewrite(sink, ren)) is not None: sink = gpu_sink

  # **** optimizations are done, now we lower to actual code ****

  sink = graph_rewrite(sink, symbolic_simple+unbroadcast+pm_add_loads, name="*** unbroadcast / add loads")

  # devectorize
  sink = graph_rewrite(sink, symbolic_simple+devectorizer2,
                       ctx=DevectorizeContext(ren, devectorize_caches[-1] if devectorize_caches else {}), name="devectorize2")

  # some coalesing misses without this
  sink = graph_rewrite(sink, sym, name="early symbolic")

  # do memory coalesing (late)
  sink = memory_coalesing(sink, ren)
  if IMAGE: sink = graph_rewrite(sink, symbolic_simple+ew_devectorizer+pm_simplify_add_image, name="add images", ctx=({}, ren), bottom_up=True)

  has_invalid = (sink.op is Ops.CONST and sink.arg is Invalid) or any(u.op is Ops.CONST and u.arg is Invalid for u in sink.backward_slice)
  if has_invalid:
    sink = graph_rewrite(sink, pm_simplify_valid, name="simplify valid after coalescing")
    has_invalid = (sink.op is Ops.CONST and sink.arg is Invalid) or any(u.op is Ops.CONST and u.arg is Invalid for u in sink.backward_slice)

  # lower index dtype
  # NOTE: we need indexing_simplify to remove the cast to long using the Invalid
  sink = graph_rewrite(sink, pm_lower_index_dtype+indexing_simplify, name="lower all index dtypes")

  # **** decomps ****

  # final symbolic + floordiv/mod + dtype decomp
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic+get_simplifying_rewrite_patterns(supported_ops)

  # late decomps + move gates from unrenderable INVALID where
  candidate_dtypes = {*dtypes.fp8s, dtypes.bfloat16, dtypes.half, dtypes.long, dtypes.ulong}
  emulated_dtypes = set(EMULATED_DTYPES.tolist(dtypes)) | (candidate_dtypes - ren.supported_dtypes())
  needs_dtype_decomp = sink.dtype in emulated_dtypes or any(u.dtype in emulated_dtypes for u in sink.backward_slice)
  if needs_dtype_decomp:
    sink = graph_rewrite(sink, pm_decomp, name="early decompositions")
    sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren), name="decomp dtypes")
  pm_decomp = pm_decomp+\
    get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))+\
    get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="late decompositions")
  if has_invalid: sink = graph_rewrite(sink, pm_move_gates_from_index, name="move gates from index")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else empty_matcher
  pm_final_rewrite = symbolic_simple+extra_matcher+pm_split_ends+pm_no_index
  sink = graph_rewrite(sink, pm_final_rewrite+pm_remove_invalid, ctx=ren, name="final rewrite")

  # this was the linearizer
  final_topo = sink.backward_slice_with_self
  if any(x.op is Ops.RANGE for x in final_topo):
    sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink, itertools.chain(sink.backward_slice, (sink,))),
                         name="add control flow", bottom_up=True)
    final_topo = sink.backward_slice_with_self

  # put unnumbered variable PARAMs in slots
  params = [x for x in final_topo if x.op is Ops.PARAM]
  if any(x.arg.slot == -1 for x in params):
    sink = graph_rewrite(sink, pm_number_params, ctx=[sum(x.arg.slot != -1 for x in params)], name="number params with -1", walk=True)

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output AST")
  if SPEC: type_verify(sink, spec_program)

  # return the rewritten sink
  if use_postopt_cache: postopt_codegen_caches[-1][postopt_key] = sink
  return sink

# inject IF/ENDIF. only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError, "if not allowed in graph")),
  # gated STORE becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat((Ops.INDEX, Ops.SHRINK)).or_casted(), UPat(), UPat(name="gate", dtype=dtypes.bool))),
   lambda u, gate: ((st:=u.replace(src=u.src[0:2])), [mif:=UOp(Ops.IF, src=(gate, u.src[0])), st, UOp(Ops.ENDIF, src=(mif,))]))
])

# requires lst be toposorted. like graph rewrite, but for lines
def line_rewrite(lst:list[UOp], pm:PatternMatcher, ctx=None) -> list[UOp]:
  newlst = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple([replaced.get(x, x) for x in u.src]))
    ret: tuple[UOp, list[UOp]] = pm.rewrite(nu, ctx) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def do_linearize(ctx:Renderer, prg:UOp, sink:UOp) -> UOp:
  if DEBUG >= 3 and sink.arg.applied_opts: print(f"{sink.arg.function_name:<25} opts: {sink.arg.applied_opts}")
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
  # isa renderers need to allocate registers
  if isinstance(ctx, ISARenderer):
    if ctx.pre_regalloc_matcher is not None: lst = line_rewrite(lst, ctx.pre_regalloc_matcher, PreRegAllocContext())
    # register definitions (INS without srcs) move to the top so regalloc sees their live ranges span the whole program (callee saved regs)
    lst = sorted(lst, key=lambda u: u.op is not Ops.INS or bool(u.src))
    regalloc_ctx = LinearScanRegallocContext(lst, ctx)
    lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
    lst = line_rewrite(lst, ctx.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 4: print(ctx.asm_str(lst, sink.arg.function_name))
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))

def do_estimates(prg:UOp, sink:UOp, lin:UOp) -> UOp|None:
  if sink.arg.estimates is not None: return None
  return prg.replace(src=(sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(lin.src, ignore_indexing=True))),)+prg.src[1:])

def do_assemble(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = "\n".join(str(u.arg) for u in lin.src)
  if DEBUG >= 4: print(src)
  binary = ctx.asm(prg, lin)
  return prg.replace(src=prg.src[:2]+(UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  new_arg = replace(prg.arg, aux=tuple(ctx.aux(list(lin.src)))) if ctx.has_aux else prg.arg
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),), arg=new_arg)

def do_compile(ctx:Renderer, prg:UOp, source:UOp) -> UOp|None:
  if DEBUG >= 4: print(source.arg)
  lib = ctx.compiler.compile_cached(source.arg)
  if DEBUG >= 7: ctx.compiler.disassemble(lib)
  return prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))

pm_to_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"),), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR, src=UPat(Ops.INS), name="lin")), name="prg"), do_assemble),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR), UPat(Ops.SOURCE, name="source")), name="prg"), do_compile),
])

@track_rewrites(name=lambda ast,renderer,ret,**kwargs: TracingKey(ret.src[0].arg.name,(ret.src[0].arg.function_name, ast), ret=renderer), replay=True)
@Context(ALLOW_DEVICE_USAGE=0)
def do_to_program(ast:UOp, renderer:Renderer, compile_binary=True) -> UOp:
  """
  Transform an AST into a compiled PROGRAM. May trigger BEAM search.

  Args:
    ast: The Ops.SINK/Ops.PROGRAM rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.PROGRAM with SINK/LINEAR/SOURCE/BINARY.
  """
  from tinygrad.codegen.opt.gemm import cooperative_gemm_program, direct_conv_bwd_activation_program
  from tinygrad.codegen.opt.reduce import activation_var_grad_program, bn_grad_512_program, channel_reduce_program, col2im_program
  from tinygrad.codegen.opt.reduce import im2col_program, moments_512_program
  from tinygrad.codegen.opt.reduce import maxpool_backward_program, maxpool_program
  if ast.op is Ops.SINK and (prg:=direct_conv_bwd_activation_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=cooperative_gemm_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=activation_var_grad_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=moments_512_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=bn_grad_512_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=channel_reduce_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=col2im_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=im2col_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=maxpool_backward_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.SINK and (prg:=maxpool_program(ast, renderer, compile_binary)) is not None: return prg
  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to to_program"
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    # instruction selection
    if isinstance(renderer, ISARenderer):
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre instruction selection", bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=IselContext(full_sink), name="instruction selection", bottom_up=True)
    prg = UOp(Ops.PROGRAM, src=(full_sink,))
  else: raise RuntimeError(f"can't call to_program on {ast.op}")
  if VIZ:
    if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0]))
    prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")
    graph_rewrite(prg, PatternMatcher([]), name="View Program")
    return prg
  # PROGRAM lowering is a linear root-only pipeline. Driving it through graph_rewrite
  # needlessly walks the full SINK and LINEAR graphs between each stage.
  if len(prg.src) == 1: prg = do_linearize(renderer, prg, prg.src[0])
  if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0], uops=prg.src[1].src))
  if prg.src[0].arg.estimates is None and (estimated:=do_estimates(prg, prg.src[0], prg.src[1])) is not None: prg = estimated
  if len(prg.src) == 2:
    prg = do_assemble(renderer, prg, prg.src[1]) if isinstance(renderer, ISARenderer) else do_render(renderer, prg, prg.src[1])
  if compile_binary and len(prg.src) == 3 and (compiled:=do_compile(renderer, prg, prg.src[2])) is not None: prg = compiled
  return prg

to_program_cache: dict[tuple, UOp] = {}
def to_program(ast:UOp, renderer:Renderer, compile_binary=True) -> UOp:
  config = (NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32)
  # UOps are structurally interned, so identity is already a collision-free structural
  # cache key within this process and avoids recursively hashing every kernel graph.
  key = (ast, type(renderer), renderer.target, compile_binary, *[x.value for x in config])
  if (prg:=to_program_cache.get(key)) is None: to_program_cache[key] = prg = do_to_program(ast, renderer, compile_binary=compile_binary)
  return prg
