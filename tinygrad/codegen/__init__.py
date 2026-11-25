from dataclasses import replace, dataclass, field
from collections import deque
import itertools, functools, time, inspect
from tinygrad.helpers import DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, CHECK_OOB, DEBUG, VIZ, IMAGE, NOOPT, EMULATED_DTYPES, USE_TC, SCACHE, BASEDIR
from tinygrad.helpers import ALLOW_TF32, DEFAULT_FLOAT, DEFAULT_INT, TC_SELECT, TC_OPT, TC_MIN_GLOBALS, TracingKey, Context, cpu_profile, prod
from tinygrad.helpers import all_same, argsort, partition, to_function_name, dedup, all_int, diskcache_get, diskcache_put, colored, NO_MEMORY_PLANNER
from tinygrad.helpers import canonicalize_device
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, MapCtx, UOp, Ops, UPat, rewrite_group, KernelInfo, ProgramInfo, GroupOp, AxisType
from tinygrad.uop.ops import UOpMetaClass, gate_kernel_sink, identity_element, _broadcast_shape, pm_flatten_linear
from tinygrad.uop.weak import pm_lower_weak, pm_commit_weak, pm_cast_const
from tinygrad.uop.uir import render_uir
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates, TinyELF
from tinygrad.renderer.x86 import ISARenderer, IselContext
from tinygrad.dtype import dtypes, AddrSpace, Invalid

# import all pattern matchers here
from tinygrad.codegen.gpudims import group_gpudims, pm_range_to_special
from tinygrad.uop.symbolic import sym, symbolic_simple, pm_move_where_on_load, pm_clean_up_group_sink, pm_remove_invalid
from tinygrad.uop.movement import mop_cleanup
from tinygrad.codegen.decomp.dtype import emulated_dtypes, decomp_dtypes
from tinygrad.codegen.decomp.op import get_late_rewrite_patterns
from tinygrad.codegen.decomp.transcendental import get_transcendental_patterns
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse, pm_reduce_unparented
from tinygrad.codegen.multi import multi_pm
from tinygrad.codegen.prepare import pm_mops, expand_bitcast, prepare_rangeify
from tinygrad.codegen.indexing import _stack_select
from tinygrad.codegen.late.linearizer import CFGContext, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.codegen.late.coalesce import indexing_simplify, memory_coalescing, pm_simplify_add_image
from tinygrad.codegen.rangeify import BufferizeOpts, get_kernel_graph
from tinygrad.codegen.memory import memory_plan_rewrite

_empty_pm = PatternMatcher([])
pm_cast_and_number = pm_cast_const+PatternMatcher([
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: x.replace(arg=replace(x.arg, slot=ctx[0] + ctx[1].setdefault(x.arg.name, len(ctx[1]))))
   if x.is_variable and x.arg.slot == -1 else None),
])

def expand_reduce(r:UOp):
  range_srcs, expanded = partition(r.src[1:], lambda u: u.op is Ops.RANGE)
  new_axes = [i for u in expanded for i,s in enumerate(u.shape) if s > 1]
  if len(new_axes) == 0: return None
  assert r.arg[1] == 0
  # permute so new_axes come to front, then reduce
  perm = tuple(new_axes) + tuple(i for i in range(len(r.src[0].shape)) if i not in new_axes)
  out_shape = tuple([1 if i in new_axes else s for i,s in enumerate(r.src[0].shape)])
  return r.src[0].permute(perm).reduce(*range_srcs, arg=(r.arg[0], len(new_axes))).reshape(out_shape)

def contract_axis(u:UOp, dims:list[int]) -> UOp:
  return u.permute([i for i in range(u.ndim) if i not in dims]+dims).flatten(-len(dims))

def unroll_axis(u:UOp, dims:list[int], sizes:list[int]) -> UOp:
  out = u.unflatten(-1, tuple(sizes))
  return out.permute(argsort([i for i in range(out.ndim) if i not in dims]+dims))

def expand_wmma(ctx:dict[tuple, int], u:UOp):
  if u.arg[3] is None: return None
  in0, in1, out0 = [[ctx[rn] for rn,_ in upcast_axes] for upcast_axes in u.arg[3]]
  wmma = u.replace(src=(contract_axis(u.src[0], in0), contract_axis(u.src[1], in1), u.src[2]), arg=(*u.arg[:3], None))
  return unroll_axis(wmma, out0, [sz for _,sz in u.arg[3][2]])

expander = PatternMatcher([
  (UPat(Ops.REDUCE, name="r"), expand_reduce),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(tuple(range(r.vmax+1)), r.dtype) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg] else 1 for i in range(len(ctx))])) if r.arg in ctx else None),
  (UPat(Ops.WMMA, name="u"), expand_wmma),
])+pm_flatten_range+mop_cleanup

def expand_broadcast(x:UOp):
  shapes = [u._shape for u in x.src]
  if any(s is None for s in shapes) or all_same(shapes): return None
  shape = _broadcast_shape(*shapes)
  return x.replace(src=tuple([u.expand(shape) for u in x.src]))

def broadcast_and_devec_wmma(b:UOp):
  if not (shape:=b.shape[:-1]): return None
  ex = tuple(u.expand(shape+(u.shape[-1],)) for u in b.src)
  return UOp.stack(*[b.replace(src=tuple(x.index(*idx) for x in ex)) for idx in itertools.product(*map(range, shape))]).reshape(b.shape)

pm_wmma_add = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
   lambda add, wmma: UOp(wmma.op, src=(wmma.src[0], wmma.src[1], wmma.src[2]+add), arg=wmma.arg)),
  # push permute/reshape to the other side of the add
  (UPat(Ops.PERMUTE, src=(UPat(Ops.WMMA, name="wmma"), UPat()), name="permute") + UPat.var("add"),
    lambda wmma,permute,add: (wmma + add.permute(argsort(permute.marg))).permute(permute.marg)),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name="wmma"), UPat()), name="reshape"), UPat()), name="permute") + UPat.var("add"),
    lambda wmma,reshape,permute,add: (wmma + add.permute(argsort(permute.marg)).reshape(wmma.shape)).reshape(reshape.shape).permute(permute.marg)),
])

def index_lanes(x:UOp) -> list[UOp]:
  # RESHAPE preserves row-major lane order and EXPAND prepends broadcast axes.
  if x.max_numel() == 0: return []
  if x.op is Ops.RESHAPE and all_int(x.src[0].shape): return index_lanes(x.src[0])
  if x.op is Ops.EXPAND: return index_lanes(x.src[0]) * int(prod(x.marg))
  if x.op is Ops.STACK: return [v for s in x.src for v in index_lanes(s)]
  if not x.shape: return [x]
  return [x.index(*idx) for idx in itertools.product(*map(range, x.shape))]

def do_devectorize(b:UOp):
  if b.shape == () or not all(x.shape == b.shape for x in b.src): return None
  src = [b.replace(src=values) for values in zip(*map(index_lanes, b.src), strict=True)]
  return UOp.stack(*src).reshape(b.shape) if b.op is not Ops.STORE else UOp.group(*src)

def do_stack_wmma(u:UOp):
  if all(x.op in (Ops.STACK, Ops.WMMA) for x in u.src): return None
  assert len(u.shape) == 1
  return u.replace(src=tuple(b if b.op is Ops.STACK else UOp.stack(*[b.index(i) for i in range(b.max_numel())]) for b in u.src))

devectorizer2 = pm_mops+PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.LOAD,Ops.STORE}, name="b"), do_devectorize),
  # INDEX without src is nothing (TODO: this should be in mop_cleanup)
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
  # unpack WMMA
  (UPat(Ops.WMMA, name="u"), do_stack_wmma),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(0) if x.marg == () and x.src[0].shape == (1,) else None),
  # a scalar index into a STACK is a read of the selected lane, a masked index reads 0. renderers only extract lanes at const indices
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="a"), UPat.var("i"))),
   lambda a,i: i.get_valid().where(_stack_select(i.get_idx(), list(a.src)), a.dtype.const(0)) if i.shape == () and a.ndim == 1 else None),
])

def fix_group_for_reduce(x:UOp):
  threads = (AxisType.WARP, AxisType.LOCAL)
  reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.axis_type in threads)
  if len(reduce_gfr) == 0: return None

  # do only the non grouped reduces early
  ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
  reduce_weak = [x.replace(arg=(AxisType.WEAK, x.axis_id[0]+100, *x.axis_id[1:])) for x in reduce_gfr]
  buf = ret.bufferize(*reduce_gfr, arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*reduce_weak)

  # do the final reduce (if/barrier are added in gpudims step)
  # NOTE: we remove all horizontal reduces here, they remain in the first reduce
  return buf.reduce(*reduce_weak, arg=(x.arg[0], 0))

def merge_reduce_ends(sink:UOp):
  # merge ENDs that share the same range and nesting context
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  # dependent ENDs are sequential, not merged
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END: range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.axis_id[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1 or any(a in b.backward_slice for a in ends for b in ends): continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(rr.axis_type, next_axis + j, *rr.axis_id[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

def lower_reduce(ctx:itertools.count, r:UOp):
  # horizontal: int, how many leading axes of inp are reduced in place, the rest of the reduce is over the ranges in src[1:]
  inp, (op, horizontal) = r.src[0], r.arg
  vals = [inp.index(*idx) for idx in itertools.product(*[range(inp.max_shape[a]) for a in range(horizontal)])] if horizontal else [inp]
  ret = functools.reduce(lambda x,y: x.alu(op, y), vals)
  if len(r.src) == 1: return ret
  acc = UOp.alloc_like(r, next(ctx), AddrSpace.REG)
  input_ranges = tuple(x for x in inp.ranges if x not in r.src[1:])
  acc_init = acc.after(*input_ranges).store(UOp.const(identity_element(op, r.dtype)))
  acc_initted = acc.after(acc_init, *r.src[1:])
  acc_out = acc_initted.store(acc_initted.alu(op, ret)).end(*r.src[1:])
  return acc.after(acc_out).load()

def add_local_buffer(ctx, x:UOp):
  upstream_locals = [u for u in x.ranges if u.axis_type in (AxisType.WARP, AxisType.LOCAL)] if x.arg.addrspace is AddrSpace.LOCAL else []
  buf = UOp.alloc(tuple(int(u.vmax+1) for u in upstream_locals)+x.max_shape, x.dtype, slot=next(ctx), addrspace=x.arg.addrspace)
  return buf.after(buf.index(*upstream_locals, *x.src[1:]).store(x.src[0]).end(*x.src[1:])).index(*upstream_locals)

pm_reduce_local = mop_cleanup+pm_wmma_add+PatternMatcher([
  (UPat(Ops.REDUCE, name="x"), fix_group_for_reduce),
  (UPat(Ops.REDUCE, name="r"), lower_reduce),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
  (UPat(Ops.STAGE, name="x"), add_local_buffer),
  # buffer and alloc are treated the same way, canonicalize to one op
  (UPat(Ops.ALLOC, name="x"), lambda ctx,x: x.replace(op=Ops.BUFFER, arg=replace(x.arg, slot=next(ctx)))),
])+pm_clean_up_group_sink+pm_mops

def in_memory(u:UOp): return u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG)
def maybe_load(u:UOp): return u.load() if in_memory(u) else u
pm_devectorize = pm_move_where_on_load+pm_flatten_range+pm_wmma_add+PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), expand_broadcast),
  (UPat(Ops.WMMA, name="b"), broadcast_and_devec_wmma),
  (UPat(GroupOp.Elementwise|{Ops.WMMA,Ops.STACK}, name="x"),
   lambda x: None if x.op is Ops.BITCAST and x.shape != x.src[0].shape else x.replace(src=tuple(map(maybe_load, x.src)))),
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(src=(x.src[0], maybe_load(x.src[1]))+x.src[2:])),
  (UPat(Ops.INDEX, name="x"), lambda x: x.replace(src=(x.src[0], *map(maybe_load, x.src[1:])))),
])+devectorizer2+symbolic_simple+indexing_simplify

# a shape-changing bitcast of memory reinterprets the pointer, of a value it is expanded here (custom kernels bypass the scheduler)
pm_early_mops = pm_mops+PatternMatcher([(UPat(Ops.BITCAST, name="bc"), lambda bc: None if in_memory(bc) else expand_bitcast(bc))])

def _is_local_store(x:UOp): return x.op is Ops.STORE and x.src[0].addrspace is AddrSpace.LOCAL

def add_raw_barrier(after:UOp):
  # loads from a LOCAL buffer that depend (via AFTER) on stores to LOCAL memory need a workgroup barrier
  if after.addrspace is not AddrSpace.LOCAL: return None
  # one toposort over all the deps
  deps = UOp.sink(*after.src[1:]).toposort(gate=lambda x: x.op is not Ops.BARRIER)
  if not any(_is_local_store(x) for x in deps): return None
  return after.src[0].after(UOp(Ops.BARRIER, src=after.src[1:]))

def add_war_barrier(end:UOp):
  # a LOCAL buffer stored and loaded in the same loop needs a barrier at the end of the loop body
  rngs = [r for r in end.ended_ranges if r.axis_type in (AxisType.WEAK, AxisType.LOOP) and r.vmax > 0]
  if not rngs or end.src[0].op is Ops.BARRIER: return None
  sl = end.src[0].backward_slice_with_self
  # only stores that are inside this loop body (not in the backward slice through AFTER chains from other loops)
  store_bufs = {x.buf_uop for x in sl if _is_local_store(x) and any(r in x.ranges for r in rngs)}
  # a load whose buffer matches a local store's buffer is necessarily a local load
  if not any(x.op is Ops.LOAD and x.src[0].buf_uop in store_bufs for x in sl): return None
  return end.replace(src=(UOp(Ops.BARRIER, src=(end.src[0],)),)+end.src[1:])

pm_implicit_barriers = PatternMatcher([
  (UPat(Ops.AFTER, name="after"), add_raw_barrier),
  (UPat((Ops.END, Ops.BACKEDGE), name="end"), add_war_barrier),
])

# here we create the alt value for load to be 0s and remove the where Invalid
def _move_where_load(gate, l, a): return l.replace(src=(l.src[0], a.cast(l.dtype), l.src[2]))
def _ungate(x, gate): return x.replace(src=tuple(s.src[1] if s.op is Ops.WHERE and s.src[0] is gate and s.src[2].is_invalid else s for s in x.src))
_gated_mop = UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat(), UPat(arg=Invalid))), name="mop", allow_any_len=True)
pm_ungate_mem = PatternMatcher([
  (_gated_mop.load(name="l"), lambda mop,gate,l: _ungate(mop, gate).load(l.vconst_like(0), gate)),
  (_gated_mop.store(UPat.var("data")), lambda mop,gate,data: _ungate(mop, gate).store(data, gate)),
])
# Where after gated load becomes alt value
pm_move_where_load = PatternMatcher([
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l"), UPat.var("a")), _move_where_load),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l")), _move_where_load),
])

def full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if DEBUG >= 5: print(render_uir(ast))
  if SPEC: type_verify(ast, spec_tensor)

  # resolve UNSHARDs (multi-device UNSHARDs are already resolved by the scheduler; this handles in-kernel shards, e.g. fragments)
  sink = graph_rewrite(ast, multi_pm, name="multi_pm") if ast.op_in_backward_slice_with_self(Ops.UNSHARD) else ast

  # preprocess
  sink = graph_rewrite(sink, pm_early_mops, name="early movement ops", bottom_up=True)

  # first we optimize
  if optimize:
    # collapse loads reduce (indexing by a tensor)
    if sink.op_in_backward_slice_with_self(Ops.REDUCE): sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")
    # split ranges
    if sink.op_in_backward_slice_with_self(Ops.MOD): sink = graph_rewrite(sink, pm_split_ranges, init_ctx=MapCtx, name="split ranges")
    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, pm_flatten_range+sym, name="initial symbolic")
    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_flatten_range+pm_simplify_ranges, init_ctx=MapCtx, name="simplify ranges")
    # do postrange optimization, BEAM or hand_coded_optimizations
    sink = apply_opts(sink, ren, beam=ast.arg.beam)

  # ** expander (expand_rewrite) **
  topo = sink.toposort()
  if any(u.op is Ops.REDUCE or (u.op is Ops.RANGE and (u.axis_type is AxisType.UPCAST or u.vmax == 0)) for u in topo):
    # reduce_unparented: a REDUCE whose src folded to a CONST (e.g. x*0) has no parented ranges, collapse it before the expander
    sink = graph_rewrite(sink, sym+pm_move_where_on_load+pm_flatten_range+pm_reduce_unparented, name="postopt symbolic")
    topo = sink.toposort()
    # expand
    if upcast_map := {x.arg:i for i,x in enumerate(dedup(x for x in topo if x.op is Ops.RANGE and x.axis_type is AxisType.UPCAST))}:
      sink = graph_rewrite(sink, expander, ctx=upcast_map, name="expander")
      topo = sink.toposort()
  # remove reduces and add local buffers
  if any(u.op in {Ops.REDUCE, Ops.STAGE, Ops.ALLOC} for u in topo) or sum(u.op is Ops.END for u in topo) > 1:
    slots = itertools.count(max([u.arg.slot+1 for u in topo if u.op in {Ops.ALLOC, Ops.BUFFER}], default=0))
    sink = graph_rewrite(sink, pm_reduce_local, ctx=slots, name="remove reduces / add locals")
    topo = sink.toposort()

  # group GPU dimensions early so their index arithmetic goes through normal lowering
  if any(u.op is Ops.RANGE and u.axis_type in {AxisType.GLOBAL, AxisType.LOCAL, AxisType.WARP} for u in topo):
    sink = group_gpudims(ren, sink)

  # **** optimizations are done, now we lower to actual code ****
  # devectorize
  sink = graph_rewrite(sink, pm_devectorize, name="devectorize")
  # some coalescing misses without this
  sink = graph_rewrite(sink, sym+pm_commit_weak, name="early symbolic")
  # do memory coalescing (late)
  csink = memory_coalescing(sink, ren)
  if IMAGE:
    # an image is read_only or write_only, so a PARAM that is both loaded and stored stays a buffer
    loaded, stored = [{u.src[0].buf_uop for u in csink.toposort() if u.op is op} for op in (Ops.LOAD, Ops.STORE)]
    csink = graph_rewrite(csink, pm_simplify_add_image, name="add images", ctx=({}, ren, loaded & stored), bottom_up=True)
  if csink is not sink:
    # NOTE: also run indexing_simplify here, while the index is still weakint and (x+y)*c -> x*c+y*c applies
    # commit widths minted in this fixpoint before lowering inspects INDEX shapes
    sink = graph_rewrite(csink, sym+indexing_simplify+pm_commit_weak, name="extra symbolic")

  # the boundary: required compute dtypes settle here; derivable const edges may stay bare
  # NOTE: we need indexing_simplify to remove the cast to long using the Invalid
  # NOTE: symbolic must NOT be composed here -- pm_data_invalid pushes the weak result CAST into a gated WHERE, remaking the weak node, and it cycles
  sink = graph_rewrite(sink, pm_lower_weak+indexing_simplify, name="lower all index dtypes")

  # **** decomps ****
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))+\
    get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  # Transcendental decompositions can introduce unsupported dtypes, e.g. uint64 in Payne-Hanek reduction.
  if (emulated:=emulated_dtypes(ren)) and (topo:=sink.toposort()) and (any(u.dtype in emulated for u in topo) or
      ((dtypes.long in emulated or dtypes.ulong in emulated) and
       any(u.op in {Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.DIV, Ops.MOD, Ops.TRUNCDIV, Ops.TRUNCMOD} for u in topo))):
    sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="late decompositions")
    sink = graph_rewrite(sink, pm_commit_weak, name="commit weak")
    sink = decomp_dtypes(sink, emulated)

  # final rules for the renderer (without sym), move gates from index, and convert hardware ranges to SPECIAL
  # pm_commit_weak precedes pm_decomp so weak consts commit before decomposition rules inspect operand dtypes
  pm_final_rewrite = pm_move_where_load+pm_commit_weak+pm_decomp+(ren.extra_matcher or _empty_pm)+pm_range_to_special+pm_remove_invalid
  sink = graph_rewrite(sink, pm_final_rewrite, bpm=pm_ungate_mem, ctx=ren, name="final rewrite")

  # commit every const still bare and put free variables in slots
  topo = sink.toposort()
  num_params = max([x.arg.slot + 1 for x in topo if x.op is Ops.PARAM and x.arg.slot != -1], default=0)
  sink = graph_rewrite(sink, pm_cast_and_number, ctx=(num_params, {}), name="cast consts / number variables", walk=True)

  # add implicit barriers (stores/loads through LOCAL memory ordered by AFTER or across loop iterations need workgroup barriers)
  if any(u.addrspace is AddrSpace.LOCAL for u in topo): sink = graph_rewrite(sink, pm_implicit_barriers, name="add implicit barriers")
  if sum(u.op in (Ops.END, Ops.BACKEDGE) for u in topo) >= 2 and (cfg:=CFGContext(sink)).edges:
    sink = graph_rewrite(sink, pm_add_control_flow, ctx=cfg, name="add control flow", bottom_up=True)

  if VIZ: graph_rewrite(sink, _empty_pm, name="View Output AST")
  if SPEC: type_verify(sink, spec_program)
  return sink

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

# inject IF/ENDIF. only needed if device doesn't support gated stores
def linearize_cleanups(lst:list[UOp]) -> list[UOp]:
  if not any(u.op in (Ops.IF, Ops.ENDIF) or (u.op is Ops.STORE and len(u.src) == 3) for u in lst): return lst
  out: list[UOp] = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    # if statements are not allowed in the graph
    if u.op in (Ops.IF, Ops.ENDIF): raise RuntimeError("if not allowed in graph")
    nu = u.replace(src=tuple(replaced.get(x, x) for x in u.src))
    # gated STORE becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
    if nu.op is Ops.STORE and len(nu.src) == 3 and nu.src[2].dtype is dtypes.bool:
      replaced[u] = st = nu.replace(src=nu.src[:2])
      out += [mif:=UOp(Ops.IF, src=(nu.src[2], nu.src[0])), st, UOp(Ops.ENDIF, src=(mif,))]
    else: out.append(replaced.setdefault(u, nu))
  return out

@dataclass
class LowerCallsCtx:
  renderer: Renderer
  names: dict[str, dict[UOp, int]] = field(default_factory=dict)

def lower_call(ctx:LowerCallsCtx, call:UOp, b:UOp) -> UOp|None:
  if b.op is Ops.PROGRAM and b.src[-1].op is Ops.BINARY: return None
  bodies = ctx.names.setdefault(name:=to_function_name((b.src[0].arg if b.op is Ops.PROGRAM else b.arg).name), {})
  i = bodies.setdefault(b, len(bodies))
  ki = KernelInfo(f"{name}_{i}" if i else name, optimized=True, kernel=False)
  if ctx.renderer.inline_prefix:
    if b.op_in_backward_slice_with_self(Ops.CALL):
      b = graph_rewrite(b, pm_lower_calls, ctx=ctx, name="lower calls", walk=True)
    if (lin:=to_program_cache.get(key:=("inline", *to_program_key(b, ctx.renderer)))) is None:
      to_program_cache[key] = lin = UOp(Ops.LINEAR, src=tuple(linearize_cleanups(list(b.src[1].src) if b.op is Ops.PROGRAM \
          else linearize(full_rewrite_to_sink(b, ctx.renderer, optimize=False)))), arg=ki.name)
    b = lin.replace(arg=ki.name)
  else: b = to_program(b.substitute({b.src[0]:b.src[0].replace(arg=ki)}) if b.op is Ops.PROGRAM else b.replace(arg=ki), ctx.renderer)
  return call.replace(src=(b,)+call.src[1:])

pm_lower_calls = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM), name="b"),), allow_any_len=True, name="call"), lower_call),
])

def pack_args(sink:UOp, params:tuple[UOp, ...]) -> UOp:
  sig = TinyELF.iter_sig(tuple((None, p.addrspace, p.dtype, ()) for p in params))
  args = UOp.param(0, dtypes.uint8, len(params) * 8, name="args")
  loads = [args.bitcast(dt).index(UOp.cconst(off // dt.itemsize, dtypes.int)).load() for off, dt in sig]
  reads = {p: v if p.addrspace is AddrSpace.ALU else UOp(Ops.REFPARAM, src=(p.src[0], v), arg=p.arg) for p, v in zip(params, loads)}
  return sink.substitute(reads, name="pack args", walk=True)

@rewrite_group(name=lambda ast,renderer,ret,**_: TracingKey((k:=ret.src[0].arg).name,(ret.key,k.function_name),ret=renderer),replay="process_replay")
@Context(ALLOW_DEVICE_USAGE=0)
def do_to_program(ast:UOp, renderer:Renderer) -> UOp:
  """
  Transform an AST into a compiled PROGRAM. May trigger BEAM search.

  Args:
    ast: The Ops.SINK/Ops.PROGRAM rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.PROGRAM with SINK/LINEAR/SOURCE/BINARY.
  """
  if VIZ: graph_rewrite(ast, _empty_pm, name="View Base AST")
  if ast.op_in_backward_slice_with_self(Ops.CALL):
    ast = graph_rewrite(ast, pm_lower_calls, ctx=LowerCallsCtx(renderer), name="lower calls", walk=True)
  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to to_program"
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=not ast.arg.optimized)
    prog_info = ProgramInfo.from_sink(full_sink, renderer.target)
    if full_sink.arg.estimates is None and (renderer.packed_args or isinstance(renderer, ISARenderer)):
      full_sink = full_sink.replace(arg=replace(full_sink.arg, estimates=Estimates.from_uops(tuple(linearize(full_sink)), ignore_indexing=True)))
    if renderer.packed_args and full_sink.arg.kernel: full_sink = pack_args(full_sink, prog_info.params)
    # instruction selection
    if isinstance(renderer, ISARenderer):
      slots = itertools.count(min([0]+[u.arg.slot for u in full_sink.toposort() if u.op is Ops.BUFFER])-1, -1)
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=slots, name="pre instruction selection", bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=IselContext(), name="instruction selection", bottom_up=True)
    prg = UOp(Ops.PROGRAM, src=(full_sink,), arg=prog_info)
  else: raise RuntimeError(f"can't call to_program on {ast.op}")
  if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0], renderer.target))
  if len(prg.src) == 1:
    if DEBUG >= 3 and (s:=prg.src[0]).arg.applied_opts: print(f"{s.arg.function_name:<25} opts: {s.arg.applied_opts}")
    lst = linearize_cleanups(lin:=linearize(prg.src[0]))
    if (s:=lst[-1]).arg.estimates is None:
      lst[-1] = s = s.replace(arg=replace(s.arg, estimates=Estimates.from_uops(tuple(lin), ignore_indexing=True)))
    # isa renderers need to allocate registers
    if isinstance(renderer, ISARenderer):
      lin_ctx = renderer.linear_ctx_type(renderer)
      lst = line_rewrite(lst, renderer.pre_regalloc_matcher, lin_ctx)
      lst = line_rewrite(lst, pm_regalloc_rewrite, LinearScanRegallocContext(lin_ctx, lst, renderer))
      lst = line_rewrite(lin_ctx.insert_linearized(lst), renderer.post_regalloc_matcher, lin_ctx)
    prg = prg.replace(src=(s, UOp(Ops.LINEAR, src=tuple(lst))))
  if len(prg.src) == 2:
    if getattr(type(renderer), "asm") is not Renderer.asm and all(x.op is Ops.CALL and x.body.op is Ops.NATIVE for x in prg.src[1].src):
      src = prg.src[1].render_uir()
      if DEBUG >= 4: print(src)
      prg = prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=renderer.asm(prg, prg.src[1]))))
    else: prg = prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=renderer.render(list(prg.src[1].src))),))
  try:
    if len(prg.src) == 3:
      if DEBUG >= 4: print(prg.src[2].arg)
      lib = renderer.compiler.compile_cached(prg.src[2].arg)
      if DEBUG >= 7: renderer.compiler.disassemble(lib)
      prg = prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))
  except Exception as e:
    if VIZ: prg = prg.replace(src=prg.src + (UOp(Ops.REWRITE_ERROR, arg=repr(e)),))
    raise
  finally:
    if VIZ: graph_rewrite(prg, _empty_pm, name="View Program")
  return prg

# config affects generated programs and cache keys; context also carries compile-only behavior to workers
to_program_config = (NOOPT, EMULATED_DTYPES, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32,
                     DEFAULT_FLOAT, DEFAULT_INT, TC_SELECT, TC_OPT, TC_MIN_GLOBALS)
to_program_context = (*to_program_config, SPEC, CHECK_OOB, DEBUG)
def to_program_key(ast:UOp, renderer:Renderer) -> tuple:
  return (ast.key, type(renderer), renderer.target, *[x.value for x in to_program_config])

to_program_cache: dict[tuple, UOp] = {}
def to_program(ast:UOp, renderer:Renderer) -> UOp:
  if (prg:=to_program_cache.get(key:=to_program_key(ast, renderer))) is None: to_program_cache[key] = prg = do_to_program(ast, renderer)
  return prg

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in GroupOp.Defines|{Ops.AFTER, Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
  return s

# a buffer state is a define or an AFTER. MSELECT/MSTACK join per-device states
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  if s.op is Ops.CONST: return []  # a constant arg, e.g. the device number of one device, reads no buffer
  assert s.op in GroupOp.Defines|{Ops.AFTER}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build dependency graph: kernels and buffer-state barriers
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, tuple[UOp, tuple[UOp, ...]]] = {}  # superseded state -> (AFTER, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      read_states = {k:[st for s in (k.src[0].src[1:] if k.op is Ops.END else k.src[1:]) for st in _states(s)] for k in kernels}
      # the prior state is the first of its source and the same-buffer states its kernels read that is not superseded yet
      prior_states = [_unwrap_src(u.src[0]), *[st for k in kernels for st in read_states[k] if st.storage_base is u.storage_base]]
      prev_state = next((st for st in prior_states if st not in writes), prior_states[0])
      # writes to one buffer are a chain of states: at most one AFTER supersedes a state
      if prev_state in writes: raise RuntimeError("sibling AFTERs detected in assign graph")
      new_kernels = tuple(k for k in kernels if all(a.storage_base is not u.storage_base for a in children.get(k, [])))
      writes[prev_state] = (u, new_kernels)
      # a buffer state is a node after its prior state, its joins and its kernels, so readers inherit its history
      in_degree.setdefault(u, 0)
      for st in ([prev_state] if prev_state.op is Ops.AFTER else []) + [*after_deps, *kernels]:
        children.setdefault(st, []).append(u)
        in_degree[u] += 1
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        reads += [(u, k, st) for st in read_states[k]]
        # RAW deps: a kernel runs after the states it reads or joins
        for st in read_states[k] + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            children.setdefault(st, []).append(k)
            in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; a kernel already on an earlier state of the same buffer is an ordering dep, not a write
    for u, k, s in reads:
      a, write_kernels = writes.get(s, (u, ()))
      if a is u: continue
      for t in write_kernels:
        if t is not k:
          children.setdefault(k, []).append(t)
          in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is not Ops.AFTER:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:])
        linearized.append(k.replace(src=(k.body, *buf_uops)))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))


def resolve_linear_call(linear_call:UOp):
  # a scheduled function is shared by all its calls, so the storage it allocates gets a fresh identity in each call
  args = linear_call.src[1:]
  subs = {u: (u.replace(arg=replace(u.arg, slot=next(UOp.unique_num))) if u.op is Ops.ALLOC else args[u.arg.slot])
          for u in linear_call.body.toposort() if u.op is Ops.ALLOC or (u.op is Ops.PARAM and u.arg.slot >= 0)}
  return linear_call.body.substitute(subs, walk=True)

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

# Body hashes cache schedules; (exported function, NO_MEMORY_PLANNER) keys cache final LINEARs.
schedule_cache: dict[bytes|tuple[UOp, int], UOp] = {}
def lower_sink_to_linear(call:UOp) -> UOp|None:
  function = call.body
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo) or not call.arg.precompile: return None
  st = time.perf_counter()
  cache_key = function.key
  # SCACHE >= 2 also persists the cache to disk
  sc_ret, disk_hit = schedule_cache.get(cache_key, None) if SCACHE else None, False
  if sc_ret is None and SCACHE >= 2: disk_hit = (sc_ret:=diskcache_get("schedule_cache", {"key": cache_key})) is not None
  if sc_ret is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
    if any(c.op is Ops.CALL and c.arg.precompile and c.body.op is Ops.SINK for c in linear.src):
      linear = graph_rewrite(linear, pm_schedule, name="schedule to linear", enter_calls=True, walk=True)
    if SCACHE: schedule_cache[cache_key] = linear
    if SCACHE >= 2: diskcache_put("schedule_cache", {"key": cache_key}, linear)
  else:
    # schedule cache hit (memory or disk)
    linear = schedule_cache[cache_key] = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    frm = next((f for f in inspect.stack() if f.filename != "<string>" and not f.filename.startswith(str(BASEDIR))
                and not f.filename.endswith("/contextlib.py")), None)
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {colored(' cache hit', 'yellow') if disk_hit else (' cache hit' if sc_ret is not None else 'CACHE MISS')} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return call.replace(src=(linear,)+call.src[1:])

pm_schedule = PatternMatcher([(UPat(Ops.CALL, name="call"), lower_sink_to_linear)])

def assert_all_same_devices(call:UOp):
  if len(devices:=dedup([canonicalize_device(x.device) for x in call.src[1:] if x.device is not None])) >= 2:
    raise RuntimeError(f"all buffers must be on the same device: {devices}")

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.is_virtual or src.is_virtual or (dst.device == src.device and not dst.on_disk()): return None
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, init_ctx=MapCtx, name="simplify ranges in copy")
  # the simplified body is kept only if it is a copy, any other kernel goes to codegen unchanged
  return pm_copy_kernel.rewrite(call.replace(src=(sink,) + call.src[1:]))

# dst[i] = src[i] for two n-element buffers
def copy_pat(n:UPat, i:UPat) -> UPat: return UPat(Ops.PARAM, src=(n,), name="dst").index(i).store(UPat(Ops.PARAM, src=(n,), name="src").index(i))

# a bulk STORE copies whole buffers, so lower only kernels that copy every i in range(n). a size 1 copy has its range folded to 0
pm_copy_kernel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat.any(copy_pat(UPat.cvar("n", arg=1), UPat.cvar(arg=0)),
    copy_pat(UPat.var("n"), UPat(Ops.RANGE, src=(UPat.var("n"),), name="r")).end(UPat(Ops.RANGE, name="r"))).sink(),),
    name="call", allow_any_len=True),
   lambda call,dst,src,n,r=None: None if dst.device == src.device and not dst.on_disk() else call.replace(src=(dst.store(src),) + call.src[1:])),
])

pm_copy_from_store = pm_copy_kernel+PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),
  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), assert_all_same_devices),
])

def contiguous_mops_to_view(ctx:dict[UOp, int]|None, c:UOp, src:UOp):
  """MOPS(BUFFER) → SHRINK when movement ops collapse to a contiguous range. a ctx collects the views, mapped to their slot"""
  if not all_int(c.shape): return None
  buf = src.base
  while buf.op is Ops.BITCAST: buf = buf.src[0].base
  if buf.op is Ops.UNSHARD:
    if isinstance(c.device, str): return None
    if (unshard := graph_rewrite(src, multi_pm, name="multi_buffer_view")).op is not Ops.UNSHARD: return None
    if (view := contiguous_mops_to_view(ctx, unshard.src[0], unshard.src[0])) is None: return None
    view = view.unshard(unshard.arg, unshard.src[1:])
  elif buf.op in {Ops.BUFFER, Ops.EXTERN} and (cv := src.contiguous_view()) is not None and cv[0].op in {Ops.BUFFER, Ops.EXTERN}:
    buf, offset = cv
    view = buf[offset:offset + src.max_numel() * src.element_size() // buf.element_size()].bitcast(src.dtype)
    if ctx is not None: ctx.setdefault(view, len(ctx))
    view = view.reshape(c.shape)
  else: return None
  return c.replace(src=(view,)+c.src[1:]) if c.op in {Ops.COPY, Ops.STORE} else view

# NOTE: scheduling rewrites belong in prepare; only storage/interface normalization belongs here.
pm_contiguous_views = PatternMatcher([
  # fold MOPS+BITCAST over BUFFER into SHRINK when movement ops collapse to contiguous range
  (UPat((Ops.COPY, Ops.STAGE), src=(UPat(GroupOp.Movement|{Ops.BITCAST}, name="src"),), allow_any_len=True, name="c"), contiguous_mops_to_view),
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, name="src"), UPat()), name="c", allow_any_len=True), contiguous_mops_to_view),
])

@rewrite_group(lambda fn,ret,**_: f"Schedule {len(fn.src)} -> {len(ret.src)}")
def create_linear(fn:UOp) -> UOp:
  """an exported function (no runtime state) to the LINEAR of calls that runs it, on its EXTERNs, view args and ALLOCs"""
  key = (fn, NO_MEMORY_PLANNER.value)
  if SCACHE and (hit:=schedule_cache.get(key)) is not None: return hit
  # a contiguous view of storage is a different buffer: it's an input of the scheduled function, the call arg is the view
  views: dict[UOp, int] = {}
  fn = graph_rewrite(fn, pm_contiguous_views, ctx=views, walk=True, name="contiguous views")
  if views: fn = fn.substitute({v:v.param_like(i) for v,i in views.items()}, walk=True, name="views to params")
  linear_call = graph_rewrite(fn.call(*views, precompile=True), pm_schedule, name="schedule to linear", enter_calls=True, walk=True)

  # resolve linear_call (storage stays unbound until link) and lower copy kernels to STORE calls
  linear = graph_rewrite(linear_call, pm_resolve_linear_call+pm_copy_from_store, name="resolve linear call")

  # plan the storage the function allocates
  linear = memory_plan_rewrite(linear)

  # the storage it allocates gets canonical slots, so the linear only depends on fn. link binds each to fresh storage
  allocs = [u for u in linear.toposort(gate=lambda u: u.op is not Ops.SINK) if u.op is Ops.ALLOC and u.addrspace is AddrSpace.GLOBAL]
  if any(b.device is None for b in allocs): raise RuntimeError("global storage needs a device, give the ALLOC one")
  ret = linear.substitute({b:b.replace(arg=replace(b.arg, slot=-1-i)) for i,b in enumerate(allocs)}, walk=True, name="canonical alloc slots")
  if SCACHE: schedule_cache[key] = ret
  return ret
