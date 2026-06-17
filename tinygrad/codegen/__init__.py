from typing import cast
from dataclasses import replace
import itertools
import functools
from tinygrad.helpers import DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, DEBUG, VIZ, IMAGE, NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC
from tinygrad.helpers import ALLOW_TF32, TracingKey, Context, panic, all_same, flatten
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, Ops, UPat, track_rewrites, KernelInfo, ProgramInfo, GroupOp
from tinygrad.uop.ops import ParamArg, AxisType, _align_left, _broadcast_shape, identity_element
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load, pm_clean_up_group_sink, pm_remove_invalid
from tinygrad.uop.decompositions import get_late_rewrite_patterns, get_transcendental_patterns, pm_dtype_decomps
from tinygrad.codegen.late.expander import expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize_buf_and_index, devectorize_alu, pm_reduce, \
  ReduceContext, correct_load_store, pm_render, pm_add_loads, pm_make_images
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen, pm_mops, pm_syntactic_sugar, pm_store_ranges
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite

pm_index_is_shrink = PatternMatcher([
  # rewrite non-image INDEX to SHRINK
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))).cast(name="x"), lambda buf,idx,x:
    UOp(Ops.SHRINK, dtype=x.dtype.base, src=(buf, idx, UOp.const(dtypes.int, x.dtype.count))) \
      if isinstance(buf.dtype, PtrDType) and x.dtype.count > 1 else None),
  # rewrite GEP to INDEX
  (UPat(Ops.GEP, name="x"), lambda x: x.replace(op=Ops.INDEX, src=x.src+(UOp.const(dtypes.int, x.arg if len(x.arg) > 1 else x.arg[0]),), arg=None)),
])

pm_remove_vec_dtypes = PatternMatcher([
  # rewrite PARAM to non pointer
  (UPat((Ops.PARAM, Ops.BUFFER, Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), lambda buf:
   buf.replace(dtype=buf.dtype.base, src=(UOp.const(dtypes.int, buf.ptrdtype.size),)) \
    if isinstance(buf.dtype, PtrDType) and not isinstance(buf.dtype, ImageDType) else None),
  # remove all vec dtypes
  (UPat(GroupOp.All-{Ops.PARAM, Ops.BUFFER, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}, name="x"),
   lambda x: x.replace(dtype=x.dtype.base.scalar().base)),
  # replace DEFINE_LOCAL/DEFINE_REG with BUFFER
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="x"), lambda x:
   x.replace(op=Ops.BUFFER, arg=ParamArg(x.arg, addrspace=AddrSpace.LOCAL if x.op == Ops.DEFINE_LOCAL else AddrSpace.REG))),
  # replace DEFINE_VAR with PARAM
  (UPat(Ops.DEFINE_VAR, name="x"), lambda ctx,x:
   x.replace(op=Ops.PARAM, src=(UOp(Ops.STACK),), arg=ParamArg(slot=-1, name=x.arg[0], vmin_vmax=x.arg[1:], addrspace=None))),
])+pm_clean_up_group_sink

def maybe_load(u:UOp): return u.load() if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG) else u
pm_move_regs = PatternMatcher([
  # BITCAST?
  (UPat(GroupOp.Elementwise, name="x"), lambda x: x.replace(src=tuple([maybe_load(u) for u in x.src]))),
  (UPat(Ops.STORE, name="x"), lambda x: x.replace(src=(x.src[0], maybe_load(x.src[1]))+x.src[2:])),
])

pm_lower_weakints = PatternMatcher([
  (UPat(GroupOp.All, dtype=dtypes.weakint, name="x"), lambda x: x.replace(dtype=dtypes.int)),
])

def build_range_map(ctx, sink:UOp):
  for x in sink.toposort():
    if x.op is Ops.RANGE and x.arg[1] in {AxisType.UNROLL, AxisType.UPCAST}:
      ctx[x.arg[0]] = len(ctx)

def fix_reduce(ctx, r:UOp):
  range_to_axis = {u:ctx[u.arg[0]] for u in r.ended_ranges if u.arg[0] in ctx if u.arg[1] == AxisType.UNROLL}
  return r.replace(src=tuple([u for u in r.src if u not in range_to_axis]), arg=(r.arg[0], r.arg[1]+tuple(range_to_axis.values())))

expander2 = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), build_range_map),
  (UPat(Ops.REDUCE, name="r"), fix_reduce),
  (UPat(Ops.RANGE, name="r"),
   lambda ctx, r: UOp.const(r.dtype, tuple(range(r.vmax+1))) \
    .reshape(tuple([r.vmax+1 if i == ctx[r.arg[0]] else 1 for i in range(len(ctx))])) if r.arg[0] in ctx else None),
])+pm_flatten_range

def broadcast_binary(x:UOp):
  shapes = [u.shape for u in x.src]
  if all_same(shapes): return None
  shaped_aligned = _align_left(*shapes)
  broadcasted = _broadcast_shape(*shapes)
  src_reshaped = [u.reshape(shp).expand(broadcasted) for u,shp in zip(x.src, shaped_aligned)]
  return x.replace(src=tuple(src_reshaped))

unbroadcast = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), broadcast_binary),
])

def do_devectorize(b:UOp):
  if b.shape == (): return None
  # broadcasting needs to be already unpacked
  if not all_same([x.shape for x in b.src]): return None
  src = []
  for idx in itertools.product(*[range(x) for x in b.shape]):
    idx_c = [UOp.const(dtypes.weakint, i) for i in idx]
    src.append(b.replace(src=tuple([x.index(*idx_c) for x in b.src])))
  return UOp.vectorize(*src).reshape(b.shape)

devectorizer2 = pm_mops+PatternMatcher([
  # unpack broadcasting
  (UPat(GroupOp.Elementwise|{Ops.LOAD, Ops.STORE}, name="b"), do_devectorize),
  # INDEX into STACK is src
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="a"), UPat.cvar("i"))), lambda a,i: a.src[i.arg]),
  # stacked INDEX is many INDEX
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.STACK, name="s"))),
   lambda b,s: UOp.vectorize(*[b.index(u) for u in s.src])),
  # INDEX into RESHAPE moves the RESHAPE
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.BUFFER), name="b"), UPat(Ops.RESHAPE, name="s"))),
   lambda b,s: b.index(s.src[0]).reshape(s.shape)),
  # RESHAPE a void is removed (hack for AFTER)
  (UPat(Ops.RESHAPE, dtype=dtypes.void, name="x"), lambda x: x.src[0]),
  # reshape of a single element shaped value to scalar is an index
  (UPat(Ops.RESHAPE, name="x"), lambda x: x.src[0].index(UOp.const(dtypes.weakint, 0)) if x.marg == () and x.src[0].shape == (1,) else None),
])

def reduce_ranges_to_acc(ctx:ReduceContext, r:UOp):
  acc = UOp.placeholder_like(r, ctx.acc_num, AddrSpace.REG)
  ctx.acc_num += 1
  topo = r.src[0].toposort()
  ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
  input_ranges = tuple(x for x in topo if x.op is Ops.RANGE and x not in r.src[1:] and x not in ended_ranges)
  acc_init = acc.after(*input_ranges).store(identity_element(r.arg[0], r.dtype.scalar()))
  acc_initted = acc.after(acc_init, *r.src[1:])
  inp = r.src[0].reduce(arg=r.arg) if r.arg[1] else r.src[0]
  acc_out = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:])
  return acc.after(acc_out)

def expand_horizontal_reduce(r:UOp):
  axes = r.arg[1]
  vals = [r.src[0].shrink(tuple((idx[axes.index(i)], idx[axes.index(i)]+1) if i in axes else None for i in range(r.src[0].ndim)))
          for idx in itertools.product(*[range(r.src[0].max_shape[a]) for a in axes])]
  return functools.reduce(lambda x,y: x.alu(r.arg[0], y), vals)

pm_reduce_local = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat(), UPat()), allow_any_len=True, name="r"), reduce_ranges_to_acc),
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), expand_horizontal_reduce),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x:
   x.replace(op=Ops.PARAM, src=(UOp(Ops.STACK),), arg=ParamArg(slot=-1, name=x.arg[0], vmin_vmax=x.arg[1:], addrspace=None))),
])+pm_clean_up_group_sink

def do_number_param(ctx:list[int], x:UOp):
  if x.arg.slot != -1: return None
  ctx[0] += 1
  return x.replace(arg=replace(x.arg, slot=ctx[0]-1))

pm_number_params = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), do_number_param),
])

def full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if VIZ: graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))
  if SPEC: type_verify(ast, spec_tensor)
  sink = ast

  # preprocess. we need to simplify these
  sink = graph_rewrite(ast, pm_mops+pm_syntactic_sugar+pm_store_ranges, ctx=itertools.count(1000), name="early movement ops", bottom_up=True)

  # this is new style
  sink = graph_rewrite(sink, pm_index_is_shrink, name="index is shrink")
  sink = graph_rewrite(sink, pm_remove_vec_dtypes, name="transform to new style")

  # first we optimize
  if optimize:
    # do postrange optimization, BEAM or hand_coded_optimizations
    sink = apply_opts(sink, ren, beam=ast.arg.beam)

  # do expander
  sink = graph_rewrite(sink, expander2, ctx={}, name="expander", bottom_up=True)

  # add locals (STAGE -> BUFFER)
  sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen, ctx=itertools.count(0), name="add local buffers")

  # rewrite reduce after optimizations
  sink = graph_rewrite(sink, pm_reduce_local, ctx=ReduceContext(), name="remove_reduce")

  # add gpu dims
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # add loads
  sink = graph_rewrite(sink, pm_move_regs, name="move to registers", walk=True)

  # symbolic (note: this does POW decomp)
  sink = graph_rewrite(sink, sym, name="post index symbolic")

  # ***** make it rendererable (within spec, tighten) *****

  # decompositions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))
  pm_transcendental = symbolic_simple+get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="*** decompositions")
  sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren), name="decomp dtypes")
  sink = graph_rewrite(sink, pm_transcendental, name="transcendental")
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="decompositions more")

  # split ends
  sink = graph_rewrite(sink, pm_split_ends, name="split ends")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  # ***** this is where it gets large *****

  # unbroadcast
  sink = graph_rewrite(sink, unbroadcast, name="*** unbroadcast")

  # devectorizer
  sink = graph_rewrite(sink, symbolic_simple+devectorizer2, name="devectorizer")

  # ***** make it rendererable (outside spec, transform) *****

  # final symbolic
  sink = graph_rewrite(sink, sym, name="post devectorizer sym")

  # move gates from unrenderable INVALID where
  sink = graph_rewrite(sink, pm_move_gates_from_index, name="move gates from index")

  # put registers in slots
  num_params = len([x for x in sink.toposort() if x.op is Ops.PARAM and x.arg.slot != -1])
  name_to_slot = {x:x.replace(arg=replace(x.arg, slot=num_params+i))
                  for i,x in enumerate(sorted([x for x in sink.toposort() if x.op is Ops.PARAM and x.arg.slot == -1]))}
  sink = sink.substitute(name_to_slot, name="put variables in slots")

  # remove all weakints
  sink = graph_rewrite(sink, pm_lower_weakints, name="lower weakints", bottom_up=True)

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output AST")
  if SPEC: type_verify(sink, spec_program)

  # return the rewritten sink
  return sink

def old_full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if VIZ: graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))
  if SPEC: type_verify(ast, spec_tensor)

  # preprocess
  sink = graph_rewrite(ast, pm_mops+pm_syntactic_sugar+pm_store_ranges, ctx=itertools.count(1000), name="early movement ops", bottom_up=True)

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
    sink = apply_opts(sink, ren, beam=ast.arg.beam)

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+pm_move_where_on_load, name="postopt symbolic")

  # expand
  sink = graph_rewrite(sink, sym+pm_pre_expander+pm_group_for_reduce+expander, name="expander")

  # add locals
  sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen, ctx=itertools.count(0), name="add local buffers")

  # ** devectorizer (full_graph_rewrite) **
  # remove reduce
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")

  # add gpu dims (late). this works after devectorize, but it's faster here
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # **** optimizations are done, now we lower to actual code ****

  # add loads and remove invalids
  sink = graph_rewrite(sink, pm_add_loads+pm_remove_invalid, name="** add loads (code)")

  # create image buffers
  if IMAGE and ren.target.device in {"QCOM", "CL", "PYTHON", "NULL"}:
    sink = graph_rewrite(sink, pm_make_images, name="create image buffers", bottom_up=True, ctx=ren.target.arch)

  # devectorize
  sink = graph_rewrite(sink, sym+devectorize_alu+devectorize_buf_and_index+load_store_folding+correct_load_store+load_store_indexing,
                       ctx=ren, name="devectorize")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing+gep_pushing, name="lower all index dtypes")
  sink = graph_rewrite(sink, symbolic, name="post index symbolic")

  # optional pre matcher
  if ren.pre_matcher is not None: sink = graph_rewrite(sink, ren.pre_matcher, name="pre_matcher")

  # decompositions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))
  pm_transcendental = symbolic_simple+get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="decompositions")
  sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren), name="decomp dtypes")
  sink = graph_rewrite(sink, pm_transcendental, name="transcendental")

  # GEP/STACK stuff
  sink = graph_rewrite(sink, pm_render, name="pm_render gep/stack")

  # this is new style
  sink = graph_rewrite(sink, pm_index_is_shrink, name="index is shrink")
  sink = graph_rewrite(sink, pm_remove_vec_dtypes, name="transform to new style")

  # move gates from unrenderable INVALID where
  sink = graph_rewrite(sink, pm_move_gates_from_index, name="move gates from index")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_decomp+extra_matcher+pm_split_ends
  sink = graph_rewrite(sink, pm_final_rewrite, ctx=ren, name="final rewrite")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  # put unnumbered DEFINE_VAR in slots
  num_params = len([x for x in sink.toposort() if x.op is Ops.PARAM and x.arg.slot != -1])
  sink = graph_rewrite(sink, pm_number_params, ctx=[num_params], name="number params with -1", walk=True)

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output AST")
  if SPEC: type_verify(sink, spec_program)

  # return the rewritten sink
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
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu, ctx)) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def do_linearize(ctx:Renderer, prg:UOp, sink:UOp) -> UOp:
  if DEBUG >= 3 and sink.arg.applied_opts: print(f"{sink.arg.function_name:<25} opts: {sink.arg.applied_opts}")
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
  # isa renderers need to allocate registers
  if isinstance(ctx, ISARenderer):
    if ctx.pre_regalloc_matcher is not None: lst = line_rewrite(lst, ctx.pre_regalloc_matcher, PreRegAllocContext())
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
  return prg.replace(src=prg.src[:3]+(UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

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
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE)), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, src=UPat(Ops.INS), name="lin")), name="prg"), do_assemble),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE, name="source")), name="prg"), do_compile),
])

@track_rewrites(name=lambda ast,renderer,ret,**kwargs: TracingKey(ret.src[0].arg.name,(ret.src[0].arg.function_name, ast), ret=renderer), replay=True)
@Context(ALLOW_DEVICE_USAGE=0)
def do_to_program(ast:UOp, renderer:Renderer) -> UOp:
  """
  Transform an AST into a compiled PROGRAM. May trigger BEAM search.

  Args:
    ast: The Ops.SINK/Ops.PROGRAM rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.PROGRAM with SINK/DEVICE/LINEAR/SOURCE/BINARY.
  """
  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to to_program"
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    prog_info = ProgramInfo.from_sink(full_sink)
    # instruction selection
    if isinstance(renderer, ISARenderer):
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre instruction selection", bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=IselContext(full_sink), name="instruction selection", bottom_up=True)
    prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=renderer.target.device)), arg=prog_info)
  else: raise RuntimeError(f"can't call to_program on {ast.op}")
  if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0]))
  prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")
  if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")
  return prg

to_program_cache: dict[tuple, UOp] = {}
def to_program(ast:UOp, renderer:Renderer) -> UOp:
  config = (NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32)
  key = (ast.key, type(renderer), renderer.target, *[x.value for x in config])
  if (prg:=to_program_cache.get(key)) is None: to_program_cache[key] = prg = do_to_program(ast, renderer)
  return prg
