from typing import cast
from dataclasses import replace
import itertools
from tinygrad.helpers import DISABLE_FAST_IDIV, DEVECTORIZE, TRANSCENDENTAL, SPEC, DEBUG, VIZ, IMAGE, TracingKey, Context
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, Ops, UPat, track_rewrites, KernelInfo, pyrender
from tinygrad.uop.spec import type_verify, program_spec, kernel_spec
from tinygrad.renderer import Renderer, ProgramSpec, Estimates
from tinygrad.dtype import dtypes
from tinygrad.helpers import panic
from tinygrad.codegen.opt import Opt

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load
from tinygrad.uop.decompositions import get_late_rewrite_patterns, get_transcendental_patterns, pm_dtype_decomps, needs_dtype_decomps
from tinygrad.codegen.late.expander import expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize, pm_reduce, \
  ReduceContext, correct_load_store, pm_render, pm_add_loads, pm_make_images
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen, pm_mops, pm_syntactic_sugar, pm_store_ranges
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.renderer.amd.elf import do_assemble_amd

pm_early_movement_ops = pm_mops + pm_syntactic_sugar + pm_store_ranges
pm_split_and_flatten_ranges = pm_split_ranges + pm_flatten_range
pm_initial_symbolic = sym + pm_flatten_range
pm_simplify_ranges_full = pm_flatten_range + pm_simplify_ranges
pm_postopt_symbolic = sym + pm_move_where_on_load
pm_expander_full = sym + pm_pre_expander + pm_group_for_reduce + expander
pm_add_local_buffers = pm_add_buffers_local + rangeify_codegen
pm_remove_reduce = pm_reduce + gep_pushing
pm_lower_all_index_dtypes = pm_lower_index_dtype + load_store_indexing + gep_pushing

def full_rewrite_to_sink(sink:UOp, ren:Renderer|None=None, optimize:bool=True) -> UOp:
  if ren is None: ren = Renderer()

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(sink))
  if SPEC: type_verify(sink, kernel_spec)

  # preprocess
  sink = graph_rewrite(sink, pm_early_movement_ops, ctx=itertools.count(1000), name="early movement ops", bottom_up=True)

  # first we optimize
  if optimize:
    # collapse loads reduce (indexing by a tensor)
    sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

    # split ranges
    sink = graph_rewrite(sink, pm_split_and_flatten_ranges, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, pm_initial_symbolic, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_simplify_ranges_full, ctx={}, name="simplify ranges")

    # do postrange optimization, BEAM or hand_coded_optimizations
    sink = apply_opts(sink, ren)

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, pm_postopt_symbolic, name="postopt symbolic")

  # expand
  sink = graph_rewrite(sink, pm_expander_full, name="expander")

  # add locals
  sink = graph_rewrite(sink, pm_add_local_buffers, ctx=itertools.count(0), name="add local buffers")

  # ** devectorizer (full_graph_rewrite) **
  # remove reduce
  sink = graph_rewrite(sink, pm_remove_reduce, ctx=ReduceContext(), name="remove_reduce")

  # **** optimizations are done, now we lower to actual code ****

  # add loads
  sink = graph_rewrite(sink, pm_add_loads, name="** add loads (code)")

  # NULL only needs a balanced linear UOp stream for ProgramSpec metadata and estimates.
  # The renderer/codegen-only tail below is unnecessary once loads are materialized.
  if ren.device == "NULL":
    sink = graph_rewrite(sink, pm_split_ends, name="null split ends")
    return sink

  # add gpu dims (late). this works after devectorize, but it's faster here.
  # NULL doesn't execute kernels, and its runtime ignores global/local sizes entirely.
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # create image buffers
  if IMAGE and ren.device in {"QCOM", "CL", "PYTHON"}: sink = graph_rewrite(sink, pm_make_images, name="create image buffers", bottom_up=True)

  # devectorize (TODO: does this need opts?)
  if DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
  elif DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
  else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
  if DEVECTORIZE >= 0: sink = graph_rewrite(sink, pm_devectorize, ctx=ren, name="devectorize")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_all_index_dtypes, ctx=ren.device, name="lower all index dtypes")
  sink = graph_rewrite(sink, symbolic, name="post index symbolic")

  # optional pre matcher
  if ren.pre_matcher is not None: sink = graph_rewrite(sink, ren.pre_matcher, name="pre_matcher")

  # decompositions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, ren.device, bool(DISABLE_FAST_IDIV))
  pm_transcendental = symbolic_simple+get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren.device, name="decompositions")
  if needs_dtype_decomps(sink, ren.device, getattr(ren, "arch", "")):
    sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren.device, getattr(ren, "arch", "")), name="decomp dtypes")
  if sink.op_in_backward_slice_with_self(Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT):
    sink = graph_rewrite(sink, pm_transcendental, ctx=ren.device, name="transcendental")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_decomp+pm_render+extra_matcher+pm_split_ends
  sink = graph_rewrite(sink, pm_final_rewrite, ctx=ren.device, name="final rewrite")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  # return the rewritten sink
  return sink

# inject IF/ENDIF. only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError, "if not allowed in graph")),
  # gated INDEX becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat(name="gate", dtype=dtypes.bool))).or_casted(), UPat())),
   lambda u, gate: (u, [mif:=UOp(Ops.IF, src=(gate, u.src[0])), u, UOp(Ops.ENDIF, src=(mif,))]))
])

# requires lst be toposorted. like graph rewrite, but for lines
def line_rewrite(lst:list[UOp], pm:PatternMatcher) -> list[UOp]:
  newlst = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple([replaced[x] for x in u.src]))
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu)) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def do_linearize(prg:UOp, sink:UOp) -> UOp:
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
  if SPEC: type_verify(lst, program_spec)
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))

def do_estimates(prg:UOp, sink:UOp, lin:UOp) -> UOp|None:
  if sink.arg.estimates is not None: return None
  return prg.replace(src=(sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(lin.src, ignore_indexing=True))),)+prg.src[1:])

def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),), arg=ctx.aux(list(lin.src)) if ctx.has_aux else prg.arg)

def do_compile(ctx:Renderer, prg:UOp, source:UOp) -> UOp|None:
  lib = ctx.compiler.compile_cached(source.arg)
  return prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))

def get_null_program(prg:UOp, renderer:Renderer) -> ProgramSpec:
  if prg.op is Ops.SINK: sink = prg
  else:
    assert prg.op is Ops.PROGRAM, f"expected PROGRAM or SINK, got {prg.op}"
    sink = prg.src[0]
  assert sink.op is Ops.SINK
  return ProgramSpec.from_null_sink(sink, renderer.device, linearize(sink))

pm_to_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE)), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, src=UPat(Ops.INS), name="lin")), name="prg"), do_assemble_amd),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE, name="source")), name="prg"), do_compile),
])
pm_to_null_program = PatternMatcher(pm_to_program.patterns[:2])

@Context(ALLOW_DEVICE_USAGE=0)
@track_rewrites(name=lambda ast,renderer,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ast), ret=renderer), replay=True)
def get_program(ast:UOp, renderer:Renderer, opts:list[Opt]|None=None) -> ProgramSpec:
  """
  Transform an AST into a ProgramSpec. May trigger BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The ProgramSpec of the program.
  """

  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    # rewrite to prg
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to get_program"
    if opts is not None:
      # TODO: should this be here?
      assert ast.arg.opts_to_apply is None, "can't apply opts if there's already opts to apply"
      ast = ast.replace(arg=replace(ast.arg, opts_to_apply=tuple(opts)))
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    if renderer.device == "NULL":
      return get_null_program(full_sink, renderer)
    prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=renderer.device)))
  else:
    raise RuntimeError(f"can't call get_program on {ast.op}")

  if renderer.device == "NULL":
    return get_null_program(prg, renderer)
  else:
    prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")
  if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")

  # create the ProgramSpec
  return ProgramSpec.from_uop(prg)
