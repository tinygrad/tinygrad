from typing import cast
from tinygrad.helpers import QUANTIZE, DEVECTORIZE, TRANSCENDENTAL, SPEC
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, test_pyrender, Ops, UPat
from tinygrad.uop.spec import type_verify, program_spec, kernel_spec
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes
from tinygrad.helpers import panic

# import all pattern matchers here
from tinygrad.codegen.quantize import pm_quant
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load
from tinygrad.uop.decompositions import get_late_rewrite_patterns
from tinygrad.codegen.late.expander import migrate_indexing, expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize, pm_reduce, \
  ReduceContext, correct_load_store, pm_render
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize

def full_rewrite_to_sink(sink:UOp, ren:Renderer|None=None, optimize:bool=True) -> UOp:
  if ren is None: ren = Renderer()

  if SPEC: type_verify(list(sink.toposort()), kernel_spec)
  if SPEC > 1: test_pyrender(sink)

  # first we optimize
  if optimize:
    if QUANTIZE and ren.device in {"CPU", "DSP"}: sink = graph_rewrite(sink, pm_quant, name="quantize")

    # TODO: fix expander and remove this
    sink = graph_rewrite(sink, pm_add_buffers_local, name="add locals early")

    # collapse loads reduce (indexing by a tensor)
    sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

    # split ranges
    sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_simplify_ranges, name="simplify ranges")

    # do postrange optimization, BEAM or hand_coded_optimizations
    sink = apply_opts(sink, ren)

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+migrate_indexing+pm_move_where_on_load, name="postopt symbolic")

  # expand
  sink = graph_rewrite(sink, sym+pm_pre_expander+pm_group_for_reduce+expander, name="expander")

  # add locals
  sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen, name="add local buffers")

  # ** devectorizer (full_graph_rewrite) **
  # remove reduce
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")

  # add gpu dims (late). this works after devectorize, but it's faster here
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # devectorize (TODO: does this need opts?)
  if DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
  elif DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
  else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
  sink = graph_rewrite(sink, pm_devectorize, ctx=ren, name="devectorize")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing, ctx=ren.device, name="lower all index dtypes")
  sink = graph_rewrite(sink, symbolic, name="post index symbolic")

  # optional pre matcher
  if ren.pre_matcher is not None: sink = graph_rewrite(sink, ren.pre_matcher, name="pre_matcher")

  # decompositions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren.device, name="decompositions")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_decomp+pm_render+extra_matcher+pm_split_ends
  sink = graph_rewrite(sink, pm_final_rewrite, ctx=ren.device, name="final rewrite")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  # return the rewritten sink
  if SPEC > 1: test_pyrender(sink)
  return sink

# inject IF/ENDIF. only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError("if not allowed in graph"))),
  # gated INDEX becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat(name="gate", dtype=dtypes.bool))).or_casted(), UPat()),
        allow_any_len=True), lambda u, gate: (u, [mif:=UOp(Ops.IF, src=(gate, u.src[0])), u, UOp(Ops.ENDIF, src=(mif,))]))
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

def full_rewrite(sink:UOp, ren:Renderer|None=None) -> list[UOp]:
  """
  Function to transform the Kernel UOp graph into a linearized program.

  Args:
    sink: The Ops.SINK rooting the Kernel graph.
    ren: The Renderer (can change how things are processed, fix this).

  Returns:
    Linear program in UOps.
  """

  full_sink = full_rewrite_to_sink(sink, ren, optimize=sink.tag is None)
  assert len(full_sink.ranges) == 0, "all ranges must end by the sink"
  lst = line_rewrite(linearize(full_sink), pm_linearize_cleanups)
  if SPEC: type_verify(lst, program_spec)
  return lst
