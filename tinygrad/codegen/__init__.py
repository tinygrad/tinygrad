from tinygrad.helpers import QUANTIZE, DEVECTORIZE, TRANSCENDENTAL
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype
from tinygrad.uop.spec import type_verify
from tinygrad.renderer import Renderer

# import all pattern matchers here
from tinygrad.codegen.quantize import pm_quant
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load
from tinygrad.uop.decompositions import get_late_rewrite_patterns
from tinygrad.codegen.late.expander import migrate_indexing, expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize, pm_reduce, \
  ReduceContext, correct_load_store, pm_render
from tinygrad.codegen.opt.postrange import pm_postrange_opt
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_reduce_simplify, pm_flatten_range, pm_split_ranges
from tinygrad.schedule.rangeify import pm_add_buffers, rangeify_codegen
from tinygrad.codegen.late.control_flow import CFGContext, pm_merge_ends, pm_add_control_flow, linearize

def full_rewrite_to_sink(sink:UOp, opts:Renderer|None=None, optimize:bool=True) -> UOp:
  if opts is None: opts = Renderer()

  # first we optimize
  if optimize:
    if QUANTIZE and opts.device in {"CPU", "DSP"}: sink = graph_rewrite(sink, pm_quant, name="quantize")

    # split ranges
    sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_simplify_ranges, name="simplify ranges")
    sink = graph_rewrite(sink, pm_reduce_simplify, name="simplify reduces")
    sink = graph_rewrite(sink, pm_postrange_opt, ctx=opts, name="post optimize ast")

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+migrate_indexing+pm_move_where_on_load, name="postopt symbolic")

  # expand
  sink = graph_rewrite(sink, sym+pm_pre_expander+pm_group_for_reduce+expander, name="expander")

  # add locals
  sink = graph_rewrite(sink, pm_add_buffers+rangeify_codegen, name="add local buffers")

  # ** devectorizer (full_graph_rewrite) **
  # remove reduce
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")

  # add gpu dims (late). this works after devectorize, but it's faster here
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=opts, name="add gpudims")

  # devectorize (TODO: does this need opts?)
  if DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
  elif DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
  else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
  sink = graph_rewrite(sink, pm_devectorize, ctx=opts, name="devectorize")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing, ctx=opts.device, name="lower all index dtypes")
  sink = graph_rewrite(sink, symbolic, name="post index symbolic")

  # optional pre matcher
  if opts.pre_matcher is not None: sink = graph_rewrite(sink, opts.pre_matcher, name="pre_matcher")

  # decompositions
  supported_ops = tuple(opts.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=opts.device, name="decompositions")

  # final rules for the renderer (without sym)
  extra_matcher = opts.extra_matcher if opts.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_decomp+pm_render+extra_matcher
  sink = graph_rewrite(sink, pm_final_rewrite, ctx=opts.device, name="final rewrite")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_merge_ends, name="merge ends")
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow starts", bottom_up=True)

  # return the rewritten sink
  return sink

def full_rewrite(sink:UOp, opts:Renderer|None=None) -> list[UOp]:
  """
  Function to transform the Kernel UOp graph into a linearized program.

  Args:
    sink: The Ops.SINK rooting the Kernel graph.
    opts: The Renderer (can change how things are processed, fix this).

  Returns:
    Linear program in UOps.
  """

  lst = linearize(full_rewrite_to_sink(sink, opts, optimize=sink.tag is None))
  if __debug__: type_verify(lst)
  return lst
