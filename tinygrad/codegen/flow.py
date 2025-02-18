# this takes the AST from a graph AST to a ProgramSpec
import itertools
from tinygrad.ops import UOp, Ops, track_rewrites, graph_rewrite, PatternMatcher, UPat, GroupOp, print_uops
from tinygrad.spec import type_verify
from tinygrad.renderer import Renderer
from tinygrad.helpers import DEBUG, NOOPT, getenv, BEAM, DEVECTORIZE, TRANSCENDENTAL, to_function_name, dedup, CAPTURE_PROCESS_REPLAY, diskcache_put
from tinygrad.renderer import ProgramSpec
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.lowerer import pm_lowerer, get_index
from tinygrad.codegen.symbolic import sym
from tinygrad.codegen.expander import migrate_indexing, expander
from tinygrad.codegen.rewriter import devectorize, float4_folding, devectorize_load_store, load_store_indexing, mulacc_unrolled
from tinygrad.codegen.rewriter import symbolic_simple, get_late_rewrite_patterns, pm_render
from tinygrad.codegen.linearize import linearize_to_uop

logkerns, logkerns_level = open(getenv("LOGKERNS", ""), "a") if getenv("LOGKERNS", "") else None, getenv("LOGKERNS_LEVEL", 1)

def optimize_kernel_ast(ast:UOp, renderer:Renderer) -> tuple[str, UOp]:
  k = Kernel(ast, opts=renderer).required_optimizations()
  if not NOOPT:
    if not k.apply_tensor_cores(getenv("TC", 1)): k.hand_coded_optimizations()
    if BEAM >= 1:
      from tinygrad.engine.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer).required_optimizations()
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  if logkerns is not None: logkerns.writelines([f"{(k.ast, k.applied_opts)}\n"])
  if DEBUG >= 5: print((k.ast, k.applied_opts)) # print here to show final applied_opts for all kernels instead of just in beam_search
  if DEBUG >= 3:
    print(k.name)
    if getenv("RAWAST"): print(k.ast)
    for i,(buf,st) in enumerate([(buf,st) for buf,st in zip(k.bufs, k.sts) if buf.op not in {Ops.CONST, Ops.VALID}]):
      print(f"{i:2d}: {str(st.shape):25s} {str(buf.src[0].dtype).replace('dtypes.',''):20s}", st.real_strides())
    print(k.applied_opts)
  return k.name, k.get_optimized_ast()

render_pm = PatternMatcher([(UPat(Ops.BLOCK, src=(UPat(Ops.NAME, name="name"),), name="block"),
                             lambda ctx,name,block: UOp(Ops.PROGRAM, arg=ctx.render(name.arg, block.arg.lst)))])

@track_rewrites()
def ast_to_program(ast:UOp, renderer:Renderer) -> ProgramSpec:
  if DEBUG >= 5: print(ast)

  # 1. Generate the modified_ast with the OptOps applied
  name, sink = optimize_kernel_ast(ast, renderer)

  # 2. verify AST matches the spec after applying opts
  if __debug__: type_verify(list(sink.toposort))

  # do indexing
  sink = graph_rewrite(sink, pm_lowerer, ctx=get_index(sink, renderer))

  # initial symbolic + migrate indexing (remove this)
  sink = graph_rewrite(sink, sym+migrate_indexing)

  # expand
  sink = graph_rewrite(sink, sym+expander)

  # two options for devectorize
  if DEVECTORIZE:
    # devectorize + load_store_indexing + mulacc_unrolled, mulacc_unrolled must be last because it can break loop_collapse
    sink = graph_rewrite(sink, sym+(devectorize+float4_folding if renderer.supports_float4 else devectorize)+load_store_indexing+mulacc_unrolled)
  else:
    # new devectorize only for load/store
    sink = graph_rewrite(sink, sym+devectorize_load_store+mulacc_unrolled)

  # optional pre matcher
  if renderer.pre_matcher is not None: sink = graph_rewrite(sink, renderer.pre_matcher)

  # final rules for the renderer (without sym)
  supported_ops = tuple(renderer.code_for_op.keys())
  extra_matcher = renderer.extra_matcher if renderer.extra_matcher is not None else PatternMatcher([])
  sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)+pm_render+extra_matcher)

  # linearize to an Ops.BLOCK
  sink = linearize_to_uop(sink)
  assert sink.op is Ops.BLOCK

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if DEBUG >= 5: print_uops(sink.arg.lst)
  if __debug__: type_verify(sink.arg.lst)

  # add the name
  sink = sink.replace(src=(UOp(Ops.NAME, arg=to_function_name(name)),)+sink.src)

  # render
  prg = graph_rewrite(sink, render_pm, ctx=renderer)
  assert prg.op is Ops.PROGRAM

  # group non-local bufs by the op type (LOAD or STORE) and the buffer arg. take the max access of that buffer in bytes
  # TODO: these max and min don't work on symbolic, and results are very wrong.
  mem_bytes = sum(max(x.src[0].dtype.itemsize * x.st_arg.real_size() for x in group)
    for _, group in itertools.groupby([x for x in ast.toposort if x.op in GroupOp.Buffer and x.src[0].op is Ops.DEFINE_GLOBAL],
                      key=lambda x: (x.op, x.src[0].arg)))

  # return the ProgramSpec
  return ProgramSpec(name, prg.arg, renderer.device, ast, sink.arg.lst, mem_estimate=mem_bytes,
                     global_size=[1,1,1] if renderer.has_local else None, local_size=[1,1,1] if renderer.has_local else None)
