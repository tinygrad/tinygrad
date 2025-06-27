# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search

from tinygrad.opt.kernel import Kernel
from tinygrad.opt.heuristic import hand_coded_optimizations
from tinygrad.uop.ops import UOp, KernelInfo, graph_rewrite, PatternMatcher, UPat, Ops
from tinygrad.helpers import NOOPT, BEAM, USE_TC, getenv
from tinygrad.renderer import Renderer

def get_optimized_ast(ast:UOp, renderer:Renderer) -> UOp:
  """
  Optimize an AST based on heuristics or BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.SINK rooted AST transformed to apply the opts and with a KernelInfo in the arg.
  """

  k = Kernel(ast, opts=renderer)
  if ast.arg is not None and ast.arg.opts_to_apply is not None: k.apply_opts(ast.arg.opts_to_apply)
  elif not NOOPT:
    if not k.apply_tensor_cores(USE_TC.value): k.apply_opts(hand_coded_optimizations(k))
    if BEAM >= 1:
      from tinygrad.opt.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer)
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  return k.get_optimized_ast()

def remove_ones_reduce(r:UOp):
  ones_count = 0
  new_axis_arg = []
  for i,x in enumerate(r.src[0].shape):
    if x == 1: ones_count += 1
    elif i in r.arg[1]: new_axis_arg.append(i-ones_count)
  # TODO: why do i need this tuple equality check?
  return r.replace(arg=(r.arg[0], tuple(new_axis_arg))) if tuple(new_axis_arg) != r.arg[1] else None

pm_no_ones = PatternMatcher([
  (UPat(Ops.VIEW, name="v"),
   lambda v: v.replace(arg=v.arg.reshape(tuple([x for x in v.arg.shape if x != 1]))) if 1 in v.arg.shape else None),
  (UPat(Ops.REDUCE_AXIS, name="r"), remove_ones_reduce),
])

def get_optimized_ast_2(ast:UOp, renderer:Renderer) -> UOp:
  # remove all 1s from views
  ast = graph_rewrite(ast, pm_no_ones, name="remove ones", bottom_up=True)
  return ast.replace(arg=KernelInfo(name="test"))
