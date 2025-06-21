# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search

from tinygrad.opt.kernel import Kernel, Opt
from tinygrad.opt.heuristic import hand_coded_optimizations
from tinygrad.uop.ops import UOp
from tinygrad.helpers import NOOPT, BEAM, getenv
from tinygrad.renderer import Renderer

# TODO: this probably shouldn't return the Opt list
def get_optimized_ast(ast:UOp, renderer:Renderer) -> tuple[UOp, list[Opt]]:
  k = Kernel(ast, opts=renderer)
  if not NOOPT:
    if not k.apply_tensor_cores(getenv("TC", 1)): k.apply_opts(hand_coded_optimizations(k))
    if BEAM >= 1:
      from tinygrad.opt.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer)
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  return k.get_optimized_ast(), k.applied_opts
