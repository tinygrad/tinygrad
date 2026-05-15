from tinygrad.uop.ops import UOp, graph_rewrite, PatternMatcher, GroupOp, UPat
from tinygrad.uop.spec import type_verify, spec_program
from tinygrad.dtype import dtypes
from tinygrad.helpers import SPEC
from tinygrad.renderer import Renderer
from tinygrad.codegen.late.devectorizer import pm_add_loads

pm_lower_weakint = PatternMatcher([
  (UPat(GroupOp.All, dtypes.weakint, name="x"), lambda x: x.replace(dtype=dtypes.int))
])

def minigen_to_sink(ast:UOp, ren:Renderer, optimize:bool) -> UOp:
  sink = ast

  # we need to lower weakint for the program
  # this will be simpler when we have implicit dtype
  sink = graph_rewrite(sink, pm_lower_weakint)

  # we need to add loads
  # this is really a store to DEFINE_REG, but load is simpler
  # LOAD is kind of anonymous store
  sink = graph_rewrite(sink, pm_add_loads)

  if SPEC: type_verify(sink, spec_program)
  return sink