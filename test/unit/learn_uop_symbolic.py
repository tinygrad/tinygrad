from test_uop_symbolic import Variable, render, NumNode
from tinygrad.engine.graph import print_tree

from tinygrad.codegen.uops import UOpGraph, UOp, UOps
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import DEBUG
from tinygrad.ops import BinaryOps
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer


def render(self, language = CStyleLanguage) -> str:
  # NOTE: we need STORE so the ALU op has children
  glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=(0,True))
  graph = UOpGraph([UOp(UOps.STORE, None, (glbl, UOp.const(dtypes.int, 0), self))])
  graph.linearize()
  class TestRenderer(language):
    code_for_op = {**CStyleLanguage().code_for_op, BinaryOps.IDIV: lambda a,b,dtype: f"({a}//{b})"}
  return TestRenderer().render("fn", graph)


class TempVar:
  def __init__(self, expr:str): self.expr = expr

v = UOp(UOps.DEFINE_VAR, dtypes.int, (), TempVar("a"))
def var(name:str, min = 0 ,max=1):
  return UOp(UOps.DEFINE_VAR, dtypes.int, src=(UOp.const(dtypes.int, min), UOp.const(dtypes.int, max)), arg=TempVar(name))
  # return UOp(UOps.DEFINE_VAR, dtypes.int, src=(), arg=TempVar(name))


v = var("a") + var("b")

print_tree(v)
print(v)
print(render(v))
print(render(v, LLVMRenderer))

