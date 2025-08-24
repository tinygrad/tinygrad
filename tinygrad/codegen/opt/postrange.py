from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.renderer import Renderer

class RKernel(Kernel):
  def __init__(self, ast:UOp, opts:Renderer|None=None):
    super().__init__(ast, opts)
