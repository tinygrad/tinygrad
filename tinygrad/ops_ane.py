from .tensor import Tensor, Function, register
from functools import lru_cache

@lru_cache
def compile_wrapper(ane, dat):
  return ane.compile(dat)

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ret = ctx.ane.tensor(input.shape)
    comp = compile_wrapper(ctx.ane, open("ane/ops/relu.hwx", "rb").read())
    ctx.ane.run(comp, input, ret)
    return ret
register('relu', ReLU, device=Tensor.ANE)

