from .tensor import Tensor, Function, register
from functools import lru_cache

@lru_cache
def compile_wrapper(ane, dat):
  return ane.compile(dat)

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ane = ctx.ane
    ret = ane.tensor(input.shape)
    comp = compile_wrapper(ane, open("ane/ops/relu.hwx", "rb").read())
    ane.run(comp, input, ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    raise Exception("not implemented")
register('relu', ReLU, device=Tensor.ANE)

