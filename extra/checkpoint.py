from tinygrad import Tensor
from tinygrad.tensor import Function
from typing import Type
from typing import Callable
from tinygrad.tensor import _METADATA

class Checkpoint(Function):
  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x, metadata=_METADATA.get())

    ret = Tensor.__new__(Tensor)
    fn = kwargs.get('fn')

    ctx.fn = fn
    ctx.x = x[0]

    Tensor.no_grad = True
    ret = fn(x[0])
    Tensor.no_grad = False

    ret.requires_grad, ret.grad = ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

  def backward(self, grad_output):
    detached_input = Tensor(self.x.lazydata, device=self.x.device, requires_grad=True)
    detached_input._ctx = None
    self.fn(detached_input).backward(Tensor(grad_output, device=self.x.device, requires_grad=False))
    return detached_input.grad.lazydata;

def checkpoint(x:Tensor, fn:Callable[[Tensor], Tensor]) -> Tensor:
  return Checkpoint.apply(x, fn=fn)