from typing import Generic, TypeVar, Callable
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

ReturnType = TypeVar('ReturnType')
class function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
    input_uops = [x[1].uop.multibase for x in input_tensors]

    # TODO: disable realize/schedule while this is running
    # run it and do surgery later
    ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"
    uret = ret.uop.substitute({x:UOp.param(i, x.dtype, x.shape, x.device) for i,x in enumerate(input_uops)})
    return Tensor(uret.call(*[x.contiguous() for x in input_uops]), device=ret.device)

