from typing import TypeVar, Generic, Callable
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def __call__(self, *args, **kwargs) -> ReturnType:
    # realize all inputs to the JIT
    input_state_dict = get_state_dict((args, kwargs))
    Tensor.realize(*input_state_dict.values())

    print(f"JIT input {len(input_state_dict)}")

    # capture the schedules that are run
    ret = self.fxn(*args, **kwargs)

    # this gets all tensors referenced in the output
    output_state_dict = get_state_dict(ret)
    Tensor.realize(*output_state_dict.values())

    print(f"JIT output {len(output_state_dict)}")

    # copy output tensors
    return ret

  # we are going to capture at the
