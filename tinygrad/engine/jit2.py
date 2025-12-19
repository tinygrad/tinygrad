from typing import TypeVar, Generic, Callable
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.engine.schedule import schedule_capturing, schedule_cache

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def add(self, input_buffers, sched_cache_key):
    self.schedule_caches.append((input_buffers, sched_cache_key))

  def __call__(self, *args, **kwargs) -> ReturnType:
    global schedule_capturing
    # realize all inputs to the JIT
    input_state_dict = get_state_dict((args, kwargs))
    Tensor.realize(*input_state_dict.values())

    # capture the schedules that are run
    self.schedule_caches = []
    schedule_capturing.append(self)
    ret = self.fxn(*args, **kwargs)
    # this gets all tensors referenced in the output
    output_state_dict = get_state_dict(ret)
    Tensor.realize(*output_state_dict.values())
    schedule_capturing = []

    print(f"JIT schedules:{len(self.schedule_caches)} inp:{len(input_state_dict)}")

    # process schedule caches to be symbolic
    for k,v in input_state_dict.items(): print(k, v.uop.base)
    for input_buffers, sched_cache_key in self.schedule_caches:
      pre_schedule, combined_sink = schedule_cache[sched_cache_key]
      for si in pre_schedule:
        print(si.bufs)

      for k,v in input_buffers.items():
        print(k.pyrender())
        print(v.pyrender())

    # copy output tensors
    return ret
