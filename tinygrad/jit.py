from typing import Callable, List, Tuple, Any, Dict, cast
import itertools
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters, DeviceBuffer

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn = fxn
    self.cnt = 0
    self.jit_cache : List[Tuple[Callable, Any]] = []  # TODO: Any should be List[DeviceBuffer], but this fails
    self.ret = None
    self.input_replace : Dict[DeviceBuffer, Any]= {}

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in ["GPU", "CLANG"]: return self.fxn(*args, **kwargs)  # only jit on the GPU
    # NOTE: this cast is needed since although we know realize will create a ".realized" DeviceBuffer, the type checker doesn't
    input_tensors = {k:cast(DeviceBuffer, v.realize().lazydata.realized)._buf for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    assert len(input_tensors) != 0, "no inputs to JIT"
    if self.cnt >= 2:
      for a,idx in self.input_replace.items(): a._buf = input_tensors[idx]
      for prg, args in self.jit_cache: prg(*args)
    elif self.cnt == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache) != 0, "didn't JIT anything!"

      # get the inputs for replacement
      for prg, args in self.jit_cache:  # pylint: disable=E1133
        self.input_replace.update({a:[k for k,v in input_tensors.items() if v == a._buf][0] for a in args if a._buf in input_tensors.values()})
      assert set(self.input_replace.values()) == set(input_tensors.keys()), "some input tensors not found"
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret
