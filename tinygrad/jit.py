from typing import Callable, List, Tuple, Any, Dict, cast, Union
import functools, itertools
from tinygrad.helpers import DEBUG, DType

from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters, RawBuffer

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: Dict[str, int] = {}
    self.jit_cache: Dict[str, List[Tuple[Callable, Any]]] = {}  # TODO: Any should be List[RawBuffer], but this fails
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], int, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_size, expected_type)

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in ["GPU", "CLANG", "METAL", "CUDA"]: return self.fxn(*args, **kwargs)  # only jit on the GPU codegen
    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {cast(Union[int, str], k):cast(RawBuffer, v.realize().lazydata.realized) for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    specialised_key = f'{hash("".join([str(input_rawbuffers[i].size) for i,k in input_rawbuffers.items()]))}'
    #initiate call count and jit cache for different instances
    if specialised_key not in self.jit_cache: self.jit_cache[specialised_key] = []
    if specialised_key not in self.cnt: self.cnt[specialised_key] = 0
    return self._jit_call(specialised_key, input_rawbuffers, *args, **kwargs)

  def _jit_call(self, specialised_key, input_rawbuffers, *args, **kwargs) -> Any:
    # NOTE: this cast is needed since although we know realize will create a ".realized" DeviceBuffer, the type checker doesn't
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt[specialised_key] >= 2:
      for (j,i),(input_name, expected_size, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name].size == expected_size and input_rawbuffers[input_name].dtype == expected_type, f"size or type mismatch in JIT, {input_rawbuffers[input_name]} != <{expected_size}, {expected_type}>"
        self.jit_cache[specialised_key][j][1][i] = input_rawbuffers[input_name]
      for prg, args in self.jit_cache[specialised_key]: prg(args, jit=True)
      for (j,i) in self.input_replace.keys(): self.jit_cache[specialised_key][j][1][i] = None
    elif self.cnt[specialised_key] == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache[specialised_key] = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache[specialised_key]) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache[specialised_key])} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,(prg,args) in enumerate(self.jit_cache[specialised_key]):  # pylint: disable=E1133
        for i,a in enumerate(args):
          if a in input_rawbuffers.values():
            self.input_replace[(j,i)] = [(k, v.size, v.dtype) for k,v in input_rawbuffers.items() if v == a][0]
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      for (j,i) in self.input_replace.keys(): self.jit_cache[specialised_key][j][1][i] = None
    elif self.cnt[specialised_key] == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt[specialised_key] += 1
    return self.ret
