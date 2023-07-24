from typing import Callable, List, Tuple, Any, Dict, cast, Union
import functools, itertools
import numpy as np

from tinygrad.helpers import DEBUG, DType
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters, RawBuffer
from tinygrad.shape.shapetracker import ShapeTracker

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU"]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Callable, Any]] = []  # TODO: Any should be List[RawBuffer], but this fails
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], int, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_st, expected_type)
    self.outputs = {}

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {}
    input_shapetrackers: Dict[Union[int, str], ShapeTracker] = {}
    input_symbols = {}
    for k,v in itertools.chain(enumerate(args), kwargs.items()):
      if isinstance(v, Tensor):
        k = cast(Union[int, str], k)
        # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
        input_rawbuffers[k] = cast(RawBuffer, v.realize().lazydata.realized)
        input_shapetrackers[k] = v.lazydata.st
        input_symbols.update(v.lazydata.symbols)
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      for (j,i), (input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name].size == expected_st.inferred_size(input_symbols) and input_rawbuffers[input_name].dtype == expected_type, f"size or type mismatch in JIT, {input_rawbuffers[input_name]} != <{expected_st.inferred_size(input_symbols)}, {expected_type}>"
        self.jit_cache[j][1][i] = input_rawbuffers[input_name]

      # resize the output buffers
      for (j,i), (arg,st,symbol) in self.outputs.items():
        symbol.update(input_symbols)
        newsize = st.inferred_size(input_symbols)
        if arg.size < newsize: arg.resize(newsize)

      # fill the symbol buffers and update symbols for output sts
      for j, (prg,args,sts,symbols) in enumerate(self.jit_cache):
        pos = -len(input_symbols)
        for i, val in enumerate(input_symbols.values()): args[pos+i] = args[pos+i].fromCPU(np.array([val], dtype=np.int32))
        prg(args, jit=True, symbols=input_symbols)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,(prg,args,sts,symbols) in enumerate(self.jit_cache):  # pylint: disable=E1133
        for i,(a,st) in enumerate(zip(args,sts)):
          if a in input_rawbuffers.values():
            self.input_replace[(j,i)] = [(k, st, v.dtype) for k,v in input_rawbuffers.items() if v == a][0]
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      # find all outputs to resize later
      for j,(prg,args,sts,symbols) in enumerate(self.jit_cache):  # pylint: disable=E1133
        for i,(a,st,symbol) in enumerate(zip(args,sts,symbols)):
          if (j,i) not in self.input_replace.keys():
            self.outputs[(j,i)] = (a,st,symbol)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret
