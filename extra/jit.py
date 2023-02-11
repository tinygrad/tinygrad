from typing import Callable, List, Tuple
import itertools
from tinygrad.tensor import Tensor
from tinygrad.ops import DEBUG, GlobalCounters

class TinyJit:
  def __init__(self, fxn):
    self.fxn = fxn
    self.cnt = 0
    self.jit_cache : List[Tuple[Callable, List]] = []
    self.ret = None
    self.input_replace = {}

  def __call__(self, *args, **kwargs):
    input_tensors = {k:v.realize().lazydata.realized._buf for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    if self.cnt >= 2:
      for a,idx in self.input_replace.items(): a._buf = input_tensors[idx]
      for prg, args in self.jit_cache: prg(*args)
    else:
      if self.cnt == 1: GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs).realize()
      if self.cnt == 1:
        self.jit_cache = GlobalCounters.cache
        GlobalCounters.cache = None

        # get the inputs for replacement
        for prg, args in self.jit_cache:  # pylint: disable=E1133
          self.input_replace.update({a:[k for k,v in input_tensors.items() if v == a._buf][0] for a in args if a._buf in input_tensors.values()})
        assert set(self.input_replace.values()) == set(input_tensors.keys()), "some input tensors not found"
    self.cnt += 1
    return self.ret
