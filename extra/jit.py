from typing import Callable, List, Tuple
from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters

def _jit(fxn):
  cnt = 0
  jit_cache : List[Tuple[Callable, List]] = []
  ret = None
  input_replace = {}
  def hook(*args, **kwargs):
    nonlocal cnt, jit_cache, ret, input_replace
    assert len(kwargs) == 0
    input_tensors = [x.realize().lazydata.realized._buf for x in args if isinstance(x, Tensor)]
    if cnt >= 2:
      for a,idx in input_replace.items(): a._buf = input_tensors[idx]
      for prg, args in jit_cache: prg(*args)
    else:
      if cnt == 1: GlobalCounters.cache = []
      ret = fxn(*args, **kwargs).realize()
      if cnt == 1:
        jit_cache = GlobalCounters.cache
        GlobalCounters.cache = None

        # get the inputs for replacement
        for prg, args in jit_cache:  # pylint: disable=E1133
          input_replace.update({a:input_tensors.index(a._buf) for a in args if a._buf in input_tensors})
        assert set(input_replace.values()) == set(range(len(input_tensors))), "some input tensors not found"
    cnt += 1
    return ret
  return hook

def tinyjit(enable=True):
  if enable: return _jit
  else: return lambda x: x
