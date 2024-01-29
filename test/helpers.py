from tinygrad.device import JITRunner
from tinygrad.nn.state import get_parameters
from tinygrad import Tensor
from tinygrad.helpers import Context

def derandomize_model(model):
  with Context(GRAPH=0):
    for p in get_parameters(model):
      p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
      p.realize()

def assert_jit_cache_len(fxn, expected_len):
  assert len(fxn.jit_cache) > 0
  if issubclass(type(fxn.jit_cache[0].prg), JITRunner):
    assert len(fxn.jit_cache) == expected_len
  else:
    assert len(fxn.jit_cache) == 1
    # until we have a better way of typing the prg in JitItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len