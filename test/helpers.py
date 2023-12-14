from tinygrad.device import JITRunner
from tinygrad.ops import LoadOps
from tinygrad.nn.state import get_parameters

def derandomize_model(model):
  for p in get_parameters(model):
    if p.lazydata.op == LoadOps.CUSTOM:
      p.lazydata.op = LoadOps.EMPTY
      p.lazydata.arg = None
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