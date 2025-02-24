#!POPCORN leaderboard grayscale
#!POPCORN gpu A100
# not a stable API, but works

import torch, functools
try:
  import tinygrad
except ImportError:
  import pip
  pip.main(['install', 'tinygrad'])

from tinygrad import Tensor, TinyJit
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import get_single_element, Context

@functools.lru_cache(None)
def generate_tinygrad_cuda(shape:tuple[int, ...], beam=0) -> CompiledRunner:
  print("generating code for", shape)
  tg_data = Tensor.zeros(*shape, device="CUDA").contiguous().realize()
  with Context(BEAM=beam):
    @TinyJit
    def f(x): return x[:, :, 0] * 0.2989 + x[:, :, 1] * 0.5870 + x[:, :, 2] * 0.1140
    for _ in range(3): f(tg_data)
    fxn = get_single_element(f.captured.jit_cache).prg
  print("generated with", fxn.p.global_size, fxn.p.local_size)
  return fxn

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
  assert data.dtype == torch.float32
  out = torch.empty((data.shape[0], data.shape[1]), dtype=data.dtype, device=data.device)
  fxn = generate_tinygrad_cuda(tuple(data.shape), beam=2)
  fxn._prg(out.data_ptr(), data.data_ptr(), global_size=fxn.p.global_size, local_size=fxn.p.local_size)
  return out

if __name__ == "__main__":
  for i in range(3):
    out = custom_kernel(inp:=torch.rand(16, 16, 3))
    torch.cuda.synchronize()
    assert torch.allclose(out, inp[:, :, 0] * 0.2989 + inp[:, :, 1] * 0.5870 + inp[:, :, 2] * 0.1140)
