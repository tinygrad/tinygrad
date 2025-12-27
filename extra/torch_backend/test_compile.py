# NOTE: we patch torch.compile instead of using register_backend because register_backend
# runs after Dynamo traces the function into FX graphs. We need to wrap the raw Python
# function before tracing to capture the full training step (fwd+bwd+optimizer) in TinyJit.
import torch
from extra.torch_backend.backend import unwrap, wrap
from tinygrad import Tensor, TinyJit, dtypes

def _tiny_compile(fn):
  model = next((v for v in fn.__globals__.values() if isinstance(v, torch.nn.Module) and list(v.parameters())), None)
  assert model, "torch.compile(backend='tiny') requires step to reference a nn.Module"
  params, loss_out = [unwrap(p) for p in model.parameters()], Tensor.zeros((), dtype=dtypes.float32)
  @TinyJit
  def _jit(samples: Tensor):
    with Tensor.train(): loss_out.assign(unwrap(fn(wrap(samples))))
    Tensor.realize(loss_out, *params)
    return wrap(loss_out)
  return lambda samples: _jit(unwrap(samples))

_orig = torch.compile
torch.compile = lambda fn=None, /, **kw: (lambda f: _tiny_compile(f)) if kw.get("backend") == "tiny" and fn is None \
  else _tiny_compile(fn) if kw.get("backend") == "tiny" else _orig(fn, **kw)

