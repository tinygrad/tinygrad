import torch
import torch._dynamo
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified

from tinygrad import Tensor, TinyJit
from extra.torch_backend.backend import unwrap, wrap


@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args:Tensor):
      torch_outs = gm(*[wrap(x) for x in args])
      torch_outs = (torch_outs,) if isinstance(torch_outs, torch.Tensor) else tuple(torch_outs)
      tiny_outs = tuple(unwrap(x) for x in torch_outs)
      for x in tiny_outs: x.realize()
      return tiny_outs
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args:torch.Tensor):
      return tuple(wrap(x) for x in tiny_function(*[unwrap(x.tiny()) for x in args]))
    return torch_function
  return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)
