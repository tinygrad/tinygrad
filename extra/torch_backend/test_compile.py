# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
from extra.torch_backend.backend import unwrap, wrap

from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified

from tinygrad import Tensor, TinyJit

@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  # convert sample_inputs to CPU for dynamo tracing
  cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) and x.device.type == "tiny" else x for x in sample_inputs]
  for name, param in gm.named_parameters():
    if param.device.type == "tiny": param.data = param.data.cpu()
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args):
      wrapped = [wrap(x) if isinstance(x, Tensor) else x for x in args]
      outs = gm(*wrapped)
      for x in outs:
        if x is not None and isinstance(x, torch.Tensor): unwrap(x).realize()
      return outs
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args):
      converted = [unwrap(x.tiny()) if isinstance(x, torch.Tensor) else x for x in args]
      outs = tiny_function(*converted)
      if isinstance(outs, tuple): return tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in outs)
      return outs.cpu() if isinstance(outs, torch.Tensor) else outs
    return torch_function
  return aot_module_simplified(gm, cpu_inputs, decompositions={}, fw_compiler=my_compiler)

if __name__ == "__main__":
  def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

  print("calling compile")
  opt_foo1 = torch.compile(foo, backend="tiny")
  print("compiled")
  for i in range(5):
    out = opt_foo1(torch.randn(10, 10), torch.randn(10, 10))
    print(out.device)
