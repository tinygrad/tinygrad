# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
from extra.torch_backend.backend import unwrap, wrap

from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func

from tinygrad import Tensor, TinyJit

@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args:Tensor):
      outs = gm(*[wrap(x) if isinstance(x, Tensor) else x for x in args])
      Tensor.realize(*[unwrap(x) for x in outs if isinstance(x, torch.Tensor)])
      return outs
    def torch_function(*args):
      tiny_args = [unwrap(x.tiny()) if isinstance(x, torch.Tensor) else x for x in args]
      return [x.cpu() if isinstance(x, torch.Tensor) else x for x in tiny_function(*tiny_args)]
    return make_boxed_func(torch_function)
  return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)

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
