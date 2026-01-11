# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
from extra.torch_backend.backend import unwrap, wrap

from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified

from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import Ops

@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args:Tensor):
      outs = gm(*[wrap(x) for x in args])
      # Filter out None outputs (e.g., for inputs without gradients in backward pass)
      tiny_outs = tuple(unwrap(x) if x is not None else None for x in outs)
      for x in tiny_outs:
        if x is not None: x.realize()
      return tiny_outs
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args:torch.Tensor):
      # Check if inputs are on CPU (need to convert outputs back to CPU)
      inputs_on_cpu = any(x.device.type == 'cpu' for x in args if isinstance(x, torch.Tensor))
      # Convert to tinygrad tensors, ensuring constants are made contiguous
      tiny_args = []
      for x in args:
        t = unwrap(x.tiny())
        # Constants need to be made contiguous to have a buffer for JIT
        if t.uop.base.op is Ops.CONST: t = t.contiguous()
        tiny_args.append(t)
      tiny_outs = tiny_function(*tiny_args)
      # Wrap outputs and convert to CPU if inputs were on CPU
      result = []
      for x in tiny_outs:
        if x is None:
          result.append(None)
        else:
          out = wrap(x)
          if inputs_on_cpu: out = out.cpu()
          result.append(out)
      return tuple(result)
    return torch_function
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
