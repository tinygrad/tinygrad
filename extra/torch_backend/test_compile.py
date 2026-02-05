# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
from extra.torch_backend.backend import unwrap, wrap

from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch._aot_autograd.runtime_wrappers import make_boxed_func

from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import capturing, JitError

@register_backend
def tiny(gm: torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm: torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    def tiny_function(*args: Tensor):
      torch_outs = gm(*[wrap(x) if x is not None else None for x in args])
      unwrapped_outs = []
      for x in torch_outs:
        if x is not None:
          t = unwrap(x).realize() 
          unwrapped_outs.append(t)
        else:
          unwrapped_outs.append(None)
      return unwrapped_outs
    jit_wrapped = TinyJit(tiny_function)
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args:torch.Tensor):
      tiny_args = []
      for x in args:
          if x is None:
            tiny_args.append(None)
          else:
              t = unwrap(x.tiny()).contiguous()
              tiny_args.append(t)
      if len(capturing) > 0: return [wrap(x) for x in tiny_function(*tiny_args)] 
      try:
        result = jit_wrapped(*tiny_args)
      except JitError:
        result = tiny_function(*tiny_args)
      return [wrap(x) for x in result]
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
