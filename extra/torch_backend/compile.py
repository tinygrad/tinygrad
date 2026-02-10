import torch
import torch._dynamo
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func
from torch._subclasses.fake_tensor import is_fake

from tinygrad import Tensor, TinyJit
from extra.torch_backend.backend import unwrap, wrap

def _to_tiny_tensor(x:torch.Tensor) -> Tensor:
  return unwrap(x) if x.device.type == "tiny" else unwrap(x.tiny())

@torch._dynamo.disable
def _to_fake_cpu_tensor(x:torch.Tensor) -> torch.Tensor:
  # Fake tiny tensors can dispatch privateuse1 view kernels during AOT tracing.
  # Move those trace-time tensors to fake CPU to keep tracing in torch-space.
  return x.to("cpu") if is_fake(x) and x.device.type == "tiny" else x


@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  sample_inputs = [_to_fake_cpu_tensor(x) if isinstance(x, torch.Tensor) else x for x in sample_inputs]
  is_backward_graph = any("backward" in str(n.target) for n in gm.graph.nodes if n.op == "call_function")
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
      # Let fake tensor tracing run in torch-space to avoid touching opaque tiny storage.
      if torch._dynamo.is_compiling() or any(is_fake(x) for x in args): return gm(*[_to_fake_cpu_tensor(x) for x in args])
      try:
        outs = tuple(wrap(x) for x in tiny_function(*[_to_tiny_tensor(x) for x in args]))
      except NotImplementedError:
        # fall back to torch graph when tiny backend misses an op in training graphs
        return gm(*args)
      # Grad-bearing paths can cross graph breaks in training; keep boundary tensors on CPU.
      if is_backward_graph or any(x.requires_grad for x in args): return tuple(x.cpu() for x in outs)
      return outs
    return make_boxed_func(torch_function)
  return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)
