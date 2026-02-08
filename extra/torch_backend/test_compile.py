# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
from extra.torch_backend.backend import unwrap, wrap
from extra.torch_backend.compile_backend import register_tiny_backend

register_tiny_backend(wrap, unwrap)

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
