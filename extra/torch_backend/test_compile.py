# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
import extra.torch_backend.backend

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
