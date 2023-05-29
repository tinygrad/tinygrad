from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG
from tinygrad.jit import TinyJit
if DEBUG < 2: DEBUG(2)

if __name__ == "__main__":
  print("add in L2")
  a = Tensor.randn(4*1024*1024).realize()
  b = Tensor.randn(4*1024*1024).realize()
  for _ in range(10): (a+b).realize()

  print("add in memory")
  a = Tensor.randn(32*1024*1024)
  b = Tensor.randn(32*1024*1024)
  for _ in range(10): (a+b).realize()
