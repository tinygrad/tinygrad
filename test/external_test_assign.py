from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters
from tinygrad.graph import nm

if __name__ == "__main__":
  GlobalCounters.cache = []
  a = Tensor.ones(4,4)
  b = Tensor.ones(4,4)
  a.realize()
  b.realize()
  a += b
  print(a.numpy())
  runner, args = GlobalCounters.cache[0]
  b0, b1, b2 = args
  print(nm(b0), b0)
  print(nm(b1), b1)
  print(nm(b2), b2)
