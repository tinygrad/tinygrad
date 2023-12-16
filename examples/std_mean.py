from tinygrad.tensor import Tensor

if __name__ == '__main__':
  a = Tensor([1, 2, 3, 4])
  b = Tensor([1, 2, 3, 4])
  print((a.std() + b.mean()).numpy())
