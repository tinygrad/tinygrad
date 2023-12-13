from tinygrad.tensor import Tensor

if __name__ == '__main__':
  a = Tensor.rand([5, 5])
  b = Tensor.rand([5, 5])
  (a.std() + b.mean()).numpy()
