from tinygrad import Tensor
from tinygrad.dtype import dtypes

if __name__ == '__main__':
    x = Tensor.rand(3)
    y = x.numpy()
    print(y)