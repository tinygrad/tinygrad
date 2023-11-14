from tinygrad.tensor import Tensor
import numpy as np

if __name__ == "__main__":
    a = Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    b = Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    c = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    d = c.sum(axis=1)
    print(d.numpy())
    #print((a + b + c.square()).numpy())
    #print((a + b * c).numpy())