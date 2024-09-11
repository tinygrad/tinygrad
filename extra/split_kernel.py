from tinygrad import Tensor

a = Tensor.rand(3, 250, 200, 1024).realize()
a = a.sum((2,3))

print(a.numpy())