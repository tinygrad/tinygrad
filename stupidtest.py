from tinygrad import Tensor

a = Tensor.rand(1, 5, 4, 255, 256).realize()
a = a.sum((3,4))

print(a.numpy())