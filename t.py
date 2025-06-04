from tinygrad import Tensor

a = Tensor.empty(4, 4, 1)
b = a.squeeze()
print(b.unsqueeze(0))
