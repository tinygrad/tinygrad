from tinygrad import Tensor
x = Tensor.arange(6).reshape(2,3)

print(x.sum(axis=1           ).shape)  # expect (2,)
print(x.sum(axis=1, keepdim=True).shape)  # expect (2,1)
