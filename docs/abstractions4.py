# UOps are used everywhere, but they aren't always the same UOps
from tinygrad import Tensor

print("**** some math ****")
a = Tensor([1.,2,3])
b = Tensor([4.,5,6])
c: Tensor = (a+b)*2
print(c.lazydata)

print("\n**** gradient ****")
da = c.sum().gradient(a)[0]
print(da.lazydata)


