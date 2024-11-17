from tinygrad import Tensor

a = Tensor([6])
b1 = Tensor([8])
b2 = 8

a.__mod__(b1)

a % b1
b1 % a

a % b2
b2 % a
