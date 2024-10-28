from tinygrad import Tensor, dtypes
#a = Tensor.ones((10,), dtype=dtypes.int32)
#b = Tensor.ones((10,), dtype=dtypes.int32)
#print((a+b).numpy())

#a = Tensor.arange(0, 6)
#b = Tensor.arange(6, 12)
#c = Tensor.arange(0, 6)
#print((a+b*c).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a == b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a > b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a / b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a // b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a & b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a | b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a - b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a ^ b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a.max()).numpy())

#a = Tensor([2,3,4])
#b = Tensor([1,2,3])
#c = Tensor([False, True, False])
#print(c.where(a, b).numpy())

#a = Tensor([2,2,3])
#b = Tensor([1,2,3])
#print((a.min()).numpy())

a = Tensor([2,2,3])
b = Tensor([1,2,3])
print((a.matmul(b)).numpy())

#a = Tensor([2,2,3])
#print((a.sqrt()).numpy())

#a = Tensor([2,2,3])
#print(a.sin().numpy())

'''
# hell nawww, this is the final boss
a = Tensor([2,2,3])
b = Tensor([1,2,3])
print((a.log()).numpy())
'''