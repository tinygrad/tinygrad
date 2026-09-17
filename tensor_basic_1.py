from tinygrad import Tensor

# N = 10
# cmp = Tensor.empty(N)
# cmp[0:10] = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
# print(cmp.tolist())

N = 10
cmp = Tensor.empty(N)
for i in range(N): cmp[i] = i
print(cmp.tolist())

"""
Replacing last line with self.assign(res) and no realizes in setitem made result:
[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
"""

# from tinygrad import Tensor
# t = Tensor([1,2,3,4])
# t_plus_3_plus_4 = t + 3 + 4
# print(t_plus_3_plus_4.tolist())

