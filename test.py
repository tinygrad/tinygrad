from tinygrad import Tensor
import copy
import torch
import numpy as np
# t = Tensor([1,2,3,4]).realize()
# nt = copy.deepcopy(t)
# t =t +1
# print(t.numpy())
# print(nt.numpy())

# a1 = [1,2,3,4]
# a1 = Tensor([4,5,6,7]).realize()

# torTens = torch.rand((1,90000))
# v, m = torTens.max(dim=0)
# print(v)
# print(torTens.shape)
# print(torTens)

t = Tensor.arange(5)
t[2] = Tensor(56).realize()
t.realize()
print(t.numpy())
# b = Tensor([True, True, False, False, True])
# # a = t*b
# # print(t.numpy())
# # print(a.numpy())

# for tt,bb in zip(t,b):
#     print(tt.numpy(),bb.numpy())

# print(np.arange(8))

t = torch.rand((90000))
ans = torch.where(t<=0.2)
print(t)
print(ans)