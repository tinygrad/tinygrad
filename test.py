from tinygrad import Tensor
import copy
import torch
import numpy as np

from tinygrad.dtype import dtypes
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

# t = Tensor.arange(5)
# t[2] = Tensor(56).realize()
# t.realize()
# print(t.numpy())
# b = Tensor([True, True, False, False, True])
# # a = t*b
# # print(t.numpy())
# # print(a.numpy())

# for tt,bb in zip(t,b):
#     print(tt.numpy(),bb.numpy())

# print(np.arange(8))

# t = torch.rand((90000))
# ans = torch.where(t<=0.2)
# print(t)
# print(ans)

# t = Tensor([18,22,34,6,8])
# t1 = t>10
# print(t1.numpy(), t1.dtype)

# l = [Tensor(1),Tensor(2)]
# t = Tensor.stack(l)
# print(t.numpy())

# to = torch.arange(5)
# tnew = to.view(-1)
# print(tnew)

# xt = torch.Tensor([1,2,3])
# yt = torch.Tensor([4,5,6,7])
# x = Tensor([1,2,3])
# y = Tensor([4,5,6,7])

# def cust_meshgrid(x:Tensor, y:Tensor):
#     xs = x.shape[0]
#     ys = y.shape[0]
#     y = Tensor.stack([y]*xs)
#     x = x.reshape(xs, 1).expand((xs,ys))
#     return x, y

# xt, yt = torch.meshgrid(xt, yt)
# x,y = cust_meshgrid(x, y)
# print(xt)
# print(yt)
# print(x.numpy())
# print(y.numpy())

# t = [(2,3)]*5
# print(t)

# t = Tensor.arange(16).reshape(4,4)
# ans =[]
# for i in range(4):
#     ans.append(t[i])
# print(ans)
# for i in ans:
#     print(i.numpy())

# t = Tensor([11,22,34,44])
# print(t[[2,0]].numpy())
# print(t[[]].numpy())

# tor = torch.tensor([11,22,34,44])
# print(tor[[2,0]].numpy())
# print(tor[[]].numpy())

# tor1 = torch.ones(5,4)+1
# t1 = Tensor.ones(5,4)+1
# tor1_unsqueeze = tor1[:, None, :2]
# t1_unsqueeze = t1
# tor2 = torch.ones(40029,4)
# tor_ans = torch.max(tor1[:, None, :2], tor2[:, :2])
# print(tor1)
# print(tor1_unsqueeze.shape)
# print(t1_unsqueeze.shape)
# print(tor_ans.shape)

# print(tor_ans)
# t = Tensor([[11,22,33,44,55]]*5)+1
# a = [1,2,3,4,5,6]

# for tt, aa in zip(t, a):
#     print(tt.numpy())
#     print(aa)
t = Tensor.arange(4*4).reshape(4,4)
b = Tensor([True, True, False, True]).reshape(-1,1)

print(t.numpy())
print(b.numpy(), b.shape)
a = t*b
print(a.numpy())