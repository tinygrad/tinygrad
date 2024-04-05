from tinygrad import Tensor
import copy, sys
import torch
import numpy as np
from typing import List, Optional

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

# t = Tensor.arange(4*4).reshape(4,4)
# b = Tensor([True, True, False, True]).reshape(-1,1)

# print(t.numpy())
# print(b.numpy(), b.shape)
# a = t*b
# print(a.numpy())

# def gen_anchor_torch(scales: List[int], aspect_ratios: List[float], dtype: torch.dtype = torch.float32,
#                          device: torch.device = torch.device("cpu")):
#         scales = torch.as_tensor(scales, dtype=dtype, device=device)
#         aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
#         h_ratios = torch.sqrt(aspect_ratios)
#         w_ratios = 1 / h_ratios
#         print('TORCH', w_ratios.shape, scales.shape)

#         ws = (w_ratios[:, None] * scales[None, :]).view(-1)
#         hs = (h_ratios[:, None] * scales[None, :]).view(-1)
#         print(ws.shape, hs.shape)
#         base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
#         return base_anchors.round()
# def generate_anchors(scales: List[int], aspect_ratios: List[float], dtype=dtypes.float):
#     scales = Tensor(list(scales), dtype=dtype)
#     aspect_ratios = Tensor(list(aspect_ratios), dtype=dtype)
#     h_ratios = aspect_ratios.sqrt()
#     w_ratios = 1 / h_ratios
#     print('TINY', w_ratios.shape, scales.shape)

#     # ws = (w_ratios[:, None] * scales[None, :])  #.view(-1)
#     # hs = (h_ratios[:, None] * scales[None, :])  #.view(-1)

#     ws = (w_ratios.unsqueeze(1) * scales.unsqueeze(0)).reshape(-1)  #.view(-1)
#     hs = (h_ratios.unsqueeze(1) * scales.unsqueeze(0)).reshape(-1)  #.view(-1)
#     print(ws.shape, hs.shape)

#     base_anchors = Tensor.stack([-ws, -hs, ws, hs], dim=1) / 2
#     return base_anchors.round()
# sizes = ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))
# aspect_ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
# tens_tor = gen_anchor_torch(sizes[0], aspect_ratios[0])
# tens_tiny = generate_anchors(sizes[0], aspect_ratios[0])

# print('TORCH', tens_tor.shape)
# print(tens_tor)
# print('TINY', tens_tiny.shape)
# print(tens_tiny.numpy())

# print(list(sizes[0]))
# print(sizes[0])

# t = Tensor.arange(16*4).reshape(4,4,4)
# l = t.chunk(4)
# l = [ll.squeeze(0) for ll in l]
# print(t.numpy())
# print(l)
# for ll in l:
#     print(ll.numpy())

# t1 = Tensor([True, False, True, True, False]).reshape(-1,1)
# t2 = Tensor.arange(3).reshape(1,-1)+2

# a = t1*t2
# print(a.numpy())
# print(t1.numpy())
# print(t2.numpy())

# t = Tensor([1,2,3,4,5]).flip(axis=1)
# print(t.numpy())

# t = Tensor(2)
# a = t.one_hot(5)
# print(a.numpy())

# b = (1,2,3,4)
# print(b[-2:])


# np.set_printoptions(threshold=sys.maxsize)
# R_SIZE = 5
# C_SIZE = 3
# t = Tensor([True, True, False, False, True]).reshape(-1,1)
# i = Tensor.arange(5).reshape(-1,1)+1
# r = Tensor.arange(5).reshape(-1,1)+10
# # temp = t*i-1
# # print('temp', temp.numpy())
# # a = Tensor.where(t, r[0].one_hot(20), 7)
# # print(a.numpy())
# # print('A SHAPE', a.shape, 'T SHpae', t.shape, 'TEMP', temp.shape,)
# for_temp = Tensor.arange(16).reshape(4,4)
# # t = t.cast(dtypes.int)
# print('T ORIG', t.shape)
# for ii in t:
#     print(ii.numpy(), ii.shape, Tensor(True).shape, ii.numel())
#     if (ii==True):
#     # if (ii is Tensor(True)):
#         print('ii is TRUE')
#     else:
#         print('ii is FALSE')
#     print('*********')

# p = Tensor([44,55,66,77])
# for pp in p:
#     print(pp.numpy())
#     if(pp.__eq__(55)):
#         print('Hit PP')
#     else:
#         print('Miss PP')

print(Tensor([-1,1,2,50]).one_hot(5).numpy())

np.set_printoptions(threshold=sys.maxsize)
R_SIZE = 5
C_SIZE = 3
rows = Tensor([True, True, False, False, True])
row_idx = Tensor.arange(5)
col_temp = Tensor([2,3,4,5,6])
masked_row_idx = Tensor.where(rows, row_idx, -5)
print('masked_row_idx', masked_row_idx.numpy())
logits = []
for m,r,c in zip(rows, masked_row_idx, col_temp):
    print(m.numpy(),'||', r.numpy(), '||', c.numpy())
    print('*******')
    # temp_append = Tensor.where(m, )
    # if(m):
    #     print('hit')
    # else:
    #     print('miss')



t = Tensor([2,3,4,5,6])
print(t[[2,3,7,1]].numpy())

print(Tensor([0]).log().numpy())
print((Tensor([5])/0).numpy())