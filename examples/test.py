from tinygrad import Tensor, dtypes
from einops import rearrange, repeat
import numpy as np
import torch

# dim = 10
# t = np.arange(16).reshape(2,2,2,2).astype(np.float32)
# tNew = repeat(t, "B G N L -> B (G H) N L", H=dim // t.shape[1])
# # tNew = rearrange(t, "b d l -> (b l) d")
# tTiny = Tensor.arange(16).reshape(2,2,2,2).cast(dtypes.float32)
# tTinyNew = tTiny.repeat((1,dim//tTiny.shape[1],1,1))#.reshape(4,2)
# # tTinyNew = tTiny.permute(0,2,1)#.reshape(4,2)
# print(t)
# print(tNew)
# print(tTiny.numpy())
# # print(tTiny.shape)
# print(tTinyNew.numpy())

# print(tNew.shape, tTinyNew.shape)

t1 = torch.arange(16).float()
t2 = Tensor.arange(16).float()

v1 = t1.masked_fill(t1>10, -1e9)


print(t1.numpy())
print(v1.numpy())









# t1 = torch.arange(16).reshape(4,4)+10
# t2 = Tensor.arange(16).reshape(4,4)+10

# v1k = t1.kthvalue(3).values
# v1 = t1[3]
# v2 = t2[3]

# print(t1.numpy(), t2.numpy())
# print(v1.numpy(), v2.numpy())
# print(v1k.numpy())