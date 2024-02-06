from tinygrad import Tensor, dtypes
from einops import rearrange, repeat
import numpy as np

dim = 10
t = np.arange(16).reshape(2,2,2,2).astype(np.float32)
tNew = repeat(t, "B G N L -> B (G H) N L", H=dim // t.shape[1])
# tNew = rearrange(t, "b d l -> (b l) d")
tTiny = Tensor.arange(16).reshape(2,2,2,2).cast(dtypes.float32)
tTinyNew = tTiny.repeat((1,dim//tTiny.shape[1],1,1))#.reshape(4,2)
# tTinyNew = tTiny.permute(0,2,1)#.reshape(4,2)
print(t)
print(tNew)
print(tTiny.numpy())
# print(tTiny.shape)
print(tTinyNew.numpy())

print(tNew.shape, tTinyNew.shape)