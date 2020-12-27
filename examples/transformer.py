#!/usr/bin/env python3
import numpy as np
import random
from tinygrad.tensor import Tensor

# dataset idea from https://github.com/karpathy/minGPT/blob/master/play_math.ipynb
def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds)
  ds_X = ds[:, 0:6]
  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]

  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

#X_train, Y_train, X_test, Y_test = make_dataset()

class TransformerBlock:
  def __init__(self, embed_dim, num_heads):
    # Multi-Head Attention
    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    assert self.head_size * self.num_heads == embed_dim

    # looks like bias is useless
    self.query_dense = Tensor.uniform(embed_dim, embed_dim)
    self.key_dense = Tensor.uniform(embed_dim, embed_dim)
    self.value_dense = Tensor.uniform(embed_dim, embed_dim)

    self.ff1 = Tensor.uniform(embed_dim, embed_dim)
    self.ff2 = Tensor.uniform(embed_dim, embed_dim)

  def __call__(self, x):
    # bs x T x embed_dim
    bs = x.shape[0]
    x = x.reshape(shape=(-1, self.num_heads * self.head_size))

    # run multi head attention (bs, T, num_heads, head_size)
    query, key, value = [x.dot(y) \
      .reshape(shape=(bs, -1, self.num_heads, self.head_size)) \
      for y in [self.query_dense, self.key_dense, self.value_dense]]

    query = query.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)
    key = key.transpose(order=(0,2,3,1))      # (bs, num_heads, head_size, T)
    

    #score = query.reshape(shape=(-1, self.projection_dim)).dot(
    #  key.reshape(shape=(-1, self.projection_dim)).transpose(order=(1,0)))
    #scaled_score = score * (1/np.sqrt(self.projection_dim))

    print(query.shape)
    print(key.shape)
    #print(value.shape)
    #print(scaled_score.shape)

    #query = self.query_dense(x).reshape((bs, -1, self.num_heads, self.projection_dim))
    #key = self.key_dense(x).reshape((bs, -1, self.num_heads, self.projection_dim))
    #value = self.value_dense(x).reshape((bs, -1, self.num_heads, self.projection_dim))

    #x = self.ff2(self.ff1(x).relu())
    #return x

if __name__ == "__main__":
  tb = TransformerBlock(128, 4)
  tmp = Tensor.zeros(20, 10, 128)
  ret = tb(tmp)
  print(ret)

