import numpy as np
from tinygrad.tensor import Tensor

def layernorm(x, sz, eps=1e-5):
  in_shape = x.shape
  x = x.reshape(shape=(-1, sz))
  layer_mean = x.mean(axis=(1,))
  y = (x - layer_mean.reshape(shape=[-1, 1]))
  layer_var = (y*y).mean(axis=(1,))
  ret = y.div(layer_var.add(eps).reshape(shape=[-1, 1]).sqrt())
  return ret.reshape(shape=in_shape)

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

    self.final = Tensor.uniform(embed_dim, embed_dim)

    self.ff1 = Tensor.uniform(embed_dim, embed_dim)
    self.ff2 = Tensor.uniform(embed_dim, embed_dim)

  def __call__(self, x):
    # bs x T x embed_dim
    bs = x.shape[0]
    embed_dim = self.num_heads * self.head_size
    inputs = x.reshape(shape=(-1, embed_dim))

    # run multi head attention (bs, T, num_heads, head_size)
    query, key, value = [inputs.dot(y) \
      .reshape(shape=(bs, -1, self.num_heads, self.head_size)) \
      for y in [self.query_dense, self.key_dense, self.value_dense]]

    query = query.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)
    key = key.transpose(order=(0,2,3,1))      # (bs, num_heads, head_size, T)
    value = value.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)

    score = query.dot(key) * (1 / np.sqrt(self.head_size))
    weights = score.softmax()                                   # (bs, num_heads, T, T)
    attention = weights.dot(value).transpose(order=(0,2,1,3))   # (bs, T, num_heads, head_size)

    x = inputs + attention.reshape(shape=(-1, embed_dim)).dot(self.final).dropout(0.1)
    x = layernorm(x, embed_dim)
    x = x + x.dot(self.ff1).relu().dot(self.ff2).dropout(0.1)
    x = layernorm(x, embed_dim)
    return x.reshape(shape=(bs, -1, embed_dim))

class Transformer:
  # L = layers, H = embed_dim, A = num_heads
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = []
    for i in range(layers):
      self.tbs.append(TransformerBlock(embed_dim, num_heads))
    self.final = Tensor.uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]
    xnp = x.cpu().data.astype(np.int32)
    onehot = np.zeros((bs, x.shape[1], self.maxlen+self.syms), dtype=np.float32)
    for i in range(x.shape[1]):
      onehot[range(bs), i, i] = 1
      onehot[range(bs), i, self.maxlen + xnp[:, i]] = 1
    onehot = onehot.reshape(bs*x.shape[1], self.maxlen+self.syms)

    x = Tensor(onehot, device=x.device).dot(self.embed).reshape(shape=(bs, x.shape[1], -1))
    for t in self.tbs:
      x = t(x)
    x = x.reshape(shape=(-1, x.shape[-1])).dot(self.final).logsoftmax()
    return x.reshape(shape=(bs, -1, x.shape[-1]))

