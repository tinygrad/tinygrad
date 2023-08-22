import numpy as np
from tinygrad.tensor import Tensor

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class Transformer:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = []
    for i in range(layers):
      self.tbs.append(TransformerBlock(embed_dim, num_heads, ff_dim))
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]
    xnp = x.numpy().astype(np.int32)
    onehot = np.zeros((bs, x.shape[1], self.maxlen+self.syms), dtype=np.float32)
    for i in range(x.shape[1]):
      onehot[range(bs), i, i] = 1
      onehot[range(bs), i, self.maxlen + xnp[:, i]] = 1
    onehot = onehot.reshape(bs*x.shape[1], self.maxlen+self.syms)

    x = Tensor(onehot, device=x.device).dot(self.embed).reshape(shape=(bs, x.shape[1], -1))
    x = x.sequential(self.tbs)
    x = x.reshape(shape=(-1, x.shape[-1])).dot(self.final).log_softmax()
    return x.reshape(shape=(bs, -1, x.shape[-1]))

