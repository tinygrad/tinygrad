import numpy as np
from tinygrad.tensor import Tensor

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False):
    # Multi-Head Attention
    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    assert self.head_size * self.num_heads == embed_dim
    self.prenorm = prenorm

    # added bias
    self.query_dense = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key_dense = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value_dense = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.final = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    embed_dim = self.num_heads * self.head_size

    query, key, value = [x.linear(y) \
      .reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)) \
      for y in [self.query_dense, self.key_dense, self.value_dense]]

    query = query.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)
    key = key.transpose(order=(0,2,3,1))      # (bs, num_heads, head_size, T)
    value = value.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)

    score = query.dot(key) * (1 / np.sqrt(self.head_size))
    weights = score.softmax()                                   # (bs, num_heads, T, T)
    attention = weights.dot(value).transpose(order=(0,2,1,3))   # (bs, T, num_heads, head_size)

    return attention.reshape(shape=(x.shape[0], -1, embed_dim)).linear(self.final)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(self.ln1)).dropout(0.1)
      x = x + x.layernorm().linear(self.ln2).linear(self.ff1).gelu().linear(self.ff2).dropout(0.1)
    else:
      x = x + self.attn(x).dropout(0.1)
      x = x.layernorm().linear(self.ln1)
      x = x + x.linear(self.ff1).relu().linear(self.ff2).dropout(0.1)
      x = x.layernorm().linear(self.ln2)
    return x

class Transformer:
  # L = layers, H = embed_dim, A = num_heads
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = []
    for i in range(layers):
      self.tbs.append(TransformerBlock(embed_dim, num_heads, ff_dim))
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

class ViT:
  def __init__(self, embed_dim=192):
    self.conv_weight = Tensor.uniform(embed_dim, 3, 16, 16)
    self.conv_bias = Tensor.zeros(embed_dim)
    self.cls_token = Tensor.ones(1, 1, embed_dim)
    self.tbs = [TransformerBlock(embed_dim=embed_dim, num_heads=3, ff_dim=768, prenorm=True) for i in range(12)]
    self.pos_embed = Tensor.ones(1, 197, embed_dim)
    self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))
    self.norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))

  def patch_embed(self, x):
    x = x.conv2d(self.conv_weight, stride=16)
    x = x.add(self.conv_bias.reshape(shape=(1,-1,1,1)))
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).transpose(order=(0,2,1))
    return x

  def forward(self, x):
    pe = self.patch_embed(x)
    x = self.cls_token.add(Tensor.zeros(pe.shape[0],1,1)).cat(pe, dim=1) + self.pos_embed
    for l in self.tbs:
      x = l(x)
    x = x.layernorm().linear(self.norm)
    return x[:, 0].linear(self.head)

