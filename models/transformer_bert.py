import math
import numpy as np
from tinygrad.tensor import Tensor
from models.transformer import TransformerBlock


# src: https://github.com/karpathy/nanoGPT/blob/7fe4a099ad2a4654f96a51c0736ecf347149c34c/model.py#L19
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + Tensor.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * Tensor.pow(x, 3.0))))

class Bert:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim)
    self.tbs = []
    for i in range(layers):
      self.tbs.append(TransformerBlock(embed_dim, num_heads, ff_dim, prenorm=True, act=new_gelu))
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]
    xnp = x.cpu().numpy().astype(np.int32)
    onehot = np.zeros((bs, x.shape[1], self.maxlen+self.syms), dtype=np.float32)
    for i in range(x.shape[1]):
      onehot[range(bs), i, i] = 1
      onehot[range(bs), i, self.maxlen + xnp[:, i]] = 1
    onehot = onehot.reshape(bs*x.shape[1], self.maxlen+self.syms)

    x = Tensor(onehot, device=x.device).dot(self.embed).reshape(shape=(bs, x.shape[1], -1))
    x = x.sequential(self.tbs)
    x = x.reshape(shape=(-1, x.shape[-1])).dot(self.final).log_softmax()
    return x.reshape(shape=(bs, -1, x.shape[-1]))
