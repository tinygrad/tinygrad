import math
import numpy as np
from tinygrad.tensor import Tensor

import torch.nn.functional as F
import math
from typing import List, Tuple, Union, Optional
from tinygrad.helpers import dtypes
from tinygrad.nn import Embedding

def SparsifyIndices(
    x: Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[int, Tensor, Optional[Tensor]]:
    b, n, c = x.size()

    x_indices = Tensor.arange(0, n, dtype=dtypes.int64, device=x.device)[None, :, None]

    num_subatt = sum([int(math.ceil(n / w)) for w in ws])
    max_subatt_n = min(n, max([w // r for w, r in zip(ws, rs)]))

    sparse_indices = -1*Tensor.ones((b, num_subatt * max_subatt_n, c), device=x.device, dtype=np.int64)

    subatt_idx = 0
    for w, r in zip(ws, rs):
        for segment_indices in np.split(x_indices, w, 1):
            offset = head_idx % r
            cur_sparse_indices = segment_indices[:, offset::r, :]
            start_idx = subatt_idx*max_subatt_n
            end_idx = start_idx+cur_sparse_indices.shape[1]
            sparse_indices[:, start_idx:end_idx] = cur_sparse_indices
            subatt_idx += 1

    if -1 in sparse_indices:
        padding_mask = sparse_indices[:, :, 0] != -1

        # to allow gather work for batching
        sparse_indices[~padding_mask] = 0

        # combine batch and subattention dims
        padding_mask = padding_mask.view((-1, max_subatt_n))
    else:
        padding_mask = None

    return max_subatt_n, sparse_indices, padding_mask


class RelativePositionBias():
    def __init__(
        self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(dtypes.int64) * num_buckets
            n = Tensor.abs(n)
        else:
            n = Tensor.max(n, Tensor.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            Tensor.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(dtypes.int64)
        val_if_large = Tensor.min(
            val_if_large, Tensor.full_like(val_if_large, num_buckets - 1)
        )

        ret += Tensor.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        step = 0 if step is None else step
        context_position = Tensor.arange(
            step,
            step + qlen,
            dtype=dtypes.int64,
            device=self.relative_attention_bias.weight.device,
        )[:, None]
        memory_position = Tensor.arange(
            klen, dtype=dtypes.int64, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        # shape (batch * num_heads, qlen, klen)
        return (
            self.compute_bias(qlen, klen, step)
            .repeat(batch_size, 1, 1, 1)
            .view(-1, qlen, klen)
        )

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = Tensor.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (Tensor.arange(0, dim) / dim))
    sinusoid_inp = (
        np.einsum("i , j -> i j", Tensor.arange(0, seq_len, dtype=Tensor.float), inv_freq).to(x)
    )
    return Tensor.sin(sinusoid_inp), Tensor.cos(sinusoid_inp)

class XPOS():
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (Tensor.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** Tensor.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
class DilatedTransformerBlock:
  def __init__(
      self, 
      embed_dim, 
      num_heads, 
      ff_dim, 
      dilation_rate, 
      segment_size, 
      casual, 
      use_xpos, 
      use_rel_pos_bias,
      prenorm=False, 
      act=lambda x: x.relu(), 
      dropout=0.1
      ):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.dilation_rate = dilation_rate
    self.segment_size = segment_size

    self.casual = casual

    self.use_xpos = use_xpos
    self.use_rel_pos_bias = use_rel_pos_bias

    if use_xpos:
        self.xpos = XPOS(head_dim=embed_dim//num_heads)
    if use_rel_pos_bias:
        self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)


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

    # get dimensions
    batch_size, seq_len, _ = x.shape

    # calculate the necessary padding
    padding_len = -seq_len % self.segment_size
    x = F.pad(x, (0,0,0,padding_len))
    seq_len = seq_len + padding_len

    if self.use_xpos:
        x = self.xpos(x)

    # Prepare sparse indices
    max_subatt_n, sparse_indices, padding_mask = SparsifyIndices(x, [self.segment_size], [self.dilation_rate], self.head_offsets)

    # Split and sparsify
    x = x.view(batch_size, -1, self.segment_size, self.d_model)
    x = x.gather(1, sparse_indices[:, :, :x.size(1)])

    query, key, value = [x.linear(*y) \
      .reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)) \
      for y in [self.query, self.key, self.value]]

    query = query.permute(order=(0,2,1,3))  # (bs, num_heads, time, head_size)
    key = key.permute(order=(0,2,3,1))      # (bs, num_heads, head_size, time)
    value = value.permute(order=(0,2,1,3))  # (bs, num_heads, time, head_size)

    # SCORE
    score = query.dot(key) * (1 / np.sqrt(self.head_size))
    # NORM
    weights = score.softmax()                                   # (bs, num_heads, time, time)
    # VALUES
    attention = weights.dot(value).permute(order=(0,2,1,3))   # (bs, time, num_heads, head_size)

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

class DilatedTransformer:
  def __init__(
      self, 
      syms, 
      maxlen, 
      layers, 
      embed_dim, 
      num_heads, 
      ff_dim,
      dilation_rate = 1,  # additional parameter for DilatedAttention
      segment_size = 1,  # additional parameter for DilatedAttention
      casual = False,  # additional parameter for DilatedAttention
      use_xpos = False,  # additional parameter for DilatedAttention
      use_rel_pos_bias = False,  # additional parameter for DilatedAttention
      distributed = False,  # additional parameter for DilatedAttention
    ):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = []
    for i in range(layers):
      self.tbs.append(DilatedTransformerBlock(
        embed_dim, 
        num_heads, 
        ff_dim,
        dilation_rate=dilation_rate, 
        segment_size=segment_size, 
        casual=casual, 
        use_xpos=use_xpos, 
        use_rel_pos_bias=use_rel_pos_bias
      ))
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

