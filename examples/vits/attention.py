import math

from tinygrad.nn import Conv1d, LayerNorm
from tinygrad.tensor import Tensor

def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

class MultiHeadAttention:
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    assert channels % n_heads == 0
    self.channels, self.out_channels, self.n_head, self.p_dropout, self.window_size, self.heads_share, self.block_length, self.proximal_bias, self.proximal_init = channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init
    self.attn = None
    self.k_channels = channels // n_heads
    self.conv_q, self.conv_k, self.conv_v = [Conv1d(channels, channels, 1)] * 3
    self.conv_o = Conv1d(channels, out_channels, 1)
    if window_size is not None:
      self.emb_rel_k, self.emb_rel_v = [Tensor.randn(1 if heads_share else n_heads, window_size * 2 + 1, self.k_channels) * (self.k_channels ** -0.5)] * 2
    # TODO: init weights as xavier_uniform for conv_q, conv_k, conv_v
    # TODO: figure out what this is for
    # if proximal_init:
    #   with torch.no_grad():
    #     self.conv_k.weight.copy_(self.conv_q.weight)
    #     self.conv_k.bias.copy_(self.conv_q.bias)

  def forward(self, x, c, attn_mask=None):
    q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    return self.conv_o(x)

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    # TODO: view equivalent in tinygrad
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = Tensor.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = scores.softmax(dim=-1)  # [b, n_h, t_t, t_s]
    p_attn = p_attn.dropout(self.p_dropout)
    output = p_attn.matmul(value)
    if self.window_size is not None:
      # TODO: implement _absolute_position_to_relative_position
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn
  def _matmul_with_relative_values(self, x, y): return x.matmul(y.unsqueeze(0))                 # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]
  def _matmul_with_relative_keys(self, x, y): return x.matmul(y.unsqueeze(0).transpose(-2, -1)) # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]
  def _get_relative_embeddings(self, relative_embeddings, length):
    pad_length, slice_start_position = max(length - (self.window_size + 1), 0), max((self.window_size + 1) - length, 0)
    padded_relative_embeddings = relative_embeddings if pad_length <= 0 else relative_embeddings.pad(convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    return padded_relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)] #used_relative_embeddings


class FFN:
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    self.in_channels, self.out_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.activation, self.causal = in_channels, out_channels, filter_channels, kernel_size, p_dropout, activation, causal
    self.padding = self._causal_padding if causal else self._same_padding
    self.conv_1, self.conv_2 = Conv1d(in_channels, filter_channels, kernel_size), Conv1d(filter_channels, out_channels, kernel_size)
  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    x = x * (1.702 * x).sigmoid() if self.activation == "gelu" else x.relu()
    x = x.dropout(self.p_dropout)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask
  def _causal_padding(self, x):
    if self.kernel_size == 1: return x
    return x.pad(convert_pad_shape([[0, 0], [0, 0], [self.kernel_size - 1, 0]]))
  def _same_padding(self, x):
    if self.kernel_size == 1: return x
    return x.pad(convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1) // 2, self.kernel_size // 2]]))

class Encoder:
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.window_size = hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size
    self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2 = [], [], [], []
    for _ in range(n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i].forward(x, x, attn_mask)
      y = y.dropout(self.p_dropout)
      x = self.norm_layers_1[i](x + y)
      y = self.ffn_layers[i].forward(x, x_mask)
      y = y.dropout(self.p_dropout)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x