import json, logging, math, os, re, sys, time, wave, argparse, numpy as np
from functools import reduce
from pathlib import Path
from typing import List
from extra.utils import download_file
from tinygrad import nn
from tinygrad.helpers import dtypes
from tinygrad.state import torch_load
from tinygrad.tensor import Tensor
from unidecode import unidecode


class ResidualCouplingBlock:
  def __init__(self,
               channels,
               hidden_channels,
               kernel_size,
               dilation_rate,
               n_layers,
               n_flows=4,
               gin_channels=0,
               share_parameter=False
               ):
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = []

    self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
      self.flows.append(Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class LayerNorm(nn.LayerNorm):
  def __init__(self, channels, eps=1e-5): super().__init__(channels, eps, elementwise_affine=True)
  def forward(self, x: Tensor): return self.__call__(x.transpose(1, -1)).transpose(1, -1)


class WN:
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    assert (kernel_size % 2 == 1)
    self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels, self.p_dropout = hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout
    self.in_layers, self.res_skip_layers = [], []
    if gin_channels != 0: self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
    for i in range(n_layers):
      dilation = dilation_rate ** i
      self.in_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation,
                                      padding=int((kernel_size * dilation - dilation) / 2)))
      self.res_skip_layers.append(
        nn.Conv1d(hidden_channels, 2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1))

  def forward(self, x, x_mask, g=None, **kwargs):
    output = Tensor.zeros_like(x)
    if g is not None: g = self.cond_layer(g)
    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = Tensor.zeros_like(x_in)
      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts
    return output * x_mask


class ResidualCouplingLayer:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.mean_only = channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only
    self.half_channels = channels // 2
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = split(x, [self.half_channels] * 2, 1)
    stats = self.post(self.enc.forward(self.pre(x0) * x_mask, x_mask, g=g)) * x_mask
    if not self.mean_only:
      m, logs = split(stats, [self.half_channels] * 2, 1)
    else:
      m = stats
      logs = Tensor.zeros_like(m)
    if not reverse: return x0.cat((m + x1 * logs.exp() * x_mask), dim=1)
    return x0.cat(((x1 - m) * (-logs).exp() * x_mask), dim=1)

class Flip:
  def forward(self, x: Tensor, *args, reverse=False, **kwargs):
    return x.flip([1]) if reverse else (x.flip([1]), Tensor.zeros(x.shape[0], dtype=x.dtype).to(device=x.device))

class Log:
  def forward(self, x : Tensor, x_mask, reverse=False):
    if not reverse:
      y = x.maximum(1e-5).log() * x_mask
      return y, (-y).sum([1, 2])
    return x.exp() * x_mask


class TextEncoder:
  def __init__(self,
               out_channels,
               hidden_channels,
               kernel_size,
               n_layers,
               gin_channels=0,
               filter_channels=None,
               n_heads=None,
               p_dropout=None):

    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.f0_emb = nn.Embedding(256, hidden_channels)

    self.enc_ = Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)

  #TODO: remove torch
  def forward(self, x, x_mask, f0=None, noice_scale=1):
    x = x + self.f0_emb(f0).transpose(1, 2)
    x = self.enc_(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

    return z, m, logs, x_mask


class MultiHeadAttention:
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    assert channels % n_heads == 0
    self.channels, self.out_channels, self.n_heads, self.p_dropout, self.window_size, self.heads_share, self.block_length, self.proximal_bias, self.proximal_init = channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init
    self.attn, self.k_channels  = None, channels // n_heads
    self.conv_q, self.conv_k, self.conv_v = [nn.Conv1d(channels, channels, 1) for _ in range(3)]
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    if window_size is not None: self.emb_rel_k, self.emb_rel_v = [Tensor.randn(1 if heads_share else n_heads, window_size * 2 + 1, self.k_channels) * (self.k_channels ** -0.5) for _ in range(2)]

  def forward(self, x, c, attn_mask=None):
    q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    return self.conv_o(x)

  def attention(self, query: Tensor, key: Tensor, value: Tensor, mask=None):# reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = key.shape[0], key.shape[1], key.shape[2], query.shape[2]
    query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
      scores = scores + self._relative_position_to_absolute_position(rel_logits)
    if mask is not None:
      scores = Tensor.where(mask, scores, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        scores = Tensor.where(Tensor.ones_like(scores).triu(-self.block_length).tril(self.block_length), scores, -1e4)
    p_attn = scores.softmax(axis=-1)  # [b, n_h, t_t, t_s]
    output = p_attn.matmul(value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().reshape(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y): return x.matmul(y.unsqueeze(0))                 # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]

  def _matmul_with_relative_keys(self, x, y): return x.matmul(y.unsqueeze(0).transpose(-2, -1)) # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]

  def _get_relative_embeddings(self, relative_embeddings, length):
    pad_length, slice_start_position = max(length - (self.window_size + 1), 0), max((self.window_size + 1) - length, 0)
    padded_relative_embeddings = relative_embeddings if pad_length <= 0\
      else relative_embeddings.pad(convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    return padded_relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)] #used_relative_embeddings

  def _relative_position_to_absolute_position(self, x: Tensor): # x: [b, h, l, 2*l-1] -> [b, h, l, l]
    batch, heads, length, _ = x.shape
    x = x.pad(convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
    x_flat = x.reshape([batch, heads, length * 2 * length]).pad(convert_pad_shape([[0,0],[0,0],[0,length-1]]))
    return x_flat.reshape([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]

  def _absolute_position_to_relative_position(self, x: Tensor): # x: [b, h, l, l] -> [b, h, l, 2*l-1]
    batch, heads, length, _ = x.shape
    x = x.pad(convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.reshape([batch, heads, length**2 + length*(length -1)]).pad(convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    return x_flat.reshape([batch, heads, length, 2*length])[:,:,:,1:]


class FFN:
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    self.in_channels, self.out_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.activation, self.causal = in_channels, out_channels, filter_channels, kernel_size, p_dropout, activation, causal
    self.padding = self._causal_padding if causal else self._same_padding
    self.conv_1, self.conv_2 = nn.Conv1d(in_channels, filter_channels, kernel_size), nn.Conv1d(filter_channels, out_channels, kernel_size)

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    x = x * (1.702 * x).sigmoid() if self.activation == "gelu" else x.relu()
    return self.conv_2(self.padding(x.dropout(self.p_dropout) * x_mask)) * x_mask

  def _causal_padding(self, x):return x if self.kernel_size == 1 else x.pad(convert_pad_shape([[0, 0], [0, 0], [self.kernel_size - 1, 0]]))

  def _same_padding(self, x): return x if self.kernel_size == 1 else x.pad(convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1) // 2, self.kernel_size // 2]]))


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
    attn_mask, x = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1), x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i].forward(x, x, attn_mask).dropout(self.p_dropout)
      x = self.norm_layers_1[i].forward(x + y)
      y = self.ffn_layers[i].forward(x, x_mask).dropout(self.p_dropout)
      x = self.norm_layers_2[i].forward(x + y)
    return x * x_mask


def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)