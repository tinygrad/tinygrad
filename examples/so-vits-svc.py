import json, logging, math, os, re, sys, time, wave, argparse, numpy as np
from functools import reduce
from pathlib import Path
from typing import List
from extra.utils import download_file
from tinygrad import nn
from tinygrad.nn import Conv1d, Conv2d
from tinygrad.helpers import dtypes
from tinygrad.state import torch_load
from tinygrad.tensor import Tensor
from unidecode import unidecode

# TODO: review and refactor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1

class Synthesizer:
  def __init__(
    self,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    gin_channels,
    ssl_dim,
    n_speakers,
    sampling_rate=44100,
    vol_embedding=False,
    vocoder_name="nsf-hifigan",
    use_depthwise_conv=False,
    use_automatic_f0_prediction=True,
    flow_share_parameter=False,
    n_flow_layer=4,
    **kwargs
  ):

    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.vol_embedding = vol_embedding
    self.emb_g = nn.Embedding(n_speakers, gin_channels)
    self.use_depthwise_conv = use_depthwise_conv
    self.use_automatic_f0_prediction = use_automatic_f0_prediction
    if vol_embedding:
      self.emb_vol = nn.Linear(1, hidden_channels)

    self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

    self.enc_p = TextEncoder(
      inter_channels,
      hidden_channels,
      filter_channels=filter_channels,
      n_heads=n_heads,
      n_layers=n_layers,
      kernel_size=kernel_size,
      p_dropout=p_dropout
    )
    hps = {
      "sampling_rate": sampling_rate,
      "inter_channels": inter_channels,
      "resblock": resblock,
      "resblock_kernel_sizes": resblock_kernel_sizes,
      "resblock_dilation_sizes": resblock_dilation_sizes,
      "upsample_rates": upsample_rates,
      "upsample_initial_channel": upsample_initial_channel,
      "upsample_kernel_sizes": upsample_kernel_sizes,
      "gin_channels": gin_channels,
      "use_depthwise_conv": use_depthwise_conv
    }

    modules.set_Conv1dModel(self.use_depthwise_conv)

    if vocoder_name == "nsf-hifigan":
      from vdecoder.hifigan.models import Generator
      self.dec = Generator(h=hps)
    elif vocoder_name == "nsf-snake-hifigan":
      from vdecoder.hifiganwithsnake.models import Generator
      self.dec = Generator(h=hps)
    else:
      print("[?] Unkown vocoder: use default(nsf-hifigan)")
      from vdecoder.hifigan.models import Generator
      self.dec = Generator(h=hps)

    self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels,
                                      share_parameter=flow_share_parameter)
    if self.use_automatic_f0_prediction:
      self.f0_decoder = F0Decoder(
        1,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        spk_channels=gin_channels
      )
    self.emb_uv = nn.Embedding(2, hidden_channels)
    self.character_mix = False

  def EnableCharacterMix(self, n_speakers_map, device):
    self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
    for i in range(n_speakers_map):
      self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
    self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
    self.character_mix = True

  def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol=None):
    g = self.emb_g(g).transpose(1, 2)

    # vol proj
    vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0

    # ssl prenet
    x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
    x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

    # f0 predict
    if self.use_automatic_f0_prediction:
      lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
      norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
      pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
    else:
      lf0 = 0
      norm_lf0 = 0
      pred_lf0 = 0
    # encoder
    z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

    # flow
    z_p = self.flow(z, spec_mask, g=g)
    z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)

    # nsf decoder
    o = self.dec(z_slice, g=g, f0=pitch_slice)

    return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0


class ResidualCouplingBlock:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0, share_parameter=False):
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = []

    self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

    for _ in range(n_flows):
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
  def __init__(self, out_channels, hidden_channels, kernel_size, n_layers, gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None):

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
      p_dropout
    )

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


class DiscriminatorP:
  def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
    self.period = period
    self.use_spectral_norm = use_spectral_norm
    norm_f = weight_norm if use_spectral_norm is False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
    ])
    self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x):
    fmap = []

    # 1d to 2d
    b, c, t = x.shape
    if t % self.period != 0:  # pad first
      n_pad = self.period - (t % self.period)
      x = F.pad(x, (0, n_pad), "reflect")
      t = t + n_pad
    x = x.view(b, c, t // self.period, self.period)

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class DiscriminatorS:
  def __init__(self, use_spectral_norm=False):
    norm_f = weight_norm if use_spectral_norm is False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv1d(1, 16, 15, 1, padding=7)),
      norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
      norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
      norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
      norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
      norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
    ])
    self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

  def forward(self, x):
    fmap = []

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, LRELU_SLOPE)
      fmap.append(x)
      x = self.conv_post(x)
      fmap.append(x)
      x = torch.flatten(x, 1, -1)
    return x, fmap


class MultiPeriodDiscriminator:
  def __init__(self, use_spectral_norm=False):
    periods = [2, 3, 5, 7, 11]
    discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
    discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
    self.discriminators = nn.ModuleList(discs)

  def forward(self, y, y_hat):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []
    for i, d in enumerate(self.discriminators):
      y_d_r, fmap_r = d(y)
      y_d_g, fmap_g = d(y_hat)
      y_d_rs.append(y_d_r)
      y_d_gs.append(y_d_g)
      fmap_rs.append(fmap_r)
      fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class FFT:
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.,
               proximal_bias=False, proximal_init=True, **kwargs):

    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(
        MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(
        FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
    x = x * x_mask
    return x


class F0Decoder(nn.Module):
  def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=0):

    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.spk_channels = spk_channels
    self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
    self.decoder = FFT(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
    self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
      x = torch.detach(x)
      if (spk_emb is not None):
        x = x + self.cond(spk_emb)
      x += self.f0_prenet(norm_f0)
      x = self.prenet(x) * x_mask
      x = self.decoder(x * x_mask, x_mask)
      x = self.proj(x) * x_mask
      return x


def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)


def get_padding(kernel_size, dilation=1): return int((kernel_size*dilation - dilation)/2)


def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: int):
    n_channels_int, in_act = n_channels, input_a + input_b
    t_act, s_act = in_act[:, :n_channels_int, :].tanh(), in_act[:, n_channels_int:, :].sigmoid()
    return t_act * s_act


def split(tensor, split_sizes, dim=0):  # if split_sizes is an integer, convert it to a tuple of size split_sizes elements
  if isinstance(split_sizes, int): split_sizes = (split_sizes,) * (tensor.shape[dim] // split_sizes)
  assert sum(split_sizes) == tensor.shape[
    dim], "Sum of split_sizes must equal the dimension size of tensor along the given dimension."
  start, slices = 0, []
  for size in split_sizes:
    slice_range = [(start, start + size) if j == dim else None for j in range(len(tensor.shape))]
    slices.append(slice_range)
    start += size
  return [tensor.slice(s) for s in slices]
