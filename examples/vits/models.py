import math

from examples.vits.attention import Encoder
from tinygrad.helpers import dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Conv1d
from tinygrad.tensor import Tensor


# PAPER: https://arxiv.org/abs/2106.06103
# CODE: https://github.com/jaywalnut310/vits/tree/main
# TODO: maybe use pre-existing attention mechanisms?

LRELU_SLOPE = 0.1

@TinyJit
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int, in_act = n_channels[0], input_a + input_b
  t_act, s_act = in_act[:, :n_channels_int, :].tanh(), in_act[:, n_channels_int:, :].sigmoid()
  return t_act * s_act
def sequence_mask(length: Tensor, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = Tensor.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0)
  return Tensor(x.unsqueeze(0).numpy() < length.unsqueeze(1).numpy())

class TextEncoder:
  def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
               p_dropout):
    self.n_vocab, self.out_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout = n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
    self.emb = Embedding(n_vocab, hidden_channels)
    # TODO: normal weight initialisation,  nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
    self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = (self.emb(x) * math.sqrt(self.hidden_channels)).transpose(1, -1)  # [b, t, h] -transpose-> [b, h, t]
    x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).cast(x.dtype)  # TODO: verify this cast works
    x = self.encoder.forward(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = stats.split(self.out_channels, dim=1)
    return x, m, logs, x_mask

class WN:
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    assert (kernel_size % 2 == 1)
    self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels, self.p_dropout = hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout
    self.in_layers, self.res_skip_layers = [], []
    if gin_channels != 0:
      self.cond_layer = Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
      # TODO: self.cond_layer = utils.weight_norm(cond_layer, name='weight')
    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                        dilation=dilation, padding=padding)
      # TODO: in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)
      res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
      res_skip_layer = Conv1d(hidden_channels, res_skip_channels, 1)
      # TODO: res_skip_layer = utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = Tensor.zeros_like(x)
    n_channels_tensor = Tensor([self.hidden_channels], dtype=dtypes.int32)  # TODO: same behaviour as IntTensor

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = Tensor.zeros_like(x_in)

      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
      acts = acts.dropout(self.p_dropout)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:, :self.hidden_channels, :]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    pass  # TODO: remove_weight_norm
  # if self.gin_channels != 0: torch.nn.utils.remove_weight_norm(self.cond_layer)
  # for l in self.in_layers: torch.nn.utils.remove_weight_norm(l)
  # for l in self.res_skip_layers: torch.nn.utils.remove_weight_norm(l)


class ResBlock1:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
    super(ResBlock1, self).__init__()
    self.convs1 = [
      # weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
      # weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
      # weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
    ]
    # self.convs1.apply(init_weights)
    self.convs2 = [
      #     weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
      #     weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
      #     weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
    ]
    # self.convs2.apply(init_weights)

  def forward(self, x: Tensor, x_mask=None):
    for c1, c2 in zip(self.convs1, self.convs2):
      xt = x.leakyrelu(LRELU_SLOPE)
      xt = c1(xt if x_mask is None else xt * x_mask).leaky_relu(LRELU_SLOPE)
      x = c2(xt if x_mask is None else xt * x_mask) + x
    return x if x_mask is None else x * x_mask

  def remove_weight_norm(self): pass  # TODO: implement
  # for l in self.convs1: remove_weight_norm(l)
  # for l in self.convs2: remove_weight_norm(l)


class ResBlock2:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
    self.convs = [
      # weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
      # weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))
    ]
    # self.convs.apply(init_weights)

  def forward(self, x, x_mask=None):
    for c in self.convs:
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c(xt if x_mask is None else xt * x_mask)
      x = xt + x
    return x if x_mask is None else x * x_mask

  def remove_weight_norm(self): pass  # TODO: implement
  # for l in self.convs: remove_weight_norm(l)


class PosteriorEncoder:
  def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
    self.in_channels, self.out_channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels = in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels
    self.pre = Conv1d(in_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).cast(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc.forward(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = stats.split(self.out_channels, dim=1)
    z = (m + Tensor.randn(m.shape, m.dtype) * logs.exp()) * x_mask
    return z, m, logs, x_mask


class Generator:
  def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
               upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
    super(Generator, self).__init__()
    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups = []
    # TODO: weight norm, ConvTranspose1d
    # for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
    #     self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),k, u, padding=(k-u)//2)))
    self.resblocks = []
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)): self.resblocks.append(
        resblock(ch, k, d))
    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    # TODO: self.ups.apply(init_weights)
    if gin_channels != 0: self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

  def forward(self, x, g=None):
    x = self.conv_pre(x)
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x, xs = self.ups[i](x.leaky_relu(LRELU_SLOPE)), None
      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i * self.num_kernels + j].forward(x)
        else:
          xs += self.resblocks[i * self.num_kernels + j].forward(x)
      x = xs / self.num_kernels
    return self.conv_post(x.leaky_relu()).tanh()

  def remove_weight_norm(self):
    print('Removing weight norm...')
    # TODO: for l in self.ups: remove_weight_norm(l)
    for l in self.resblocks: l.remove_weight_norm()

class SynthesizerTrn:  # Synthesizer for Training
  def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, use_sdp=True):
    self.n_vocab, self.spec_channels, self.inter_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.segment_size, self.n_speakers, self.gin_channels, self.use_sdp = n_vocab, spec_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, segment_size, n_speakers, gin_channels, use_sdp
    self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
                             p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                         upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
