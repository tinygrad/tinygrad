from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, getenv
from vits import ResidualCouplingBlock, PosteriorEncoder, Encoder, ResBlock1, ResBlock2, LRELU_SLOPE, sequence_mask, split, download_if_not_present
import numpy as np
import logging
import sys
from pathlib import Path

# original code: https://github.com/svc-develop-team/so-vits-svc

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

# TODO: add speech encoder ContentVec256L9 (most weights trained on 4.0 architecture require this)
# TODO: add Synthesizer
# TODO: add Volume Extractor
class Svc():
  def __init__(self, net_g_path): pass

# TODO: depthwise conv (optional) #self.use_depthwise_conv = use_depthwise_conv
# TODO: f0 decoder (optional for infer) #self.use_automatic_f0_prediction = use_automatic_f0_prediction
# TODO: flow_share_parameter
class Synthesizer:
  def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, ssl_dim, n_speakers, sampling_rate=44100,
               vol_embedding=False,
               #use_depthwise_conv=False,
               #use_automatic_f0_prediction=True,
               #flow_share_parameter=False,
               n_flow_layer=4,
               **kwargs):
    self.spec_channels, self.inter_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.segment_size, self.n_speakers, self.gin_channels, self.vol_embedding = spec_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, segment_size, n_speakers, gin_channels, vol_embedding
    self.emb_g = nn.Embedding(n_speakers, gin_channels)
    if vol_embedding:
      self.emb_vol = nn.Linear(1, hidden_channels)
    self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)
    self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.dec = Generator(sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flows=n_flow_layer, gin_channels=gin_channels) # TODO: This is already defined in super Synthesizer. Maybe adjust super constructor? Only diff is n_flows param
    self.emb_uv = nn.Embedding(vocab_size=2, embed_size=hidden_channels)
    self.character_mix = False
  def EnableCharacterMix(self, n_speakers_map, device):
    self.speaker_map = Tensor.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
    for i in range(n_speakers_map):
      self.speaker_map[i] = self.emb_g(Tensor([[i]], dtype=dtypes.int64).to(device))
    self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
    self.character_mix = True
  def infer(self, c, f0, uv, g=None, noise_scale=0.35, seed=52468, vol=None):
    Tensor.manual_seed(getenv('SEED', seed))
    c_lengths = (Tensor.ones([c.shape[0]]) * c.shape[-1]).to(c.device)
    if self.character_mix and g.shape[0] > 1:  # [N, S]  *  [S, B, 1, H]
      g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
      g = g * self.speaker_map  # [N, S, B, 1, H]
      g = g.sum(dim=1) # [N, 1, B, 1, H]
      g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
    else:
      if g.dim() == 1:
        g = g.unsqueeze(0)
      g = self.emb_g(g).transpose(1, 2)
    x_mask = sequence_mask(c_lengths, c.shape[2]).unsqueeze(1).cast(c.dtype)
    vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0
    x = self.pre(c) * x_mask + self.emb_uv(uv.cast(dtypes.int64)).transpose(1, 2) + vol
    z_p, _, _, c_mask = self.enc_p.forward(x, x_mask, f0=f0_to_coarse(f0), noise_scale=noise_scale)
    z = self.flow.forward(z_p, c_mask, g=g, reverse=True)
    o = self.dec.forward(z * c_mask, g=g, f0=f0)
    return o,f0

def randn_like(x): return Tensor.randn(x.shape, dtype=x.dtype).to(device=x.device)

class TextEncoder:
  def __init__(self, out_channels, hidden_channels, kernel_size, n_layers, gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None):
    self.out_channels, self.hidden_channels, self.kernel_size, self.n_layers, self.gin_channels = out_channels, hidden_channels, kernel_size, n_layers, gin_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.f0_emb = nn.Embedding(256, hidden_channels)  # n_vocab = 256
    self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
  def forward(self, x, x_mask, f0=None, noise_scale=1):
    x = x + self.f0_emb(f0).transpose(1, 2)
    x = self.encoder.forward(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = split(stats, self.out_channels, dim=1)
    z = (m + randn_like(m) * logs.exp() * noise_scale) * x_mask
    return z, m, logs, x_mask

# TODO: this is tragic. remove this
import torch
class Upsample:
  def __init__(self, scale_factor):
    self.scale_factor=scale_factor
    self.torch_ups = torch.nn.Upsample(scale_factor=scale_factor)
  def forward(self, x):
    return Tensor(self.torch_ups(torch.from_numpy(x.numpy())).numpy()).to(x.device)

class SineGen():
  def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voice_threshold=0, flag_for_pulse=False):
    self.sine_amp, self.noise_std, self.harmonic_num, self.sampling_rate, self.voiced_threshold, self.flag_for_pulse = sine_amp, noise_std, harmonic_num, samp_rate, voice_threshold, flag_for_pulse
    self.dim = self.harmonic_num + 1
  def _f02uv(self, f0):
    return (f0 > self.voiced_threshold).cast(dtypes.float32)  #generate uv signal
  def _f02sine(self, f0_values):
    def padDiff(x : Tensor): return (x.pad2d((0,0,-1,1)) - x).pad2d((0,0,0,-1))
    rad_values = (f0_values / self.sampling_rate) % 1  # convert to F0 in rad
    rand_ini = Tensor.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)  # initial phase noise
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    # TODO: optional flag_for_pulse. Not required for now.
    tmp_over_one = rad_values.cumsum(1) % 1
    tmp_over_one_idx = padDiff(tmp_over_one) < 0
    cumsum_shift = Tensor.zeros_like(rad_values)
    cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
    sines = ((rad_values + cumsum_shift).cumsum(1) * 2 * np.pi).sin()
    return sines
  def forward(self, f0, upp=None):
    fn = f0.mul(Tensor([[range(1, self.harmonic_num + 2)]], dtype=dtypes.float32).to(f0.device))
    sine_waves = self._f02sine(fn) * self.sine_amp  #generate sine waveforms
    uv = self._f02uv(f0)  # generate uv signal
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * randn_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise

# TODO: verify tanh
class SourceHnNSF:
  def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
    self.sine_amp, self.noise_std = sine_amp, add_noise_std
    self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
    self.l_linear = nn.Linear(harmonic_num + 1, 1)
  def forward(self, x, upp=None):
    sine_waves, uv, _ = self.l_sin_gen.forward(x, upp)
    sine_merge = self.l_linear(sine_waves.cast(self.l_linear.weight.dtype)).tanh()
    noise = randn_like(uv) * self.sine_amp / 3
    return sine_merge, noise, uv

# TODO: most of the hifigan in standard vits is reused here
class Generator:
  def __init__(self, sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels):
    self.sampling_rate, self.inter_channels, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.gin_channels = sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels
    self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
    self.conv_pre = nn.Conv1d(inter_channels, upsample_initial_channel, 7, 1, padding=3)
    self.f0_upsamp = Upsample(scale_factor=np.prod(upsample_rates))
    self.m_source = SourceHnNSF(sampling_rate, harmonic_num=8)
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups, self.noise_convs = [], []
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      c_curr = upsample_initial_channel//(2**(i+1))
      self.ups.append(nn.ConvTranspose1d(upsample_initial_channel//(2**i), c_curr, k, u, padding=(k-u)//2))
      if i + 1 < len(upsample_rates):  # TODO: make oneliner after debugging
        stride_f0 = np.prod(upsample_rates[i + 1:])
        self.noise_convs.append(nn.Conv1d(1, c_curr, stride_f0 * 2, stride_f0, (stride_f0+1) // 2))
      else:
        self.noise_convs.append(nn.Conv1d(1, c_curr, kernel_size=1))
    self.resblocks = []
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))
    self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    self.upp = np.prod(upsample_rates)
  def forward(self, x, f0, g=None):
    f0 = self.f0_upsamp.forward(f0[:, None]).transpose(1, 2)  # bs,n,t
    har_source, _, _ = self.m_source.forward(f0, self.upp)
    har_source = har_source.transpose(1, 2)
    x = self.conv_pre(x)
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x, xs = self.ups[i](x.leakyrelu(LRELU_SLOPE)), None
      x_source = self.noise_convs[i](har_source)
      x = x + x_source
      for j in range(self.num_kernels):
        if xs is None: xs = self.resblocks[i * self.num_kernels + j].forward(x)
        else: xs += self.resblocks[i * self.num_kernels + j].forward(x)
      x = xs / self.num_kernels
    return self.conv_post(x.leakyrelu()).tanh()

# TODO: this is ugly
def f0_to_coarse(f0 : Tensor):
  f0_mel = 1127 * (1 + f0 / 700).log()
  a = (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN)
  b = F0_MEL_MIN * a - 1.
  f0_mel = (f0_mel > 0).where(f0_mel * a - b, f0_mel)
  # TODO: use round() instead of ceil()
  f0_coarse = f0_mel.ceil().cast(dtype=dtypes.int64) # TODO: make oneliner after debugging
  f0_coarse = f0_coarse * (f0_coarse > 0)
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  f0_coarse = f0_coarse * (f0_coarse < F0_BIN)
  f0_coarse = f0_coarse + ((f0_coarse >= F0_BIN) * (F0_BIN - 1))
  return f0_coarse

"""
TODO: optional F0 decoder
class FFT():
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.proximal_bias, self.proximal_init = hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, proximal_bias, proximal_init
    #self.drop = nn.Dropout()
    #self.self_attn_layers = nn.
    self.attn_layers, self.norm_layers_0, self.ffn_layers, self.norm_layers_1 = []
    for _ in range(self.n_layers):
      self.attn_layers.append(
          MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init)
        )

class F0Decoder():
  def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=0):
    self.out_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.spk_channels = out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels
    self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
    self.decoder = None
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
    self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)
"""

SO_VITS_SVC_PATH = Path(__file__).parent.parent / "weights/So-VITS-SVC"
VITS_MODELS = { # config_path, weights_path, config_url, weights_url
  "saul_goodman" : (SO_VITS_SVC_PATH / "config_saul_gman.json", SO_VITS_SVC_PATH / "pretrained_saul_gman.pth", "https://huggingface.co/Amo/so-vits-svc-4.0_GA/resolve/main/ModelsFolder/Saul_Goodman_80000/config.json", "https://huggingface.co/Amo/so-vits-svc-4.0_GA/resolve/main/ModelsFolder/Saul_Goodman_80000/G_80000.pth")
}

if __name__=="__main__":
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  test_model = "saul_goodman"
  model_config = VITS_MODELS[test_model]
  
  config_path = model_config[0]
  download_if_not_present(config_path, model_config[2])
  
  model_path = model_config[1]
  download_if_not_present(model_path, model_config[3])
