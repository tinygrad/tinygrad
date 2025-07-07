from typing import Optional

import numpy as np
from tinygrad import Tensor

from tinygrad.dtype import DTypeLike, dtypes
import math


def stft(x:Tensor, weight:Tensor, n_fft, stride, pad, pad_mode="constant")->Tensor:
  cutoff = int(n_fft // 2) + 1
  x_padded = x.pad(pad, mode=pad_mode)
  stft_raw = x_padded.unsqueeze(1).conv2d(weight, stride=stride)

  # NOTE(irwin): magnitudes only atm
  magnitudes = (stft_raw[:, :cutoff, :]**2 + stft_raw[:, cutoff:, :]**2).sqrt()
  return magnitudes

def hann_window(N:int, periodic=True) -> Tensor:
  M = N+(periodic*1)
  return ((1.0 - (Tensor.arange(M) * 2.0 * math.pi / (M - 1)).cos()) * 0.5)[:N]

def make_stft_basis_buffers(n_fft:int, window:Tensor) -> Tensor:
  return Tensor.cat(*make_basis_buffers(n_fft, Tensor.arange((n_fft // 2) + 1)[None].T, window)).reshape(n_fft+2, 1, n_fft)

class STFT:
  def __init__(self, n_fft:int, stride:int, pad:tuple[int, int], window="hann", pad_mode="constant"):
    assert window == "hann", "other window types not implemented yet"
    self.n_fft = n_fft
    self.stride = stride
    self.pad = pad
    self.pad_mode = pad_mode
    self.forward_basis_buffers = make_stft_basis_buffers(n_fft, hann_window(n_fft)).realize()

  def __call__(self, waveforms):
    return self.forward(waveforms)

  def forward(self, x:Tensor) -> Tensor:
    x = x.reshape(-1, x.shape[-1])
    spec = stft(x, self.forward_basis_buffers, self.n_fft, self.stride, self.pad, self.pad_mode)
    return spec

def make_basis_buffers(N_FFT: int, k_freq_bin: int|Tensor, window:Tensor) -> tuple[Tensor, Tensor]:
  n = Tensor.arange(N_FFT)
  angle = 2 * math.pi * k_freq_bin * n / N_FFT

  w = window
  cos_basis = w * angle.cos()
  # NOTE(irwin): negate sin_basis to match torch
  sin_basis = w * -angle.sin()
  return cos_basis, sin_basis

def stft_full(x:Tensor, n_fft:int, stride:int, pad:tuple[int, int], window="hann", pad_mode="constant") -> Tensor:
  assert window == "hann", "other window types not implemented yet"
  bb = make_stft_basis_buffers(n_fft, hann_window(n_fft))
  res = stft(x, bb, n_fft, stride, pad, pad_mode)
  return res

# rewritten from numpy
def rfftfreq(n, d=1.0, device=None):
  val = 1.0 / (n * d)
  N = n // 2 + 1
  results = Tensor.arange(N, device=device)
  return results * val

# just like in librosa
def fft_frequencies(sr:float, n_fft:int):
  return rfftfreq(n=n_fft, d=1.0 / sr)

def hz_to_mel(freq:Tensor)->Tensor:
  # Fill in the linear part
  f_min = 0.0
  f_sp = 200.0 / 3

  mels = (freq - f_min) / f_sp

  # Fill in the log-scale part

  min_log_hz = 1000.0  # beginning of log region (Hz)

  mask = freq >= min_log_hz
  return mask.where(((min_log_hz - f_min) / f_sp) + (freq / min_log_hz).log() / (Tensor(6.4).log() / 27.0), mels)

def mel_to_hz(mels:Tensor)->Tensor:
  # Fill in the linear scale
  f_min = 0.0
  f_sp = 200.0 / 3
  freqs = f_min + f_sp * mels

  # And now the nonlinear scale
  min_log_hz = 1000.0  # beginning of log region (Hz)
  min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
  logstep = Tensor(6.4).log() / 27.0  # step size for log region

  # If we have vector data, vectorize
  log_t = mels >= min_log_mel
  freqs = log_t.where(min_log_hz * ((logstep * (mels - min_log_mel)).exp()), freqs)

  return freqs

def mel_frequencies(n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0)->Tensor:
  # 'Center freqs' of mel bands - uniformly spaced between limits
  min_max_mel = hz_to_mel(Tensor([fmin, fmax]))

  mels = Tensor.linspace(min_max_mel[0], min_max_mel[1], n_mels)
  hz = mel_to_hz(mels)
  return hz

def mel(
  *,
  sr: float,
  n_fft: int,
  n_mels: int = 128,
  fmin: float = 0.0,
  fmax: Optional[float] = None,
  dtype: DTypeLike = dtypes.default_float,
) -> Tensor:

  if fmax is None:
    fmax = float(sr) / 2

  # Initialize the weights
  n_mels = int(n_mels)
  weights = Tensor.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype).contiguous()

  # Center freqs of each FFT bin
  fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

  # 'Center freqs' of mel bands - uniformly spaced between limits
  mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

  fdiff = mel_f[1:] - mel_f[:-1]
  ramps = mel_f[None].T.expand(-1, fftfreqs.shape[-1]) - fftfreqs

  lower = -ramps[:n_mels] / fdiff[:n_mels][None].T
  upper = ramps[2:n_mels + 2] / fdiff[1:n_mels + 1][None].T
  weights = lower.minimum(upper).maximum(0)

  # Slaney-style mel is scaled to be approx constant energy per channel
  enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
  weights *= enorm[:, None]

  return weights

def resample(x, L, M, num_taps=64):
  fc = 0.5 / max(L, M)
  t = Tensor.arange(-num_taps//2, num_taps//2 + 1)
  # NOTE(irwin): poor man's sinc
  h:Tensor = (t * fc * np.pi).sin() / (t * fc * np.pi)
  # hack fix NaN
  h[num_taps//2] = 1.0
  h *= 0.54 - 0.46 * (2 * np.pi * (t + num_taps//2) / num_taps).cos()  # hamming
  h /= h.sum()
  # TODO(irwin): contiguous sped up things before replacing decimation with stride=M, retest if still relevant
  upsampled = x.reshape(-1, 1, x.shape[-1]).pad((None, (0, L-1), None)).transpose(1, 2).flatten(1).unsqueeze(1)
  # upsampled = xx.cat(Tensor.zeros(L-1, *x.shape)).T.flatten().contiguous()
  padding = (len(h) // 2)
  filtered = upsampled.conv2d(h.reshape(1, 1, -1), stride=M, padding=padding).flatten(1)
  return filtered

def next_power_of_2(n):
  if n <= 0:
    return 1
  return 1 << (n - 1).bit_length()

def resample2(samples, source, target):
  gcd = math.gcd(source, target)
  M = source // gcd
  L = target // gcd
  # NOTE(irwin): overkill but works
  taps = next_power_of_2(max(M, L))*2
  #print(M, L, taps)
  return resample(samples, L, M, taps)

def resample_batched(samples, source, target):
  count = samples.shape[-1]
  rbs = source*10
  samples = samples.pad(((0, math.ceil(count / rbs) * rbs - count))).reshape(-1, rbs)
  resampled = resample2(samples, source, target)

  return resampled[:int((count / source) * target)]