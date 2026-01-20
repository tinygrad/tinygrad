from typing import Optional, Sequence
from tinygrad import Tensor
from tinygrad.dtype import DTypeLike, dtypes
from tinygrad.uop.ops import sint
import math


def stft(
  x: Tensor, weight: Tensor, n_fft: int, stride: int, pad: Sequence[sint] | Sequence[tuple[sint, sint] | None], pad_mode: str = "constant"
) -> Tensor:
  cutoff = int(n_fft // 2) + 1
  x_padded = x.pad(pad, mode=pad_mode)
  stft_raw = x_padded.unsqueeze(1).conv2d(weight, stride=stride)

  # magnitudes only atm
  magnitudes = (stft_raw[:, :cutoff, :] ** 2 + stft_raw[:, cutoff:, :] ** 2).sqrt()
  return magnitudes


def hann_window(N: int, periodic=True) -> Tensor:
  M = N + (periodic * 1)
  return ((1.0 - (Tensor.arange(M) * 2.0 * math.pi / (M - 1)).cos()) * 0.5)[:N]


def make_stft_basis_buffers(n_fft: int, window: Tensor) -> Tensor:
  return Tensor.cat(*make_basis_buffers(n_fft, Tensor.arange((n_fft // 2) + 1)[None].T, window)).reshape(n_fft + 2, 1, n_fft)


def make_basis_buffers(N_FFT: int, k_freq_bin: int | Tensor, window: Tensor) -> tuple[Tensor, Tensor]:
  n = Tensor.arange(N_FFT)
  angle = 2 * math.pi * k_freq_bin * n / N_FFT

  w = window
  cos_basis = w * angle.cos()
  sin_basis = w * -angle.sin()  # negate sin_basis to match torch
  return cos_basis, sin_basis


def stft_full(x: Tensor, n_fft: int, stride: int, pad: tuple[int, int], window="hann", pad_mode="constant") -> Tensor:
  assert window == "hann", "other window types not implemented yet"
  bb = make_stft_basis_buffers(n_fft, hann_window(n_fft))
  res = stft(x, bb, n_fft, stride, pad, pad_mode)
  return res


# rewritten from numpy
def rfftfreq(n: int, d: float = 1.0, device=None) -> Tensor:
  val = 1.0 / (n * d)
  N = n // 2 + 1
  results = Tensor.arange(N, device=device)
  return results * val

# just like in librosa
def fft_frequencies(sr: float, n_fft: int) -> Tensor:
  return rfftfreq(n=n_fft, d=1.0 / sr)

def hz_to_mel(freq: Tensor) -> Tensor:
  # linear part
  f_min = 0.0
  f_sp = 200.0 / 3
  mels = (freq - f_min) / f_sp

  # log-scale part
  min_log_hz = 1000.0  # beginning of log region (Hz)
  mask = freq >= min_log_hz
  return mask.where(((min_log_hz - f_min) / f_sp) + (freq / min_log_hz).log() / (math.log(6.4) / 27.0), mels)

def mel_to_hz(mels: Tensor) -> Tensor:
  # linear scale
  f_min = 0.0
  f_sp = 200.0 / 3
  freqs = f_min + f_sp * mels

  # nonlinear scale
  min_log_hz = 1000.0  # beginning of log region (Hz)
  min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
  logstep = math.log(6.4) / 27.0  # step size for log region

  log_t = mels >= min_log_mel
  freqs = log_t.where(min_log_hz * ((logstep * (mels - min_log_mel)).exp()), freqs)
  return freqs

def mel_frequencies(n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0) -> Tensor:
  # center freqs of mel bands - uniformly spaced between limits
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

  n_mels = int(n_mels)

  fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)  # center freqs of each FFT bin
  mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)  # center freqs of mel bands

  fdiff = mel_f[1:] - mel_f[:-1]
  ramps = mel_f[None].T.expand(-1, fftfreqs.shape[-1]) - fftfreqs

  lower = -ramps[:n_mels] / fdiff[:n_mels][None].T
  upper = ramps[2 : n_mels + 2] / fdiff[1 : n_mels + 1][None].T
  weights = lower.minimum(upper).maximum(0)

  # Slaney-style mel is scaled to be approx constant energy per channel
  enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
  weights *= enorm[:, None]

  return weights
