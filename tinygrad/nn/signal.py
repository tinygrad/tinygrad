import math
from tinygrad.tensor import Tensor, dtypes
from tinygrad.helpers import make_pair
from scipy.signal import get_window
import numpy as np
import math
from typing import List


def pad_center(data, size, axis=-1, **kwargs):
  kwargs.setdefault("mode", "constant")
  n = data.shape[axis]
  lpad = int((size - n) // 2)
  lengths = [(0, 0)] * data.ndim
  lengths[axis] = (lpad, int(size - n - lpad))
  assert lpad <= 0
  return np.pad(data, lengths, **kwargs)


def create_fourier_kernels(
    n_fft,
    win_length=None,
    freq_bins=None,
    fmin=50,
    fmax=6000,
    sr=44100,
    freq_scale="linear",
    window="hann",
):
  if freq_bins == None:
    freq_bins = n_fft // 2 + 1
  if win_length == None:
    win_length = n_fft

  s = np.arange(0, n_fft, 1.0)
  wsin = np.empty((freq_bins, 1, n_fft))
  wcos = np.empty((freq_bins, 1, n_fft))
  start_freq = fmin
  end_freq = fmax
  bins2freq = []
  binslist = []

  # Choosing window shape
  window_mask = get_window(window, int(win_length), fftbins=True)
  window_mask = pad_center(window_mask, n_fft)

  if freq_scale == "linear":
    start_bin = start_freq * n_fft / sr
    scaling_ind = (end_freq - start_freq) * (n_fft / sr) / freq_bins

    for k in range(freq_bins):  # Only half of the bins contain useful info
      bins2freq.append((k * scaling_ind + start_bin) * sr / n_fft)
      binslist.append((k * scaling_ind + start_bin))
      wsin[k, 0, :] = np.sin(2 * np.pi * (k * scaling_ind + start_bin) * s /
                             n_fft)
      wcos[k, 0, :] = np.cos(2 * np.pi * (k * scaling_ind + start_bin) * s /
                             n_fft)

  elif freq_scale == "log":
    start_bin = start_freq * n_fft / sr
    scaling_ind = np.log(end_freq / start_freq) / freq_bins

    for k in range(freq_bins):  # Only half of the bins contain useful info
      # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
      bins2freq.append(np.exp(k * scaling_ind) * start_bin * sr / n_fft)
      binslist.append((np.exp(k * scaling_ind) * start_bin))
      wsin[k, 0, :] = np.sin(2 * np.pi *
                             (np.exp(k * scaling_ind) * start_bin) * s / n_fft)
      wcos[k, 0, :] = np.cos(2 * np.pi *
                             (np.exp(k * scaling_ind) * start_bin) * s / n_fft)

  elif freq_scale == "log2":
    start_bin = start_freq * n_fft / sr
    scaling_ind = np.log2(end_freq / start_freq) / freq_bins

    for k in range(freq_bins):  # Only half of the bins contain useful info
      # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
      bins2freq.append(2**(k * scaling_ind) * start_bin * sr / n_fft)
      binslist.append((2**(k * scaling_ind) * start_bin))
      wsin[k, 0, :] = np.sin(2 * np.pi * (2**(k * scaling_ind) * start_bin) *
                             s / n_fft)
      wcos[k, 0, :] = np.cos(2 * np.pi * (2**(k * scaling_ind) * start_bin) *
                             s / n_fft)

  elif freq_scale == "no":
    for k in range(freq_bins):  # Only half of the bins contain useful info
      bins2freq.append(k * sr / n_fft)
      binslist.append(k)
      wsin[k, 0, :] = np.sin(2 * np.pi * k * s / n_fft)
      wcos[k, 0, :] = np.cos(2 * np.pi * k * s / n_fft)
  else:
    raise NotImplementedError(
        'Please select the correct frequency scale: "linear", "log", "log2", "no"'
    )
  return (
      wsin.astype(np.float32),
      wcos.astype(np.float32),
      bins2freq,
      binslist,
      window_mask.astype(np.float32),
  )


class STFT:
  def __init__(self,
               n_fft=128,
               win_length=128,
               freq_bins=None,
               hop_length=64,
               window="hann",
               freq_scale="no",
               center=True,
               fmin=50,
               fmax=6000,
               sr=16000,
               trainable=False,
               eps=1e-10):

    super().__init__()

    # Trying to make the default setting same as librosa
    if win_length == None:
      win_length = n_fft
    if hop_length == None:
      hop_length = int(win_length // 4)

    self.trainable = trainable
    self.stride = hop_length
    self.center = center
    self.n_fft = n_fft
    self.freq_bins = freq_bins
    self.trainable = trainable
    self.pad_amount = self.n_fft // 2
    self.window = window
    self.win_length = win_length
    self.eps = eps

    # Create filter windows for stft
    (
        kernel_sin,
        kernel_cos,
        self.bins2freq,
        self.bin_list,
        window_mask,
    ) = create_fourier_kernels(
        n_fft,
        win_length=win_length,
        freq_bins=freq_bins,
        window=window,
        freq_scale=freq_scale,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )

    kernel_sin = Tensor(kernel_sin, dtype=dtypes.float32)
    kernel_cos = Tensor(kernel_cos, dtype=dtypes.float32)

    self.kernel_sin_inv = kernel_sin.cat(-kernel_sin[1:-1].flip(0),
                                         dim=0).unsqueeze(-1)
    self.kernel_cos_inv = kernel_cos.cat(kernel_cos[1:-1].flip(0),
                                         dim=0).unsqueeze(-1)

    # Applying window functions to the Fourier kernels
    window_mask = Tensor(window_mask)
    self.wsin = kernel_sin * window_mask
    self.wcos = kernel_cos * window_mask
    self.window_mask = window_mask.unsqueeze(0).unsqueeze(-1)

  def __call__(self, x, inverse=False, *args, **kwargs):
    return self.forward(x, *args, **kwargs) if not inverse else self.inverse(
        x, *args, **kwargs)

  def forward(self, x, return_spec=False):
    self.num_samples = x.shape[-1]

    assert len(x.shape) == 2, "Input shape must be (batch, len) "
    if self.center:
      x = x.pad(((0, 0), (self.pad_amount, self.pad_amount)), )
    x = x[:, None, :]

    spec_imag = x.conv2d(self.wsin, stride=self.stride)[:, :self.freq_bins, :]
    spec_real = x.conv2d(self.wcos, stride=self.stride)[:, :self.freq_bins, :]
    if return_spec:
      spec = (spec_real.pow(2) + spec_imag.pow(2)).sqrt()
      spec = (spec + self.eps) if self.trainable else spec
      return spec
    else:
      return Tensor.stack((spec_real, -spec_imag), -1)

  def inverse(self, X, onesided=True, length=None):
    assert len(X.shape) == 4, (
        "Tensor must be in the shape of (batch, freq_bins, timesteps, 2)."
        "Where last dim is real and imaginary number dim")
    # If the input spectrogram contains only half of the n_fft
    # Use extend_fbins function to get back another half
    if onesided:
      # Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
      # reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
      X_ = X[:, 1:-1].flip(1)
      X_upper1 = X_[:, :, :, 0]
      X_upper2 = -X_[:, :, :, 1]
      X_upper = Tensor.stack([X_upper1, X_upper2], dim=3)
      X = X.cat(X_upper, dim=1)
    X_real, X_imag = X[:, :, :, 0][:, None], X[:, :, :, 1][:, None]
    a1 = X_real.conv2d(self.kernel_cos_inv, stride=(1, 1))
    b2 = X_imag.conv2d(self.kernel_sin_inv, stride=(1, 1))
    real = a1 - b2
    real = real[:, :, 0, :] * self.window_mask
    real = real / self.n_fft

    # Overlap and Add algorithm to connect all the frames
    n_fft = real.shape[1]
    output_len = n_fft + self.stride * (real.shape[2] - 1)
    real = col2im(real, (1, output_len),
                  kernel_size=(1, n_fft),
                  stride=(self.stride, self.stride),
                  dilation=(1, 1),
                  padding=(0, 0)).flatten(1)
    win = self.window_mask.flatten()
    n_frames = X.shape[2]
    win_stacks = win[:, None].repeat((1, n_frames))[None, :]
    output_len = win_stacks.shape[1] + self.stride * (win_stacks.shape[2] - 1)
    w_sum = col2im(win_stacks**2, (1, output_len),
                   kernel_size=(1, n_fft),
                   stride=(self.stride, self.stride),
                   dilation=(1, 1),
                   padding=(0, 0)).flatten(1)
    real = real / w_sum

    if length is None:
      if self.center:
        real = real[:, self.pad_amount:-self.pad_amount]
    else:
      if self.center:
        real = real[:, self.pad_amount:self.pad_amount + length]
      else:
        real = real[:, :length]
    return real


def col2im(input: Tensor,
           output_size: List[int],
           kernel_size: List[int],
           dilation: List[int],
           padding: List[int],
           stride: List[int],
           dtype=dtypes.float32) -> Tensor:
  assert len(kernel_size) == 2, "only 2D kernel supported"
  assert len(dilation) == 2, "only 2D dilation supported"
  assert len(padding) == 2, "only 2D padding supported"
  assert len(stride) == 2, "only 2D stride supported"

  assert all(e >= 0 for e in kernel_size), "kernel_size must be positive"
  assert all(e >= 0 for e in dilation), "dilation must be positive"
  assert all(e >= 0 for e in padding), "padding must be positive"
  assert all(e >= 0 for e in stride), "stride must be positive"
  assert all(e >= 0 for e in output_size), "output_size must be positive"

  shape = input.shape
  ndim = len(shape)
  assert ndim in (2, 3) and all(d != 0 for d in shape[-2:]), (
      f"Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
      f"and non-zero dimensions, but got: {tuple(shape)}", )
  prod_kernel_size = kernel_size[0] * kernel_size[1]
  assert shape[-2] % prod_kernel_size == 0, (
      f"Expected size of input's first non-batch dimension to be divisible by the "
      f"product of kernel_size, but got input.shape[-2] = {shape[-2]} and "
      f"kernel_size={kernel_size}", )
  col = [
      1 + (out + 2 * pad - dil * (ker - 1) - 1) // st for out, pad, dil, ker,
      st in zip(output_size, padding, dilation, kernel_size, stride)
  ]
  L = col[0] * col[1]
  assert shape[-1] == L, (
      f"Given output_size={output_size}, kernel_size={kernel_size}, "
      f"dilation={dilation}, padding={padding}, stride={stride}, "
      f"expected input.size(-1) to be {L} but got {shape[-1]}.", )
  assert L > 0, (
      f"Given output_size={output_size}, kernel_size={kernel_size}, "
      f"dilation={dilation}, padding={padding}, stride={stride}, "
      f"expected input.size(-1) to be {L} but got {shape[-1]}.", )
  batched_input = ndim == 3
  if not batched_input:
    input = input.unsqueeze(0)

  shape = input.shape

  out_h, out_w = output_size
  stride_h, stride_w = stride
  padding_h, padding_w = padding
  dilation_h, dilation_w = dilation
  kernel_h, kernel_w = kernel_size

  input = input.reshape([shape[0], shape[1] // prod_kernel_size] +
                        list(kernel_size) + col)
  input = input.permute(0, 1, 2, 4, 3, 5)

  def indices_along_dim(input_d, kernel_d, dilation_d, padding_d, stride_d):
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)
    blocks_d_indices = np.arange(0, blocks_d, stride_d)[None, ...]
    kernel_grid = np.arange(0, kernel_d * dilation_d, dilation_d)[..., None]
    return blocks_d_indices + kernel_grid

  indices_row = indices_along_dim(out_h, kernel_h, dilation_h, padding_h,
                                  stride_h)
  for _ in range(4 - len(indices_row.shape)):
    indices_row = indices_row[..., None]
  indices_col = indices_along_dim(out_w, kernel_w, dilation_w, padding_w,
                                  stride_w)

  output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
  output = np.zeros([shape[0], shape[1] // math.prod(kernel_size)] +
                    output_padded_size).astype(dtype.np)
  output = Tensor(output)
  output_shape = output.shape
  input = input.reshape(input.shape[0], -1)
  output = output.reshape(output.shape[0], -1)
  indices_col = indices_col.flatten()
  # TODO: SLOW, vectorize this
  for i, idx_col in enumerate(indices_col):
    out = Tensor.zeros_like(output)
    idxs = (Tensor.arange(math.prod(out.shape[1:]))).reshape(out.shape[1:])[None,:].expand(out.shape[0],out.shape[1])
    mask = idxs.eq(idx_col)
    masked = input[:,i:i+1]*mask
    output = output + masked
  output = output.reshape(output_shape)
  output = output[:, :, padding_w:(-padding_w if padding_w != 0 else None),padding_h:(-padding_h if padding_h != 0 else None)]

  if not batched_input:
    output = output.squeeze(0)

  return output
