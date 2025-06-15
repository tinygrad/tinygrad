import unittest
import numpy as np
import torch
import importlib.util

from tinygrad import Tensor, Variable, Device
from tinygrad.helpers import OSX


class TestSample(unittest.TestCase):
  def test_hann(self):
    ref = torch.hann_window(256, False)
    result = Tensor.hann_tg(256, False)
    np.testing.assert_allclose(result.numpy(), ref.numpy(), atol=1e-6, rtol=1e-5)

  def test_hann_periodic(self):
    ref = torch.hann_window(256)
    result = Tensor.hann_tg(256)
    np.testing.assert_allclose(result.numpy(), ref.numpy(), atol=1e-7)

  def test_stft(self):
    N_FFT = 400
    HOP_LENGTH = 160
    BS = 16

    Tensor.manual_seed(42)
    X = Tensor.rand(BS, 2400).realize()
    reference = torch.stft(torch.Tensor(X.numpy()), N_FFT, HOP_LENGTH, center=False, return_complex=True, window=torch.hann_window(N_FFT))
    reference = reference.abs()
    result = Tensor.stft_full(X, N_FFT, HOP_LENGTH, (0, 0))
    np.testing.assert_allclose(result.numpy(), reference.numpy(), atol=1e-4, rtol=1e-2)

  @unittest.skipUnless(importlib.util.find_spec("librosa") is not None)
  def test_stft_librosa(self):
    import librosa
    N_FFT = 400
    HOP_LENGTH = 160
    BS = 16

    Tensor.manual_seed(42)
    X = Tensor.rand(BS, 2400).realize()
    reference = librosa.stft(X.numpy(), n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, window="hann", dtype=np.csingle)
    reference = np.abs(reference)
    result = Tensor.stft_full(X, N_FFT, HOP_LENGTH, (0, 0))
    # NOTE(irwin): why do we pass at atol=1e-7 here? it's much lower with librosa than torch.stft
    np.testing.assert_allclose(result.numpy(), reference, atol=1e-7, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()