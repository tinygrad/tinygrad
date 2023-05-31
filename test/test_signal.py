import unittest
import numpy as np
from tinygrad.tensor import Tensor
from examples.mlperf.audio_helpers import STFT
import numpy as np

class TestSignal(unittest.TestCase):
  def test_stft_reconstruction(self):
    # x = Tensor.ones(5,2000)
    x = Tensor.ones(2,300)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    y = stft_layer(x)
    x_hat = stft_layer(y, inverse=True)
    np.testing.assert_allclose(x[:, :x_hat.shape[1]].numpy(), x_hat.numpy(), atol=1e-3) # TODO: improve precision

  def test_stft_spectrogram(self):
    x = Tensor.ones(5,2000)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    spec = stft_layer(x, return_spec=True)
    np.testing.assert_((spec.numpy()>0).all())

if __name__ == '__main__':
  unittest.main()