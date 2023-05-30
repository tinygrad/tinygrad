import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.signal import STFT
import numpy as np

class TestSignal(unittest.TestCase):
  def test_stft_reconstruction(self):
    x = Tensor.ones(5,2000)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    y = stft_layer(x)
    x_hat = stft_layer(y, inverse=True) 
    assert np.allclose(x[:, :x_hat.shape[1]].numpy(), x_hat.numpy(), atol=1e-4) # TODO: improve precision
    spec = stft_layer(x, return_spec=True)
    assert (spec.numpy()>0).all()

  def test_stft_spectrogram(self):
    x = Tensor.ones(5,2000)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    spec = stft_layer(x, return_spec=True)
    assert (spec.numpy()>0).all()

if __name__ == '__main__':
  unittest.main()