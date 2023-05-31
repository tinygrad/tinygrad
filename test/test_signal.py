import unittest
import numpy as np
from tinygrad.tensor import Tensor
from examples.mlperf.audio_helpers import STFT, MelSpectrogram, MFCC
import numpy as np

class TestSignal(unittest.TestCase):
  def test_stft_reconstruction(self):
    x = Tensor.ones(2,300)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    y = stft_layer(x)
    x_hat = stft_layer(y, inverse=True)
    np.testing.assert_allclose(x[:, :x_hat.shape[1]].numpy(), x_hat.numpy(), atol=1e-3) # TODO: improve precision

  def test_stft_spectrogram(self):
    x = Tensor.ones(2,2000)
    stft_layer = STFT(n_fft=512, window="hann", hop_length=160, win_length=320)
    spec = stft_layer(x, return_spec=True)
    np.testing.assert_((spec.numpy()>0).all())
  
  def test_log_mel_spectrogram(self):
    x = Tensor.ones(2,2000)
    mel_spec_layer = MelSpectrogram(n_mels=80, n_fft=512, window="hann", hop_length=160, win_length=320,
                                    sr=16000, fmin=0, fmax=8000)
    mel_spec = mel_spec_layer(x, return_log=True)
    np.testing.assert_((mel_spec.numpy()>-25.).all())

if __name__ == '__main__':
  unittest.main()