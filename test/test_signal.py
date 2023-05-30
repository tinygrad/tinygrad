import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.signal import STFT
import numpy as np

class TestSignal(unittest.TestCase):
  def test_fft(self):
    data = np.arange(5)
    np_out = np.fft.fft(data)
    np_out = np.stack(np_out.real, np_out.imag, -1)
    # tg_out = fft.fft(Tensor(data))
    
  def test_stft(self):
    x = Tensor.ones(5,2000)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    y = stft_layer(x)
    x_hat = stft_layer(y, inverse=True) 

    import torch
    from nnAudio.features import STFT as STFT2
    m = STFT2(pad_mode="constant",n_fft=128, win_length=128, hop_length=64, iSTFT=True)
    y2 = m(torch.from_numpy(x.numpy())).numpy()
    x_hat2 = m.inverse(torch.from_numpy(y2)).numpy()
    assert np.allclose(y,y2, atol=.5e-5)

if __name__ == '__main__':
  unittest.main()