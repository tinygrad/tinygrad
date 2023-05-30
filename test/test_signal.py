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
    x = Tensor.ones(5,2048)
    stft_layer = STFT(n_fft=128, win_length=128, hop_length=64)
    y = stft_layer(x)
    x_hat = stft_layer(y, inverse=True)
    breakpoint()
    
    # from nnAudio.features import STFT as STFT2
    # import torch
    # spec_imag = torch.nn.functional.conv1d(torch.from_numpy(x.numpy()), torch.from_numpy(self.wsin.numpy()), stride=self.stride)
    # spec_real = torch.nn.functional.conv1d(torch.from_numpy(x.numpy()), torch.from_numpy(self.wcos.numpy()), stride=self.stride)
    # return torch.stack((spec_real, -spec_imag), -1) 
    # import torch
    # y = y.numpy()
    # m = STFT2(pad_mode="constant",n_fft=128, win_length=128, hop_length=64, iSTFT=True)
    # y2 = m(torch.from_numpy(x.numpy())).numpy()
    # assert np.allclose(y,y2, atol=5e-5)
    

if __name__ == '__main__':
  unittest.main()