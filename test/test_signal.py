import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.signal import fft, stft
import numpy as np

class TestSignal(unittest.TestCase):
  def test_fft(self):
    data = np.arange(5)
    np_out = np.fft.fft(data)
    np_out = np.stack(np_out.real, np_out.imag, -1)
    tg_out = fft.fft(Tensor(data))
    
  def test_stft(self):
    x = Tensor.ones(1,1000)
    stft_layer = stft.STFT()
    y = stft_layer.transform(x)
    breakpoint()
    

if __name__ == '__main__':
  unittest.main()