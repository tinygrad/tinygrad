# Testing for yolov3

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Upsample
import torch
import numpy as np

def test_conv2d():
  conv = Conv2d(16, 33, 3, stride=2)
  conv.weights = Tensor.randn(33, 16, 3, 3)
  conv.biases = Tensor.randn(1, 33, 1, 1)

  x = Tensor.randn(20, 16, 50, 100)
  out = conv(x)

  # Torch output
  tconv = torch.nn.Conv2d(16, 33, 3, stride=2)
  def fromNumpy(ndarray): # have to use this since PyTorch on M1 doesn't have NumPy support… smh
    return torch.Tensor(ndarray.tolist())

  tconv.bias = torch.nn.Parameter(fromNumpy(conv.biases.cpu().data).squeeze())
  tconv.weight = torch.nn.Parameter(fromNumpy(conv.weights.cpu().data))
  tout = tconv(fromNumpy(x.cpu().data))
  #print("Output (tinygrad)")
  #print(out.cpu().data[0][0][0])
  #print("Output (Torch)")
  #print(tout[0][0][0])
  np.testing.assert_allclose(out.cpu().data, tout.detach().numpy(), rtol=5e-2)

def test_upsample():
  x = Tensor(np.arange(1,5)).reshape(shape=(1,1,2,2))
  tx = torch.arange(1,5).view(1,1,2,2).float()
  m = Upsample(scale_factor=2, mode='nearest')
  tm = torch.nn.Upsample(scale_factor=2, mode='nearest')
  #print("Output (tinygrad)")
  #print(m(x).cpu().data)
  np.testing.assert_allclose(m(x).cpu().data, tm(tx).detach().numpy(), rtol=5e-5)

np.random.seed(1337)
test_upsample()
test_conv2d()
print("Tests passed ✅")
