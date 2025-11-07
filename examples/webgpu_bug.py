import os, numpy as np
from tinygrad import Tensor, nn, Device
import torch
np.random.seed(42)

BACKEND1 = "METAL"
BACKEND2 = "WEBGPU"

def test_simpler():
  BS, N, C, G =  1, 164, 5, 5
  # create in torch
  torch_layer = torch.nn.GroupNorm(G, C).eval()

  # create in tinygrad
  layer = nn.GroupNorm(G, C)
  layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
  layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

  x = Tensor.randn(BS, C, N, N, N, requires_grad=False)
  z = layer(x)

  torch_x = torch.tensor(x.numpy(), requires_grad=False)
  torch_z = torch_layer(torch_x)


  np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)

    #torch_x = torch.tensor(x.numpy(), requires_grad=True)
    #torch_z = torch_layer(torch_x)
def test_groupnorm(b,c1,c2,N):
  '''print("Channels", b,c1,c2, "Dims", N)
  data = np.random.randn(b, c1, N,N,N).astype(np.float32)
  weights = np.random.randn(c1, c2, 3, 3, 3).astype(np.float32)# * 0.1

  for backend in [BACKEND1, BACKEND2]:
    os.environ.pop(BACKEND2, None) if backend == BACKEND1 else os.environ.update({BACKEND2: '1'})
    Device.DEFAULT = backend
    x = Tensor(data)
    conv = nn.Conv2d(c1, c2, kernel_size=(3, 3, 3), padding=1, bias=False)
    conv.weight.assign(Tensor(weights)).realize()
    gn = nn.GroupNorm(c2, c2, affine=False)
    
    out = gn(x).realize().numpy()
    print(f"{backend:6s}: Min={out.min():.3f}, Max={out.max():.3f}, Mean={out.mean():.3f}")
    if backend == BACKEND1: metal_out = out
    else: print(f"Max diff: {np.abs(metal_out - out).max():.3f}")

# by "breaks" i mean sizable diff. in this case > 1.0. typical diff is 5.0 

b,c1,c2,N = [1,30,30,128] # works 
test_groupnorm(b,c1,c2,N)

print("sweep over input dimensions") # ==============
print("="*80)
#b,c1,c2,N = [1,30,30,128+32] # works on everything smaller
#test_groupnorm(b,c1,c2,N)

#b,c1,c2,N = [1,30,30,128+32] # breaks on everything bigger
#test_groupnorm(b,c1,c2,N)

#b,c1,c2,N = [1,30,30,128+36] # breaks on everything bigger
#test_groupnorm(b,c1,c2,N)

print("sweep over inner channels") # ==============
print("="*80)
b,c1,c2,N = [1,5,5,128+34] # channels [5,10,15,30] break
test_groupnorm(b,c1,c2,N)'''

test_simpler()