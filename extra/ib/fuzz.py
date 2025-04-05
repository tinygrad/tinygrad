import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import getenv

ibd = f"IB:0:{getenv('REMOTE', '')}"

def send(tensor: Tensor):
  tensor.contiguous().to(ibd).realize()

def receive(*shape, d=Device.DEFAULT):
  return Tensor.empty(shape, device=ibd).contiguous().to(d).realize()

def rdma_fuzz(side:bool):
  if side:
    for i in range(N):
      l = Tensor.arange(float(i), float(SZ+i))
      print(l.numpy())
      send(l)
  else:
    for i in range(N):
      l = Tensor.arange(float(i), float(SZ+i))
      r = receive(SZ)
      print(l.numpy())
      print(r.numpy())
      np.testing.assert_equal(l.numpy(), r.numpy())

if __name__ == '__main__':
  tiny31 = getenv('REMOTE', '').endswith('30')

  SZ = 64*1024*1024//4 # 64MB
  # SZ=1

  N = 5

  rdma_fuzz(tiny31)
  rdma_fuzz(not tiny31)
