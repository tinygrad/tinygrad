import time
import torch

N = 2048
a = torch.randn((N, N), dtype=torch.float32).to('mps')
b = torch.randn((N, N), dtype=torch.float32).to('mps')
def fxn():
  print("starting matmul")
  st = time.monotonic()
  c = a @ b
  torch.zeros(1, device='mps').cpu()
  return time.monotonic() - st

flops = N*N*N*2

et = min([fxn() for _ in range(10)])
print(f"{et*1e6:.2f} us, {flops*1e-9/et:.2f} GFLOPS")