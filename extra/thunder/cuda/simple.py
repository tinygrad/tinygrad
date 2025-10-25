import os, pathlib
os.environ["CUDA_NVCC"] = '1'
from tinygrad import Device, Tensor
from tinygrad.helpers import Context, getenv
from tinygrad.runtime.support.compiler_cuda import pretty_ptx

if __name__ == "__main__":
  code = pathlib.Path("simple.cu").read_text()
  device = Device["CUDA"]
  lib = device.compiler.compile(code)
  kernel_name = lib.decode().split(".globl\t")[1].split("\n")[0]
  print("kernel name", kernel_name)
  #print(pretty_ptx(lib.decode()))

  prg = device.runtime(kernel_name, lib)
  prg.smem = 10000

  N = 1024
  a = Tensor.randn(N, N, device='CUDA')
  b = Tensor.randn(N, N, device='CUDA')
  c = Tensor.empty(N, N, device='CUDA')
  Tensor.realize(a, b, c)

  TILE_DIM = 8
  N_BLOCK = 4
  M_BLOCK = 4

  gsz = (N // (M_BLOCK * TILE_DIM), N // (N_BLOCK * TILE_DIM), 1)
  for _ in range(5):
    et = prg(c.uop.buffer.ensure_allocated()._buf, a.uop.buffer._buf, b.uop.buffer._buf,
             global_size=gsz, local_size=(32,1,1), wait=True)
    print(f"{N*N*N*2/(et*1e9):2f} GFLOPS")

  for _ in range(5):
    with Context(DEBUG=2):
      ref = (a@b).realize()

  print((ref-c).mean().item(), (ref-c).max().item())

