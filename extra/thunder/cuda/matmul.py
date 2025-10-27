import pathlib
from tinygrad import Device, Tensor
from tinygrad.helpers import Context
from tinygrad.runtime.support.compiler_cuda import pretty_ptx, NVCCCompiler

if __name__ == "__main__":
  code = (pathlib.Path(__file__).parent / "matmul.cu").read_text()
  device = Device["CUDA"]
  kitten_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "--expt-relaxed-constexpr", "-DKITTENS_HOPPER"]
  lib = NVCCCompiler(device.compiler.arch, kitten_args).compile(code)
  kernel_name = lib.decode().split(".globl\t")[1].split("\n")[0]
  print("kernel name", kernel_name)
  print(pretty_ptx(lib.decode()))

  prg = device.runtime(kernel_name, lib)
  prg.smem = 10000

  N = 8192
  a = Tensor.randn(N, N, device='CUDA', dtype="bfloat16")
  b = Tensor.randn(N, N, device='CUDA', dtype="bfloat16")
  c = Tensor.empty(N, N, device='CUDA', dtype="bfloat16")
  Tensor.realize(a, b, c)

  BLOCK_SIZE = 32

  gsz = (N // BLOCK_SIZE, N // BLOCK_SIZE, 1)
  for _ in range(5):
    et = prg(c.uop.buffer.ensure_allocated()._buf, a.uop.buffer._buf, b.uop.buffer._buf,
             global_size=gsz, local_size=(32,1,1), wait=True)
    print(f"{N*N*N*2/(et*1e9):2f} GFLOPS")

  for _ in range(5):
    with Context(DEBUG=2):
      ref = (a@b).realize()

  ref, c = ref.float(), c.float()
  print((ref-c).mean().item(), (ref-c).max().item())
