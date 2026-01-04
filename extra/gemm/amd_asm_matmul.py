# RDNA3 128x128 tiled GEMM kernel
# Uses DSL-generated assembly from kernel8_batched_gmem.s

import numpy as np
from tinygrad import Tensor, Device, dtypes, Context, GlobalCounters
from tinygrad.helpers import getenv, colored
from tinygrad.engine.realize import Runner, Estimates, ExecItem

N = getenv("N", 4096)
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
THREADS = 128
LDS_SIZE = 8320

def build_kernel(arch: str = "gfx1100") -> str:
  """Build GEMM kernel using DSL."""
  from extra.gemm.amd_asm_kernel_dsl import build_kernel as build_kernel_dsl
  return build_kernel_dsl(arch)

def test_matmul():
  dev = Device[Device.DEFAULT]
  arch = dev.arch
  print(f"Device arch: {arch}")

  asm = build_kernel(arch)
  if getenv("PRINT_ASM", 0):
    print("Generated assembly:")
    print(asm)

  try:
    binary = dev.compiler.compile(asm)
    print(f"Compiled! Binary size: {len(binary)} bytes")
  except Exception as e:
    print(f"Compilation failed: {e}")
    return

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  b = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  grid = (N // BLOCK_N, N // BLOCK_M, 1)
  local = (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  # Create a custom runner for DEBUG=2 support
  _prg = dev.runtime("kernel", binary)
  class AsmRunner(Runner):
    def __init__(self):
      super().__init__(colored("kernel", "cyan"), Device.DEFAULT, Estimates(ops=N*N*N*2, mem=N*N*4*3))
    def __call__(self, rawbufs, var_vals, wait=False):
      # Kernel expects (a, b, c) but rawbufs is [c, a, b] to match tinygrad convention
      c_buf, a_buf, b_buf = [x.ensure_allocated()._buf for x in rawbufs]
      return _prg(a_buf, b_buf, c_buf, global_size=grid, local_size=local, wait=wait)

  runner = AsmRunner()
  bufs = [c.uop.buffer, a.uop.buffer, b.uop.buffer]
  ei = ExecItem(None, bufs, prg=runner)

  run_count = getenv("CNT", 5)
  ets = []
  try:
    with Context(DEBUG=2):
      for _ in range(run_count):
        et = ei.run(wait=True)
        ets.append(et)
    print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")
  except Exception as e:
    print(f"Kernel execution failed: {e}")
    import traceback
    traceback.print_exc()
    return

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a @ b).realize()
    with Context(DEBUG=0):
      err = (c - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-06:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  if getenv("ASM", 0):
    print(build_kernel(Device[Device.DEFAULT].arch))
  else:
    test_matmul()
