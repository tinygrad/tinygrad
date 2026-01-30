import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv
from extra.gemm.asm.cdna.gemm import asm_gemm

def verify_asm_gemm(batch:int, M:int, K:int, N:int, dtype=dtypes.bfloat16, multi=False) -> None:
  a = Tensor.randn((batch, M, K), dtype=dtypes.float).sub(0.5).cast(dtype)
  b = Tensor.randn((K, N), dtype=dtypes.float).sub(0.5).cast(dtype)
  with Context(DEBUG=0):
    Tensor.realize(a, b)

  if multi:
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(8))
    a = a.shard(devs, axis=0)
    b = b.shard(devs, axis=None)

  asm = asm_gemm(a, b)
  Tensor.realize(asm)

  with Context(ASM_GEMM=0):
    ref = a @ b
  Tensor.realize(ref)

  err = (asm - ref).square().max().float()
  with Context(DEBUG=0):
    assert err.item() < 1e-6

class TestGemm(unittest.TestCase):
  def test_simple(self): verify_asm_gemm(8, 8192, 4096, 1024)
  def test_square(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 1024, multi=True)
  def test_gemm2(self): verify_asm_gemm(8, 8192, 14336, 4096, multi=True)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 4096, 128256, multi=True)
  def test_gemm4(self): verify_asm_gemm(8, 8192, 4096, 14336, multi=True)

if __name__ == "__main__":
  unittest.main()
