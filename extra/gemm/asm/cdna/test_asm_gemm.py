import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv
from extra.gemm.asm.cdna.gemm import asm_gemm

def verify_asm_gemm(batch:int, M:int, K:int, N:int, dtype=dtypes.bfloat16, multi=False) -> None:
  a_rand = Tensor.randn((batch, M, K), dtype=dtypes.float).sub(0.5).cast(dtype)
  b_rand = Tensor.randn((K, N), dtype=dtypes.float).sub(0.5).cast(dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(8)) if multi else None

  a, b = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a, b = a.shard(devs, axis=0), b.shard(devs, axis=None)
  tst = asm_gemm(a, b)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  a_ref, b_ref = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a_ref, b_ref = a_ref.shard(devs, axis=0), b_ref.shard(devs, axis=None)
  with Context(ASM_GEMM=0): ref = a_ref @ b_ref
  ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  with Context(DEBUG=0):
    assert (tst - ref).square().max().float().item() < 1e-6, "forward mismatch"
    assert (a.grad - a_ref.grad).square().max().float().item() < 1e-3, "grad_a mismatch"
    assert (b.grad - b_ref.grad).square().max().float().item() < 1e-3, "grad_b mismatch"

class TestGemm(unittest.TestCase):
  def test_simple(self): verify_asm_gemm(8, 8192, 4096, 1024)
  def test_square(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 1024, multi=True)
  def test_gemm2(self): verify_asm_gemm(8, 8192, 14336, 4096, multi=True)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 4096, 128256, multi=True)
  def test_gemm4(self): verify_asm_gemm(8, 8192, 4096, 14336, multi=True)

if __name__ == "__main__":
  unittest.main()
