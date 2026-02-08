import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv
from extra.gemm.asm.cdna.gemm import asm_gemm
from test.helpers import needs_second_gpu

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
# Use NULL=1 EMULATE=AMD_CDNA4 to also test the assembly
def is_cdna4(): return getattr(Device[Device.DEFAULT].renderer, "arch", "").startswith("gfx950")

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  a_rand = Tensor.randn((batch, M, K), dtype=dtypes.float).sub(0.5).cast(dtype)
  b_rand = Tensor.randn((K, N), dtype=dtypes.float).sub(0.5).cast(dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  a, b = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a, b = a.shard(devs, axis=0), b.shard(devs, axis=None)
  with Context(ASM_GEMM=1):
    tst = asm_gemm(a, b)
    tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  a_ref, b_ref = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a_ref, b_ref = a_ref.shard(devs, axis=0), b_ref.shard(devs, axis=None)
  with Context(ASM_GEMM=0):
    ref = asm_gemm(a_ref, b_ref)
    ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  # no validation on the NULL device
  if a_rand.device.startswith("NULL"): return None
  with Context(DEBUG=0):
    assert (tst - ref).square().max().float().item() < 1e-6, "forward mismatch"
    assert (a.grad - a_ref.grad).square().max().float().item() < 1e-3, "grad_a mismatch"
    assert (b.grad - b_ref.grad).square().max().float().item() < 1e-3, "grad_b mismatch"

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
@unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
class TestGemm(unittest.TestCase):
  def setUp(self):
    if is_cdna4(): self.skipTest("shapes are too small for the assembly GEMM")
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 32), N, N, dtype=dtypes.half)
  def test_gemm(self): verify_asm_gemm(1, 64, 32, 112)
  def test_gemm_batched(self): verify_asm_gemm(2, 64, 32, 32)
  @needs_second_gpu
  def test_gemm_multi(self): verify_asm_gemm(2, 64, 32, 32, gpus=2)

# uses the Asm GEMM on CDNA4 only for speed reasons
class TestGemmLarge(unittest.TestCase):
  def setUp(self):
    if not is_cdna4():
      self.skipTest("very slow on non mi350x")

  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N, dtype=dtypes.half)
  def test_gemm(self): verify_asm_gemm(1, 8192, 4096, 14336)
  def test_gemm_batched(self): verify_asm_gemm(2, 8192, 4096, 4096)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 14336, dtype=dtypes.bfloat16, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, dtype=dtypes.bfloat16, gpus=8)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, dtype=dtypes.bfloat16, gpus=8)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, dtype=dtypes.bfloat16, gpus=8)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, dtype=dtypes.bfloat16, gpus=8)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, dtype=dtypes.bfloat16, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm7(self): verify_asm_gemm(1, 8192, 128256, 4096)
  def test_gemm_unsupported(self):
    with self.assertRaisesRegex(AssertionError, "shape not supported"):
      verify_asm_gemm(8, 1024, 1024, 4096, gpus=8)

if __name__ == "__main__":
  unittest.main()
