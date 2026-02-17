import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv
from extra.gemm.asm.cdna.gemm import asm_gemm
from test.helpers import needs_second_gpu

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
# Use NULL=1 EMULATE=AMD_CDNA4 to also test the assembly
def is_cdna4(): return getattr(Device[Device.DEFAULT].renderer, "arch", "").startswith("gfx950")

def run_asm_gemm(a_shape, b_shape, dtype=dtypes.float16, a_shard=None, b_shard=None, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  a_rand = Tensor.randn(a_shape, dtype=dtypes.float).sub(0.5).cast(dtype)
  b_rand = Tensor.randn(b_shape, dtype=dtypes.float).sub(0.5).cast(dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  a, b = a_rand.clone().requires_grad_(), b_rand.clone().requires_grad_()
  if multi: a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  with Context(ASM_GEMM=1):
    tst = asm_gemm(a, b)
    tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  a_ref, b_ref = a_rand.clone().requires_grad_(), b_rand.clone().requires_grad_()
  if multi: a_ref, b_ref = a_ref.shard(devs, axis=a_shard), b_ref.shard(devs, axis=b_shard)
  with Context(ASM_GEMM=0):
    ref = asm_gemm(a_ref, b_ref)
    ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  # no validation on the NULL device
  if a_rand.device.startswith("NULL"): return None
  atol, rtol = (1e-2, 1e-3)
  with Context(DEBUG=0):
    assert tst.allclose(ref, atol=atol, rtol=rtol), "forward mismatch"
    assert a.grad.allclose(a_ref.grad, atol=atol, rtol=rtol), "grad_a mismatch"
    assert b.grad.allclose(b_ref.grad, atol=atol, rtol=rtol), "grad_b mismatch"


def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=1) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_k_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=8) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)

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
  @needs_second_gpu
  def test_gemm_k_sharded(self): verify_asm_gemm_k_sharded(64, 64, 2*64, gpus=2)

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
  def test_gemm8(self): verify_asm_gemm(1, 4096, 14336, 8192)
  def test_gemm9(self): verify_asm_gemm(8, 4096, 14336, 8192, dtype=dtypes.bfloat16, gpus=8)
  def test_gemm10(self): verify_asm_gemm(1, 4096, 8192, 4096)
  def test_k_sharded_1(self): verify_asm_gemm_k_sharded(14336, 4096, 8*8192, gpus=8)
  def test_k_sharded_2(self): verify_asm_gemm_k_sharded(4096, 14336, 8*8192, gpus=8)
  def test_k_sharded_3(self): verify_asm_gemm_k_sharded(4096, 4096, 8*8192, gpus=8)
  def test_unsupported_k(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1024, 1024, 100)
  def test_unsupported_m(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1000, 256, 256)
  def test_unsupported_n(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 256, 1000, 256)
  def test_unsupported_batch(self):
    with self.assertRaisesRegex(AssertionError, "batch size"):
      verify_asm_gemm(3, 256, 256, 256)
  def test_gemm_previously_unsupported(self): verify_asm_gemm(8, 1024, 1024, 4096, gpus=8)

  # more shapes: vary M, N, K independently
  def test_shape_small_square(self): verify_asm_gemm(1, 256, 256, 256)
  def test_shape_small_rect_m(self): verify_asm_gemm(1, 512, 256, 256)
  def test_shape_small_rect_n(self): verify_asm_gemm(1, 256, 512, 256)
  def test_shape_small_rect_k(self): verify_asm_gemm(1, 256, 256, 512)
  def test_shape_tall(self): verify_asm_gemm(1, 2048, 256, 256)
  def test_shape_wide(self): verify_asm_gemm(1, 256, 2048, 256)
  def test_shape_deep(self): verify_asm_gemm(1, 256, 256, 4096)
  def test_shape_non_square(self): verify_asm_gemm(1, 1024, 2048, 512)
  def test_shape_batched_small(self): verify_asm_gemm(2, 256, 256, 256)
  def test_shape_batched_rect(self): verify_asm_gemm(2, 512, 1024, 256)
  # K edge cases: iters=1,2,3 exercise different loop paths
  def test_shape_k64(self): verify_asm_gemm(1, 256, 256, 64)
  def test_shape_k128(self): verify_asm_gemm(1, 256, 256, 128)
  def test_shape_k192(self): verify_asm_gemm(1, 256, 256, 192)

  def test_llama3_out1(self): verify_asm_gemm(1, 8192, 128256, 4096)
  def test_llama3_out2(self): verify_asm_gemm(1, 8192, 4096, 128256)
  def test_llama3_out3(self): verify_asm_gemm(1, 4096, 128256, 8192)

class TestMagicGu(unittest.TestCase):
  def test_magicgu_matches_old(self):
    from extra.gemm.asm.cdna.asm import _magicgu_mulhi, TILE_M, TILE_N, TILE_K
    old_iters_args = {64: (67108864, 0), 128: (33554432, 0), 224: (613566757, 2147483656)}
    old_gemm_shapes = [
      (8192, 4096, 4096), (8192, 14336, 4096), (8192, 4096, 14336),
      (8192, 8192, 8192), (4096, 4096, 4096), (4096, 14336, 4096),
      (4096, 14336, 8192), (4096, 4096, 14336), (14336, 4096, 8192),
      (4096, 8192, 14336), (4096, 4096, 8192), (4096, 8192, 4096),
    ]
    for M, N, K in old_gemm_shapes:
      iters = K // TILE_K
      total = (M // TILE_M) * (N // TILE_N) * iters
      for batch in [1, 2]:
        magic, shift = _magicgu_mulhi(iters, total * batch)
        old_magic, old_shift = old_iters_args[iters]
        self.assertEqual((magic, shift), (old_magic, old_shift), f"mismatch for ({M},{N},{K}) batch={batch} iters={iters}")

if __name__ == "__main__":
  unittest.main()
