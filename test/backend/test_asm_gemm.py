import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv, system, DEV
from extra.gemm.cdna_asm_gemm import asm_gemm
from test.helpers import needs_second_gpu
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8, FP8_MAX

def _assert_allclose(name:str, got:Tensor, expected:Tensor, atol:float, rtol:float) -> None:
  if got.allclose(expected, atol=atol, rtol=rtol).item(): return
  raise AssertionError(_allclose_mismatch_message(name, got, expected, atol, rtol))

def _allclose_mismatch_message(name:str, got:Tensor, expected:Tensor, atol:float, rtol:float) -> str:
  import numpy as np
  got_np, expected_np = got.numpy(), expected.numpy()
  got_flat, expected_flat = got_np.reshape(-1), expected_np.reshape(-1)
  total, chunk_size = got_flat.size, 1_000_000
  mismatch_count = first_bad = max_abs_idx = max_rel_idx = max_margin_idx = None
  max_abs = max_rel = max_margin = -1.0
  top:list[tuple[float, int]] = []
  for start in range(0, total, chunk_size):
    end = min(start+chunk_size, total)
    g, e = got_flat[start:end].astype(np.float32), expected_flat[start:end].astype(np.float32)
    diff = np.abs(g-e)
    tol = atol + rtol*np.abs(e)
    bad = ~(diff <= tol)
    bad_count = int(bad.sum())
    if bad_count:
      mismatch_count = (mismatch_count or 0) + bad_count
      if first_bad is None: first_bad = start + int(np.argmax(bad))
      bad_idxs = np.nonzero(bad)[0]
      take = bad_idxs[np.argsort(diff[bad_idxs])[-8:]]
      top.extend((float(diff[i]), start+int(i)) for i in take)
      top = sorted(top, reverse=True)[:8]
    local_abs_idx = int(np.argmax(diff))
    if float(diff[local_abs_idx]) > max_abs:
      max_abs, max_abs_idx = float(diff[local_abs_idx]), start+local_abs_idx
    rel = diff / np.maximum(np.abs(e), 1e-30)
    local_rel_idx = int(np.argmax(rel))
    if float(rel[local_rel_idx]) > max_rel:
      max_rel, max_rel_idx = float(rel[local_rel_idx]), start+local_rel_idx
    margin = diff - tol
    local_margin_idx = int(np.argmax(margin))
    if float(margin[local_margin_idx]) > max_margin:
      max_margin, max_margin_idx = float(margin[local_margin_idx]), start+local_margin_idx

  def fmt_idx(idx:int) -> str: return f"flat={idx} idx={tuple(int(x) for x in np.unravel_index(idx, got_np.shape))}"
  def fmt_val(idx:int) -> str:
    g, e = float(got_flat[idx]), float(expected_flat[idx])
    diff, tol = abs(g-e), atol + rtol*abs(e)
    return f"{fmt_idx(idx)} got={g:.9g} expected={e:.9g} diff={diff:.9g} tol={tol:.9g} rel={diff/max(abs(e), 1e-30):.9g}"

  lines = [
    f"{name} mismatch",
    f"shape={got_np.shape} dtype(got)={got.dtype} dtype(expected)={expected.dtype} atol={atol} rtol={rtol}",
    f"mismatches={mismatch_count}/{total} ({(mismatch_count or 0)/total:.6%})",
    f"first mismatch: {fmt_val(first_bad)}" if first_bad is not None else "first mismatch: none found after numpy conversion",
    f"max abs diff: {fmt_val(max_abs_idx)}",
    f"max rel diff: {fmt_val(max_rel_idx)}",
    f"max tol overrun: {fmt_val(max_margin_idx)}",
  ]
  if top:
    lines.append("top mismatches by abs diff:")
    lines += [f"  {i+1}. {fmt_val(idx)}" for i, (_, idx) in enumerate(top)]
  if got_np.ndim and max_margin_idx is not None:
    def row_summary(label:str, flat_idx:int) -> str:
      idx = tuple(int(x) for x in np.unravel_index(flat_idx, got_np.shape))
      row_sl = idx[:-1] + (slice(None),)
      g, e = got_np[row_sl].astype(np.float32), expected_np[row_sl].astype(np.float32)
      diff, tol = np.abs(g-e), atol + rtol*np.abs(e)
      bad = ~(diff <= tol)
      bad_cols = np.nonzero(bad)[0]
      worst_col = int(np.argmax(diff - tol))
      if bad_cols.size == 0: return f"{label} row prefix={idx[:-1]} has no mismatches after numpy conversion"
      return (f"{label} row prefix={idx[:-1]} mismatches={bad_cols.size}/{g.size} "
              f"first_col={int(bad_cols[0])} last_col={int(bad_cols[-1])} "
              f"worst_col={worst_col} got={float(g[worst_col]):.9g} expected={float(e[worst_col]):.9g} "
              f"diff={float(diff[worst_col]):.9g} tol={float(tol[worst_col]):.9g}")
    if first_bad is not None: lines.append(row_summary("first mismatch", first_bad))
    lines.append(row_summary("max tol overrun", max_margin_idx))
    idx = tuple(int(x) for x in np.unravel_index(max_margin_idx, got_np.shape))
    axis, center = got_np.ndim-1, idx[-1]
    lo, hi = max(0, center-3), min(got_np.shape[-1], center+4)
    sl = tuple(idx[i] if i != axis else slice(lo, hi) for i in range(got_np.ndim))
    lines += [
      f"local slice around max tol overrun on last axis [{lo}:{hi}] at prefix={idx[:-1]}:",
      f"  got={got_np[sl].astype(np.float32)}",
      f"  expected={expected_np[sl].astype(np.float32)}",
      f"  diff={np.abs(got_np[sl].astype(np.float32)-expected_np[sl].astype(np.float32))}",
    ]
  return "\n".join(lines)

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
# Use DEV=NULL:HIP:gfx950 to also test the assembly
def is_cdna4(): return Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950")

def run_asm_gemm(a_shape, b_shape, dtype=dtypes.float16, a_shard=None, b_shard=None, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  input_dtype = dtypes.bfloat16 if dtype == FP8_DTYPE else dtype
  a_rand = Tensor.randn(a_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  b_rand = Tensor.randn(b_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  if dtype == FP8_DTYPE:
    a_rand, x_scale, _ = quantize_fp8(a_rand)
    b_rand, w_scale, _ = quantize_fp8(b_rand)
    grad_amax_state = Tensor.full((), FP8_MAX, dtype=dtypes.float32, device=devs).contiguous()
    with Context(DEBUG=0):
      Tensor.realize(a_rand, x_scale, b_rand, w_scale, grad_amax_state)

  # clone all inputs before any backward: a clone copies the source's current .grad
  a, b = a_rand.clone(), b_rand.clone()
  if dtype == FP8_DTYPE:
    a_ref, b_ref = a_rand.detach().cast(dtypes.bfloat16), b_rand.detach().cast(dtypes.bfloat16)
  else:
    a_ref, b_ref = a_rand.clone(), b_rand.clone()
  if multi: a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    tst = asm_gemm(a, b, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state)
  else:
    tst = asm_gemm(a, b)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  if multi: a_ref, b_ref = a_ref.shard(devs, axis=a_shard), b_ref.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    ref = ((a_ref @ b_ref) * x_scale * w_scale).cast(dtypes.bfloat16)
  else:
    ref = a_ref @ b_ref
  ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  # no validation on the NULL device
  if a_rand.device.startswith("NULL"): return None
  atol, rtol = (2e-1, 1e-2) if dtype == dtypes.bfloat16 else (256, 1e-2) if dtype == FP8_DTYPE else (1e-2, 1e-3)
  # allow more rtol for multi because of ALLREDUCE_CAST
  grad_atol, grad_rtol = (16895, 0.125) if dtype == FP8_DTYPE else (atol, 2e-2 if multi else rtol)
  with Context(DEBUG=0):
    # enable for debugging, slow for larger gemms
    if getenv("USE_NPY"):
      import numpy as np
      np.testing.assert_allclose(tst.numpy(), ref.numpy(), atol=atol, rtol=rtol)
      np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
      np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
    _assert_allclose("forward", tst, ref, atol=atol, rtol=rtol)
    _assert_allclose("grad_a", a.grad, a_ref.grad, atol=grad_atol, rtol=grad_rtol)
    _assert_allclose("grad_b", b.grad, b_ref.grad, atol=grad_atol, rtol=grad_rtol)

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=1) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_k_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=8) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)

def verify_asm_gemm_n_sharded(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_m_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_n_sharded_2d(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_k_sharded_3d(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus)

def run_llama_final_logits_gemm(gpus:int=1) -> None:
  Tensor.manual_seed(0)
  batch, seqlen, dim, vocab = 2*gpus, 8192, 4096, 128256
  x = Tensor.randn(batch, seqlen, dim, dtype=dtypes.float).sub(0.5).cast(dtypes.bfloat16)
  w = Tensor.randn(vocab, dim, dtype=dtypes.float).sub(0.5).cast(dtypes.bfloat16)
  dy = Tensor.randn(batch, seqlen, vocab, dtype=dtypes.float).sub(0.5).cast(dtypes.bfloat16)
  if gpus > 1:
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus))
    x, w, dy = x.shard(devs, axis=0), w.shard(devs, axis=None), dy.shard(devs, axis=0)
  with Context(DEBUG=0): Tensor.realize(x, w, dy)
  logits = asm_gemm(x, w.T)
  (logits * dy).sum().backward()
  Tensor.realize(logits, x.grad, w.grad)
  dname = x.device[0] if isinstance(x.device, tuple) else x.device
  if dname.startswith("NULL"): return None
  x_ref, w_ref, dy_ref = x.detach(), w.detach(), dy.detach()
  ref = x_ref @ w_ref.T
  (ref * dy_ref).sum().backward()
  Tensor.realize(ref, x_ref.grad, w_ref.grad)
  _assert_allclose("final logits forward", logits, ref, atol=2e-1, rtol=1e-2)
  _assert_allclose("final logits grad_x", x.grad, x_ref.grad, atol=2e-1, rtol=2e-2 if gpus > 1 else 1e-2)
  _assert_allclose("final logits grad_w", w.grad, w_ref.grad, atol=2e-1, rtol=2.5e-2 if gpus > 1 else 1e-2)

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
@unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
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
  @needs_second_gpu
  def test_gemm_m_sharded(self): verify_asm_gemm_m_sharded(2*64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded(self): verify_asm_gemm_n_sharded(1, 64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded_2d(self): verify_asm_gemm_n_sharded_2d(64, 2*64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_k_sharded_3d(self): verify_asm_gemm_k_sharded_3d(1, 64, 32, 2*64, gpus=2)

# uses the smallest size for the cdna assembly gemm
class TestAsmGEMM(unittest.TestCase):
  def setUp(self):
    if not is_cdna4():
      self.skipTest("assembly gemm is only for cdna4")

  def test_tiny(self): verify_asm_gemm(1, 256, 256, 64)

  def test_verify_with_numpy(self):
    import numpy as np
    M, N, K = 256, 256, 64
    rng = np.random.default_rng(0)
    a_np = (rng.random((M, K), dtype=np.float32) - 0.5).astype(np.half)
    b_np = (rng.random((K, N), dtype=np.float32) - 0.5).astype(np.half)
    c_np = a_np @ b_np
    a, b = Tensor(a_np), Tensor(b_np)
    c = asm_gemm(a, b)
    c.realize()
    # no validation on the NULL device
    if a.device.startswith("NULL"): return None
    np.testing.assert_allclose(c.numpy(), c_np, atol=2e-3, rtol=5e-2)

  def test_unsupported_batch(self):
    with self.assertRaisesRegex(AssertionError, "batch size"):
      verify_asm_gemm(3, 256, 256, 256)

  def test_unsupported_k(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1024, 1024, 100)
  def test_unsupported_m(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1000, 256, 256)
  def test_unsupported_n(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 256, 1000, 256)

# test the Asm GEMM with Llama shapes, only run on the real machine for speed
class TestGemmLlama(unittest.TestCase):
  dtype = dtypes.bfloat16

  def setUp(self):
    if not is_cdna4() or DEV.interface.startswith("MOCK"):
      self.skipTest("very slow on non mi350x")

  def test_empty(self): asm_gemm(Tensor.empty(N:=getenv("N", 4096), N, dtype=self.dtype), Tensor.empty(N, N, dtype=self.dtype)).realize()

  def test_empty_bw(self):
    x = Tensor.empty(1, N:=getenv("N", 4096), N, dtype=self.dtype)
    y = Tensor.empty((N, N), dtype=self.dtype)
    if self.dtype == FP8_DTYPE:
      x_scale = Tensor.empty((), dtype=dtypes.float32)
      w_scale = Tensor.empty((), dtype=dtypes.float32)
      grad_amax_state = Tensor.empty((), dtype=dtypes.float32).contiguous()
      z = asm_gemm(x, y, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state)
    else:
      z = asm_gemm(x, y)
    z.sum().backward()
    Tensor.realize(z, x.grad, y.grad)
    # FP8 GEMM stores bf16 output and its backward produces bf16 gradients.
    grad_dtype = dtypes.bfloat16 if self.dtype == FP8_DTYPE else self.dtype
    assert z.dtype == dtypes.bfloat16
    assert x.grad.dtype == y.grad.dtype == grad_dtype

  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N, dtype=self.dtype)
  def test_gemm(self): verify_asm_gemm(1, 8192, 4096, 14336, dtype=self.dtype)
  def test_gemm_batched(self): verify_asm_gemm(2, 8192, 4096, 4096, dtype=self.dtype)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 14336, dtype=self.dtype, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, dtype=self.dtype, gpus=8)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm7(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_gemm8(self): verify_asm_gemm(1, 4096, 14336, 8192, dtype=self.dtype)
  def test_gemm9(self): verify_asm_gemm(8, 4096, 14336, 8192, dtype=self.dtype, gpus=8)
  def test_gemm10(self): verify_asm_gemm(1, 4096, 8192, 4096, dtype=self.dtype)
  def test_gemm_previously_unsupported(self): verify_asm_gemm(8, 1024, 1024, 4096, gpus=8)
  def test_k_sharded_1(self): verify_asm_gemm_k_sharded(14336, 4096, 8*8192, dtype=self.dtype, gpus=8)
  def test_k_sharded_2(self): verify_asm_gemm_k_sharded(4096, 14336, 8*8192, dtype=self.dtype, gpus=8)
  def test_k_sharded_3(self): verify_asm_gemm_k_sharded(4096, 4096, 8*8192, dtype=self.dtype, gpus=8)

  # M-sharded 2D
  def test_m_sharded_1(self): verify_asm_gemm_m_sharded(8*8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_m_sharded_2(self): verify_asm_gemm_m_sharded(8*4096, 14336, 4096, dtype=self.dtype, gpus=8)

  # N-sharded 2D
  def test_n_sharded_2d_1(self): verify_asm_gemm_n_sharded_2d(8192, 8*4096, 4096, dtype=self.dtype, gpus=8)
  def test_n_sharded_2d_2(self): verify_asm_gemm_n_sharded_2d(4096, 8*14336, 4096, dtype=self.dtype, gpus=8)

  # tensor parallel shapes (Llama 8B, MP=8)
  def test_tp_n_sharded_wq(self): verify_asm_gemm_n_sharded(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_tp_n_sharded_w1(self): verify_asm_gemm_n_sharded(1, 8192, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_tp_k_sharded_wo(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_tp_k_sharded_w2(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 14336, dtype=self.dtype, gpus=8)

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

  def test_llama3_out1(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_llama3_out2(self): verify_asm_gemm(1, 8192, 4096, 128256, dtype=self.dtype)
  def test_llama3_out3(self): verify_asm_gemm(1, 4096, 128256, 8192, dtype=self.dtype)

  def test_llama3_final_logits_bw_hk_bf16_single(self):
    if self.dtype != dtypes.bfloat16: self.skipTest("final logits HK bf16 test is bf16-only")
    if not has_hipcc(): self.skipTest("HK bf16 gemm requires hipcc to compile")
    if not getenv("USE_HK_BF16_GEMM", 0): self.skipTest("set USE_HK_BF16_GEMM=1 to test HK bf16 final logits")
    run_llama_final_logits_gemm(gpus=1)

  @needs_second_gpu
  def test_llama3_final_logits_bw_hk_bf16_multi(self):
    if self.dtype != dtypes.bfloat16: self.skipTest("final logits HK bf16 test is bf16-only")
    if not has_hipcc(): self.skipTest("HK bf16 gemm requires hipcc to compile")
    if not getenv("USE_HK_BF16_GEMM", 0): self.skipTest("set USE_HK_BF16_GEMM=1 to test HK bf16 final logits")
    run_llama_final_logits_gemm(gpus=getenv("GPUS", 8))

def has_hipcc():
  try: system("hipcc --version")
  except Exception: return False
  return True

@unittest.skipUnless(has_hipcc(), "FP8 gemm requires hipcc to compile")
class TestGemmLlamaFP8(TestGemmLlama): dtype = FP8_DTYPE

class TestMagicGu(unittest.TestCase):
  def test_magicgu_matches_old(self):
    from extra.gemm.cdna_asm_gemm import _magicgu_mulhi, TILE_M, TILE_N, TILE_K
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
