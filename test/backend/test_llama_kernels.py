import unittest, functools
from tinygrad import Tensor, Device, dtypes, Context, GlobalCounters
from tinygrad.helpers import getenv
from examples.mlperf.optim import GradAccClipAdamW
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8
from extra.llama_kernels.fused_ce import fused_ce_loss
from extra.llama_kernels import NUM_WG, local_abs_max
from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed, quantize_fp8_scalar
from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import _custom_bwd as custom_rmsnorm_bwd
from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import fused_add_rmsnorm_mul_quantize_fp8, fused_rmsnorm_mul_quantize_fp8
from extra.llama_kernels.cast_amax import _custom_fused_bwd_w13 as custom_fused_bwd_w13
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis
from extra.thunder.amd.fa import custom_fa_backward_pre, custom_fused_qkv_rope_backward, fused_qkv_rope
from test.helpers import needs_second_gpu
from test.backend.test_asm_gemm import has_hipcc

def run_fused_ce(bs:int, seqlen:int, vocab:int, label_smoothing:float=0.0) -> None:
  Tensor.manual_seed(0)
  logits_rand = Tensor.randn(bs, seqlen, vocab).cast(dtypes.bfloat16)
  targets = Tensor.randint(bs, seqlen, high=vocab, dtype=dtypes.int32)
  logits, logits_ref = logits_rand.clone(), logits_rand.detach().float().contiguous()
  with Context(DEBUG=0):
    Tensor.realize(logits, logits_ref, targets)

  loss = fused_ce_loss(logits, targets, label_smoothing=label_smoothing)
  loss.backward()
  Tensor.realize(loss, logits.grad)

  ref = logits_ref.sparse_categorical_crossentropy(targets, label_smoothing=label_smoothing)
  ref.backward()
  Tensor.realize(ref, logits_ref.grad)

  assert logits.grad.shape == (bs, seqlen, vocab)
  with Context(DEBUG=0):
    assert loss.allclose(ref, atol=2e-3, rtol=2e-3).item(), "forward mismatch"
    assert logits.grad.allclose(logits_ref.grad, atol=2e-3, rtol=2e-3).item(), "grad mismatch"

class TestFusedCE(unittest.TestCase):
  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("need bfloat16")

  def test_fused_ce_1_2_16(self): run_fused_ce(1, 2, 16, label_smoothing=0.2)
  def test_fused_ce_2_16_128(self): run_fused_ce(2, 16, 128)
  def test_fused_ce_4_128_1024(self): run_fused_ce(4, 128, 1024, label_smoothing=0.2)

  @unittest.skipUnless(Device.DEFAULT.split(":")[0] == "AMD", "requires AMD custom kernel")
  def test_fused_ce_llama31_8b(self): run_fused_ce(2, 8192, 128256)

def run_quantize_fp8(shape:tuple[int, ...], delayed:bool=True) -> None:
  Tensor.manual_seed(0)
  x = Tensor.randn(*shape).cast(dtypes.bfloat16).contiguous()
  amax_state = Tensor.full((), 2.0, dtype=dtypes.float32).contiguous()
  with Context(DEBUG=0): Tensor.realize(x, amax_state)

  if delayed:
    amax_out = Tensor.zeros((), dtype=dtypes.float32, device=x.device).realize()
    fp8, inv_scale = quantize_fp8_delayed(x, amax_state, amax_out, FP8_DTYPE)
    ref_fp8, ref_inv_scale, ref_new_amax = quantize_fp8(x, amax_state=amax_state)
    Tensor.realize(fp8, inv_scale)
    Tensor.realize(ref_fp8, ref_inv_scale, ref_new_amax)
  else:
    fp8 = quantize_fp8_scalar(x, amax_state, FP8_DTYPE)
    ref_fp8, _, _ = quantize_fp8(x, amax_state=amax_state)
    Tensor.realize(fp8)
    Tensor.realize(ref_fp8)

  with Context(DEBUG=0):
    assert fp8.cast(dtypes.float).allclose(ref_fp8.cast(dtypes.float), atol=0, rtol=0).item(), "fp8 mismatch"
    if delayed:
      assert inv_scale.allclose(ref_inv_scale, atol=0, rtol=0).item(), "inv_scale mismatch"
      assert amax_out.allclose(ref_new_amax, atol=0, rtol=0).item(), \
        f"amax mismatch: got={amax_out.item()} ref={ref_new_amax.item()} diff={abs(amax_out.item()-ref_new_amax.item())}"

def run_quantize_fp8_layer(shape:tuple[int, ...]) -> None:
  Tensor.manual_seed(0)
  x = Tensor.randn(*shape).cast(dtypes.bfloat16).contiguous()
  amax_state = Tensor([1.0, 2.0], dtype=dtypes.float32).contiguous()
  amax_out = Tensor.zeros(2, dtype=dtypes.float32).contiguous()
  layer_num = Tensor([1], dtype=dtypes.int32).contiguous()
  with Context(DEBUG=0): Tensor.realize(x, amax_state, amax_out, layer_num)

  fp8, _ = quantize_fp8_delayed(x, amax_state, amax_out, FP8_DTYPE, layer_num=layer_num)
  ref_fp8, _, ref_amax = quantize_fp8(x, amax_state=amax_state[1])
  Tensor.realize(fp8, amax_out, ref_fp8, ref_amax)

  with Context(DEBUG=0):
    assert fp8.cast(dtypes.float).allclose(ref_fp8.cast(dtypes.float), atol=0, rtol=0).item(), "fp8 mismatch"
    assert amax_out[0].item() == 0.0, "atomic max modified the wrong layer"
    assert amax_out[1].allclose(ref_amax, atol=0, rtol=0).item(), "layer amax mismatch"

class TestQuantizeFP8(unittest.TestCase):
  def setUp(self):
    ren = Device[Device.DEFAULT].renderer
    if dtypes.bfloat16 not in ren.supported_dtypes(): self.skipTest("need bfloat16")
    if not ren.has_local or not ren.has_shared: self.skipTest("need local/shared")

  def test_scalar(self): run_quantize_fp8((getenv("N", 1024), 32), delayed=False)
  def test_delayed(self): run_quantize_fp8((getenv("N", 2048), 1024))
  @unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD atomic max")
  def test_delayed_layer(self): run_quantize_fp8_layer((getenv("N", 2048), 1024))
  @unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD custom kernel")
  def test_delayed_llama31_8b_shapes(self):
    for n in (65536, 98304):
      with self.subTest(n=n): run_quantize_fp8((n, 1024))

  @needs_second_gpu
  def test_multi(self):
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(8))
    x = Tensor.empty(2048*8, 1024, dtype=dtypes.bfloat16, device=devs).uop.multi(0)
    x = Tensor(x, device=devs)
    amax_state = Tensor.full((), 2.0, dtype=dtypes.float32, device=devs).contiguous()
    amax_out = Tensor.zeros((), dtype=dtypes.float32, device=devs).realize()
    fp8, _ = quantize_fp8_delayed(x, amax_state, amax_out, FP8_DTYPE)
    Tensor.realize(fp8)
    assert fp8.uop.shape == x.uop.shape
    assert amax_out.shape == ()

class TestLocalAmax(unittest.TestCase):
  def test_multi_tensor_local_shard_amax(self):
    devices = ("CPU:0", "CPU:1")
    x = Tensor.arange(16).reshape(4, 4).cast(dtypes.float).clone(devices[0]).realize().shard(devices, axis=0).realize()
    GlobalCounters.reset()
    out = (x * local_abs_max(x)).clone().realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(out.tolist(), [[0., 7., 14., 21.], [28., 35., 42., 49.], [120., 135., 150., 165.], [180., 195., 210., 225.]])

class TestMasterWeightUpdate(unittest.TestCase):
  def test_master_is_quantization_source(self):
    initial = Tensor([[[1., 2.], [3., 4.]]]).contiguous().realize()
    param = initial.cast(dtypes.bfloat16).contiguous().realize().is_param_()
    master = initial.clone().realize()
    update = Tensor.full(param.shape, 0.125).contiguous().realize()
    optim = GradAccClipAdamW([param], lr=0.25, weight_decay=0.1)
    expected_master = (initial - (update + 0.25 * 0.1 * initial)).realize()
    expected_param = expected_master.cast(dtypes.bfloat16).contiguous().realize()
    optim.lr.realize()
    param.assign(optim._apply_update(param, update, master))
    GlobalCounters.reset()
    Tensor.realize(param, master)
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertTrue(master.allclose(expected_master, atol=1e-6, rtol=1e-6).item())
    self.assertTrue(param.allclose(expected_param, atol=0, rtol=0).item())

@unittest.skipUnless(has_hipcc() and Device.DEFAULT.split(":")[0] == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedFP8AdamW(unittest.TestCase):
  def run_update(self, shape:tuple[int, ...], random:bool=False):
    Tensor.manual_seed(0)
    b1, b2, eps, wd = 0.9, 0.95, 1e-5, 0.1
    if random:
      master = (Tensor.rand(*shape) * 0.2 - 0.1).contiguous().realize()
      m = (Tensor.rand(*shape) * 0.02 - 0.01).cast(dtypes.bfloat16).contiguous().realize()
      v = (Tensor.rand(*shape) * 0.01).cast(dtypes.bfloat16).contiguous().realize()
      grad = (Tensor.rand(*shape) * 0.02 - 0.01).cast(dtypes.bfloat16).contiguous().realize()
    else:
      master = Tensor.full(shape, 0.05).contiguous().realize()
      m = Tensor.full(shape, -0.005, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.full(shape, 0.002, dtype=dtypes.bfloat16).contiguous().realize()
      grad = Tensor.full(shape, 0.007, dtype=dtypes.bfloat16).contiguous().realize()
    grad_scale = Tensor([0.5], dtype=dtypes.float32).contiguous().realize()
    weight = Tensor.empty(*shape, dtype=dtypes.fp8e4m3).realize()
    next_inv = Tensor.zeros(shape[0], dtype=dtypes.float32).contiguous().realize()
    inv_scale = Tensor([0.001 * (i+1) for i in range(shape[0])], dtype=dtypes.float32).contiguous().realize()
    inv_scale_ref = inv_scale.clone().realize()
    lr = Tensor([0.001], dtype=dtypes.float32).contiguous().realize()
    b1_t, b2_t = Tensor([b1], dtype=dtypes.float32).realize(), Tensor([b2], dtype=dtypes.float32).realize()

    master_ref, m_ref, v_ref, grad_ref = master.clone().realize(), m.clone().realize(), v.clone().realize(), grad.clone().realize()
    scaled_grad = (grad.float() * grad_scale).cast(dtypes.bfloat16).float()
    m_ref = b1 * m_ref.float() + (1.0 - b1) * scaled_grad
    v_ref = b2 * v_ref.float() + (1.0 - b2) * scaled_grad.square()
    update = lr * ((m_ref / (1.0 - b1_t)) / ((v_ref / (1.0 - b2_t)).sqrt() + eps))
    master_ref = master_ref - update - lr * wd * master_ref
    scale = inv_scale.reciprocal().reshape(shape[0], 1, 1)
    weight_ref = (master_ref * scale).clamp(-448.0, 448.0).cast(dtypes.fp8e4m3)
    next_inv_ref = (weight_ref.float().abs().max(axis=(1, 2)) * inv_scale * 1.1 + 1e-8) / 448.0

    from extra.llama_kernels.fused_fp8_adamw import fused_fp8_adamw
    master, weight, next_inv, m, v = fused_fp8_adamw(master, weight, next_inv, m, v, grad, grad_scale, inv_scale, lr, b1_t, b2_t,
      b1=b1, b2=b2, eps=eps, wd=wd)
    Tensor.realize(master, weight, next_inv, m, v, master_ref, weight_ref, next_inv_ref)
    with Context(DEBUG=0):
      self.assertTrue(inv_scale.allclose(inv_scale_ref, atol=0, rtol=0).item(), "delayed scale changed")
      self.assertTrue(grad.allclose(grad_ref, atol=0, rtol=0).item(), "gradient changed")
      self.assertTrue(m.allclose(m_ref.cast(dtypes.bfloat16), atol=6.2e-5, rtol=0).item(), "first moment mismatch")
      self.assertTrue(v.allclose(v_ref.cast(dtypes.bfloat16), atol=6.2e-5, rtol=0).item(), "second moment mismatch")
      self.assertTrue(master.allclose(master_ref, atol=2e-6, rtol=2e-6).item(), "master mismatch")
      weight_diff = (weight.float() - weight_ref.float()).abs()
      self.assertLessEqual(weight_diff.max().item(), 1.0, "FP8 weight error exceeds one quantization unit")
      self.assertLessEqual((weight_diff != 0).sum().item(), 1, "too many FP8 boundary differences")
      self.assertTrue(next_inv.allclose(next_inv_ref, atol=1e-8, rtol=1e-6).item(), "next scale mismatch")

  def test_values(self): self.run_update((2, 1024, 512), random=True)

  def test_llama31_8b_shapes(self):
    for name, shape in (("wo", (1, 4096, 4096)), ("wqkv", (1, 6144, 4096)),
                        ("w2", (1, 4096, 14336)), ("w13", (1, 28672, 4096))):
      with self.subTest(name=name): self.run_update(shape)

@unittest.skipUnless(has_hipcc() and Device.DEFAULT.split(":")[0] == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedBF16AdamW(unittest.TestCase):
  def test_update(self):
    from extra.llama_kernels.fused_bf16_adamw import fused_bf16_adamw
    for wd in (0.0, 0.1):
      with self.subTest(wd=wd):
        Tensor.manual_seed(0)
        shape, b1, b2, eps = (2, 4096), 0.9, 0.95, 1e-5
        master = (Tensor.rand(*shape) * 0.2 - 0.1).contiguous().realize()
        weight = master.cast(dtypes.bfloat16).contiguous().realize()
        m = (Tensor.rand(*shape) * 0.02 - 0.01).cast(dtypes.bfloat16).contiguous().realize()
        v = (Tensor.rand(*shape) * 0.01).cast(dtypes.bfloat16).contiguous().realize()
        grad = (Tensor.rand(*shape) * 0.02 - 0.01).cast(dtypes.bfloat16).contiguous().realize()
        grad_scale = Tensor([0.5], dtype=dtypes.float32).contiguous().realize()
        lr = Tensor([0.001], dtype=dtypes.float32).contiguous().realize()
        b1_t, b2_t = Tensor([b1], dtype=dtypes.float32).realize(), Tensor([b2], dtype=dtypes.float32).realize()

        master_ref, m_ref, v_ref = master.clone().realize(), m.clone().realize(), v.clone().realize()
        scaled_grad = (grad.float() * grad_scale).cast(dtypes.bfloat16).float()
        m_ref = b1 * m_ref.float() + (1.0 - b1) * scaled_grad
        v_ref = b2 * v_ref.float() + (1.0 - b2) * scaled_grad.square()
        update = lr * ((m_ref / (1.0 - b1_t)) / ((v_ref / (1.0 - b2_t)).sqrt() + eps))
        master_ref = master_ref - update - lr * wd * master_ref
        weight_ref = master_ref.cast(dtypes.bfloat16)

        master, weight, m, v = fused_bf16_adamw(master, weight, m, v, grad, grad_scale, lr, b1_t, b2_t,
                                                b1=b1, b2=b2, eps=eps, wd=wd)
        Tensor.realize(master, weight, m, v, master_ref, weight_ref, m_ref, v_ref)
        with Context(DEBUG=0):
          self.assertTrue(m.allclose(m_ref.cast(dtypes.bfloat16), atol=6.2e-5, rtol=0).item(), "first moment mismatch")
          self.assertTrue(v.allclose(v_ref.cast(dtypes.bfloat16), atol=6.2e-5, rtol=0).item(), "second moment mismatch")
          self.assertTrue(master.allclose(master_ref, atol=2e-6, rtol=2e-6).item(), "master mismatch")
          self.assertTrue(weight.allclose(weight_ref, atol=0, rtol=0).item(), "weight mismatch")

@unittest.skipUnless(has_hipcc() and Device.DEFAULT == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedAddRMSNorm(unittest.TestCase):
  def test_llama31_8b_forward(self):
    shape, eps = (2, 8192, 4096), 1e-5
    x = Tensor.full(shape, 0.25, dtype=dtypes.bfloat16).contiguous()
    residual = Tensor.full(shape, 0.125, dtype=dtypes.bfloat16).contiguous()
    weight = Tensor.full((shape[-1],), 0.75, dtype=dtypes.bfloat16).contiguous()
    amax = Tensor.full((), 1.0, dtype=dtypes.float32).contiguous()
    amax_out, ref_amax_out = Tensor.zeros((), dtype=dtypes.float32), Tensor.zeros((), dtype=dtypes.float32)
    Tensor.realize(x, residual, weight, amax, amax_out, ref_amax_out)

    fp8, h, x_normed, rrms = fused_add_rmsnorm_mul_quantize_fp8(x, residual, weight, amax, eps, FP8_DTYPE, amax_out)
    ref_h = x + residual
    ref_fp8, ref_x_normed, ref_rrms = fused_rmsnorm_mul_quantize_fp8(ref_h, weight, amax, eps, FP8_DTYPE, ref_amax_out)
    Tensor.realize(fp8, h, x_normed, rrms, ref_fp8, ref_h, ref_x_normed, ref_rrms, amax_out, ref_amax_out)

    with Context(DEBUG=0):
      for got, ref in ((h, ref_h), (x_normed, ref_x_normed), (rrms, ref_rrms)):
        self.assertTrue(got.allclose(ref, atol=2e-2, rtol=2e-2).item())
      self.assertTrue(fp8.cast(dtypes.float).allclose(ref_fp8.cast(dtypes.float), atol=2e-2, rtol=2e-2).item())
      self.assertTrue(amax_out.allclose(ref_amax_out, atol=0, rtol=0).item())

  def test_backward_with_residual_grad(self):
    shape = (2, 8192, 4096)
    grad_fp8 = Tensor.full(shape, 0.01, dtype=dtypes.bfloat16).contiguous().realize()
    x_normed = Tensor.full(shape, 0.125, dtype=dtypes.bfloat16).contiguous().realize()
    rrms = Tensor.full(shape[:-1], 0.5, dtype=dtypes.float32).contiguous().realize()
    weight = Tensor.full((shape[-1],), 0.25, dtype=dtypes.bfloat16).contiguous().realize()
    grad_h = Tensor.full(shape, 0.02, dtype=dtypes.bfloat16).contiguous().realize()
    amax = Tensor.full((), 2.0, dtype=dtypes.float32).contiguous().realize()
    grad_weight_shape = (NUM_WG, shape[-1])
    args = (grad_fp8, x_normed, rrms, weight, amax)
    grad_x, grad_weight, *_ = Tensor.custom_kernel(
      Tensor.empty(*shape, dtype=dtypes.bfloat16), Tensor.empty(*grad_weight_shape, dtype=dtypes.float32), *args, grad_h,
      fxn=functools.partial(custom_rmsnorm_bwd, has_h_grad=True, has_layer_num=False, dname=Device.DEFAULT))
    grad_x_ref, grad_weight_ref, *_ = Tensor.custom_kernel(
      Tensor.empty(*shape, dtype=dtypes.bfloat16), Tensor.empty(*grad_weight_shape, dtype=dtypes.float32), *args,
      fxn=functools.partial(custom_rmsnorm_bwd, has_h_grad=False, has_layer_num=False, dname=Device.DEFAULT))
    Tensor.realize(grad_x, grad_weight, grad_x_ref, grad_weight_ref)
    expected_grad_x = (grad_x_ref + grad_h).realize()

    with Context(DEBUG=0):
      self.assertTrue(grad_x.allclose(expected_grad_x, atol=0, rtol=0).item(), "residual add mismatch")
      self.assertTrue(grad_weight.allclose(grad_weight_ref, atol=0, rtol=0).item(), "weight grad changed")

@unittest.skipUnless(has_hipcc() and Device.DEFAULT == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedSiluMulBwdW13(unittest.TestCase):
  def test_llama31_8b_shape(self):
    shape, hidden, layer = (2, 8192, 28672), 14336, 7
    xw1 = Tensor.full((*shape[:-1], hidden), 0.125, dtype=dtypes.bfloat16)
    xw3 = Tensor.full((*shape[:-1], hidden), -0.25, dtype=dtypes.bfloat16)
    xw13 = xw1.cat(xw3, dim=-1).contiguous().realize()
    grad_x2 = Tensor.full((*shape[:-1], hidden), 0.01, dtype=dtypes.bfloat16).contiguous().realize()
    amax_state = Tensor.full((32,), 448.0, dtype=dtypes.float32).contiguous().realize()
    grad_amax_state = Tensor.full((32,), 1.0, dtype=dtypes.float32).contiguous().realize()
    grad_amax_next = Tensor.zeros(32, dtype=dtypes.float32).contiguous().realize()
    layer_num = Tensor([layer], dtype=dtypes.int32).contiguous().realize()

    grad_xw13, grad_amax_next, *_ = Tensor.custom_kernel(
      Tensor.empty(*shape, dtype=dtypes.fp8e4m3), grad_amax_next, xw13, grad_x2, amax_state, grad_amax_state, layer_num,
      fxn=functools.partial(custom_fused_bwd_w13, dname=Device.DEFAULT))
    Tensor.realize(grad_xw13, grad_amax_next)

    x1, x3, grad = Tensor([0.125]), Tensor([-0.25]), Tensor([0.01]).cast(dtypes.bfloat16).float()
    sig = x1.sigmoid()
    silu = x1 * sig
    grad_w1 = grad * (sig + silu * (1.0 - sig)) * x3
    grad_w3 = grad * silu
    grad_ref = Tensor.cat(grad_w1, grad_w3).mul(448.0).clamp(-448.0, 448.0).cast(dtypes.fp8e4m3).float().realize()
    amax_ref = Tensor.cat(grad_w1.abs(), grad_w3.abs()).max().item()
    grad_xw1, grad_xw3 = grad_xw13[..., :hidden].float(), grad_xw13[..., hidden:].float()
    Tensor.realize(grad_xw1, grad_xw3)

    with Context(DEBUG=0):
      self.assertEqual(grad_xw1.min().item(), grad_ref[0].item())
      self.assertEqual(grad_xw1.max().item(), grad_ref[0].item())
      self.assertEqual(grad_xw3.min().item(), grad_ref[1].item())
      self.assertEqual(grad_xw3.max().item(), grad_ref[1].item())
      self.assertAlmostEqual(grad_amax_next[layer].item(), amax_ref, places=7)
      self.assertEqual((grad_amax_next != 0).sum().item(), 1)

  def test_values(self):
    Tensor.manual_seed(0)
    hidden, layer = 14336, 7
    xw13 = (Tensor.randn(1, 1, hidden * 2) * 1.5).cast(dtypes.bfloat16).contiguous().realize()
    grad_x2 = (Tensor.randn(1, 1, hidden) * 0.02).cast(dtypes.bfloat16).contiguous().realize()
    amax_state = Tensor.full((32,), 448.0, dtype=dtypes.float32).contiguous().realize()
    grad_amax_state = Tensor.full((32,), 0.05, dtype=dtypes.float32).contiguous().realize()
    grad_amax_next = Tensor.zeros(32, dtype=dtypes.float32).contiguous().realize()
    layer_num = Tensor([layer], dtype=dtypes.int32).contiguous().realize()
    grad_xw13, grad_amax_next, *_ = Tensor.custom_kernel(
      Tensor.empty(*xw13.shape, dtype=dtypes.fp8e4m3), grad_amax_next, xw13, grad_x2, amax_state, grad_amax_state, layer_num,
      fxn=functools.partial(custom_fused_bwd_w13, dname=Device.DEFAULT))

    xw1, xw3 = xw13[..., :hidden].float(), xw13[..., hidden:].float()
    sig = xw1.sigmoid()
    silu = xw1 * sig
    grad_w1 = grad_x2.float() * (sig + silu * (1.0 - sig)) * xw3
    grad_w3 = grad_x2.float() * silu
    grad_ref = Tensor.cat(grad_w1, grad_w3, dim=-1).mul(8960.0).clamp(-448.0, 448.0).cast(dtypes.fp8e4m3)
    amax_ref = Tensor.cat(grad_w1.abs(), grad_w3.abs(), dim=-1).max()
    Tensor.realize(grad_xw13, grad_amax_next, grad_ref, amax_ref)

    with Context(DEBUG=0):
      self.assertTrue(grad_xw13.float().allclose(grad_ref.float(), atol=0, rtol=0).item())
      self.assertTrue(grad_amax_next[layer].allclose(amax_ref, atol=1e-7, rtol=0).item())
      self.assertEqual((grad_amax_next != 0).sum().item(), 1)

@unittest.skipUnless(has_hipcc() and Device.DEFAULT == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedQKVRoPE(unittest.TestCase):
  SHAPE = (2, 8192, 32, 8, 128)

  def rand_bf16(self, *shape:int) -> Tensor:
    return (Tensor.randn(*shape) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

  def freqs_cis(self) -> Tensor:
    _, N, _, _, D = self.SHAPE
    return precompute_freqs_cis(D, N * 2).cast(dtypes.bfloat16).clone().realize()

  def test_llama31_8b_forward(self):
    Tensor.manual_seed(0)
    _, N, H, H_KV, D = self.SHAPE
    GROUP = H // H_KV
    freqs_cis = self.freqs_cis()

    for B in (1, 2):
      with self.subTest(B=B):
        x = self.rand_bf16(B, N, H_KV * (GROUP + 2) * D)
        q, k, v = fused_qkv_rope(x, freqs_cis, H, H_KV, D)
        Tensor.realize(q, k, v)
        packed_ref = x.reshape(B, N, H_KV, GROUP + 2, D)
        q_ref = packed_ref[:, :, :, :GROUP].reshape(B, N, H, D)
        k_ref, v_ref = packed_ref[:, :, :, GROUP], packed_ref[:, :, :, GROUP+1]
        q_ref, k_ref = apply_rotary_emb(q_ref, k_ref, freqs_cis[:, :N])
        q_ref, k_ref, v_ref = q_ref.cast(dtypes.bfloat16), k_ref.cast(dtypes.bfloat16), v_ref.cast(dtypes.bfloat16)
        Tensor.realize(q_ref, k_ref, v_ref)

        with Context(DEBUG=0):
          self.assertTrue(q.allclose(q_ref, atol=2e-2, rtol=0).item(), "Q forward mismatch")
          self.assertTrue(k.allclose(k_ref, atol=2e-2, rtol=0).item(), "K forward mismatch")
          self.assertTrue(v.allclose(v_ref, atol=0, rtol=0).item(), "V forward mismatch")

  def test_llama31_8b_backward(self):
    Tensor.manual_seed(1)
    B, N, H, H_KV, D = self.SHAPE
    PARTIALS = 4
    GROUP = H // H_KV
    freqs_cis = self.freqs_cis()
    dq = self.rand_bf16(B, N, H, D)
    dk_partial = self.rand_bf16(B * PARTIALS, N, H_KV, D)
    dv_partial = self.rand_bf16(B * PARTIALS, N, H_KV, D)

    # Invert Flash Attention's dQ layout transform to reproduce its native buffer.
    dq_native = dq.transpose(1, 2).reshape(B, H, N//16, 4, 4, 4, 2, D//32, 2, 2) \
      .permute(0, 1, 2, 5, 6, 8, 7, 3, 4, 9).reshape(B, H, N, D).contiguous().realize()
    dx = Tensor.empty(B, N, H_KV * (GROUP + 2) * D, dtype=dtypes.bfloat16)
    arch = Device[Device.DEFAULT].renderer.target.arch
    fxn = functools.partial(custom_fused_qkv_rope_backward, device=Device.DEFAULT, arch=arch,
                            B=B, N=N, H=H, H_KV=H_KV, D=D)
    dx = Tensor.custom_kernel(dx, dq_native, dk_partial, dv_partial, freqs_cis, fxn=fxn)[0].realize()

    def inverse_rope(x:Tensor) -> Tensor:
      x = x.reshape(*x.shape[:-1], D//2, 2).float()
      cs = freqs_cis[:, :N].float()
      return Tensor.stack(x[..., 0] * cs[..., 0] + x[..., 1] * cs[..., 1],
                          -x[..., 0] * cs[..., 1] + x[..., 1] * cs[..., 0], dim=-1).flatten(-2).cast(dtypes.bfloat16)

    dq_ref = inverse_rope(dq).reshape(B, N, H_KV, GROUP, D)
    dk_ref = inverse_rope(dk_partial.float().reshape(B, PARTIALS, N, H_KV, D).sum(1).cast(dtypes.bfloat16)).unsqueeze(3)
    dv_ref = dv_partial.float().reshape(B, PARTIALS, N, H_KV, D).sum(1).cast(dtypes.bfloat16).unsqueeze(3)
    ref = Tensor.cat(dq_ref, dk_ref, dv_ref, dim=3).reshape(*dx.shape).realize()
    with Context(DEBUG=0): self.assertTrue(dx.allclose(ref, atol=2e-2, rtol=2e-2).item(), "backward mismatch")

  def test_flash_attention_backward_pre_scale(self):
    B, N, H, H_KV, D = self.SHAPE
    shape, dq_shape, delta_shape = (B, N, H, D), (B, H, N, D), (B, H, 1, N)
    attn = Tensor.full(shape, 0.125, dtype=dtypes.bfloat16).contiguous().realize()
    grad = Tensor.full(shape, 0.01, dtype=dtypes.bfloat16).contiguous().realize()
    amax = Tensor.full((32,), 2.0, dtype=dtypes.float32).contiguous().realize()
    layer_num = Tensor([3], dtype=dtypes.int32).contiguous().realize()
    scaled_ref = (grad.float() * (448.0 / (amax[layer_num[0]] + 1e-8))).cast(dtypes.bfloat16).contiguous().realize()
    arch = Device[Device.DEFAULT].renderer.target.arch
    common = dict(device=Device.DEFAULT, arch=arch, B=B, N=N, H=H, H_KV=H_KV, D=D)
    delta_ref, dq_ref, *_ = Tensor.custom_kernel(
      Tensor.empty(*delta_shape, dtype=dtypes.float32), Tensor.empty(*dq_shape, dtype=dtypes.bfloat16), attn, scaled_ref,
      fxn=functools.partial(custom_fa_backward_pre, **common))
    delta, dq, scaled, *_ = Tensor.custom_kernel(
      Tensor.empty(*delta_shape, dtype=dtypes.float32), Tensor.empty(*dq_shape, dtype=dtypes.bfloat16),
      Tensor.empty(*shape, dtype=dtypes.bfloat16), attn, grad, amax, layer_num,
      fxn=functools.partial(custom_fa_backward_pre, scale_do=True, has_layer_num=True, **common))
    Tensor.realize(delta_ref, dq_ref, delta, dq, scaled)
    with Context(DEBUG=0):
      self.assertTrue(scaled.allclose(scaled_ref, atol=0, rtol=0).item(), "scaled dO mismatch")
      self.assertTrue(delta.allclose(delta_ref, atol=0, rtol=0).item(), "delta changed")
      self.assertTrue(dq.allclose(dq_ref, atol=0, rtol=0).item(), "dQ initialization changed")

if __name__ == '__main__':
  unittest.main()
