import unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import ASM_ATN
from extra.gemm.asm.cdna.atn import asm_sdpa, can_use_asm_atn
from test.helpers import needs_second_gpu

def is_cdna4(): return getattr(Device[Device.DEFAULT].renderer, "arch", "").startswith("gfx950")

class TestAsmAtn(unittest.TestCase):
  def setUp(self):
    if not is_cdna4():
      self.skipTest("ASM ATN only works on CDNA4 (MI350X)")
    self.prev_atn = ASM_ATN.value
    ASM_ATN.value = 1
  def tearDown(self):
    ASM_ATN.value = self.prev_atn

  def test_sdpa_forward(self):
    B, H, S, D = 8, 8, 8192, 128
    Tensor.manual_seed(0)
    q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    with Context(DEBUG=0): Tensor.realize(q, k, v)

    out_ref = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_asm = asm_sdpa(q, k, v)
    Tensor.realize(out_ref, out_asm)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)

  def test_sdpa_backward(self):
    B, H, S, D = 8, 8, 8192, 128
    Tensor.manual_seed(0)
    q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    with Context(DEBUG=0): Tensor.realize(q, k, v)

    dout = Tensor.empty(B, H, S, D, dtype=dtypes.bfloat16)
    with Context(DEBUG=0): Tensor.realize(dout)

    q_ref, k_ref, v_ref = [t.clone().requires_grad_() for t in [q, k, v]]
    out_ref = Tensor.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    dq_ref, dk_ref, dv_ref = Tensor.gradient(out_ref, q_ref, k_ref, v_ref, gradient=dout.clone())

    q_asm, k_asm, v_asm = [t.clone().requires_grad_() for t in [q, k, v]]
    out_asm = asm_sdpa(q_asm, k_asm, v_asm)
    dq_asm, dk_asm, dv_asm = Tensor.gradient(out_asm, q_asm, k_asm, v_asm, gradient=dout.clone())

    Tensor.realize(out_asm, out_ref, dq_asm, dk_asm, dv_asm, dq_ref, dk_ref, dv_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dq_asm.numpy(), dq_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dk_asm.numpy(), dk_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dv_asm.numpy(), dv_ref.numpy(), atol=atol, rtol=rtol)

  def test_can_use_asm_atn(self):
    B, H, S, D = 8, 8, 8192, 128
    q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    self.assertTrue(can_use_asm_atn(q, k, v, is_causal=True))
    self.assertFalse(can_use_asm_atn(q, k, v, is_causal=False))

  def test_can_use_asm_atn_wrong_dtype(self):
    B, H, S, D = 8, 8, 8192, 128
    q = Tensor.randn(B, H, S, D, dtype=dtypes.float16)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.float16)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.float16)
    self.assertFalse(can_use_asm_atn(q, k, v, is_causal=True))

  def test_can_use_asm_atn_wrong_head_dim(self):
    B, H, S, D = 8, 8, 8192, 64
    q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    self.assertFalse(can_use_asm_atn(q, k, v, is_causal=True))

  def test_can_use_asm_atn_wrong_seq_len(self):
    B, H, S, D = 8, 8, 1000, 128  # not divisible by 512
    q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
    self.assertFalse(can_use_asm_atn(q, k, v, is_causal=True))

  def test_sdpa_gqa_backward(self):
    """Test GQA (Grouped Query Attention) where Q has more heads than K/V."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8  # GQA ratio of 4 (like LLaMA 8B)
    Tensor.manual_seed(0)
    # K starts as (B, S, H_kv*D) then reshaped to (B, H_kv, S, D) - like in LLaMA
    k_flat = Tensor.randn(B, S, H_kv * D, dtype=dtypes.bfloat16, requires_grad=True)
    with Context(DEBUG=0): Tensor.realize(k_flat)
    k = k_flat.reshape(B, S, H_kv, D).permute(0, 2, 1, 3)

    q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    with Context(DEBUG=0): Tensor.realize(q, v)

    out_asm = asm_sdpa(q, k, v)
    out_asm.sum().backward()

    self.assertEqual(q.grad.shape, q.shape)
    self.assertEqual(k_flat.grad.shape, k_flat.shape)
    self.assertEqual(v.grad.shape, v.shape)

  def test_sdpa_gqa_backward_realize(self):
    """Test GQA backward with explicit realize (triggers scheduler)."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8  # GQA ratio of 4 (like LLaMA 8B)
    Tensor.manual_seed(0)
    q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    with Context(DEBUG=0): Tensor.realize(q, k, v)

    out_asm = asm_sdpa(q, k, v)
    loss = out_asm.sum()
    grads = Tensor.gradient(loss, q, k, v)
    # This triggers the scheduler - should not fail
    Tensor.realize(loss, *grads)

  def test_sdpa_gqa_backward_jit(self):
    """Test GQA backward with JIT (like training)."""
    from tinygrad import TinyJit
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8  # GQA ratio of 4 (like LLaMA 8B)

    @TinyJit
    def forward_backward(q, k, v):
      out = asm_sdpa(q, k, v)
      loss = out.sum()
      grads = Tensor.gradient(loss, q, k, v)
      Tensor.realize(loss, *grads)
      return loss, *grads

    Tensor.manual_seed(0)
    q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    with Context(DEBUG=0): Tensor.realize(q, k, v)

    # First call captures
    loss, dq, dk, dv = forward_backward(q, k, v)
    self.assertEqual(dq.shape, q.shape)
    self.assertEqual(dk.shape, k.shape)
    self.assertEqual(dv.shape, v.shape)

  def test_sdpa_gqa_multi_layer(self):
    """Test multiple GQA layers like in LLaMA."""
    B, S, D = 2, 512, 128
    H_q, H_kv = 4, 2  # GQA ratio of 2
    n_layers = 2

    Tensor.manual_seed(0)
    x = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16, requires_grad=True)
    ks = [Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True) for _ in range(n_layers)]
    vs = [Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True) for _ in range(n_layers)]
    with Context(DEBUG=0): Tensor.realize(x, *ks, *vs)

    # Multiple attention layers
    for k, v in zip(ks, vs):
      x = asm_sdpa(x, k, v)

    loss = x.sum()
    all_grads = Tensor.gradient(loss, *ks, *vs)
    Tensor.realize(loss, *all_grads)

  @needs_second_gpu
  def test_sdpa_gqa_multidevice_forward(self):
    """Test GQA (Grouped Query Attention) with multi-device sharding - forward only."""
    B, S, D = 2, 8192, 128
    H_q, H_kv = 32, 8  # GQA ratio of 4 (like LLaMA 8B)
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v)

    # shard on batch axis
    q = base_q.shard(GPUS, axis=0)
    k = base_k.shard(GPUS, axis=0)
    v = base_v.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    out_ref = Tensor.scaled_dot_product_attention(base_q, base_k, base_v, is_causal=True)
    Tensor.realize(out_asm, out_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)

  @needs_second_gpu
  def test_sdpa_gqa_multidevice_backward(self):
    """Test GQA (Grouped Query Attention) with multi-device sharding - forward + backward."""
    B, S, D = 2, 8192, 128
    H_q, H_kv = 32, 8  # GQA ratio of 4 (like LLaMA 8B)
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_dout = Tensor.ones(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v, base_dout)

    # shard on batch axis
    q = base_q.clone().requires_grad_(True).shard(GPUS, axis=0)
    k = base_k.clone().requires_grad_(True).shard(GPUS, axis=0)
    v = base_v.clone().requires_grad_(True).shard(GPUS, axis=0)
    dout = base_dout.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    dq_asm, dk_asm, dv_asm = Tensor.gradient(out_asm, q, k, v, gradient=dout)

    # reference on single device
    q_ref = base_q.clone().requires_grad_(True)
    k_ref = base_k.clone().requires_grad_(True)
    v_ref = base_v.clone().requires_grad_(True)
    out_ref = Tensor.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    dq_ref, dk_ref, dv_ref = Tensor.gradient(out_ref, q_ref, k_ref, v_ref, gradient=base_dout)

    Tensor.realize(out_asm, out_ref, dq_asm, dk_asm, dv_asm, dq_ref, dk_ref, dv_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dq_asm.numpy(), dq_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dk_asm.numpy(), dk_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dv_asm.numpy(), dv_ref.numpy(), atol=atol, rtol=rtol)

  @needs_second_gpu
  def test_sdpa_multidevice_forward(self):
    B, H, S, D = 2, 8, 8192, 128
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v)

    # shard on batch axis
    q = base_q.shard(GPUS, axis=0)
    k = base_k.shard(GPUS, axis=0)
    v = base_v.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    out_ref = Tensor.scaled_dot_product_attention(base_q, base_k, base_v, is_causal=True)
    Tensor.realize(out_asm, out_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)

  @needs_second_gpu
  def test_sdpa_multidevice_backward(self):
    B, H, S, D = 2, 8, 8192, 128
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      base_dout = Tensor.ones(B, H, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v, base_dout)

    # shard on batch axis
    q = base_q.clone().requires_grad_(True).shard(GPUS, axis=0)
    k = base_k.clone().requires_grad_(True).shard(GPUS, axis=0)
    v = base_v.clone().requires_grad_(True).shard(GPUS, axis=0)
    dout = base_dout.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    dq_asm, dk_asm, dv_asm = Tensor.gradient(out_asm, q, k, v, gradient=dout)

    # reference on single device
    q_ref = base_q.clone().requires_grad_(True)
    k_ref = base_k.clone().requires_grad_(True)
    v_ref = base_v.clone().requires_grad_(True)
    out_ref = Tensor.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    dq_ref, dk_ref, dv_ref = Tensor.gradient(out_ref, q_ref, k_ref, v_ref, gradient=base_dout)

    Tensor.realize(out_asm, out_ref, dq_asm, dk_asm, dv_asm, dq_ref, dk_ref, dv_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dq_asm.numpy(), dq_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dk_asm.numpy(), dk_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dv_asm.numpy(), dv_ref.numpy(), atol=atol, rtol=rtol)

  # ============================================================================
  # LLaMA 8B trainer exact shapes (8 GPUs, GQA with 32 query heads, 8 kv heads)
  # q: (8, 32, 8192, 128) axis=0, k: (8, 8, 8192, 128) axis=0, v: (8, 8, 8192, 128) axis=0
  # ============================================================================

  def test_llama8b_gqa_single_shard_forward(self):
    """Test single-shard shape from LLaMA 8B trainer (B=1)."""
    B, S, D = 1, 8192, 128
    H_q, H_kv = 32, 8

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()

    out = asm_sdpa(q, k, v)
    out.realize()

  @needs_second_gpu
  def test_llama8b_gqa_3gpu_forward(self):
    """Test 3-GPU sharding with LLaMA 8B GQA shapes."""
    B, S, D = 3, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(3))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()

    q = q.shard(GPUS, axis=0)
    k = k.shard(GPUS, axis=0)
    v = v.shard(GPUS, axis=0)

    out = asm_sdpa(q, k, v)
    out.realize()

  @needs_second_gpu
  def test_llama8b_gqa_4gpu_forward(self):
    """Test 4-GPU sharding with LLaMA 8B GQA shapes."""
    B, S, D = 4, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(4))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()

    q = q.shard(GPUS, axis=0)
    k = k.shard(GPUS, axis=0)
    v = v.shard(GPUS, axis=0)

    out = asm_sdpa(q, k, v)
    out.realize()

  @needs_second_gpu
  def test_llama8b_gqa_2gpu_forward(self):
    """Test 2-GPU sharding with LLaMA 8B GQA shapes."""
    B, S, D = 2, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = (f'{Device.DEFAULT}:0', f'{Device.DEFAULT}:1')

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()

    q = q.shard(GPUS, axis=0)
    k = k.shard(GPUS, axis=0)
    v = v.shard(GPUS, axis=0)

    out = asm_sdpa(q, k, v)
    out.realize()

  @needs_second_gpu
  def test_llama8b_gqa_8gpu_forward_tiny(self):
    """Test exact shapes from LLaMA 8B trainer - forward only, no reference."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous().realize()

    q = q.shard(GPUS, axis=0)
    k = k.shard(GPUS, axis=0)
    v = v.shard(GPUS, axis=0)

    out = asm_sdpa(q, k, v)
    out.realize()

  @needs_second_gpu
  def test_llama8b_gqa_8gpu_backward_tiny(self):
    """Test exact shapes from LLaMA 8B trainer - backward only, no reference."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16, requires_grad=True).shard(GPUS, axis=0)
      k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True).shard(GPUS, axis=0)
      v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16, requires_grad=True).shard(GPUS, axis=0)

    out = asm_sdpa(q, k, v)
    loss = out.sum()
    grads = Tensor.gradient(loss, q, k, v)
    Tensor.realize(loss, *grads)

  @needs_second_gpu
  def test_llama8b_gqa_8gpu_forward(self):
    """Test exact shapes from LLaMA 8B trainer - forward with reference."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v)

    q = base_q.shard(GPUS, axis=0)
    k = base_k.shard(GPUS, axis=0)
    v = base_v.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    out_ref = Tensor.scaled_dot_product_attention(base_q, base_k, base_v, is_causal=True)
    Tensor.realize(out_asm, out_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)

  @needs_second_gpu
  def test_llama8b_gqa_8gpu_backward(self):
    """Test exact shapes from LLaMA 8B trainer - backward with reference."""
    B, S, D = 8, 8192, 128
    H_q, H_kv = 32, 8
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(B))

    Tensor.manual_seed(0)
    with Context(DEBUG=0):
      base_q = Tensor.randn(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      base_k = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_v = Tensor.randn(B, H_kv, S, D, dtype=dtypes.bfloat16).contiguous()
      base_dout = Tensor.ones(B, H_q, S, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(base_q, base_k, base_v, base_dout)

    q = base_q.clone().requires_grad_(True).shard(GPUS, axis=0)
    k = base_k.clone().requires_grad_(True).shard(GPUS, axis=0)
    v = base_v.clone().requires_grad_(True).shard(GPUS, axis=0)
    dout = base_dout.shard(GPUS, axis=0)

    out_asm = asm_sdpa(q, k, v)
    dq_asm, dk_asm, dv_asm = Tensor.gradient(out_asm, q, k, v, gradient=dout)

    q_ref = base_q.clone().requires_grad_(True)
    k_ref = base_k.clone().requires_grad_(True)
    v_ref = base_v.clone().requires_grad_(True)
    out_ref = Tensor.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    dq_ref, dk_ref, dv_ref = Tensor.gradient(out_ref, q_ref, k_ref, v_ref, gradient=base_dout)

    Tensor.realize(out_asm, out_ref, dq_asm, dk_asm, dv_asm, dq_ref, dk_ref, dv_ref)

    atol, rtol = 2e-2, 2e-2
    np.testing.assert_allclose(out_asm.numpy(), out_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dq_asm.numpy(), dq_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dk_asm.numpy(), dk_ref.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(dv_asm.numpy(), dv_ref.numpy(), atol=atol, rtol=rtol)

if __name__ == "__main__":
  unittest.main()
