import unittest, time

from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.jit import TinyJit
import numpy as np

from extra.thunder.amd.fa import flash_attention

def assert_allclose(cmp:Tensor, ref:Tensor, **kwargs) -> None:
  if Device.DEFAULT == "NULL": Tensor.realize(cmp, ref)
  else: np.testing.assert_allclose(cmp.numpy(), ref.numpy(), **kwargs)

class TestFA(unittest.TestCase):
  def setUp(self):
    arch = getattr(Device[Device.DEFAULT].renderer, "arch", "")
    if not arch.startswith("gfx9"):
      self.skipTest(f"arch {arch} not supported")

  def test_fast_fa_causal(self):
    B, N, H, H_KV, D = 1, 8192, 32, 8, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    fa_jitted = TinyJit(flash_attention)

    for _ in range(10):
      st = time.perf_counter()
      out = fa_jitted(q, k, v, is_causal=True)
      et = time.perf_counter() - st
      attn_flops = 2 * B * H * N * N * D + \
                   4 * B * H * N * N + \
                   2 * B * H * N * N * D
      print(f"{attn_flops/(et*1e9):2f} GFLOPS")
    out = out.float().transpose(1, 2)

    ref = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float().transpose(1, 2)

    assert_allclose(out, ref, atol=2e-2, rtol=2e-2)

  def test_fast_fa_bwd_causal(self):
    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 1, 8192, 32, 8, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)

      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)

    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attention(q_, k_, v_, is_causal=True)
    out = out.float().transpose(1, 2)
    out.backward(do)
    Tensor.realize(q.grad, k.grad, v.grad)

    with Context(DEBUG=0):
      q_ref = q.detach().clone().requires_grad_(True)
      k_ref = k.detach().clone().requires_grad_(True)
      v_ref = v.detach().clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = q_ref_.scaled_dot_product_attention(k_ref_, v_ref_, is_causal=True, enable_gqa=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    assert_allclose(q.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    assert_allclose(v.grad, v_ref.grad, atol=2e-2, rtol=2e-2)
    assert_allclose(k.grad, k_ref.grad, atol=6e-2, rtol=2e-2)

  def test_fast_fa_bwd_causal_jitted(self):
    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 1, 8192, 32, 8, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)

      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)

    def fn(q, k, v, do):
      q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
      out = flash_attention(q_, k_, v_, is_causal=True)
      out = out.float().transpose(1, 2)
      out.backward(do)
      Tensor.realize(out, q.grad, k.grad, v.grad)
      return q.grad, k.grad, v.grad

    fn_jitted = TinyJit(fn)

    for _ in range(10):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)
      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)
      q.grad, k.grad, v.grad = fn_jitted(q, k, v, do)

    with Context(DEBUG=0):
      q_ref = q.detach().clone().requires_grad_(True)
      k_ref = k.detach().clone().requires_grad_(True)
      v_ref = v.detach().clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = flash_attention(q_ref_, k_ref_, v_ref_, is_causal=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    assert_allclose(q.grad, q_ref.grad, atol=3e-3, rtol=3e-3)
    assert_allclose(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(v.grad, v_ref.grad, atol=1e-5, rtol=1e-5)

  def test_fast_fa_bwd_multidevice(self):
    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 2, 1024, 32, 8, 128
    GPUS = tuple(f"AMD:{i}" for i in range(B))

    with Context(DEBUG=0):
      base_q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      base_k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      base_v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()

      base_do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()

    with Context(DEBUG=0):
      q = base_q.clone().requires_grad_(True).shard(GPUS, axis=0)
      k = base_k.clone().requires_grad_(True).shard(GPUS, axis=0)
      v = base_v.clone().requires_grad_(True).shard(GPUS, axis=0)
      Tensor.realize(q, k, v)

      do = base_do.clone().shard(GPUS, axis=0)
      Tensor.realize(do)

    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attention(q_, k_, v_, is_causal=True)
    out = out.float().transpose(1, 2)
    out.backward(do)
    Tensor.realize(q.grad, k.grad, v.grad)

    with Context(DEBUG=0):
      q_ref = base_q.clone().requires_grad_(True)
      k_ref = base_k.clone().requires_grad_(True)
      v_ref = base_v.clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

      do_ref = base_do.clone()
      Tensor.realize(do_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = flash_attention(q_ref_, k_ref_, v_ref_, is_causal=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do_ref)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    assert_allclose(q.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(v.grad, v_ref.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
  unittest.main()
