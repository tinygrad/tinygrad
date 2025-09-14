import unittest
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.apps.llm import apply_rope

class TestAttention(unittest.TestCase):
  def _get_attn_and_scheduler(self, BS=10, seqlen=4, dim=100, dtype=dtypes.half):
    q = Tensor.ones(BS, seqlen, dim, dtype=dtype).contiguous().realize()
    k = Tensor.ones(BS, seqlen, dim, dtype=dtype).contiguous().realize()
    v = Tensor.ones(BS, seqlen, dim, dtype=dtype).contiguous().realize()
    attn = q.scaled_dot_product_attention(k, v)
    sched = attn.schedule()
    return attn, sched
  def test_half_qkv_buffers(self):
    _, sched = self._get_attn_and_scheduler()
    # attention has 5 kernels now
    self.assertEqual(len(sched), 5)
    softmax_inputs = sched[1:4]
    for si in softmax_inputs:
      assert all(b.dtype == dtypes.half for b in si.bufs), f"non half {si.bufs=}"
  def test_uint_qkv_buffers(self):
    _, sched = self._get_attn_and_scheduler(dtype=dtypes.uint)
    # attention has 6 kernels now
    self.assertEqual(len(sched), 6)
    qkv_uint_scheds = [si for si in sched if all(b.dtype == dtypes.uint for b in si.bufs)]
    assert len(qkv_uint_scheds) >= 1, f"no uint found in scheduler buffers {sched[0].bufs=} "
    found_cast_uint_to_float = any(
        any(b.dtype == dtypes.uint for b in si.bufs) and any(b.dtype == dtypes.float for b in si.bufs)
        for si in sched
    )
    assert found_cast_uint_to_float, f"no cast found from uint to float in scheduler buffers {sched=}"

  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_apply_rope_jit_prune(self):
    def rope_fn(x_in, pos): return apply_rope(x_in, pos)
    rope_noprune = TinyJit(rope_fn)
    rope_prune = TinyJit(rope_fn, prune=True)

    v_pos = UOp.variable("start_pos", 0, 100)
    for _ in range(3):
      rope_noprune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
      rope_prune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
    noprune_size = len(rope_noprune.captured.jit_cache)
    prune_size = len(rope_prune.captured.jit_cache)

    self.assertGreater(noprune_size, prune_size)
    self.assertGreaterEqual(noprune_size, 3)
    self.assertEqual(prune_size, 1)

if __name__ == '__main__':
  unittest.main()
