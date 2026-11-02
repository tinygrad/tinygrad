import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_state_dict
from tinygrad.llm.model import TransformerConfig, HCConfig, IndexerConfig, QwenMTPBlock, _qsa_dense
from tinygrad.llm.kernels.amd import q8_quantize_cache_clear

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestMTPCache(unittest.TestCase):
  def test_selected_and_prefill_caches(self):
    if Device[Device.DEFAULT].target[0] != 11: self.skipTest("requires RDNA3")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    N, D, C, T = len(devices), 128, 4, 5
    Tensor.manual_seed(39)
    cfg = TransformerConfig(num_blocks=1, dim=D, hidden_dim=256, n_heads=N, n_kv_heads=N, norm_eps=1e-6,
                            vocab_size=128, head_dim=128, rope_theta=10000, rope_dim=128, v_head_dim=128, max_context=64,
                            qk_norm=128, attn_output_gate=True, hc=HCConfig(C, 1, 1e-6, low_rank=32),
                            indexer=IndexerConfig(2, 128, 32, kpool=4, rope_dim=128), qsa=True)
    block = QwenMTPBlock(cfg)
    block.hnorm["weight"] = Tensor.ones(C*D)
    for hc in (block.hc_attn, block.hc_ffn, block.hc_head): hc.norm["weight"] = Tensor.ones(C*D)
    for name, value in get_state_dict(block).items():
      axis = 0 if name in ("attn_q.weight", "attn_k.weight", "attn_v.weight") else 1 if name == "attn_output.weight" else None
      value.shard_(devices, axis=axis).realize()
    block._init_state(Tensor.zeros(1, T, C, D, device=devices))
    caches = [block.cache_k, block.indexer.cache_k, block.indexer.cache_pk]
    def copies(t): return [Tensor(UOp.from_buffer(b, t.dtype, d)).numpy().copy() for b, d in zip(t.uop.buffer.bufs, devices)]
    def clear(): Tensor.realize(*[c.assign(Tensor.zeros_like(c)) for c in caches])
    @TinyJit
    def select(e, h, start, row): return block.predict_selected(e, h, start, row).realize()
    old = _qsa_dense[0]
    _qsa_dense[0] = True
    try:
      # Isolate attention from the stateless FFN; one-query decode and batched attention have different floating-point reductions.
      with patch.object(block, "hc_ffn", lambda x, *args, **kwargs: x):
        for iteration in range(3):
          e, h = [Tensor.randn(*shape).shard(devices).realize() for shape in ((1, T, D), (1, T, C*D))]
          start = iteration*8
          clear()
          if start:
            pe, ph = [Tensor.randn(*shape).shard(devices).realize() for shape in ((1, start, D), (1, start, C*D))]
            block.prefill_cache(pe, ph, 0, start)[:, :, :1, :1].contiguous().realize()
          seed = [c.clone().realize() for c in caches]
          before = [copies(c) for c in seed]
          sp = UOp.variable("start_pos", 0, 32).bind(start)
          reference = copies(block.predict(e, h, sp).realize())
          state = [copies(c) for c in caches]
          def check_caches():
            for index, (actual, ref, prior) in enumerate(zip([copies(c) for c in caches], state, before)):
              rows, lo, hi = (64, start, start+T) if index < 2 else (16, start//4, (start+T+3)//4)
              for rank, (aa, bb, prev) in enumerate(zip(actual, ref, prior)):
                aa, bb, prev = [v.reshape(rows, -1) for v in (aa, bb, prev)]
                with self.subTest(iteration=iteration, cache=index, rank=rank):
                  # Independent FP32 fusion boundaries can flip an FP16 rounding tie. Untouched cache rows must stay bit-exact.
                  np.testing.assert_allclose(aa[lo:hi], bb[lo:hi], rtol=1e-3, atol=2e-6)
                  np.testing.assert_array_equal(aa[:lo], prev[:lo])
                  np.testing.assert_array_equal(aa[hi:], prev[hi:])
          q8_quantize_cache_clear()
          for row in range(T):
            Tensor.realize(*[c.assign(s) for c, s in zip(caches, seed)])
            actual = select(e, h, sp, UOp.variable("selected_row", 0, T-1).bind(row))
            for rank, (a, b) in enumerate(zip(copies(actual), reference)):
              b = b.reshape(1, T, C, D)[:, row:row+1].reshape(-1)
              with self.subTest(iteration=iteration, row=row, rank=rank):
                np.testing.assert_allclose(a, b, rtol=2e-3, atol=np.abs(b).max()*2e-3)
            check_caches()
          Tensor.realize(*[c.assign(s) for c, s in zip(caches, seed)])
          block.prefill_cache(e, h, sp, T)[:, :, :1, :1].contiguous().realize()
          check_caches()
          q8_quantize_cache_clear()
    finally: _qsa_dense[0] = old

if __name__ == "__main__": unittest.main()
