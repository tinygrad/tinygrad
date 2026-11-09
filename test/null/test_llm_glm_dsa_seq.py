import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor
from tinygrad.llm import glm_dsa
from tinygrad.llm.glm_dsa import Indexer, MLATransformerBlock, Transformer


class TestGLMDSASequence(unittest.TestCase):
  def setUp(self):
    self.devs = ("CPU",)

  def test_direct_key_prefill_region(self):
    idx = Indexer.__new__(Indexer)
    idx.idx = SimpleNamespace(kpool=1, n_heads=16, head_dim=4, top_k=2, rope_dim=2, rope_interleave=False)
    idx.max_context, idx.shard, idx.cache_dk = 16, 0, Tensor.zeros(32, 4, dtype="half")
    idx.k_norm = idx.attn_k = lambda x: x
    idx.attn_q_b = lambda x: x.unsqueeze(1).expand(x.shape[0], 16, x.shape[1], x.shape[2]).reshape(*x.shape[:2], 16*x.shape[2])
    idx.proj = lambda x: Tensor.ones(*x.shape[:2], 16)
    x = Tensor(np.arange(1, 9, dtype=np.float16).reshape(1, 2, 4))
    q = Tensor.zeros(1, 2, 4)
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = 8
    try:
      with patch.object(glm_dsa, "indexer_decode_index_rows", return_value=Tensor.zeros(2, 4, dtype="int32")) as score:
        idx.dsa_picks(x, q, 2, Tensor([[1, 0], [1, 0]]))
      score.assert_called_once()
      self.assertEqual(score.call_args.kwargs["pbase"], 8)
      self.assertEqual(score.call_args.kwargs["width"], 16)
      score.call_args.args[1].realize()  # the mocked scorer does not carry pk's store into its return graph
      # start=2, base=8: the two direct-key rows land at 10..12, not 2..4.
      np.testing.assert_array_equal(idx.cache_dk[10:12].numpy(), x[0].half().numpy())
      np.testing.assert_array_equal(idx.cache_dk[2:4].numpy(), np.zeros((2, 4), dtype=np.float16))
    finally: glm_dsa._seq_base[0] = old_base

  def test_token_after_prefill_uses_region_base(self):
    model = Transformer.__new__(Transformer)
    model._token_buf = Tensor.full((1, 32), -1, dtype="int32")
    model.output_norm = lambda x: x
    model._sample = lambda h, temp, device: Tensor([9], dtype="int32")
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = 8
    try:
      tokens = model._token_buf[:, 8:10]
      model._tail(Tensor.zeros(1, 2, 3), tokens, 0, None).realize()
      self.assertEqual(model._token_buf[0, 10].item(), 9)
      self.assertEqual(model._token_buf[0, 2].item(), -1)
    finally: glm_dsa._seq_base[0] = old_base

  def test_dsa_stash_roundtrip(self):
    def t(shape, axis=None):
      x = Tensor(np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
      return x.shard(self.devs, axis) if axis is not None else x.to(self.devs)

    model = Transformer.__new__(Transformer)
    model.config = SimpleNamespace(spec_drafts=2, cache_rows=32, kv_lora_rank=512)
    model.token_embd = SimpleNamespace(weight=SimpleNamespace(device=self.devs))
    model.SEQ_MOVE, model.seq_rows = 32, lambda: 32
    idx = Indexer.__new__(Indexer)
    idx.idx, idx.shard, idx.cache_dk = SimpleNamespace(kpool=1), 0, t((32, 2))
    block = MLATransformerBlock.__new__(MLATransformerBlock)
    block.indexer, block.cache_s = idx, t((32, 2), axis=0)
    model.blk, model.mtp = [block], None
    model.seq_spec_h, model.seq_ckpt_h = t((12, 3)), t((2, 3))
    model.seq_ckpt_rings = []
    sources = [block.cache_s, idx.cache_dk, model.seq_spec_h, model.seq_ckpt_h]
    values = [x.numpy().copy() for x in sources]
    with patch.dict(os.environ, {"SEQ_STASH_GB": "0.000002", "SEQ_STASH_ENTRIES": "1"}): model._seq_stash_init()
    self.assertEqual(model.stash_entries, 1)
    self.assertGreater(model.stash_chunks, 0)
    self.assertEqual(model._stash_rings, [])

    model._stash_kv_copy(True, 1, 0, 0)
    model._stash_state_copy(True, 0, 0, 0, 0, 0)
    for x in sources: x.assign(Tensor.zeros_like(x)).realize()
    model._stash_kv_copy(False, 1, 0, 0)
    model._stash_state_copy(False, 1, 0, 0, 0, 0)
    np.testing.assert_array_equal(block.cache_s.numpy(), values[0])
    np.testing.assert_array_equal(idx.cache_dk.numpy(), values[1])
    spec, ckpt = np.zeros((12, 3), dtype=np.float16), np.zeros((2, 3), dtype=np.float16)
    spec[6], ckpt[1] = values[2][0], values[3][0]
    np.testing.assert_array_equal(model.seq_spec_h.numpy(), spec)
    np.testing.assert_array_equal(model.seq_ckpt_h.numpy(), ckpt)


if __name__ == '__main__': unittest.main()
