import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.llm import glm_dsa
from tinygrad.llm.glm_dsa import Indexer, MLATransformerBlock, Transformer


class TestGLMDSASequence(unittest.TestCase):
  def setUp(self):
    self.devs = ("CPU",)

  def test_region_alignment_drives_move_chunks(self):
    self.assertEqual(Transformer.SEQ_MOVE, Transformer.REGION_ALIGN)
    self.assertEqual(Transformer.REGION_ALIGN % (64*14), 0)

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

  def test_seq_embedding_is_a_gather(self):
    # (a view of the token buffer at seq_base+start_pos (two variables): its embedding a lookup of the rows, not a reduce over the vocabulary)
    model = Transformer.__new__(Transformer)
    model.token_embd = nn.Embedding(1024, 64)
    buf = Tensor.zeros(1, 4096, dtype="int32").contiguous().realize()
    vb, sp = UOp.variable("seq_base", 0, 2048), UOp.variable("start_pos", 0, 1023)
    nt = UOp.variable("toks", 1, 256)
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = vb
    try: x = model._embed(buf[:, vb+sp:vb+sp+nt])
    finally: glm_dsa._seq_base[0] = old_base
    kernels = [si for si in x.schedule_linear().src if si.src[0].op is Ops.SINK]
    # (the reduce of a one-hot: the one-hot (tokens, vocabulary) of bools stored by a kernel of its own)
    stores = [u for k in kernels for u in k.src[0].toposort() if u.op is Ops.STORE]
    self.assertFalse(any(u.src[1].dtype == dtypes.bool for u in stores), "the embedding is a reduce of a one-hot of the vocabulary")

  def test_main_path_token_buffer_has_the_regions(self):
    # (the main path making the token buffer first (generate, warmup): GLM_SEQ's regions in it too, not max_context+1)
    model = Transformer.__new__(Transformer)
    model.max_context, model.config = 65536, SimpleNamespace(cache_rows=107520)
    self.assertEqual(model._main_token_rows(), model._token_rows())
    self.assertGreaterEqual(model._main_token_rows(), model.seq_rows() + Transformer.REGION_ALIGN + 1)
    model.config = SimpleNamespace(cache_rows=0)
    self.assertEqual(model._main_token_rows(), 65537)

  def test_q8_pack_needs_whole_warps_of_groups(self):
    # (MLA's 576 dims (18 groups of 32): the kernels pack 8 groups a warp, the last 2 would be left out: refused, mla_sparse_seq sends half)
    from tinygrad.llm.kernels.amd_glm_dsa import _q8_pack_kernel, _q8_unpack_kernel
    self.assertRaises(AssertionError, _q8_pack_kernel, Tensor.empty(4, 18, 9, dtype="uint32").uop, Tensor.empty(4, 576, dtype="half").uop)
    self.assertRaises(AssertionError, _q8_unpack_kernel, Tensor.empty(4, 576, dtype="half").uop, Tensor.empty(4, 18, 9, dtype="uint32").uop)

  def test_dsa_stash_roundtrip(self):
    def t(shape, axis=None):
      x = Tensor(np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
      return x.shard(self.devs, axis) if axis is not None else x.to(self.devs)

    model = Transformer.__new__(Transformer)
    model.config = SimpleNamespace(spec_drafts=2, cache_rows=128, kv_lora_rank=512)
    model.token_embd = SimpleNamespace(weight=SimpleNamespace(device=self.devs))
    model.SEQ_MOVE, model.seq_rows = 32, lambda: 128
    idx = Indexer.__new__(Indexer)
    idx.idx, idx.shard, idx.cache_dk = SimpleNamespace(kpool=1), 0, t((128, 2))
    block = MLATransformerBlock.__new__(MLATransformerBlock)
    block.indexer, block.cache_s = idx, t((128, 2), axis=0)
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
    model._stash_kv_copy(False, 1, 32, 0)  # region row 32 restore from host chunk 0
    model._stash_state_copy(False, 1, 0, 0, 0, 0)
    for actual, saved in zip((block.cache_s, idx.cache_dk), values):
      expected = np.zeros_like(actual.numpy())
      expected[32:64] = saved[:32]
      np.testing.assert_array_equal(actual.numpy(), expected)
    spec, ckpt = np.zeros((12, 3), dtype=np.float16), np.zeros((2, 3), dtype=np.float16)
    spec[6], ckpt[1] = values[2][0], values[3][0]
    np.testing.assert_array_equal(model.seq_spec_h.numpy(), spec)
    np.testing.assert_array_equal(model.seq_ckpt_h.numpy(), ckpt)


if __name__ == '__main__': unittest.main()
