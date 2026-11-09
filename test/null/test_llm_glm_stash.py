import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor
from tinygrad.llm.glm5next import GatedDeltaNetBlock, Indexer, MLATransformerBlock, Transformer, replay_q


class TestGLMStash(unittest.TestCase):
  def setUp(self):
    self.devs = ("CPU",)

    def t(shape, axis=None):
      x = Tensor(np.arange(math.prod(shape), dtype=np.float16).reshape(shape))
      return x.shard(self.devs, axis) if axis is not None else x.to(self.devs)

    self.model = Transformer.__new__(Transformer)
    self.model.config = SimpleNamespace(spec_drafts=2, cache_rows=32, kv_lora_rank=512)
    self.model.token_embd = SimpleNamespace(weight=SimpleNamespace(device=self.devs))
    self.model.SEQ_MOVE = 32
    self.model.seq_rows = lambda: 32

    self.gdn = GatedDeltaNetBlock.__new__(GatedDeltaNetBlock)
    self.gdn.pool_rec, self.gdn.pool_conv = t((4, 3)), [t((4, 2))]
    self.gdn.slot_rec, self.gdn.slot_conv = [t((2, 3))], [[t((2, 2))]]
    self.gdn.pool_rp = t((2*replay_q(2), 3))
    self.idx = Indexer.__new__(Indexer)
    self.idx.idx = SimpleNamespace(kpool=2)
    self.idx.ring = t((2, 2, 2))
    self.idx.cache_pk = t((16, 2))
    self.idx.pool_ring = t((4, 2, 2))
    self.idx.shard = 0
    self.mla = MLATransformerBlock.__new__(MLATransformerBlock)
    self.mla.indexer = self.idx
    self.mla.cache_s = t((32, 2), axis=0)

    self.model.blk, self.model.mtp = [self.gdn, self.mla], self.mla
    self.model.seq_spec_h = t((12, 3))
    self.model.seq_ckpt_h = t((2, 3))
    self.model.seq_ckpt_states = [[t((4, 3)), t((4, 2))]]
    self.ckpt_ring = t((4, 2, 2))
    self.model.seq_ckpt_rings = [(self.idx, self.ckpt_ring)]

    # Every tensor participates in the round trip, including current/checkpoint rings and replay inputs.
    self.states = [self.gdn.pool_rec, *self.gdn.pool_conv, self.gdn.pool_rp, self.idx.cache_pk,
                   self.idx.pool_ring, self.ckpt_ring, self.mla.cache_s,
                   self.model.seq_spec_h, self.model.seq_ckpt_h, *self.model.seq_ckpt_states[0]]
    self.snapshots = {}

  def zero_like(self, tensor):
    return Tensor(np.zeros_like(tensor.numpy()), device=tensor.device)

  def reset_sources(self):
    self.gdn.pool_rec = self.zero_like(self.gdn.pool_rec)
    self.gdn.pool_conv = [self.zero_like(t) for t in self.gdn.pool_conv]
    self.gdn.pool_rp = self.zero_like(self.gdn.pool_rp)
    self.idx.cache_pk = self.zero_like(self.idx.cache_pk)
    self.idx.pool_ring = self.zero_like(self.idx.pool_ring)
    self.ckpt_ring = self.zero_like(self.ckpt_ring)
    self.mla.cache_s = Tensor(np.zeros_like(self.mla.cache_s.numpy())).shard(self.devs, 0)
    self.model.seq_spec_h = self.zero_like(self.model.seq_spec_h)
    self.model.seq_ckpt_h = self.zero_like(self.model.seq_ckpt_h)
    self.model.seq_ckpt_states = [[self.zero_like(t) for t in self.model.seq_ckpt_states[0]]]

  def fill_and_save(self):
    with patch.dict(os.environ, {"SEQ_STASH_GB": "0.000002", "SEQ_STASH_ENTRIES": "1"}):
      self.model._seq_stash_init()
    self.assertGreater(self.model.stash_chunks, 0)
    # Save every field to the host before replacing sources with clean zero tensors.
    for tensor in self.states: self.snapshots[tensor] = tensor.numpy().copy()

  def test_init_and_kv_roundtrip(self):
    self.fill_and_save()
    self.assertEqual(self.model.stash_entries, 1)
    self.assertGreater(self.model.stash_chunk_nbytes, 0)
    self.assertGreater(self.model.stash_entry_nbytes, 0)
    self.model._stash_kv_copy(True, 1, 0, 0)
    before_reset = self.idx.cache_pk, self.mla.cache_s
    self.reset_sources()
    self.model._stash_kv_copy(False, 1, 0, 0)
    for tensor in before_reset:
      np.testing.assert_array_equal(tensor.numpy(), self.snapshots[tensor])

  def test_state_and_ring_replay_roundtrip(self):
    self.fill_and_save()
    values = {t: t.numpy().copy() for t in self.states}
    # Copy slot 0 -> entry 0. Replace sources with clean zero buffers, then restore entry 0 into slot 1.
    self.model._stash_state_copy(True, 0, 0, 0, 0, 0)
    self.reset_sources()
    self.model._stash_state_copy(False, 1, 0, 0, 0, 0)
    for tensor in self.states:
      expected = values[tensor]
      if tensor is self.idx.pool_ring:
        expected = np.concatenate([np.zeros_like(tensor.numpy()[:2]), expected[:2]])
      elif tensor is self.gdn.pool_rp:
        expected = np.concatenate([np.zeros_like(tensor.numpy()[:replay_q(2)]), expected[:replay_q(2)]])
      elif tensor is self.model.seq_spec_h or tensor is self.model.seq_ckpt_h:
        expected = np.concatenate([np.zeros_like(tensor.numpy()[:tensor.shape[0]//2]), expected[:tensor.shape[0]//2]])
      else:
        expected = values[tensor]
      np.testing.assert_array_equal(tensor.numpy(), expected)


if __name__ == '__main__': unittest.main()
