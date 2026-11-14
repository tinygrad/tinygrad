import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes, UOp, Device
from tinygrad.llm import glm_dsa
from tinygrad.llm.glm_dsa import apply_rope, mla_q_cat, Indexer
from tinygrad.llm.kernels import amd_glm_dsa

def _bits(t:Tensor) -> np.ndarray: return t.numpy().view(np.uint32 if t.dtype == dtypes.float32 else np.uint16)

class TestMlaQCat(unittest.TestCase):
  # (MLA_QCAT: the matmul and the roped q_rope stored into one buffer's slices = the cat of the realized matmul, bit for bit)
  def _check(self, devs):
    Tensor.manual_seed(3)
    H, R, nope, rope, L, M = 2*len(devs or (1,)), 40, 24, 8, 32, 256
    q = Tensor.randn(1, R, H*(nope+rope)).cast(dtypes.half)
    kb, fc = Tensor.randn(H, L, nope).cast(dtypes.half), Tensor.randn(M, rope)
    if devs: q, kb, fc = q.shard(devs, 2), kb.shard(devs, 0), fc.shard(devs, None)
    sp, T = UOp.variable("start_pos", 0, M-1).bind(7), UOp.variable("toks", 1, R).bind(R-3)
    qh = q.reshape(1, R, H, nope+rope).transpose(1, 2)
    q_rope = apply_rope(qh[..., nope:].float(), fc[sp:sp+T].pad_to((R, rope)), True)
    def mm() -> Tensor: return qh[..., :nope].half().contiguous().matmul(kb.transpose(-1, -2), dtype=dtypes.float)
    ref = mm().contiguous().cat(q_rope.cast(dtypes.float), dim=-1).contiguous().realize()
    new = mla_q_cat(mm(), q_rope).contiguous().realize()
    self.assertEqual(new.shape, (1, H, R, L+rope))
    np.testing.assert_array_equal(_bits(new), _bits(ref))
  def test_one_device(self): self._check(None)
  def test_sharded_heads(self): self._check(tuple(f"{Device.DEFAULT}:{i}" for i in range(2)))

class TestHalfRowsG(unittest.TestCase):
  # (HALF_ROWS_G: the row max in G parts, then of the parts: the same scales and halves as one reduce, rows past HALF_ROWS_MAX included)
  def test_same_bits(self):
    Tensor.manual_seed(5)
    x = (Tensor.randn(1, 1024, 1024) * Tensor([1.0, 3e5, 1e-3, 7e4] * 256).reshape(1, 1024, 1)).realize()
    g0 = amd_glm_dsa.HALF_ROWS_G
    try:
      amd_glm_dsa.HALF_ROWS_G = 1
      h1, s1 = (t.realize() for t in amd_glm_dsa._half_rows(x, 1024))
      for g in (16, 4):
        amd_glm_dsa.HALF_ROWS_G = g
        h2, s2 = amd_glm_dsa._half_rows(x, 1024)
        np.testing.assert_array_equal(_bits(s2.realize()), _bits(s1))
        np.testing.assert_array_equal(_bits(h2.realize()), _bits(h1))
    finally: amd_glm_dsa.HALF_ROWS_G = g0
    self.assertTrue((s1.numpy() > 1).any())

  def _vs_one(self, x:Tensor, tokens:int, splits:int, G:int):
    g0 = amd_glm_dsa.HALF_ROWS_G
    try:
      amd_glm_dsa.HALF_ROWS_G = 1
      h1, s1 = (t.realize() for t in amd_glm_dsa._half_rows(x, tokens, splits))
      amd_glm_dsa.HALF_ROWS_G = 16
      self.assertEqual(amd_glm_dsa._half_rows_parts(tokens, x.shape[-1] // splits), G)
      h2, s2 = amd_glm_dsa._half_rows(x, tokens, splits)
      np.testing.assert_array_equal(_bits(s2.realize()), _bits(s1))
      np.testing.assert_array_equal(_bits(h2.realize()), _bits(h1))
    finally: amd_glm_dsa.HALF_ROWS_G = g0
    self.assertTrue((s1.numpy() > 1).any())

  def test_parts_divide_width(self):
    # (a width not of 16 x 32s: the most parts that divide it, 2816 = 11 x 256)
    Tensor.manual_seed(6)
    self._vs_one((Tensor.randn(1, 512, 2816) * Tensor([1.0, 3e5, 1e-3, 7e4] * 128).reshape(1, 512, 1)).realize(), 512, 1, 11)

  def test_splits_parts(self):
    # (a row parallel layer's input: every device's part of a row its own scale, the parts inside the device's part)
    Tensor.manual_seed(7)
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    x = Tensor.randn(1, 512, 2*2816) * Tensor([1.0, 3e5, 1e-3, 7e4] * 128).reshape(1, 512, 1)
    self._vs_one(x.shard(devs, 2).realize(), 512, 2, 11)

  def test_row_split_parts(self):
    # (a row split over the devices (axis 1): G parts of every device's part (11 of 2816), then the max over the devices: the one reduce's bits)
    Tensor.manual_seed(8)
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    x = Tensor.randn(1, 512, 2*2816) * Tensor([1.0, 3e5, 1e-3, 7e4] * 128).reshape(1, 512, 1)
    self._vs_one(x.shard(devs, 2).realize(), 512, 1, 16)
    self.assertEqual(amd_glm_dsa._half_rows_parts(512, 2816), 11)

  def test_row_split_unchanged(self):
    # (a row split over the devices (axis 1): the one-reduce path, the reshape into parts would move items between shards)
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    x = Tensor.randn(1, 512, 2048).shard(devs, 2).realize()
    h, s = amd_glm_dsa._half_rows(x, 512)
    np.testing.assert_array_equal(_bits(h.realize()), _bits((x.float() / s).cast(dtypes.half).realize()))

class TestIdxQBlk(unittest.TestCase):
  # (IDX_QBLK: every block of the indexer's queries roped from the projection's rows = the block of all of them roped: the same float ops;
  # on the CPU the compiler contracts the rope's mul-sub into fma differently in the two kernels (1 ulp), the device's bits: the CHECK hashes)
  def _blocks(self, qblk:int, Tq:int):
    Tensor.manual_seed(9)
    idx = Indexer.__new__(Indexer)
    idx.idx = SimpleNamespace(kpool=1, n_heads=4, head_dim=8, top_k=2, rope_dim=4, rope_interleave=True)
    idx.max_context, idx.shard, idx.cache_dk = 2048, 0, Tensor.zeros(2048, 8, dtype="half").contiguous().realize()
    idx.k_norm = lambda x: x
    idx.attn_k = lambda x: x[..., :8]
    idx.attn_q_b = lambda q: q
    idx.proj = lambda x: x[..., :4]
    x, q = Tensor.randn(1, Tq, 8).realize(), Tensor.randn(1, Tq, 32).realize()
    freqs = Tensor.randn(Tq, 4).realize()
    got:list[np.ndarray] = []
    def scores(qb, *a, **kw):
      got.append(qb.numpy())
      return Tensor.zeros(qb.shape[0], 4)
    real_getenv = glm_dsa.getenv
    with patch.object(glm_dsa, "indexer_scores", side_effect=scores), \
         patch.object(glm_dsa, "pool_index_rows", side_effect=lambda sc, *a, out=None, **kw: out), \
         patch.object(glm_dsa, "getenv", side_effect=lambda k, d=0: qblk if k == "IDX_QBLK" else real_getenv(k, d)):
      idx.dsa_picks(x, q, 0, freqs)
    return got
  def test_same_bits(self):
    for Tq in (1100, 2200):
      a, b = self._blocks(0, Tq), self._blocks(1, Tq)
      self.assertEqual(len(a), len(b))
      self.assertGreater(len(a), 1)
      for u, v in zip(a, b):
        np.testing.assert_array_equal(u[..., 4:].view(np.uint32), v[..., 4:].view(np.uint32))  # (the unroped dims: copies)
        np.testing.assert_allclose(u, v, rtol=1e-6, atol=1e-7)
      self.assertTrue((b[-1][Tq - 1056*(len(b)-1):] == 0).all() and len(b[-1]) > Tq - 1056*(len(b)-1))  # (the padding rows: 0)

@unittest.skipUnless(Device.DEFAULT == "AMD", "the pack / sum kernels are wave kernels (lane exchanges): AMD only")
class TestSpRsSkip(unittest.TestCase):
  # (SP_RS_SKIP: a device packs only the rows it sends, not its own (summed from its float partial): the same sums, bit for bit)
  def _check(self, fmt:str):
    Tensor.manual_seed(11)
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(3))
    p = (Tensor.randn(3, 1, 48, 256) * Tensor([1.0, 3e3, 1e-3] * 16).reshape(1, 1, 48, 1)).shard(devs, 0).realize()
    outs = []
    real_getenv = amd_glm_dsa.getenv
    for sk in (0, 1):
      with patch.object(amd_glm_dsa, "getenv", side_effect=lambda k, d=0, sk=sk: sk if k == "SP_RS_SKIP" else real_getenv(k, d)):
        outs.append(amd_glm_dsa.sp_reduce_scatter(p, fmt).realize())
    self.assertEqual(outs[1].shape, (1, 48, 256))
    np.testing.assert_array_equal(_bits(outs[1]), _bits(outs[0]))
    # (and near the float sum: the others' parts quantized, a value at most half a step of its group (q4: amax / 7, q8: amax / 127) off)
    pn = p.numpy()[:, 0]
    groups = np.abs(pn.reshape(3, 48, 8, 32)).max(-1, keepdims=True)
    bound = (groups / (14 if fmt == "q4" else 254)).sum(0).repeat(32, -1).reshape(48, 256) * 1.01 + 1e-5
    err = np.abs(outs[1].numpy()[0] - pn.sum(0))
    self.assertTrue((err <= bound).all(), f"max err / bound {(err / bound).max()}")
  def test_q4(self): self._check("q4")
  def test_q8(self): self._check("q8")

if __name__ == "__main__":
  unittest.main()
