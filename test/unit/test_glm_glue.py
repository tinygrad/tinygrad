import unittest
import numpy as np
from tinygrad import Tensor, dtypes, UOp, Device
from tinygrad.llm.glm_dsa import apply_rope, mla_q_cat
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

  def test_row_split_unchanged(self):
    # (a row split over the devices (axis 1): the one-reduce path, the reshape into parts would move items between shards)
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    x = Tensor.randn(1, 512, 2048).shard(devs, 2).realize()
    h, s = amd_glm_dsa._half_rows(x, 512)
    np.testing.assert_array_equal(_bits(h.realize()), _bits((x.float() / s).cast(dtypes.half).realize()))

if __name__ == "__main__":
  unittest.main()
