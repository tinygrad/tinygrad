import unittest
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.llm.kernels import amd_qwen as aq

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestQSAPrefill(unittest.TestCase):
  def test_qsa_prefill_vs_exact(self):
    # QSA_PREFILL's kernel (_qsa_prefill_kernel) on compact pool lists of a 4-bit cache: against the float64 attention of the cache's values,
    # and close to qsa_decode's (whose queries are int8)
    if Device[Device.DEFAULT].target[0] not in (11, 12): self.skipTest("requires RDNA3/4")
    B, H, HD, KP, TOP, CTX = 288, 12, 256, 4, 40, 4096
    rng = np.random.default_rng(7)
    N = CTX + B + 64
    cache = aq.kv_q4_pack(Tensor(rng.normal(size=(N, 2*HD)).astype(np.float16))).realize()
    q = Tensor(rng.normal(size=(B, H, HD)).astype(np.float16)).realize()
    idx = np.zeros((B, TOP + 2), dtype=np.int32)
    for r in range(B):
      npk = TOP - r % 3  # (a few lists shorter: their pools past the count unused)
      idx[r, :npk], idx[r, TOP], idx[r, TOP + 1] = np.sort(rng.choice(CTX // KP, size=npk, replace=False)), npk, CTX + r
    index, bases = Tensor(idx).realize(), Tensor(np.zeros(B, dtype=np.int32)).realize()
    outs = {}
    for flag in (0, 1):
      aq.QSA_PREFILL = flag
      try: outs[flag] = aq.qsa_decode_attention(q, cache.reshape(1, N, -1), index, bases, HD**-0.5, kpool=KP, bits=4).numpy().reshape(B, H, HD)
      finally: aq.QSA_PREFILL = 0
    cw = cache.numpy().reshape(N, -1)
    vals = np.zeros((N, 2*HD))
    for j in range(8): vals[:, j::8] = ((cw[:, :2*HD//8] >> ((j % 4) * 8 + (j // 4) * 4)) & 15).astype(np.float64) - 8
    vals *= np.repeat(cw[:, 2*HD//8:].view(np.float16).astype(np.float64).reshape(N, -1), 32, axis=1)
    qn = q.numpy().astype(np.float64)
    for r in range(0, B, 7):
      npk, qp = idx[r, TOP], idx[r, TOP + 1]
      keys = np.concatenate([(idx[r, :npk, None] * KP + np.arange(KP)).ravel(), np.arange((qp + 1) // KP * KP, qp + 1)])
      s = qn[r] @ vals[keys, :HD].T * HD**-0.5
      p = np.exp(s - s.max(-1, keepdims=True))
      ref = (p / p.sum(-1, keepdims=True)) @ vals[keys, HD:]
      np.testing.assert_allclose(outs[1][r], ref, atol=2e-3 * np.abs(ref).max(), rtol=0)
      np.testing.assert_allclose(outs[0][r], outs[1][r], atol=3e-2 * np.abs(ref).max(), rtol=0)

if __name__ == "__main__":
  unittest.main()
