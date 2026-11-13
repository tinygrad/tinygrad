# QSA_KV_BITS4_LAYERS: the attention blocks named by index get 4-bit cache rows, the others QSA_KV_BITS (a cache's bits are read from its shape)
import os, unittest
from types import SimpleNamespace
from unittest import mock
from tinygrad import Tensor
import tinygrad.llm.qwen4exp as q

def make(idx:int):
  b = object.__new__(q.QSABlock)
  b.layer_index, b.indexer = idx, None
  b.config = SimpleNamespace(n_kv_heads=2, head_dim=256, max_context=4096, cache_rows=0, kv_split=1, rope_dim=64, rope_theta=1e7, yarn=None)
  return b

class TestKVMix(unittest.TestCase):
  def widths(self, env:dict[str, str], idxs:list[int]) -> dict[int, tuple[int, int]]:
    out = {}
    with mock.patch.dict(os.environ, env), mock.patch.object(q, "amd_custom_kernels_supported", lambda d: True), \
         mock.patch.object(q, "_freqs", lambda *a, **k: Tensor.zeros(1)), \
         mock.patch.object(q, "getenv", lambda k, d=0: type(d)(os.environ.get(k, d))):
      for i in idxs:
        b = make(i)
        b._init_state(Tensor.zeros(1, 1, 8))
        out[i] = (int(b.cache_k.shape[-1]), b._kv_bits)
    return out
  def test_default_all_8(self):
    self.assertEqual(self.widths({"QSA_KV_BITS": "8"}, [3, 7, 48]), {3: (144, 8), 7: (144, 8), 48: (144, 8)})
  def test_mix(self):
    w = self.widths({"QSA_KV_BITS": "8", "QSA_KV_BITS4_LAYERS": "3,48"}, [3, 7, 48])
    self.assertEqual(w, {3: (72, 4), 7: (144, 8), 48: (72, 4)})  # (4-bit: half the bytes of a row)
  def test_16_unchanged(self):
    self.assertEqual(self.widths({"QSA_KV_BITS": "16", "QSA_KV_BITS4_LAYERS": "3"}, [3]), {3: (512, 16)})

if __name__ == "__main__": unittest.main()
