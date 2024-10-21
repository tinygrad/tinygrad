import hashlib
import random
import unittest
from test.helpers import is_dtype_supported
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

@unittest.skipUnless(is_dtype_supported(dtypes.uint8) and is_dtype_supported(dtypes.uint64), "Device must support uint8 and uint64")
class TestKeccak(unittest.TestCase):
  def setUp(self) -> None: random.seed(1337)

  def test_shape_keeping(self):
    s = (1, 2, 3, 4)
    for i in range(len(s)):
      out_shape = Tensor.randint(*s[i:], high=255, dtype=dtypes.uint8).keccak().shape
      self.assertTupleEqual(s[i:-1], out_shape[:-1])

  def test_sha3_224(self): self._test_preset("sha3_224", [143, 144])
  def test_sha3_256(self): self._test_preset("sha3_256", [135, 136])
  def test_sha3_384(self): self._test_preset("sha3_384", [103, 104])
  def test_sha3_512(self): self._test_preset("sha3_512", [71, 72])
  def _test_preset(self, name: str, special_sizes: list[int]):
    hasher: type[hashlib._Hash] = getattr(hashlib, name)

    for n in (special_sizes + [1, 128]):
      a, b = random.randbytes(n), random.randbytes(n)

      ha_ref, hb_ref = hasher(a).digest(), hasher(b).digest()
      tres = Tensor.stack(*(Tensor(d) for d in (a, b))).keccak(name)
      ha, hb = tres[0].data(), tres[1].data()

      self.assertEqual(ha_ref, ha)
      self.assertEqual(ha_ref, Tensor(a).keccak(name).data())
      self.assertEqual(hb_ref, hb)

if __name__ == "__main__":
  unittest.main()
