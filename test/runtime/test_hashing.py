import hashlib, unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp

supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.uint64 in supported_dtypes, "Device must support uint8 and uint64")
@unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestHashing(unittest.TestCase):
  def _python_hash_1mb(self, data:bytes):
    chunks = [data[i:i+4096] for i in range(0, len(data), 4096)]
    chunk_hashes = [hashlib.shake_128(chunk).digest(16) for chunk in chunks]
    return hashlib.shake_128(b''.join(chunk_hashes)).digest(16)

  @unittest.skip("very slow")
  def test_abc(self):
    expected = self._python_hash_1mb(b"abc" + b"\x00" * (2**20 - 3))
    out = Tensor(b"abc").hash()
    self.assertEqual(bytes(out.data()), expected)

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.uint64 in supported_dtypes, "Device must support uint8 and uint64")
@unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestKeccak(unittest.TestCase):
  def test_referenced(self):
    # https://www.di-mgt.com.au/sha_testvectors.html
    self.assertEqual(bytes(Tensor(b"abc").keccak().tolist()),
                     bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))

  def test_variable_bs(self):
    data = Tensor([b"abc", b"abc", b"def"], dtype=dtypes.uint8).repeat(2048, 1)
    bs = UOp.variable("bs", 1, 4096).bind(3)
    out = data.shrink_to(bs, data.shape[-1]).keccak().shrink_to(3, 32).realize()
    self.assertEqual(bytes(out[0].tolist()), bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))
    self.assertEqual(bytes(out[1].tolist()), bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))
    self.assertEqual(bytes(out[2].tolist()), bytearray.fromhex("8e0d8f672252acb0 ffc5093db8653b18 1513bf9a2097e737 b4f73533dcaf46df"))

if __name__ == "__main__":
  unittest.main()
