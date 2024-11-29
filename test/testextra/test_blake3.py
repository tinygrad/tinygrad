import unittest
from extra.utilities.blake3 import BLAKE3
from tinygrad.tensor import Tensor

class TestBLAKE3(unittest.TestCase):
  """Test against official tests from: https://github.com/BLAKE3-team/BLAKE3"""

  def _test(self, input_len: int, expected_hash: str):
    input_data = bytes(i % 251 for i in range(input_len))
    actual = BLAKE3().hash(Tensor(input_data))
    self.assertEqual(actual, expected_hash)

  def test_empty_input(self):
    """Test empty input (0 bytes)"""
    self._test(0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")

  def test_single_byte(self):
    """Test single byte input"""
    self._test(1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213")

  def test_two_bytes(self):
    """Test two bytes input"""
    self._test(2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63")

  def test_three_bytes(self):
    """Test three bytes input"""
    self._test(3, "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f")

  def test_four_bytes(self):
    """Test four bytes input"""
    self._test(4, "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32")

  def test_five_bytes(self):
    """Test five bytes input"""
    self._test(5, "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2")

  def test_six_bytes(self):
    """Test six bytes input"""
    self._test(6, "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844")

  def test_seven_bytes(self):
    """Test seven bytes input"""
    self._test(7, "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe")

  def test_eight_bytes(self):
    """Test eight bytes input"""
    self._test(8, "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb")

  def test_63_bytes(self):
    """Test 63 bytes (just under block size)"""
    self._test(63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b")

  def test_64_bytes(self):
    """Test 64 bytes (exactly one block)"""
    self._test(64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98")

  def test_65_bytes(self):
    """Test 65 bytes (just over block size)"""
    self._test(65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee")

  def test_127_bytes(self):
    """Test 127 bytes"""
    self._test(127, "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d")

  def test_128_bytes(self):
    """Test 128 bytes"""
    self._test(128, "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef")

  def test_129_bytes(self):
    """Test 129 bytes"""
    self._test(129, "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12")

  def test_1023_bytes(self):
    """Test 1023 bytes (just under chunk size)"""
    self._test(1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11")

  def test_1024_bytes(self):
    """Test 1024 bytes (exactly one chunk)"""
    self._test(1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7")

  def test_2048_bytes(self):
    """Test 2048 bytes (two chunks)"""
    self._test(2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a")

  def test_2049_bytes(self):
    """Test 2049 bytes"""
    self._test(2049, "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030")

  def test_3072_bytes(self):
    """Test 3072 bytes"""
    self._test(3072, "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2")

  def test_3073_bytes(self):
    """Test 3073 bytes"""
    self._test(3073, "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3")

  def test_4096_bytes(self):
    """Test 4096 bytes"""
    self._test(4096, "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969")

  def test_4097_bytes(self):
    """Test 4097 bytes"""
    self._test(4097, "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995")

  def test_5120_bytes(self):
    """Test 5120 bytes"""
    self._test(5120, "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833")

  def test_5121_bytes(self):
    """Test 5121 bytes"""
    self._test(5121, "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff")

  def test_6144_bytes(self):
    """Test 6144 bytes"""
    self._test(6144, "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205")

  def test_6145_bytes(self):
    """Test 6145 bytes"""
    self._test(6145, "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f")

  def test_7168_bytes(self):
    """Test 7168 bytes"""
    self._test(7168, "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a")

  def test_7169_bytes(self):
    """Test 7169 bytes"""
    self._test(7169, "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817")

  def test_8192_bytes(self):
    """Test 8192 bytes"""
    self._test(8192, "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63")

  def test_8193_bytes(self):
    """Test 8193 bytes"""
    self._test(8193, "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b")

  def test_16384_bytes(self):
    """Test 16384 bytes"""
    self._test(16384, "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4")

  def test_31744_bytes(self):
    """Test 31744 bytes"""
    self._test(31744, "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47")

  def test_102400_bytes(self):
    """Test 102400 bytes"""
    self._test(102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085")

if __name__ == "__main__":
  unittest.main()
