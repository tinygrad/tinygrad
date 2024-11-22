import tempfile
import unittest
from extra.utilities.blake3 import blake3

#@unittest.skip("slow")
class TestBLAKE3(unittest.TestCase):
  def setUp(self):
    self.vectors = [
      {
        "input_len": 0,
        "hash": "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
      },
      {
        "input_len": 1,
        "hash": "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"
      },
      {
        "input_len": 2,
        "hash": "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"
      },
      {
        "input_len": 3,
        "hash": "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"
      },
      {
        "input_len": 4,
        "hash": "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"
      },
      {
        "input_len": 5,
        "hash": "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"
      },
      {
        "input_len": 6,
        "hash": "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"
      },
      {
        "input_len": 7,
        "hash": "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"
      },
      {
        "input_len": 8,
        "hash": "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"
      },
      {
        "input_len": 63,
        "hash": "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"
      },
      {
        "input_len": 64,
        "hash": "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"
      },
      {
        "input_len": 65,
        "hash": "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"
      },
      {
        "input_len": 127,
        "hash": "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d"
      },
      {
        "input_len": 128,
        "hash": "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef"
      },
      {
        "input_len": 129,
        "hash": "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12"
      },
      {
        "input_len": 1023,
        "hash": "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"
      },
      {
        "input_len": 1024,
        "hash": "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"
      },
      {
        "input_len": 2048,
        "hash": "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"
      },
      {
        "input_len": 2049,
        "hash": "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"
      },
      {
        "input_len": 3072,
        "hash": "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"
      },
      {
        "input_len": 3073,
        "hash": "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"
      },
      {
        "input_len": 4096,
        "hash": "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"
      },
      {
        "input_len": 4097,
        "hash": "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995"
      },
      {
        "input_len": 5120,
        "hash": "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833"
      },
      {
        "input_len": 5121,
        "hash": "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff"
      },
      {
        "input_len": 6144,
        "hash": "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205"
      },
      {
        "input_len": 6145,
        "hash": "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"
      },
      {
        "input_len": 7168,
        "hash": "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a"
      },
      {
        "input_len": 7169,
        "hash": "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817"
      },
      {
        "input_len": 8192,
        "hash": "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"
      },
      {
        "input_len": 8193,
        "hash": "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b"
      },
      {
        "input_len": 16384,
        "hash": "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4"
      },
      {
        "input_len": 31744,
        "hash": "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47"
      },
      {
        "input_len": 102400,
        "hash": "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"
      }
    ]

  def generate_input(self, length: int) -> bytes:
    return bytes(i % 251 for i in range(length))

  def test_official_vectors(self):
    """Test against the official test vectors from: https://github.com/BLAKE3-team/BLAKE"""
    for vector in self.vectors:
        input_len = vector["input_len"]
        expected = vector["hash"]
        input = self.generate_input(input_len)
        actual = blake3(input)
        self.assertEqual(actual, expected)

  def test_file_input(self):
    with tempfile.NamedTemporaryFile(delete=True) as file:
      file.write(self.generate_input(102400))
      file.flush()
      actual = blake3(file=file.name, bufsize=1024 * 50) # expect 2 reads
      self.assertEqual(actual, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085")
    with tempfile.NamedTemporaryFile(delete=True) as file:
      file.write(self.generate_input(4097))
      file.flush()
      actual = blake3(file=file.name, bufsize=1024) # expect 5 reads
      self.assertEqual(actual, "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995")

if __name__ == "__main__":
  unittest.main()
