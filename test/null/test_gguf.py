import struct, unittest
from tinygrad import dtypes, Tensor, Device
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_shard
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def test_gguf_shard_contiguous_view_fallbacks(self):
    vals_a, vals_b = [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
    raw = Tensor(struct.pack("<4f", *vals_a) + struct.pack("<4f", *vals_b) + b"\x00" + struct.pack("<4f", *vals_b),
                 dtype=dtypes.uint8, device="CPU").realize()
    for entries in ({"a": (raw[0:16], (4,), 0), "b": (raw[16:32], (4,), 0)},
                    {"a": (raw[0:16], (4,), 0), "b": (raw[33:49], (4,), 0)},
                    {"a": (raw[0:16], (4,), 0), "b": (raw[16:32], (4,), 0), "e": (raw[32:32], (0,), 0)}):
      out = gguf_shard(entries, ("CPU",))
      self.assertEqual((out["a"].tolist(), out["b"].tolist()), (vals_a, vals_b))
      if "e" in out: self.assertEqual(out["e"].tolist(), [])

if __name__ == '__main__':
  unittest.main()
