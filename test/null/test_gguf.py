import io, struct, unittest
from tinygrad import dtypes, Tensor, Device
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_shard, read_arr, read_str
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def test_read_arr_bulk_and_nested(self):
    # scalar int32 array
    buf = io.BufferedReader(io.BytesIO(struct.pack("<iQ3i", 5, 3, -1, 0, 42)))
    self.assertEqual(read_arr(buf), [-1, 0, 42])
    # string array
    sdata = b"".join(struct.pack("<Q", len(s)) + s for s in (b"hi", b"", b"gguf"))
    buf = io.BufferedReader(io.BytesIO(struct.pack("<iQ", 8, 3) + sdata))
    self.assertEqual(read_arr(buf), ["hi", "", "gguf"])
    # nested array of uint16 arrays
    inner = struct.pack("<iQ2H", 2, 2, 1, 2) + struct.pack("<iQ1H", 2, 1, 9)
    buf = io.BufferedReader(io.BytesIO(struct.pack("<iQ", 9, 2) + inner))
    self.assertEqual(read_arr(buf), [[1, 2], [9]])
    # truncated string header must raise struct.error
    with self.assertRaises(struct.error):
      read_str(io.BufferedReader(io.BytesIO(b"\x05\x00\x00")))

  def test_gguf_shard_contiguous_view_and_unaligned(self):
    vals_a, vals_b = [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
    payload = struct.pack("<4f", *vals_a) + struct.pack("<4f", *vals_b) + b"\x00" + struct.pack("<4f", *vals_b)
    raw = Tensor(payload, dtype=dtypes.uint8, device="CPU").realize()
    # aligned contiguous slices share one Buffer via Buffer.view (pin CPU so value asserts always run even under DEV=NULL)
    out = gguf_shard({"a": (raw[0:16], (4,), 0), "b": (raw[16:32], (4,), 0)}, ("CPU",))
    self.assertEqual(out["a"].tolist(), vals_a)
    self.assertEqual(out["b"].tolist(), vals_b)
    # unaligned slice falls back cleanly without creating misaligned Buffer.view
    out_unaligned = gguf_shard({"a": (raw[0:16], (4,), 0), "b": (raw[33:49], (4,), 0)}, ("CPU",))
    self.assertEqual(out_unaligned["a"].tolist(), vals_a)
    self.assertEqual(out_unaligned["b"].tolist(), vals_b)
    # trailing empty tensor falls back cleanly instead of calling Buffer.view with offset == nbytes
    out_empty = gguf_shard({"a": (raw[0:16], (4,), 0), "b": (raw[16:32], (4,), 0), "e": (raw[32:32], (0,), 0)}, ("CPU",))
    self.assertEqual(out_empty["a"].tolist(), vals_a)
    self.assertEqual(out_empty["b"].tolist(), vals_b)
    self.assertEqual(out_empty["e"].tolist(), [])
    if Device.DEFAULT != "CPU":
      out_def = gguf_shard({"a": (raw[0:16], (4,), 0), "b": (raw[16:32], (4,), 0)}, (Device.DEFAULT,))
      self.assertEqual((out_def["a"].shape, out_def["b"].shape), ((4,), (4,)))

if __name__ == '__main__':
  unittest.main()
