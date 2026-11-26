import struct, unittest
from tinygrad import dtypes, Tensor, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_shard, _bulk_load
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def test_gguf_shard_bulk_load(self):
    vals_a, vals_b, vals_c = [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]
    bc = struct.pack("<4f", *vals_c)
    part0 = Tensor(b"\x00" * 16 + struct.pack("<4f4f", *vals_a, *vals_b) + b"\x00" + bc + b"\x00" * 15 + bc, device="CPU").realize()
    part1 = Tensor(b"\x00" * 16 + bc, device="CPU").realize()
    a, b, unaligned = (part0[16:32], (4,), 0), (part0[32:48], (4,), 0), (part0[49:65], (4,), 0)
    gap, empty, d = (part0[80:96], (4,), 0), (part0[48:48], (0,), 0), (part1[16:32], (4,), 0)
    for entries, exp, bulk in (({"a": a, "b": b}, {"a": vals_a, "b": vals_b}, True),
                               ({"a": a, "b": b, "d": d}, {"a": vals_a, "b": vals_b, "d": vals_c}, True),
                               ({"a": a, "b": b, "u": unaligned, "d": d}, {"a": vals_a, "b": vals_b, "u": vals_c, "d": vals_c}, False),
                               ({"a": a, "gap": gap}, {"a": vals_a, "gap": vals_c}, False),
                               ({"a": a, "b": b, "e": empty}, {"a": vals_a, "b": vals_b, "e": []}, False)):
      self.assertEqual(bool(res:=_bulk_load(entries, "CPU")), bulk)
      if bulk: self.assertEqual(len({t.uop.buffer.base for t in res.values()}), len({e[0].uop.base for e in entries.values()}))
      self.assertEqual({k: v.tolist() for k, v in gguf_shard(entries, ("CPU",)).items()}, exp)
    with Context(MAX_BUFFER_SIZE=16): self.assertEqual(_bulk_load({"a": a, "b": b}, "CPU"), {})
    self.assertEqual(gguf_shard({}, ("CPU",)), {})

if __name__ == '__main__':
  unittest.main()
