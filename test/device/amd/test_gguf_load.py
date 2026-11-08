import os, pathlib, tempfile, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
from tinygrad.llm.gguf_load import _parallel_load, _preadv_exact

class TestFileReads(unittest.TestCase):
  def test_repeated_short_reads(self):
    source = b"abcdefghijklmnop"
    out = bytearray(20)
    offsets = []
    def short_read(fd, views, offset):
      offsets.append(offset)
      data = source[offset:offset+3]
      copied = 0
      for view in views:
        n = min(len(view), len(data)-copied)
        view[:n] = data[copied:copied+n]
        copied += n
      return copied
    with patch("os.preadv", short_read): _preadv_exact(0, [memoryview(out)[1:6], memoryview(out)[9:20]], 0)
    self.assertEqual(out[1:6]+out[9:20], source)
    self.assertEqual(offsets, [0, 3, 6, 9, 12, 15])

  def test_eof(self):
    with patch("os.preadv", return_value=0), self.assertRaises(EOFError): _preadv_exact(0, [memoryview(bytearray(4))], 0)

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD required")
class TestParallelLoad(unittest.TestCase):
  def test_shard_layouts_and_stage_reuse(self):
    # One test owns all devices: pytest workers must not run concurrent userspace drivers on the same GPUs.
    n = getenv("TEST_GPUS", 2)
    devs = tuple(Device.canonicalize(f"AMD:{i}") for i in range(n))
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, LOAD_STAGE="4096", LOAD_STAGES="2"):
      getenv.cache_clear()
      entries, expected, units, shards, pads, fuse, rows = {}, {}, {}, {}, {}, {}, {}
      data = bytearray()
      def add(name, shape):
        arr = rng.standard_normal(shape).astype(np.float32)
        off = len(data)
        data.extend(arr.tobytes())
        data.extend(bytes((-len(data)) % 256))
        entries[name] = (off, arr)
        return arr
      # A gather between two copies must not be overwritten by a copy spanning its destination.
      before = add("before", (17,))
      expected["before"] = [before.tobytes()]*n
      # More gathers than the three shared buffers also exercise reuse of those buffers.
      for name in ("col", "col_reuse1", "col_reuse2", "col_reuse3"):
        col = add(name, (7, n*5-1))
        shards[name], pads[name] = 1, n*5
        expected[name] = [x.tobytes() for x in np.split(np.pad(col, ((0, 0), (0, 1))), n, axis=1)]
      after = add("after", (13,))
      expected["after"] = [after.tobytes()]*n
      row = add("row", (n*3-1, 4))
      shards["row"], pads["row"] = 0, n*3
      expected["row"] = [x.tobytes() for x in np.split(np.pad(row, ((0, 1), (0, 0))), n)]
      selected = add("selected", (n*4, 3))
      rows["selected"] = (n*4, [(4*(n-i-1), 4) for i in range(n)])
      expected["selected"] = [selected[lo:lo+cnt].tobytes() for lo, cnt in rows["selected"][1]]
      # Nonconsecutive experts, repeated units (padding), and fused gate/up rows.
      selections = [[i, 2*n+i, i] for i in range(n)]
      gate = add("ffn_gate_exps.weight", (4, n*32, 8))
      up = add("ffn_up_exps.weight", gate.shape)
      down = add("ffn_down_exps.weight", (4, 7, n*8))
      for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"): units[name] = (n, selections)
      fuse["ffn_gate_exps.weight"] = "ffn_up_exps.weight"
      expected["ffn_gate_exps.weight"] = []
      expected["ffn_down_exps.weight"] = []
      for us in selections:
        chunks = []
        for u in us:
          e, s = divmod(u, n)
          for k in range(2): chunks += [gate[e, s*32+k*16:s*32+(k+1)*16], up[e, s*32+k*16:s*32+(k+1)*16]]
        expected["ffn_gate_exps.weight"].append(b"".join(x.tobytes() for x in chunks))
        expected["ffn_down_exps.weight"].append(b"".join(down[u//n, :, (u%n)*8:(u%n+1)*8].tobytes() for u in us))
      path = pathlib.Path(tmp)/"weights.bin"
      path.write_bytes(data)
      disk = Tensor(path)
      source = {name: (disk[off:off+arr.nbytes], arr.shape, 0) for name, (off, arr) in entries.items()}
      for deferred in (False, True):
        pending = [] if deferred else None
        packed = _parallel_load(source, devs, shards, units, pads, fuse, pending, row_sel=rows)
        if pending is not None:
          self.assertEqual(len(pending), 1)
          pending[0]()
        self.assertEqual(set(packed), set(expected))
        for name, refs in expected.items():
          for i, (buf, ref) in enumerate(zip(packed[name].uop.base.buffer.bufs, refs)):
            with self.subTest(deferred=deferred, name=name, device=i): self.assertEqual(bytes(buf.as_memoryview()), ref)
      # A short source file must fail promptly, after joining readers and draining outstanding GPU work.
      path.write_bytes(bytes(64))
      with self.assertRaises(EOFError): _parallel_load(source, devs, shards, units, pads, fuse, row_sel=rows)
      getenv.cache_clear()

if __name__ == "__main__": unittest.main()
