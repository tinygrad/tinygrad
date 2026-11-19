# the vector lowering of kernels on the DSP (Renderer.vector_alu): values stay vectors, memory is accessed in windows of vector registers
import re, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context

@unittest.skipUnless(Device.DEFAULT == "DSP", "DSP only")
class TestVectorize(unittest.TestCase):
  def setUp(self):
    import tinygrad.runtime.ops_dsp as dsp
    self.srcs: list[str] = []
    self.orig, srcs = dsp.DSPRenderer.render, self.srcs
    def render(ren, uops):
      srcs.append(s:=self.orig(ren, uops))
      return s
    dsp.DSPRenderer.render = render
    # (rendered here, not in workers)
    self.ctx = Context(PARALLEL=0)
    self.ctx.__enter__()
  def tearDown(self):
    import tinygrad.runtime.ops_dsp as dsp
    dsp.DSPRenderer.render = self.orig
    self.ctx.__exit__(None, None, None)

  def test_padded_store(self):
    # an output axis padded to the vector width stores its lanes with a mask
    for shape in [(3, 200), (5, 77), (130,)]:
      for dt in [dtypes.uint8, dtypes.int32]:
        a, b = [np.random.randint(0, 100, shape).astype(dt.fmt) for _ in range(2)]
        np.testing.assert_equal(((Tensor(a)+Tensor(b))*3).numpy(), ((a+b)*3).astype(dt.fmt))
    self.assertTrue(any("store_m(" in s for s in self.srcs))

  def test_conv(self):
    for cin, cout, h, w, k, pad in [(1, 1, 4, 128, 1, 0), (16, 32, 16, 128, 1, 0), (1, 1, 4, 128, 3, 1), (16, 32, 16, 128, 3, 1)]:
      x, wt = np.random.randint(0, 255, (1, cin, h, w)).astype(np.uint8), np.random.randint(-128, 127, (cout, cin, k, k)).astype(np.int8)
      xp = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
      ref = np.zeros((1, cout, h+2*pad-k+1, w+2*pad-k+1), np.int64)
      for i in range(k):
        for j in range(k): ref += np.einsum("nchw,oc->nohw", xp[:, :, i:i+ref.shape[2], j:j+ref.shape[3]], wt[:, :, i, j].astype(np.int64))
      np.testing.assert_equal(Tensor(x).int().conv2d(Tensor(wt).int(), padding=pad).numpy(), ref)

  def test_no_wide_values(self):
    # a sum of strided windows (an unrolled reduce) is the lanes of each, not a gather of all of them: no value is wider than a few registers
    x = np.random.randint(0, 255, (1, 3, 226, 226)).astype(np.uint8)
    out = Tensor(x).int().conv2d(Tensor.ones(1, 3, 3, 3, dtype=dtypes.int), stride=2).numpy()
    ref = sum(x[:, :, i:i+223:2, j:j+223:2].astype(np.int64).sum(1, keepdims=True) for i in range(3) for j in range(3))
    np.testing.assert_equal(out, ref)
    widths = [int(n) for s in self.srcs for n in re.findall(r"^\s+[a-z_]+?(\d+) \w+ =", s, re.M)]
    self.assertLessEqual(max(widths, default=0), 512)

if __name__ == "__main__":
  unittest.main()
