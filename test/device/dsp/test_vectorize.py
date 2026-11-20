# shaped values are vectors on the DSP (Renderer.vector_alu)
import re, unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import Context

@unittest.skipUnless(Device.DEFAULT == "DSP", "DSP only")
class TestVectorize(unittest.TestCase):
  def test_elementwise_padded(self):
    for shape in [(3, 200), (5, 77), (130,)]:
      a, b = [np.random.randint(0, 100, shape).astype(np.uint8) for _ in range(2)]
      np.testing.assert_equal(((Tensor(a)+Tensor(b))*3).numpy(), ((a+b)*3).astype(np.uint8))

  def test_conv_no_wide_values(self):
    # each tap of a conv is a vector of its lanes, no value is wider than a few registers
    import tinygrad.runtime.ops_dsp as dsp
    srcs, render = [], dsp.DSPRenderer.render
    dsp.DSPRenderer.render = lambda ren, uops: srcs.append(s:=render(ren, uops)) or s
    try:
      x, w = np.random.randint(0, 255, (1, 16, 16, 128)).astype(np.uint8), np.random.randint(-128, 127, (32, 16, 3, 3)).astype(np.int8)
      with Context(PARALLEL=0): out = Tensor(x).int().conv2d(Tensor(w).int(), padding=1).numpy()
    finally: dsp.DSPRenderer.render = render
    xp = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    np.testing.assert_equal(out, sum(np.einsum("nchw,oc->nohw", xp[:, :, i:i+16, j:j+128], w[:, :, i, j]) for i in range(3) for j in range(3)))
    self.assertLessEqual(max(int(n) for s in srcs for n in re.findall(r"^\s+[a-z_]+?(\d+) \w+ =", s, re.M)), 512)

if __name__ == "__main__":
  unittest.main()
