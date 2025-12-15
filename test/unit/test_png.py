#!/usr/bin/env python
import io, unittest
import numpy as np
from tinygrad import Tensor, fetch
from tinygrad.nn.state import png_load
try:
  from PIL import Image
except ImportError:
  raise unittest.SkipTest("PIL not installed")

class TestPNGLoad(unittest.TestCase):
  def test_real_png(self):
    # test against a real PNG file (uses only filters 0, 1)
    fp = fetch('https://upload.wikimedia.org/wikipedia/en/d/d4/Norwegian_Forest_Cat_in_Norway.png')
    with open(fp, 'rb') as f: png_bytes = f.read()
    expected = np.array(Image.open(io.BytesIO(png_bytes)))[:, :, :3]
    result = png_load(Tensor(np.frombuffer(png_bytes, dtype=np.uint8))).numpy()
    np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
  unittest.main()
