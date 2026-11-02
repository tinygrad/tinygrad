import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import all_gather

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestLargeAllgather(unittest.TestCase):
  def test_chunked_gather(self):
    if Device[Device.DEFAULT].target[0] != 11: self.skipTest("requires RDNA3")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(92)
    for dtype in (np.float16, np.float32):
      @TinyJit
      def run(x): return all_gather(x, 1).realize()
      for iteration in range(3):
        data = rng.normal(size=(1, len(devices)*352, 10240)).astype(dtype)
        data.reshape(-1)[:4] = [-0., np.inf, -np.inf, np.nan]
        y = run(Tensor(data).shard(devices, axis=1).realize())
        for d, buf in zip(devices, y.uop.buffer.bufs):
          actual = Tensor(UOp.from_buffer(buf, y.dtype, d)).reshape(y.shape).numpy()
          with self.subTest(dtype=dtype, iteration=iteration, device=d):
            np.testing.assert_array_equal(actual.view(np.uint16 if dtype == np.float16 else np.uint32),
                                          data.view(np.uint16 if dtype == np.float16 else np.uint32))

if __name__ == "__main__": unittest.main()
