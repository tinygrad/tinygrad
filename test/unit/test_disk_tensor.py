import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor

class TestDiskTensor(unittest.TestCase):
  def test_empty(self):
    pathlib.Path("/tmp/dt1").unlink(missing_ok=True)

    Tensor.empty(100, 100, device="disk:/tmp/dt1")

  def test_write_ones(self):
    pathlib.Path("/tmp/dt2").unlink(missing_ok=True)

    out = Tensor.ones(10, 10, device="CPU")
    outdisk = out.to("disk:/tmp/dt2")
    print(outdisk)
    outdisk.realize()
    del out, outdisk

    # test file
    with open("/tmp/dt2", "rb") as f:
      assert f.read() == b"\x00\x00\x80\x3F" * 100

    # test load alt
    reloaded = Tensor.empty(10, 10, device="disk:/tmp/dt2")
    out = reloaded.numpy()
    assert np.all(out == 1.)

if __name__ == "__main__":
  unittest.main()

