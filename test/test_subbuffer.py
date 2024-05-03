import unittest
from tinygrad import Device, dtypes
from tinygrad.buffer import Buffer

class TestSubBuffer(unittest.TestCase):
  def test_subbuffer(self):
    buf = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()
    buf.copyin(memoryview(bytearray(range(10))))
    vbuf = buf.view(2, dtypes.uint8, 3).ensure_allocated()
    tst = bytearray(2)
    vbuf.copyout(memoryview(tst))
    assert tst[0] == 3
    assert tst[1] == 4

if __name__ == '__main__':
  unittest.main()
