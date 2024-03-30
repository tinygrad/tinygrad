import unittest
from tinygrad.device import Device
from tinygrad.buffer import Buffer
from tinygrad.dtype import dtypes

DEVICE = Device.DEFAULT

def mvs(mv: memoryview) -> str:
  return str(list(mv))

def check(buf: Buffer, exp):
  assert buf.as_buffer() == memoryview(bytearray(exp)), f"expected {mvs(memoryview(bytearray(exp)))} but got {mvs(buf.as_buffer())}"

@unittest.skipUnless(hasattr(Device[DEVICE].allocator, 'offset'), f"{DEVICE} doesn't support offset buffers")
class TestOffsetBuffer(unittest.TestCase):
  def test_simple(self):
    base = Buffer(DEVICE, 16, dtype=dtypes.uint8, initial_value=bytearray(range(16)))

    v = base.view(4,4)
    check(v, [4,5,6,7]) # should offset
    v.copyin(memoryview(bytearray([125,126,127,128]))) # copy new stuff to view
    check(v, [125,126,127,128]) # view should change
    check(base, [0,1,2,3,125,126,127,128,8,9,10,11,12,13,14,15]) # part of base should change to view
    v2 = base.view(8,4) # create another view
    check(v2, [8,9,10,11])
    v3 = v2.view(1,2) # create view of that view
    check(v3, [9,10])
    v3.copyin(memoryview(bytearray([13,37]))) # change v3
    check(v3, [13,37])
    check(v2, [8,13,37,11])
    check(base, [0,1,2,3,125,126,127,128,8,13,37,11,12,13,14,15]) # base should change too

  def test_base_assign(self):
    base = Buffer(DEVICE, 16, dtype=dtypes.uint8, initial_value=bytearray(range(16)))

    v = base.view(4,4)
    check(v, [4,5,6,7]) # should offset
    base.copyin(memoryview(bytearray([255-x for x in range(16)]))) # copy stuff to base
    check(v, [251,250,249,248]) # d2 should change

  def test_changing_dtype(self):
    base = Buffer(DEVICE, 16, dtype=dtypes.uint8, initial_value=bytearray(range(16)))
    i32 = base.view(4,4, dtype=dtypes.int32)
    check(i32, [4,5,6,7])
    assert i32.size == 1 and i32.nbytes == 4
    u16 = i32.view(0,4,dtype=dtypes.uint16)
    check(u16, [4,5,6,7])
    assert u16.size == 2 and i32.nbytes == 4
    with self.assertRaises(AssertionError):
      base.view(0,6).view(0,6,dtype=dtypes.int32)

if __name__ == "__main__":
  unittest.main()