import io, unittest
from array import array
from tinygrad import Tensor
from tinygrad.nn.state import TensorIO

class TestTensorIO(unittest.TestCase):
  def test_read(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    self.assertEqual(fobj.read(1), data[:1])
    self.assertEqual(fobj.read(5), data[1:6])
    self.assertEqual(fobj.read(100), data[6:])
    self.assertEqual(fobj.read(100), b"")

  def test_read_nolen(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    fobj.seek(2)
    self.assertEqual(fobj.read(), data[2:])

  def test_readinto_buffers(self):
    data = b"Hello World!"
    for buffer in (bytearray(8), array('I', [0, 0]), memoryview(bytearray(8)).cast('B', shape=(2, 4))):
      with self.subTest(buffer_type=type(buffer).__name__):
        fobj, reference = TensorIO(Tensor(data)), io.BytesIO(data)
        expected = bytearray(memoryview(buffer).nbytes)
        for _ in range(3):
          self.assertEqual(fobj.readinto(buffer), reference.readinto(expected))
          self.assertEqual(memoryview(buffer).cast('B').tobytes(), bytes(expected))
          self.assertEqual(fobj.tell(), reference.tell())

if __name__ == '__main__':
  unittest.main()
