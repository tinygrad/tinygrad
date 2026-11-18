import pathlib, unittest
from tinygrad.runtime.support.mt.compiler import MTCompiler, DEFAULT_MUSA_PATH

@unittest.skipUnless(pathlib.Path(DEFAULT_MUSA_PATH, 'bin/mcc').exists(), 'MUSA toolchain is not installed')
class TestMTCompiler(unittest.TestCase):
  def test_compile_and_metadata(self):
    image = MTCompiler().compile('#include <musa.h>\nextern "C" __global__ void mt_compute(int *out) { out[0] = 43; }\n')
    self.assertEqual(image.entry, 0x1700)
    self.assertEqual(image.text, bytes.fromhex('4e2ac341400040c01c070080'))
    self.assertEqual(image.metadata.immediates[-1].value, 43)
    self.assertEqual(image.metadata.kernel['.name'], 'mt_compute')

if __name__ == '__main__': unittest.main()
