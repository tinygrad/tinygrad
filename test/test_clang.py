import unittest
from tinygrad import Device
from tinygrad.runtime.ops_clang import ISOLATE_DLOPEN, ClangCompiler, ClangProgram

@unittest.skipUnless(Device.DEFAULT == "CLANG", "Clang is being tested")
class TestClang(unittest.TestCase):
  @unittest.skipUnless(ISOLATE_DLOPEN, "Isolated dlopen is being tested")
  def test_isolated_dlopen(self):
    src = '''
    extern float sqrtf(float x);
    void kernel(float* restrict dst, float* restrict src) {
      *dst = sqrtf(*src);
    }
    '''
    lib = ClangCompiler().compile_cached(src)
    with self.assertRaisesRegex(RuntimeError, "failed to import dynamic library"): ClangProgram("kernel", lib)

if __name__ == "__main__":
  unittest.main()