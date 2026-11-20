import pathlib, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.runtime.support.mt.compiler import MTCompiler, DEFAULT_MUSA_PATH
from tinygrad.runtime.ops_mt import MUSACompiler

@unittest.skipUnless(pathlib.Path(DEFAULT_MUSA_PATH, 'bin/mcc').exists(), 'MUSA toolchain is not installed')
class TestMTCompiler(unittest.TestCase):
  def test_compile_and_metadata(self):
    image = MTCompiler().compile('#include <musa.h>\nextern "C" __global__ void mt_compute(int *out) { out[0] = 43; }\n')
    self.assertEqual(image.entry, 0x1700)
    self.assertEqual(image.text, bytes.fromhex('4e2ac341400040c01c070080'))
    self.assertEqual(image.metadata.immediates[-1].value, 43)
    self.assertEqual(image.metadata.kernel['.name'], 'mt_compute')

class TestMUSACompilerFallback(unittest.TestCase):
  def test_high_spill_local_arrays_are_volatile(self):
    compiler = MUSACompiler()
    compiler.mt.compile_raw = Mock(side_effect=[b'normal',b'volatile'])
    metadata = SimpleNamespace(kernel={'.private_memory_size':104,'.temp_reg_count':254})
    with patch('tinygrad.runtime.ops_mt.MTKernelImage.from_elf',return_value=SimpleNamespace(metadata=metadata)):
      self.assertEqual(compiler.compile('  float buf0[1];\n'),b'volatile')
    self.assertEqual(compiler.mt.compile_raw.call_args_list[1].args[0],'  volatile float buf0[1];\n')

if __name__ == '__main__': unittest.main()
