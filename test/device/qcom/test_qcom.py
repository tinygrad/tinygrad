import ctypes, platform, struct, unittest
from unittest.mock import patch
from tinygrad import Device
from tinygrad.renderer.cstyle import ClangRenderer

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

class TestQCOMCompilerDisassemble(unittest.TestCase):
  def test_disassemble_passes_gpu_generation(self):
    from tinygrad.runtime.support import compiler_qcom
    for arch in ["a630", "a630,IMAGE_PITCH_ALIGNMENT=64"]:
      with self.subTest(arch=arch):
        with patch("platform.machine", return_value="aarch64"), \
             patch.object(compiler_qcom.llvm_qcom, "cl_compiler_create_llvm_instance", return_value=ctypes.c_void_p(1)), \
             patch.object(compiler_qcom.llvm_qcom, "cl_compiler_destroy_llvm_instance"), \
             patch.object(compiler_qcom, "disas_adreno") as disas:
          c = compiler_qcom.QCOMCompiler(arch)
          self.assertEqual(c.gpu_id, 630)
          lib = bytearray(0x210)
          struct.pack_into("I", lib, 0xc0, 0x200)
          struct.pack_into("I", lib, 0x100, 16)
          lib[0x200:0x210] = bytes(range(16))
          c.disassemble(bytes(lib))
          disas.assert_called_once_with(bytes(range(16)), 630)
          del c

if __name__ == '__main__':
  unittest.main()
