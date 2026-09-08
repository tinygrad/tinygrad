import struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.runtime.support.compiler_qcom import QCOMCompiler

class TestQCOMCompiler(unittest.TestCase):
  def test_disassemble_gpu_id(self):
    shader = b"shader payload"
    offset = 0x120
    lib = bytearray(offset + len(shader) + 8)
    struct.pack_into("I", lib, 0xc0, offset)
    struct.pack_into("I", lib, 0x100, len(shader))
    lib[offset:offset+len(shader)] = shader
    lib[-8:] = b"trailer!"
    for arch in ("a630", "a630,IMAGE_PITCH_ALIGNMENT=64"):
      with self.subTest(arch=arch), patch("tinygrad.runtime.support.compiler_qcom.disas_adreno") as disassemble:
        QCOMCompiler.disassemble(SimpleNamespace(arch=arch, chip_id=0x6030001), bytes(lib))
        disassemble.assert_called_once_with(shader, 630)

if __name__ == '__main__':
  unittest.main()
