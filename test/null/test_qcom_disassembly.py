import struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.runtime.support.compiler_qcom import QCOMCompiler

class TestQCOMDisassembly(unittest.TestCase):
  def test_numeric_generation_and_shader_slice(self):
    shader = struct.pack("<2Q", 0, 6 << 55)
    library = bytearray(b"\xa5" * 0x160)
    struct.pack_into("<I", library, 0xc0, 0x140)
    struct.pack_into("<I", library, 0x100, len(shader))
    library[0x140:0x150] = shader

    for arch in ("a630", "a630,IMAGE_PITCH_ALIGNMENT=64"):
      with self.subTest(arch=arch), patch("tinygrad.runtime.support.compiler_qcom.disas_adreno") as decode:
        QCOMCompiler.disassemble(SimpleNamespace(arch=arch, chip_id=0x6030001), bytes(library))
        decode.assert_called_once_with(shader, 630)

if __name__ == "__main__":
  unittest.main()
