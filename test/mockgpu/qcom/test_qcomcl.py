import platform, shutil, struct, unittest

from tinygrad.helpers import DEV


@unittest.skipUnless(DEV.interface.startswith("MOCK") and DEV.device == "QCOM", "QCOM mock device required")
@unittest.skipUnless(platform.machine() == "aarch64" or shutil.which("qemu-aarch64-static"), "QCOMCL compiler needs qemu-aarch64-static")
class TestQCOMCLMachineCode(unittest.TestCase):
  def test_compiled_add_is_decodable_ir3(self):
    from tinygrad.runtime.support.compiler_qcom import QCOMCompiler
    from test.mockgpu.qcom.decoder import decode_ir3

    compiler = QCOMCompiler("a630")
    binary = compiler.compile("""
      __kernel void add(__global int *out, __global const int *lhs, __global const int *rhs) {
        int index = get_global_id(0);
        out[index] = lhs[index] + rhs[index];
      }
    """)
    self.assertGreater(len(binary), 0x104)
    image_offset, image_size = struct.unpack_from("<I", binary, 0xc0)[0], struct.unpack_from("<I", binary, 0x100)[0]
    self.assertGreater(image_size, 0)
    self.assertEqual(image_size % 8, 0)
    self.assertLessEqual(image_offset + image_size, len(binary))
    instructions = decode_ir3(binary[image_offset:image_offset + image_size])
    self.assertTrue(any(instruction.name == "end" for instruction in instructions))
    self.assertTrue(any(instruction.name == "ldg.a" for instruction in instructions))
    self.assertTrue(any(instruction.name == "stg.a" for instruction in instructions))


if __name__ == "__main__": unittest.main()
