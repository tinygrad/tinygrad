import ctypes, platform, unittest
from types import SimpleNamespace
from tinygrad import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, MMIOInterface

class TestQCOM(unittest.TestCase):
  def test_args_use_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState

    gpu_memory, cpu_memory = bytearray([0xaa] * 64), bytearray([0xaa] * 64)
    args = HCQBuffer(mv_address(memoryview(gpu_memory)), len(gpu_memory),
                     view=MMIOInterface(mv_address(memoryview(cpu_memory)), len(cpu_memory)))
    data = HCQBuffer(0x123456789abcdef0, 16)
    prg = SimpleNamespace(kernargs_alloc_size=64, signature=((None, 0, dtypes.float32, (1,)),),
                          ibo_cnt=0, tex_cnt=0, samp_cnt=0, NIR=True, tex_to_image=[], consts_info=[(0x12345678, 24, 4)],
                          buf_off=8, tex_off=64, ibo_off=64, samplers=[])

    state = QCOMArgsState(args, prg, (data,), vals=(0x87654321,))
    HWQueue().bind_args_state(state)

    self.assertEqual(cpu_memory[:8], bytes(8))
    self.assertEqual(int.from_bytes(cpu_memory[8:16], "little"), data.va_addr)
    self.assertEqual(int.from_bytes(cpu_memory[16:20], "little"), 0x87654321)
    self.assertEqual(int.from_bytes(cpu_memory[24:28], "little"), 0x12345678)
    self.assertEqual(cpu_memory[28:], bytes(36))
    self.assertEqual(gpu_memory, bytes([0xaa] * 64))

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
