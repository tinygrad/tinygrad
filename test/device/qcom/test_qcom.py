import ctypes, platform, unittest
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.uop.ops import Ops, UOp

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, IR3Renderer), "needs the ir3 compiler")
  def test_kernargs_fit_big_consts(self):
    from tinygrad.llm.gguf import _ggml_iq_grid
    from tinygrad.runtime.autogen import ggml_common as _ggml
    from tinygrad.runtime.ops_qcom import QCOMProgramData
    from tinygrad.runtime.support.hcq2 import pack_args
    # the kernel that builds the iq2_s grid has 2180 bytes of immediates, past the 2048 that used to sit before the descriptors
    for call in _ggml_iq_grid(Device.DEFAULT, _ggml.iq2s_grid, (1024, 8)).schedule_linear().src:
      if (ast:=call.src[0]).op is not Ops.SINK: continue
      data = QCOMProgramData(Device[Device.DEFAULT], to_program(ast, Device[Device.DEFAULT].renderer).to_elf())
      pack_args([(off, UOp.const(val, dtypes.uint32)) for val,off,_ in data.consts_info], data.kernargs_alloc_size)

if __name__ == '__main__':
  unittest.main()
