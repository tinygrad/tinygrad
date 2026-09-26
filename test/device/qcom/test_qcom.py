import ctypes, platform, unittest
from tinygrad import Device, Tensor, Variable
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
  def test_exec_symbolic_global_size(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    from tinygrad.runtime.support.hcq2 import EncodeCtx
    n = Variable("n", 1, 100000).bind(10)
    linear, _ = Tensor.linear_with_vars((Tensor.empty(100000)[:n] + 1).contiguous())
    call = next(c for c in linear.src if c.src[0].op is Ops.SINK)
    prg = to_program(call.src[0], Device[Device.DEFAULT].renderer)
    submit = UOp(Ops.CUSTOM_FUNCTION, arg="submit_qcom_compute", src=(UOp(Ops.LINEAR, arg=((Device.DEFAULT,), "compute")),))
    QCOMComputeQueue(EncodeCtx((Device.DEFAULT,)), submit).exec(call.replace(src=(prg,)+call.src[1:]), prg)

if __name__ == '__main__':
  unittest.main()
