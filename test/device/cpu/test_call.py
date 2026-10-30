import unittest, ctypes
from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.device import TinyELF

# an external call is a CALL on a CUSTOM_FUNCTION body naming the symbol, the program links against the loaded libraries
def call_out_kernel(C:UOp) -> UOp: # frexp writes the exponent through its pointer arg
  call = UOp.custom_function("frexp", dtype=dtypes.float64).call(UOp.const(8.0, dtypes.float64), C[0])
  return C.after(call)[1].store(C.after(call)[0].load() + 1).sink(arg=KernelInfo(name="call_out"))

def call_ret_kernel(C:UOp) -> UOp:
  val = UOp.custom_function("sqrt", dtype=dtypes.float64).call(UOp.const(16.0, dtypes.float64))
  return C[0].store(val.cast(dtypes.int) * 2).sink(arg=KernelInfo(name="call_ret"))

@unittest.skipUnless(isinstance(Device["CPU"].renderer, (CStyleLanguage, CPULLVMRenderer)), "requires C or LLVM calls")
class TestExternalCall(unittest.TestCase):
  def test_direct_args(self):
    # Spill multiple integers: Darwin ARM64 packs stack arguments, while generic AAPCS64 uses 8-byte slots.
    dev = Device["CPU"]
    src = """void test_args(double* out, signed char a, unsigned short b, int c, unsigned int d,
                            long e, unsigned long f, int g, int h, int k, int l, float i, double j) {
      out[0] = a + b + c + d + e + f + g + h + k + l + i + j;
    }"""
    types = (dtypes.int8, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64,
             dtypes.int32, dtypes.int32, dtypes.int32, dtypes.int32, dtypes.float32, dtypes.float64)
    if isinstance(dev.renderer, CPULLVMRenderer):
      params = [UOp.param(i+1, dt, addrspace=AddrSpace.ALU) for i,dt in enumerate(types)]
      value = params[0].cast(dtypes.float64)
      for p in params[1:]: value = value + p.cast(dtypes.float64)
      out_param = UOp.param(0, dtypes.float64, 1)
      sink = out_param.index(UOp.const(0, dtypes.int)).store(value).sink(arg=KernelInfo(name="test_args"))
      src = dev.renderer.render(list(sink.toposort()))
    sig = ((None, AddrSpace.GLOBAL, dtypes.float64, (1,)),) + tuple((None, AddrSpace.ALU, dt, ()) for dt in types)
    prg = dev.runtime(TinyELF(dev.compiler.compile(src), "test_args", dev.renderer.target, sig))
    out = ctypes.c_double()
    args = (-3, 65535, -100, 100, -1000, 1000, -7, 8, -17, 23, 0.5, 0.25)
    self.assertIsNotNone(prg(ctypes.addressof(out), *args, wait=True))
    self.assertEqual(out.value, sum(args))

  def test_call_out_param(self):
    c = Tensor.custom_kernel(Tensor.empty(2, dtype=dtypes.int, device="CPU"), fxn=call_out_kernel)[0]
    self.assertEqual(c.tolist(), [4, 5])

  def test_call_ret(self):
    c = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="CPU"), fxn=call_ret_kernel)[0]
    self.assertEqual(c.item(), 8)

if __name__ == "__main__": unittest.main()
