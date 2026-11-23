import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.codegen import create_linear
from tinygrad.device import Buffer
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
from tinygrad.runtime.ops_cl import CLDevice, CLAllocator, CLCompiler
from tinygrad.uop.uir import parse_uir

def compile_batch(fn:str) -> str: return compile_linear(create_linear(parse_uir(fn))).render_uir()

@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestOpenCL(unittest.TestCase):
  def test_link_compiled_batch(self):
    out = Tensor([1, 2, 3], device="CL").realize() + 7
    fn, externs = out.export()
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool: batch = pool.submit(compile_batch, fn.render_uir()).result()
    with patch.object(CLCompiler, "compile_cached", side_effect=AssertionError("link tried to compile OpenCL source")):
      run_linear(link_linear(parse_uir(batch), list(externs)), jit=True, wait=True)
    self.assertEqual(out.tolist(), [8, 9, 10])

  @unittest.skip("allocates tons of memory")
  def test_oom(self):
    with self.assertRaises(RuntimeError) as err:
      allocator = CLAllocator(CLDevice())
      for i in range(1_000_000):
        allocator.alloc(1_000_000_000)
    assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"

  def test_invalid_kernel_name(self):
    device = Device[Device.DEFAULT]
    lib = device.compile(b"__kernel void test(__global int* a) { a[0] = 1; }")
    with self.assertRaises(RuntimeError) as err: device.function(lib, "")
    assert str(err.exception) == "OpenCL Error -46: CL_INVALID_KERNEL_NAME"

  def test_unaligned_copy(self):
    data = list(range(65))
    unaligned = memoryview(bytearray(data))[1:]
    buffer = Buffer("CL", 64).allocate()
    buffer.copy_from(Buffer("PYTHON", 64, opaque=unaligned))
    result = memoryview(bytearray(len(data) - 1))
    result[:] = buffer.as_memoryview()
    assert unaligned == result, "Unaligned data copied in must be equal to data copied out."
