import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
from tinygrad.runtime.ops_cl import CLDevice, CLAllocator, CLCompiler
from tinygrad.uop.uir import render_uir, parse_uir

def run_compiled(src:str, roles:list[str]) -> list[int]:
  # a fresh process, with its own CL context and queue, links and runs the batch another process compiled
  a, out = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize(), Tensor.empty(4, dtype=dtypes.int).realize()
  inputs = [{"a": a, "out": out}[r].uop.buf_uop for r in roles]
  run_linear(link_linear(parse_uir(src), inputs), input_uops=inputs, jit=True, wait=True)
  return out.tolist()

@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestCLError(unittest.TestCase):
  @unittest.skip("allocates tons of memory")
  def test_oom(self):
    with self.assertRaises(RuntimeError) as err:
      allocator = CLAllocator(CLDevice())
      for i in range(1_000_000):
        allocator.alloc(1_000_000_000)
    assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"

  def test_invalid_kernel_name(self):
    device = Device[Device.DEFAULT]
    lib = CLCompiler(device).compile_cached("__kernel void test(__global int* a) { a[0] = 1; }")
    with self.assertRaises(RuntimeError) as err: device.function(lib, "")
    assert str(err.exception) == "OpenCL Error -46: CL_INVALID_KERNEL_NAME"

  def test_link_in_other_process(self):
    a = Tensor([0, 0, 0, 0], dtype=dtypes.int).realize()
    out = (a + 1).contiguous()
    compiled = compile_linear(out.schedule_linear(), input_uops=(inputs:=[]), cache=True)
    roles = ["out" if u is out.uop.base.buf_uop else "a" for u in inputs]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as runner:
      self.assertEqual(runner.submit(run_compiled, render_uir(compiled), roles).result(), [2, 3, 4, 5])

  def test_unaligned_copy(self):
    data = list(range(65))
    unaligned = memoryview(bytearray(data))[1:]
    buffer = Buffer("CL", 64).allocate()
    buffer.copy_from(Buffer("PYTHON", 64, opaque=unaligned))
    result = memoryview(bytearray(len(data) - 1))
    result[:] = buffer.as_memoryview()
    assert unaligned == result, "Unaligned data copied in must be equal to data copied out."
