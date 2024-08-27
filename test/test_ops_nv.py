import unittest

from tinygrad import Device, dtypes
from tinygrad.device import BufferOptions, Buffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, to_mv
from tinygrad.renderer import Program
from tinygrad.runtime.ops_nv import NVDevice

MOCKGPU = getenv("MOCKGPU")

@unittest.skipUnless(isinstance(Device[Device.DEFAULT], NVDevice), "NV device required to run")
class TestOpsNV(unittest.TestCase):
  @unittest.skipIf(MOCKGPU, "Can't run on MOCKGPU for now due to constbuf[0] hack")
  def test_blockdim_non_zero(self):
    program = Program("test_blockdim_nonzero",
                      "extern \"C\" __global__ void test_blockdim_nonzero(float *dst) {"
                      "int idx = blockDim.x * blockIdx.x + threadIdx.x;dst[idx] = 1.0f;}",
                      "nv")
    runner = CompiledRunner(program)
    N = 1024
    block_size = 256
    grid_size = N // block_size
    dst_buf = Buffer(Device.DEFAULT, N, dtypes.float32, options=BufferOptions(cpu_access=True)).ensure_allocated()
    runner.clprg(
      dst_buf._buf,
      global_size=(grid_size, 1, 1),
      local_size=(block_size, 1, 1),
      wait=True
    )
    dst_mv = to_mv(dst_buf._buf.va_addr, 4 * N).cast('f')
    assert dst_mv.tolist() == ([1.0] * N)

if __name__ == "__main__":
  unittest.main()
