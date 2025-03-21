import unittest
from tinygrad.runtime.ops_gpu import CLDevice
from tinygrad.device import BufferSpec, Device

class TestIntelOcloc(unittest.TestCase):
  @unittest.expectedFailure # we can only test on Intel GPUs
  @unittest.skipIf(Device.DEFAULT != "GPU", f"not supported on {Device.DEFAULT}")
  def test_simple_compilation(self):
    cl_kernel = """__kernel void test(__global int* data0) {
                      int gidx0 = get_group_id(0);
                      *(data0+gidx0) =  get_group_id(0);
                    }"""

    from tinygrad import intel_offline_compiler as ioc

    def cl_compile(src, ioc_compile_func):
      device = CLDevice()
      array_len = 100
      # allocate buffer and compile
      buf = device.allocator.alloc(array_len*4, BufferSpec(host=True, cpu_access=True))
      binary = device.compiler.compile(cl_kernel) if ioc_compile_func is False else ioc.IntelOfflineCompiler().compile(cl_kernel, "0x"+format(device.device_ip, "x"))
      prog = device.runtime("test", binary)
      res = prog(buf, global_size = (array_len,1,1), local_size=(4,1,1), wait=True)
      # read out results and free cl buffer
      array = bytearray(array_len*4)
      mv_array = memoryview(array)
      device.allocator._copyout(mv_array, buf)
      device.allocator._free(buf, BufferSpec(host=True, cpu_access=True))
      return res, binary, list(bytearray(mv_array))
    cl_time, cl_binary, cl_result = cl_compile(cl_kernel, False)
    ioc_time, ioc_binary, ioc_result = cl_compile(cl_kernel, True)
    assert cl_result == ioc_result

if __name__ == '__main__':
  unittest.main()