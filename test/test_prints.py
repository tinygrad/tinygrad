import unittest, contextlib, io
from tinygrad import Tensor, Context, Device
from tinygrad.engine.realize import method_cache
from tinygrad.codegen.kernel import Kernel

@unittest.skipIf(Device.DEFAULT != "CUDA", "test prints for cuda")
class TestPrints(unittest.TestCase):
  def test_cuda_print_debug_6(self):
    db5_out = io.StringIO()
    with contextlib.redirect_stdout(db5_out):
      with Context(DEBUG=5):
        Tensor.arange(0,6).realize()
    method_cache.clear()
    Kernel.kernel_cnt.clear()

    db6_out = io.StringIO()
    with contextlib.redirect_stdout(db6_out):
      with Context(DEBUG=6):
        Tensor.arange(0,6).realize()
    
    # get rif of trailing "\n" and final print with kernel time/flops/mem usage
    str = db5_out.getvalue()[:-1].rpartition("\n")[0]
    assert db6_out.getvalue().startswith(str), "DEBUG=6 output should start with DEBUG=5 output"

if __name__ == "__main__":
  unittest.main()