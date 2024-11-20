import unittest, contextlib, io
from tinygrad import Tensor, Context, Device
from tinygrad.engine.realize import method_cache
from tinygrad.codegen.kernel import Kernel

def clear_cache():
  method_cache.clear()
  Kernel.kernel_cnt.clear()

@unittest.skipIf(Device.DEFAULT != "CUDA", "test prints for cuda")
class TestCUDAPrints(unittest.TestCase):
  clear_cache()
  def test_cuda_print_order_debug_6(self):
    db5_out = io.StringIO()
    with contextlib.redirect_stdout(db5_out):
      with Context(DEBUG=5):
        Tensor.ones(3).realize()

    clear_cache()
    db6_out = io.StringIO()
    with contextlib.redirect_stdout(db6_out):
      with Context(DEBUG=6):
        Tensor.ones(3).realize()
    # get rif of trailing "\n" and final print with kernel time/flops/mem usage
    db5_str = db5_out.getvalue()[:-1].rpartition("\n")[0]
    assert db6_out.getvalue().startswith(db5_str), "DEBUG=6 output should start with DEBUG=5 output"

  def test_cuda_disassemble(self):
    clear_cache()
    db6_out = io.StringIO()
    with contextlib.redirect_stdout(db6_out):
      with Context(DEBUG=6):
        Tensor.ones(3).realize()
    assert "returned non-zero exit status 1" not in db6_out.getvalue(), "failed disassembly from ptx"

if __name__ == "__main__":
  unittest.main()