import unittest, contextlib, io
from tinygrad import Tensor, Context, Device
from tinygrad.engine.realize import method_cache
from tinygrad.codegen.kernel import Kernel

def clear_cache():
  method_cache.clear()
  Kernel.kernel_cnt.clear()

def print_debug(level:int) -> str:
  clear_cache()
  db_out = io.StringIO()
  with contextlib.redirect_stdout(db_out):
    with Context(DEBUG=level):
      Tensor.ones(3).realize()
  return db_out.getvalue()

@unittest.skipIf(Device.DEFAULT != "CUDA", "test prints for cuda")
class TestCUDAPrints(unittest.TestCase):
  def test_cuda_print_order_debug_6(self):
    db5_str = print_debug(5)
    db6_str = print_debug(6)
    # get rif of trailing "\n" and final print with kernel time/flops/mem usage
    db5_str = db5_str[:-1].rpartition("\n")[0]
    assert db6_str.startswith(db5_str), "DEBUG=6 output should start with DEBUG=5 output"

  def test_cuda_disassemble(self):
    db6_str = print_debug(6)
    assert "returned non-zero exit status 1" not in db6_str, "failed disassembly from ptx"

if __name__ == "__main__":
  unittest.main()