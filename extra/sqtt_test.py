import os, unittest
os.environ["SQTT"] = "1"
os.environ["VIZ"] = "1"
os.environ["PRINT_MATCH_STATS"] = "0"

os.environ["AMD_LLVM"] = "0"
os.environ["AM_RESET"] = "1"
os.environ["DEBUG"] = "2"

from tinygrad import Tensor
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.realize import CompiledRunner

class TestSQTT(unittest.TestCase):
  def test_asm(self):
    test = """
    typedef long unsigned int size_t;
    extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
    extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
    extern "C" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);
    extern "C" __attribute__((global)) void test(float* data0) {
      asm volatile("v_mov_b32 v0, 0");
      #pragma clang loop unroll(disable)
      for (unsigned i = 0; i < 100; i++) {
        asm volatile("v_add_u32 v0, v0, 1" ::: "v0");
      }
    }
    """
    global_size = [1, 1, 1]
    local_size  = [1, 1, 1]
    prg = ProgramSpec("test", test, "AMD", UOp(Ops.NOOP), None, global_size=global_size, local_size=local_size)
    cr = CompiledRunner(prg)
    cr([Tensor.empty(1).uop.buffer.ensure_allocated()], {}, wait=True)

  def test_wave_sched(self):
    test = """extern "C" __attribute__((global)) void test(float* data0) {}"""
    global_size = [32, 1, 1]
    local_size  = [32, 1, 1]
    prg = ProgramSpec("test", test, "AMD", UOp(Ops.NOOP), None, global_size=global_size, local_size=local_size)
    cr = CompiledRunner(prg)
    cr([Tensor.empty(1).uop.buffer.ensure_allocated()], {}, wait=True)

if __name__ == "__main__":
  unittest.main()
