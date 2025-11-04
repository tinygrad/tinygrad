import os
os.environ["PYTHONPATH"] = "."
os.environ["SQTT"] = "1"
os.environ["AMD"] = "1"
os.environ["VIZ"] = "1"
os.environ["AMD_LLVM"] = "0"

import unittest
import sys
from tinygrad import Tensor
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.realize import CompiledRunner

def run_asm(asm:list[str]):
  name = sys._getframe(1).f_code.co_name
  def fxn(A:UOp, B:UOp):
    ops:list[str] = [UOp(Ops.CUSTOM, arg="asm volatile (")]
    for inst in asm: ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=f'  "{inst}\\n\\t"'))
    ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=");"))
    i = UOp.range(A.size, 0)
    return A[i].store(B[i]).end(i).sink(*ops, arg=KernelInfo(name=name))
  Tensor.custom_kernel(Tensor.empty(1), Tensor.empty(1), fxn=fxn)[0].realize()

class TestSQTT(unittest.TestCase):
  def test_v_add(self):
    run_asm([
      "v_mov_b32_e32 v10 10",
      "v_mov_b32_e32 v11 20",
      "v_add_f32 v11 v10 v11",
    ])

if __name__ == "__main__":
  unittest.main()
