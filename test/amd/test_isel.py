import unittest
import numpy as np
from tinygrad import Tensor, Device, Context
from tinygrad.codegen import to_program
from tinygrad.viz.serve import amd_decode
from tinygrad.renderer.amd.dsl import Inst

def get_insts(t:Tensor) -> list[Inst]:
  prg_uop = to_program(t.schedule_linear().src[-1].src[0], Device[t.device].renderer)
  return list(amd_decode(prg_uop.src[4].arg, Device[t.device].renderer.target.arch).values())

class TestIsel(unittest.TestCase):
  @Context(NOOPT=1)
  def test_v_max(self):
    a = Tensor(np.arange(2, dtype=np.uint32)).realize()
    insts = get_insts(a.max())
    assert any((op:=getattr(i, "op_name", "")).startswith("V_MAX") or op.startswith("S_MAX") for i in insts)

if __name__ == "__main__":
  unittest.main()
