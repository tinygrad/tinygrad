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
    insts = get_insts(Tensor(np.arange(2, dtype=np.uint32)).max())
    assert any("MAX" in getattr(i, "op_name", "") for i in insts)

  # there's no native u64 max
  @Context(NOOPT=1)
  def test_v_max_long(self):
    insts = get_insts(Tensor(np.arange(2, dtype=np.uint64)).max())
    assert not any("MAX" in getattr(i, "op_name", "") for i in insts)

if __name__ == "__main__":
  unittest.main()
