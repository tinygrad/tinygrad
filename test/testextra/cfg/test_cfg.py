import unittest
from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, track_rewrites
from tinygrad.renderer import ProgramSpec
from tinygrad.helpers import TracingKey
from tinygrad.engine.realize import ExecItem, CompiledRunner
from extra.sqtt.active_sqtt_parse import template

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, ret=ret))
def run_asm(asm:list[str]):
  a = Tensor([1]).realize()
  prg = ProgramSpec("test_cfg", template.replace("INSTRUCTION", '\n'.join(asm)), Device.DEFAULT, UOp(Ops.SINK))
  ei = ExecItem(UOp(Ops.SINK), [a.uop.buffer], prg=CompiledRunner(prg))
  ei.run()
  return prg

@unittest.skipUnless(Device.DEFAULT == "AMD", "only on AMD")
class TestCfg(unittest.TestCase):
  def setUp(self):
    arch = Device["AMD"].arch
    if not arch.startswith("gfx11"):
      self.skipTest(f"tests written for RDNA3, got arch {arch}")

  def test_simple(self):
    run_asm(["s_add_u32 s0 s0 s1", "s_endpgm"])

if __name__ == "__main__":
  unittest.main()
