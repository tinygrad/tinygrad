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
from tinygrad.device import Device, ProfileDeviceEvent

from extra.sqtt.roc import decode, InstExec

def get_sqtt(asm:list[str]) -> list[InstExec]:
  name = sys._getframe(1).f_code.co_name
  def fxn(_):
    ops:list[str] = [UOp(Ops.CUSTOM, arg="asm volatile (")]
    for inst in asm: ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=f'  "{inst}\\n\\t"'))
    ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=");"))
    return UOp.sink(*ops, arg=KernelInfo(name=name))
  Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0].realize()
  dev = Device["AMD"]
  rctx = decode(dev.profile_events+[ProfileDeviceEvent("AMD", arch=dev.device_info())])
  dev.profile_events.clear()
  assert len(rctx.inst_execs) == 1, f"expected one trace event, got {len(rctx.inst_execs)}"
  return list(rctx.inst_execs.values())[0]

class TestSQTT(unittest.TestCase):
  def test_v_add(self):
    sqtt = get_sqtt([
      "v_mov_b32_e32 v10 10",
      "v_mov_b32_e32 v11 20",
      "v_add_f32 v11 v10 v11",
    ])
    self.assertEqual(sqtt[0].dur, 1)
    self.assertEqual(sqtt[1].dur, 1)
    self.assertEqual(sqtt[2].dur, 1)
    assert all(s.stall == 0 for s in sqtt)

if __name__ == "__main__":
  unittest.main()
