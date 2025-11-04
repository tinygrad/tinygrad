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
  # clear the old traces
  dev = Device["AMD"]
  dev.profile_events.clear()
  # setup custom_kernel
  name = sys._getframe(1).f_code.co_name
  def fxn(_):
    ops:list[str] = [UOp(Ops.CUSTOM, arg="asm volatile (")]
    for inst in asm: ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=f'  "{inst}\\n\\t"'))
    ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=");"))
    return UOp.sink(*ops, arg=KernelInfo(name=name))
  k = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
  # exec and decode sqtt
  k.realize()
  rctx = decode(dev.profile_events+[ProfileDeviceEvent("AMD", arch=dev.device_info())])
  assert len(rctx.inst_execs) == 1, f"expected one trace event, got {len(rctx.inst_execs)}"
  return list(rctx.inst_execs.values())[0][:-1]

class TestTiming(unittest.TestCase):
  def test_v_add(self):
    sqtt = get_sqtt([f"v_add_f32 v{10+i} v{10+i+1} {10+i}" for i in range(3)])
    for s in sqtt:
      print(s.time, s.inst)
    assert all(s.dur == 1 for s in sqtt)
    assert all(s.stall == 0 for s in sqtt)

  def test_chain_v_add(self):
    sqtt = get_sqtt([
      "v_add_f32_e32 v1 v0 v0",
      "v_add_f32_e32 v2 v1 v1",
    ])
    self.assertEqual([s.dur for s in sqtt], [1,1])
    for s in sqtt:
      print(s)

  def test_multi_cycle_inst(self):
    sqtt = get_sqtt([
      "v_mov_b32_e32 v4 0x3f800000",
      "v_rcp_f32_e32 v5 v4",
      "v_mul_f32_e32 v6 v5 v4",
    ])
    rcp, mul = sqtt[1], sqtt[2]
    self.assertGreater(rcp.dur, 1) # 4 cycles on gfx11
    self.assertEqual(mul.dur, 1)
    # why doesn't this work?
    print(mul.time, mul.stall, mul.dur, rcp.time, rcp.dur, rcp.stall)
    #self.assertGreaterEqual(mul.time + mul.stall, rcp.time + rcp.dur)

if __name__ == "__main__":
  unittest.main()
