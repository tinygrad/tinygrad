import os
os.environ["PYTHONPATH"] = "."
os.environ["SQTT"] = "1"
os.environ["AMD"] = "1"
os.environ["VIZ"] = "1"
os.environ["AMD_LLVM"] = "0"

import unittest
import sys
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.realize import CompiledRunner
from tinygrad.device import Device, ProfileDeviceEvent

from extra.sqtt.roc import decode, InstExec

dev = Device["AMD"]
def get_sqtt(asm:list[str], l:int=1, g:int=1) -> list[InstExec]:
  # clear the old traces
  dev.profile_events.clear()
  # setup custom_kernel
  name = sys._getframe(1).f_code.co_name
  def fxn(_):
    L = UOp.special(l, "lidx0")
    G = UOp.special(g, "gidx0")
    ops:list[str] = [UOp(Ops.CUSTOM, arg="asm volatile (")]
    for inst in asm: ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=f'  "{inst}\\n\\t"'))
    ops.append(UOp(Ops.CUSTOM, src=(ops[-1],), arg=");"))
    return UOp.sink(*ops, L, G, arg=KernelInfo(name=name))
  k = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
  # exec and decode sqtt
  k.realize()
  rctx = decode(dev.profile_events+[ProfileDeviceEvent("AMD", props=dev.device_props())])
  assert len(rctx.inst_execs) > 0, "empty sqtt output"
  return list(rctx.inst_execs.values())[0][:-1]

class TestTiming(unittest.TestCase):
  def test_v_add(self):
    sqtt = get_sqtt([f"v_add_f32 v{10+i} v{10+i+1} {10+i}" for i in range(3)])
    assert all(s.dur == 1 for s in sqtt)
    assert all(s.stall == 0 for s in sqtt)

  def test_chain_v_add_1l(self):
    sqtt = get_sqtt([
      "v_add_f32_e32 v1 v0 v0",
      "v_add_f32_e32 v2 v1 v1",
    ])
    assert all(s.dur == 1 for s in sqtt)
    assert all(s.stall == 0 for s in sqtt)

  def test_multi_cycle_inst(self):
    sqtt = get_sqtt([
      "v_mov_b32_e32 v4 0x3f800000",
      "v_rcp_f32_e32 v5 v4",
      "v_mul_f32_e32 v6 v5 v4",
    ])
    rcp, mul = sqtt[1], sqtt[2]
    self.assertGreater(rcp.dur, 1) # 4 cycles on gfx11
    self.assertEqual(mul.dur, 1)
    # mul depends on v5, how can it run before rcp is done?
    self.assertGreaterEqual(mul.time, rcp.time+rcp.dur)

  def test_wmma(self):
    sqtt = get_sqtt([
      "v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]",
      "v_add_f32_e32 v0 v16 v0",
    ], 32*4)
    wmma = sqtt[0]
    self.assertGreater(wmma.dur, 1) # rgp says 32 clocks

  def test_sleep(self):
    n = 1
    def sleep_kernel(data0):
      assert data0.dtype.base == dtypes.ulong
      ops:list[UOp] = []
      ops.append(UOp(Ops.CUSTOM, arg="unsigned long long t0 = __builtin_readcyclecounter();"))
      ops.append(UOp(Ops.CUSTOM, arg=f"__builtin_amdgcn_s_sleep({n});", src=(ops[-1],)))
      ops.append(UOp(Ops.CUSTOM, arg="unsigned long long t1 = __builtin_readcyclecounter();", src=(ops[-1],)))
      ops.append(UOp(Ops.CUSTOM, arg=f"data0_{data0.size}[0] = t1 - t0;", src=(ops[-1],)))
      return UOp.sink(data0, *ops, arg=KernelInfo(name=f"sleep_{n}"))
    diff_hw_reg = Tensor.empty(1, dtype=dtypes.ulong)
    diff_hw_reg = Tensor.custom_kernel(diff_hw_reg, fxn=sleep_kernel)[0]
    diff_hw_reg.realize()
    rctx = decode(dev.profile_events+[ProfileDeviceEvent("AMD", props=dev.device_props())])
    diff_sqtt = list(rctx.inst_execs.values())[0][2]
    self.assertEqual(diff_sqtt.dur, diff_hw_reg.item()-1) # 1 cycle for reading the counter register

if __name__ == "__main__":
  unittest.main()
