import unittest
from tinygrad.renderer.amd.dsl import v, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import v_rcp_f32_e32, v_add_f32_e32, ds_load_b32, s_waitcnt_lgkmcnt
from test.mockgpu.amd.sqtt_model import Step, emulate, TRANS_II, FETCH_START, LDS_LAT, WAITCNT_SEE

def stalls(*insts) -> list[tuple[str, int]]:
  steps, pc = [], 0
  for inst in insts:
    steps.append(Step(pc, inst))
    pc += inst.size()
  return emulate([steps])[0].stalls

class TestStalls(unittest.TestCase):
  def test_fetch(self): self.assertEqual(stalls(v_add_f32_e32(v[0], v[1], v[2]))[0], ("fetch: wave start", FETCH_START))
  def test_back_to_back(self): self.assertEqual(stalls(v_add_f32_e32(v[0], v[1], v[2]), v_add_f32_e32(v[3], v[1], v[2]))[1], ("", 0))
  def test_trans_port(self):
    self.assertEqual(stalls(v_rcp_f32_e32(v[0], v[1]), v_rcp_f32_e32(v[2], v[3]))[1], ("TRANS: port", TRANS_II - 1))
  def test_lgkmcnt(self):
    self.assertEqual(stalls(ds_load_b32(vdst=v[0], addr=v[1]), s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))[1],
                     ("s_waitcnt: lgkmcnt", LDS_LAT + WAITCNT_SEE))

if __name__ == "__main__":
  unittest.main()
