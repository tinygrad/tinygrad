import unittest, itertools
from tinygrad.renderer.amd.dsl import v, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, v_add_f32_e32, v_fma_f32, v_rcp_f32_e32, v_wmma_f32_16x16x16_f16, ds_load_b32, \
  ds_store_b32, s_waitcnt_lgkmcnt, s_barrier, s_nop
from test.mockgpu.amd.sqtt_model import Op, CU

def run(*waves, simd=None, addrs=tuple(range(0, 128, 4))):
  ops = []
  for insts in waves:
    pcs = itertools.accumulate([0] + [inst.size() for inst in insts])
    ops.append([Op(pc, inst, 0xffffffff, addrs if isinstance(inst, DS) else ()) for pc, inst in zip(pcs, insts)])
  cu = CU(ops, simd or [0] * len(ops))
  cu.run()
  return [([x.t for x in w.ops], [x.stamp for x in w.ops]) for w in cu.waves]
def deltas(xs): return [b - a for a, b in zip(xs, xs[1:])]
def add(k): return v_add_f32_e32(v[k], v[20], v[21])
# a wave's first vector op is slower
warm = [add(30), s_nop(simm16=15)]

class TestSQTTModel(unittest.TestCase):
  def test_valu(self):
    (issue, stamp), = run(warm + [add(k) for k in range(8)])
    self.assertEqual(deltas(issue[2:]), [1] * 7)
    self.assertEqual([x - t for x, t in zip(stamp[2:], issue[2:])], [7] * 8)

  def test_vector_queue(self):
    (issue, stamp), = run(warm + [v_fma_f32(v[0], v[0], v[1], v[2])] * 20)
    self.assertEqual(set(deltas(stamp[2:])), {5})
    self.assertEqual(deltas(issue[2:16]), [1] * 13)
    self.assertEqual(set(deltas(issue[16:])), {5})

  def test_matrix_queue(self):
    (issue, _), = run(warm + [v_wmma_f32_16x16x16_f16(v[0:7], v[8:15], v[16:23], v[0:7])] + [add(30 + k) for k in range(10)])
    self.assertEqual(deltas(issue[2:10]), [1] * 7)
    self.assertGreater(issue[10] - issue[9], 30)

  def test_trans_port(self):
    (issue, _), = run(warm + [v_rcp_f32_e32(v[0], v[1]), v_rcp_f32_e32(v[2], v[3])])
    self.assertEqual(issue[3] - issue[2], 4)

  def test_lds_port(self):
    # one port for both SIMDs, SIMD 1 goes first
    (a, a_stamp), (b, b_stamp) = run(*[[s_barrier(), ds_store_b32(addr=v[1], data0=v[2])]] * 2, simd=[0, 1])
    self.assertEqual(a[1], b[1])
    self.assertEqual(a_stamp[1] - b_stamp[1], 2)

  def test_lds_request_queue(self):
    (issue, _), = run([ds_store_b32(addr=v[1], data0=v[2])] * 30)
    self.assertEqual(deltas(issue), [1] * 15 + [2] * 14)

  def test_lds_banks(self):
    # all 32 lanes in bank 0
    (issue, stamp), = run([ds_load_b32(vdst=v[0], addr=v[1]), s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)], addrs=tuple(range(0, 4096, 128)))
    self.assertEqual(issue[1] - stamp[0], 59)

  def test_icache_fill(self):
    (issue, _), = run([add(0)] * 128)
    self.assertEqual(issue[64] - issue[15], 625)

  def test_barrier(self):
    (early, _), (late, _) = run([s_barrier(), add(0)], [s_nop(simm16=15), s_nop(simm16=15), s_barrier(), add(0)], simd=[0, 1])
    self.assertEqual(early[1], late[3])
    self.assertEqual(early[1] - late[2], 12)

if __name__ == "__main__":
  unittest.main()
