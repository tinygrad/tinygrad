import unittest
import numpy as np
from tinygrad import Tensor, Variable, dtypes
from tinygrad.helpers import DEV

@unittest.skipUnless(DEV.device == "QCOM" and DEV.interface.startswith("MOCK") and DEV.renderer == "IR3", "MOCK+QCOM:IR3")
class TestIR3Emu(unittest.TestCase):
  def test_int_add(self):
    np.testing.assert_equal((Tensor([1,2,3])+1).numpy(), [2,3,4])
  def test_int_sub(self):
    np.testing.assert_equal((Tensor([3,2,1])-1).numpy(), [2,1,0])
  def test_int_sub_two(self):
    np.testing.assert_equal((Tensor([5,6,7])-Tensor([1,2,3])).numpy(), [4,4,4])
  def test_int_mul(self):
    np.testing.assert_equal((Tensor([1,2,3])*2).numpy(), [2,4,6])
    np.testing.assert_equal((Tensor([0x10000], dtype=dtypes.int32)*3).numpy(), [0x30000])
    np.testing.assert_equal((Tensor([100, -50], dtype=dtypes.int32)*Tensor([200, 3])).numpy(), [20000, -150])
  def test_uint_shr(self):
    np.testing.assert_equal((Tensor([16,32,64], dtype=dtypes.uint32)>>2).numpy(), [4,8,16])
    np.testing.assert_equal((Tensor([0x80000000], dtype=dtypes.uint32)>>1).numpy(), [0x40000000])
  def test_float_add(self):
    np.testing.assert_allclose((Tensor([1.0,2.0,3.0])+1).numpy(), [2,3,4])
  def test_float_mul(self):
    np.testing.assert_allclose((Tensor([1.0,2.0,3.0])*2).numpy(), [2,4,6])
  def test_float_add_two(self):
    np.testing.assert_allclose((Tensor([1.0,2.0,3.0])+Tensor([10.0,20.0,30.0])).numpy(), [11,22,33])
  def test_float_neg(self):
    np.testing.assert_allclose((-Tensor([1.0,2.0,3.0])).numpy(), [-1,-2,-3])
  def test_float_max(self):
    np.testing.assert_allclose(Tensor([1.0,3.0,2.0]).maximum(2.0).numpy(), [2,3,2])
  def test_float_min(self):
    np.testing.assert_allclose(Tensor([1.0,3.0,2.0]).minimum(2.0).numpy(), [1,2,2])
  def test_relu(self):
    np.testing.assert_allclose(Tensor([-1.0,2.0,-3.0]).relu().numpy(), [0,2,0])
  def test_int_max(self):
    np.testing.assert_equal(Tensor([1,3,2]).maximum(2).numpy(), [2,3,2])
    np.testing.assert_equal(Tensor([0], dtype=dtypes.int16).maximum(Tensor([-1], dtype=dtypes.int16)).numpy(), [0])
    np.testing.assert_equal(Tensor([0], dtype=dtypes.int8).maximum(Tensor([-1], dtype=dtypes.int8)).numpy(), [0])
  def test_int_min(self):
    np.testing.assert_equal(Tensor([1,3,2]).minimum(2).numpy(), [1,2,2])
    np.testing.assert_equal(Tensor([-5,3,-1]).minimum(0).numpy(), [-5,0,-1])
    np.testing.assert_equal(Tensor([0], dtype=dtypes.int16).minimum(Tensor([-1], dtype=dtypes.int16)).numpy(), [-1])
    np.testing.assert_equal(Tensor([0], dtype=dtypes.int8).minimum(Tensor([-1], dtype=dtypes.int8)).numpy(), [-1])
  def test_int_and(self):
    np.testing.assert_equal((Tensor([1,3,7])&Tensor([3,3,3])).numpy(), [1,3,3])
  def test_int_or(self):
    np.testing.assert_equal((Tensor([1,2,4])|Tensor([1,1,1])).numpy(), [1,3,5])
  def test_int_xor(self):
    np.testing.assert_equal((Tensor([1,3,7])^Tensor([1,1,1])).numpy(), [0,2,6])
  def test_andg(self):
    x, y, z = Tensor([0xF0, 0x0F, 0xAA], dtype=dtypes.int32), Tensor([0x3C, 0x3C, 0x0F], dtype=dtypes.int32), Tensor([1, 2, 4], dtype=dtypes.int32)
    np.testing.assert_equal(((x & y) | z).numpy(), [0x31, 0x0E, 0x0E])
  def test_sign(self):
    np.testing.assert_allclose(Tensor([-2.0, 0.0, 3.0]).sign().numpy(), [-1, 0, 1])
  def test_half_sign(self):
    np.testing.assert_allclose(Tensor([-2.0, 0.0, 3.0], dtype=dtypes.half).sign().numpy(), [-1, 0, 1])
  def test_bitcast_i16_i32(self):
    np.testing.assert_equal(Tensor([1, 2], dtype=dtypes.int16).bitcast(dtypes.int32).numpy(),
                            np.array([1, 2], dtype=np.int16).view(np.int32))
  def test_cast_i8_i16(self):
    np.testing.assert_equal(Tensor([-6, 0, 7], dtype=dtypes.int8).cast(dtypes.int16).numpy(), [-6, 0, 7])
  def test_cast_i8_u16(self):
    np.testing.assert_equal(Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.ushort).numpy(),
                            [65535, 65534, 65533, 65532])
  def test_cast_i8_float(self):
    np.testing.assert_allclose(Tensor([-6, 0, 7], dtype=dtypes.int8).cast(dtypes.float).numpy(), [-6, 0, 7])
  def test_cast_i64_float(self):
    np.testing.assert_allclose(Tensor([7, 74, 23, -78], dtype=dtypes.long).cast(dtypes.float).numpy(), [7, 74, 23, -78])
    np.testing.assert_allclose(Tensor([0, 1 << 32], dtype=dtypes.long).cast(dtypes.float).numpy(), [0.0, float(1 << 32)])
  def test_half_add_ulong(self):
    np.testing.assert_allclose((Tensor([1,2,3,4], dtype=dtypes.half)+Tensor([1,2,3,4], dtype=dtypes.ulong)).numpy(), [2,4,6,8])
  def test_cast_i8_emulated_i64(self):
    from tinygrad import Context
    with Context(EMULATED_DTYPES="long"):
      np.testing.assert_equal(Tensor([-99, 12], dtype=dtypes.char).cast(dtypes.long).numpy(), [-99, 12])
  def test_half_cmp_mask(self):
    x = Tensor([0.0, 2.0, -3.0])
    np.testing.assert_allclose((x * (x != 0)).numpy(), [0, 2, -3])
  def test_int_neg(self):
    np.testing.assert_equal((-Tensor([1,2,3])).numpy(), [-1,-2,-3])
  def test_int_abs(self):
    np.testing.assert_equal(Tensor([-1,2,-3]).abs().numpy(), [1,2,3])
  def test_expand(self):
    np.testing.assert_allclose(Tensor([1.0,2.0,3.0]).expand(2,3).numpy(), [[1,2,3],[1,2,3]])
  def test_sum_2d(self):
    np.testing.assert_allclose(Tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]).sum(1).numpy(), [6,15])
  def test_int_lt(self):
    np.testing.assert_equal((Tensor([1,2,3])<2).numpy(), [True,False,False])
  def test_float_abs(self):
    np.testing.assert_allclose(Tensor([-1.0,2.0,-3.0]).abs().numpy(), [1,2,3])
  def test_sqrt(self):
    np.testing.assert_allclose(Tensor([4.0,9.0,16.0]).sqrt().numpy(), [2,3,4])
  def test_sqrt_neg(self):
    self.assertTrue(np.isnan(Tensor([-1.0]).sqrt().numpy()[0]))
  def test_rcp(self):
    np.testing.assert_allclose(Tensor([2.0,4.0,5.0]).reciprocal().numpy(), [0.5,0.25,0.2])
  def test_rsqrt(self):
    np.testing.assert_allclose(Tensor([4.0,16.0,25.0]).rsqrt().numpy(), [0.5,0.25,0.2])
  def test_trunc(self):
    np.testing.assert_allclose(Tensor([1.9,-1.9,3.2]).trunc().numpy(), [1,-1,3])
  def test_floor(self):
    np.testing.assert_allclose(Tensor([1.2,-1.2,3.9]).floor().numpy(), [1,-2,3])
  def test_ceil(self):
    np.testing.assert_allclose(Tensor([1.2,-1.2,3.1]).ceil().numpy(), [2,-1,4])
  def test_sin(self):
    np.testing.assert_allclose(Tensor([0.0, 0.5*np.pi]).sin().numpy(), [0, 1], atol=1e-5)
    from tinygrad import Context
    with Context(TRANSCENDENTAL=2):
      np.testing.assert_allclose(Tensor([30.0]).sin().numpy(), [np.sin(np.float32(30))], atol=2e-5, rtol=1e-5)
      np.testing.assert_allclose(Tensor([1.0], dtype=dtypes.half).sin().numpy(), [np.sin(np.float16(1.0))], atol=1e-2, rtol=5e-3)
      np.testing.assert_allclose(Tensor([30.0], dtype=dtypes.half).sin().numpy(), [np.sin(np.float16(30.0))], atol=1e-2, rtol=5e-3)
  def test_exp2(self):
    np.testing.assert_allclose(Tensor([0.0,1.0,2.0]).exp2().numpy(), [1,2,4])
  def test_log2(self):
    np.testing.assert_allclose(Tensor([1.0,2.0,4.0]).log2().numpy(), [0,1,2])
  def test_log2_rpt(self):
    np.testing.assert_allclose(Tensor([1.0,2.0,4.0,8.0]).log2().numpy(), [0,1,2,3])
  def test_plus_big(self):
    np.testing.assert_allclose((Tensor.ones(16).contiguous()+Tensor.ones(16).contiguous()).numpy(), [2]*16)
  def test_eye_clone(self):
    np.testing.assert_allclose(Tensor.eye(2).clone().numpy(), [[1.0,0.0],[0.0,1.0]])
  def test_gemm(self):
    N = 2
    a = Tensor.ones(N, N).contiguous()
    np.testing.assert_allclose((a @ Tensor.eye(N).clone()).numpy(), [[1.0,1.0],[1.0,1.0]])
  def test_gemm_multi_wg(self):
    a = Tensor.ones(8, 16).contiguous()
    np.testing.assert_allclose((a @ Tensor.ones(16, 4).contiguous()).numpy(), np.full((8, 4), 16.0))
  def test_mad_f32(self):
    np.testing.assert_allclose((Tensor([1.0,2.0,3.0])*Tensor([4.0,5.0,6.0])+7).numpy(), [11,17,25])
  def test_float_lt(self):
    np.testing.assert_equal((Tensor([1.0,2.0,3.0])<2).numpy(), [True,False,False])
  def test_cast_f32_i32(self):
    np.testing.assert_equal(Tensor([1.0,2.0,3.0]).cast(dtypes.int32).numpy(), [1,2,3])
    np.testing.assert_equal(Tensor([1.9,-1.9]).cast(dtypes.int32).numpy(), [1,-1])
  def test_bool_and(self):
    np.testing.assert_equal((Tensor([True,True,False])&Tensor([True,False,False])).numpy(), [True,False,False])
  def test_bool_or(self):
    np.testing.assert_equal((Tensor([True,True,False])|Tensor([True,False,False])).numpy(), [True,True,False])
  def test_bool_all(self):
    np.testing.assert_equal(Tensor.ones(16).bool().all().numpy(), True)
    np.testing.assert_equal(Tensor.ones(32).bool().all().numpy(), True)
  def test_sum(self):
    np.testing.assert_equal(Tensor.ones(256).contiguous().sum().numpy(), 256)
  def test_symbolic_reduce(self):
    i = Variable('i', 1, 10)
    ones = Tensor.ones(10).contiguous()
    for s in (2, 5):
      np.testing.assert_equal(ones[:i.bind(s)].sum().numpy(), s)
  def test_symbolic(self):
    i = Variable('i', 1, 10)
    ones = Tensor.ones(10).contiguous()
    for s in (2, 5):
      np.testing.assert_allclose((ones[:i.bind(s)] + 1).contiguous()[:s].numpy(), [2.0]*s)
  def test_rand(self):
    Tensor.manual_seed(42)
    got = Tensor.rand(4).numpy()
    Tensor.manual_seed(42)
    np.testing.assert_allclose(got, Tensor.rand(4, device="CPU").numpy())
  def test_half_add(self):
    np.testing.assert_allclose((Tensor([1,2,3], dtype=dtypes.half)+1).numpy(), [2,3,4])
  def test_half_mul(self):
    np.testing.assert_allclose((Tensor([1,2,3], dtype=dtypes.half)*2).numpy(), [2,4,6])
  def test_half_neg(self):
    np.testing.assert_allclose((-Tensor([1,2,3], dtype=dtypes.half)).numpy(), [-1,-2,-3])
  def test_half_sqrt(self):
    np.testing.assert_allclose(Tensor([4,9,16], dtype=dtypes.half).sqrt().numpy(), [2,3,4])
  def test_half_min(self):
    np.testing.assert_allclose(Tensor([1.,3.,2.], dtype=dtypes.half).minimum(2.).numpy(), [1,2,2])
  def test_half_max(self):
    np.testing.assert_allclose(Tensor([1.,3.,2.], dtype=dtypes.half).maximum(2.).numpy(), [2,3,2])
  def test_half_exp2(self):
    np.testing.assert_allclose(Tensor([0.,1.,2.], dtype=dtypes.half).exp2().numpy(), [1,2,4])
  def test_half_log2(self):
    np.testing.assert_allclose(Tensor([1.,2.,4.], dtype=dtypes.half).log2().numpy(), [0,1,2])
  def test_half_trunc(self):
    np.testing.assert_allclose(Tensor([1.9,-1.9,3.2], dtype=dtypes.half).trunc().numpy(), [1,-1,3])
  def test_half_floor(self):
    np.testing.assert_allclose(Tensor([1.2,-1.2,3.9], dtype=dtypes.half).floor().numpy(), [1,-2,3])
  def test_half_ceil(self):
    np.testing.assert_allclose(Tensor([1.2,-1.2,3.1], dtype=dtypes.half).ceil().numpy(), [2,-1,4])
  def test_half_rsqrt(self):
    np.testing.assert_allclose(Tensor([4,16,25], dtype=dtypes.half).rsqrt().numpy(), [0.5,0.25,0.2], atol=1e-3)
  def test_clip01(self):
    np.testing.assert_allclose(Tensor([-2.0,-0.5,0.0,0.5,2.0]).clip(0,1).numpy(), [0,0,0,0.5,1])
  def test_hgemm(self):
    N = 4
    a = Tensor.ones(N, N, dtype=dtypes.half).contiguous()
    np.testing.assert_allclose((a @ Tensor.eye(N, dtype=dtypes.half).clone()).numpy(), np.ones((N, N)))
  def test_diag(self):
    np.testing.assert_allclose(Tensor([1.0, 2.0, 3.0, 4.0, 5.0]).diag().numpy(), np.diag([1, 2, 3, 4, 5]))
  def test_argmax_first(self):
    np.testing.assert_equal(Tensor([2, 2]).argmax().numpy(), 0)
    np.testing.assert_equal(Tensor([1, 2, 2]).argmax().numpy(), 1)
  def test_argmax_intmin(self):
    np.testing.assert_equal(Tensor([-2**31, 0], dtype=dtypes.int32).argmax().numpy(), 1)
    np.testing.assert_equal(Tensor([0, -2**31], dtype=dtypes.int32).argmin().numpy(), 1)
  def test_half_mem_isa(self):
    # stp.u16 decode omits TYPE_HALF; file still follows ISA #type-half
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.emu import _half_mem
    self.assertTrue(_half_mem({"TYPE": mesa.TYPE_U16}))
    self.assertTrue(_half_mem({"TYPE": mesa.TYPE_F16}))
    self.assertTrue(_half_mem({"TYPE": mesa.TYPE_S16}))
    self.assertTrue(_half_mem({"TYPE": mesa.TYPE_U8}))
    self.assertFalse(_half_mem({"TYPE": mesa.TYPE_U32}))
    self.assertFalse(_half_mem({"TYPE": mesa.TYPE_F32}))
    self.assertTrue(_half_mem({"TYPE": mesa.TYPE_U16, "TYPE_HALF": 1}))

  def _emu_wave(self, nt=4):
    from test.mockgpu.qcom.emu import _Wave, _wg
    wg = _wg(nt)
    wg.reset()
    return wg, _Wave(0, nt)

  def test_divergent_br(self):
    from test.mockgpu.qcom.emu import _exec_cat0, _step_wave, DONE
    wg, wave = self._emu_wave(4)
    for tid in (0, 2): wg.gpr_mv[tid * 256 + 248] = 1  # p0 taken on lanes 0,2
    instrs = [{"NAME": "br", "IMMED": 2, "COMP1": 0}, {"NAME": "nop"}, {"NAME": "end"}]
    _exec_cat0(wave, wg, instrs[0], "br")
    # forward br parks taken (0b0101) at pc 2, falls through with 0b1010
    self.assertEqual(int(wave.pc_mv[0]), 1)
    self.assertEqual(int(wave.act_mv[0]), 0b1010)
    self.assertEqual(int(wave.sp_mv[0]), 1)
    self.assertEqual(int(wave.ppc_mv[0]), 2)
    self.assertEqual(int(wave.pmsk_mv[0]), 0b0101)
    _step_wave(wave, instrs, wg, len(instrs))
    self.assertEqual(int(wave.flg_mv[0]) & DONE, DONE)

  def test_getone_jp(self):
    from test.mockgpu.qcom.emu import _exec_cat0, _step_wave, DONE, _JP
    wg, wave = self._emu_wave(4)
    instrs = [{"NAME": "getone", "IMMED": 2}, {"NAME": "end"}, {"NAME": "nop", "JP": 1}, {"NAME": "end"}]
    _exec_cat0(wave, wg, instrs[0], "getone")
    self.assertEqual(int(wave.pc_mv[0]), 2)
    self.assertEqual(int(wave.act_mv[0]), 0b0001)
    self.assertEqual(int(wave.sp_mv[0]), 1)
    self.assertEqual(int(wave.ppc_mv[0]), _JP)
    self.assertEqual(int(wave.pmsk_mv[0]), 0b1110)
    _step_wave(wave, instrs, wg, len(instrs))
    self.assertEqual(int(wave.flg_mv[0]) & DONE, DONE)

  def test_predt_br(self):
    from test.mockgpu.qcom.emu import _exec_cat0, _step_wave, DONE
    wg, wave = self._emu_wave(4)
    # already predicated to lanes 0,1; p0 is 1 on everyone (stale on 2,3)
    for tid in range(4): wg.gpr_mv[tid * 256 + 248] = 1
    wg.gpr_mv[0 * 256 + 249] = 1  # p1 on exec lane 0
    wg.gpr_mv[2 * 256 + 249] = 1  # p1 on predicated-off lane 2 — must not vote
    wave.pmode_mv[0], wave.pmask_mv[0] = 1, 0b0011
    instrs = [{"NAME": "predt"}, {"NAME": "br", "IMMED": 2, "COMP1": 1}, {"NAME": "nop"}, {"NAME": "end"}]
    _exec_cat0(wave, wg, instrs[0], "predt")
    self.assertEqual(int(wave.pmode_mv[0]), 1)
    self.assertEqual(int(wave.pmask_mv[0]), 0b0011)  # snap exec only, not 0b1111
    _exec_cat0(wave, wg, instrs[1], "br")
    # em=0b0011, taken=p1 among em=0b0001, idle=0b1100; park taken at pc 3
    self.assertEqual(int(wave.pc_mv[0]), 2)
    self.assertEqual(int(wave.act_mv[0]), 0b1110)
    self.assertEqual(int(wave.sp_mv[0]), 1)
    self.assertEqual(int(wave.ppc_mv[0]), 3)
    self.assertEqual(int(wave.pmsk_mv[0]), 0b0001)
    _step_wave(wave, instrs, wg, len(instrs))
    self.assertEqual(int(wave.flg_mv[0]) & DONE, DONE)

  def test_park_branchstack(self):
    wg, wave = self._emu_wave(2)
    wave.park_limit = 1
    wave.park(2, 0b01)
    with self.assertRaisesRegex(RuntimeError, "park overflow"):
      wave.park(3, 0b10)

  def test_park_same_pc(self):
    from test.mockgpu.qcom.emu import _exec_cat0
    wg, wave = self._emu_wave(4)
    wave.park_limit = 1
    # divergent loop: one lane breaks per iter onto the same PC; HW shares that frame
    instrs = [{"NAME": "nop"}, {"NAME": "br", "IMMED": 2, "COMP1": 0},
              {"NAME": "jump", "IMMED": 0xfffffffe}, {"NAME": "end"}]
    for lane in range(3):
      wave.pc_mv[0] = 1
      for tid in range(4): wg.gpr_mv[tid * 256 + 248] = int(tid == lane)
      _exec_cat0(wave, wg, instrs[1], "br")
      self.assertEqual(int(wave.sp_mv[0]), 1)
      self.assertEqual(int(wave.ppc_mv[0]), 3)
      self.assertEqual(int(wave.pmsk_mv[0]), (1 << (lane + 1)) - 1)
      _exec_cat0(wave, wg, instrs[2], "jump")
    self.assertEqual(int(wave.act_mv[0]), 0b1000)

  def test_cmps_01(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.emu import _make_enc, _run_wave_op
    wg, wave = self._emu_wave(1)
    wg.gpr_mv[1], wg.gpr_mv[2] = 1, 2
    def cmps(cond):
      raw = {"NAME": "cmps.s", "DST": 0, "SRC1": {"SRC": 1, "_": 1}, "SRC2": {"SRC": 2, "_": 2}, "COND": cond}
      enc, enc_addr = _make_enc([raw])
      wave.keep.append(enc)
      wave.pc_mv[0] = 0
      wave.c_bufs = wg.c_bufs(0, wave.addr, enc_addr)
      _run_wave_op(raw, 1, wave.c_bufs)
    cmps(mesa.IR3_COND_LT)
    self.assertEqual(int(wg.gpr_mv[0]), 1)
    cmps(mesa.IR3_COND_GT)
    self.assertEqual(int(wg.gpr_mv[0]), 0)

  def test_absneg_f_sat(self):
    import struct
    from test.mockgpu.qcom.emu import _make_enc, _run_wave_op
    wg, wave = self._emu_wave(1)
    wg.gpr_mv[1] = struct.unpack("I", struct.pack("f", -2.0))[0]
    def absneg(sat):
      raw = {"NAME": "absneg.f", "DST": 0, "SRC1": {"SRC": 1, "_": 1, "ABSNEG": 2}, "SAT": sat}
      enc, enc_addr = _make_enc([raw])
      wave.keep.append(enc)
      wave.pc_mv[0] = 0
      wave.c_bufs = wg.c_bufs(0, wave.addr, enc_addr)
      _run_wave_op(raw, 1, wave.c_bufs)
      return struct.unpack("f", struct.pack("I", int(wg.gpr_mv[0])))[0]
    self.assertEqual(absneg(0), 2.0)
    self.assertEqual(absneg(1), 1.0)

  def test_ldg_neg_off(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.emu import _make_enc, _run_wave_op, _host_buf
    wg, wave = self._emu_wave(1)
    addr, mem, mv = _host_buf(2, "I")
    mv[0], mv[1] = 0xA1A1A1A1, 0xB2B2B2B2
    ptr = addr + 4  # r2:r3 points at word 1; OFF=-4 should read word 0
    wg.gpr_mv[2], wg.gpr_mv[3] = ptr & 0xffffffff, ptr >> 32
    def ldg(off):
      raw = {"NAME": "ldg", "DST": 0, "SRC1": {"SRC": 2, "_": 2}, "OFF": off, "TYPE": mesa.TYPE_U32, "SIZE": 1}
      enc, enc_addr = _make_enc([raw])
      wave.keep.append(enc)
      wave.pc_mv[0] = 0
      wave.c_bufs = wg.c_bufs(0, wave.addr, enc_addr)
      _run_wave_op(raw, 1, wave.c_bufs)
    ldg(0)
    self.assertEqual(int(wg.gpr_mv[0]), 0xB2B2B2B2)
    ldg(-4)
    self.assertEqual(int(wg.gpr_mv[0]), 0xA1A1A1A1)

  def test_stl_neg_off(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.emu import _make_enc, _run_wave_op
    wg, wave = self._emu_wave(1)
    wg.reset(lds=True)
    def run(name, off, **fields):
      raw = {"NAME": name, "OFF": off, "TYPE": mesa.TYPE_U32, "SIZE": 1, **fields}
      enc, enc_addr = _make_enc([raw])
      wave.keep.append(enc)
      wave.pc_mv[0] = 0
      wave.c_bufs = wg.c_bufs(0, wave.addr, enc_addr)
      _run_wave_op(raw, 1, wave.c_bufs)
    def stl(off):
      run("stl", off, DST=0, SRC={"SRC": 1, "_": 1})
    def ldl(off):
      wg.gpr_mv[2] = 0
      run("ldl", off, DST=2, SRC=0)
    wg.gpr_mv[0], wg.gpr_mv[1] = 4, 0xA1A1A1A1  # ptr at word 1; OFF=-4 / 0x1ffc → word 0
    for off in (-4, 0x1ffc):
      stl(off)
      ldl(off)
      self.assertEqual(int(wg.gpr_mv[2]), 0xA1A1A1A1)
    wg.gpr_mv[0], wg.gpr_mv[1] = 8192, 0xB2B2B2B2  # mesa stp_ldp_offset: 8192 + s13(0x1000) = 4096
    stl(0x1000)
    wg.gpr_mv[0] = 4096
    ldl(0)
    self.assertEqual(int(wg.gpr_mv[2]), 0xB2B2B2B2)

  def test_bar_waves(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.emu import _wg, _waves, _make_enc, _run_waves, DONE
    nt = 128
    wg = _wg(nt)
    wg.reset(lds=True)
    waves = _waves(wg)
    for tid in range(nt):
      wg.gpr_mv[tid * 256 + 0] = tid * 4
      wg.gpr_mv[tid * 256 + 1] = 0xA0000000 | tid
      wg.gpr_mv[tid * 256 + 3] = (tid ^ 64) * 4
    instrs = [
      {"NAME": "stl", "DST": 0, "SRC": {"SRC": 1, "_": 1}, "OFF": 0, "TYPE": mesa.TYPE_U32, "SIZE": 1},
      {"NAME": "bar"},
      {"NAME": "ldl", "DST": 2, "SRC": 3, "OFF": 0, "TYPE": mesa.TYPE_U32, "SIZE": 1},
      {"NAME": "end"},
    ]
    enc, enc_addr = _make_enc(instrs)
    for w in waves:
      w.keep.append(enc)
      w.c_bufs = wg.c_bufs(w.base, w.addr, enc_addr)
    _run_waves(waves, instrs, wg)
    for w in waves: self.assertEqual(int(w.flg_mv[0]) & DONE, DONE)
    for tid in range(nt):
      self.assertEqual(int(wg.gpr_mv[tid * 256 + 2]), 0xA0000000 | (tid ^ 64))

if __name__ == '__main__':
  unittest.main()
