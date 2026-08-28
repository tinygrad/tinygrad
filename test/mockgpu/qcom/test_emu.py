import ctypes, struct, unittest
from test.mockgpu.qcom.emu import decode_shader, Wave, CSState, execute_inst, Inst, Operand, _mem_width, run_wave

# Real IR3 words from DEBUG=7 TestTiny.test_plus (little-endian 64-bit instructions)
PLUS_HEX = """
47180803_201f0000
46d80803_20020003
65000403_0003301e
46d80000_20020000
42100008_00031003
42100004_00001002
4210000d_00001004
42100009_00031001
42100003_00031005
42904000_10020004
42984002_1004000d
2009400a_00000000
2009400f_00000002
00000000_00000000
42100006_00001000
42100005_0008000a
4210080e_0003000f
42984801_10000006
c006000b_01810001
2009400c_00000001
c0060010_01834001
00000200_00000000
42100007_0009000c
5018080b_0010000b
00000200_00000000
c0c60d00_01800016
03000000_00000000
""".split()

def _image(hexes=PLUS_HEX) -> bytes:
  words = []
  for h in hexes:
    hi, lo = h.split('_')
    words += [int(lo, 16), int(hi, 16)]
  return struct.pack('<' + 'I' * len(words), *words)

class TestIR3Decode(unittest.TestCase):
  def test_plus_listing_names(self):
    insts = decode_shader(_image())
    names = [i.name for i in insts]
    self.assertEqual(names[0], 'ashr.b')
    self.assertEqual(names[4], 'add.u')
    self.assertIn('cmps.u.lt', names)
    self.assertEqual(names[-2], 'stg')
    self.assertEqual(names[-1], 'end')
    self.assertEqual(insts[0].dst.packed if insts[0].dst is not None else None, 3)  # r0.w
    self.assertEqual(insts[0].srcs[0].packed, 0)  # r0.x
    self.assertEqual(insts[0].srcs[1].kind, 'imm')
    self.assertEqual(insts[0].srcs[1].imm, 31)

class TestIR3ALU(unittest.TestCase):
  def _run(self, inst: Inst, gprs: dict[int, int], const: list[int]|None=None) -> Wave:
    wave = Wave(1)
    for p, v in gprs.items(): wave.write_gpr(p, 0, False, v)
    st = CSState(const or [0] * 16, bytearray(1024), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    return wave

  def test_add_u(self):
    insts = decode_shader(_image())
    add = next(i for i in insts if i.name == 'add.u' and i.dst and i.dst.packed == 8)
    # add.u r2.x, c0.w, r0.w
    wave = self._run(add, {3: 10}, const=[0, 0, 0, 20])
    self.assertEqual(wave.read_gpr(8, 0, False), 30)

  def test_cmps_to_p0(self):
    # cmps.s.eq p0.x, r0.x, 0  from test_sum reduce
    insts = decode_shader(_image(["42b400f8_20000000", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cmps.s.eq')
    self.assertIsNotNone(inst.dst)
    assert inst.dst is not None
    self.assertEqual(inst.dst.kind, 'pred')
    wave = self._run(inst, {0: 0})
    self.assertEqual(wave.pred[0], 1)
    wave = self._run(inst, {0: 3})
    self.assertEqual(wave.pred[0], 0)

  def test_br_offset_is_relative_to_branch(self):
    wave = Wave(1)
    wave.pc, wave.pred[0] = 175, 1
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(Inst(name='br', line='br p0.x, #4', extra={'immed': 4}), wave, st, 0)
    self.assertEqual(wave.pc, 179)

  def test_brao_inverted_preds(self):
    # brao !p0.y, !p0.x, #14 — take iff NOT (p0.x AND p0.y)
    insts = decode_shader(_image(["00b02020_0000000e", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'brao')
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    wave = Wave(1)
    wave.pc, wave.pred[0] = 19, 0b11  # both in-bounds
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.pc, 19)  # not taken
    wave.pc, wave.pred[0] = 19, 0b01  # p0.x set, p0.y clear → OOB
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.pc, 33)

  def test_divergent_brao_runs_both_paths(self):
    # if p0.x: r0.x = 2; else: r0.x = 1. Both lanes must execute their path.
    insts = [
      Inst(name='br', line='br p0.x, #3', extra={'immed': 3, 'comp1': 0}),
      Inst(name='mov.u32u32', line='mov r0.x, 1', dst=Operand('DST', packed=0),
           srcs=[Operand('SRC1', kind='imm', imm=1)]),
      Inst(name='jump', line='jump #2', extra={'immed': 2}),
      Inst(name='mov.u32u32', line='mov r0.x, 2', dst=Operand('DST', packed=0),
           srcs=[Operand('SRC1', kind='imm', imm=2)]),
      Inst(name='end', line='end'),
    ]
    wave = Wave(2)
    wave.pred[0], wave.pred[1] = 0, 1
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    run_wave(insts, wave, st)
    self.assertEqual(wave.read_gpr(0, 0, False), 1)
    self.assertEqual(wave.read_gpr(0, 1, False), 2)

  def test_pred_y_does_not_clobber_x(self):
    px = decode_shader(_image(["42b400f8_20000000", "03000000_00000000"]))[0]  # cmps.s.eq p0.x
    py = decode_shader(_image(["42b000f9_000727ff", "03000000_00000000"]))[0]  # cmps.s.lt p0.y, -1, r1.w
    self.assertEqual(py.dst.packed if py.dst else None, 249)
    wave = self._run(px, {0: 0})
    self.assertEqual(wave.pred[0] & 1, 1)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    wave.write_gpr(7, 0, False, 3)
    execute_inst(py, wave, st, 0)
    self.assertEqual(wave.pred[0] & 1, 1)
    self.assertEqual((wave.pred[0] >> 1) & 1, 1)

  def test_predf_uses_saved_exec_not_then_mask(self):
    # predt; predf must run the else lanes, not exec&=~pred on the then-mask (which is empty)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    wave = Wave(2)
    wave.pred[0], wave.pred[1] = 1, 0
    execute_inst(Inst(name='predt', line='predt'), wave, st, 0)
    self.assertEqual(wave.exec, 0b01)
    execute_inst(Inst(name='predf', line='predf'), wave, st, 0)
    self.assertEqual(wave.exec, 0b10)
    execute_inst(Inst(name='prede', line='prede'), wave, st, 0)
    self.assertEqual(wave.exec, 0b11)
    buf = (ctypes.c_uint16 * 2)(0x3c00, 0x4000)  # 1.0, 2.0 in f16
    addr = ctypes.addressof(buf)
    insts = decode_shader(_image(["c0040001_01828001", "03000000_00000000"]))
    ldg = insts[0]
    self.assertEqual(ldg.name, 'ldg')
    assert ldg.dst is not None
    self.assertTrue(ldg.dst.half)
    wave = Wave(1)
    wave.write_gpr(ldg.srcs[0].packed, 0, False, addr & 0xffffffff)
    wave.write_gpr(ldg.srcs[0].packed + 1, 0, False, addr >> 32)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(ldg, wave, st, 0)
    self.assertEqual(wave.read_gpr(ldg.dst.packed, 0, True), 0x3c00)

  def test_mul_f_half_srcs_to_full_dst(self):
    inst = Inst(name='mul.f', line='mul.f r6.y, hr0.y, hr0.w', dst=Operand('DST', packed=25),
                srcs=[Operand('SRC1', packed=1, half=True), Operand('SRC2', packed=3, half=True)])
    wave = Wave(1)
    wave.write_gpr(1, 0, True, 0x3c00)
    wave.write_gpr(3, 0, True, 0x3c00)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(25, 0, False), 0x3f800000)

  def test_swz_swaps_two_regs(self):
    insts = decode_shader(_image(["240cc0a4_00a5a4a5", "03000000_00000000"]))
    inst = insts[0]
    self.assertTrue(inst.name.startswith('swz'))
    wave = Wave(1)
    wave.write_gpr(164, 0, False, 0x1111)
    wave.write_gpr(165, 0, False, 0x2222)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(164, 0, False), 0x2222)
    self.assertEqual(wave.read_gpr(165, 0, False), 0x1111)

  def test_floor_f(self):
    insts = decode_shader(_image(["4130001c_00000018", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'floor.f')
    wave = self._run(inst, {24: 0x3fc00000})  # 1.5f
    self.assertEqual(wave.read_gpr(28, 0, False), 0x3f800000)  # 1.0f

  def test_trunc_f(self):
    insts = decode_shader(_image(["51b00006_00000002", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'trunc.f')
    wave = self._run(inst, {2: 0xc0c80000})  # -6.25f
    self.assertEqual(wave.read_gpr(6, 0, False), 0xc0c00000)  # -6.0f

  def test_mova_sets_a0(self):
    insts = decode_shader(_image(["201100f4_00000000", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'mova')
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 3)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.a0[0], 3)

  def test_cmps_f_lt_flut_imm(self):
    insts = decode_shader(_image(["50b00009_28030009", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cmps.f.lt')
    wave = self._run(inst, {9: 0x3f800000})  # 1.0 < 2.0
    self.assertEqual(wave.read_gpr(9, 0, False), 1)
    wave = self._run(inst, {9: 0x40000000})  # 2.0 < 2.0
    self.assertEqual(wave.read_gpr(9, 0, False), 0)
    # sel.b32 dst, true, cond, false
    inst = Inst(name='sel.b32', line='sel.b32 r0.x, r1.x, r2.x, r3.x', dst=Operand('DST', packed=0),
                srcs=[Operand('SRC1', packed=4), Operand('SRC2', packed=8), Operand('SRC3', packed=12)])
    wave = self._run(inst, {4: 11, 8: 1, 12: 22})
    self.assertEqual(wave.read_gpr(0, 0, False), 11)
    wave = self._run(inst, {4: 11, 8: 0, 12: 22})
    self.assertEqual(wave.read_gpr(0, 0, False), 22)

  def test_sel_b16_reads_half_cond(self):
    # cat3 SRC2 HALF is emitted before SRC2; cond is hr0.x, not full r0.x
    insts = decode_shader(_image(["64000801_00028001", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'sel.b16')
    self.assertTrue(inst.srcs[1].half, msg=f"SRC2 half: {inst.srcs}")
    wave = Wave(1)
    wave.write_gpr(1, 0, True, 0xbc00)  # -1.0 f16 true-val
    wave.write_gpr(0, 0, True, 0)       # cond false in half file
    wave.write_gpr(0, 0, False, 0x1234) # full r0.x must be ignored
    wave.write_gpr(2, 0, True, 0xc400)  # -4.0 f16 false-val
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(1, 0, True), 0xc400)

  def test_sel_b32_cond_is_src1(self):
    inst = Inst(name='sel.b32', line='sel.b32 r0.x, r1.x, r2.x, r3.x', dst=Operand('DST', packed=0),
                srcs=[Operand('SRC1', packed=4), Operand('SRC2', packed=8), Operand('SRC3', packed=12)])
    wave = self._run(inst, {4: 11, 8: 1, 12: 22})
    self.assertEqual(wave.read_gpr(0, 0, False), 11)
    wave = self._run(inst, {4: 11, 8: 0, 12: 22})
    self.assertEqual(wave.read_gpr(0, 0, False), 22)

  def test_shrm_merges_shifted_src(self):
    insts = decode_shader(_image(["64070400_00003010", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'shrm')
    val_p, merge_p = inst.srcs[1].packed, inst.srcs[2].packed
    dst_p = inst.dst.packed if inst.dst is not None else 0
    # shrm dst, 16, src, merge  →  (src >> 16) & merge
    wave = self._run(inst, {val_p: 0xABCD1234, merge_p: 0x00005678})
    self.assertEqual(wave.read_gpr(dst_p, 0, False), 0x0000ABCD & 0x00005678)

  def test_half_const_is_low16_of_same_slot(self):
    # and.b r3.y, hr0.y, hc4.y — HALF const is c4.y low 16, not packed>>1
    insts = decode_shader(_image(["5380400d_10110001", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'and.b')
    self.assertTrue(inst.srcs[1].half)
    self.assertEqual(inst.srcs[1].packed, 17)
    wave = Wave(1)
    wave.write_gpr(1, 0, True, 0x461c)
    const = [0] * 20
    const[17] = 0x7fff
    st = CSState(const, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(13, 0, False), 0x461c)

  def test_half_float_const_from_f32_slot(self):
    # mul.f hr, hr, hc with f32 1.333 (0x3faaa000) in the slot; low 16 is 0xa000, not a packed f16
    inst = Inst(name='mul.f', line='mul.f hr0.x, hr0.x, hc4.w', dst=Operand('DST', packed=0, half=True),
                srcs=[Operand('SRC1', packed=0, half=True), Operand('SRC2', kind='const', packed=19, half=True)])
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 0x3c00)  # 1.0 f16
    const = [0] * 24
    const[19] = 0x3faaa000  # 1.333 f32
    st = CSState(const, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    got = struct.unpack('<e', struct.pack('<H', wave.read_gpr(0, 0, True) & 0xFFFF))[0]
    self.assertAlmostEqual(got, 1.333, places=2)

  def test_relative_gpr(self):
    inst = Inst(name='add.u', line='add.u r0.x, r<a0.x + 2>.y, r1.x',
                dst=Operand('DST', packed=0),
                srcs=[Operand('SRC1', kind='rel_gpr', packed=1, rel_off=2), Operand('SRC2', packed=4)])
    wave = Wave(1)
    wave.a0[0] = 1
    # r<(1+2)>.y = packed (3<<2)|1 = 13
    wave.write_gpr(13, 0, False, 10)
    wave.write_gpr(4, 0, False, 5)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1<<30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(0, 0, False), 15)

  def test_shr_full_src_to_half_dst_uses_5bit_shift(self):
    # shr.b hr1.x, r3.x, 16  must shift by 16, not 16&15==0
    insts = decode_shader(_image(["46f04004_2010000c", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'shr.b')
    self.assertTrue(inst.dst.half if inst.dst else False)
    self.assertFalse(inst.srcs[0].half)
    wave = Wave(1)
    wave.write_gpr(12, 0, False, 0x01020304)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(4, 0, True), 0x0102)

  def test_stg_u8_width_is_one_byte(self):
    # HALF on the data GPR must not widen stg.u8 to 2 bytes
    insts = decode_shader(_image(["c0cc2100_01800018", "03000000_00000000"]))
    inst = insts[0]
    self.assertTrue(inst.name.startswith('stg'))
    self.assertEqual(_mem_width(inst), 1)

  def test_cov_u8s32_sign_extends(self):
    insts = decode_shader(_image(["20194009_00000001", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cov.u8s32')
    wave = Wave(1)
    wave.write_gpr(1, 0, True, 0xd3)  # -45 as i8
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(9, 0, False), 0xFFFFFFD3)

  def test_cmps_s_half_uses_s16(self):
    # cmps.s.lt hr0.y, hr0.x, h(0)  must compare i16, not zero-extended u16
    insts = decode_shader(_image(["52a00001_20000000", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cmps.s.lt')
    self.assertTrue(inst.srcs[0].half)
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 0xffad)  # -83
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(1, 0, True), 1)

  def test_cov_u8s16_sign_extends(self):
    insts = decode_shader(_image(["30190000_00000000", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cov.u8s16')
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 0xfa)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(0, 0, True) & 0xFFFF, 0xFFFA)

  def test_cmps_f_half_uses_f16(self):
    # cmps.f.lt hr2.w, hr1.w, hc5.y  must compare f16, not f32 bitcast
    insts = decode_shader(_image(["40a0000b_10150007", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cmps.f.lt')
    self.assertTrue(inst.dst.half if inst.dst else False)
    self.assertTrue(inst.srcs[0].half)
    wave = Wave(1)
    wave.write_gpr(7, 0, True, 0x3c00)  # 1.0 f16
    const = [0] * 24
    const[21] = 0x4000  # hc5.y = 2.0 f16 in low 16 of c5.y
    st = CSState(const, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(11, 0, True), 1)  # 1.0 < 2.0

  def test_cmps_f_lt_half_flut_zero(self):
    # (rpt1)cmps.f.lt hr0.x, h(0.0), (r)hr0.x  is CMPLT(0, t) for f16; h(0.0) is not SRC1 HALF
    insts = decode_shader(_image(["50a80100_00002c00", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'cmps.f.lt')
    self.assertTrue(inst.srcs[0].half, msg=f"h(0.0) half: {inst.srcs}")
    self.assertTrue(inst.srcs[1].half)
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 0xbc00)  # hr0.x = -1.0
    wave.write_gpr(1, 0, True, 0x3c00)  # hr0.y =  1.0
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    execute_inst(inst, wave, st, 1)
    self.assertEqual(wave.read_gpr(0, 0, True), 0)  # 0.0 < -1.0
    self.assertEqual(wave.read_gpr(1, 0, True), 1)  # 0.0 <  1.0

  def test_hlog2(self):
    insts = decode_shader(_image(["91400000_00000000", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'hlog2')
    wave = Wave(1)
    wave.write_gpr(0, 0, True, 0x3e00)  # 1.5 f16
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    got = struct.unpack('<e', struct.pack('<H', wave.read_gpr(0, 0, True) & 0xFFFF))[0]
    self.assertAlmostEqual(got, 0.5849625, places=2)

  def test_log2_abs_neg_inf_is_pos_inf(self):
    # (abs) must be a float modifier; integer-abs of 0xff800000 yields log2(2^-126)=-126
    insts = decode_shader(_image(["90500003_00008002", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'log2')
    self.assertEqual(inst.srcs[0].absneg, 2)
    wave = Wave(1)
    wave.write_gpr(2, 0, False, 0xff800000)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(3, 0, False), 0x7f800000)

  def test_cat2_immed_is_11bit_signed(self):
    # add.u r3.y, r3.y, -23
    insts = decode_shader(_image(["4218080d_27e9000d", "03000000_00000000"]))
    inst = insts[0]
    self.assertEqual(inst.name, 'add.u')
    wave = Wave(1)
    wave.write_gpr(13, 0, False, 32)
    st = CSState([0]*16, bytearray(64), 0, 512, 0, 0, 0, 0, lambda a: 1 << 30)
    execute_inst(inst, wave, st, 0)
    self.assertEqual(wave.read_gpr(13, 0, False), 9)

if __name__ == '__main__': unittest.main()
