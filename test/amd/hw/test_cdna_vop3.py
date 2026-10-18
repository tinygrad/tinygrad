"""CDNA VOP3 instruction coverage.

Exercises generated CDNA pcode end-to-end in the emulator and compares against
gfx950 hardware when USE_HW=1.
"""
import ctypes, unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import USE_HW, assemble
from test.mockgpu.amd.emu import run_asm

def _code(instructions: list, out_reg: int = 2) -> list:
  return [
    cdna.s_mov_b32(cdna.s[80], cdna.s[0]),
    cdna.s_mov_b32(cdna.s[81], cdna.s[1]),
    cdna.v_mov_b32_e32(cdna.v[255], cdna.v[0]),
    *instructions,
    cdna.s_load_dwordx2(cdna.s[92:93], cdna.s[80:81], 0, soffset=NULL, imm=1),
    cdna.s_waitcnt(0),
    cdna.v_lshlrev_b32_e32(cdna.v[240], 2, cdna.v[255]),
    cdna.global_store_dword(addr=cdna.v[240], data=cdna.v[out_reg], saddr=cdna.s[92:93], offset=0),
    cdna.s_endpgm(),
  ]

def _run_emu(instructions: list, out_reg: int = 2, lanes: int = 1) -> list[int]:
  out_buf = (ctypes.c_uint32 * lanes)(*([0] * lanes))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = assemble(_code(instructions, out_reg))
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, lanes, 1, 1, ctypes.addressof(args),
                   0x19c | (128 << 15), 0x10000, arch="cdna")
  assert result == 0, f"run_asm failed with {result}"
  return list(out_buf)

def _run_hw(instructions: list, out_reg: int = 2, lanes: int = 1) -> list[int]:
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.engine.realize import run_linear

  dev = Device["AMD"]
  if dev.arch != "gfx950": raise unittest.SkipTest("requires gfx950 hardware")
  out = Tensor.zeros(lanes, dtype=dtypes.uint32, device=dev.device).contiguous().realize()
  sink = UOp.sink(UOp.param(0, dtypes.uint32, (lanes,)), UOp.special(lanes, "lidx0"), arg=KernelInfo("test"))
  prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in _code(instructions, out_reg)))))
  run_linear(UOp(Ops.LINEAR, src=(prg.call(out.uop.buf_uop),)), wait=True)
  return out.tolist()

def run_cdna_lanes(instructions: list, out_reg: int = 2, lanes: int = 64) -> list[int]:
  emu = _run_emu(instructions, out_reg, lanes)
  if not USE_HW: return emu
  hw = _run_hw(instructions, out_reg, lanes)
  if emu != hw: raise AssertionError(f"Emulator vs Hardware mismatch: emu={[hex(x) for x in emu]} hw={[hex(x) for x in hw]}")
  return hw

def run_cdna(instructions: list, out_reg: int = 2) -> int: return run_cdna_lanes(instructions, out_reg, 1)[0]

class TestCDNAVOP3(unittest.TestCase):
  def test_cvt_pk_fp8_f32_preserves_upper_half(self):
    """V_CVT_PK_FP8_F32 with OPSEL[3]=0 writes only D[15:0]."""
    out = run_cdna([
      cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
      cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
      cdna.v_mov_b32_e32(cdna.v[0], 1.0),
      cdna.v_mov_b32_e32(cdna.v[1], 2.0),
      cdna.v_cvt_pk_fp8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
    ])
    self.assertEqual(out, 0xdead4038)

  def test_cvt_pk_bf8_f32_overflow_and_inf(self):
    """V_CVT_PK_BF8_F32 converts finite overflow and infinities to E5M2 infinities."""
    for name, bits, expected in [
      ("finite_overflow", 0x47700000, 0x7c),
      ("pos_inf", 0x7f800000, 0x7c),
      ("neg_inf", 0xff800000, 0xfc),
    ]:
      with self.subTest(name=name):
        out = run_cdna([
          cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
          cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
          cdna.s_mov_b32(cdna.s[0], bits),
          cdna.v_mov_b32_e32(cdna.v[0], cdna.s[0]),
          cdna.v_mov_b32_e32(cdna.v[1], 1.0),
          cdna.v_cvt_pk_bf8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
        ])
        self.assertEqual(out, 0xdead3c00 | expected)
