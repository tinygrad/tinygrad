"""End-to-end scaled MFMA semantics in the CDNA emulator."""
import ctypes, struct, unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import assemble
from test.mockgpu.amd.emu import run_asm

def _f32(x:int) -> float: return struct.unpack('<f', struct.pack('<I', x))[0]
def _genexp(lane:int, shift:int) -> int: return 124 + ((lane + (lane >> shift)) & 7)

def _scale_insts(shift:int, a_byte:int, b_byte:int) -> list:
  # The selected bytes of scales A and B are lane-dependent in [124,131]. Other bytes are neutral.
  # Adding the lane-group index makes the selected exponent differ across 32-wide K chunks.
  a_neutral = 0x7f7f7f7f & ~(0xff << (a_byte * 8))
  b_neutral = 0x7f7f7f7f & ~(0xff << (b_byte * 8))
  return [
    cdna.v_and_b32_e32(cdna.v[200], 7, cdna.v[255]),
    cdna.v_lshrrev_b32_e32(cdna.v[201], shift, cdna.v[255]),
    cdna.v_add_u32_e32(cdna.v[200], cdna.v[201], cdna.v[200]),
    cdna.v_and_b32_e32(cdna.v[200], 7, cdna.v[200]),
    cdna.v_add_u32_e32(cdna.v[200], 124, cdna.v[200]),
    cdna.v_lshlrev_b32_e32(cdna.v[201], a_byte * 8, cdna.v[200]),
    cdna.v_mov_b32_e32(cdna.v[202], a_neutral),
    cdna.v_or_b32_e32(cdna.v[20], cdna.v[201], cdna.v[202]),
    cdna.v_lshlrev_b32_e32(cdna.v[201], b_byte * 8, cdna.v[200]),
    cdna.v_mov_b32_e32(cdna.v[202], b_neutral),
    cdna.v_or_b32_e32(cdna.v[21], cdna.v[201], cdna.v[202]),
  ]

def _run_scaled(shape:str, opsel:int=3, opsel_hi:int=0) -> list[list[float]]:
  lanes, nouts, shift = 64, 4 if shape == '16' else 16, 4 if shape == '16' else 5
  ins = [cdna.v_mov_b32_e32(cdna.v[r], 3.5) for r in range(16)]
  # A is 1.0 in the first half of K and 0.5 in the second half; B is all 1.0 (E4M3 encodings, four per register).
  ins += [cdna.v_mov_b32_e32(cdna.v[r], 0x38383838) for r in range(32, 36)]
  ins += [cdna.v_mov_b32_e32(cdna.v[r], 0x30303030) for r in range(36, 40)]
  ins += [cdna.v_mov_b32_e32(cdna.v[r], 0x38383838) for r in range(40, 48)]
  a_byte = ((opsel_hi & 1) << 1) | (opsel & 1)
  b_byte = (((opsel_hi >> 1) & 1) << 1) | ((opsel >> 1) & 1)
  ins += _scale_insts(shift, a_byte, b_byte)
  args = dict(opsel=opsel, opsel_hi=opsel_hi, cbsz=0, blgp=0, scale_src0=cdna.v[20].offset, scale_src1=cdna.v[21].offset)
  if shape == '16': mfma = cdna.v_mfma_scale_f32_16x16x128_f8f6f4(cdna.v[0:3], cdna.v[32:39], cdna.v[40:47], cdna.v[0:3], **args)
  else: mfma = cdna.v_mfma_scale_f32_32x32x64_f8f6f4(cdna.v[0:15], cdna.v[32:39], cdna.v[40:47], cdna.v[0:15], **args)
  ins.append(mfma)
  code = [
    cdna.s_mov_b32(cdna.s[80], cdna.s[0]), cdna.s_mov_b32(cdna.s[81], cdna.s[1]), cdna.v_mov_b32_e32(cdna.v[255], cdna.v[0]),
    *ins, cdna.s_load_dwordx2(cdna.s[92:93], cdna.s[80:81], 0, soffset=NULL, imm=1), cdna.s_waitcnt(0),
    cdna.v_lshlrev_b32_e32(cdna.v[240], 2, cdna.v[255]),
    *[cdna.global_store_dword(addr=cdna.v[240], data=cdna.v[r], saddr=cdna.s[92:93], offset=r*lanes*4) for r in range(nouts)],
    cdna.s_endpgm(),
  ]
  out = (ctypes.c_uint32 * (lanes * nouts))()
  kern_args = (ctypes.c_uint64 * 1)(ctypes.addressof(out))
  code_bytes = assemble(code)
  kernel_buf = (ctypes.c_char * len(code_bytes)).from_buffer_copy(code_bytes)
  result = run_asm(ctypes.addressof(kernel_buf), len(code_bytes), 1, 1, 1, lanes, 1, 1, ctypes.addressof(kern_args),
                   0x19c | (128 << 15), 0x10000, arch='cdna')
  assert result == 0, f"run_asm failed with {result}"
  return [[_f32(out[r*lanes+lane]) for lane in range(lanes)] for r in range(nouts)]

def _expected(shape:str) -> list[list[float]]:
  lanes, nouts, shift = 64, 4 if shape == '16' else 16, 4 if shape == '16' else 5
  dim, chunks = (16, 4) if shape == '16' else (32, 2)
  ret = []
  for out_reg in range(nouts):
    vals = []
    for lane in range(lanes):
      row = (lane//16)*4 + out_reg if shape == '16' else (out_reg//4)*8 + (lane//32)*4 + out_reg%4
      col, acc = lane%dim, 3.5
      for chunk in range(chunks):
        dot = 32.0 if chunk < chunks//2 else 16.0
        exp = _genexp(chunk*dim+row, shift) - 127 + _genexp(chunk*dim+col, shift) - 127
        acc += dot * 2.0**exp
      vals.append(acc)
    ret.append(vals)
  return ret

class TestCDNAScaledMFMA(unittest.TestCase):
  def check_shape(self, shape:str):
    # The test values and every partial sum are exact binary fractions, so exact equality is intentional.
    out, expected = _run_scaled(shape), _expected(shape)
    self.assertEqual(out, expected)

  def test_mfma_scale_fp8_16x16x128(self): self.check_shape('16')
  def test_mfma_scale_fp8_32x32x64(self): self.check_shape('32')
  def test_mfma_scale_fp8_asymmetric_selectors(self):
    # OP_SEL/OP_SEL_HI=(0,1) selects byte 2 for A and byte 0 for B under the CDNA4 split-bit encoding.
    self.assertEqual(_run_scaled('16', opsel=0, opsel_hi=1), _expected('16'))

if __name__ == "__main__":
  unittest.main()
