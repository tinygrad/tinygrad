import unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from test.amd.hw.test_cdna_vop3 import run_cdna_lanes

class TestCDNADSSwizzle(unittest.TestCase):
  def test_modes(self):
    # the source lane of lanes 0-7 for each mode of the ISA pseudocode
    for offset, src in {0x041f: [1, 0, 3, 2, 5, 4, 7, 6],                  # bitmask: xor 1
                        0x00a0: [5]*8,                                     # bitmask: or 5, broadcast
                        0x801b: [3, 2, 1, 0, 7, 6, 5, 4],                  # quad permute: reversed
                        0xc020: [1, 2, 3, 4, 5, 6, 7, 8],                  # rotate by 1
                        0xe000: [0, 16, 8, 24, 4, 20, 12, 28]}.items():    # fft: bit reversed
      with self.subTest(offset=hex(offset)):
        out = run_cdna_lanes([cdna.ds_swizzle_b32(vdst=cdna.v[2], addr=cdna.v[255], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0)])
        self.assertEqual(out[:8], src)

def _tr_elem(lane: int, elem: int, bits: int) -> int:
  # lane id in the low bits, plus the 16-lane group, so a gather from the wrong group cannot pass
  return ((lane & 15) + 3 * elem + (lane >> 4)) & ((1 << bits) - 1)

def _tr_expected(elem_bits: int, out_bits: int) -> list[list[int]]:
  """Column tx of the 16-wide tile. Row j lives in lanes j*(16//m) + c, m elements per lane."""
  m, n_out = out_bits // elem_bits, out_bits // 32
  per_row = 16 // m
  words = [[0] * 64 for _ in range(n_out)]
  for lane in range(64):
    group, tx = divmod(lane, 16)
    acc = 0
    for row in range(m):
      src = group * 16 + row * per_row + tx // m
      acc |= _tr_elem(src, tx % m, elem_bits) << (row * elem_bits)
    for w in range(n_out): words[w][lane] = (acc >> (32 * w)) & 0xffffffff
  return words

def _tr_program(elem_bits: int, out_bits: int, offset: int = 0, acc: int = 0, vdst: int = 2) -> list:
  m, n_dwords = out_bits // elem_bits, out_bits // 32
  code = [cdna.v_mov_b32_e32(cdna.v[4 + w], 0) for w in range(n_dwords)]
  if out_bits == 96:
    code += [cdna.v_lshlrev_b32_e32(cdna.v[0], 2, cdna.v[255]), cdna.v_lshlrev_b32_e32(cdna.v[1], 3, cdna.v[255]),
             cdna.v_add_u32_e32(cdna.v[0], cdna.v[1], cdna.v[0])]
  else: code.append(cdna.v_lshlrev_b32_e32(cdna.v[0], 3, cdna.v[255]))
  # v[8] = (lane & 15) + (lane >> 4), the part of the element value that does not depend on the element index
  code += [cdna.v_and_b32_e32(cdna.v[8], 15, cdna.v[255]), cdna.v_lshrrev_b32_e32(cdna.v[7], 4, cdna.v[255]),
           cdna.v_add_u32_e32(cdna.v[8], cdna.v[7], cdna.v[8])]
  mask = (1 << elem_bits) - 1
  for e in range(m):
    code.append(cdna.v_add_u32_e32(cdna.v[9], 3 * e, cdna.v[8]) if e else cdna.v_mov_b32_e32(cdna.v[9], cdna.v[8]))
    if mask <= 64: code.append(cdna.v_and_b32_e32(cdna.v[9], mask, cdna.v[9]))
    bit = e * elem_bits
    dword, shift = divmod(bit, 32)
    if shift == 0: code.append(cdna.v_or_b32_e32(cdna.v[4 + dword], cdna.v[9], cdna.v[4 + dword]))
    else: code.append(cdna.v_lshl_or_b32(cdna.v[4 + dword], cdna.v[9], shift, cdna.v[4 + dword]))
    if (spill := shift + elem_bits - 32) > 0:
      code += [cdna.v_lshrrev_b32_e32(cdna.v[7], 32 - shift, cdna.v[9]),
               cdna.v_or_b32_e32(cdna.v[5 + dword], cdna.v[7], cdna.v[5 + dword])]
      assert spill < 32
  data = cdna.v[4:4 + n_dwords - 1]
  write = cdna.ds_write_b96 if out_bits == 96 else cdna.ds_write_b64
  read = getattr(cdna, f"ds_read_b{out_bits}_tr_b{elem_bits}")
  code += [write(addr=cdna.v[0], data0=data, offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0),
           read(vdst=cdna.v[vdst:vdst + n_dwords - 1], addr=cdna.v[0], offset0=offset & 255, offset1=offset >> 8, acc=acc),
           cdna.s_waitcnt(0)]
  if acc: code += [cdna.v_accvgpr_read(cdna.v[vdst + i], cdna.v[vdst + i]) for i in range(n_dwords)]
  return code

class TestCDNADSReadTr(unittest.TestCase):
  """CDNA4 ISA 11.4 lane map: one instruction, one 16-lane transpose. Not compared to gfx950 silicon here."""
  def test_spec_lanes(self):
    # frozen lanes, including lane 16, so the Python spec cannot drift into "every group looks the same"
    b16, b4 = _tr_expected(16, 64), _tr_expected(4, 64)
    self.assertEqual(b16[0][0], 0x00040000)
    self.assertEqual(b16[1][0], 0x000C0008)
    self.assertEqual(b16[0][1], 0x00070003)
    self.assertEqual(b16[1][1], 0x000F000B)
    self.assertEqual(b16[0][16], 0x00050001)
    self.assertEqual(b4[0][0], 0x76543210)
    self.assertEqual(b4[0][16], 0x87654321)
    self.assertNotEqual(b4[0][0], b4[0][16])

  def test_transpose(self):
    for elem_bits, out_bits in ((16, 64), (8, 64), (4, 64), (6, 96)):
      with self.subTest(elem_bits=elem_bits):
        code = _tr_program(elem_bits, out_bits)
        got = [run_cdna_lanes(code, out_reg=2 + w) for w in range(out_bits // 32)]
        self.assertEqual(got, _tr_expected(elem_bits, out_bits))

  def test_b16_offset_acc_alias(self):
    expected = _tr_expected(16, 64)
    for kwargs in ({'offset': 0x108}, {'acc': 1}, {'vdst': 0}):
      with self.subTest(**kwargs):
        vdst = kwargs.get('vdst', 2)
        code = _tr_program(16, 64, **kwargs)
        self.assertEqual([run_cdna_lanes(code, out_reg=vdst + w) for w in range(2)], expected)

if __name__ == "__main__":
  unittest.main()
