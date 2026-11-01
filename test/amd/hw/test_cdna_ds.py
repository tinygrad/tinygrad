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

class TestCDNADSReadTr(unittest.TestCase):
  def _run_tr_b16(self, offset: int = 0, acc: int = 0, vdst: int = 2) -> list[list[int]]:
    # lane l writes the 16-bit values 4l..4l+3 to LDS[8l], then reads its own address back with the transpose
    code = [cdna.v_lshlrev_b32_e32(cdna.v[0], 3, cdna.v[255]),
            cdna.v_lshlrev_b32_e32(cdna.v[1], 2, cdna.v[255]),
            cdna.v_add_u32_e32(cdna.v[6], 1, cdna.v[1]), cdna.v_lshl_or_b32(cdna.v[4], cdna.v[6], 16, cdna.v[1]),
            cdna.v_add_u32_e32(cdna.v[6], 2, cdna.v[1]), cdna.v_add_u32_e32(cdna.v[7], 3, cdna.v[1]),
            cdna.v_lshl_or_b32(cdna.v[5], cdna.v[7], 16, cdna.v[6]),
            cdna.ds_write_b64(addr=cdna.v[0], data0=cdna.v[4:5], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0),
            cdna.ds_read_b64_tr_b16(vdst=cdna.v[vdst:vdst+1], addr=cdna.v[0], offset0=offset & 255, offset1=offset >> 8, acc=acc),
            cdna.s_waitcnt(0)]
    # with acc, the result lands in a[vdst:vdst+1]
    if acc: code += [cdna.v_accvgpr_read(cdna.v[vdst], cdna.v[vdst]), cdna.v_accvgpr_read(cdna.v[vdst+1], cdna.v[vdst+1])]
    return [run_cdna_lanes(code, out_reg=r) for r in (vdst, vdst+1)]

  def _expected(self) -> list[list[int]]:
    # each group of 16 lanes read a 4x16 tile of 16-bit values (lane 4r+c holds row r, columns 4c..4c+3), lane j gets column j
    elem = lambda l, r: 4 * ((l & ~15) + 4 * r + (l & 15) // 4) + (l & 15) % 4  # noqa: E731
    return [[elem(l, 2*d) | (elem(l, 2*d+1) << 16) for l in range(64)] for d in range(2)]

  def test_read_b64_tr_b16(self): self.assertEqual(self._run_tr_b16(), self._expected())
  def test_read_b64_tr_b16_offset(self): self.assertEqual(self._run_tr_b16(offset=0x108), self._expected())
  def test_read_b64_tr_b16_acc(self): self.assertEqual(self._run_tr_b16(acc=1), self._expected())
  def test_read_b64_tr_b16_vdst_is_addr(self): self.assertEqual(self._run_tr_b16(vdst=0), self._expected())

if __name__ == "__main__":
  unittest.main()
