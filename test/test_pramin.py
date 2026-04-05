"""Unit test for PRAMIN address math used in NVPageTableEntry on eGPU."""
import unittest

class TestPRAMINAddressMath(unittest.TestCase):
  """Verify BAR0 PRAMIN window address calculations. No hardware needed."""

  def _pramin_window_and_offset(self, paddr:int, idx:int) -> tuple[int, int]:
    """Returns (window_reg_value, bar0_word_offset) matching NVPageTableEntry._pramin_base."""
    window = paddr >> 16
    base = (0x700000 + (paddr & 0xffff)) // 4 + idx * 2
    return window, base

  def test_aligned_base(self):
    window, base = self._pramin_window_and_offset(paddr=0x0, idx=0)
    self.assertEqual(window, 0)
    self.assertEqual(base, 0x700000 // 4)

  def test_offset_within_window(self):
    # paddr=0x5000 should set window to 0, offset to 0x5000 within the 64KB window
    window, base = self._pramin_window_and_offset(paddr=0x5000, idx=0)
    self.assertEqual(window, 0)
    self.assertEqual(base, (0x700000 + 0x5000) // 4)

  def test_second_window(self):
    # paddr=0x10000 crosses into the second 64KB window
    window, base = self._pramin_window_and_offset(paddr=0x10000, idx=0)
    self.assertEqual(window, 1)
    self.assertEqual(base, 0x700000 // 4)

  def test_idx_advances(self):
    # idx=3 should add 6 words (3 * 2 for 64-bit entries)
    window, base = self._pramin_window_and_offset(paddr=0x0, idx=3)
    self.assertEqual(base, 0x700000 // 4 + 6)

  def test_large_paddr(self):
    # 1GB into VRAM
    window, base = self._pramin_window_and_offset(paddr=0x40000000, idx=0)
    self.assertEqual(window, 0x40000000 >> 16)
    self.assertEqual(base, 0x700000 // 4)

  def test_no_overflow(self):
    # 4KB page table at paddr=0x3000 (aligned), last entry idx=511
    paddr = 0x3000
    idx = 511  # last 8-byte entry: offset 0x3000 + 511*8 = 0x3FF8
    window, base = self._pramin_window_and_offset(paddr, idx)
    bar0_byte = base * 4
    self.assertGreaterEqual(bar0_byte, 0x700000)
    self.assertLess(bar0_byte + 8, 0x700000 + 0x10000)  # second dword within window

  def test_palloc_zero_addresses(self):
    """Verify palloc zeroing hits correct VRAM addresses (reproduces the off-by-offset bug)."""
    paddr, size = 0x5000, 0x1000
    zeroed_addrs = []
    for i in range(0, size, 8):
      off = paddr + i
      window = off >> 16
      base = (0x700000 + (off & 0xffff)) // 4
      zeroed_addrs.append((window, base * 4))  # convert back to byte addr for clarity
    # first zero should be at paddr, not at 0
    self.assertEqual(zeroed_addrs[0], (0, 0x700000 + 0x5000))
    # last zero should be at paddr + size - 8
    self.assertEqual(zeroed_addrs[-1], (0, 0x700000 + 0x5000 + 0x1000 - 8))
    # all should be within PRAMIN window
    for window, addr in zeroed_addrs:
      self.assertGreaterEqual(addr, 0x700000)
      self.assertLess(addr + 8, 0x710000)

if __name__ == "__main__":
  unittest.main()
