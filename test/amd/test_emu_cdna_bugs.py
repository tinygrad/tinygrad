"""Regression tests for emulator bugs found during CDNA4 GEMM debugging.

Each test corresponds to a specific bug that was discovered and fixed.
Tests are designed to fail WITHOUT the fix and pass WITH the fix.

Run: python -m pytest test/amd/test_emu_cdna_bugs.py -v
"""
import ctypes, struct, unittest
from test.mockgpu.amd.emu import run_asm
from test.amd.hw.helpers import assemble, parse_output, _out_bytes, get_prologue_epilogue

# RDNA3 instruction imports (for LDS large size tests that are arch-independent)
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_mov_b32_e32 as r3_v_mov_b32_e32, v_lshlrev_b32_e32 as r3_v_lshlrev_b32_e32,
  v_or_b32_e32 as r3_v_or_b32_e32, ds_store_b32 as r3_ds_store_b32,
  ds_load_b32 as r3_ds_load_b32, s_waitcnt as r3_s_waitcnt,
)
from tinygrad.runtime.autogen.amd.rdna3.ins import v, s

# CDNA instruction imports (for CDNA-specific bug tests)
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from tinygrad.renderer.amd.dsl import NULL

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_rdna3_program(instructions: list, n_lanes: int = 1, lds_granules: int = 128):
  """Run RDNA3 instructions via emulator with configurable LDS size."""
  buf_sz = _out_bytes(n_lanes)
  out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
  out_addr = ctypes.addressof(out_buf)
  prologue, epilogue = get_prologue_epilogue(n_lanes)
  code = assemble(prologue + instructions + epilogue)
  args = (ctypes.c_uint64 * 1)(out_addr)
  args_ptr = ctypes.addressof(args)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  lib_ptr = ctypes.addressof(kernel_buf)
  rsrc2 = 0x19c | (lds_granules << 15)
  scratch_size = 0x10000
  result = run_asm(lib_ptr, len(code), 1, 1, 1, n_lanes, 1, 1, args_ptr, rsrc2, scratch_size)
  assert result == 0, f"run_asm failed with {result}"
  return parse_output(bytes(out_buf), n_lanes)

def run_cdna_program_raw(instructions: list, n_lanes: int = 64, lds_granules: int = 128) -> bytes:
  """Run CDNA instructions via emulator, return raw output bytes."""
  buf_sz = max(n_lanes * 4 * 16, 4096)
  out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
  out_addr = ctypes.addressof(out_buf)
  code = assemble(instructions)
  args = (ctypes.c_uint64 * 1)(out_addr)
  args_ptr = ctypes.addressof(args)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  lib_ptr = ctypes.addressof(kernel_buf)
  rsrc2 = 0x19c | (lds_granules << 15)
  result = run_asm(lib_ptr, len(code), 1, 1, 1, n_lanes, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
  assert result == 0, f"run_asm failed with {result}"
  return bytes(out_buf)

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 1: _Ctx.lds hardcoded to ptr(16384) - crashes for LDS > 64KB
# ═══════════════════════════════════════════════════════════════════════════════

class TestLDSLargeSize(unittest.TestCase):
  def test_lds_access_at_64kb_boundary(self):
    """Write/read LDS at byte address 0x10000 (64KB) = uint32 index 16384."""
    magic = 0xDEAD1234
    instructions = [
      r3_v_mov_b32_e32(v[1], magic & 0xFFFF),
      r3_v_mov_b32_e32(v[2], magic >> 16),
      r3_v_lshlrev_b32_e32(v[2], 16, v[2]),
      r3_v_or_b32_e32(v[1], v[1], v[2]),
      # Construct 0x10000 = 1 << 16
      r3_v_mov_b32_e32(v[3], 1),
      r3_v_lshlrev_b32_e32(v[3], 16, v[3]),
      r3_ds_store_b32(addr=v[3], data0=v[1]),
      r3_s_waitcnt(0),
      r3_ds_load_b32(vdst=v[4], addr=v[3]),
      r3_s_waitcnt(0),
    ]
    st = run_rdna3_program(instructions, n_lanes=1, lds_granules=256)
    self.assertEqual(st.vgpr[0][4], magic, f"got 0x{st.vgpr[0][4]:08x}, expected 0x{magic:08x}")

  def test_lds_access_at_96kb(self):
    """Write/read LDS at byte address 0x18000 (96KB) = uint32 index 24576."""
    magic = 0xCAFEBABE
    # 0x18000 = 0x18 << 12 = 24 << 12. Build as 0x18 << 12.
    instructions = [
      r3_v_mov_b32_e32(v[1], magic & 0xFFFF),
      r3_v_mov_b32_e32(v[2], magic >> 16),
      r3_v_lshlrev_b32_e32(v[2], 16, v[2]),
      r3_v_or_b32_e32(v[1], v[1], v[2]),
      r3_v_mov_b32_e32(v[3], 0x18),
      r3_v_lshlrev_b32_e32(v[3], 12, v[3]),  # v[3] = 0x18000
      r3_ds_store_b32(addr=v[3], data0=v[1]),
      r3_s_waitcnt(0),
      r3_ds_load_b32(vdst=v[5], addr=v[3]),
      r3_s_waitcnt(0),
    ]
    st = run_rdna3_program(instructions, n_lanes=1, lds_granules=256)
    self.assertEqual(st.vgpr[0][5], magic, f"got 0x{st.vgpr[0][5]:08x}, expected 0x{magic:08x}")

  def test_lds_access_at_gemm_size(self):
    """Access LDS at offset matching GEMM kernel lds_size (133120 bytes / ~130KB)."""
    magic = 0x12345678
    addr_val = 133112  # 133120 - 8, safely within 260-granule LDS
    # Build addr_val: 133112 = 0x20808. Build as (0x208 << 4) | 0x8... easier: use two halves.
    # 133112 = 0x00020808. lo=0x0808, hi=0x0002.
    lo, hi = addr_val & 0xFFFF, addr_val >> 16
    instructions = [
      r3_v_mov_b32_e32(v[1], magic & 0xFFFF),
      r3_v_mov_b32_e32(v[2], magic >> 16),
      r3_v_lshlrev_b32_e32(v[2], 16, v[2]),
      r3_v_or_b32_e32(v[1], v[1], v[2]),
      r3_v_mov_b32_e32(v[3], lo),
      r3_v_mov_b32_e32(v[4], hi),
      r3_v_lshlrev_b32_e32(v[4], 16, v[4]),
      r3_v_or_b32_e32(v[3], v[3], v[4]),  # v[3] = addr_val
      r3_ds_store_b32(addr=v[3], data0=v[1]),
      r3_s_waitcnt(0),
      r3_ds_load_b32(vdst=v[5], addr=v[3]),
      r3_s_waitcnt(0),
    ]
    st = run_rdna3_program(instructions, n_lanes=1, lds_granules=260)
    self.assertEqual(st.vgpr[0][5], magic, f"got 0x{st.vgpr[0][5]:08x}, expected 0x{magic:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 2: CDNA DS_WRITE_B128 / DS_READ_B128 pcode uses DATA[127:96] on 32-bit DATA
# Fix: direct handlers that bypass pcode and use separate VGPR reads.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_DS_B128(unittest.TestCase):
  def test_ds_write_read_b128(self):
    """Write 4 dwords to LDS via ds_write_b128, read back via ds_read_b128."""
    magic = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      cdna.v_mov_b32_e32(v[1], 0),
      cdna.ds_write_b128(addr=v[1], data0=v[4:7]),
      cdna.s_waitcnt(0),
      cdna.ds_read_b128(vdst=v[8:11], addr=v[1]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_ds_write_read_b128_with_offset(self):
    """DS_WRITE_B128 / DS_READ_B128 with non-zero LDS offset."""
    magic = [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      cdna.v_mov_b32_e32(v[1], 256),  # LDS byte offset 256
      cdna.ds_write_b128(addr=v[1], data0=v[4:7]),
      cdna.s_waitcnt(0),
      cdna.ds_read_b128(vdst=v[8:11], addr=v[1]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_ds_write_b128_large_lds(self):
    """DS_WRITE_B128 to LDS address > 64KB — combines ptr(16384) and direct handler bugs."""
    magic = [0xFEED0001, 0xFEED0002, 0xFEED0003, 0xFEED0004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      cdna.v_mov_b32_e32(v[1], 1),
      cdna.v_lshlrev_b32_e32(v[1], 16, v[1]),  # v[1] = 0x10000
      cdna.ds_write_b128(addr=v[1], data0=v[4:7]),
      cdna.s_waitcnt(0),
      cdna.ds_read_b128(vdst=v[8:11], addr=v[1]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=256)
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 3: DS_WRITE_B96 / DS_READ_B96 — same pcode UB issue as B128 but 3 dwords
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_DS_B96(unittest.TestCase):
  def test_ds_write_read_b96(self):
    """Write 3 dwords to LDS via ds_write_b96, read back via ds_read_b96."""
    magic = [0x11110001, 0x22220002, 0x33330003]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[1], 0),
      cdna.ds_write_b96(addr=v[1], data0=v[4:6]),
      cdna.s_waitcnt(0),
      cdna.ds_read_b96(vdst=v[8:10], addr=v[1]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    results = struct.unpack_from('<3I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 4: MUBUF OOB accesses should be silently dropped (not crash)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMUBUF_OOB(unittest.TestCase):
  def test_mubuf_store_oob_no_crash(self):
    """BUFFER_STORE_DWORD with num_records=0 should be silently dropped."""
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.s_mov_b32(s[4], 0),   # SRD base_lo
      cdna.s_mov_b32(s[5], 0),   # SRD base_hi | stride
      cdna.s_mov_b32(s[6], 0),   # SRD num_records = 0 (all OOB)
      cdna.s_mov_b32(s[7], 0),   # SRD flags
      cdna.v_mov_b32_e32(v[1], 0),
      cdna.v_mov_b32_e32(v[2], 0xDEADBEEF),
      cdna.buffer_store_dword(vdata=v[2], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.v_mov_b32_e32(v[3], 0x0000CAFE),
      cdna.global_store_dword(addr=v[0], data=v[3], saddr=s[10:11], offset=0),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    result = struct.unpack_from('<I', out, 0)[0]
    self.assertEqual(result, 0x0000CAFE, f"got 0x{result:08x}, expected 0x0000CAFE")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 5: ds_write_b128 crashes after barrier with multi-wave workgroup
# This reproduces the exact GEMM crash pattern:
# - 4 waves (256 threads, lx=256, wave_size=64)
# - barrier synchronization between waves
# - ds_write_b128 with per-lane LDS address computed from thread ID + wave offset
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_DS_B128_MultiWave(unittest.TestCase):
  def test_ds_write_b128_after_barrier_4_waves(self):
    """ds_write_b128 after s_barrier with 4 waves — reproduces GEMM crash pattern.

    Each wave writes 4 dwords per lane to a different LDS region.
    Wave offset is in s[2] (workgroup_id_x is placed after user SGPRs).
    """
    # The kernel:
    #   1. Load output address from args (s[0:1])
    #   2. Compute per-lane LDS addr: (tid_in_wave & 63) * 16 + wave_offset
    #   3. Write magic values to LDS via ds_write_b128
    #   4. Barrier
    #   5. Read back from LDS via ds_read_b128
    #   6. Thread 0 stores result to output
    #
    # With 4 waves: total_threads = 256, lx = 256
    # Wave 0: lanes 0-63, v[0] packed = 0-63
    # Wave 1: lanes 64-127, v[0] packed = 64-127
    # Wave 2: lanes 128-191, v[0] packed = 128-191
    # Wave 3: lanes 192-255, v[0] packed = 192-255
    magic = [0xAA000001, 0xBB000002, 0xCC000003, 0xDD000004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Set up data in v[4:7]
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      # Compute per-lane LDS address:
      #   v[0] has packed tid (for lx=256, ly=1, lz=1: v[0] = tid)
      #   addr = (v[0] & 63) * 16   (each lane writes 16 bytes = 4 dwords)
      cdna.v_and_b32_e32(v[1], 63, v[0]),      # v[1] = tid_in_wave = v[0] & 63
      cdna.v_lshlrev_b32_e32(v[1], 4, v[1]),   # v[1] = tid_in_wave * 16
      # Write to LDS — each wave writes to lane*16 (overlapping regions, that's ok for this test)
      cdna.ds_write_b128(addr=v[1], data0=v[4:7]),
      cdna.s_waitcnt(0),
      # Barrier to synchronize all 4 waves
      cdna.s_barrier(),
      # After barrier: read back from LDS (wave 0 reads)
      cdna.ds_read_b128(vdst=v[8:11], addr=v[1]),
      cdna.s_waitcnt(0),
      # Thread 0 stores result to output
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    # 4 waves: lx=256, wave_size=64 → 4 waves
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    # rsrc2: USER_SGPR_COUNT=2, ENABLE_SGPR_WORKGROUP_ID_X/Y/Z=1, LDS_SIZE=128 (64KB)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 256, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    assert result == 0, f"run_asm failed with {result}"
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_ds_write_b128_after_barrier_separate_lds_regions(self):
    """Each wave writes to a DIFFERENT LDS region (closer to GEMM pattern).

    Wave offset = wave_id * 64 * 16 = wave_id * 1024
    Each wave: lane addr = wave_offset + (tid_in_wave & 63) * 16
    """
    magic = [0x11110001, 0x22220002, 0x33330003, 0x44440004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      # Compute wave_offset: v[0] has packed tid.
      # For lx=256: tid = v[0] & 0xFF. wave_id = tid >> 6. wave_offset = wave_id * 1024.
      cdna.v_lshrrev_b32_e32(v[2], 6, v[0]),   # v[2] = wave_id = tid >> 6
      cdna.v_lshlrev_b32_e32(v[2], 10, v[2]),  # v[2] = wave_offset = wave_id * 1024
      # Per-lane addr: (tid & 63) * 16 + wave_offset
      cdna.v_and_b32_e32(v[1], 63, v[0]),
      cdna.v_lshlrev_b32_e32(v[1], 4, v[1]),
      cdna.v_add_u32_e32(v[1], v[2], v[1]),     # v[1] = final LDS addr
      # Write to LDS
      cdna.ds_write_b128(addr=v[1], data0=v[4:7]),
      cdna.s_waitcnt(0),
      cdna.s_barrier(),
      # Read back and verify (wave 0, lane 0)
      cdna.v_mov_b32_e32(v[1], 0),  # lane 0 reads from addr 0
      cdna.ds_read_b128(vdst=v[8:11], addr=v[1]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    # 4 waves * 64 lanes * 16 bytes = 4096 bytes of LDS needed. Use 16 granules (8KB).
    rsrc2 = 0x19c | (16 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 256, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    assert result == 0, f"run_asm failed with {result}"
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 6: ds_write_b128 crash with high VGPR indices (GEMM exact pattern)
# The GEMM kernel uses v[82] as addr, v[18:21] as data, v[180] as packed tid.
# This test reproduces the exact register layout that crashes in the GEMM.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_DS_B128_GEMMPattern(unittest.TestCase):
  def test_ds_write_b128_high_vgpr_gemm_regs(self):
    """ds_write_b128 with GEMM register layout: addr=v[82], data=v[18:21], tid in v[180].

    Reproduces the exact instruction sequence that crashes in the GEMM kernel:
      v_mov_b32_e32(v[180], v[0])         # copy packed tid
      v_and_b32_e32(v[82], 63, v[180])    # v[82] = tid_in_wave
      v_lshlrev_b32_e32(v[82], 4, v[82])  # v[82] *= 16
      s_mov_b32(s[53], 0)                  # wave offset = 0
      v_add_u32_e32(v[82], s[53], v[82])  # v[82] += wave_offset
      ds_write_b128(addr=v[82], data0=v[18:21])
    """
    magic = [0xAA000001, 0xBB000002, 0xCC000003, 0xDD000004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Copy packed tid to v[180] (GEMM pattern: high VGPR for tid)
      cdna.v_mov_b32_e32(v[180], v[0]),
      # Set data in v[18:21] (GEMM data registers)
      cdna.v_mov_b32_e32(v[18], magic[0]),
      cdna.v_mov_b32_e32(v[19], magic[1]),
      cdna.v_mov_b32_e32(v[20], magic[2]),
      cdna.v_mov_b32_e32(v[21], magic[3]),
      # Set wave offset in s[53]
      cdna.s_mov_b32(s[53], 0),
      # Compute LDS address exactly like GEMM: v[82] = (v[180] & 63) * 16 + s[53]
      cdna.v_and_b32_e32(v[82], 63, v[180]),
      cdna.v_lshlrev_b32_e32(v[82], 4, v[82]),
      cdna.v_add_u32_e32(v[82], s[53], v[82]),
      # THE CRASH INSTRUCTION: ds_write_b128 with addr=v[82], data0=v[18:21]
      cdna.ds_write_b128(addr=v[82], data0=v[18:21]),
      cdna.s_waitcnt(0),
      # Read back from LDS
      cdna.ds_read_b128(vdst=v[8:11], addr=v[82]),
      cdna.s_waitcnt(0),
      # Thread 0 stores result
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=260)
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_ds_write_b128_high_vgpr_4_waves_barrier(self):
    """GEMM exact crash: 4 waves, barrier, ds_write_b128 with v[82]/v[180]/v[18:21].

    This is the CLOSEST reproduction of the GEMM crash:
    - 4 waves (lx=256, wave_size=64)
    - s_barrier before ds_write_b128
    - v[180] = packed tid, v[82] = addr, v[18:21] = data
    - Large LDS (260 granules = 133120 bytes, GEMM size)
    """
    magic = [0xAA000001, 0xBB000002, 0xCC000003, 0xDD000004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Copy packed tid to v[180]
      cdna.v_mov_b32_e32(v[180], v[0]),
      # Set data in v[18:21]
      cdna.v_mov_b32_e32(v[18], magic[0]),
      cdna.v_mov_b32_e32(v[19], magic[1]),
      cdna.v_mov_b32_e32(v[20], magic[2]),
      cdna.v_mov_b32_e32(v[21], magic[3]),
      # Set wave offset s[53] = 0
      cdna.s_mov_b32(s[53], 0),
      # Compute v[82] = (v[180] & 63) * 16 + s[53]
      cdna.v_and_b32_e32(v[82], 63, v[180]),
      cdna.v_lshlrev_b32_e32(v[82], 4, v[82]),
      cdna.v_add_u32_e32(v[82], s[53], v[82]),
      # Barrier (all 4 waves sync here)
      cdna.s_barrier(),
      # THE CRASH INSTRUCTION
      cdna.ds_write_b128(addr=v[82], data0=v[18:21]),
      cdna.s_waitcnt(0),
      # Read back
      cdna.ds_read_b128(vdst=v[8:11], addr=v[82]),
      cdna.s_waitcnt(0),
      # Thread 0 stores result
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    # GEMM-size LDS: 260 granules = 133120 bytes
    rsrc2 = 0x19c | (260 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 256, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    assert result == 0, f"run_asm failed with {result}"
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_ds_write_b128_high_vgpr_4_waves_wave_offset(self):
    """4 waves with per-wave LDS offset — closest to real GEMM computation.

    Each wave computes wave_offset = (tid >> 6) * 1024 (each wave gets 1KB LDS region).
    addr = (v[180] & 63) * 16 + wave_offset, so each wave writes to a different LDS region.
    """
    magic = [0x11110001, 0x22220002, 0x33330003, 0x44440004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Copy packed tid to v[180]
      cdna.v_mov_b32_e32(v[180], v[0]),
      # Set data in v[18:21]
      cdna.v_mov_b32_e32(v[18], magic[0]),
      cdna.v_mov_b32_e32(v[19], magic[1]),
      cdna.v_mov_b32_e32(v[20], magic[2]),
      cdna.v_mov_b32_e32(v[21], magic[3]),
      # Compute wave_offset in v[83]: wave_id = tid >> 6, wave_offset = wave_id * 1024
      cdna.v_lshrrev_b32_e32(v[83], 6, v[180]),   # v[83] = wave_id = tid >> 6
      cdna.v_lshlrev_b32_e32(v[83], 10, v[83]),   # v[83] = wave_id * 1024
      # Compute v[82] = (v[180] & 63) * 16 + wave_offset
      cdna.v_and_b32_e32(v[82], 63, v[180]),
      cdna.v_lshlrev_b32_e32(v[82], 4, v[82]),
      cdna.v_add_u32_e32(v[82], v[83], v[82]),
      # Barrier
      cdna.s_barrier(),
      # ds_write_b128
      cdna.ds_write_b128(addr=v[82], data0=v[18:21]),
      cdna.s_waitcnt(0),
      # Read back (wave 0, lane 0 reads from addr 0)
      cdna.v_mov_b32_e32(v[82], 0),
      cdna.ds_read_b128(vdst=v[8:11], addr=v[82]),
      cdna.s_waitcnt(0),
      # Store result
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (260 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 256, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    assert result == 0, f"run_asm failed with {result}"
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 7: s_addc_u32 carry chain — 64-bit addition via s_add_u32 + s_addc_u32
# The GEMM kernel uses s_add_u32/s_addc_u32 to advance SRD base address.
# If carry (SCC) is wrong, s[13] is wrong → buffer_store writes to invalid address.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_SAddCarry(unittest.TestCase):
  def test_s_add_u32_no_carry(self):
    """s_add_u32 + s_addc_u32 with no overflow (SCC=0)."""
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.s_mov_b32(s[4], 100),       # lo = 100
      cdna.s_mov_b32(s[5], 0),          # hi = 0
      cdna.s_mov_b32(s[6], 50),         # add 50
      cdna.s_add_u32(s[4], s[4], s[6]), # s[4] = 150, SCC=0
      cdna.s_addc_u32(s[5], s[5], 0),   # s[5] = 0 + SCC = 0
      # Store s[4] and s[5] to output
      cdna.v_mov_b32_e32(v[1], s[4]),
      cdna.v_mov_b32_e32(v[2], s[5]),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[1], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[2], saddr=s[10:11], offset=4),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64)
    lo, hi = struct.unpack_from('<II', out, 0)
    self.assertEqual(lo, 150, f"lo: got {lo}, expected 150")
    self.assertEqual(hi, 0, f"hi: got {hi}, expected 0 (no carry)")

  def test_s_add_u32_with_carry(self):
    """s_add_u32 + s_addc_u32 with overflow (SCC=1): 0xFFFFFF00 + 0x200 = 0x100000100."""
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.s_mov_b32(s[4], 0xFFFFFF00),  # lo near max
      cdna.s_mov_b32(s[5], 0),            # hi = 0
      cdna.s_mov_b32(s[6], 0x200),        # add 0x200
      cdna.s_add_u32(s[4], s[4], s[6]),   # s[4] = 0x100 (overflow!), SCC=1
      cdna.s_addc_u32(s[5], s[5], 0),     # s[5] = 0 + SCC = 1
      cdna.v_mov_b32_e32(v[1], s[4]),
      cdna.v_mov_b32_e32(v[2], s[5]),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[1], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[2], saddr=s[10:11], offset=4),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64)
    lo, hi = struct.unpack_from('<II', out, 0)
    self.assertEqual(lo, 0x100, f"lo: got 0x{lo:x}, expected 0x100")
    self.assertEqual(hi, 1, f"hi: got {hi}, expected 1 (carry from overflow)")

  def test_s_add_u32_repeated_carry_chain(self):
    """Multiple s_add_u32/s_addc_u32 — like GEMM advancing SRD base through loop iterations."""
    # Start at 0x00000000_FFFF0000, add 0x20000 four times
    # After 4 adds: 0x00000000_FFFF0000 + 4*0x20000 = 0x00000001_00070000
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      cdna.s_mov_b32(s[4], 0xFFFF0000),  # lo
      cdna.s_mov_b32(s[5], 0),            # hi
      cdna.s_mov_b32(s[6], 0x20000),      # stride
      # Iteration 1
      cdna.s_add_u32(s[4], s[4], s[6]),
      cdna.s_addc_u32(s[5], s[5], 0),
      # Iteration 2
      cdna.s_add_u32(s[4], s[4], s[6]),
      cdna.s_addc_u32(s[5], s[5], 0),
      # Iteration 3
      cdna.s_add_u32(s[4], s[4], s[6]),
      cdna.s_addc_u32(s[5], s[5], 0),
      # Iteration 4
      cdna.s_add_u32(s[4], s[4], s[6]),
      cdna.s_addc_u32(s[5], s[5], 0),
      cdna.v_mov_b32_e32(v[1], s[4]),
      cdna.v_mov_b32_e32(v[2], s[5]),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[1], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[2], saddr=s[10:11], offset=4),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64)
    lo, hi = struct.unpack_from('<II', out, 0)
    expected = 0xFFFF0000 + 4 * 0x20000  # = 0x100070000
    expected_lo = expected & 0xFFFFFFFF   # = 0x00070000
    expected_hi = expected >> 32          # = 1
    self.assertEqual(lo, expected_lo, f"lo: got 0x{lo:x}, expected 0x{expected_lo:x}")
    self.assertEqual(hi, expected_hi, f"hi: got {hi}, expected {expected_hi}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 8: buffer_store_dwordx4 with SRD base address advancing (GEMM output path)
# The GEMM writes output rows via buffer_store_dwordx4 with s[12:15] SRD.
# s[12:13] base address is advanced each iteration via s_add_u32/s_addc_u32.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_BufferStoreDwordx4(unittest.TestCase):
  def test_buffer_store_dwordx4_basic(self):
    """buffer_store_dwordx4 stores 4 dwords to global memory via SRD."""
    magic = [0xAA000001, 0xBB000002, 0xCC000003, 0xDD000004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Set up SRD in s[4:7] pointing to same output buffer
      cdna.s_mov_b32(s[4], s[10]),   # base_lo = output buf lo
      cdna.s_mov_b32(s[5], s[11]),   # base_hi = output buf hi
      cdna.s_mov_b32(s[6], 1024),    # num_records = 1024 bytes
      cdna.s_mov_b32(s[7], 0),       # flags
      # Set data
      cdna.v_mov_b32_e32(v[4], magic[0]),
      cdna.v_mov_b32_e32(v[5], magic[1]),
      cdna.v_mov_b32_e32(v[6], magic[2]),
      cdna.v_mov_b32_e32(v[7], magic[3]),
      # vaddr = lane * 16 (each lane writes 16 bytes)
      cdna.v_lshlrev_b32_e32(v[1], 4, v[0]),
      # buffer_store_dwordx4 with offen=1
      cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    # Lane 0 writes to offset 0
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_buffer_store_dwordx4_advancing_srd(self):
    """buffer_store_dwordx4 with SRD base advancing via s_add_u32/s_addc_u32 — GEMM pattern.

    Simulates the GEMM output loop: each iteration advances s[12:13] by stride,
    then writes via buffer_store_dwordx4. Verifies data at multiple offsets.
    """
    magic1 = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    magic2 = [0x55555555, 0x66666666, 0x77777777, 0x88888888]
    stride = 64  # advance 64 bytes per iteration
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # SRD in s[12:15]
      cdna.s_mov_b32(s[12], s[10]),
      cdna.s_mov_b32(s[13], s[11]),
      cdna.s_mov_b32(s[14], 4096),   # num_records large enough
      cdna.s_mov_b32(s[15], 0),
      cdna.s_mov_b32(s[36], stride), # stride for advancing
      # First store: magic1 at offset 0
      cdna.v_mov_b32_e32(v[4], magic1[0]),
      cdna.v_mov_b32_e32(v[5], magic1[1]),
      cdna.v_mov_b32_e32(v[6], magic1[2]),
      cdna.v_mov_b32_e32(v[7], magic1[3]),
      cdna.v_mov_b32_e32(v[1], 0),   # vaddr = 0
      cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[12:15], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      # Advance SRD base: s[12:13] += s[36] (same pattern as GEMM)
      cdna.s_add_u32(s[12], s[12], s[36]),
      cdna.s_addc_u32(s[13], s[13], 0),
      # Second store: magic2 at new base (offset 64 from original)
      cdna.v_mov_b32_e32(v[4], magic2[0]),
      cdna.v_mov_b32_e32(v[5], magic2[1]),
      cdna.v_mov_b32_e32(v[6], magic2[2]),
      cdna.v_mov_b32_e32(v[7], magic2[3]),
      cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[12:15], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    # Check first store at offset 0
    r1 = struct.unpack_from('<4I', out, 0)
    for i, (got, exp) in enumerate(zip(r1, magic1)):
      self.assertEqual(got, exp, f"store1 dword[{i}]: got 0x{got:08x}, expected 0x{exp:08x}")
    # Check second store at offset=stride
    r2 = struct.unpack_from('<4I', out, stride)
    for i, (got, exp) in enumerate(zip(r2, magic2)):
      self.assertEqual(got, exp, f"store2 dword[{i}]: got 0x{got:08x}, expected 0x{exp:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 9: buffer_load_dwordx4 — GEMM input loading path
# The GEMM reads input matrix tiles via buffer_load_dwordx4 with SRD pointing
# to the input tensor. If the load address is wrong or OOB handling is broken,
# garbage data flows into VGPRs → MFMA → output, corrupting the result.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_BufferLoadDwordx4(unittest.TestCase):
  def _make_input_buffer(self, data: list[int]) -> tuple:
    """Create a ctypes buffer filled with data, return (buffer, address)."""
    buf = (ctypes.c_uint32 * len(data))(*data)
    return buf, ctypes.addressof(buf)

  def test_buffer_load_dwordx4_basic(self):
    """buffer_load_dwordx4 loads 4 dwords from global memory via SRD into VGPRs.

    Lane 0 reads from input buffer offset 0, verifies all 4 dwords arrive in VGPRs.
    """
    input_data = [0xAABBCCDD, 0x11223344, 0x55667788, 0x99AABBCC]
    in_buf, in_addr = self._make_input_buffer(input_data)
    # Output buffer
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Set up SRD s[4:7] pointing to input buffer
      cdna.s_mov_b32(s[4], in_addr & 0xFFFFFFFF),
      cdna.s_mov_b32(s[5], (in_addr >> 32) & 0xFFFF),
      cdna.s_mov_b32(s[6], len(input_data) * 4),  # num_records in bytes
      cdna.s_mov_b32(s[7], 0),
      # Load 4 dwords from input buffer at vaddr=0
      cdna.v_mov_b32_e32(v[1], 0),
      cdna.buffer_load_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      # Store loaded values to output
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[4], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[5], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[6], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[7], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 64, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    self.assertEqual(result, 0, f"run_asm failed with {result}")
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, expected) in enumerate(zip(results, input_data)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

  def test_buffer_load_dwordx4_per_lane_offsets(self):
    """buffer_load_dwordx4 with per-lane vaddr offsets (offen=1).

    Each lane loads from a different 16-byte aligned offset.
    Lane i reads 4 dwords at offset i*16.
    Verifies lane 0 and lane 1 read correct data.
    """
    # 64 lanes * 4 dwords = 256 dwords of input data
    input_data = [0x10000 + i for i in range(256)]
    in_buf, in_addr = self._make_input_buffer(input_data)
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # SRD pointing to input buffer
      cdna.s_mov_b32(s[4], in_addr & 0xFFFFFFFF),
      cdna.s_mov_b32(s[5], (in_addr >> 32) & 0xFFFF),
      cdna.s_mov_b32(s[6], len(input_data) * 4),
      cdna.s_mov_b32(s[7], 0),
      # vaddr = lane_id * 16 (each lane reads 16 bytes = 4 dwords)
      cdna.v_lshlrev_b32_e32(v[1], 4, v[0]),
      cdna.buffer_load_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      # Store loaded values to output (all 64 lanes store, each at lane*16)
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[1], data=v[4], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[1], data=v[5], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[1], data=v[6], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[1], data=v[7], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 64, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    self.assertEqual(result, 0, f"run_asm failed with {result}")
    # Verify lane 0 (offset 0): should be input_data[0:4]
    r0 = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, exp) in enumerate(zip(r0, input_data[0:4])):
      self.assertEqual(got, exp, f"lane0 dword[{i}]: got 0x{got:08x}, expected 0x{exp:08x}")
    # Verify lane 1 (offset 16): should be input_data[4:8]
    r1 = struct.unpack_from('<4I', bytes(out_buf), 16)
    for i, (got, exp) in enumerate(zip(r1, input_data[4:8])):
      self.assertEqual(got, exp, f"lane1 dword[{i}]: got 0x{got:08x}, expected 0x{exp:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 10: buffer_load + buffer_store loop with SRD advancing (GEMM data flow)
# The GEMM inner loop: load tile from input A via buffer_load_dwordx4,
# then store result to output C via buffer_store_dwordx4, advancing the SRD
# base each iteration. This reproduces the exact GEMM data movement that
# crashes with non-deterministic segfaults.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_BufferLoadStoreLoop(unittest.TestCase):
  def _make_input_buffer(self, data: list[int]) -> tuple:
    buf = (ctypes.c_uint32 * len(data))(*data)
    return buf, ctypes.addressof(buf)

  def test_buffer_load_store_copy(self):
    """Load 4 dwords from input buffer, store to output buffer — single iteration copy.

    Verifies the full load→store pipeline works: data flows correctly from
    input SRD through VGPRs to output SRD.
    """
    input_data = [0xDEAD0001, 0xDEAD0002, 0xDEAD0003, 0xDEAD0004]
    in_buf, in_addr = self._make_input_buffer(input_data)
    buf_sz = 4096
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Input SRD s[4:7]
      cdna.s_mov_b32(s[4], in_addr & 0xFFFFFFFF),
      cdna.s_mov_b32(s[5], (in_addr >> 32) & 0xFFFF),
      cdna.s_mov_b32(s[6], len(input_data) * 4),
      cdna.s_mov_b32(s[7], 0),
      # Output SRD s[12:15]
      cdna.s_mov_b32(s[12], s[10]),
      cdna.s_mov_b32(s[13], s[11]),
      cdna.s_mov_b32(s[14], buf_sz),
      cdna.s_mov_b32(s[15], 0),
      # Load from input
      cdna.v_mov_b32_e32(v[1], 0),
      cdna.buffer_load_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      # Store to output
      cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[12:15], soffset=NULL, offen=1, offset=0),
      cdna.s_waitcnt(0),
      cdna.s_endpgm(),
    ]
    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 64, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    self.assertEqual(result, 0, f"run_asm failed with {result}")
    results = struct.unpack_from('<4I', bytes(out_buf), 0)
    for i, (got, exp) in enumerate(zip(results, input_data)):
      self.assertEqual(got, exp, f"dword[{i}]: got 0x{got:08x}, expected 0x{exp:08x}")

  def test_buffer_load_store_advancing_srd_loop(self):
    """Load from input, store to output, advancing both SRDs — GEMM loop pattern.

    Simulates 4 iterations of the GEMM output loop:
      - Each iteration loads 4 dwords from input[iter*16..iter*16+15]
      - Stores to output[iter*stride..iter*stride+15]
      - Advances output SRD base by stride
      - Advances input SRD base by 16 bytes

    This is the EXACT data movement pattern that crashes in the GEMM.
    """
    n_iters = 4
    stride = 64  # output stride (simulates advancing through output rows)
    # Input: 4 dwords per iteration, 4 iterations = 16 dwords
    input_data = [0x10000000 + i * 0x1000 + j for i in range(n_iters) for j in range(4)]
    in_buf, in_addr = self._make_input_buffer(input_data)
    buf_sz = stride * n_iters + 16  # enough for all stores
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)

    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Input SRD s[4:7] — advances by 16 bytes per iteration
      cdna.s_mov_b32(s[4], in_addr & 0xFFFFFFFF),
      cdna.s_mov_b32(s[5], (in_addr >> 32) & 0xFFFF),
      cdna.s_mov_b32(s[6], len(input_data) * 4),
      cdna.s_mov_b32(s[7], 0),
      # Output SRD s[12:15] — advances by stride per iteration
      cdna.s_mov_b32(s[12], s[10]),
      cdna.s_mov_b32(s[13], s[11]),
      cdna.s_mov_b32(s[14], buf_sz),
      cdna.s_mov_b32(s[15], 0),
      # Constants
      cdna.s_mov_b32(s[36], stride),  # output stride
      cdna.s_mov_b32(s[37], 16),      # input stride (4 dwords = 16 bytes)
      cdna.v_mov_b32_e32(v[1], 0),    # vaddr always 0 (offset managed by SRD base)
    ]

    # Unroll 4 iterations of load + store + advance
    for i in range(n_iters):
      instructions += [
        # Load 4 dwords from input
        cdna.buffer_load_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
        cdna.s_waitcnt(0),
        # Store 4 dwords to output
        cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[12:15], soffset=NULL, offen=1, offset=0),
        cdna.s_waitcnt(0),
        # Advance input SRD: s[4:5] += 16
        cdna.s_add_u32(s[4], s[4], s[37]),
        cdna.s_addc_u32(s[5], s[5], 0),
        # Advance output SRD: s[12:13] += stride
        cdna.s_add_u32(s[12], s[12], s[36]),
        cdna.s_addc_u32(s[13], s[13], 0),
      ]
    instructions.append(cdna.s_endpgm())

    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 64, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    self.assertEqual(result, 0, f"run_asm failed with {result}")

    # Verify each iteration's output at the correct offset
    for i in range(n_iters):
      offset = i * stride
      r = struct.unpack_from('<4I', bytes(out_buf), offset)
      expected = input_data[i*4:(i+1)*4]
      for j, (got, exp) in enumerate(zip(r, expected)):
        self.assertEqual(got, exp, f"iter{i} dword[{j}] at offset {offset}: got 0x{got:08x}, expected 0x{exp:08x}")

  def test_buffer_load_store_64_lanes_advancing_srd(self):
    """All 64 lanes load and store with per-lane offsets, advancing SRD each iteration.

    This is closest to the actual GEMM: each lane loads from input at
    lane*16 offset, stores to output at lane*16 offset, then SRD advances.
    Multiple iterations with all 64 lanes active.
    """
    n_iters = 4
    row_stride = 64 * 16  # 64 lanes * 16 bytes each = 1024 bytes per row
    # Input: enough for 4 iterations of 64 lanes * 4 dwords
    total_input_dwords = n_iters * 64 * 4
    input_data = [(i + 1) for i in range(total_input_dwords)]
    in_buf, in_addr = self._make_input_buffer(input_data)
    buf_sz = n_iters * row_stride + 16
    out_buf = (ctypes.c_uint8 * buf_sz)(*([0] * buf_sz))
    out_addr = ctypes.addressof(out_buf)

    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Input SRD s[4:7]
      cdna.s_mov_b32(s[4], in_addr & 0xFFFFFFFF),
      cdna.s_mov_b32(s[5], (in_addr >> 32) & 0xFFFF),
      cdna.s_mov_b32(s[6], total_input_dwords * 4),
      cdna.s_mov_b32(s[7], 0),
      # Output SRD s[12:15]
      cdna.s_mov_b32(s[12], s[10]),
      cdna.s_mov_b32(s[13], s[11]),
      cdna.s_mov_b32(s[14], buf_sz),
      cdna.s_mov_b32(s[15], 0),
      # Per-lane offset: vaddr = lane_id * 16
      cdna.v_lshlrev_b32_e32(v[1], 4, v[0]),
      # Strides
      cdna.s_mov_b32(s[36], row_stride),  # advance both by row_stride each iteration
    ]

    for i in range(n_iters):
      instructions += [
        cdna.buffer_load_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[4:7], soffset=NULL, offen=1, offset=0),
        cdna.s_waitcnt(0),
        cdna.buffer_store_dwordx4(vdata=v[4:7], vaddr=v[1], srsrc=s[12:15], soffset=NULL, offen=1, offset=0),
        cdna.s_waitcnt(0),
        # Advance both SRDs by row_stride
        cdna.s_add_u32(s[4], s[4], s[36]),
        cdna.s_addc_u32(s[5], s[5], 0),
        cdna.s_add_u32(s[12], s[12], s[36]),
        cdna.s_addc_u32(s[13], s[13], 0),
      ]
    instructions.append(cdna.s_endpgm())

    code = assemble(instructions)
    args = (ctypes.c_uint64 * 1)(out_addr)
    args_ptr = ctypes.addressof(args)
    kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
    lib_ptr = ctypes.addressof(kernel_buf)
    rsrc2 = 0x19c | (128 << 15)
    result = run_asm(lib_ptr, len(code), 1, 1, 1, 64, 1, 1, args_ptr, rsrc2, 0, arch="cdna")
    self.assertEqual(result, 0, f"run_asm failed with {result}")

    # Verify: each iteration, lane j gets input_data[iter*256 + j*4 .. +3]
    for i in range(n_iters):
      for lane in [0, 1, 31, 63]:  # spot check a few lanes
        out_offset = i * row_stride + lane * 16
        r = struct.unpack_from('<4I', bytes(out_buf), out_offset)
        in_base = i * 256 + lane * 4
        expected = input_data[in_base:in_base+4]
        for j, (got, exp) in enumerate(zip(r, expected)):
          self.assertEqual(got, exp,
            f"iter{i} lane{lane} dword[{j}] at offset {out_offset}: got 0x{got:08x}, expected 0x{exp:08x}")

# ═══════════════════════════════════════════════════════════════════════════════
# Bug 11: DS_WRITE_B128 crashes when inactive lanes have garbage in address VGPR
# ROOT CAUSE OF GEMM SEGFAULT: The CPU backend evaluates both branches of
# where() — including lds_idx.load() for inactive lanes. If an inactive lane's
# address VGPR contains garbage (e.g., MFMA output data), the LDS index is
# out-of-bounds, causing a segfault.
# Fix: clamp inactive lane addresses to 0 (same as DS_READ_B128 already does).
# ═══════════════════════════════════════════════════════════════════════════════

class TestCDNA_DS_Write_InactiveLaneGarbage(unittest.TestCase):
  def test_ds_write_b128_garbage_addr_inactive_lanes(self):
    """ds_write_b128 must not crash when inactive lanes have garbage in address VGPR.

    This reproduces the GEMM crash root cause:
    1. Set v[82] = garbage (0xDEADBEEF) for ALL 64 lanes
    2. Use s_and_saveexec_b64 to restrict EXEC to lanes 0-31
    3. Set valid LDS addresses for active lanes (0-31 only)
    4. Call ds_write_b128 — lanes 32-63 have garbage in v[82] but are inactive

    Without the fix (missing inactive lane address clamping), this segfaults
    because the CPU backend evaluates lds_idx.load() even for inactive lanes,
    and the garbage address (0xDEADBEEF) causes an out-of-bounds LDS access.
    """
    magic = [0x11110001, 0x22220002, 0x33330003, 0x44440004]
    instructions = [
      cdna.s_load_dwordx2(sdata=s[10:11], sbase=s[0:1], offset=0),
      cdna.s_waitcnt(0),
      # Set data in v[18:21]
      cdna.v_mov_b32_e32(v[18], magic[0]),
      cdna.v_mov_b32_e32(v[19], magic[1]),
      cdna.v_mov_b32_e32(v[20], magic[2]),
      cdna.v_mov_b32_e32(v[21], magic[3]),
      # Step 1: Set v[82] = 0xDEADBEEF for ALL 64 lanes (garbage address, simulates MFMA overwrite)
      cdna.v_mov_b32_e32(v[82], 0xBEEF),
      cdna.v_mov_b32_e32(v[83], 0xDEAD),
      cdna.v_lshlrev_b32_e32(v[83], 16, v[83]),
      cdna.v_or_b32_e32(v[82], v[82], v[83]),   # v[82] = 0xDEADBEEF for all 64 lanes
      # Step 2: Restrict EXEC to lanes 0-31 only (saves old EXEC to s[60:61])
      cdna.s_mov_b32(s[20], 0xFFFFFFFF),  # lo 32 bits = all 1s
      cdna.s_mov_b32(s[21], 0),            # hi 32 bits = all 0s -> mask = 0x00000000_FFFFFFFF
      cdna.s_and_saveexec_b64(s[60:61], s[20:21]),  # EXEC = EXEC & mask = lanes 0-31; save old to s[60:61]
      # Step 3: Set valid LDS addresses for active lanes (0-31 only)
      cdna.v_and_b32_e32(v[82], 63, v[0]),       # v[82] = lane_id & 63 (only for active lanes 0-31)
      cdna.v_lshlrev_b32_e32(v[82], 4, v[82]),   # v[82] = lane_id * 16 (only for active lanes 0-31)
      # NOW: lanes 0-31 have valid addr in v[82], lanes 32-63 still have 0xDEADBEEF
      # Step 4: ds_write_b128 — EXEC = lanes 0-31, MUST NOT CRASH on inactive lanes 32-63
      cdna.ds_write_b128(addr=v[82], data0=v[18:21]),
      cdna.s_waitcnt(0),
      # Step 5: Restore full EXEC via s_or_saveexec_b64 (EXEC = s[60:61] | EXEC = full mask)
      cdna.s_or_saveexec_b64(s[62:63], s[60:61]),
      # Step 6: Read back lane 0's data from LDS offset 0 and write to output
      cdna.v_mov_b32_e32(v[82], 0),
      cdna.ds_read_b128(vdst=v[8:11], addr=v[82]),
      cdna.s_waitcnt(0),
      cdna.v_mov_b32_e32(v[0], 0),
      cdna.global_store_dword(addr=v[0], data=v[8], saddr=s[10:11], offset=0),
      cdna.global_store_dword(addr=v[0], data=v[9], saddr=s[10:11], offset=4),
      cdna.global_store_dword(addr=v[0], data=v[10], saddr=s[10:11], offset=8),
      cdna.global_store_dword(addr=v[0], data=v[11], saddr=s[10:11], offset=12),
      cdna.s_endpgm(),
    ]
    out = run_cdna_program_raw(instructions, n_lanes=64, lds_granules=128)
    results = struct.unpack_from('<4I', out, 0)
    for i, (got, expected) in enumerate(zip(results, magic)):
      self.assertEqual(got, expected, f"dword[{i}]: got 0x{got:08x}, expected 0x{expected:08x}")

if __name__ == '__main__':
  unittest.main()
