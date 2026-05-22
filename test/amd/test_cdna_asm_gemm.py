import ctypes, math, unittest
import numpy as np
from tinygrad import dtypes
from extra.gemm.cdna_asm_gemm import NUM_WG, build_kernel
from test.mockgpu.amd.emu import run_asm

TILE_M = TILE_N = 256
TILE_K = 64
LDS_BYTES = 133_120

def _run_cdna_asm_gemm(batch: int):
  rng = np.random.default_rng(batch)
  a = (rng.random((batch, TILE_M, TILE_K), dtype=np.float32) - 0.5).astype(np.float16)
  b = (rng.random((TILE_K, TILE_N), dtype=np.float32) - 0.5).astype(np.float16)

  out = (ctypes.c_uint16 * (batch * TILE_M * TILE_N))()
  a_buf = (ctypes.c_uint16 * (batch * TILE_M * TILE_K)).from_buffer_copy(a.view(np.uint16).tobytes())
  b_buf = (ctypes.c_uint16 * (TILE_K * TILE_N)).from_buffer_copy(b.view(np.uint16).tobytes())
  args = (ctypes.c_uint64 * 3)(ctypes.addressof(out), ctypes.addressof(a_buf), ctypes.addressof(b_buf))

  code = b''.join(inst.to_bytes() for inst in build_kernel(batch, TILE_M, TILE_N, TILE_K, dtypes.float16))
  kernel = (ctypes.c_char * len(code)).from_buffer_copy(code)

  # Match tinygrad/renderer/amd/elf.py: two user SGPRs for the kernargs pointer, then workgroup_id_x in s2.
  rsrc2 = (2 << 1) | (1 << 7) | (math.ceil(LDS_BYTES / 512) << 15)
  run_asm(ctypes.addressof(kernel), len(code), NUM_WG, 1, 1, 256, 1, 1, ctypes.addressof(args),
          rsrc2=rsrc2, scratch_size=0x10000, arch="cdna")

  got = np.frombuffer(out, dtype=np.uint16).view(np.float16).reshape(batch, TILE_M, TILE_N)
  expected = np.stack([(a[i].astype(np.float32) @ b.astype(np.float32)).astype(np.float16) for i in range(batch)])
  return got, expected

class TestCDNAAsmGEMM(unittest.TestCase):
  def test_single_batch_matches_numpy(self):
    got, expected = _run_cdna_asm_gemm(batch=1)
    np.testing.assert_allclose(got, expected, atol=1e-2, rtol=1e-3)

  def test_two_batches_match_numpy(self):
    got, expected = _run_cdna_asm_gemm(batch=2)
    np.testing.assert_allclose(got, expected, atol=1e-2, rtol=1e-3)

if __name__ == "__main__":
  unittest.main()
