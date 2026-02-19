"""Tests for MFMA (Matrix Fused Multiply-Add) instructions on CDNA."""
import unittest
import numpy as np
from test.amd.hw.helpers import run_program, i2f
from tinygrad.runtime.autogen.amd.cdna.ins import v_mov_b32_e32, v_mfma_f32_16x16x16_f16, v_mfma_f32_4x4x4_16b_f16, v_mfma_f32_32x32x8_f16, \
  v_accvgpr_write, v_accvgpr_read, v, s, v_add_u32_e32, v_cvt_f32_u32_e32, v_cvt_f16_f32_e32, v_pack_b32_f16
from tinygrad.renderer.amd.dsl import acc
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.realize import get_runner
from tinygrad.engine.schedule import ExecItem
from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import ST_16X32, RT_16X32, RT_16X16, TileLayout

def assert_allclose(cmp:Tensor, ref:Tensor, **kwargs) -> None:
  if Device.DEFAULT == "NULL": Tensor.realize(cmp, ref)
  else: np.testing.assert_allclose(cmp.numpy(), ref.numpy(), **kwargs)

class TestMFMA(unittest.TestCase):
  def test_mfma_f32_16x16x16_f16_all_ones(self):
    """16x16 MFMA: A(16x16) * B(16x16) + C(16x16), all f16 1.0 -> every output = 16.0"""
    instructions = [
      v_mov_b32_e32(v[0], 0x3c003c00),  # packed f16 [1.0, 1.0]
      v_mov_b32_e32(v[1], 0x3c003c00),
      v_mov_b32_e32(v[2], 0x3c003c00),
      v_mov_b32_e32(v[3], 0x3c003c00),
      v_accvgpr_write(acc[0], 0),
      v_accvgpr_write(acc[1], 0),
      v_accvgpr_write(acc[2], 0),
      v_accvgpr_write(acc[3], 0),
      v_mfma_f32_16x16x16_f16(acc[0:3], v[0:1], v[2:3], acc[0:3]),
      v_accvgpr_read(v[0], acc[0]),
      v_accvgpr_read(v[1], acc[1]),
      v_accvgpr_read(v[2], acc[2]),
      v_accvgpr_read(v[3], acc[3]),
    ]
    st = run_program(instructions, n_lanes=64, arch='cdna')
    for lane in range(64):
      for reg in range(4):
        self.assertAlmostEqual(i2f(st.vgpr[lane][reg]), 16.0, places=1, msg=f"v[{reg}] lane {lane}")

  def test_mfma_f32_4x4x4_f16_all_ones(self):
    """4x4 MFMA (16 blocks): all f16 1.0 -> each dot product output = 4.0 (lanes 0-15, ACCVGPR 0)"""
    instructions = [
      v_mov_b32_e32(v[0], 0x3c003c00),  # packed f16 [1.0, 1.0]
      v_mov_b32_e32(v[1], 0x3c003c00),
      v_mov_b32_e32(v[2], 0x3c003c00),
      v_mov_b32_e32(v[3], 0x3c003c00),
      v_accvgpr_write(acc[0], 0),
      v_accvgpr_write(acc[1], 0),
      v_accvgpr_write(acc[2], 0),
      v_accvgpr_write(acc[3], 0),
      v_mfma_f32_4x4x4_16b_f16(acc[0:3], v[0:1], v[2:3], acc[0:3]),
      v_accvgpr_read(v[0], acc[0]),
    ]
    st = run_program(instructions, n_lanes=64, arch='cdna')
    for lane in range(16):  # only lanes 0-15 participate in 4x4 MFMA
      self.assertAlmostEqual(i2f(st.vgpr[lane][0]), 4.0, places=1, msg=f"v[0] lane {lane}")

  def test_mfma_f32_32x32x8_f16_all_ones(self):
    """32x32 MFMA: A(32x8) * B(8x32) + C(32x32), all f16 1.0 -> every output = 8.0"""
    instructions = [
      v_mov_b32_e32(v[0], 0x3c003c00),  # packed f16 [1.0, 1.0]
      v_mov_b32_e32(v[1], 0x3c003c00),
      v_mov_b32_e32(v[2], 0x3c003c00),
      v_mov_b32_e32(v[3], 0x3c003c00),
    ]
    for r in range(16):
      instructions.append(v_accvgpr_write(acc[r], 0))
    instructions.append(v_mfma_f32_32x32x8_f16(acc[0:15], v[0:1], v[2:3], acc[0:15]))
    for r in range(16):
      instructions.append(v_accvgpr_read(v[r], acc[r]))
    st = run_program(instructions, n_lanes=64, arch='cdna')
    for lane in range(64):
      for reg in range(16):
        self.assertAlmostEqual(i2f(st.vgpr[lane][reg]), 8.0, places=1, msg=f"v[{reg}] lane {lane}")

  def test_mfma_f32_4x4x4_f16_per_lane(self):
    """4x4 MFMA with per-lane unique f16 inputs to exercise lane-to-matrix mapping."""
    instructions = [
      # A input: v[0] = f16(lane + 1), lower 16 bits only
      v_add_u32_e32(v[10], 1, v[255]),
      v_cvt_f32_u32_e32(v[10], v[10]),
      v_cvt_f16_f32_e32(v[0], v[10]),
      # B input: v[2] = f16(lane + 17)
      v_add_u32_e32(v[10], 17, v[255]),
      v_cvt_f32_u32_e32(v[10], v[10]),
      v_cvt_f16_f32_e32(v[2], v[10]),
      # Zero accumulator
      v_accvgpr_write(acc[0], 0),
      # Run MFMA
      v_mfma_f32_4x4x4_16b_f16(acc[0:3], v[0:1], v[2:3], acc[0:3]),
      v_accvgpr_read(v[0], acc[0]),
    ]
    st = run_program(instructions, n_lanes=64, arch='cdna')
    # Build reference: M=4, K=4, groups=4, k_per_grp=1
    M, K = 4, 4
    A = np.zeros((M, K), dtype=np.float32)
    B = np.zeros((M, K), dtype=np.float32)
    for lane in range(16):
      row, grp = lane % M, lane // M
      A[row, grp] = float(np.float16(lane + 1))
      B[row, grp] = float(np.float16(lane + 17))
    C = A @ B.T
    for lane in range(16):
      expected = C[lane // M, lane % M]
      self.assertAlmostEqual(i2f(st.vgpr[lane][0]), expected, places=1, msg=f"lane {lane}")

  def test_mfma_f32_32x32x8_f16_per_lane(self):
    """32x32 MFMA with per-lane unique f16 inputs to exercise lane-to-matrix mapping."""
    # Build packed f16 pairs for A (v[0:1]) and B (v[2:3])
    # v[r] = pack(f16(lane+off_lo), f16(lane+off_hi))
    # A offsets: (1,65), (129,193)  B offsets: (257,321), (385,449)
    a_offsets = [(1, 65), (129, 193)]
    b_offsets = [(257, 321), (385, 449)]
    instructions = []
    for reg, (lo, hi) in zip([0, 1], a_offsets):
      instructions += [
        v_add_u32_e32(v[10], lo, v[255]),
        v_cvt_f32_u32_e32(v[10], v[10]),
        v_cvt_f16_f32_e32(v[10], v[10]),
        v_add_u32_e32(v[11], hi, v[255]),
        v_cvt_f32_u32_e32(v[11], v[11]),
        v_cvt_f16_f32_e32(v[11], v[11]),
        v_pack_b32_f16(v[reg], v[10], v[11]),
      ]
    for reg, (lo, hi) in zip([2, 3], b_offsets):
      instructions += [
        v_add_u32_e32(v[10], lo, v[255]),
        v_cvt_f32_u32_e32(v[10], v[10]),
        v_cvt_f16_f32_e32(v[10], v[10]),
        v_add_u32_e32(v[11], hi, v[255]),
        v_cvt_f32_u32_e32(v[11], v[11]),
        v_cvt_f16_f32_e32(v[11], v[11]),
        v_pack_b32_f16(v[reg], v[10], v[11]),
      ]
    for r in range(16):
      instructions.append(v_accvgpr_write(acc[r], 0))
    instructions.append(v_mfma_f32_32x32x8_f16(acc[0:15], v[0:1], v[2:3], acc[0:15]))
    for r in range(16):
      instructions.append(v_accvgpr_read(v[r], acc[r]))
    st = run_program(instructions, n_lanes=64, arch='cdna')
    # Build reference: M=32, K=8, groups=2, k_per_grp=4
    M, K = 32, 8
    A = np.zeros((M, K), dtype=np.float32)
    B = np.zeros((M, K), dtype=np.float32)
    a_flat = [1, 65, 129, 193]  # offsets for kl=0,1,2,3
    b_flat = [257, 321, 385, 449]
    for lane in range(64):
      row, grp = lane % M, lane // M
      for kl in range(4):
        A[row, grp * 4 + kl] = float(np.float16(lane + a_flat[kl]))
        B[row, grp * 4 + kl] = float(np.float16(lane + b_flat[kl]))
    C = A @ B.T  # (32, 32)
    for lane in range(64):
      n_idx, c_grp = lane % M, lane // M
      for reg in range(16):
        expected = C[c_grp * 16 + reg, n_idx]
        self.assertAlmostEqual(i2f(st.vgpr[lane][reg]), expected, places=0, msg=f"v[{reg}] lane {lane}")

  def test_mfma_f32_4x4x4_f16_accumulator(self):
    """4x4 MFMA with per-lane unique inputs AND non-zero initial accumulator."""
    instructions = [
      # A input: v[0] = f16(lane + 1)
      v_add_u32_e32(v[10], 1, v[255]),
      v_cvt_f32_u32_e32(v[10], v[10]),
      v_cvt_f16_f32_e32(v[0], v[10]),
      # B input: v[2] = f16(lane + 17)
      v_add_u32_e32(v[10], 17, v[255]),
      v_cvt_f32_u32_e32(v[10], v[10]),
      v_cvt_f16_f32_e32(v[2], v[10]),
      # Non-zero accumulator: acc[0] = f32(lane + 100)
      v_add_u32_e32(v[10], 100, v[255]),
      v_cvt_f32_u32_e32(v[10], v[10]),
      v_accvgpr_write(acc[0], v[10]),
      # Run MFMA
      v_mfma_f32_4x4x4_16b_f16(acc[0:3], v[0:1], v[2:3], acc[0:3]),
      v_accvgpr_read(v[0], acc[0]),
    ]
    st = run_program(instructions, n_lanes=64, arch='cdna')
    # Build reference (same as per_lane test, but with accumulator)
    M, K = 4, 4
    A = np.zeros((M, K), dtype=np.float32)
    B = np.zeros((M, K), dtype=np.float32)
    for lane in range(16):
      row, grp = lane % M, lane // M
      A[row, grp] = float(np.float16(lane + 1))
      B[row, grp] = float(np.float16(lane + 17))
    C = A @ B.T
    for lane in range(16):
      expected = C[lane // M, lane % M] + float(lane + 100)
      self.assertAlmostEqual(i2f(st.vgpr[lane][0]), expected, places=1, msg=f"lane {lane}")

  def test_mfma_f32_32x32x8_f16_accumulator(self):
    """32x32 MFMA with per-lane unique inputs AND non-zero initial accumulator."""
    a_offsets = [(1, 65), (129, 193)]
    b_offsets = [(257, 321), (385, 449)]
    instructions = []
    for reg, (lo, hi) in zip([0, 1], a_offsets):
      instructions += [
        v_add_u32_e32(v[10], lo, v[255]),
        v_cvt_f32_u32_e32(v[10], v[10]),
        v_cvt_f16_f32_e32(v[10], v[10]),
        v_add_u32_e32(v[11], hi, v[255]),
        v_cvt_f32_u32_e32(v[11], v[11]),
        v_cvt_f16_f32_e32(v[11], v[11]),
        v_pack_b32_f16(v[reg], v[10], v[11]),
      ]
    for reg, (lo, hi) in zip([2, 3], b_offsets):
      instructions += [
        v_add_u32_e32(v[10], lo, v[255]),
        v_cvt_f32_u32_e32(v[10], v[10]),
        v_cvt_f16_f32_e32(v[10], v[10]),
        v_add_u32_e32(v[11], hi, v[255]),
        v_cvt_f32_u32_e32(v[11], v[11]),
        v_cvt_f16_f32_e32(v[11], v[11]),
        v_pack_b32_f16(v[reg], v[10], v[11]),
      ]
    # Non-zero accumulator: acc[r] = f32(lane * 16 + r + 1) for each output register
    for r in range(16):
      instructions += [
        v_mov_b32_e32(v[10], r + 1),
        v_add_u32_e32(v[10], v[10], v[255]),
        v_cvt_f32_u32_e32(v[10], v[10]),
        v_accvgpr_write(acc[r], v[10]),
      ]
    instructions.append(v_mfma_f32_32x32x8_f16(acc[0:15], v[0:1], v[2:3], acc[0:15]))
    for r in range(16):
      instructions.append(v_accvgpr_read(v[r], acc[r]))
    st = run_program(instructions, n_lanes=64, arch='cdna')
    # Build reference
    M, K = 32, 8
    A = np.zeros((M, K), dtype=np.float32)
    B = np.zeros((M, K), dtype=np.float32)
    a_flat = [1, 65, 129, 193]
    b_flat = [257, 321, 385, 449]
    for lane in range(64):
      row, grp = lane % M, lane // M
      for kl in range(4):
        A[row, grp * 4 + kl] = float(np.float16(lane + a_flat[kl]))
        B[row, grp * 4 + kl] = float(np.float16(lane + b_flat[kl]))
    C = A @ B.T
    for lane in range(64):
      n_idx, c_grp = lane % M, lane // M
      for reg in range(16):
        acc_init = float(lane + reg + 1)
        expected = C[c_grp * 16 + reg, n_idx] + acc_init
        self.assertAlmostEqual(i2f(st.vgpr[lane][reg]), expected, places=0, msg=f"v[{reg}] lane {lane}")

class TestMFMAKernels(unittest.TestCase):
  def setUp(self):
    arch = getattr(Device[Device.DEFAULT].renderer, "arch", "")
    if not arch.startswith("gfx9"):
      self.skipTest(f"arch {arch} not supported")

  def test_gemm(self):
    N = 64
    BLOCK_SIZE = 64
    with Kernel("small_matmul", (N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      c_reg_col = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      c_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg_col = warp.zero(c_reg_col)
      for tile in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, tile, col), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg_col = warp.mma_AB(c_reg_col, a_reg, b_reg)
      c_reg_col = ker.endrange()

      c_smem = warp.store(c_smem, c_reg_col)
      c_reg = warp.load(c_reg, c_smem)

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(sink, [t.uop.buffer for t in (c, a, b)], prg=get_runner(Device.DEFAULT, sink))
    ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b, dtype=dtypes.float32).float()

    assert_allclose(c, ref)

  def test_gemm_transposed(self):
    N = 128
    BLOCK_N, BLOCK_M, BLOCK_K = 64, 64, 128
    with Kernel("small_matmul_transposed", (N // BLOCK_N, N // BLOCK_M, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)
      b_smem = ker.st((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)

      a_reg = ker.rt((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      b_reg = ker.rt((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      c_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL, base_shape=RT_16X16)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_K):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, col, tile), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg = warp.mma_ABt(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(sink, [t.uop.buffer for t in (c, a, b)], prg=get_runner(Device.DEFAULT, sink))
    ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b.transpose(2, 3), dtype=dtypes.float32).float()

    assert_allclose(c, ref)

  def test_flash_attention(self):
    from extra.thunder.tiny.fa import flash_attention

    B, N, H, H_KV, D = 1, 64, 2, 1, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    out = flash_attention(q, k, v, is_causal=False)
    out = out.float().transpose(1, 2)

    ref = q.scaled_dot_product_attention(k, v, is_causal=False, enable_gqa=True).float().transpose(1, 2)

    assert_allclose(out, ref, atol=2e-2, rtol=2e-2)

  def test_flash_attention_causal(self):
    from extra.thunder.tiny.fa import flash_attention

    B, N, H, H_KV, D = 1, 64, 2, 1, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    out = flash_attention(q, k, v, is_causal=True)
    out = out.float().transpose(1, 2)

    ref = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float().transpose(1, 2)

    assert_allclose(out, ref, atol=2e-2, rtol=2e-2)

if __name__ == "__main__":
  unittest.main()
