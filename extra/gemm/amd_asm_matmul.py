# RDNA3 128x128 tiled GEMM kernel using Python DSL
# Generates LLVM-style assembly compiled by HIPCompiler
# Based on kernel8_batched_gmem.s algorithm

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from extra.assembly.amd.dsl import Inst, Inst32, Inst64, Inst96, s, v, NULL, VCC_LO
from extra.assembly.amd.autogen.rdna3.ins import (
  VOPD, VOPDOp, VOP3, VOP3Op,
  s_load_b128, s_load_b64, s_mov_b32, s_add_u32, s_addc_u32, s_lshl_b32, s_add_i32, s_cmp_ge_i32, s_cmp_lt_i32,
  s_waitcnt, s_barrier, s_endpgm, s_nop, s_sendmsg, s_cbranch_scc0, s_cbranch_scc1, s_setprio, s_branch,
  v_mov_b32_e32 as v_mov_b32, v_add_nc_u32_e32 as v_add_nc_u32,
  v_lshlrev_b32_e32 as v_lshlrev_b32, v_lshrrev_b32_e32 as v_lshrrev_b32,
  v_and_b32_e32 as v_and_b32, v_or_b32_e32 as v_or_b32,
  v_lshl_add_u32, v_lshl_or_b32, v_and_or_b32, v_bfe_u32, v_mad_u32_u24,
  v_mul_lo_u32, v_add_co_u32, v_add_co_ci_u32_e32 as v_add_co_ci_u32,
  v_ashrrev_i32_e32 as v_ashrrev_i32,
  ds_store_b32, ds_store_2addr_stride64_b32, ds_load_b64,
  global_load_b32, global_store_b128,
)

# ============================================================================
# Constants - matching kernel8
# ============================================================================
N = getenv("N", 4096)
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
THREADS = 128
LDS_SIZE = 8320  # bytes

# ============================================================================
# Kernel class - accumulates instructions and generates assembly
# ============================================================================
class Kernel:
  def __init__(self, arch: str = "gfx1100"):
    self.instructions: list[Inst] = []
    self.labels: dict[str, int] = {}
    self.arch = arch

  def emit(self, inst: Inst) -> Inst:
    self.instructions.append(inst)
    return inst

  def label(self, name: str):
    self.labels[name] = len(self.instructions)

  def inst_size(self, inst: Inst) -> int:
    """Return instruction size in bytes"""
    if isinstance(inst, Inst96): return 12
    if isinstance(inst, Inst64): return 8
    return 4  # Inst32

  def byte_offset(self, from_idx: int, to_idx: int) -> int:
    """Calculate byte offset from instruction from_idx to to_idx"""
    if from_idx <= to_idx:
      return sum(self.inst_size(self.instructions[i]) for i in range(from_idx, to_idx))
    else:
      return -sum(self.inst_size(self.instructions[i]) for i in range(to_idx, from_idx))

  def to_asm(self) -> str:
    lines = [
      "\t.text",
      f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
      "\t.protected\tkernel",
      "\t.globl\tkernel",
      "\t.p2align\t8",
      "\t.type\tkernel,@function",
      "kernel:",
    ]

    label_at = {pos: name for name, pos in self.labels.items()}
    for i, inst in enumerate(self.instructions):
      if i in label_at:
        lines.append(f".{label_at[i]}:")
      lines.append("\t" + inst.disasm())

    lines.extend([
      "\t.section\t.rodata,\"a\",@progbits",
      "\t.p2align\t6, 0x0",
      "\t.amdhsa_kernel kernel",
      f"\t\t.amdhsa_group_segment_fixed_size {LDS_SIZE}",
      "\t\t.amdhsa_private_segment_fixed_size 0",
      "\t\t.amdhsa_kernarg_size 24",
      "\t\t.amdhsa_user_sgpr_count 14",
      "\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1",
      "\t\t.amdhsa_wavefront_size32 1",
      "\t\t.amdhsa_system_sgpr_workgroup_id_x 1",
      "\t\t.amdhsa_system_sgpr_workgroup_id_y 1",
      "\t\t.amdhsa_next_free_vgpr 216",
      "\t\t.amdhsa_next_free_sgpr 56",
      "\t\t.amdhsa_float_denorm_mode_32 3",
      "\t\t.amdhsa_float_denorm_mode_16_64 3",
      "\t\t.amdhsa_dx10_clamp 1",
      "\t\t.amdhsa_ieee_mode 1",
      "\t\t.amdhsa_memory_ordered 1",
      "\t.end_amdhsa_kernel",
      "\t.text",
      ".Lfunc_end0:",
      "\t.size\tkernel, .Lfunc_end0-kernel",
      "\t.amdgpu_metadata",
      "---",
      "amdhsa.kernels:",
      "  - .args:",
      "      - .address_space: global",
      "        .offset: 0",
      "        .size: 8",
      "        .value_kind: global_buffer",
      "      - .address_space: global",
      "        .offset: 8",
      "        .size: 8",
      "        .value_kind: global_buffer",
      "      - .address_space: global",
      "        .offset: 16",
      "        .size: 8",
      "        .value_kind: global_buffer",
      f"    .group_segment_fixed_size: {LDS_SIZE}",
      "    .kernarg_segment_align: 8",
      "    .kernarg_segment_size: 24",
      "    .max_flat_workgroup_size: 128",
      "    .name: kernel",
      "    .private_segment_fixed_size: 0",
      "    .sgpr_count: 60",
      "    .symbol: kernel.kd",
      "    .vgpr_count: 216",
      "    .wavefront_size: 32",
      f"amdhsa.target: amdgcn-amd-amdhsa--{self.arch}",
      "amdhsa.version:",
      "  - 1",
      "  - 2",
      "...",
      "\t.end_amdgpu_metadata",
    ])
    return "\n".join(lines)

# ============================================================================
# FMA block - 64 dual FMAs matching kernel8 exactly
# ============================================================================
def emit_fma_block(k: Kernel):
  """Emit 64 dual FMAs for one k-step - matching kernel8 exactly"""
  fmas = [
    (5, 186, 184, 2, 187, 185), (3, 186, 185, 4, 187, 184),
    (9, 186, 188, 6, 187, 189), (7, 187, 188, 8, 186, 189),
    (13, 190, 188, 10, 191, 189), (11, 190, 189, 12, 191, 188),
    (17, 190, 184, 14, 191, 185), (15, 191, 184, 16, 190, 185),
    (21, 194, 184, 18, 195, 185), (19, 194, 185, 20, 195, 184),
    (25, 194, 188, 22, 195, 189), (23, 195, 188, 24, 194, 189),
    (29, 198, 188, 26, 199, 189), (27, 198, 189, 28, 199, 188),
    (33, 198, 192, 30, 199, 193), (31, 199, 192, 32, 198, 193),
    (37, 186, 192, 34, 187, 193), (35, 186, 193, 36, 187, 192),
    (41, 186, 196, 38, 187, 197), (39, 187, 196, 40, 186, 197),
    (45, 190, 196, 42, 191, 197), (43, 190, 197, 44, 191, 196),
    (49, 190, 192, 46, 191, 193), (47, 191, 192, 48, 190, 193),
    (53, 194, 192, 50, 195, 193), (51, 194, 193, 52, 195, 192),
    (57, 194, 196, 54, 195, 197), (55, 195, 196, 56, 194, 197),
    (61, 198, 196, 58, 199, 197), (59, 198, 197, 60, 199, 196),
    (65, 198, 200, 62, 199, 201), (63, 199, 200, 64, 198, 201),
    (69, 186, 200, 66, 187, 201), (67, 186, 201, 68, 187, 200),
    (73, 186, 204, 70, 187, 205), (71, 187, 204, 72, 186, 205),
    (77, 190, 204, 74, 191, 205), (75, 190, 205, 76, 191, 204),
    (81, 190, 200, 78, 191, 201), (79, 191, 200, 80, 190, 201),
    (85, 194, 200, 82, 195, 201), (83, 194, 201, 84, 195, 200),
    (89, 194, 204, 86, 195, 205), (87, 195, 204, 88, 194, 205),
    (93, 198, 204, 90, 199, 205), (91, 198, 205, 92, 199, 204),
    (97, 198, 208, 94, 199, 209), (95, 199, 208, 96, 198, 209),
    (101, 186, 208, 98, 187, 209), (99, 186, 209, 100, 187, 208),
    (105, 186, 212, 102, 187, 213), (103, 187, 212, 104, 186, 213),
    (109, 190, 212, 106, 191, 213), (107, 190, 213, 108, 191, 212),
    (113, 190, 208, 110, 191, 209), (111, 191, 208, 112, 190, 209),
    (117, 194, 208, 114, 195, 209), (115, 194, 209, 116, 195, 208),
    (121, 194, 212, 122, 195, 213), (123, 195, 212, 120, 194, 213),
    (129, 198, 212, 126, 199, 213), (127, 198, 213, 124, 199, 212),
    (133, 198, 184, 214, 199, 185), (131, 199, 184, 128, 198, 185),
  ]
  for vdx, ax0, bx1, vdy, ay0, by1 in fmas:
    k.emit(VOPD(opx=VOPDOp.V_DUAL_FMAC_F32, opy=VOPDOp.V_DUAL_FMAC_F32,
                vdstx=v[vdx], srcx0=v[ax0], vsrcx1=v[bx1],
                vdsty=v[vdy], srcy0=v[ay0], vsrcy1=v[by1]))

def emit_lds_loads(k: Kernel, a_addr: int, b_addr: int):
  """Emit LDS loads for A (4 b64) and B (8 b64) - matching kernel8"""
  # A on bank 2-3
  k.emit(ds_load_b64(vdst=v[186:187], addr=v[a_addr], offset0=0))
  k.emit(ds_load_b64(vdst=v[190:191], addr=v[a_addr], offset0=8))
  k.emit(ds_load_b64(vdst=v[194:195], addr=v[a_addr], offset0=64))
  k.emit(ds_load_b64(vdst=v[198:199], addr=v[a_addr], offset0=72))
  # B on bank 0-1
  k.emit(ds_load_b64(vdst=v[184:185], addr=v[b_addr], offset0=0))
  k.emit(ds_load_b64(vdst=v[188:189], addr=v[b_addr], offset0=8))
  k.emit(ds_load_b64(vdst=v[192:193], addr=v[b_addr], offset0=128))
  k.emit(ds_load_b64(vdst=v[196:197], addr=v[b_addr], offset0=136))
  k.emit(ds_load_b64(vdst=v[200:201], addr=v[b_addr], offset0=256))
  k.emit(ds_load_b64(vdst=v[204:205], addr=v[b_addr], offset0=264))
  k.emit(ds_load_b64(vdst=v[208:209], addr=v[b_addr], offset0=384))
  k.emit(ds_load_b64(vdst=v[212:213], addr=v[b_addr], offset0=392))

# ============================================================================
# Build the kernel
# ============================================================================
def build_kernel(arch: str = "gfx1100") -> str:
  k = Kernel(arch)

  # ========== PROLOGUE: Load kernarg pointers ==========
  # s[0:1] = kernarg ptr, s[14] = blockIdx.x, s[15] = blockIdx.y, v[0] = threadIdx.x
  k.emit(s_load_b128(sdata=s[20:23], sbase=s[0:1], soffset=NULL))  # A[20:21], B[22:23]
  k.emit(s_waitcnt(simm16=0))  # lgkmcnt(0)

  # B base addresses with row offsets (8 rows of 16KB each for N=4096)
  for i, off in enumerate([0x0, 0x4000, 0x8000, 0xc000, 0x10000, 0x14000, 0x18000, 0x1c000]):
    k.emit(s_add_u32(sdst=s[24 + i*2], ssrc0=s[22], ssrc1=off))
    k.emit(s_addc_u32(sdst=s[25 + i*2], ssrc0=s[23], ssrc1=0))

  # B thread offset: v203 = (blockIdx.x * 128 + threadIdx.x) * 4
  k.emit(s_lshl_b32(sdst=s[19], ssrc0=s[14], ssrc1=7))  # blockIdx.x * 128
  k.emit(v_add_nc_u32(vdst=v[203], src0=s[19], vsrc1=v[0]))
  k.emit(v_lshlrev_b32(vdst=v[203], src0=2, vsrc1=v[203]))

  # A base addresses with row offsets (8 rows of 256KB each for N=4096)
  for i, off in enumerate([0x0, 0x40000, 0x80000, 0xc0000, 0x100000, 0x140000, 0x180000, 0x1c0000]):
    k.emit(s_add_u32(sdst=s[40 + i*2], ssrc0=s[20], ssrc1=off))
    k.emit(s_addc_u32(sdst=s[41 + i*2], ssrc0=s[21], ssrc1=0))

  # A thread offset: v215 = (blockIdx.y * 128 * N + (tid/8) * N + tid%8) * 4 bytes
  # First compute in elements, then multiply by 4 at the end
  # blockIdx.y * 128 * N = blockIdx.y * 128 * 4096 = blockIdx.y << 19 (in elements)
  k.emit(s_lshl_b32(sdst=s[19], ssrc0=s[15], ssrc1=19))  # blockIdx.y * 128 * N elements
  k.emit(v_lshrrev_b32(vdst=v[1], src0=3, vsrc1=v[0]))   # tid / 8
  k.emit(v_lshlrev_b32(vdst=v[1], src0=12, vsrc1=v[1])) # * 4096
  k.emit(v_and_b32(vdst=v[215], src0=7, vsrc1=v[0]))    # tid % 8
  k.emit(v_add_nc_u32(vdst=v[215], src0=v[1], vsrc1=v[215]))
  k.emit(v_add_nc_u32(vdst=v[215], src0=s[19], vsrc1=v[215]))
  k.emit(v_lshlrev_b32(vdst=v[215], src0=2, vsrc1=v[215]))

  # ========== Setup constants and C pointer ==========
  k.emit(s_mov_b32(sdst=s[4], ssrc0=N))
  k.emit(s_mov_b32(sdst=s[5], ssrc0=0x3F800000))  # 1.0f
  k.emit(s_mov_b32(sdst=s[6], ssrc0=0))

  # C output address setup (simplified)
  k.emit(s_lshl_b32(sdst=s[2], ssrc0=s[14], ssrc1=7))  # blockIdx.x * 128
  k.emit(s_lshl_b32(sdst=s[3], ssrc0=s[15], ssrc1=7))  # blockIdx.y * 128
  k.emit(v_and_b32(vdst=v[118], src0=7, vsrc1=v[0]))   # tid % 8
  k.emit(v_bfe_u32(vdst=v[2], src0=v[0], src1=3, src2=2)) # (tid >> 3) & 3
  k.emit(v_lshlrev_b32(vdst=v[135], src0=2, vsrc1=v[118]))  # (tid%8) * 4
  k.emit(v_lshlrev_b32(vdst=v[136], src0=2, vsrc1=v[2]))    # ((tid>>3)&3) * 4

  # Load C pointer
  k.emit(s_load_b64(sdata=s[0:1], sbase=s[0:1], soffset=NULL, offset=0x10))

  # ========== INITIAL GLOBAL LOAD (before loop) ==========
  # First batch of global loads for B (8 values)
  for i in range(8):
    k.emit(global_load_b32(vdst=v[167+i], addr=v[203], saddr=s[24+i*2:25+i*2]))

  # First batch of global loads for A (8 values)
  for i in range(8):
    k.emit(global_load_b32(vdst=v[175+i], addr=v[215], saddr=s[40+i*2:41+i*2]))

  # Compute LDS store addresses for A (v141-v148)
  k.emit(v_mad_u32_u24(vdst=v[141], src0=0x210, src1=v[118], src2=0))
  for i in range(1, 8):
    k.emit(v_add_nc_u32(vdst=v[141+i], src0=0x210, vsrc1=v[140+i]))

  # Compute LDS store address for B (v156 - preserved across acc zeroing)
  k.emit(v_lshrrev_b32(vdst=v[4], src0=3, vsrc1=v[0]))  # tid / 8
  k.emit(v_lshlrev_b32(vdst=v[156], src0=2, vsrc1=v[4])) # * 4
  k.emit(v_add_nc_u32(vdst=v[156], src0=0x80, vsrc1=v[156])) # + 0x80

  # Wait for global loads and store to LDS
  k.emit(s_waitcnt(simm16=0))

  # Store B values to LDS
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[167], data1=v[168], offset0=16, offset1=18))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[169], data1=v[170], offset0=20, offset1=22))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[171], data1=v[172], offset0=24, offset1=26))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[173], data1=v[174], offset0=28, offset1=30))

  # Store A values to LDS
  for i in range(8):
    k.emit(ds_store_b32(addr=v[141+i], data0=v[175+i]))

  k.emit(s_waitcnt(simm16=0))
  k.emit(s_barrier())

  # ========== ZERO ACCUMULATORS ==========
  k.emit(s_mov_b32(sdst=s[12], ssrc0=0))
  # Zero accumulators v2-v133
  for i in range(0, 132, 2):
    k.emit(VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
                vdstx=v[2+i], srcx0=s[12], vsrcx1=v[0],
                vdsty=v[3+i], srcy0=s[12], vsrcy1=v[0]))
  k.emit(v_mov_b32(vdst=v[214], src0=0))

  # ========== LDS POINTER SETUP ==========
  k.emit(v_lshl_add_u32(vdst=v[155], src0=v[135], src1=2, src2=0x1080))
  k.emit(v_lshl_or_b32(vdst=v[166], src0=v[118], src1=4, src2=0x1080))
  k.emit(v_lshlrev_b32(vdst=v[3], src0=2, vsrc1=v[0]))
  k.emit(v_and_or_b32(vdst=v[165], src0=0x180, src1=v[3], src2=v[136]))

  k.emit(s_add_i32(sdst=s[7], ssrc0=s[4], ssrc1=-8))  # N - 8
  k.emit(s_mov_b32(sdst=s[12], ssrc0=0))  # loop counter

  # ========== MAIN LOOP ==========
  # Note: The initial global loads (before loop) loaded data for k=0..7.
  # Each iteration of the loop:
  # 1. Processes current k-tile (k, k+8, ...) from LDS
  # 2. Prefetches NEXT k-tile (k+8, k+16, ...)
  # 3. Stores prefetched data to LDS
  # Last valid k-tile is k=N-8. When processing it, we should NOT prefetch k=N.

  k.label("LOOP_START")

  # Check if we should prefetch for next iteration (counter < N-8)
  # s[7] = N - 8, s[12] = counter
  k.emit(s_cmp_lt_i32(ssrc0=s[12], ssrc1=s[7]))
  # SCC=1 if counter < N-8 (should prefetch), SCC=0 if counter >= N-8 (skip prefetch)
  k.emit(s_cbranch_scc0(simm16=0))  # skip prefetch if SCC=0, placeholder offset
  prefetch_skip_branch_idx = len(k.instructions) - 1

  # Advance global load pointers and prefetch for next iteration
  k.emit(v_add_nc_u32(vdst=v[203], src0=0x20000, vsrc1=v[203]))
  k.emit(v_add_nc_u32(vdst=v[215], src0=0x20, vsrc1=v[215]))

  # Issue global loads for B (8 values)
  for i in range(4):
    k.emit(global_load_b32(vdst=v[167+i], addr=v[203], saddr=s[24+i*2:25+i*2]))
  for i in range(4, 8):
    k.emit(global_load_b32(vdst=v[167+i], addr=v[203], saddr=s[24+i*2:25+i*2]))

  # Issue global loads for A (8 values)
  for i in range(8):
    k.emit(global_load_b32(vdst=v[175+i], addr=v[215], saddr=s[40+i*2:41+i*2]))

  k.label("AFTER_PREFETCH")

  k.emit(s_setprio(simm16=0))
  k.emit(v_mov_b32(vdst=v[183], src0=v[165]))
  k.emit(v_mov_b32(vdst=v[202], src0=v[166]))

  # Inner loop: 8 k-steps - compute using data already in LDS
  for step in range(8):
    emit_lds_loads(k, 183, 202)
    k.emit(v_add_nc_u32(vdst=v[183], src0=0x210, vsrc1=v[183]))
    k.emit(v_add_nc_u32(vdst=v[202], src0=0x200, vsrc1=v[202]))
    k.emit(s_waitcnt(simm16=0))
    k.emit(s_setprio(simm16=1))
    emit_fma_block(k)
    k.emit(s_setprio(simm16=0))

  # Wait for global loads and barrier
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_barrier())

  # Store prefetched data to LDS (for next iteration to consume)
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[167], data1=v[168], offset0=16, offset1=18))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[169], data1=v[170], offset0=20, offset1=22))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[171], data1=v[172], offset0=24, offset1=26))
  k.emit(ds_store_2addr_stride64_b32(addr=v[156], data0=v[173], data1=v[174], offset0=28, offset1=30))
  for i in range(8):
    k.emit(ds_store_b32(addr=v[141+i], data0=v[175+i]))

  k.emit(s_waitcnt(simm16=0))
  k.emit(s_barrier())

  # Loop control
  k.emit(s_add_i32(sdst=s[12], ssrc0=s[12], ssrc1=8))
  k.emit(s_cmp_ge_i32(ssrc0=s[12], ssrc1=s[4]))
  k.emit(s_cbranch_scc1(simm16=2))  # skip to epilogue (+2 dwords = skip s_branch)
  # s_branch offset: Target = PC + 4 + simm16*4, so simm16 = (Target - PC - 4) / 4
  branch_idx = len(k.instructions)
  loop_start_idx = k.labels["LOOP_START"]
  byte_offset = k.byte_offset(branch_idx, loop_start_idx)
  dword_offset = (byte_offset - 4) // 4
  k.emit(s_branch(simm16=dword_offset))

  # Fix up the prefetch skip branch offset
  after_prefetch_idx = k.labels["AFTER_PREFETCH"]
  skip_byte_offset = k.byte_offset(prefetch_skip_branch_idx + 1, after_prefetch_idx)
  k.instructions[prefetch_skip_branch_idx] = s_cbranch_scc0(simm16=skip_byte_offset // 4)

  # ========== EPILOGUE: Store C ==========
  k.label("EPILOGUE")
  k.emit(s_waitcnt(simm16=0))

  # Compute C address: row = blockIdx.y*128 + (tid&0x60) + (tid>>3)&3
  #                    col = blockIdx.x*128 + (tid&7)
  # addr = C + (row * N + col) * 4
  # Recompute (tid>>3)&3 since v[2] was overwritten by accumulator zeroing
  k.emit(v_bfe_u32(vdst=v[149], src0=v[0], src1=3, src2=2))  # v[149] = (tid>>3)&3
  k.emit(v_and_b32(vdst=v[151], src0=0x60, vsrc1=v[0]))  # v[151] = (tid & 0x60)
  k.emit(v_or_b32(vdst=v[151], src0=s[3], vsrc1=v[151])) # + blockIdx.y*128
  k.emit(v_or_b32(vdst=v[149], src0=v[151], vsrc1=v[149])) # row = blockIdx.y*128 + (tid&0x60) + (tid>>3)&3
  k.emit(v_mul_lo_u32(vdst=v[150], src0=v[149], src1=s[4])) # row * N (in elements)
  k.emit(v_or_b32(vdst=v[151], src0=s[2], vsrc1=v[118]))  # col = blockIdx.x*128 + tid%8 (in elements)
  k.emit(v_add_nc_u32(vdst=v[150], src0=v[150], vsrc1=v[151])) # row*N + col (in elements)
  k.emit(v_ashrrev_i32(vdst=v[151], src0=31, vsrc1=v[150]))  # sign extend for 64-bit
  k.emit(VOP3(VOP3Op.V_LSHLREV_B64, vdst=v[150:151], src0=2, src1=v[150])) # * 4 (convert to bytes)
  k.emit(v_add_co_u32(vdst=v[150], sdst=VCC_LO, src0=s[0], src1=v[150]))  # add C base address
  k.emit(v_add_co_ci_u32(vdst=v[151], src0=s[1], vsrc1=v[151]))  # add high bits with carry

  # Store 128 values (32 stores of 4 floats each)
  # Each thread stores one 4x4 tile of the output
  # Row stride = N * 4 bytes
  k.emit(s_lshl_b32(sdst=s[7], ssrc0=s[4], ssrc1=2))  # N * 4

  # Store first column of 4 values
  k.emit(global_store_b128(addr=v[150:151], data=v[2:5], saddr=NULL))

  # Move to next row and store remaining rows
  for row in range(1, 4):
    k.emit(v_add_co_u32(vdst=v[150], sdst=VCC_LO, src0=s[7], src1=v[150]))
    k.emit(v_add_co_ci_u32(vdst=v[151], src0=0, vsrc1=v[151]))
    base = 2 + row * 4
    k.emit(global_store_b128(addr=v[150:151], data=v[base:base+3], saddr=NULL))

  k.emit(s_nop(simm16=0))
  k.emit(s_sendmsg(simm16=3))
  k.emit(s_endpgm())

  return k.to_asm()

# ============================================================================
# Test harness
# ============================================================================
def test_matmul():
  from tinygrad import Context, GlobalCounters

  dev = Device[Device.DEFAULT]
  arch = dev.arch
  print(f"Device arch: {arch}")

  asm = build_kernel(arch)
  if getenv("PRINT_ASM", 0):
    print("Generated assembly:")
    print(asm)

  try:
    binary = dev.compiler.compile(asm)
    print(f"Compiled! Binary size: {len(binary)} bytes")
  except Exception as e:
    print(f"Compilation failed: {e}")
    print("\nFull assembly:")
    print(asm)
    return

  prg = dev.runtime("kernel", binary)

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  b = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  a_hcq = a.uop.buffer.ensure_allocated()._buf
  b_hcq = b.uop.buffer.ensure_allocated()._buf
  c_hcq = c.uop.buffer.ensure_allocated()._buf

  grid = (N // BLOCK_N, N // BLOCK_M, 1)
  local = (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  run_count = getenv("CNT", 5)
  ets = []
  try:
    for _ in range(run_count):
      et = prg(a_hcq, b_hcq, c_hcq, global_size=grid, local_size=local, wait=True)
      ets.append(et)
    print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")
  except Exception as e:
    print(f"Kernel execution failed: {e}")
    import traceback
    traceback.print_exc()
    return

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a @ b).realize()
    with Context(DEBUG=0):
      err = (c - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-06:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  if getenv("ASM", 0):
    print(build_kernel(Device[Device.DEFAULT].arch))
  else:
    test_matmul()
