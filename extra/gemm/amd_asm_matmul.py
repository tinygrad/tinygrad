# RDNA3 128x128 tiled GEMM kernel - DSL version
# Computes C = A @ B for 4096x4096 float32 matrices using 128x128 tiles
#
# Architecture: RDNA3 (gfx1100)
# Tile size: 128x128 (each workgroup computes one tile of C)
# Workgroup: 128 threads (arranged as 32x4 for coalesced memory access)
# Inner loop: 8 iterations per K-block, processing 8 columns of A and 8 rows of B
#
# Accumulators: 128 vgprs (v[2-117], v[120-124], v[126-129], v[131-133])

import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.helpers import getenv, colored
from tinygrad.engine.realize import Runner, Estimates, ExecItem
from extra.assembly.amd.dsl import s, v, VCC_LO, RawImm, EXEC_LO
from extra.assembly.amd.autogen.rdna3.ins import *

# =============================================================================
# Kernel constants
# =============================================================================
LDS_SIZE = 8320       # Local data share size in bytes
MATRIX_DIM = 4096     # Matrix dimension N (assumes square NxN matrices)
LDS_A_STRIDE = 0x210  # LDS stride for A tile (528 bytes)
LDS_B_STRIDE = 0x200  # LDS stride for B tile (512 bytes)
LDS_BASE_OFFSET = 0x1080  # Base LDS offset for tiles
ADDR_MASK = 0x3fffff80    # Address alignment mask

# s_waitcnt encodings: wait for memory operations to complete
WAIT_LGKM = 64519    # wait for LDS/GDS/KMEM (lgkm_cnt=0)
WAIT_ALL = 0         # wait for everything
WAIT_VMEM = 1015     # wait for VMEM only (vm_cnt=0, lgkm_cnt=63)

# Float constants as hex
F32_ONE = 0x3F800000  # 1.0f

# =============================================================================
# Named register assignments (VGPRs) - COMPACT LAYOUT
# =============================================================================
V_LANE_ID_MOD8 = 118      # lane_id & 7 (column within 8-wide tile chunk)
V_OUTPUT_ROW = 119        # output row coordinate
V_LANE_MOD8_X4 = 134      # V_LANE_ID_MOD8 << 2 (byte offset)
V_LANE_DIV8_X4 = 135      # (lane_id >> 3) << 2
V_ADDR_HI_ZERO = 136      # always 0 (for 64-bit address high bits)
V_LDS_A_BASE = 125        # LDS A-tile base address for inner loop (in ACC_RESERVED gap)
V_LDS_B_BASE = 130        # LDS B-tile base address for inner loop (in ACC_RESERVED gap)
V_GLOBAL_A_ADDR = 119     # global memory A prefetch address (reuses V_OUTPUT_ROW slot during main loop)
V_GLOBAL_B_ADDR = 154     # global memory B prefetch address

# LDS tile register destinations - SEPARATE from DATA to avoid overlap
# DATA regs (v155-170) receive global prefetch
# A TILE reuses ROW_REGS (v137-144) which are free after prologue
# B TILE at v171-186
V_A_TILE_REGS = [137, 139, 141, 143]  # A tile: 4 pairs (v137-144, reusing ROW_REGS)
V_B_TILE_REGS = [171, 173, 175, 177, 179, 181, 183, 185]  # B tile: 8 pairs (v171-186)

# =============================================================================
# Named register assignments (SGPRs)
# =============================================================================
S_OUT_PTR = (0, 1)        # output C matrix base pointer
S_TILE_X = 2              # workgroup_x << 7
S_TILE_Y = 3              # workgroup_y << 7
S_DIM_N = 4               # matrix dimension N
S_ALPHA = 5               # alpha scalar (1.0f = 0x3F800000)
S_LOOP_BOUND = 7          # K-8 (loop termination bound)
S_A_PTR = (8, 9)          # A matrix base pointer
S_B_PTR = (10, 11)        # B matrix base pointer
S_LOOP_CTR = 12           # loop counter (increments by 8)
S_PREFETCH_FLAG = 13      # prefetch condition flag / row stride in epilogue
S_WORKGROUP_X = 14        # workgroup_id_x
S_WORKGROUP_Y = 15        # workgroup_id_y

# =============================================================================
# Data tables
# =============================================================================

# Accumulator grid: ACC_GRID[a_idx][b_idx] = vgpr for C[a,b]
# a_idx: which A value (0-7), b_idx: which B value (0-15)
# Scattered due to VOPD bank constraints (vdst_x % 4 != vdst_y % 4)
ACC_GRID = [
  [  5,  3,  9,  8,   37, 35, 41, 40,   69, 67, 73, 72,  101, 99,105,104],  # a0
  [  4,  2,  7,  6,   36, 34, 39, 38,   68, 66, 71, 70,  100, 98,103,102],  # a1
  [ 17, 16, 13, 11,   49, 48, 45, 43,   81, 80, 77, 75,  113,112,109,107],  # a2
  [ 15, 14, 12, 10,   47, 46, 44, 42,   79, 78, 76, 74,  111,110,108,106],  # a3
  [ 21, 19, 25, 24,   53, 51, 57, 56,   85, 83, 89, 88,  117,115,121,120],  # a4
  [ 20, 18, 23, 22,   52, 50, 55, 54,   84, 82, 87, 86,  116,114,123,122],  # a5
  [133,128, 29, 27,   33, 32, 61, 59,   65, 64, 93, 91,   97, 96,129,127],  # a6
  [131,132, 28, 26,   31, 30, 60, 58,   63, 62, 92, 90,   95, 94,124,126],  # a7 (was v214, moved to v132)
]

# Derived: all 128 accumulator registers to zero before loop
ACC_REGS = sorted(set(acc for row in ACC_GRID for acc in row))

# Reserved registers in the accumulator range (not used for accumulators)
ACC_RESERVED = {118, 119, 125, 130}

# Optimized (a_pair, b_pair) iteration order for better GPU scheduling
# Interleaves A and B pairs to maximize instruction-level parallelism
FMAC_PAIR_ORDER = [
  (0,0),(0,1),(1,1),(1,0), (2,0),(2,1),(3,1),(3,2), (0,2),(0,3),(1,3),(1,2), (2,2),(2,3),(3,3),(3,4),
  (0,4),(0,5),(1,5),(1,4), (2,4),(2,5),(3,5),(3,6), (0,6),(0,7),(1,7),(1,6), (2,6),(2,7),(3,7),(3,0),
]

def derive_fmac_pattern(acc_grid, a_tile_regs=None, b_tile_regs=None):
  """Generate 64 dual FMAC ops from accumulator grid with optimized iteration order."""
  if a_tile_regs is None: a_tile_regs = V_A_TILE_REGS
  if b_tile_regs is None: b_tile_regs = V_B_TILE_REGS
  pattern = []
  for idx, (a_pair, b_pair) in enumerate(FMAC_PAIR_ORDER):
    a_even, a_odd = a_pair * 2, a_pair * 2 + 1
    b_even, b_odd = b_pair * 2, b_pair * 2 + 1
    a_base, b_base = a_tile_regs[a_pair], b_tile_regs[b_pair]
    # Op 1: normal order -> C[a_even, b_even] + C[a_odd, b_odd]
    pattern.append((acc_grid[a_even][b_even], acc_grid[a_odd][b_odd],
                   a_base, b_base, a_base+1, b_base+1))
    # Op 2: alternate swapping A vs B to vary register banks
    if idx % 2 == 0:  # swap B
      pattern.append((acc_grid[a_even][b_odd], acc_grid[a_odd][b_even],
                     a_base, b_base+1, a_base+1, b_base))
    else:  # swap A
      pattern.append((acc_grid[a_odd][b_even], acc_grid[a_even][b_odd],
                     a_base+1, b_base, a_base, b_base+1))
  return pattern

# Derived: 64 dual FMAC operations
FMAC_PATTERN = derive_fmac_pattern(ACC_GRID)

def derive_permute_swaps(acc_grid, reserved=ACC_RESERVED):
  """Derive swap sequence to permute accumulators from FMAC layout to output order.

  After FMAC loop: acc_grid[a][b] holds C[a,b]
  Output order: for row_half in 0,1; col_group in 0-3; row_in_group in 0-3; b_off in 0-3
    -> need C[row_half*4 + row_in_group, col_group*4 + b_off] in descending reg order
  """
  out_regs = [r for r in range(133, 1, -1) if r not in reserved]
  out_regs.remove(124)
  out_regs = [124] + out_regs  # 124 at front due to epilogue temp reg usage

  def target_ab(i):
    row_half, col_group = i // 64, (i // 16) % 4
    row_in_group, b_off = (i // 4) % 4, i % 4
    return (row_half * 4 + row_in_group, col_group * 4 + b_off)

  reg_contents = {acc_grid[a][b]: (a, b) for a in range(8) for b in range(16)}
  ab_location = {ab: r for r, ab in reg_contents.items()}

  swaps = []
  for i in range(128):
    target_reg, needed_ab = out_regs[i], target_ab(i)
    current_reg = ab_location[needed_ab]
    if current_reg != target_reg:
      swaps.append((current_reg, target_reg))
      ab_at_target = reg_contents.get(target_reg)
      reg_contents[target_reg], ab_location[needed_ab] = needed_ab, target_reg
      if ab_at_target is not None:
        reg_contents[current_reg], ab_location[ab_at_target] = ab_at_target, current_reg
  return swaps

# Derived: swap sequence to arrange accumulators for output
PERMUTE_SWAPS = derive_permute_swaps(ACC_GRID)

# Derived: output register order after PERMUTE_SWAPS (descending, 124 at front)
OUT_REGS = [r for r in range(133, 1, -1) if r not in ACC_RESERVED]
OUT_REGS.remove(124)
OUT_REGS = [124] + OUT_REGS

# =============================================================================
# LDS tile staging registers - COMPACT LAYOUT
# =============================================================================
# DATA regs receive contiguous global prefetch, then write to LDS
# TILE regs receive scattered LDS loads (ds_load_b64 pairs), then feed FMACs
# These are SEPARATE - DATA lives during prefetch/store, TILE lives during inner loop
V_LDS_A_ADDR = [153]                          # single base register for A stores (use +512 offsets)
V_LDS_A_DATA = list(range(155, 163))          # 8 data registers for A prefetch (v155-162)
V_LDS_B_ADDR = list(range(145, 153))          # 8 address registers for B stores
V_LDS_B_DATA = list(range(163, 171))          # 8 data registers for B prefetch (v163-170)

# Derived: interleaved A/B store pairs for LDS writes
# A stores use V_LDS_A_ADDR[0] + i*512 offset, B stores use precomputed addresses
# Format: (addr_vreg, data_vreg, offset0, offset1)
LDS_STORE_PAIRS = []
for i, (a_data, b_addr, b_data) in enumerate(zip(V_LDS_A_DATA, V_LDS_B_ADDR, V_LDS_B_DATA)):
  a_off = i * 512  # A stores are strided by 512
  LDS_STORE_PAIRS.append((V_LDS_A_ADDR[0], a_data, a_off & 0xFF, a_off >> 8))  # A store with offset
  LDS_STORE_PAIRS.append((b_addr, b_data, 0, 0))  # B store with no offset

# Global memory prefetch schedule: (vdst1, vdst2, addr_vreg, saddr_lo1, saddr_lo2)
# Prefetch into DATA regs (v171-182, spanning A_DATA[4:8] and B_DATA[0:8])
# First 2 pairs from B address, next 4 pairs from A address
PREFETCH_LOADS = [(V_LDS_A_DATA[4+2*i], V_LDS_A_DATA[4+2*i+1], V_GLOBAL_B_ADDR, 32+4*i, 34+4*i) for i in range(2)] + \
                 [(V_LDS_B_DATA[2*(i-2)], V_LDS_B_DATA[2*(i-2)+1], V_GLOBAL_A_ADDR, 32+4*i, 34+4*i) for i in range(2, 6)]

# Initial tile prefetch: (vdst, saddr_lo) - load into A data regs
INIT_PREFETCH = [(V_LDS_A_DATA[i], 24+2*i) for i in range(4)]

# Initial tile loads: (vdst, addr_lo) pairs - use temp regs in accumulator gaps
INIT_TILE_LOADS = [(23,5),(24,9),(25,7),(26,2),(27,11),(28,13),(29,6),(30,8),(31,10),(12,12),(13,14),(3,2),(4,4),(5,8),(6,6),(7,10)]

# Initial LDS stores: (addr, d0, d1, off0, off1) for stride64 or (addr, data) for single
# Updated to use new V_LDS_B_ADDR
INIT_LDS_STORES = [
  (8,23,24,16,18), (145,29), (8,25,26,20,22), (146,30), (147,31),
  (8,27,28,24,26), (148,12), (149,13), (8,3,4,28,30), (150,5), (151,6), (152,7)]

# A matrix row offset registers (scattered to avoid accumulator conflicts)
ROW_REGS = list(range(137, 145))  # v137-v144 (8 regs)

# =============================================================================
# Kernel class
# =============================================================================

class Kernel:
  def __init__(self, arch='gfx1100'):
    self.instructions, self.labels, self.branch_targets, self.arch = [], {}, {}, arch

  def emit(self, inst): self.instructions.append(inst); return inst
  def label(self, name): self.labels[name] = len(self.instructions)
  def branch_to(self, label): self.branch_targets[len(self.instructions) - 1] = label

  def add64(self, dst_lo, dst_hi, src_lo, src_hi, off):
    """s[dst_lo:dst_hi] = s[src_lo:src_hi] + off"""
    if off: self.emit(s_add_u32(s[dst_lo], s[src_lo], off)); self.emit(s_addc_u32(s[dst_hi], s[src_hi], 0))
    elif dst_lo != src_lo: self.emit(s_mov_b64(s[dst_lo:dst_hi], s[src_lo:src_hi]))

  def global_load(self, vdst, addr, saddr=None):
    """Global load b32"""
    self.emit(global_load_b32(vdst=v[vdst], addr=v[addr:addr+1],
              saddr=s[saddr:saddr+2] if saddr else RawImm(124)))

  def waitcnt(self, lgkm=None, vm=None):
    """Wait for memory operations. lgkm=N waits until N lgkm ops remain, vm=N waits until N vmem ops remain."""
    from extra.assembly.amd.asm import waitcnt as encode_waitcnt
    if lgkm == 0 and vm is None: self.emit(s_waitcnt(simm16=WAIT_LGKM))
    elif vm == 0 and lgkm is None: self.emit(s_waitcnt(simm16=WAIT_VMEM))
    elif lgkm == 0 and vm == 0: self.emit(s_waitcnt(simm16=WAIT_ALL))
    elif vm is not None and lgkm is None:
      self.emit(s_waitcnt(simm16=encode_waitcnt(vmcnt=vm, expcnt=7, lgkmcnt=63)))
    elif lgkm is not None and vm is None:
      self.emit(s_waitcnt(simm16=encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=lgkm)))
    else: raise ValueError(f"unsupported waitcnt: lgkm={lgkm}, vm={vm}")

  def barrier(self): self.emit(s_barrier())

  def to_asm(self):
    import re
    # Instruction stream with labels
    label_at = {pos: name for name, pos in self.labels.items()}
    body = []
    for i, inst in enumerate(self.instructions):
      if i in label_at: body.append(f'.{label_at[i]}:')
      asm = inst.disasm()
      if i in self.branch_targets:
        asm = re.sub(r'(s_cbranch_\w+|s_branch)\s+\S+', rf'\1 .{self.branch_targets[i]}', asm)
      body.append('\t' + asm)

    # HSA kernel descriptor attributes (zeros included for compatibility)
    hsa = [
      ('group_segment_fixed_size', LDS_SIZE), ('private_segment_fixed_size', 0), ('kernarg_size', 36),
      ('user_sgpr_count', 14), ('user_sgpr_dispatch_ptr', 0), ('user_sgpr_queue_ptr', 0),
      ('user_sgpr_kernarg_segment_ptr', 1), ('user_sgpr_dispatch_id', 0), ('user_sgpr_private_segment_size', 0),
      ('wavefront_size32', 1), ('uses_dynamic_stack', 0), ('enable_private_segment', 0),
      ('system_sgpr_workgroup_id_x', 1), ('system_sgpr_workgroup_id_y', 1), ('system_sgpr_workgroup_id_z', 0),
      ('system_sgpr_workgroup_info', 0), ('system_vgpr_workitem_id', 0), ('next_free_vgpr', 187),
      ('next_free_sgpr', 16), ('float_round_mode_32', 0), ('float_round_mode_16_64', 0),
      ('float_denorm_mode_32', 3), ('float_denorm_mode_16_64', 3), ('dx10_clamp', 1), ('ieee_mode', 1),
      ('fp16_overflow', 0), ('workgroup_processor_mode', 0), ('memory_ordered', 1), ('forward_progress', 0),
      ('shared_vgpr_count', 0)]

    return '\n'.join([
      '\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
      '\t.protected\tkernel', '\t.globl\tkernel', '\t.p2align\t8', '\t.type\tkernel,@function', 'kernel:',
      *body,
      '\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0', '\t.amdhsa_kernel kernel',
      *[f'\t\t.amdhsa_{k} {v}' for k, v in hsa],
      '\t.end_amdhsa_kernel', '\t.text', '.Lfunc_end0:', '\t.size\tkernel, .Lfunc_end0-kernel',
      '\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:',
      *[f'      - .address_space: global\n        .offset: {i*8}\n        .size: 8\n        .value_kind: global_buffer' for i in range(3)],
      f'    .group_segment_fixed_size: {LDS_SIZE}', '    .kernarg_segment_align: 8',
      '    .kernarg_segment_size: 24', '    .max_flat_workgroup_size: 128', '    .name: kernel',
      '    .private_segment_fixed_size: 0', '    .sgpr_count: 60', '    .symbol: kernel.kd',
      '    .vgpr_count: 187', '    .wavefront_size: 32', f'amdhsa.target: amdgcn-amd-amdhsa--{self.arch}',
      'amdhsa.version:', '  - 1', '  - 2', '...', '\t.end_amdgpu_metadata'])


# =============================================================================
# Kernel builder
# =============================================================================

def build_kernel(arch='gfx1100'):
  k = Kernel(arch)

  # ===========================================================================
  # PROLOGUE: Load kernel arguments and compute tile base addresses
  # ===========================================================================
  # Load A and B matrix pointers from kernarg (s[20:21] = A, s[22:23] = B)
  k.emit(s_load_b128(sdata=s[20:23], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))
  k.waitcnt(lgkm=0)

  # Compute 8 B matrix tile base pointers (s[24:39]) - each tile is 16KB apart (0x4000)
  for i in range(8): k.add64(24 + i*2, 25 + i*2, 22, 23, i * 0x4000)

  # B prefetch address: (workgroup_x * 128 + lane_id) * 4
  k.emit(s_lshl_b32(s[19], s[S_WORKGROUP_X], 7))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_B_ADDR], s[19], v[0]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_B_ADDR], 2, v[V_GLOBAL_B_ADDR]))

  # Compute 8 A matrix tile base pointers (s[40:55]) - each tile is 256KB apart (0x40000)
  for i in range(8): k.add64(40 + i*2, 41 + i*2, 20, 21, i * 0x40000)

  # A prefetch address: workgroup_y * 512KB + (lane_id / 8) * 4KB + (lane_id % 8)
  k.emit(s_lshl_b32(s[19], s[S_WORKGROUP_Y], 19))  # workgroup_y * 512KB
  k.emit(v_lshrrev_b32_e32(v[1], 3, v[0]))         # lane_id / 8
  k.emit(v_lshlrev_b32_e32(v[1], 12, v[1]))        # * 4KB
  k.emit(v_and_b32_e32(v[V_GLOBAL_A_ADDR], 7, v[0]))  # lane_id % 8
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], v[1], v[V_GLOBAL_A_ADDR]))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], s[19], v[V_GLOBAL_A_ADDR]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_A_ADDR], 2, v[V_GLOBAL_A_ADDR]))

  # Initialize constants
  k.emit(s_mov_b32(s[S_DIM_N], MATRIX_DIM))
  k.emit(s_mov_b32(s[S_ALPHA], F32_ONE))
  k.emit(s_mov_b32(s[6], 0))

  # Load C matrix pointer
  k.emit(s_load_b128(sdata=s[8:11], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))

  # Compute tile coordinates: tile_x = workgroup_x * 128, tile_y = workgroup_y * 128
  k.emit(s_lshl_b32(s[S_TILE_X], s[S_WORKGROUP_X], 7))
  k.emit(v_lshrrev_b32_e32(v[4], 3, v[0]))
  k.emit(v_or_b32_e32(v[1], s[S_TILE_X], v[0]))
  k.emit(s_lshl_b32(s[S_TILE_Y], s[S_WORKGROUP_Y], 7))
  k.emit(v_and_b32_e32(v[V_LANE_ID_MOD8], 7, v[0]))
  k.emit(s_bfe_i32(s[S_LOOP_CTR], s[S_WORKGROUP_Y], 0x10018))
  k.emit(v_or_b32_e32(v[22], s[S_TILE_Y], v[4]))
  k.emit(v_ashrrev_i32_e32(v[2], 31, v[1]))
  k.emit(s_lshr_b32(s[S_LOOP_CTR], s[S_LOOP_CTR], 25))

  # Load output pointer
  k.emit(s_load_b64(sdata=s[S_OUT_PTR[0]:S_OUT_PTR[1]], sbase=s[0:1], offset=0x10, soffset=RawImm(124)))

  # Precompute lane-based offsets for epilogue
  k.emit(v_lshlrev_b32_e32(v[V_LANE_MOD8_X4], 2, v[V_LANE_ID_MOD8]))
  k.emit(v_lshlrev_b64(v[5:6], 2, v[1:2]))
  k.waitcnt(lgkm=0)

  # ===========================================================================
  # Tile address computation for initial A/B matrix loads
  # ===========================================================================
  k.emit(s_lshl_b32(s[S_LOOP_BOUND], s[S_DIM_N], 4))  # row stride = 16*N
  k.emit(v_mul_lo_u32(v[ROW_REGS[0]], v[22], s[S_DIM_N]))  # A matrix row offsets
  for i in range(1, 8): k.emit(v_add_nc_u32_e32(v[ROW_REGS[i]], s[S_LOOP_BOUND], v[ROW_REGS[i-1]]))

  def addr64(dst, base_s):  # 64-bit address: v[dst:dst+1] = s[base_s:base_s+1] + v[dst]*4
    k.emit(v_ashrrev_i32_e32(v[dst+1], 31, v[dst]))
    k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[dst:dst+1]))
    k.emit(v_add_co_u32(v[dst], VCC_LO, s[base_s], v[dst]))
    k.emit(v_add_co_ci_u32_e32(v[dst+1], s[base_s+1], v[dst+1]))

  def b_addr(dst, mult, tmp=None):  # B address for col + mult*N
    tmp = tmp if tmp is not None else dst
    k.emit(v_mad_u32_u24(v[tmp], s[S_DIM_N], mult, v[1]))
    if tmp != dst:
      k.emit(v_ashrrev_i32_e32(v[tmp+1], 31, v[tmp]))
      k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[tmp:tmp+1]))
      k.emit(v_add_co_u32(v[dst], VCC_LO, s[S_B_PTR[0]], v[dst]))
      k.emit(v_add_co_ci_u32_e32(v[dst+1], s[S_B_PTR[1]], v[dst+1]))
    else: addr64(dst, S_B_PTR[0])

  def a_addr(dst, row_reg, tmp):  # A address for row_reg + lane_id_mod8
    k.emit(v_add_nc_u32_e32(v[tmp], v[row_reg], v[V_LANE_ID_MOD8]))
    k.emit(v_ashrrev_i32_e32(v[tmp+1], 31, v[tmp]))
    k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[tmp:tmp+1]))
    k.emit(v_add_co_u32(v[dst], VCC_LO, s[S_A_PTR[0]], v[dst]))
    k.emit(v_add_co_ci_u32_e32(v[dst+1], s[S_A_PTR[1]], v[dst+1]))

  # Batch 1: B addresses (cols 0-5) and loads
  k.emit(v_add_co_u32(v[5], VCC_LO, s[S_B_PTR[0]], v[5]))
  k.emit(v_add_co_ci_u32_e32(v[6], s[S_B_PTR[1]], v[6]))
  for dst, mult in [(9,1), (7,2), (2,3), (11,4), (13,5)]: b_addr(dst, mult)
  k.emit(s_clause(simm16=5))  # 6 consecutive global loads
  for vdst, addr in INIT_TILE_LOADS[:6]: k.global_load(vdst, addr)

  # Batch 2: A addresses (rows 0-4) and loads
  for dst, ri in [(6,0), (8,1), (10,2), (12,3), (14,4)]:
    k.emit(v_add_nc_u32_e32(v[dst], v[ROW_REGS[ri]], v[V_LANE_ID_MOD8]))
    addr64(dst, S_A_PTR[0])
  k.emit(s_clause(simm16=4))  # 5 consecutive global loads
  for vdst, addr in INIT_TILE_LOADS[6:11]: k.global_load(vdst, addr)

  # Batch 3: B cols 6-7, A rows 5-7, and loads
  for dst, mult, tmp in [(2,6,15), (4,7,4)]: b_addr(dst, mult, tmp)
  for dst, ri, tmp in [(8,5,16), (6,6,18), (10,7,20)]: a_addr(dst, ROW_REGS[ri], tmp)
  k.emit(s_clause(simm16=4))  # 5 consecutive global loads
  for vdst, addr in INIT_TILE_LOADS[11:]: k.global_load(vdst, addr)

  # ===========================================================================
  # LDS store address computation (bank-conflict-avoiding swizzle)
  # ===========================================================================
  # Computes v[141-148]: 8 B-tile row store addresses with swizzling
  # Formula: v[141+i] = LDS_A_STRIDE * (lane_id % 8) + row_base[i]
  # The row_base values are swizzled based on (lane_id >> 3) to avoid bank conflicts
  #
  # Also computes v[8]: A-tile store base address (used with stride64 stores)
  # And v[V_LANE_DIV8_X4]: (lane_id >> 3) << 2 for later use

  # Temporary registers for row offset computation
  lds_r = [10, 11, 14, 15, 16, 17, 18]  # row offsets before swizzle
  lds_t = [19, 20, 21, 22, 32, 33, 34]  # masked values for swizzle

  # Compute raw row offsets: v[lds_r[i]] = 16*(i+1) | v[22]
  k.emit(v_add_nc_u32_e32(v[9], s[S_LOOP_CTR], v[22]))
  for i, r in enumerate(lds_r): k.emit(v_or_b32_e32(v[r], 16 * (i + 1), v[22]))

  # Extract sign bit of workgroup_x for swizzle
  k.emit(s_bfe_i32(s[S_LOOP_BOUND], s[S_WORKGROUP_X], 0x10018))
  k.emit(v_and_b32_e32(v[9], ADDR_MASK, v[9]))
  k.emit(s_lshr_b32(s[S_LOOP_BOUND], s[S_LOOP_BOUND], 25))

  # Compute masked versions for bank conflict avoidance
  k.emit(v_add_nc_u32_e32(v[19], s[S_LOOP_CTR], v[10]))
  k.emit(v_add_nc_u32_e32(v[8], s[S_LOOP_BOUND], v[1]))
  for d, r in zip([20, 21, 32, 33, 34, 35], lds_r[1:]): k.emit(v_add_nc_u32_e32(v[d], s[S_LOOP_CTR], v[r]))
  k.emit(v_and_b32_e32(v[8], ADDR_MASK, v[8]))
  k.emit(v_sub_nc_u32_e32(v[9], v[22], v[9]))
  for d, s_ in zip(lds_t, lds_t[1:] + [35]): k.emit(v_and_b32_e32(v[d], ADDR_MASK, v[s_]))
  k.emit(v_sub_nc_u32_e32(v[8], v[1], v[8]))

  # Apply swizzle and scale to byte offsets
  k.emit(v_lshlrev_b32_e32(v[9], 2, v[9]))
  for r, t in zip(lds_r, lds_t): k.emit(v_sub_nc_u32_e32(v[r], v[r], v[t]))
  k.emit(v_bfe_u32(v[2], v[0], 3, 2))  # v[2] = (lane_id >> 3) & 3
  k.emit(v_lshlrev_b32_e32(v[8], 2, v[8]))

  # Compute final B-tile addresses: V_LDS_B_ADDR[i] = LDS_A_STRIDE * lane_mod8 + row_base[i]
  k.emit(v_mad_u32_u24(v[V_LDS_B_ADDR[0]], LDS_A_STRIDE, v[V_LANE_ID_MOD8], v[9]))
  lds_bases = [9] + lds_r[:-1]  # base offsets for 8 rows
  for d, r in zip(lds_bases, lds_r): k.emit(v_lshlrev_b32_e32(v[d], 2, v[r]))
  k.emit(v_lshlrev_b32_e32(v[V_LANE_DIV8_X4], 2, v[2]))
  k.emit(v_add_nc_u32_e32(v[8], 0x80, v[8]))  # A-tile base + 128
  for i, base in enumerate(lds_bases):
    k.emit(v_mad_u32_u24(v[V_LDS_B_ADDR[1 + i]], LDS_A_STRIDE, v[V_LANE_ID_MOD8], v[base]))

  # Store initial tile data to LDS with software pipelining
  # We issued 16 global loads. Store as they complete rather than waiting for all.
  # Load order: v23,v24,v25,v26,v27,v28,v29,v30,v31,v12,v13,v3,v4,v5,v6,v7
  k.emit(s_mov_b32(s[S_LOOP_BOUND], 0))
  k.emit(s_cmp_gt_i32(s[S_DIM_N], 0))
  k.waitcnt(vm=14)  # wait for loads 0,1 (v23,v24)
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[23], data1=v[24], offset0=16, offset1=18))
  k.waitcnt(vm=9)   # wait for load 6 (v29)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[0]], data0=v[29], offset0=0, offset1=0))
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[25], data1=v[26], offset0=20, offset1=22))
  k.waitcnt(vm=8)   # wait for load 7 (v30)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[1]], data0=v[30], offset0=0, offset1=0))
  k.waitcnt(vm=7)   # wait for load 8 (v31)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[2]], data0=v[31], offset0=0, offset1=0))
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[27], data1=v[28], offset0=24, offset1=26))
  k.waitcnt(vm=6)   # wait for load 9 (v12)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[3]], data0=v[12], offset0=0, offset1=0))
  k.waitcnt(vm=5)   # wait for load 10 (v13)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[4]], data0=v[13], offset0=0, offset1=0))
  k.waitcnt(vm=3)   # wait for loads 11,12 (v3,v4)
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[3], data1=v[4], offset0=28, offset1=30))
  k.waitcnt(vm=2)   # wait for load 13 (v5)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[5]], data0=v[5], offset0=0, offset1=0))
  k.waitcnt(vm=1)   # wait for load 14 (v6)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[6]], data0=v[6], offset0=0, offset1=0))
  k.waitcnt(vm=0)   # wait for load 15 (v7)
  k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR[7]], data0=v[7], offset0=0, offset1=0))
  k.waitcnt(lgkm=0)
  k.barrier()

  # ===========================================================================
  # INIT: Compute LDS base addresses, then zero accumulators
  # ===========================================================================
  # Compute LDS base addresses (v[2], v[3] used here, then zeroed later)
  k.emit(s_ashr_i32(s[S_LOOP_BOUND], s[S_TILE_X], 31))
  k.emit(v_lshlrev_b32_e32(v[2], 4, v[2]))
  k.emit(s_lshr_b32(s[S_LOOP_BOUND], s[S_LOOP_BOUND], 25))
  k.emit(v_add_nc_u32_e32(v[3], s[S_LOOP_BOUND], v[1]))
  # ROW_SIGN_REGS sign-extension removed - values were never used
  k.emit(v_and_b32_e32(v[3], ADDR_MASK, v[3]))
  k.emit(v_sub_nc_u32_e32(v[3], v[1], v[3]))
  k.emit(v_lshl_or_b32(v[V_LDS_B_BASE], v[V_LANE_ID_MOD8], 4, LDS_BASE_OFFSET))
  k.emit(v_lshl_add_u32(v[V_LDS_A_ADDR[0]], v[3], 2, LDS_BASE_OFFSET))
  k.emit(v_lshlrev_b32_e32(v[3], 2, v[0]))
  # V_LDS_A_ADDR[1-7] eliminated - use offset-based addressing from V_LDS_A_ADDR[0]
  k.emit(v_and_or_b32(v[V_LDS_A_BASE], 0x180, v[3], v[2]))

  for r in ACC_REGS: k.emit(v_mov_b32_e32(v[r], 0))  # zero all 128 accumulator registers

  k.emit(s_add_i32(s[S_LOOP_BOUND], s[S_DIM_N], -8))
  k.emit(s_add_u32(s[S_A_PTR[0]], s[S_A_PTR[0]], 32))
  k.emit(s_addc_u32(s[S_A_PTR[1]], s[S_A_PTR[1]], 0))
  k.emit(s_mov_b32(s[S_LOOP_CTR], 0))
  k.emit(s_branch(simm16=0)); k.branch_to('LOOP_ENTRY')

  # ===========================================================================
  # MAIN GEMM LOOP
  # ===========================================================================

  k.label('LOOP_INC')
  k.emit(s_add_i32(s[S_LOOP_CTR], s[S_LOOP_CTR], 8))
  k.emit(s_cmp_ge_i32(s[S_LOOP_CTR], s[S_DIM_N]))
  k.emit(s_cbranch_scc1(simm16=0)); k.branch_to('EPILOGUE')

  k.label('LOOP_ENTRY')
  k.emit(s_cmp_lt_i32(s[S_LOOP_CTR], s[S_LOOP_BOUND]))
  k.emit(s_cselect_b32(s[S_PREFETCH_FLAG], -1, 0))
  k.emit(s_cmp_ge_i32(s[S_LOOP_CTR], s[S_LOOP_BOUND]))
  k.emit(s_cbranch_scc1(simm16=0)); k.branch_to('SKIP_PREFETCH')

  # Advance prefetch pointers
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_B_ADDR], 0x20000, v[V_GLOBAL_B_ADDR]))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], 0x20, v[V_GLOBAL_A_ADDR]))
  k.emit(s_setprio(0))

  k.emit(s_clause(simm16=3))  # 4 consecutive global loads
  for vdst, saddr_lo in INIT_PREFETCH:
    k.global_load(vdst, V_GLOBAL_B_ADDR, saddr_lo)

  k.label('SKIP_PREFETCH')
  # PTR registers eliminated - use offset-based addressing from BASE
  k.emit(s_mov_b32(s[S_WORKGROUP_X], 0))

  k.label('INNER_LOOP')

  # 8 inner loop iterations (6 with prefetch, 2 without)
  for iter in range(8):
    # Load A tile (4 pairs) and B tile (8 pairs) from LDS
    k.emit(s_clause(simm16=11))
    for i, vdst in enumerate(V_A_TILE_REGS):
      a_off = (i & 1) * 8 + (i >> 1) * 64 + iter * LDS_A_STRIDE
      k.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_A_BASE], offset0=a_off & 0xFF, offset1=a_off >> 8))
    for i, vdst in enumerate(V_B_TILE_REGS):
      b_off = (i & 1) * 8 + (i & 2) * 64 + (i >> 2) * 256 + iter * LDS_B_STRIDE
      k.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_B_BASE], offset0=b_off & 0xFF, offset1=b_off >> 8))

    # Issue global prefetch AFTER LDS loads (first 6 iterations only)
    # This provides some latency hiding while LDS loads complete
    if iter < 6:
      vdst1, vdst2, addr, slo1, slo2 = PREFETCH_LOADS[iter]
      k.emit(s_clause(simm16=1))
      k.global_load(vdst1, addr, slo1)
      k.global_load(vdst2, addr, slo2)

    k.waitcnt(lgkm=0)

    # 64 dual FMACs
    for i, (vdst_x, vdst_y, ax, bx, ay, by) in enumerate(FMAC_PATTERN):
      k.emit(VOPD(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_FMAC_F32,
                  vdstx=v[vdst_x], vdsty=v[vdst_y], srcx0=v[ax], vsrcx1=v[bx], srcy0=v[ay], vsrcy1=v[by]))
      if i == 4: k.emit(s_setprio(1))
    k.emit(s_setprio(0))

  k.emit(s_and_not1_b32(VCC_LO, EXEC_LO, s[S_PREFETCH_FLAG]))
  k.waitcnt(vm=0)
  k.barrier()
  k.emit(s_cbranch_vccnz(simm16=0)); k.branch_to('LOOP_INC')

  # Store prefetched data to LDS
  for addr, data, off0, off1 in LDS_STORE_PAIRS:
    k.emit(ds_store_b32(addr=v[addr], data0=v[data], offset0=off0, offset1=off1))

  k.waitcnt(lgkm=0)
  k.barrier()
  k.emit(s_branch(simm16=0)); k.branch_to('LOOP_INC')

  # ===========================================================================
  # EPILOGUE: Permute and store results
  # ===========================================================================
  k.label('EPILOGUE')

  # Rearrange accumulators from FMAC layout to contiguous output order
  for a, b in PERMUTE_SWAPS:
    k.emit(v_swap_b32_e32(v[a], v[b]))

  # Compute output coordinates: v[V_LANE_ID_MOD8] = col, v[V_OUTPUT_ROW] = row
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32,
              vdstx=v[149], vdsty=v[150], srcx0=v[V_LANE_MOD8_X4], vsrcx1=v[0], srcy0=v[V_LANE_DIV8_X4], vsrcy1=v[0]))
  k.emit(v_and_b32_e32(v[0], 0x60, v[0]))
  k.emit(v_or_b32_e32(v[V_LANE_ID_MOD8], s[S_TILE_X], v[149]))
  k.emit(v_add_nc_u32_e32(v[0], s[S_TILE_Y], v[0]))
  k.emit(v_or_b32_e32(v[V_OUTPUT_ROW], v[0], v[150]))

  # Precompute row offsets: v[144-147] for rows 0-3, v[148-151] for rows 16-19
  for base, row_off in [(144, 0), (148, 16)]:
    if row_off: k.emit(v_or_b32_e32(v[1], row_off, v[V_OUTPUT_ROW]))
    k.emit(v_mul_lo_u32(v[base], v[1] if row_off else v[V_OUTPUT_ROW], s[S_DIM_N]))
    for i in range(3): k.emit(v_add_nc_u32_e32(v[base + 1 + i], s[S_DIM_N], v[base + i]))

  k.emit(v_mov_b32_e32(v[V_ADDR_HI_ZERO], 0))
  k.emit(s_lshl_b32(s[S_PREFETCH_FLAG], s[S_DIM_N], 2))  # row stride in bytes

  # Store 128 output values as 32 groups of 4 (128-bit stores)
  # Layout: 2 row halves (0-3, 16-19) x 4 col groups x 4 rows = 32 stores of 4 floats
  epilogue_reserved = {V_LANE_ID_MOD8, V_OUTPUT_ROW, V_LANE_MOD8_X4, V_LANE_DIV8_X4, V_ADDR_HI_ZERO}

  for i, (row_half, col_off, row_in_group) in enumerate([(rh, co, ri)
      for rh in range(2) for co in [0, 32, 64, 96] for ri in range(4)]):
    row = row_half * 16 + row_in_group
    srcs = OUT_REGS[i*4:(i+1)*4]

    # Find temp register for scaled values (must not conflict with reserved regs)
    tmp = max(srcs) + 5
    while any(r in epilogue_reserved for r in range(tmp, tmp + 4)): tmp += 1

    # Scale by alpha
    for j, src in enumerate(srcs):
      k.emit(v_mul_f32_e32(v[tmp + j], s[S_ALPHA], v[src]))

    # Compute output address
    if row_in_group == 0:  # first row: compute base address for this column group
      if col_off == 0: k.emit(v_mov_b32_e32(v[0], v[V_LANE_ID_MOD8]))
      else: k.emit(v_add_nc_u32_e32(v[0], col_off, v[V_LANE_ID_MOD8]))
      row_base = 144 + row if row < 4 else 148 + row - 16
      k.emit(v_add_nc_u32_e32(v[0], v[row_base], v[0]))
      k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
      k.emit(v_add_co_u32(v[0], VCC_LO, s[S_OUT_PTR[0]], v[0]))
      k.emit(v_add_co_ci_u32_e32(v[1], s[S_OUT_PTR[1]], v[V_ADDR_HI_ZERO]))
    else:  # subsequent rows: just add stride
      k.emit(v_add_co_u32(v[0], VCC_LO, s[S_PREFETCH_FLAG], v[0]))
      k.emit(v_add_co_ci_u32_e32(v[1], v[1], v[V_ADDR_HI_ZERO]))

    k.emit(global_store_b128(addr=v[0:1], data=v[tmp:tmp+3], saddr=RawImm(124)))

  k.emit(s_nop(0))
  k.emit(s_sendmsg(simm16=3))
  k.emit(s_endpgm())

  return k.to_asm()

# =============================================================================
# Test harness
# =============================================================================

N = getenv("N", 4096)
BLOCK_M, BLOCK_N = 128, 128
THREADS = 128

def test_matmul():
  dev = Device[Device.DEFAULT]
  print(f"Device arch: {dev.arch}")

  asm = build_kernel(dev.arch)
  if getenv("PRINT_ASM", 0): print(asm)

  binary = dev.compiler.compile(asm)
  print(f"Compiled! Binary size: {len(binary)} bytes")

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  b = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  grid, local = (N // BLOCK_N, N // BLOCK_M, 1), (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  _prg = dev.runtime("kernel", binary)
  class AsmRunner(Runner):
    def __init__(self):
      super().__init__(colored("kernel", "cyan"), Device.DEFAULT, Estimates(ops=N*N*N*2, mem=N*N*4*3))
    def __call__(self, rawbufs, var_vals, wait=False):
      c_buf, a_buf, b_buf = [x.ensure_allocated()._buf for x in rawbufs]
      return _prg(a_buf, b_buf, c_buf, global_size=grid, local_size=local, wait=wait)

  ei = ExecItem(None, [c.uop.buffer, a.uop.buffer, b.uop.buffer], prg=AsmRunner())

  ets = []
  with Context(DEBUG=2):
    for _ in range(getenv("CNT", 5)): ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2): tc = (a @ b).realize()
    with Context(DEBUG=0): err = (c - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-06: raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  if getenv("ASM", 0): print(build_kernel(Device[Device.DEFAULT].arch))
  else: test_matmul()
