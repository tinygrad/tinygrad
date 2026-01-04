# RDNA3 128x128 tiled GEMM kernel - DSL version
# Computes C = A @ B for 4096x4096 float32 matrices using 128x128 tiles
#
# Architecture: RDNA3 (gfx1100)
# Tile size: 128x128 (each workgroup computes one tile of C)
# Workgroup: 128 threads (arranged as 32x4 for coalesced memory access)
# Inner loop: 8 iterations per K-block, processing 8 columns of A and 8 rows of B

from extra.assembly.amd.dsl import s, v, VCC_LO, RawImm, EXEC_LO
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.autogen.rdna3.enum import VOP1Op

# =============================================================================
# Kernel constants
# =============================================================================
LDS_SIZE = 8320   # Local data share size in bytes
MATRIX_DIM = 4096 # Matrix dimension N (assumes square NxN matrices)

# =============================================================================
# Named register assignments (VGPRs)
# =============================================================================
V_LANE_ID_MOD8 = 118      # lane_id & 7 (column within 8-wide tile chunk)
V_OUTPUT_ROW = 119        # output row coordinate
V_LANE_MOD8_X4 = 135      # V_LANE_ID_MOD8 << 2 (byte offset)
V_LANE_DIV8_X4 = 136      # (lane_id >> 3) << 2
V_ADDR_HI_ZERO = 152      # always 0 (for 64-bit address high bits)
V_LDS_A_BASE = 165        # LDS A-tile base address for inner loop
V_LDS_B_BASE = 166        # LDS B-tile base address for inner loop
V_LDS_A_PTR = 183         # current LDS A-tile read pointer
V_LDS_B_PTR = 202         # current LDS B-tile read pointer
V_GLOBAL_B_ADDR = 203     # global memory B prefetch address
V_GLOBAL_A_ADDR = 215     # global memory A prefetch address

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
# Register assignment tables
# =============================================================================

# 59 accumulator register pairs (zeroed with immediate 0)
ACC_ZERO_PAIRS = [
  (111,94),(97,80),(95,78),(81,128),(79,126),(129,108),(127,106),(109,92),(107,90),(93,76),(91,74),(77,122),(75,120),(123,104),(121,102),(105,88),
  (103,86),(89,72),(87,70),(73,116),(71,114),(117,100),(115,98),(101,84),(99,82),(85,68),(83,66),(69,64),(67,62),(65,48),(63,46),(49,32),
  (47,30),(33,16),(31,14),(17,60),(15,58),(61,44),(59,42),(45,28),(43,26),(29,12),(27,10),(13,56),(11,54),(57,40),(55,38),(41,24),
  (39,22),(25,8),(23,6),(9,52),(7,50),(53,36),(51,34),(37,20),(35,18),(21,4),(19,2)]

# 122 register swaps to permute accumulator layout before output
PERMUTE_SWAPS = [
  (128,2),(56,2),(46,2),(100,2),(77,2),(87,2),(27,2),(54,2),(42,2),(98,2),(76,2),(83,2),(32,2),(40,2),(110,2),(68,2),
  (93,2),(23,2),(59,2),(38,2),(106,2),(66,2),(92,2),(19,2),(64,2),(24,2),(62,2),(20,2),(61,2),(39,2),(107,2),(70,2),
  (90,2),(18,2),(60,2),(35,2),(112,2),(72,2),(94,2),(4,2),(129,2),(7,2),(127,2),(6,2),(126,2),(133,3),(57,3),(47,3),
  (101,3),(81,3),(89,3),(31,3),(37,3),(113,3),(73,3),(95,3),(5,3),(124,3),(131,8),(53,8),(49,8),(105,8),(79,8),(85,8),
  (33,8),(41,8),(111,8),(69,8),(97,8),(9,8),(132,8),(114,10),(12,10),(115,10),(16,10),(122,10),(120,11),(14,11),(116,11),(13,11),
  (121,11),(15,11),(117,11),(17,11),(123,11),(65,21),(25,21),(63,21),(58,22),(34,22),(108,22),(67,22),(96,22),(8,22),(50,26),(44,26),
  (99,26),(80,26),(88,26),(30,26),(36,26),(109,26),(71,26),(91,26),(22,26),(51,28),(48,28),(104,28),(78,28),(84,28),(29,28),(55,28),
  (43,28),(102,28),(74,28),(82,28),(103,45),(75,45),(86,45),(26,45),(45,52),(52,214)]

# 64 dual FMAC operations per inner loop iteration: (vdst_x, vdst_y, src_ax, src_bx, src_ay, src_by)
FMAC_PATTERN = [
  (5,2,186,184,187,185),(3,4,186,185,187,184),(9,6,186,188,187,189),(7,8,187,188,186,189),
  (13,10,190,188,191,189),(11,12,190,189,191,188),(17,14,190,184,191,185),(15,16,191,184,190,185),
  (21,18,194,184,195,185),(19,20,194,185,195,184),(25,22,194,188,195,189),(23,24,195,188,194,189),
  (29,26,198,188,199,189),(27,28,198,189,199,188),(33,30,198,192,199,193),(31,32,199,192,198,193),
  (37,34,186,192,187,193),(35,36,186,193,187,192),(41,38,186,196,187,197),(39,40,187,196,186,197),
  (45,42,190,196,191,197),(43,44,190,197,191,196),(49,46,190,192,191,193),(47,48,191,192,190,193),
  (53,50,194,192,195,193),(51,52,194,193,195,192),(57,54,194,196,195,197),(55,56,195,196,194,197),
  (61,58,198,196,199,197),(59,60,198,197,199,196),(65,62,198,200,199,201),(63,64,199,200,198,201),
  (69,66,186,200,187,201),(67,68,186,201,187,200),(73,70,186,204,187,205),(71,72,187,204,186,205),
  (77,74,190,204,191,205),(75,76,190,205,191,204),(81,78,190,200,191,201),(79,80,191,200,190,201),
  (85,82,194,200,195,201),(83,84,194,201,195,200),(89,86,194,204,195,205),(87,88,195,204,194,205),
  (93,90,198,204,199,205),(91,92,198,205,199,204),(97,94,198,208,199,209),(95,96,199,208,198,209),
  (101,98,186,208,187,209),(99,100,186,209,187,208),(105,102,186,212,187,213),(103,104,187,212,186,213),
  (109,106,190,212,191,213),(107,108,190,213,191,212),(113,110,190,208,191,209),(111,112,191,208,190,209),
  (117,114,194,208,195,209),(115,116,194,209,195,208),(121,122,194,212,195,213),(123,120,195,212,194,213),
  (129,126,198,212,199,213),(127,124,198,213,199,212),(133,214,198,184,199,185),(131,128,199,184,198,185)]

# Global memory prefetch: (vdst1, vdst2, addr_vreg, saddr_lo1, saddr_lo2)
PREFETCH_LOADS = [(171+2*i, 172+2*i, V_GLOBAL_B_ADDR if i < 2 else V_GLOBAL_A_ADDR, 32+4*i, 34+4*i) for i in range(6)]

# LDS store pairs for prefetched data: (addr_vreg, data_vreg)
# Interleaves A stores (addr v[155,158-164], data v[167-174]) with B stores (addr v[141-148], data v[175-182])
_lds_addr_a, _lds_addr_b = [155] + list(range(158, 165)), list(range(141, 149))
LDS_STORE_PAIRS = [x for pair in zip([(a, 167+i) for i, a in enumerate(_lds_addr_a)],
                                      [(b, 175+i) for i, b in enumerate(_lds_addr_b)]) for x in pair]

# Initial tile prefetch: (vdst, saddr_lo)
INIT_PREFETCH = [(167+i, 24+2*i) for i in range(4)]

# Initial tile loads: (vdst, addr_lo) pairs
INIT_TILE_LOADS = [(23,5),(24,9),(25,7),(26,2),(27,11),(28,13),(29,6),(30,8),(31,10),(12,12),(13,14),(3,2),(4,4),(5,8),(6,6),(7,10)]

# Epilogue stores: (col_offset, row_index, data_base_reg, [src0, src1, src2, src3])
EPILOGUE_STORES = [
  (0,0,138,[124,133,132,131]), (0,1,134,[129,128,127,126]), (0,2,128,[123,122,121,120]), (0,3,124,[117,116,115,114]),
  (32,0,120,[113,112,111,110]), (32,1,114,[109,108,107,106]), (32,2,110,[105,104,103,102]), (32,3,107,[101,100,99,98]),
  (64,0,103,[97,96,95,94]), (64,1,98,[93,92,91,90]), (64,2,94,[89,88,87,86]), (64,3,90,[85,84,83,82]),
  (96,0,86,[81,80,79,78]), (96,1,82,[77,76,75,74]), (96,2,78,[73,72,71,70]), (96,3,74,[69,68,67,66]),
  (0,16,70,[65,64,63,62]), (0,17,66,[61,60,59,58]), (0,18,62,[57,56,55,54]), (0,19,58,[53,52,51,50]),
  (32,16,54,[49,48,47,46]), (32,17,50,[45,44,43,42]), (32,18,46,[41,40,39,38]), (32,19,42,[37,36,35,34]),
  (64,16,38,[33,32,31,30]), (64,17,34,[29,28,27,26]), (64,18,30,[25,24,23,22]), (64,19,26,[21,20,19,18]),
  (96,16,22,[17,16,15,14]), (96,17,18,[13,12,11,10]), (96,18,14,[9,8,7,6]), (96,19,10,[5,4,3,2])]

# Initial LDS stores: (addr, d0, d1, off0, off1) for stride64 or (addr, data) for single
INIT_LDS_STORES = [
  (8,23,24,16,18), (141,29), (8,25,26,20,22), (142,30), (143,31),
  (8,27,28,24,26), (144,12), (145,13), (8,3,4,28,30), (146,5), (147,6), (148,7)]

# A matrix row offset registers
ROW_REGS = [119, 125, 130, 134, 137, 138, 139, 140]

# =============================================================================
# Kernel class
# =============================================================================

class Kernel:
  def __init__(self, arch='gfx1100'):
    self.instructions = []
    self.labels = {}
    self.branch_targets = {}
    self.arch = arch

  def emit(self, inst):
    self.instructions.append(inst)
    return inst

  def label(self, name):
    self.labels[name] = len(self.instructions)

  def branch_to(self, label):
    self.branch_targets[len(self.instructions) - 1] = label

  def add64_imm(self, dst_lo, dst_hi, src_lo, src_hi, offset):
    """64-bit add: s[dst_lo:dst_hi] = s[src_lo:src_hi] + offset"""
    if offset == 0 and (dst_lo != src_lo or dst_hi != src_hi):
      self.emit(s_mov_b64(s[dst_lo:dst_hi], s[src_lo:src_hi]))
    elif offset != 0:
      self.emit(s_add_u32(s[dst_lo], s[src_lo], offset))
      self.emit(s_addc_u32(s[dst_hi], s[src_hi], 0))

  def vopd_mov2(self, vdst1, vdst2, src1, src2):
    """Dual move: v[vdst1] = src1, v[vdst2] = src2"""
    self.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32,
                   vdstx=v[vdst1], vdsty=v[vdst2], srcx0=src1, vsrcx1=v[0], srcy0=src2, vsrcy1=v[0]))

  def fmac_block(self):
    """Emit 64 dual FMACs for one inner loop iteration."""
    for i, (vdst_x, vdst_y, ax, bx, ay, by) in enumerate(FMAC_PATTERN):
      self.emit(VOPD(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_FMAC_F32,
                     vdstx=v[vdst_x], vdsty=v[vdst_y], srcx0=v[ax], vsrcx1=v[bx], srcy0=v[ay], vsrcy1=v[by]))
      if i == 0:
        self.emit(s_setprio(1))

  def lds_load_ab_tiles(self):
    """Load A and B tiles from LDS into registers."""
    # A tile: 4 x 64-bit loads from v[V_LDS_A_PTR], storing to v[186,190,194,198]
    for i, vdst in enumerate([186, 190, 194, 198]):
      self.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_A_PTR], offset0=(i%2)*8 + (i//2)*64, offset1=0))
    # B tile: 8 x 64-bit loads from v[V_LDS_B_PTR], storing to v[184,188,192,196,200,204,208,212]
    for i, vdst in enumerate([184, 188, 192, 196, 200, 204, 208, 212]):
      self.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_B_PTR], offset0=(i%2)*8 + (i//2 % 2)*128, offset1=i//4))

  def main_loop_iter(self, prefetch_idx):
    """One iteration of the main GEMM loop."""
    self.lds_load_ab_tiles()
    self.emit(v_add_nc_u32_e32(v[V_LDS_A_PTR], 0x210, v[V_LDS_A_PTR]))
    self.emit(v_add_nc_u32_e32(v[V_LDS_B_PTR], 0x200, v[V_LDS_B_PTR]))
    self.emit(s_waitcnt(simm16=64519))
    self.fmac_block()
    self.emit(s_setprio(0))
    if prefetch_idx is not None:
      vdst1, vdst2, addr, slo1, slo2 = PREFETCH_LOADS[prefetch_idx]
      self.emit(global_load_b32(vdst=v[vdst1], addr=v[addr], saddr=s[slo1:slo1+2]))
      self.emit(global_load_b32(vdst=v[vdst2], addr=v[addr], saddr=s[slo2:slo2+2]))

  def epilogue_store_group(self, col_off, stores):
    """Store a group of 4 consecutive rows with the same column offset."""
    for i, (row_idx, data_base, srcs) in enumerate(stores):
      for j, src in enumerate(srcs):
        self.emit(v_mul_f32_e32(v[data_base + j], s[S_ALPHA], v[src]))
      if i == 0:
        if col_off == 0:
          self.emit(v_mov_b32_e32(v[0], v[V_LANE_ID_MOD8]))
        else:
          self.emit(v_add_nc_u32_e32(v[0], col_off, v[V_LANE_ID_MOD8]))
        row_reg = 144 + row_idx if row_idx < 4 else 148 + row_idx - 16
        self.emit(v_add_nc_u32_e32(v[0], v[row_reg], v[0]))
        self.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
        self.emit(v_add_co_u32(v[0], VCC_LO, s[S_OUT_PTR[0]], v[0]))
        self.emit(v_add_co_ci_u32_e32(v[1], s[S_OUT_PTR[1]], v[V_ADDR_HI_ZERO]))
      else:
        self.emit(v_add_co_u32(v[0], VCC_LO, s[S_PREFETCH_FLAG], v[0]))
        self.emit(v_add_co_ci_u32_e32(v[1], v[1], v[V_ADDR_HI_ZERO]))
      self.emit(global_store_b128(addr=v[0:1], data=v[data_base:data_base+3], saddr=RawImm(124)))

  def to_asm(self):
    import re
    lines = ['\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
             '\t.protected\tkernel', '\t.globl\tkernel', '\t.p2align\t8',
             '\t.type\tkernel,@function', 'kernel:']
    label_at = {pos: name for name, pos in self.labels.items()}
    for i, inst in enumerate(self.instructions):
      if i in label_at:
        lines.append(f'.{label_at[i]}:')
      asm = inst.disasm()
      if i in self.branch_targets:
        asm = re.sub(r'(s_cbranch_\w+|s_branch)\s+\S+', rf'\1 .{self.branch_targets[i]}', asm)
      lines.append('\t' + asm)
    # AMDHSA kernel descriptor
    hsa_attrs = [
      ('group_segment_fixed_size', LDS_SIZE), ('private_segment_fixed_size', 0), ('kernarg_size', 36),
      ('user_sgpr_count', 14), ('user_sgpr_dispatch_ptr', 0), ('user_sgpr_queue_ptr', 0),
      ('user_sgpr_kernarg_segment_ptr', 1), ('user_sgpr_dispatch_id', 0), ('user_sgpr_private_segment_size', 0),
      ('wavefront_size32', 1), ('uses_dynamic_stack', 0), ('enable_private_segment', 0),
      ('system_sgpr_workgroup_id_x', 1), ('system_sgpr_workgroup_id_y', 1), ('system_sgpr_workgroup_id_z', 0),
      ('system_sgpr_workgroup_info', 0), ('system_vgpr_workitem_id', 0), ('next_free_vgpr', 216),
      ('next_free_sgpr', 16), ('float_round_mode_32', 0), ('float_round_mode_16_64', 0),
      ('float_denorm_mode_32', 3), ('float_denorm_mode_16_64', 3), ('dx10_clamp', 1), ('ieee_mode', 1),
      ('fp16_overflow', 0), ('workgroup_processor_mode', 0), ('memory_ordered', 1), ('forward_progress', 0),
      ('shared_vgpr_count', 0)]
    lines.extend(['\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0', '\t.amdhsa_kernel kernel'])
    lines.extend([f'\t\t.amdhsa_{k} {v}' for k, v in hsa_attrs])
    lines.extend(['\t.end_amdhsa_kernel', '\t.text', '.Lfunc_end0:', '\t.size\tkernel, .Lfunc_end0-kernel'])
    lines.extend(['\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:'])
    for i in range(3):
      lines.extend([f'      - .address_space: global', f'        .offset: {i*8}', '        .size: 8', '        .value_kind: global_buffer'])
    lines.extend([f'    .group_segment_fixed_size: {LDS_SIZE}', '    .kernarg_segment_align: 8',
                  '    .kernarg_segment_size: 24', '    .max_flat_workgroup_size: 128', '    .name: kernel',
                  '    .private_segment_fixed_size: 0', '    .sgpr_count: 60', '    .symbol: kernel.kd',
                  '    .vgpr_count: 216', '    .wavefront_size: 32', f'amdhsa.target: amdgcn-amd-amdhsa--{self.arch}',
                  'amdhsa.version:', '  - 1', '  - 2', '...', '\t.end_amdgpu_metadata'])
    return '\n'.join(lines)


# =============================================================================
# Kernel builder
# =============================================================================

def build_kernel(arch='gfx1100'):
  k = Kernel(arch)

  # ===========================================================================
  # PROLOGUE: Load arguments and compute tile addresses
  # ===========================================================================

  k.emit(s_load_b128(sdata=s[20:23], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))
  k.emit(s_waitcnt(simm16=64519))

  # B matrix tile offsets: s[24:39] = base_B + i * 0x4000 for i in 0..7
  for i in range(8):
    k.add64_imm(24 + i*2, 25 + i*2, 22, 23, i * 0x4000)

  # v[V_GLOBAL_B_ADDR] = (workgroup_id_x << 7 + lane_id) << 2
  k.emit(s_lshl_b32(s[19], s[S_WORKGROUP_X], 7))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_B_ADDR], s[19], v[0]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_B_ADDR], 2, v[V_GLOBAL_B_ADDR]))

  # A matrix tile offsets: s[40:55] = base_A + i * 0x40000 for i in 0..7
  for i in range(8):
    k.add64_imm(40 + i*2, 41 + i*2, 20, 21, i * 0x40000)

  # v[V_GLOBAL_A_ADDR] = A tile address
  k.emit(s_lshl_b32(s[19], s[S_WORKGROUP_Y], 19))
  k.emit(v_lshrrev_b32_e32(v[1], 3, v[0]))
  k.emit(v_lshlrev_b32_e32(v[1], 12, v[1]))
  k.emit(v_and_b32_e32(v[V_GLOBAL_A_ADDR], 7, v[0]))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], v[1], v[V_GLOBAL_A_ADDR]))
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], s[19], v[V_GLOBAL_A_ADDR]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_A_ADDR], 2, v[V_GLOBAL_A_ADDR]))

  # Scalar constants
  k.emit(s_mov_b32(s[S_DIM_N], MATRIX_DIM))
  k.emit(s_mov_b32(s[S_ALPHA], 0x3F800000))  # alpha = 1.0f
  k.emit(s_mov_b32(s[6], 0))

  # Load C matrix pointer
  k.emit(s_load_b128(sdata=s[8:11], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))

  # Workgroup tile coordinates
  k.emit(s_lshl_b32(s[S_TILE_X], s[S_WORKGROUP_X], 7))
  k.emit(v_lshrrev_b32_e32(v[4], 3, v[0]))
  k.emit(v_or_b32_e32(v[1], s[S_TILE_X], v[0]))
  k.emit(s_lshl_b32(s[S_TILE_Y], s[S_WORKGROUP_Y], 7))
  k.emit(v_and_b32_e32(v[V_LANE_ID_MOD8], 7, v[0]))
  k.emit(s_bfe_i32(s[S_LOOP_CTR], s[S_WORKGROUP_Y], 0x10018))
  k.emit(v_or_b32_e32(v[22], s[S_TILE_Y], v[4]))
  k.emit(v_ashrrev_i32_e32(v[2], 31, v[1]))
  k.emit(s_lshr_b32(s[S_LOOP_CTR], s[S_LOOP_CTR], 25))

  # Load output matrix pointer
  k.emit(s_load_b64(sdata=s[S_OUT_PTR[0]:S_OUT_PTR[1]], sbase=s[0:1], offset=0x10, soffset=RawImm(124)))
  k.emit(v_lshlrev_b32_e32(v[V_LANE_MOD8_X4], 2, v[V_LANE_ID_MOD8]))
  k.emit(v_lshlrev_b64(v[5:6], 2, v[1:2]))
  k.emit(s_waitcnt(simm16=64519))

  # ===========================================================================
  # Tile address computation for initial A/B matrix loads
  # ===========================================================================

  # Row stride constant: s[S_LOOP_BOUND] = 16*N (for A matrix row stepping)
  k.emit(s_lshl_b32(s[S_LOOP_BOUND], s[S_DIM_N], 4))

  # A matrix row offsets: v[ROW_REGS[i]] = row*N + i*16N
  k.emit(v_mul_lo_u32(v[ROW_REGS[0]], v[22], s[S_DIM_N]))
  for i in range(1, 8):
    k.emit(v_add_nc_u32_e32(v[ROW_REGS[i]], s[S_LOOP_BOUND], v[ROW_REGS[i-1]]))

  # Helper: compute 64-bit address v[dst:dst+1] = base + offset*4
  def addr64(dst, base_s):
    k.emit(v_ashrrev_i32_e32(v[dst+1], 31, v[dst]))
    k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[dst:dst+1]))
    k.emit(v_add_co_u32(v[dst], VCC_LO, s[base_s], v[dst]))
    k.emit(v_add_co_ci_u32_e32(v[dst+1], s[base_s+1], v[dst+1]))

  # Helper: compute B address for col + mult*N -> v[dst:dst+1]
  def b_addr(dst, mult):
    k.emit(v_mad_u32_u24(v[dst], s[S_DIM_N], mult, v[1]))
    addr64(dst, S_B_PTR[0])

  # Helper: compute A address for row_reg + v[V_LANE_ID_MOD8] -> v[dst:dst+1]
  def a_addr(dst, row_reg, tmp):
    k.emit(v_add_nc_u32_e32(v[tmp], v[row_reg], v[V_LANE_ID_MOD8]))
    k.emit(v_ashrrev_i32_e32(v[tmp+1], 31, v[tmp]))
    k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[tmp:tmp+1]))
    k.emit(v_add_co_u32(v[dst], VCC_LO, s[S_A_PTR[0]], v[dst]))
    k.emit(v_add_co_ci_u32_e32(v[dst+1], s[S_A_PTR[1]], v[dst+1]))

  # --- B addresses for batch 1 (cols 0-5) ---
  k.emit(v_add_co_u32(v[5], VCC_LO, s[S_B_PTR[0]], v[5]))
  k.emit(v_add_co_ci_u32_e32(v[6], s[S_B_PTR[1]], v[6]))
  for dst, mult in [(9,1), (7,2), (2,3), (11,4), (13,5)]:
    b_addr(dst, mult)

  # Batch 1: B matrix loads
  for vdst, addr_lo in INIT_TILE_LOADS[:6]:
    k.emit(global_load_b32(vdst=v[vdst], addr=v[addr_lo:addr_lo+1], saddr=RawImm(124)))

  # --- A addresses for batch 2 (rows 0-4) ---
  for dst, row_idx in [(6,0), (8,1), (10,2), (12,3), (14,4)]:
    k.emit(v_add_nc_u32_e32(v[dst], v[ROW_REGS[row_idx]], v[V_LANE_ID_MOD8]))
    addr64(dst, S_A_PTR[0])

  # Batch 2: A matrix loads
  for vdst, addr_lo in INIT_TILE_LOADS[6:11]:
    k.emit(global_load_b32(vdst=v[vdst], addr=v[addr_lo:addr_lo+1], saddr=RawImm(124)))

  # --- Batch 3: B cols 6-7, A rows 5-7 ---
  for dst, mult, tmp in [(2,6,15), (4,7,4)]:
    k.emit(v_mad_u32_u24(v[tmp], s[S_DIM_N], mult, v[1]))
    k.emit(v_ashrrev_i32_e32(v[tmp+1], 31, v[tmp]))
    k.emit(v_lshlrev_b64(v[dst:dst+1], 2, v[tmp:tmp+1]))
    k.emit(v_add_co_u32(v[dst], VCC_LO, s[S_B_PTR[0]], v[dst]))
    k.emit(v_add_co_ci_u32_e32(v[dst+1], s[S_B_PTR[1]], v[dst+1]))

  for dst, row_idx, tmp in [(8,5,16), (6,6,18), (10,7,20)]:
    a_addr(dst, ROW_REGS[row_idx], tmp)

  # Batch 3: Mixed B+A loads
  for vdst, addr_lo in INIT_TILE_LOADS[11:]:
    k.emit(global_load_b32(vdst=v[vdst], addr=v[addr_lo:addr_lo+1], saddr=RawImm(124)))

  # ===========================================================================
  # LDS address computation
  # ===========================================================================
  lds_rows = [10, 11, 14, 15, 16, 17, 18]

  k.emit(v_add_nc_u32_e32(v[9], s[S_LOOP_CTR], v[22]))
  for i, dst in enumerate(lds_rows):
    k.emit(v_or_b32_e32(v[dst], 16 * (i + 1), v[22]))

  k.emit(s_bfe_i32(s[S_LOOP_BOUND], s[S_WORKGROUP_X], 0x10018))
  k.emit(v_and_b32_e32(v[9], 0x3fffff80, v[9]))
  k.emit(s_lshr_b32(s[S_LOOP_BOUND], s[S_LOOP_BOUND], 25))
  k.emit(v_add_nc_u32_e32(v[19], s[S_LOOP_CTR], v[10]))
  k.emit(v_add_nc_u32_e32(v[8], s[S_LOOP_BOUND], v[1]))

  for dst, src in [(20,11), (21,14), (32,15), (33,16), (34,17), (35,18)]:
    k.emit(v_add_nc_u32_e32(v[dst], s[S_LOOP_CTR], v[src]))

  k.emit(v_and_b32_e32(v[8], 0x3fffff80, v[8]))
  k.emit(v_sub_nc_u32_e32(v[9], v[22], v[9]))

  for dst, src in [(19,19), (20,20), (21,21), (22,32), (32,33), (33,34), (34,35)]:
    k.emit(v_and_b32_e32(v[dst], 0x3fffff80, v[src]))

  k.emit(v_sub_nc_u32_e32(v[8], v[1], v[8]))
  k.emit(v_lshlrev_b32_e32(v[9], 2, v[9]))

  for a, b in [(10,19), (11,20), (14,21), (15,22), (16,32), (17,33), (18,34)]:
    k.emit(v_sub_nc_u32_e32(v[a], v[a], v[b]))

  k.emit(v_bfe_u32(v[2], v[0], 3, 2))
  k.emit(v_lshlrev_b32_e32(v[8], 2, v[8]))
  k.emit(v_mad_u32_u24(v[141], 0x210, v[V_LANE_ID_MOD8], v[9]))

  for dst, src in [(9,10), (10,11), (11,14), (14,15), (15,16), (16,17), (17,18)]:
    k.emit(v_lshlrev_b32_e32(v[dst], 2, v[src]))

  k.emit(v_lshlrev_b32_e32(v[V_LANE_DIV8_X4], 2, v[2]))
  k.emit(v_add_nc_u32_e32(v[8], 0x80, v[8]))

  for dst, src in [(142,9), (143,10), (144,11), (145,14), (146,15), (147,16), (148,17)]:
    k.emit(v_mad_u32_u24(v[dst], 0x210, v[V_LANE_ID_MOD8], v[src]))

  k.emit(s_mov_b32(s[S_LOOP_BOUND], 0))
  k.emit(s_cmp_gt_i32(s[S_DIM_N], 0))
  k.emit(s_waitcnt(simm16=0))

  # Store initial tile data to LDS
  for store in INIT_LDS_STORES:
    if len(store) == 5:
      k.emit(ds_store_2addr_stride64_b32(addr=v[store[0]], data0=v[store[1]], data1=v[store[2]],
                                          offset0=store[3], offset1=store[4]))
    else:
      k.emit(ds_store_b32(addr=v[store[0]], data0=v[store[1]], offset0=0, offset1=0))

  k.emit(s_waitcnt(simm16=64519))
  k.emit(s_barrier())

  # ===========================================================================
  # INIT: Zero accumulators and set up loop
  # ===========================================================================

  k.emit(s_mov_b32(s[S_LOOP_BOUND], -1))

  k.emit(s_ashr_i32(s[S_LOOP_BOUND], s[S_TILE_X], 31))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_LSHLREV_B32,
              vdstx=v[133], vdsty=v[2], srcx0=0, vsrcx1=v[0], srcy0=4, vsrcy1=v[2]))
  k.emit(s_lshr_b32(s[S_LOOP_BOUND], s[S_LOOP_BOUND], 25))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32,
              vdstx=v[124], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=s[S_LOOP_BOUND], vsrcy1=v[1]))

  # Sign-extend row offsets for 64-bit addressing
  for vreg, src in zip([149, 150, 151, 152, 153, 154, 156, 157], ROW_REGS):
    k.emit(v_ashrrev_i32_e32(v[vreg], 31, v[src]))

  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_AND_B32,
              vdstx=v[132], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=0x3fffff80, vsrcy1=v[3]))
  k.emit(v_sub_nc_u32_e32(v[3], v[1], v[3]))
  k.emit(v_lshl_or_b32(v[V_LDS_B_BASE], v[V_LANE_ID_MOD8], 4, 0x1080))
  k.vopd_mov2(131, 110, 0, 0)
  k.emit(v_lshl_add_u32(v[155], v[3], 2, 0x1080))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_LSHLREV_B32,
              vdstx=v[112], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=2, vsrcy1=v[0]))
  k.vopd_mov2(113, 96, 0, 0)

  # LDS A-tile store addresses: v[158..164] = (i+1)<<9 + v[155]
  for i, dst in enumerate([158, 159, 160, 161, 162, 163, 164]):
    k.emit(v_lshl_add_u32(v[dst], i + 1, 9, v[155]))

  k.emit(v_and_or_b32(v[V_LDS_A_BASE], 0x180, v[3], v[2]))

  # Zero all 128 accumulator registers
  for vx, vy in ACC_ZERO_PAIRS:
    k.vopd_mov2(vx, vy, 0, 0)
  for vreg in [214, 5, 3]:
    k.emit(v_mov_b32_e32(v[vreg], 0))

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

  for vdst, saddr_lo in INIT_PREFETCH:
    k.emit(global_load_b32(vdst=v[vdst], addr=v[V_GLOBAL_B_ADDR], saddr=s[saddr_lo:saddr_lo+2]))

  k.label('SKIP_PREFETCH')
  k.emit(v_mov_b32_e32(v[V_LDS_A_PTR], v[V_LDS_A_BASE]))
  k.emit(s_mov_b32(s[S_WORKGROUP_X], 0))
  k.emit(v_mov_b32_e32(v[V_LDS_B_PTR], v[V_LDS_B_BASE]))

  k.label('INNER_LOOP')

  # 8 inner loop iterations (6 with prefetch, 2 without)
  for i in range(8):
    k.main_loop_iter(i if i < 6 else None)

  k.emit(s_and_not1_b32(VCC_LO, EXEC_LO, s[S_PREFETCH_FLAG]))
  k.emit(s_waitcnt(simm16=1015))
  k.emit(s_barrier())
  k.emit(s_cbranch_vccnz(simm16=0)); k.branch_to('LOOP_INC')

  # Store prefetched data to LDS
  for addr, data in LDS_STORE_PAIRS:
    k.emit(ds_store_b32(addr=v[addr], data0=v[data], offset0=0, offset1=0))

  k.emit(s_waitcnt(simm16=64519))
  k.emit(s_barrier())
  k.emit(s_branch(simm16=0)); k.branch_to('LOOP_INC')

  # ===========================================================================
  # EPILOGUE: Permute and store results
  # ===========================================================================

  k.label('EPILOGUE')

  for a, b in PERMUTE_SWAPS:
    k.emit(v_swap_b32_e32(v[a], v[b]))

  k.vopd_mov2(149, 150, v[V_LANE_MOD8_X4], v[V_LANE_DIV8_X4])

  # Compute output addresses
  k.emit(v_and_b32_e32(v[0], 0x60, v[0]))
  k.emit(v_or_b32_e32(v[V_LANE_ID_MOD8], s[S_TILE_X], v[149]))
  k.emit(v_add_nc_u32_e32(v[0], s[S_TILE_Y], v[0]))
  k.emit(v_or_b32_e32(v[V_OUTPUT_ROW], v[0], v[150]))

  # Row multipliers: compute row*N for rows 0-3 and 16-19
  for base_reg, row_offset in [(144, 0), (148, 16)]:
    if row_offset:
      k.emit(v_or_b32_e32(v[1], row_offset, v[V_OUTPUT_ROW]))
      k.emit(v_mul_lo_u32(v[base_reg], v[1], s[S_DIM_N]))
    else:
      k.emit(v_mul_lo_u32(v[base_reg], v[V_OUTPUT_ROW], s[S_DIM_N]))
    for i in range(3):
      k.emit(v_add_nc_u32_e32(v[base_reg + 1 + i], s[S_DIM_N], v[base_reg + i]))

  k.emit(v_mov_b32_e32(v[V_ADDR_HI_ZERO], 0))
  k.emit(s_lshl_b32(s[S_PREFETCH_FLAG], s[S_DIM_N], 2))  # row stride in bytes

  # Store all results (8 groups of 4 consecutive rows)
  for group in range(8):
    col_off = [0, 32, 64, 96][group % 4]
    stores = [(EPILOGUE_STORES[group * 4 + i][1],
               EPILOGUE_STORES[group * 4 + i][2],
               EPILOGUE_STORES[group * 4 + i][3]) for i in range(4)]
    k.epilogue_store_group(col_off, stores)

  k.emit(s_nop(0))
  k.emit(s_sendmsg(simm16=3))
  k.emit(s_endpgm())

  return k.to_asm()
