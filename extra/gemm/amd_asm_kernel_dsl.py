# RDNA3 128x128 tiled GEMM kernel - DSL version of kernel8_batched_gmem.s
from extra.assembly.amd.dsl import Inst, Inst32, Inst64, Inst96, s, v, NULL, VCC_LO, RawImm, EXEC_LO, SrcMod
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.autogen.rdna3.enum import VOP1Op

LDS_SIZE, N = 8320, 4096

# Accumulator register pairs to zero (128 vgprs: 64 pairs)
ACC_ZERO_PAIRS = [
  (2,3),(4,5),(18,19),(20,21),(34,35),(36,37),(50,51),(52,53),(6,7),(8,9),(22,23),(24,25),(38,39),(40,41),(54,55),(56,57),
  (10,11),(12,13),(26,27),(28,29),(42,43),(44,45),(58,59),(60,61),(14,15),(16,17),(30,31),(32,33),(46,47),(48,49),(62,63),(64,65),
  (66,67),(68,69),(82,83),(84,85),(98,99),(100,101),(114,115),(116,117),(70,71),(72,73),(86,87),(88,89),(102,103),(104,105),(120,121),(122,123),
  (74,75),(76,77),(90,91),(92,93),(106,107),(108,109),(126,127),(128,129),(78,79),(80,81),(94,95),(96,97),(110,111),(112,113),(131,132),(133,124),
]

# Register permutation swaps for output layout
PERMUTE_SWAPS = [
  (128,2),(56,2),(46,2),(100,2),(77,2),(87,2),(27,2),(54,2),(42,2),(98,2),(76,2),(83,2),(32,2),(40,2),(110,2),(68,2),
  (93,2),(23,2),(59,2),(38,2),(106,2),(66,2),(92,2),(19,2),(64,2),(24,2),(62,2),(20,2),(61,2),(39,2),(107,2),(70,2),
  (90,2),(18,2),(60,2),(35,2),(112,2),(72,2),(94,2),(4,2),(129,2),(7,2),(127,2),(6,2),(126,2),(133,3),(57,3),(47,3),
  (101,3),(81,3),(89,3),(31,3),(37,3),(113,3),(73,3),(95,3),(5,3),(124,3),(131,8),(53,8),(49,8),(105,8),(79,8),(85,8),
  (33,8),(41,8),(111,8),(69,8),(97,8),(9,8),(132,8),(114,10),(12,10),(115,10),(16,10),(122,10),(120,11),(14,11),(116,11),(13,11),
  (121,11),(15,11),(117,11),(17,11),(123,11),(65,21),(25,21),(63,21),(58,22),(34,22),(108,22),(67,22),(96,22),(8,22),(50,26),(44,26),
  (99,26),(80,26),(88,26),(30,26),(36,26),(109,26),(71,26),(91,26),(22,26),(51,28),(48,28),(104,28),(78,28),(84,28),(29,28),(55,28),
  (43,28),(102,28),(74,28),(82,28),(103,45),(75,45),(86,45),(26,45),(45,52),(52,214),
]

# FMAC pattern for one inner loop iteration (64 dual FMACs): (vdstx, vdsty, srcx0, vsrcx1, srcy0, vsrcy1)
FMAC_PATTERN = [
  (5,2,186,184,187,185),(3,4,186,185,187,184),(9,6,186,188,187,189),(7,8,187,188,186,189),(13,10,190,188,191,189),(11,12,190,189,191,188),(17,14,190,184,191,185),(15,16,191,184,190,185),
  (21,18,194,184,195,185),(19,20,194,185,195,184),(25,22,194,188,195,189),(23,24,195,188,194,189),(29,26,198,188,199,189),(27,28,198,189,199,188),(33,30,198,192,199,193),(31,32,199,192,198,193),
  (37,34,186,192,187,193),(35,36,186,193,187,192),(41,38,186,196,187,197),(39,40,187,196,186,197),(45,42,190,196,191,197),(43,44,190,197,191,196),(49,46,190,192,191,193),(47,48,191,192,190,193),
  (53,50,194,192,195,193),(51,52,194,193,195,192),(57,54,194,196,195,197),(55,56,195,196,194,197),(61,58,198,196,199,197),(59,60,198,197,199,196),(65,62,198,200,199,201),(63,64,199,200,198,201),
  (69,66,186,200,187,201),(67,68,186,201,187,200),(73,70,186,204,187,205),(71,72,187,204,186,205),(77,74,190,204,191,205),(75,76,190,205,191,204),(81,78,190,200,191,201),(79,80,191,200,190,201),
  (85,82,194,200,195,201),(83,84,194,201,195,200),(89,86,194,204,195,205),(87,88,195,204,194,205),(93,90,198,204,199,205),(91,92,198,205,199,204),(97,94,198,208,199,209),(95,96,199,208,198,209),
  (101,98,186,208,187,209),(99,100,186,209,187,208),(105,102,186,212,187,213),(103,104,187,212,186,213),(109,106,190,212,191,213),(107,108,190,213,191,212),(113,110,190,208,191,209),(111,112,191,208,190,209),
  (117,114,194,208,195,209),(115,116,194,209,195,208),(121,122,194,212,195,213),(123,120,195,212,194,213),(129,126,198,212,199,213),(127,124,198,213,199,212),(133,214,198,184,199,185),(131,128,199,184,198,185),
]

# Prefetch load pairs: (vdst1, vdst2, addr_vreg, saddr_lo1, saddr_lo2)
PREFETCH_LOADS = [(171,172,203,32,34),(173,174,203,36,38),(175,176,215,40,42),(177,178,215,44,46),(179,180,215,48,50),(181,182,215,52,54)]

# LDS store pairs for writing prefetched data to LDS (addr, data)
LDS_STORE_PAIRS = [(155,167),(141,175),(158,168),(142,176),(159,169),(143,177),(160,170),(144,178),(161,171),(145,179),(162,172),(146,180),(163,173),(147,181),(164,174),(148,182)]

# Initial B-tile prefetch: (vdst, saddr_lo) pairs for global_load_b32 from v[203]
INIT_PREFETCH = [(167,24),(168,26),(169,28),(170,30)]

# Initial B-tile loads: (vdst, addr_lo) pairs for global_load_b32 - consolidated
INIT_B_LOADS = [(23,5),(24,9),(25,7),(26,2),(27,11),(28,13),(29,6),(30,8),(31,10),(12,12),(13,14),(3,2),(4,4),(5,8),(6,6),(7,10)]

# Accumulator zero pairs for LBB0_4 path (59 pairs) - zeros vgprs using immediate 0
ACC_ZERO_PAIRS_B = [
  (111,94),(97,80),(95,78),(81,128),(79,126),(129,108),(127,106),(109,92),(107,90),(93,76),(91,74),(77,122),(75,120),(123,104),(121,102),(105,88),
  (103,86),(89,72),(87,70),(73,116),(71,114),(117,100),(115,98),(101,84),(99,82),(85,68),(83,66),(69,64),(67,62),(65,48),(63,46),(49,32),
  (47,30),(33,16),(31,14),(17,60),(15,58),(61,44),(59,42),(45,28),(43,26),(29,12),(27,10),(13,56),(11,54),(57,40),(55,38),(41,24),
  (39,22),(25,8),(23,6),(9,52),(7,50),(53,36),(51,34),(37,20),(35,18),(21,4),(19,2),
]

# Epilogue stores: (col_offset, row_index, data_base_reg, [src0, src1, src2, src3])
EPILOGUE_STORES = [
  (0,0,138,[124,133,132,131]),(0,1,134,[129,128,127,126]),(0,2,128,[123,122,121,120]),(0,3,124,[117,116,115,114]),
  (32,0,120,[113,112,111,110]),(32,1,114,[109,108,107,106]),(32,2,110,[105,104,103,102]),(32,3,107,[101,100,99,98]),
  (64,0,103,[97,96,95,94]),(64,1,98,[93,92,91,90]),(64,2,94,[89,88,87,86]),(64,3,90,[85,84,83,82]),
  (96,0,86,[81,80,79,78]),(96,1,82,[77,76,75,74]),(96,2,78,[73,72,71,70]),(96,3,74,[69,68,67,66]),
  (0,16,70,[65,64,63,62]),(0,17,66,[61,60,59,58]),(0,18,62,[57,56,55,54]),(0,19,58,[53,52,51,50]),
  (32,16,54,[49,48,47,46]),(32,17,50,[45,44,43,42]),(32,18,46,[41,40,39,38]),(32,19,42,[37,36,35,34]),
  (64,16,38,[33,32,31,30]),(64,17,34,[29,28,27,26]),(64,18,30,[25,24,23,22]),(64,19,26,[21,20,19,18]),
  (96,16,22,[17,16,15,14]),(96,17,18,[13,12,11,10]),(96,18,14,[9,8,7,6]),(96,19,10,[5,4,3,2]),
]

class Kernel:
  def __init__(self, arch: str = 'gfx1100'):
    self.instructions = []
    self.labels = {}
    self.branch_targets = {}  # Maps instruction index to target label
    self.arch = arch

  def emit(self, inst): self.instructions.append(inst); return inst
  def label(self, name): self.labels[name] = len(self.instructions)
  def branch_to(self, label): self.branch_targets[len(self.instructions) - 1] = label

  # Helper: emit 64-bit address = base + offset (with carry)
  def add64_imm(self, dst_lo, dst_hi, src_lo, src_hi, offset):
    self.emit(s_add_u32(s[dst_lo], s[src_lo], offset))
    self.emit(s_addc_u32(s[dst_hi], s[src_hi], 0))

  # Helper: emit dual mov to zero two vgprs (using sgpr as source)
  def dual_mov_zero(self, vdst1, vdst2, zero_sgpr):
    self.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32,
                   vdstx=v[vdst1], vdsty=v[vdst2], srcx0=s[zero_sgpr], vsrcx1=v[0], srcy0=s[zero_sgpr], vsrcy1=v[0]))

  # Helper: emit dual mov to zero two vgprs (using immediate 0)
  def dual_mov_zero_imm(self, vdst1, vdst2):
    self.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32,
                   vdstx=v[vdst1], vdsty=v[vdst2], srcx0=0, vsrcx1=v[0], srcy0=0, vsrcy1=v[0]))

  # Helper: emit dual fmac
  def dual_fmac(self, vdstx, vdsty, ax, bx, ay, by):
    self.emit(VOPD(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_FMAC_F32,
                   vdstx=v[vdstx], vdsty=v[vdsty], srcx0=v[ax], vsrcx1=v[bx], srcy0=v[ay], vsrcy1=v[by]))

  # Helper: emit the 64 dual FMACs for one inner loop iteration
  # First FMAC, then s_setprio(1), then remaining 63 FMACs
  def fmac_block(self):
    self.dual_fmac(*FMAC_PATTERN[0])
    self.emit(s_setprio(1))
    for vdstx, vdsty, ax, bx, ay, by in FMAC_PATTERN[1:]:
      self.dual_fmac(vdstx, vdsty, ax, bx, ay, by)

  # Helper: emit the standard 12 ds_load_b64 for loading A and B tiles from LDS
  def lds_load_ab_tiles(self):
    # A matrix tile from v[183]: 4 pairs at offsets 0,8,64,72 -> v[186:199] (step 4)
    for i, (vdst, off) in enumerate([(186,0), (190,8), (194,64), (198,72)]):
      self.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[183], offset0=off, offset1=0))
    # B matrix tile from v[202]: 4 pairs at offsets 0,8,128,136 with offset1=0 -> v[184:197] (step 4)
    for vdst, off in [(184,0), (188,8), (192,128), (196,136)]:
      self.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[202], offset0=off, offset1=0))
    # B matrix tile from v[202]: 4 pairs at offsets 0,8,128,136 with offset1=1 -> v[200:213] (step 4)
    for vdst, off in [(200,0), (204,8), (208,128), (212,136)]:
      self.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[202], offset0=off, offset1=1))

  # Helper: swap two vgprs using hardware swap instruction
  def vswap(self, a, b):
    self.emit(v_swap_b32_e32(v[a], v[b]))

  # Helper: one iteration of the main GEMM loop with prefetch loads
  def main_loop_iter(self, prefetch_idx):
    self.lds_load_ab_tiles()
    self.emit(v_add_nc_u32_e32(v[183], 0x210, v[183]))
    self.emit(v_add_nc_u32_e32(v[202], 0x200, v[202]))
    self.emit(s_waitcnt(simm16=64519))
    self.fmac_block()
    self.emit(s_setprio(0))
    if prefetch_idx is not None:
      vdst1, vdst2, addr, slo1, slo2 = PREFETCH_LOADS[prefetch_idx]
      self.emit(global_load_b32(vdst=v[vdst1], addr=v[addr], saddr=s[slo1:slo1+2]))
      self.emit(global_load_b32(vdst=v[vdst2], addr=v[addr], saddr=s[slo2:slo2+2]))

  # Helper: write prefetched data to LDS (16 stores)
  def lds_store_prefetch(self):
    for addr, data in LDS_STORE_PAIRS:
      self.emit(ds_store_b32(addr=v[addr], data0=v[data], offset0=0, offset1=0))

  # Helper: multiply 4 accumulators by alpha and store for epilogue
  # col_base in v[118], row multipliers in v[144-147] for rows 0-3, v[148-151] for rows 16-19
  # row_idx_map: 0->144, 1->145, 2->146, 3->147, 16->148, 17->149, 18->150, 19->151
  def epilogue_mul_store(self, col_off, row_idx, data_base, srcs):
    # Multiply 4 source accumulators by alpha (s[5]) into data_base:data_base+3
    for i, src in enumerate(srcs):
      self.emit(v_mul_f32_e32(v[data_base + i], s[5], v[src]))
    # Compute column offset: v[0] = v[118] + col_off
    if col_off == 0:
      self.emit(v_mov_b32_e32(v[0], v[118]))
    else:
      self.emit(v_add_nc_u32_e32(v[0], col_off, v[118]))
    # Use precomputed row multipliers for all row indices
    row_reg = 144 + row_idx if row_idx < 4 else 148 + (row_idx - 16)
    self.emit(v_add_nc_u32_e32(v[0], v[row_reg], v[0]))
    # Sign-extend to 64-bit, shift left by 2 (sizeof float)
    self.emit(v_ashrrev_i32_e32(v[1], 31, v[0]))
    self.emit(v_lshlrev_b64(v[0:1], 2, v[0:1]))
    # Add base address s[0:1]
    self.emit(v_add_co_u32(v[0], VCC_LO, s[0], v[0]))
    self.emit(v_add_co_ci_u32_e32(v[1], s[1], v[1]))
    # Store 4 floats
    self.emit(global_store_b128(addr=v[0:1], data=v[data_base:data_base+3], saddr=RawImm(124)))

  def to_asm(self):
    lines = ['\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
             '\t.protected\tkernel', '\t.globl\tkernel', '\t.p2align\t8',
             '\t.type\tkernel,@function', 'kernel:']
    label_at = {pos: name for name, pos in self.labels.items()}
    for i, inst in enumerate(self.instructions):
      if i in label_at: lines.append(f'.{label_at[i]}:')
      asm_line = inst.disasm()
      # Fix up branch instructions to use label references
      if i in self.branch_targets:
        target_label = self.branch_targets[i]
        import re
        asm_line = re.sub(r'(s_cbranch_\w+|s_branch)\s+\S+', rf'\1 .{target_label}', asm_line)
      lines.append('\t' + asm_line)
    lines.extend(['\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0',
      '\t.amdhsa_kernel kernel', f'\t\t.amdhsa_group_segment_fixed_size {LDS_SIZE}',
      '\t\t.amdhsa_private_segment_fixed_size 0', '\t\t.amdhsa_kernarg_size 36',
      '\t\t.amdhsa_user_sgpr_count 14', '\t\t.amdhsa_user_sgpr_dispatch_ptr 0',
      '\t\t.amdhsa_user_sgpr_queue_ptr 0', '\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1',
      '\t\t.amdhsa_user_sgpr_dispatch_id 0', '\t\t.amdhsa_user_sgpr_private_segment_size 0',
      '\t\t.amdhsa_wavefront_size32 1', '\t\t.amdhsa_uses_dynamic_stack 0',
      '\t\t.amdhsa_enable_private_segment 0', '\t\t.amdhsa_system_sgpr_workgroup_id_x 1',
      '\t\t.amdhsa_system_sgpr_workgroup_id_y 1', '\t\t.amdhsa_system_sgpr_workgroup_id_z 0',
      '\t\t.amdhsa_system_sgpr_workgroup_info 0', '\t\t.amdhsa_system_vgpr_workitem_id 0',
      '\t\t.amdhsa_next_free_vgpr 216', '\t\t.amdhsa_next_free_sgpr 16',
      '\t\t.amdhsa_float_round_mode_32 0', '\t\t.amdhsa_float_round_mode_16_64 0',
      '\t\t.amdhsa_float_denorm_mode_32 3', '\t\t.amdhsa_float_denorm_mode_16_64 3',
      '\t\t.amdhsa_dx10_clamp 1', '\t\t.amdhsa_ieee_mode 1', '\t\t.amdhsa_fp16_overflow 0',
      '\t\t.amdhsa_workgroup_processor_mode 0', '\t\t.amdhsa_memory_ordered 1',
      '\t\t.amdhsa_forward_progress 0', '\t\t.amdhsa_shared_vgpr_count 0',
      '\t.end_amdhsa_kernel',
      '\t.text', '.Lfunc_end0:', '\t.size\tkernel, .Lfunc_end0-kernel',
      '\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:',
      '      - .address_space: global', '        .offset: 0', '        .size: 8',
      '        .value_kind: global_buffer', '      - .address_space: global',
      '        .offset: 8', '        .size: 8', '        .value_kind: global_buffer',
      '      - .address_space: global', '        .offset: 16', '        .size: 8',
      '        .value_kind: global_buffer', f'    .group_segment_fixed_size: {LDS_SIZE}',
      '    .kernarg_segment_align: 8', '    .kernarg_segment_size: 24',
      '    .max_flat_workgroup_size: 128', '    .name: kernel',
      '    .private_segment_fixed_size: 0', '    .sgpr_count: 60', '    .symbol: kernel.kd',
      '    .vgpr_count: 216', '    .wavefront_size: 32',
      f'amdhsa.target: amdgcn-amd-amdhsa--{self.arch}', 'amdhsa.version:', '  - 1',
      '  - 2', '...', '\t.end_amdgpu_metadata'])
    return '\n'.join(lines)

def build_kernel(arch='gfx1100'):
  k = Kernel(arch)

  # === KERNEL SETUP: Load arguments and compute matrix tile offsets ===
  k.emit(s_load_b128(sdata=s[20:23], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))
  k.emit(s_waitcnt(simm16=64519))
  # B matrix offsets: s[24:39] = s[22:23] + i*0x4000 for i in 0..7
  for i in range(8): k.add64_imm(24 + i*2, 25 + i*2, 22, 23, i * 0x4000)
  k.emit(s_lshl_b32(s[19], s[14], 7))
  k.emit(v_add_nc_u32_e32(v[203], s[19], v[0]))
  k.emit(v_lshlrev_b32_e32(v[203], 2, v[203]))
  # A matrix offsets: s[40:55] = s[20:21] + i*0x40000 for i in 0..7
  for i in range(8): k.add64_imm(40 + i*2, 41 + i*2, 20, 21, i * 0x40000)
  k.emit(s_lshl_b32(s[19], s[15], 19))
  k.emit(v_lshrrev_b32_e32(v[1], 3, v[0]))
  k.emit(v_lshlrev_b32_e32(v[1], 12, v[1]))
  k.emit(v_and_b32_e32(v[215], 7, v[0]))
  k.emit(v_add_nc_u32_e32(v[215], v[1], v[215]))
  k.emit(v_add_nc_u32_e32(v[215], s[19], v[215]))
  k.emit(v_lshlrev_b32_e32(v[215], 2, v[215]))
  k.emit(s_mov_b32(s[4], 4096))
  k.emit(s_mov_b32(s[5], 0x3F800000))
  k.emit(s_mov_b32(s[6], 0))
  k.emit(s_load_b128(sdata=s[8:11], sbase=s[0:1], offset=0x0, soffset=RawImm(124)))
  k.emit(s_lshl_b32(s[2], s[14], 7))
  k.emit(v_lshrrev_b32_e32(v[4], 3, v[0]))
  k.emit(v_or_b32_e32(v[1], s[2], v[0]))
  k.emit(s_lshl_b32(s[3], s[15], 7))
  k.emit(v_and_b32_e32(v[118], 7, v[0]))
  k.emit(s_bfe_i32(s[12], s[15], 0x10018))
  k.emit(v_or_b32_e32(v[22], s[3], v[4]))
  k.emit(v_ashrrev_i32_e32(v[2], 31, v[1]))
  k.emit(s_lshr_b32(s[12], s[12], 25))
  k.emit(s_load_b64(sdata=s[0:1], sbase=s[0:1], offset=0x10, soffset=RawImm(124)))
  k.emit(v_lshlrev_b32_e32(v[135], 2, v[118]))
  k.emit(v_lshlrev_b64(v[5:6], 2, v[1:2]))
  k.emit(s_waitcnt(simm16=64519))
  k.emit(v_add_nc_u32_e32(v[3], s[4], v[1]))
  k.emit(v_mul_lo_u32(v[119], v[22], s[4]))
  k.emit(v_add_co_u32(v[5], VCC_LO, s[10], v[5]))
  k.emit(v_add_co_ci_u32_e32(v[6], s[11], v[6]))
  k.emit(v_add_nc_u32_e32(v[7], s[4], v[3]))
  k.emit(v_ashrrev_i32_e32(v[4], 31, v[3]))
  k.emit(s_lshl_b32(s[7], s[4], 4))
  k.emit(v_add_nc_u32_e32(v[125], s[7], v[119]))
  k.emit(v_add_nc_u32_e32(v[2], s[4], v[7]))
  k.emit(v_ashrrev_i32_e32(v[8], 31, v[7]))
  k.emit(v_lshlrev_b64(v[9:10], 2, v[3:4]))
  k.emit(v_add_nc_u32_e32(v[130], s[7], v[125]))
  k.emit(v_add_nc_u32_e32(v[11], s[4], v[2]))
  k.emit(v_ashrrev_i32_e32(v[3], 31, v[2]))
  k.emit(v_lshlrev_b64(v[7:8], 2, v[7:8]))
  k.emit(v_add_co_u32(v[9], VCC_LO, s[10], v[9]))
  k.emit(v_add_nc_u32_e32(v[13], s[4], v[11]))
  k.emit(v_ashrrev_i32_e32(v[12], 31, v[11]))
  k.emit(v_lshlrev_b64(v[2:3], 2, v[2:3]))
  k.emit(v_add_co_ci_u32_e32(v[10], s[11], v[10]))
  k.emit(v_ashrrev_i32_e32(v[14], 31, v[13]))
  k.emit(v_add_co_u32(v[7], VCC_LO, s[10], v[7]))
  k.emit(v_lshlrev_b64(v[11:12], 2, v[11:12]))
  k.emit(v_add_co_ci_u32_e32(v[8], s[11], v[8]))
  k.emit(v_add_nc_u32_e32(v[15], s[4], v[13]))
  k.emit(v_add_co_u32(v[2], VCC_LO, s[10], v[2]))
  k.emit(v_lshlrev_b64(v[13:14], 2, v[13:14]))
  k.emit(v_add_co_ci_u32_e32(v[3], s[11], v[3]))
  k.emit(v_add_co_u32(v[11], VCC_LO, s[10], v[11]))
  k.emit(v_add_nc_u32_e32(v[4], s[4], v[15]))
  k.emit(v_add_co_ci_u32_e32(v[12], s[11], v[12]))
  k.emit(v_add_co_u32(v[13], VCC_LO, s[10], v[13]))
  k.emit(v_ashrrev_i32_e32(v[16], 31, v[15]))
  k.emit(v_add_nc_u32_e32(v[134], s[7], v[130]))
  k.emit(v_add_co_ci_u32_e32(v[14], s[11], v[14]))
  for vdst, alo in INIT_B_LOADS[:6]: k.emit(global_load_b32(vdst=v[vdst], addr=v[alo:alo+1], saddr=RawImm(124)))
  k.emit(v_add_nc_u32_e32(v[6], v[119], v[118]))
  k.emit(v_ashrrev_i32_e32(v[5], 31, v[4]))
  k.emit(v_add_nc_u32_e32(v[8], v[125], v[118]))
  k.emit(v_lshlrev_b64(v[2:3], 2, v[15:16]))
  k.emit(v_add_nc_u32_e32(v[137], s[7], v[134]))
  k.emit(v_ashrrev_i32_e32(v[7], 31, v[6]))
  k.emit(v_add_nc_u32_e32(v[10], v[130], v[118]))
  k.emit(v_lshlrev_b64(v[4:5], 2, v[4:5]))
  k.emit(v_ashrrev_i32_e32(v[9], 31, v[8]))
  k.emit(v_add_nc_u32_e32(v[12], v[134], v[118]))
  k.emit(v_add_nc_u32_e32(v[138], s[7], v[137]))
  k.emit(v_add_co_u32(v[2], VCC_LO, s[10], v[2]))
  k.emit(v_lshlrev_b64(v[6:7], 2, v[6:7]))
  k.emit(v_ashrrev_i32_e32(v[11], 31, v[10]))
  k.emit(v_add_co_ci_u32_e32(v[3], s[11], v[3]))
  k.emit(v_add_nc_u32_e32(v[14], v[137], v[118]))
  k.emit(v_add_co_u32(v[4], VCC_LO, s[10], v[4]))
  k.emit(v_lshlrev_b64(v[8:9], 2, v[8:9]))
  k.emit(v_ashrrev_i32_e32(v[13], 31, v[12]))
  k.emit(v_add_nc_u32_e32(v[139], s[7], v[138]))
  k.emit(v_add_co_ci_u32_e32(v[5], s[11], v[5]))
  k.emit(v_add_co_u32(v[6], VCC_LO, s[8], v[6]))
  k.emit(v_lshlrev_b64(v[10:11], 2, v[10:11]))
  k.emit(v_ashrrev_i32_e32(v[15], 31, v[14]))
  k.emit(v_add_co_ci_u32_e32(v[7], s[9], v[7]))
  k.emit(v_add_nc_u32_e32(v[16], v[138], v[118]))
  k.emit(v_add_co_u32(v[8], VCC_LO, s[8], v[8]))
  k.emit(v_lshlrev_b64(v[12:13], 2, v[12:13]))
  k.emit(v_add_nc_u32_e32(v[140], s[7], v[139]))
  k.emit(v_add_co_ci_u32_e32(v[9], s[9], v[9]))
  k.emit(v_add_nc_u32_e32(v[18], v[139], v[118]))
  k.emit(v_add_co_u32(v[10], VCC_LO, s[8], v[10]))
  k.emit(v_lshlrev_b64(v[14:15], 2, v[14:15]))
  k.emit(v_ashrrev_i32_e32(v[17], 31, v[16]))
  k.emit(v_add_co_ci_u32_e32(v[11], s[9], v[11]))
  k.emit(v_add_nc_u32_e32(v[20], v[140], v[118]))
  k.emit(v_add_co_u32(v[12], VCC_LO, s[8], v[12]))
  k.emit(v_ashrrev_i32_e32(v[19], 31, v[18]))
  k.emit(v_add_co_ci_u32_e32(v[13], s[9], v[13]))
  k.emit(v_add_co_u32(v[14], VCC_LO, s[8], v[14]))
  k.emit(v_lshlrev_b64(v[16:17], 2, v[16:17]))
  k.emit(v_ashrrev_i32_e32(v[21], 31, v[20]))
  k.emit(v_add_co_ci_u32_e32(v[15], s[9], v[15]))
  for vdst, alo in INIT_B_LOADS[6:11]: k.emit(global_load_b32(vdst=v[vdst], addr=v[alo:alo+1], saddr=RawImm(124)))
  k.emit(v_lshlrev_b64(v[6:7], 2, v[18:19]))
  k.emit(v_add_co_u32(v[8], VCC_LO, s[8], v[16]))
  k.emit(v_lshlrev_b64(v[10:11], 2, v[20:21]))
  k.emit(v_add_co_ci_u32_e32(v[9], s[9], v[17]))
  k.emit(v_add_co_u32(v[6], VCC_LO, s[8], v[6]))
  k.emit(v_add_co_ci_u32_e32(v[7], s[9], v[7]))
  k.emit(v_add_co_u32(v[10], VCC_LO, s[8], v[10]))
  k.emit(v_add_co_ci_u32_e32(v[11], s[9], v[11]))
  for vdst, alo in INIT_B_LOADS[11:]: k.emit(global_load_b32(vdst=v[vdst], addr=v[alo:alo+1], saddr=RawImm(124)))
  k.emit(v_add_nc_u32_e32(v[9], s[12], v[22]))
  # LDS row offset ORs: v[dst] = v[22] | offset for offsets 16,32,48,64,80,96,112
  for i, dst in enumerate([10, 11, 14, 15, 16, 17, 18]): k.emit(v_or_b32_e32(v[dst], 16*(i+1), v[22]))
  k.emit(s_bfe_i32(s[7], s[14], 0x10018))
  k.emit(v_and_b32_e32(v[9], 0x3fffff80, v[9]))
  k.emit(s_lshr_b32(s[7], s[7], 25))
  k.emit(v_add_nc_u32_e32(v[19], s[12], v[10]))  # LDS_ADD_PAIRS[0]
  k.emit(v_add_nc_u32_e32(v[8], s[7], v[1]))
  # LDS base offset adds: v[dst] = s[12] + v[src]
  for dst, src in [(20,11),(21,14),(32,15),(33,16),(34,17),(35,18)]: k.emit(v_add_nc_u32_e32(v[dst], s[12], v[src]))
  k.emit(v_and_b32_e32(v[8], 0x3fffff80, v[8]))
  k.emit(v_sub_nc_u32_e32(v[9], v[22], v[9]))
  # LDS address masking: v[dst] = v[src] & 0x3fffff80
  for dst, src in [(19,19),(20,20),(21,21),(22,32),(32,33),(33,34),(34,35)]: k.emit(v_and_b32_e32(v[dst], 0x3fffff80, v[src]))
  k.emit(v_sub_nc_u32_e32(v[8], v[1], v[8]))
  k.emit(v_lshlrev_b32_e32(v[9], 2, v[9]))
  # LDS offset subtractions: v[a] = v[a] - v[b]
  for a, b in [(10,19),(11,20),(14,21),(15,22),(16,32),(17,33),(18,34)]: k.emit(v_sub_nc_u32_e32(v[a], v[a], v[b]))
  k.emit(v_bfe_u32(v[2], v[0], 3, 2))
  k.emit(v_lshlrev_b32_e32(v[8], 2, v[8]))
  k.emit(v_mad_u32_u24(v[141], 0x210, v[118], v[9]))
  # Chained lshlrev_b32: v[dst] = v[src] << 2
  for dst, src in [(9,10),(10,11),(11,14),(14,15),(15,16),(16,17),(17,18)]: k.emit(v_lshlrev_b32_e32(v[dst], 2, v[src]))
  k.emit(v_lshlrev_b32_e32(v[136], 2, v[2]))
  k.emit(v_add_nc_u32_e32(v[8], 0x80, v[8]))
  # LDS B-tile address calculations: v_mad_u32_u24(dest, 0x210, v[118], src)
  for dst, src in [(142,9),(143,10),(144,11),(145,14),(146,15),(147,16),(148,17)]: k.emit(v_mad_u32_u24(v[dst], 0x210, v[118], v[src]))
  k.emit(s_mov_b32(s[7], 0))
  k.emit(s_cmp_gt_i32(s[4], 0))
  # Wait for all global loads before storing to LDS
  k.emit(s_waitcnt(simm16=0))
  # Store all initial tile data to LDS
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[23], data1=v[24], offset0=16, offset1=18))
  k.emit(ds_store_b32(addr=v[141], data0=v[29], offset0=0, offset1=0))
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[25], data1=v[26], offset0=20, offset1=22))
  k.emit(ds_store_b32(addr=v[142], data0=v[30], offset0=0, offset1=0))
  k.emit(ds_store_b32(addr=v[143], data0=v[31], offset0=0, offset1=0))
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[27], data1=v[28], offset0=24, offset1=26))
  k.emit(ds_store_b32(addr=v[144], data0=v[12], offset0=0, offset1=0))
  k.emit(ds_store_b32(addr=v[145], data0=v[13], offset0=0, offset1=0))
  k.emit(ds_store_2addr_stride64_b32(addr=v[8], data0=v[3], data1=v[4], offset0=28, offset1=30))
  k.emit(ds_store_b32(addr=v[146], data0=v[5], offset0=0, offset1=0))
  k.emit(ds_store_b32(addr=v[147], data0=v[6], offset0=0, offset1=0))
  k.emit(ds_store_b32(addr=v[148], data0=v[7], offset0=0, offset1=0))
  k.emit(s_waitcnt(simm16=64519))
  k.emit(s_barrier())
  k.emit(s_cbranch_scc1(simm16=0))
  k.branch_to('LBB0_3')
  k.emit(v_lshlrev_b32_e32(v[149], 2, v[118]))
  k.emit(v_lshlrev_b32_e32(v[150], 2, v[2]))
  k.emit(s_mov_b32(s[12], 0))
  k.emit(s_and_not1_b32(VCC_LO, EXEC_LO, s[7]))
  k.emit(s_cbranch_vccz(simm16=0))
  k.branch_to('LBB0_4')
  # Zero 128 accumulator vgprs (64 pairs)
  for vx, vy in ACC_ZERO_PAIRS: k.dual_mov_zero(vx, vy, 12)
  k.emit(s_branch(simm16=0))
  k.branch_to('LBB0_13')
  k.label('LBB0_3')
  k.emit(s_mov_b32(s[7], -1))
  k.label('LBB0_4')
  k.emit(s_ashr_i32(s[7], s[2], 31))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_LSHLREV_B32, vdstx=v[133], vdsty=v[2], srcx0=0, vsrcx1=v[0], srcy0=4, vsrcy1=v[2]))
  k.emit(s_lshr_b32(s[7], s[7], 25))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32, vdstx=v[124], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=s[7], vsrcy1=v[1]))
  k.emit(v_ashrrev_i32_e32(v[149], 31, v[119]))
  k.emit(v_ashrrev_i32_e32(v[150], 31, v[125]))
  k.emit(v_ashrrev_i32_e32(v[151], 31, v[130]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_AND_B32, vdstx=v[132], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=0x3fffff80, vsrcy1=v[3]))
  k.emit(v_ashrrev_i32_e32(v[152], 31, v[134]))
  k.emit(v_ashrrev_i32_e32(v[153], 31, v[137]))
  k.emit(v_ashrrev_i32_e32(v[154], 31, v[138]))
  k.emit(v_ashrrev_i32_e32(v[156], 31, v[139]))
  k.emit(v_sub_nc_u32_e32(v[3], v[1], v[3]))
  k.emit(v_ashrrev_i32_e32(v[157], 31, v[140]))
  k.emit(v_lshl_or_b32(v[166], v[118], 4, 0x1080))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[131], vdsty=v[110], srcx0=0, vsrcx1=v[0], srcy0=0, vsrcy1=v[0]))
  k.emit(v_lshl_add_u32(v[155], v[3], 2, 0x1080))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_LSHLREV_B32, vdstx=v[112], vdsty=v[3], srcx0=0, vsrcx1=v[0], srcy0=2, vsrcy1=v[0]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[113], vdsty=v[96], srcx0=0, vsrcx1=v[0], srcy0=0, vsrcy1=v[0]))
  # LDS A-tile address offsets: v[dst] = (shift_amt << 9) + v[155]
  for i, dst in enumerate([158, 159, 160, 161, 162, 163, 164]): k.emit(v_lshl_add_u32(v[dst], i+1, 9, v[155]))
  k.emit(v_and_or_b32(v[165], 0x180, v[3], v[2]))
  for vx, vy in ACC_ZERO_PAIRS_B: k.dual_mov_zero_imm(vx, vy)
  k.emit(v_mov_b32_e32(v[214], 0))
  k.emit(v_mov_b32_e32(v[5], 0))
  k.emit(v_mov_b32_e32(v[3], 0))
  k.emit(s_add_i32(s[7], s[4], -8))
  k.emit(s_add_u32(s[8], s[8], 32))
  k.emit(s_addc_u32(s[9], s[9], 0))
  k.emit(s_mov_b32(s[12], 0))
  k.emit(s_branch(simm16=0))
  k.branch_to('LBB0_6')

  # === MAIN GEMM LOOP: 8 iterations of FMAC with LDS loads and global prefetch ===
  k.label('LBB0_5')
  k.emit(s_add_i32(s[12], s[12], 8))
  k.emit(s_cmp_ge_i32(s[12], s[4]))
  k.emit(s_cbranch_scc1(simm16=0))
  k.branch_to('LBB0_12')
  k.label('LBB0_6')
  k.emit(s_cmp_lt_i32(s[12], s[7]))
  k.emit(s_cselect_b32(s[13], -1, 0))
  k.emit(s_cmp_ge_i32(s[12], s[7]))
  k.emit(s_cbranch_scc1(simm16=0))
  k.branch_to('LBB0_8')
  k.emit(v_add_nc_u32_e32(v[203], 0x20000, v[203]))
  k.emit(v_add_nc_u32_e32(v[215], 0x20, v[215]))
  k.emit(s_setprio(0))
  for vdst, slo in INIT_PREFETCH: k.emit(global_load_b32(vdst=v[vdst], addr=v[203], saddr=s[slo:slo+2]))
  k.label('LBB0_8')
  k.emit(v_mov_b32_e32(v[183], v[165]))
  k.emit(s_mov_b32(s[14], 0))
  k.emit(v_mov_b32_e32(v[202], v[166]))
  k.label('LBB0_9')
  for i in range(6): k.main_loop_iter(i)  # 6 iterations with prefetch
  k.main_loop_iter(None)  # iteration 7: no prefetch
  k.main_loop_iter(None)  # iteration 8: no prefetch
  k.emit(s_and_not1_b32(VCC_LO, EXEC_LO, s[13]))
  k.emit(s_waitcnt(simm16=1015))
  k.emit(s_barrier())
  k.emit(s_cbranch_vccnz(simm16=0))
  k.branch_to('LBB0_5')
  k.lds_store_prefetch()
  k.emit(s_waitcnt(simm16=64519))
  k.emit(s_barrier())
  k.emit(s_branch(simm16=0))
  k.branch_to('LBB0_5')
  k.label('LBB0_12')
  # Register permutation for output layout (122 swaps)
  for a, b in PERMUTE_SWAPS: k.vswap(a, b)
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[149], vdsty=v[150], srcx0=v[135], vsrcx1=v[0], srcy0=v[136], vsrcy1=v[0]))

  # === EPILOGUE: Scale and store accumulated results ===
  k.label('LBB0_13')
  # Setup: compute column base (v[118]) and row base (v[119])
  k.emit(v_and_b32_e32(v[0], 0x60, v[0]))  # Extract bits for row offset
  k.emit(v_or_b32_e32(v[118], s[2], v[149]))  # col_base = block_col | lane_col
  k.emit(v_add_nc_u32_e32(v[0], s[3], v[0]))  # Add block_row offset
  k.emit(v_or_b32_e32(v[119], v[0], v[150]))  # row_base = (block_row + offset) | lane_row
  # Compute row multipliers for rows 0-3: v[144+i] = (v[119] + i) * s[4]
  k.emit(v_mul_lo_u32(v[144], v[119], s[4]))
  k.emit(v_add_nc_u32_e32(v[145], s[4], v[144]))
  k.emit(v_add_nc_u32_e32(v[146], s[4], v[145]))
  k.emit(v_add_nc_u32_e32(v[147], s[4], v[146]))
  # Compute row multipliers for rows 16-19: v[148+i] = (v[119] | (16+i)) * s[4]
  k.emit(v_or_b32_e32(v[1], 16, v[119]))
  k.emit(v_mul_lo_u32(v[148], v[1], s[4]))
  k.emit(v_add_nc_u32_e32(v[149], s[4], v[148]))
  k.emit(v_add_nc_u32_e32(v[150], s[4], v[149]))
  k.emit(v_add_nc_u32_e32(v[151], s[4], v[150]))
  # Store all 32 groups of 4 floats
  for col_off, row_idx, data_base, srcs in EPILOGUE_STORES:
    k.epilogue_mul_store(col_off, row_idx, data_base, srcs)
  # End kernel
  k.emit(s_nop(0))
  k.emit(s_sendmsg(simm16=3))
  k.emit(s_endpgm())

  return k.to_asm()
