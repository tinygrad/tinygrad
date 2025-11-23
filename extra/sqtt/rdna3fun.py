# https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf
# my goal is to write the most beautiful DSL for expressing this manual in Python
# it should be a disassembler and assembler and only require writing code that looks like the manual
from enum import IntEnum

# *** supporting code ***

class BitField:
  def __init__(self, hi, lo, fixed=None):
    self.hi, self.lo

  def enum(self, enum):
    # TODO: return object with this enum registered
    return self

  # TODO: add eq support

  def __set_name__(self, owner, name):
    print(owner, name)

class _FieldFactory:
  def __getitem__(self, sl: slice):
    # TODO: make bits[n] work
    return BitField(sl.start, sl.stop)
bits = _FieldFactory()

class int16: pass

class Inst:
  def to_int(self): pass
  def from_int(self, i): pass

class Registers: pass
class Inst32(Inst): pass
class Inst64(Inst): pass

# *** enum definitions ***

class SOP2Op(IntEnum):
  """Table 66. SOP2 Opcodes"""

  # Integer add/sub
  S_ADD_U32 = 0; S_SUB_U32 = 1; S_ADD_I32 = 2; S_SUB_I32 = 3
  S_ADDC_U32 = 4; S_SUBB_U32 = 5; S_ABSDIFF_I32 = 6

  # Shifts / shift-add
  S_LSHL_B32 = 8; S_LSHL_B64 = 9; S_LSHR_B32 = 10; S_LSHR_B64 = 11
  S_ASHR_I32 = 12; S_ASHR_I64 = 13
  S_LSHL1_ADD_U32 = 14; S_LSHL2_ADD_U32 = 15
  S_LSHL3_ADD_U32 = 16; S_LSHL4_ADD_U32 = 17

  # Integer min/max
  S_MIN_I32 = 18; S_MIN_U32 = 19; S_MAX_I32 = 20; S_MAX_U32 = 21

  # Bitwise logic
  S_AND_B32 = 22; S_AND_B64 = 23; S_OR_B32 = 24; S_OR_B64 = 25
  S_XOR_B32 = 26; S_XOR_B64 = 27; S_NAND_B32 = 28; S_NAND_B64 = 29
  S_NOR_B32 = 30; S_NOR_B64 = 31; S_XNOR_B32 = 32; S_XNOR_B64 = 33
  S_AND_NOT1_B32 = 34; S_AND_NOT1_B64 = 35
  S_OR_NOT1_B32 = 36; S_OR_NOT1_B64 = 37

  # Bitfield extract / mask
  S_BFE_U32 = 38; S_BFE_I32 = 39; S_BFE_U64 = 40; S_BFE_I64 = 41
  S_BFM_B32 = 42; S_BFM_B64 = 43

  # Integer multiply
  S_MUL_I32 = 44; S_MUL_HI_U32 = 45; S_MUL_HI_I32 = 46

  # Conditional select
  S_CSELECT_B32 = 48; S_CSELECT_B64 = 49

  # Pack
  S_PACK_LL_B32_B16 = 50; S_PACK_LH_B32_B16 = 51
  S_PACK_HH_B32_B16 = 52; S_PACK_HL_B32_B16 = 53

  # Scalar float32
  S_ADD_F32 = 64; S_SUB_F32 = 65; S_MIN_F32 = 66; S_MAX_F32 = 67
  S_MUL_F32 = 68; S_FMAAK_F32 = 69; S_FMAMK_F32 = 70; S_FMAC_F32 = 71
  S_CVT_PK_RTZ_F16_F32 = 72

  # Scalar float16
  S_ADD_F16 = 73; S_SUB_F16 = 74; S_MIN_F16 = 75; S_MAX_F16 = 76
  S_MUL_F16 = 77; S_FMAC_F16 = 78

class SMEMOp(IntEnum):
  """Table 76. SMEM Opcodes"""

  S_LOAD_B32 = 0; S_LOAD_B64 = 1; S_LOAD_B128 = 2
  S_LOAD_B256 = 3; S_LOAD_B512 = 4

  S_BUFFER_LOAD_B32 = 8; S_BUFFER_LOAD_B64 = 9; S_BUFFER_LOAD_B128 = 10
  S_BUFFER_LOAD_B256 = 11; S_BUFFER_LOAD_B512 = 12

  S_GL1_INV = 32; S_DCACHE_INV = 33

class SGPR(Registers):
  # TODO: something here that works and typechecks as a valid SSrc
  # these are the 106 scalar regs in RDNA3
  pass

class VGPR(Registers):
  # same, but the 256 vector regs in RDNA3
  pass

class TTMP(Registers):
  # 16 TTMP regs
  pass

class SSrc(IntEnum):
  """Table 65. SOP2 Fields: SSRC encoding"""

  # 0–105: SGPR0..SGPR105
  for i in range(106): locals()[f"SGPR{i}"] = i

  VCC_LO = 106; VCC_HI = 107

  # 108–123: TTMP0..TTMP15
  for i in range(16): locals()[f"TTMP{i}"] = 108 + i

  NULL    = 124
  M0      = 125
  EXEC_LO = 126; EXEC_HI = 127

  ZERO = 128  # 0
  # 129–192: +1..+64
  # 193–208: -1..-16

  SHARED_BASE   = 235; SHARED_LIMIT  = 236
  PRIVATE_BASE  = 237; PRIVATE_LIMIT = 238

  POS_HALF = 240; NEG_HALF = 241
  POS_ONE  = 242; NEG_ONE  = 243
  POS_TWO  = 244; NEG_TWO  = 245
  POS_FOUR = 246; NEG_FOUR = 247

  INV_2PI = 248
  SCC     = 253
  LITERAL = 255

# *** instruction definitions ***

class SOP2(Inst32):
  encoding   = bits[31:30] == 0b10
  op:SOP2Op  = bits[29:23]
  sdst:SSrc  = bits[22:16]
  ssrc1:SSrc = bits[15:8]
  ssrc0:SSrc = bits[7:0]

class SMEM(Inst64):
  encoding  = bits[31:26] == 0b111101
  op:SMEMOp = bits[25:18]
  glc       = bits[15]
  dlc       = bits[14]
  sdata     = bits[12:6]
  sbase     = bits[5:0]
  # extension
  soffset   = bits[63:57]
  offset    = bits[52:32]

# s_load_b128 s[4:7], s[0:1], null

if __name__ == "__main__":
  s = SGPR  # assembly-style alias
  v = VGPR  # assembly-style alias

  # assembler
  inst = SOP2(SOP2Op.S_ADD_U32, s[3], s[2], s[1])
  word = inst.to_int()

  # s_load_b128 s[4:7], s[0:1], null
  instm = SMEM(SMEMOp.S_LOAD_B128, s[4:7], s[0:1])

  # disassembler
  inst = SOP2.from_int(word)
  print(inst)

  program = [
    VOP3(VOP3Op.V_BFE_U32, v[1], v[0], 10, 10),
    SMEM(SMEMOp.S_LOAD_B128, s[4:7], s[0:1], SSrc.NULL),
    VOP2(VOP2Op.V_AND_B32_E32, v[0], 0x3FF, v[0]),
    SOP1(SOP1Op.S_MULK_I32, s[3], 0x87),
    VOP3(VOP3Op.V_MAD_U64_U32, v[1:2], SSrc.NULL, s[2], 3, v[1:2]),
    VOP2(VOP2Op.V_MUL_U32_U24_E32, v[0], 45, v[0]),
    VOP2(VOP2Op.V_ASHRREV_I32_E32, v[2], 31, v[1]),
    VOP3(VOP3Op.V_ADD3_U32, v[0], v[0], s[3], v[1]),
    VOP3(VOP3Op.V_LSHLREV_B64, v[2:3], 2, v[1:2]),
    VOP2(VOP2Op.V_ASHRREV_I32_E32, v[1], 31, v[0]),
    VOP3(VOP3Op.V_LSHLREV_B64, v[0:1], 2, v[0:1]),
    SOPP(SOPPOp.S_WAITCNT_LGKMCNT, 0),
    VOP3(VOP3Op.V_ADD_CO_U32, v[2], SSrc.VCC_LO, s[6], v[2]),
    VOP3(VOP3Op.V_ADD_CO_CI_U32_E32, v[3], SSrc.VCC_LO, s[7], v[3], SSrc.VCC_LO),
    VOP3(VOP3Op.V_ADD_CO_U32, v[0], SSrc.VCC_LO, s[4], v[0]),
    FLAT(FLATOp.GLOBAL_LOAD_B32, v[2], v[2:3], OFF),
    VOP3(VOP3Op.V_ADD_CO_CI_U32_E32, v[1], SSrc.VCC_LO, s[5], v[1], SSrc.VCC_LO),
    SOPP(SOPPOp.S_WAITCNT_VMCNT, 0),
    FLAT(FLATOp.GLOBAL_STORE_B32, v[0:1], v[2], OFF),
    SOPP(SOPPOp.S_ENDPGM),
  ]

  # autogen helpers allow this as equiv

  program = [
    v_bfe_u32(v[1], v[0], 10, 10),
    s_load_b128(s[4:7], s[0:1], NULL),
    v_and_b32_e32(v[0], 0x3FF, v[0]),
    s_mulk_i32(s[3], 0x87),
    v_mad_u64_u32(v[1:2], NULL, s[2], 3, v[1:2]),
    v_mul_u32_u24_e32(v[0], 45, v[0]),
    v_ashrrev_i32_e32(v[2], 31, v[1]),
    v_add3_u32(v[0], v[0], s[3], v[1]),
    v_lshlrev_b64(v[2:3], 2, v[1:2]),
    v_ashrrev_i32_e32(v[1], 31, v[0]),
    v_lshlrev_b64(v[0:1], 2, v[0:1]),
    s_waitcnt_lgkmcnt(0),
    v_add_co_u32(v[2], VCC_LO, s[6], v[2]),
    v_add_co_ci_u32_e32(v[3], VCC_LO, s[7], v[3], VCC_LO),
    v_add_co_u32(v[0], VCC_LO, s[4], v[0]),
    global_load_b32(v[2], v[2:3], OFF),
    v_add_co_ci_u32_e32(v[1], VCC_LO, s[5], v[1], VCC_LO),
    s_waitcnt_vmcnt(0),
    global_store_b32(v[0:1], v[2], OFF),
    s_endpgm(),
  ]

# *** more for later ***

class SOP1Op(IntEnum):
  """Table 70. SOP1 Opcodes"""

  # Move / conditional move
  S_MOV_B32 = 0; S_MOV_B64 = 1; S_CMOV_B32 = 2; S_CMOV_B64 = 3
  S_BREV_B32 = 4; S_BREV_B64 = 5

  # Count / leading / trailing / sign-extend
  S_CTZ_I32_B32 = 8; S_CTZ_I32_B64 = 9
  S_CLZ_I32_U32 = 10; S_CLZ_I32_U64 = 11
  S_CLS_I32 = 12; S_CLS_I32_I64 = 13
  S_SEXT_I32_I8 = 14; S_SEXT_I32_I16 = 15

  # Bit set / replicate
  S_BITSET0_B32 = 16; S_BITSET0_B64 = 17
  S_BITSET1_B32 = 18; S_BITSET1_B64 = 19
  S_BITREPLICATE_B64_B32 = 20

  # Integer abs
  S_ABS_I32 = 21

  # Bit counts
  S_BCNT0_I32_B32 = 22; S_BCNT0_I32_B64 = 23
  S_BCNT1_I32_B32 = 24; S_BCNT1_I32_B64 = 25

  # Quad / WQM / NOT
  S_QUADMASK_B32 = 26; S_QUADMASK_B64 = 27
  S_WQM_B32 = 28; S_WQM_B64 = 29
  S_NOT_B32 = 30; S_NOT_B64 = 31

  # Exec mask logic (SAVEEXEC)
  S_AND_SAVEEXEC_B32 = 32; S_AND_SAVEEXEC_B64 = 33
  S_OR_SAVEEXEC_B32 = 34; S_OR_SAVEEXEC_B64 = 35
  S_XOR_SAVEEXEC_B32 = 36; S_XOR_SAVEEXEC_B64 = 37
  S_NAND_SAVEEXEC_B32 = 38; S_NAND_SAVEEXEC_B64 = 39
  S_NOR_SAVEEXEC_B32 = 40; S_NOR_SAVEEXEC_B64 = 41
  S_XNOR_SAVEEXEC_B32 = 42; S_XNOR_SAVEEXEC_B64 = 43
  S_AND_NOT0_SAVEEXEC_B32 = 44; S_AND_NOT0_SAVEEXEC_B64 = 45
  S_OR_NOT0_SAVEEXEC_B32 = 46; S_OR_NOT0_SAVEEXEC_B64 = 47
  S_AND_NOT1_SAVEEXEC_B32 = 48; S_AND_NOT1_SAVEEXEC_B64 = 49
  S_OR_NOT1_SAVEEXEC_B32 = 50; S_OR_NOT1_SAVEEXEC_B64 = 51

  # Exec mask logic (WREXEC)
  S_AND_NOT0_WREXEC_B32 = 52; S_AND_NOT0_WREXEC_B64 = 53
  S_AND_NOT1_WREXEC_B32 = 54; S_AND_NOT1_WREXEC_B64 = 55

  # MOVREL* / PC / MSG
  S_MOVRELS_B32 = 64; S_MOVRELS_B64 = 65
  S_MOVRELD_B32 = 66; S_MOVRELD_B64 = 67
  S_MOVRELSD_2_B32 = 68
  S_GETPC_B64 = 71; S_SETPC_B64 = 72; S_SWAPPC_B64 = 73; S_RFE_B64 = 74
  S_SENDMSG_RTN_B32 = 76; S_SENDMSG_RTN_B64 = 77

  # Scalar float33
  S_CEIL_F32 = 96; S_FLOOR_F32 = 97; S_TRUNC_F32 = 98; S_RNDNE_F32 = 99
  S_CVT_F32_I32 = 100; S_CVT_F32_U32 = 101; S_CVT_I32_F32 = 102; S_CVT_U32_F32 = 103
  S_CVT_F16_F32 = 104; S_CVT_F32_F16 = 105; S_CVT_HI_F32_F16 = 106

  # Scalar float16
  S_CEIL_F16 = 107; S_FLOOR_F16 = 108; S_TRUNC_F16 = 109; S_RNDNE_F16 = 110



class SOP1(Inst32):
  encoding   = bits[31:23] == 0b10_1111101
  sdst       = bits[22:16]
  op:SOP1Op  = bits[15:8]
  ssrc0      = bits[7:0]

class SOPP(Inst32):
  encoding      = bits[31:23] == 0b10_1111111
  op            = bits[22:16]
  simm16:int16  = bits[15:0]
