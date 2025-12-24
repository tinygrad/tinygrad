# https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf
# my goal is to write the most beautiful DSL for expressing this manual in Python
# it should be a disassembler and assembler and only require writing code that looks like the manual
import functools
from extra.assembly.rdna3.autogen_rdna3_enum import SOP2Op, SOP1Op, SMEMOp, VOP2Op, SrcEnum

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

class Imm: pass
class SImm: pass

class Inst:
  def to_int(self): pass
  def from_int(self, i): pass

class Registers: pass
class Inst32(Inst): pass
class Inst64(Inst): pass

class SGPR(Registers):
  # TODO: something here that works and typechecks as a valid SSrc
  # these are the 106 real and 128 total scalar regs in RDNA3
  pass

class TTMP(Registers):
  # 16 TTMP regs
  pass

class VGPR(Registers):
  # same, but the 256 vector regs in RDNA3
  pass

class SSrc(SrcEnum):
  # 0–105: SGPR0..SGPR105
  for i in range(106): locals()[f"SGPR{i}"] = i

  # 108–123: TTMP0..TTMP15
  for i in range(16): locals()[f"TTMP{i}"] = 108 + i

  # 256–511: VGPR0..VGPR255
  for i in range(256): locals()[f"VGPR{i}"] = 256 + i

# *** instruction definitions ***

class SOP2(Inst32):
  encoding   = bits[31:30] == 0b10
  op:SOP2Op  = bits[29:23]
  sdst:SGPR  = bits[22:16]
  ssrc1:SSrc = bits[15:8]
  ssrc0:SSrc = bits[7:0]

class SOP1(Inst32):
  encoding   = bits[31:23] == 0b10_1111101
  sdst       = bits[22:16]
  op:SOP1Op  = bits[15:8]
  ssrc0      = bits[7:0]

class SOPP(Inst32):
  encoding      = bits[31:23] == 0b10_1111111
  op            = bits[22:16]
  simm16:SImm   = bits[15:0]

class VOP2(Inst32):
  encoding   = bits[31] == 0b0
  op:VOP2Op  = bits[30:25]
  vdst:VGPR  = bits[24:17]
  vsrc1:VGPR = bits[16:9]
  src0:VSrc  = bits[8:0]

class SMEM(Inst64):
  encoding     = bits[31:26] == 0b111101
  op:SMEMOp    = bits[25:18]
  glc          = bits[15]
  dlc          = bits[14]
  sdata:SGPR   = bits[12:6]
  sbase        = bits[5:0]  # this is SGPR shifted over 1
  soffset:SGPR = bits[63:57]
  offset:Imm   = bits[52:32]

# s_load_b128 s[4:7], s[0:1], null

s_load_b128 = functools.partial(SMEM, SMEMOp.S_LOAD_B128)
v_and_b32_e32 = functools.partial(VOP2, VOP2Op.V_AND_B32)
s = SGPR  # assembly-style alias
v = VGPR  # assembly-style alias

NULL = SSrc.NULL

if __name__ == "__main__":
  # assembler
  inst = SOP2(SOP2Op.S_ADD_U32, s[3], s[2], s[1])
  word = inst.to_int()

  # s_load_b128 s[4:7], s[0:1], null
  instm = SMEM(SMEMOp.S_LOAD_B128, s[4:7], s[0:1])

  # disassembler
  inst = SOP2.from_int(word)
  print(inst)

  s_load_b128(s[4:7], s[0:1], NULL)
  v_and_b32_e32(v[0], 0x3FF, v[0])

  # some of these are wrong
  program = [
    VOP3(VOP3Op.V_BFE_U32, v[1], v[0], 10, 10),
    SMEM(SMEMOp.S_LOAD_B128, s[4:7], s[0:1], SSrc.NULL),
    VOP2(VOP2Op.V_AND_B32, v[0], 0x3FF, v[0]),
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
    s_load_b128(s[4:7], s[0:1], SrcEnum.NULL),
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
