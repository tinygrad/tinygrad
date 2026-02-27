# RDNA2 pcode - mapped from RDNA3 pcode by instruction name normalization
# RDNA2 and RDNA3 share identical instruction semantics, only naming differs
# ruff: noqa: E501,F401
from tinygrad.runtime.autogen.amd.rdna2.enum import *
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE as _R3

# Build RDNA3 name -> pcode lookup (strip _E32/_E64 suffixes)
_r3 = {op.name.replace("_E32","").replace("_E64",""): p for op, p in _R3.items()}

# RDNA2 -> RDNA3 instruction name renames (applied sequentially, first match per pair wins)
_RENAMES = [
  # DS operations (2-variant and base)
  ("DS_WRITE2ST64_", "DS_STORE2ST64_"), ("DS_WRITE2_", "DS_STORE2_"), ("DS_WRITE_", "DS_STORE_"),
  ("DS_READ2ST64_", "DS_LOAD2ST64_"), ("DS_READ2_", "DS_LOAD2_"), ("DS_READ_", "DS_LOAD_"),
  ("DS_CMPST_", "DS_CMPSTORE_"),
  ("DS_WRXCHG2ST64_", "DS_STOREXCHG2ST64_"), ("DS_WRXCHG2_", "DS_STOREXCHG2_"), ("DS_WRXCHG_", "DS_STOREXCHG_"),
  # VOP renames
  ("V_FFBH_U32", "V_CLZ_I32_U32"), ("V_FFBH_I32", "V_CLS_I32"), ("V_FFBL_B32", "V_CTZ_I32_B32"),
  ("V_CVT_FLR_I32_F32", "V_CVT_FLOOR_I32_F32"), ("V_CVT_RPI_I32_F32", "V_CVT_NEAREST_I32_F32"),
  ("V_MUL_LEGACY_F32", "V_MUL_DX9_ZERO_F32"), ("V_FMAC_LEGACY_F32", "V_FMAC_DX9_ZERO_F32"),
  ("V_CVT_PKRTZ_F16_F32", "V_CVT_PK_RTZ_F16_F32"), ("V_DOT2C_F32_F16", "V_DOT2ACC_F32_F16"),
  ("V_CMP_TRU_", "V_CMP_T_"), ("V_CMPX_TRU_", "V_CMPX_T_"),
  # SOP renames
  ("S_ANDN2_B", "S_AND_NOT1_B"), ("S_ORN2_B", "S_OR_NOT1_B"),
  # Memory size suffixes (longer first to avoid partial match)
  ("_DWORDX16", "_B512"), ("_DWORDX8", "_B256"), ("_DWORDX4", "_B128"), ("_DWORDX3", "_B96"), ("_DWORDX2", "_B64"), ("_DWORD", "_B32"),
  ("_UBYTE_D16_HI", "_D16_HI_U8"), ("_UBYTE_D16", "_D16_U8"), ("_SBYTE_D16_HI", "_D16_HI_I8"), ("_SBYTE_D16", "_D16_I8"),
  ("_SHORT_D16_HI", "_D16_HI_B16"), ("_SHORT_D16", "_D16_B16"),
  ("_UBYTE", "_U8"), ("_SBYTE", "_I8"), ("_USHORT", "_U16"), ("_SSHORT", "_I16"), ("_SHORT", "_B16"), ("_BYTE", "_B8"),
  # Atomic type suffixes (_X2 before bare, longer before shorter)
  ("_FCMPSWAP_X2", "_FCMPSWAP_X2"), ("_CMPSWAP_X2", "_CMPSWAP_B64"), ("_SUB_X2", "_SUB_U64"),
  ("_ADD_X2", "_ADD_U64"), ("_AND_X2", "_AND_B64"), ("_OR_X2", "_OR_B64"), ("_XOR_X2", "_XOR_B64"),
  ("_SWAP_X2", "_SWAP_B64"), ("_SMIN_X2", "_SMIN_I64"), ("_SMAX_X2", "_SMAX_I64"),
  ("_UMIN_X2", "_UMIN_U64"), ("_UMAX_X2", "_UMAX_U64"), ("_INC_X2", "_INC_U64"), ("_DEC_X2", "_DEC_U64"),
  ("_CMPSWAP", "_CMPSWAP_B32"), ("_SWAP", "_SWAP_B32"),
  ("_SUB", "_SUB_U32"), ("_ADD", "_ADD_U32"), ("_AND", "_AND_B32"), ("_OR", "_OR_B32"), ("_XOR", "_XOR_B32"),
  ("_SMIN", "_SMIN_I32"), ("_SMAX", "_SMAX_I32"), ("_UMIN", "_UMIN_U32"), ("_UMAX", "_UMAX_U32"),
  ("_INC", "_INC_U32"), ("_DEC", "_DEC_U32"),
  # D16 store renames
  ("_STORE_BYTE_D16_HI", "_STORE_D16_HI_B8"),
]

def _normalize(name: str) -> str:
  for old, new in _RENAMES:
    if old in name: return name.replace(old, new, 1)
  return name

# Build RDNA2 PCODE by mapping enum members to RDNA3 pcode via normalized names
PCODE = {}
for _name in dir():
  _obj = eval(_name)
  if isinstance(_obj, type) and _name.endswith("Op") and issubclass(_obj, ReprEnum):
    for _m in _obj:
      _n = _m.name.replace("_E32", "").replace("_E64", "")
      if _n in _r3: PCODE[_m] = _r3[_n]
      else:
        _nn = _normalize(_n)
        if _nn in _r3: PCODE[_m] = _r3[_nn]
