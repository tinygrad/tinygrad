# Instruction format detection and decoding
from __future__ import annotations
from extra.assembly.amd.dsl import Inst, FixedBitField
from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP1_SDST, VOP2, VOP3, VOP3_SDST, VOP3SD, VOP3P, VOPC, VOPD, VINTERP,
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, MUBUF, MTBUF, MIMG, EXP)
from extra.assembly.amd.autogen.rdna4.ins import (VOP1 as R4_VOP1, VOP1_SDST as R4_VOP1_SDST, VOP2 as R4_VOP2,
  VOP3 as R4_VOP3, VOP3_SDST as R4_VOP3_SDST, VOP3SD as R4_VOP3SD, VOP3P as R4_VOP3P,
  VOPC as R4_VOPC, VOPD as R4_VOPD, VINTERP as R4_VINTERP, SOP1 as R4_SOP1, SOP2 as R4_SOP2, SOPC as R4_SOPC, SOPK as R4_SOPK, SOPP as R4_SOPP,
  SMEM as R4_SMEM, DS as R4_DS, VBUFFER as R4_VBUFFER, VEXPORT as R4_VEXPORT)
from extra.assembly.amd.autogen.cdna.ins import (VOP1 as C_VOP1, VOP2 as C_VOP2, VOPC as C_VOPC, VOP3A, VOP3B, VOP3P as C_VOP3P,
  SOP1 as C_SOP1, SOP2 as C_SOP2, SOPC as C_SOPC, SOPK as C_SOPK, SOPP as C_SOPP, SMEM as C_SMEM, DS as C_DS,
  FLAT as C_FLAT, MUBUF as C_MUBUF, MTBUF as C_MTBUF, SDWA, DPP)

def _matches_encoding(word: int, cls: type[Inst]) -> bool:
  """Check if word matches the encoding pattern of an instruction class."""
  enc = next(((n, f) for n, f in cls._fields if isinstance(f, FixedBitField) and n == 'encoding'), None)
  if enc is None: return False
  bf = enc[1]
  return ((word >> bf.lo) & bf.mask()) == bf.default

# Order matters: more specific encodings first, VOP2 last (it's a catch-all for bit31=0)
_RDNA_FORMATS_64 = [VOPD, VOP3P, VINTERP, VOP3, DS, FLAT, MUBUF, MTBUF, MIMG, SMEM, EXP]
_RDNA_FORMATS_32 = [SOP1, SOPC, SOPP, SOPK, VOPC, VOP1, SOP2, VOP2]  # SOP2/VOP2 are catch-alls
_CDNA_FORMATS_64 = [C_VOP3P, VOP3A, C_DS, C_FLAT, C_MUBUF, C_MTBUF, C_SMEM]
_CDNA_FORMATS_32 = [SDWA, DPP, C_SOP1, C_SOPC, C_SOPP, C_SOPK, C_VOPC, C_VOP1, C_SOP2, C_VOP2]
_CDNA_VOP3B_OPS = {281, 282, 283, 284, 285, 286, 480, 481, 488, 489}  # VOP3B opcodes
_RDNA4_FORMATS_64 = [R4_VOPD, R4_VOP3P, R4_VINTERP, R4_VOP3, R4_DS, R4_VBUFFER, R4_SMEM, R4_VEXPORT]
_RDNA4_FORMATS_32 = [R4_SOP1, R4_SOPC, R4_SOPP, R4_SOPK, R4_VOPC, R4_VOP1, R4_SOP2, R4_VOP2]
_RDNA4_VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
_RDNA3_VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
# Instructions with SGPR destination (READLANE, READFIRSTLANE, and VOP3-encoded VOPC)
_VOP1_SDST_OPS = {2}  # V_READFIRSTLANE_B32_E32
_VOP3_SDST_OPS = {386, 864}  # V_READFIRSTLANE_B32_E64, V_READLANE_B32 (V_WRITELANE_B32=865 writes to VGPR)
# VOP3-encoded VOPC instructions (opcodes < 256) also have SGPR destination

def detect_format(data: bytes, arch: str = "rdna3") -> type[Inst]:
  """Detect instruction format from machine code bytes."""
  assert len(data) >= 4, f"need at least 4 bytes, got {len(data)}"
  word = int.from_bytes(data[:4], 'little')
  if arch == "cdna":
    if (word >> 30) == 0b11:
      for cls in _CDNA_FORMATS_64:
        if _matches_encoding(word, cls):
          return VOP3B if cls is VOP3A and ((word >> 16) & 0x3ff) in _CDNA_VOP3B_OPS else cls
      raise ValueError(f"unknown CDNA 64-bit format word={word:#010x}")
    for cls in _CDNA_FORMATS_32:
      if _matches_encoding(word, cls): return cls
    raise ValueError(f"unknown CDNA 32-bit format word={word:#010x}")
  if arch == "rdna4":
    if (word >> 30) == 0b11:
      for cls in _RDNA4_FORMATS_64:
        if _matches_encoding(word, cls):
          if cls is R4_VOP3:
            opcode = (word >> 16) & 0x3ff
            if opcode in _RDNA4_VOP3SD_OPS: return R4_VOP3SD
            if opcode in _VOP3_SDST_OPS or opcode < 256: return R4_VOP3_SDST  # VOP3-encoded VOPC (op < 256) writes to SGPR
          return cls
      raise ValueError(f"unknown RDNA4 64-bit format word={word:#010x}")
    for cls in _RDNA4_FORMATS_32:
      if _matches_encoding(word, cls):
        if cls is R4_VOP1 and ((word >> 9) & 0xff) in _VOP1_SDST_OPS: return R4_VOP1_SDST
        return cls
    raise ValueError(f"unknown RDNA4 32-bit format word={word:#010x}")
  # RDNA3 (default)
  if (word >> 30) == 0b11:
    for cls in _RDNA_FORMATS_64:
      if _matches_encoding(word, cls):
        if cls is VOP3:
          opcode = (word >> 16) & 0x3ff
          if opcode in _RDNA3_VOP3SD_OPS: return VOP3SD
          if opcode in _VOP3_SDST_OPS or opcode < 256: return VOP3_SDST  # VOP3-encoded VOPC (op < 256) writes to SGPR
        return cls
    raise ValueError(f"unknown 64-bit format word={word:#010x}")
  for cls in _RDNA_FORMATS_32:
    if _matches_encoding(word, cls):
      if cls is VOP1 and ((word >> 9) & 0xff) in _VOP1_SDST_OPS: return VOP1_SDST
      return cls
  raise ValueError(f"unknown 32-bit format word={word:#010x}")

def decode_inst(data: bytes, arch: str = "rdna3") -> Inst:
  """Decode machine code bytes into an instruction."""
  return detect_format(data, arch).from_bytes(data)
