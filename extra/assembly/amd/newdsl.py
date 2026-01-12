import functools
from extra.assembly.amd.autogen.rdna3.enum import VOP1Op

class Reg:
  pass

class RegFactory:
  pass

class SGPR(Reg): pass
class VGPR(Reg): pass
class TTMP(Reg): pass

# TODO: these may vary per ISA
s: RegFactory[SGPR] = RegFactory(SGPR, "SGPR", sz=128)
v: RegFactory[VGPR] = RegFactory(VGPR, "VGPR", sz=256)

# Scalar 0-105=SGPR; 106,107=VCC, 108-123=TTMP0-15, and 124-127={NULL, M0, EXEC_LO, EXEC_HI}.
VCC = s[106:107]
ttmp = s[108:123]
NULL = s[124]
M0 = s[125]
EXEC_LO = s[126]
EXEC_HI = s[127]

class BitField:
  required_size = None
  def __init__(self, hi:int, lo:int):
    self.hi, self.lo = hi, lo
    if self.required_size is not None and hi-lo != self.required_size: raise RuntimeError("wrong size field")
  # add __eq__ to return const bit field
  # add methods here for encoding/decoding

class OpField(BitField):
  # TODO: this should work with ints
  pass

class VGPRField(BitField):
  required_size = 8

class SrcField(BitField):
  required_size = 9
  # repr should print v[6], s[6], 1.0

# rewrite enum.py to generate with BitFields. anything else we need

class VOP1Op(OpField):
  V_NOP = 0
  V_MOV_B32_E32 = 1

class VOP2Op(OpField):
  V_CNDMASK_B32_E32 = 1

class VOP3Op(OpField):
  V_MOV_B32_E64 = 385

assert VOP1Op.V_MOV_B32_E32 != VOP2Op.V_CNDMASK_B32_E32

class Inst:
  # TODO: write init to make things work. the constructor should convert the VGPRs into the fields based on the order of the things
  pass
  # add from_bytes and to_bytes

# define class like this

class VOP1(Inst):
  encoding = BitField(31,25) == 0b0111111
  op       = VOP1Op(16,9)
  vdst     = VGPRField(24,17)
  src0     = SrcField(8,0)

# define instruction like this

v_mov_b32_e32 = functools.partial(VOP1, VOP1Op.V_MOV_B32_E32)

i1 = v_mov_b32_e32(v[5], v[6])
i2 = v_mov_b32_e32(v[5], s[6])
i3 = v_mov_b32_e32(v[5], 1.0)  # should encode as const

assert repr(i1) == "v_mov_b32_e32(v[5], v[6])"
assert repr(i2) == "v_mov_b32_e32(v[5], s[6])"
assert repr(i3) == "v_mov_b32_e32(v[5], 1.0)"
