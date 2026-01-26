import unittest
from tinygrad.renderer.x86 import X86Renderer, RBP, RDI, RSP, RSI, RAX, RDX, XMM, GPR, Register, imm
from tinygrad.uop import X86Ops, Ops
from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import SPEC

@unittest.skipIf(SPEC > 1, "x86 spec not supported in full_spec")
class TestEncodingsX86(unittest.TestCase):
  # NOTE: x86 supports a single displacement as memory address and index without base memory address
  # these have no use cases so they aren't supported
  def reg(self, dt:DType, reg:Register): return UOp(X86Ops.DEFINE_REG, dt, arg=reg)
  def encode(self, u:UOp): return X86Renderer().render([u], lower=False)

  # displacement of 0 isn't emitted
  def test_base_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RDI), UOp(Ops.NOOP), imm(dtypes.int8, 0)), RDI)
    # mov edi, dword ptr [rdi]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 3F"))

  # rsp/r12 require a sib byte when used as base memory address
  def test_rsp_base_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RSP), UOp(Ops.NOOP), imm(dtypes.int8, 0)), RSP)
    # mov esp, dword ptr [rsp]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 24 24"))

  # rbp/r13 require a displacement when used as base memory address
  def test_rbp_base_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RBP), UOp(Ops.NOOP), imm(dtypes.int8, 0)), RBP)
    # mov ebp, dword ptr [rbp + 0]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 6D 00"))

  # test [base + index*scale]
  def test_base_index_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RAX), self.reg(dtypes.int32, RDX), imm(dtypes.int8, 0)), RAX)
    # mov eax, dword ptr [rax + rdx*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 04 90"))

  # rsp as index means no index
  def test_rsp_index_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RAX), self.reg(dtypes.int32, RSP), imm(dtypes.int8, 0)), RAX)
    # mov eax, dword ptr [rax]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 00"))

  # however r12 is a valid index
  def test_r12_index_address(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RAX), self.reg(dtypes.int32, GPR[12]), imm(dtypes.int8, 0)), RAX)
    # mov eax, dword ptr [rax + r12*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("42 8B 04 A0"))

  # test [base + index*scale + 8bit disp]
  def test_complex_address_8bit_disp(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RDI), self.reg(dtypes.int32, RSI), imm(dtypes.int8, 10)), RDI)
    # mov edi, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 7C B7 0A"))

  # test [base + index*scale + 32bit disp]
  def test_complex_address_32bit_disp(self):
    load = UOp(X86Ops.MOV, dtypes.int32, (self.reg(dtypes.int32.ptr(), RDI), self.reg(dtypes.int32, RSI), imm(dtypes.int32, 10000)), RDI)
    # mov edi, dword ptr [rdi + rsi*4 + 0x2710]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B BC B7 10 27 00 00"))

  # 8bit variants of legacy instructions subtract 1 from opcode
  def test_8bit_legacy_encoding(self):
    cast = UOp(X86Ops.MOVSX, dtypes.int32, (self.reg(dtypes.int8, RDX),), RAX)
    # movsx eax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("0F BE C2"))

  # accessing lower 8 bits of rsp, rbp, rsi, rdi requires rex prefix
  def test_lower_8bits_reg(self):
    cast = UOp(X86Ops.MOVSX, dtypes.int32, (self.reg(dtypes.int8, RDI),), RAX)
    # movsx eax, dil
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("40 0F BE C7"))

  # test 16 bit variant of legacy instruction
  def test_16bit_legacy_encoding(self):
    cast = UOp(X86Ops.MOVSX, dtypes.int16, (self.reg(dtypes.int8, RDX),), RAX)
    # movsx ax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("66 0F BE C2"))

  # test 64 bit variant of legacy instruction
  def test_64bit_legacy_encoding(self):
    cast = UOp(X86Ops.MOVSX, dtypes.int64, (self.reg(dtypes.int8, RDX),), RAX)
    # movsx rax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("48 0F BE C2"))

  # test compact vex encoding
  def test_compact_vex_encoding(self):
    xmm0, xmm1 = self.reg(dtypes.float32, XMM[0]), self.reg(dtypes.float32, XMM[1])
    add = UOp(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm1), XMM[0])
    # vaddss xmm0, xmm0, xmm1
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C5 FA 58 C1"))

  # test long vex encoding
  def test_long_vex_encoding(self):
    xmm0, xmm8 = self.reg(dtypes.float32, XMM[0]), self.reg(dtypes.float32, XMM[8])
    add = UOp(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm8), XMM[0])
    # vaddss xmm0, xmm0, xmm8
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C4 C1 7A 58 C0"))

  # test ymm encoding
  def test_ymm_encoding(self):
    xmm0, xmm1 = self.reg(dtypes.float32.vec(8), XMM[0]), self.reg(dtypes.float32.vec(8), XMM[1])
    add = UOp(X86Ops.VADDPS, dtypes.float32.vec(8), (xmm0, xmm1), XMM[0])
    # vaddps ymm0, ymm0, ymm1
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C5 FC 58 C1"))

  # test encoding where register is in the immediate field
  def test_reg_in_imm_field(self):
    xmm0, xmm1, xmm2 = self.reg(dtypes.float32, XMM[0]), self.reg(dtypes.float32, XMM[1]), self.reg(dtypes.float32, XMM[2])
    blend = UOp(X86Ops.VBLENDVPS, dtypes.float32, (xmm0, xmm1, xmm2), XMM[0])
    # vblendvps xmm0, xmm0, xmm1, xmm2
    self.assertEqual(bytes.fromhex(self.encode(blend)), bytes.fromhex("C4 E3 79 4A C1 20"))

  # when writting to mem the uop takes the store form where dtype is void and there's no definition
  def test_write_mem(self):
    base, index, disp = self.reg(dtypes.int32.ptr(), RDI), self.reg(dtypes.int32, RSI), imm(dtypes.int8, 10)
    xmm0 = self.reg(dtypes.float32, XMM[0])
    extr = UOp(X86Ops.VPEXTRD, dtypes.void, (base, index, disp, xmm0, imm(dtypes.uint8, 0)))
    # vpextrd dword ptr [rdi + rsi*4 + 0xa], xmm0, 0
    self.assertEqual(bytes.fromhex(self.encode(extr)), bytes.fromhex("C4 E3 79 16 44 B7 0A 00"))

  # test two address instruction with fused load works
  def test_two_address_load(self):
    base, index, disp = self.reg(dtypes.int32.ptr(), RDI), self.reg(dtypes.int32, RSI), imm(dtypes.int8, 10)
    cmove = UOp(X86Ops.CMOVE, dtypes.int32,  (base, index, disp), RAX)
    # cmove eax, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 44 B7 0A"))

  # test instruction where displacement and imm have the same value
  def test_disp_imm_same_value(self):
    base, index, disp = self.reg(dtypes.int8.ptr(), RDI), self.reg(dtypes.int8, RSI), imm(dtypes.int8, 10)
    mov = UOp(X86Ops.MOVi, dtypes.void, (base, index, disp, disp))
    # mov byte ptr [rdi + rsi + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(mov)), bytes.fromhex("40 C6 44 37 0A 0A"))

    base, index, disp = self.reg(dtypes.int32.ptr(), RDI), self.reg(dtypes.int32, RSI), imm(dtypes.int32, 10)
    imul = UOp(X86Ops.IMULi, dtypes.int32, (base, index, disp) + (imm(dtypes.int32, 10),), RDI)
    # imul edi, dword ptr [rdi + rsi*4 + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(imul)), bytes.fromhex("69 BC B7 0A 00 00 00 0A 00 00 00"))

  # cmoves have the cmp as the last src even though it is not explicitly used, the cmp doesn't define a reg and is ignored in the encoding
  def test_cmove_ignore_cmp(self):
    cmove = UOp(X86Ops.CMOVE, dtypes.int32, (self.reg(dtypes.int32, RAX), UOp(X86Ops.CMP)), RDX)
    # cmove edx, eax
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 D0"))

if __name__ == "__main__":
  unittest.main()