#!/usr/bin/env python3
"""Integration test: round-trip RDNA3 assembly through AMD toolchain."""
import unittest, re, io, sys
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import waitcnt

def get_amd_toolchain():
  """Check if AMD toolchain is available."""
  try:
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    HIPCompiler("gfx1100").compile(".text\ns_endpgm")
    return True
  except Exception:
    return False

def disassemble(lib: bytes, arch: str = "gfx1100") -> str:
  """Disassemble ELF binary using tinygrad's compiler, return raw output."""
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  HIPCompiler(arch).disassemble(lib)
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout
  return output

def parse_disassembly(raw: str) -> list[str]:
  """Parse disassembly output to list of instruction mnemonics."""
  lines = []
  for line in raw.splitlines():
    if line.startswith('\t'):
      instr = line.split('//')[0].strip()
      if instr: lines.append(instr)
  return lines

def assemble_and_disassemble(instructions: list, arch: str = "gfx1100") -> list[str]:
  """Assemble instructions with our DSL, then disassemble with AMD toolchain."""
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  # Generate bytes from our DSL
  code_bytes = b''.join(inst.to_bytes() for inst in instructions)

  # Wrap in minimal ELF-compatible assembly with .byte directives
  byte_str = ', '.join(f'0x{b:02x}' for b in code_bytes)
  asm_src = f".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n.byte {byte_str}\n"

  # Assemble with AMD COMGR and disassemble
  lib = HIPCompiler(arch).compile(asm_src)
  return parse_disassembly(disassemble(lib, arch))

@unittest.skipUnless(get_amd_toolchain(), "AMD toolchain not available")
class TestIntegration(unittest.TestCase):
  """Test our assembler output matches LLVM disassembly."""

  def test_simple_sop1(self):
    """Test SOP1 instructions round-trip."""
    instructions = [
      s_mov_b32(s[0], s[1]),
      s_mov_b32(s[2], 0),
      s_not_b32(s[3], s[4]),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_mov_b32', disasm[0])
    self.assertIn('s_mov_b32', disasm[1])
    self.assertIn('s_not_b32', disasm[2])

  def test_simple_sop2(self):
    """Test SOP2 instructions round-trip."""
    instructions = [
      s_add_u32(s[0], s[1], s[2]),
      s_sub_u32(s[3], s[4], 10),
      s_and_b32(s[5], s[6], s[7]),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_add_u32', disasm[0])
    self.assertIn('s_sub_u32', disasm[1])
    self.assertIn('s_and_b32', disasm[2])

  def test_simple_vop2(self):
    """Test VOP2 instructions round-trip."""
    instructions = [
      v_add_f32_e32(v[0], v[1], v[2]),
      v_mul_f32_e32(v[3], 1.0, v[4]),  # 1.0 is inline constant
      v_and_b32_e32(v[5], 10, v[6]),  # small inline constant
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('v_add_f32', disasm[0])
    self.assertIn('v_mul_f32', disasm[1])

  def test_control_flow(self):
    """Test control flow instructions."""
    instructions = [
      s_waitcnt(simm16=waitcnt(lgkmcnt=0)),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_waitcnt', disasm[0])
    self.assertIn('s_endpgm', disasm[1])

  def test_memory_ops(self):
    """Test memory instructions."""
    instructions = [
      s_load_b32(s[0], s[0:2], NULL),
      s_waitcnt(simm16=waitcnt(lgkmcnt=0)),
      global_store_b32(addr=v[0:2], data=v[2], saddr=OFF),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_load_b32', disasm[0])
    self.assertIn('s_waitcnt', disasm[1])
    self.assertIn('global_store_b32', disasm[2])

  def test_full_kernel(self):
    """Test a complete kernel similar to tinygrad output."""
    # Simple kernel: load value, add 1, store back
    instructions = [
      # Get thread ID
      v_mov_b32_e32(v[0], s[0]),  # base addr low
      v_mov_b32_e32(v[1], s[1]),  # base addr high
      # Load value
      global_load_b32(vdst=v[2], addr=v[0:2], saddr=OFF),
      s_waitcnt(simm16=waitcnt(vmcnt=0)),
      # Add 1.0
      v_add_f32_e32(v[2], 1.0, v[2]),
      # Store result
      global_store_b32(addr=v[0:2], data=v[2], saddr=OFF),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    # Verify key instructions are present
    self.assertTrue(any('global_load' in d for d in disasm))
    self.assertTrue(any('v_add_f32' in d for d in disasm))
    self.assertTrue(any('global_store' in d for d in disasm))
    self.assertTrue(any('s_endpgm' in d for d in disasm))

  def test_bytes_roundtrip(self):
    """Test that our bytes match what AMD assembler produces."""
    from tinygrad.runtime.support.compiler_amd import HIPCompiler

    # Simple instruction
    inst = s_mov_b32(s[0], s[1])
    our_bytes = inst.to_bytes()

    # Assemble same instruction with AMD toolchain
    asm_src = ".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\ns_mov_b32 s0, s1\n"
    compiler = HIPCompiler("gfx1100")
    lib = compiler.compile(asm_src)
    raw = disassemble(lib)

    for line in raw.splitlines():
      if 's_mov_b32' in line and '//' in line:
        # Extract hex bytes from comment: "// 000000001300: BE800001"
        comment = line.split('//')[1].strip()
        hex_str = comment.split(':')[1].strip()
        # Convert big-endian hex string to little-endian bytes
        amd_bytes = bytes.fromhex(hex_str)[::-1]  # reverse for little-endian
        self.assertEqual(our_bytes, amd_bytes, f"Bytes mismatch: ours={our_bytes.hex()} AMD={amd_bytes.hex()}")
        return
    self.fail("Could not find s_mov_b32 in disassembly")

if __name__ == "__main__":
  unittest.main()
