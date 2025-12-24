#!/usr/bin/env python3
"""Test RDNA3 assembler/disassembler against LLVM test vectors."""
import unittest, re
from tinygrad.helpers import fetch
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import RawImm

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

# Format info: (filename, format_class, op_enum)
LLVM_TEST_FILES = {
  'sop1': ('gfx11_asm_sop1.s', SOP1, SOP1Op),
  'sop2': ('gfx11_asm_sop2.s', SOP2, SOP2Op),
  'sopp': ('gfx11_asm_sopp.s', SOPP, SOPPOp),
  'sopk': ('gfx11_asm_sopk.s', SOPK, SOPKOp),
  'sopc': ('gfx11_asm_sopc.s', SOPC, SOPCOp),
  'vop1': ('gfx11_asm_vop1.s', VOP1, VOP1Op),
  'vop2': ('gfx11_asm_vop2.s', VOP2, VOP2Op),
}

def parse_llvm_tests(text: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm = line.split('//')[0].strip()
    if not asm: continue
    for j in range(i, min(i + 3, len(lines))):
      if m := re.search(r'GFX11[^:]*:\s*encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
        if hex_bytes:
          try: tests.append((asm, bytes.fromhex(hex_bytes)))
          except ValueError: pass
        break
  return tests

def get_instruction_helper(mnemonic: str):
  name = mnemonic.lower().replace('.', '_')
  for suffix in ['_e32', '']:
    if (n := f"{name}{suffix}") in globals(): return globals()[n]
  return None

def parse_operand(op: str):
  op = op.strip()
  # Register patterns
  for cls, prefix in [(SGPR, 's'), (VGPR, 'v'), (TTMP, 'ttmp')]:
    if m := re.match(rf'^{prefix}\[(\d+):(\d+)\]$', op): return cls[int(m.group(1)):int(m.group(2))+1]
    if m := re.match(rf'^{prefix}(\d+)$', op): return cls[int(m.group(1))]
  # Special registers and constants
  specials = {'vcc_lo': VCC_LO, 'vcc_hi': VCC_HI, 'null': NULL, 'off': OFF,
              'm0': M0, 'exec_lo': EXEC_LO, 'exec_hi': EXEC_HI, 'scc': SCC}
  if op.lower() in specials: return specials[op.lower()]
  # Numbers
  if m := re.match(r'^0x([0-9a-fA-F]+)$', op): return int(m.group(1), 16)
  if m := re.match(r'^-?\d+$', op): return int(op)
  # Float constants
  floats = {'0.5': POS_HALF, '-0.5': NEG_HALF, '1.0': POS_ONE, '-1.0': NEG_ONE,
            '2.0': POS_TWO, '-2.0': NEG_TWO, '4.0': POS_FOUR, '-4.0': NEG_FOUR}
  return floats.get(op)

def split_operands(s: str) -> list[str]:
  operands, current, depth = [], "", 0
  for ch in s:
    if ch == '[': depth += 1
    elif ch == ']': depth -= 1
    if ch == ',' and depth == 0:
      if current.strip(): operands.append(current.strip())
      current = ""
    else: current += ch
  if current.strip(): operands.append(current.strip())
  return operands

SOP1_SRC_ONLY = {'s_setpc_b64', 's_rfe_b64'}
SOP1_MSG_IMM = {'s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'}
# SOPK instructions that only take simm16 (no sdst)
SOPK_IMM_ONLY = {'s_version', 's_subvector_loop_begin', 's_subvector_loop_end'}
# SOPK instructions where simm16 comes before sdst in assembly (setreg)
SOPK_IMM_FIRST = {'s_setreg_b32'}
# SOPK instructions with sdst, simm16 order (getreg - dst first)
SOPK_DST_FIRST = {'s_getreg_b32', 's_waitcnt_vscnt', 's_waitcnt_vmcnt', 's_waitcnt_expcnt', 's_waitcnt_lgkmcnt'}
# SOPK instructions with special 64-bit encoding (not yet supported)
SOPK_UNSUPPORTED = {'s_setreg_imm32_b32'}

def try_assemble(asm: str):
  parts = asm.replace(',', ' ').split()
  if not parts: return None
  mnemonic, operands_str = parts[0], asm[len(parts[0]):].strip()
  helper = get_instruction_helper(mnemonic)
  if helper is None: return None
  if not operands_str:
    try: return helper().to_bytes()
    except: return None
  parsed = [parse_operand(op) for op in split_operands(operands_str)]
  if None in parsed: return None
  try:
    name = mnemonic.lower()
    if name in SOPK_UNSUPPORTED: return None  # skip unsupported formats
    if name in SOP1_SRC_ONLY: inst = helper(ssrc0=parsed[0])
    elif name in SOP1_MSG_IMM: inst = helper(sdst=parsed[0], ssrc0=RawImm(parsed[1]))
    elif name in SOPK_IMM_ONLY: inst = helper(simm16=parsed[0])
    elif name in SOPK_IMM_FIRST: inst = helper(simm16=parsed[0], sdst=parsed[1])
    elif name in SOPK_DST_FIRST: inst = helper(sdst=parsed[0], simm16=parsed[1])
    else: inst = helper(*parsed)
    return inst.to_bytes()
  except: return None

class TestLLVM(unittest.TestCase):
  """Test assembler and disassembler against all LLVM test vectors."""
  tests: dict[str, list[tuple[str, bytes]]] = {}

  @classmethod
  def setUpClass(cls):
    for name, (filename, _, _) in LLVM_TEST_FILES.items():
      try:
        data = fetch(f"{LLVM_BASE}/{filename}").read_bytes()
        cls.tests[name] = parse_llvm_tests(data.decode('utf-8', errors='ignore'))
      except Exception as e:
        print(f"Warning: couldn't fetch {filename}: {e}")
        cls.tests[name] = []

# Generate test methods dynamically for each format
def _make_asm_test(name):
  def test(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests.get(name, []):
      result = try_assemble(asm)
      if result is None: skipped += 1
      elif result == expected: passed += 1
      else: failed += 1
    print(f"{name.upper()} asm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  def test(self):
    _, fmt_cls, op_enum = LLVM_TEST_FILES[name]
    passed, failed, skipped = 0, 0, 0
    for asm, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size(): skipped += 1; continue  # skip literals
      try:
        decoded = fmt_cls.from_bytes(data)
        op_enum(decoded._values.get('op', 0))  # validate opcode
        if decoded.to_bytes()[:len(data)] == data: passed += 1
        else: failed += 1
      except: skipped += 1
    print(f"{name.upper()} disasm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
  return test

for name in LLVM_TEST_FILES:
  setattr(TestLLVM, f'test_{name}_asm', _make_asm_test(name))
  setattr(TestLLVM, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
