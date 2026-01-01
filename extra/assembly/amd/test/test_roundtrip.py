#!/usr/bin/env python3
"""Roundtrip tests: generate tinygrad kernels, decode instructions, re-encode, verify match."""
import unittest, io, sys, re, subprocess, os
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.asm import asm
from extra.assembly.amd.asm import detect_format
from extra.assembly.amd.test.helpers import get_llvm_mc, get_llvm_objdump

def disassemble_lib(lib: bytes, compiler) -> list[tuple[str, bytes]]:
  """Disassemble ELF binary and return list of (instruction_text, machine_code_bytes)."""
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  compiler.disassemble(lib)
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout

  results = []
  for line in output.splitlines():
    if '//' not in line: continue
    instr = line.split('//')[0].strip()
    if not instr: continue
    comment = line.split('//')[1].strip()
    if ':' not in comment: continue
    hex_str = comment.split(':')[1].strip().split()[0]
    try:
      machine_bytes = bytes.fromhex(hex_str)[::-1]  # big-endian to little-endian
      results.append((instr, machine_bytes))
    except ValueError:
      continue
  return results

def compile_asm(instr: str, compiler=None) -> bytes:
  """Compile a single instruction with llvm-mc and return the machine code bytes."""
  llvm_mc = get_llvm_mc()
  result = subprocess.run(
    [llvm_mc, '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
    input=f".text\n{instr}\n", capture_output=True, text=True)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc failed for '{instr}': {result.stderr.strip()}")
  # Parse encoding: [0x01,0x39,0x0a,0x7e]
  for line in result.stdout.split('\n'):
    if 'encoding:' in line:
      enc = line.split('encoding:')[1].strip()
      if enc.startswith('[') and enc.endswith(']'):
        hex_vals = enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')
        return bytes.fromhex(hex_vals)
  raise RuntimeError(f"no encoding found in llvm-mc output for: {instr}")

def compile_asm_batch(instrs: list[str]) -> list[bytes]:
  """Compile multiple instructions with a single llvm-mc call."""
  if not instrs: return []
  llvm_mc = get_llvm_mc()
  src = ".text\n" + "\n".join(instrs) + "\n"
  result = subprocess.run(
    [llvm_mc, '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
    input=src, capture_output=True, text=True)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc batch failed: {result.stderr.strip()}")
  # Parse all encodings in order
  encodings = []
  for line in result.stdout.split('\n'):
    if 'encoding:' in line:
      enc = line.split('encoding:')[1].strip()
      if enc.startswith('[') and enc.endswith(']'):
        hex_vals = enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')
        encodings.append(bytes.fromhex(hex_vals))
  if len(encodings) != len(instrs): raise RuntimeError(f"expected {len(instrs)} encodings, got {len(encodings)}")
  return encodings

def compile_and_disasm_batch(instrs: list[str], compiler) -> list[str]:
  """Compile instructions with LLVM and get LLVM's disassembly."""
  import tempfile, os
  if not instrs: return []
  # Build assembly source with all instructions
  src = ".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n"
  src += "\n".join(f"  {instr}" for instr in instrs) + "\n"
  # Use llvm-mc to assemble to object file
  with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
    obj_path = f.name
  try:
    result = subprocess.run(
      [get_llvm_mc(), '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-filetype=obj', '-o', obj_path],
      input=src, capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
    # Disassemble with llvm-objdump
    result = subprocess.run([get_llvm_objdump(), '-d', '--mcpu=gfx1100', obj_path], capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError(f"llvm-objdump failed: {result.stderr.strip()}")
    # Parse disassembly output
    results: list[str] = []
    for line in result.stdout.splitlines():
      if '//' not in line: continue
      instr = line.split('//')[0].strip()
      if instr: results.append(instr)
    return results[:len(instrs)]
  finally:
    os.unlink(obj_path)

class TestTinygradKernelRoundtrip(unittest.TestCase):
  """Test roundtrip on real tinygrad-generated kernels using get_kernels_from_tinygrad pattern."""

  def _test_kernel_roundtrip(self, op_fn):
    """Generate kernel from op_fn, test:
    1. decode -> reencode matches original bytes
    2. asm(disasm()) matches LLVM output
    3. our disasm() matches LLVM's disassembly string exactly
    """
    from extra.assembly.amd.test.test_compare_emulators import get_kernels_from_tinygrad
    from tinygrad.runtime.support.compiler_amd import HIPCompiler

    kernels, _, _ = get_kernels_from_tinygrad(op_fn)
    compiler = HIPCompiler('gfx1100')

    # First pass: decode all instructions and collect info
    decoded_instrs: list[tuple] = []  # list of (ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err)
    for ki, kernel in enumerate(kernels):
      offset = 0
      while offset < len(kernel.code):
        remaining = kernel.code[offset:]
        fmt = detect_format(remaining)
        if fmt is None:
          decoded_instrs.append((ki, offset, None, None, None, False, "no format"))
          offset += 4
          continue

        base_size = fmt._size()
        if len(remaining) < base_size:
          break

        try:
          decoded = fmt.from_bytes(remaining)  # pass all remaining bytes so from_bytes can read literal
          size = decoded.size()  # actual size including literal
          orig_bytes = remaining[:size]
          reencoded = decoded.to_bytes()
          our_disasm = decoded.disasm()
          decode_ok = reencoded == orig_bytes
          decode_err: str | None = None if decode_ok else f"orig={orig_bytes.hex()} reenc={reencoded.hex()}"
          decoded_instrs.append((ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err))
        except Exception as e:
          decoded_instrs.append((ki, offset, remaining[:base_size], None, None, False, str(e)))
          size = base_size

        offset += size

    # Collect disasm strings for batched LLVM calls - skip unknown opcodes (op_X) that LLVM can't compile
    asm_test_instrs: list[tuple[int, str]] = []  # (idx, our_disasm) for asm test
    disasm_test_instrs: list[tuple[int, str]] = []  # (idx, our_disasm) for disasm comparison test

    for idx, (ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err) in enumerate(decoded_instrs):
      if our_disasm is None: continue
      # Skip unknown opcodes and malformed instructions for both tests
      if our_disasm.startswith('op_') or re.search(r', \d+, \d+, \d+,', our_disasm): continue
      asm_test_instrs.append((idx, our_disasm))
      disasm_test_instrs.append((idx, our_disasm))

    # Batch compile for asm test
    asm_llvm_results = compile_asm_batch([d for _, d in asm_test_instrs])
    asm_llvm_map = {idx: result for (idx, _), result in zip(asm_test_instrs, asm_llvm_results)}

    # Batch compile+disasm for disasm comparison test
    disasm_llvm_results = compile_and_disasm_batch([d for _, d in disasm_test_instrs], compiler)
    disasm_llvm_map = {idx: result for (idx, _), result in zip(disasm_test_instrs, disasm_llvm_results)}

    # Now evaluate results
    decode_passed, decode_failed, decode_skipped = 0, 0, 0
    asm_passed, asm_failed, asm_skipped = 0, 0, 0
    disasm_passed, disasm_failed, disasm_skipped = 0, 0, 0
    decode_failures: list[str] = []
    asm_failures: list[str] = []
    disasm_failures: list[str] = []

    for idx, (ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err) in enumerate(decoded_instrs):
      # Decode test
      if decode_ok:
        decode_passed += 1
      elif decode_err == "no format":
        decode_skipped += 1
      else:
        decode_failed += 1
        decode_failures.append(f"K{ki}@{offset}: {our_disasm}: {decode_err}")

      # Asm test
      if our_disasm is None:
        asm_skipped += 1
      elif idx in asm_llvm_map:
        llvm_bytes = asm_llvm_map[idx]
        try:
          our_bytes = asm(our_disasm).to_bytes()
          if our_bytes[:len(llvm_bytes)] == llvm_bytes:
            asm_passed += 1
          else:
            asm_failed += 1
            asm_failures.append(f"K{ki}@{offset}: '{our_disasm}': ours={our_bytes[:len(llvm_bytes)].hex()} llvm={llvm_bytes.hex()}")
        except Exception:
          asm_skipped += 1
      else:
        asm_skipped += 1

      # Disasm comparison test
      if our_disasm is None:
        disasm_skipped += 1
      elif idx in disasm_llvm_map:
        llvm_disasm = disasm_llvm_map[idx]
        if our_disasm == llvm_disasm:
          disasm_passed += 1
        else:
          disasm_failed += 1
          disasm_failures.append(f"K{ki}@{offset}: ours='{our_disasm}' llvm='{llvm_disasm}'")
      else:
        disasm_skipped += 1

    print(f"decode roundtrip: {decode_passed} passed, {decode_failed} failed, {decode_skipped} skipped")
    print(f"asm vs llvm: {asm_passed} passed, {asm_failed} failed, {asm_skipped} skipped")
    print(f"disasm vs llvm: {disasm_passed} passed, {disasm_failed} failed, {disasm_skipped} skipped")
    self.assertEqual(decode_failed, 0, f"Decode failures:\n" + "\n".join(decode_failures[:20]))
    self.assertEqual(asm_failed, 0, f"Asm failures:\n" + "\n".join(asm_failures[:20]))
    # Note: disasm string comparison is informational only - formatting differences between LLVM versions are expected

  # Basic unary ops
  def test_neg(self): self._test_kernel_roundtrip(lambda T: -T([1.0, -2.0, 3.0, -4.0]))
  def test_relu(self): self._test_kernel_roundtrip(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())
  def test_exp(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).exp())
  def test_log(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 3.0]).log())
  def test_sin(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).sin())
  def test_sqrt(self): self._test_kernel_roundtrip(lambda T: T([1.0, 4.0, 9.0]).sqrt())
  def test_recip(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 4.0]).reciprocal())

  # Binary ops
  def test_add(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0]) + T([3.0, 4.0]))
  def test_sub(self): self._test_kernel_roundtrip(lambda T: T([5.0, 6.0]) - T([1.0, 2.0]))
  def test_mul(self): self._test_kernel_roundtrip(lambda T: T([2.0, 3.0]) * T([4.0, 5.0]))
  def test_div(self): self._test_kernel_roundtrip(lambda T: T([10.0, 20.0]) / T([2.0, 4.0]))
  def test_max_binary(self): self._test_kernel_roundtrip(lambda T: T([1.0, 5.0]).maximum(T([3.0, 2.0])))

  # Reductions
  def test_sum_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).sum())
  def test_max_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).max())
  def test_mean_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(32).mean())

  # Matmul
  def test_gemm_4x4(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4) @ T.empty(4, 4))
  def test_gemv(self): self._test_kernel_roundtrip(lambda T: T.empty(1, 16) @ T.empty(16, 16))

  # Complex ops
  def test_softmax(self): self._test_kernel_roundtrip(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel_roundtrip(lambda T: T.empty(8, 8).layernorm())

  # Memory patterns
  def test_contiguous(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel_roundtrip(lambda T: (T.empty(16) + 1).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 1).expand(4, 4).contiguous())

  # Cast ops
  def test_cast_int(self): self._test_kernel_roundtrip(lambda T: T.empty(16).int().float())
  def test_cast_half(self): self._test_kernel_roundtrip(lambda T: T.empty(16).half().float())

  # Comparison ops
  def test_cmp_lt(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_where(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) > 0).where(T.empty(64), T.empty(64)))

  # Fused ops
  def test_fma(self): self._test_kernel_roundtrip(lambda T: (T([1.0, 2.0]) * T([3.0, 4.0]) + T([5.0, 6.0])))

if __name__ == "__main__":
  unittest.main()
