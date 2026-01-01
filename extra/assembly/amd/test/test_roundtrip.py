#!/usr/bin/env python3
"""Roundtrip tests: generate tinygrad kernels, decode instructions, re-encode, verify match."""
import unittest, io, sys, re, subprocess, os
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.test.helpers import get_llvm_mc, get_llvm_objdump

def compile_asm_batch(instrs: list[str], mcpu: str = 'gfx1100') -> list[bytes]:
  """Compile multiple instructions with a single llvm-mc call."""
  if not instrs: return []
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
                          input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc batch failed: {result.stderr.strip()}")
  encodings = []
  for line in result.stdout.split('\n'):
    if 'encoding:' in line:
      enc = line.split('encoding:')[1].strip()
      if enc.startswith('[') and enc.endswith(']'):
        encodings.append(bytes.fromhex(enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')))
  if len(encodings) != len(instrs): raise RuntimeError(f"expected {len(instrs)} encodings, got {len(encodings)}")
  return encodings

def compile_and_disasm_batch(instrs: list[str], mcpu: str = 'gfx1100') -> list[str]:
  """Compile instructions with LLVM and get LLVM's disassembly."""
  import tempfile
  if not instrs: return []
  src = ".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n" + "\n".join(f"  {instr}" for instr in instrs) + "\n"
  with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
    obj_path = f.name
  try:
    result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', '-mattr=+real-true16,+wavefrontsize32', '-filetype=obj', '-o', obj_path],
                            input=src, capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
    result = subprocess.run([get_llvm_objdump(), '-d', f'--mcpu={mcpu}', obj_path], capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError(f"llvm-objdump failed: {result.stderr.strip()}")
    results: list[str] = []
    for line in result.stdout.splitlines():
      if '//' not in line: continue
      instr = line.split('//')[0].strip()
      if instr: results.append(instr)
    return results[:len(instrs)]
  finally:
    os.unlink(obj_path)

class TestRoundtripBase(unittest.TestCase):
  """Base class for roundtrip tests."""
  mcpu: str = 'gfx1100'
  arch: str = 'rdna3'

  @classmethod
  def _get_modules(cls):
    if cls.arch == 'rdna3':
      from extra.assembly.amd.autogen.rdna3 import ins
      from extra.assembly.amd.asm import detect_format, asm
    else:
      import extra.assembly.amd.autogen.rdna4.ins as ins
      from extra.assembly.amd.asm import asm
      detect_format = None  # RDNA4 uses different detection
    return ins, detect_format, asm

  def _test_kernel_roundtrip(self, op_fn):
    """Generate kernel from op_fn, test decode -> reencode and asm(disasm()) matches LLVM."""
    from extra.assembly.amd.test.test_compare_emulators import get_kernels_from_tinygrad
    from tinygrad.runtime.support.compiler_amd import HIPCompiler

    ins, detect_format, asm = self._get_modules()
    kernels, _, _ = get_kernels_from_tinygrad(op_fn)
    compiler = HIPCompiler(self.mcpu)

    # First pass: decode all instructions
    decoded_instrs: list[tuple] = []
    for ki, kernel in enumerate(kernels):
      offset = 0
      while offset < len(kernel.code):
        remaining = kernel.code[offset:]
        if len(remaining) < 4: break

        # Try to detect format
        if detect_format is not None:
          try:
            fmt = detect_format(remaining)
          except ValueError:
            decoded_instrs.append((ki, offset, None, None, None, False, "no format"))
            offset += 4
            continue
        else:
          # For RDNA4, try formats in order
          fmt = None
          from extra.assembly.amd.autogen.rdna4.ins import SOP1, SOP2, SOPC, SOPK, SOPP, VOP1, VOP2, VOP3, VOP3P, VOPC, VOPD, VDS, SMEM, VFLAT, VBUFFER, VIMAGE, VSAMPLE, VEXPORT, VDSDIR
          word = int.from_bytes(remaining[:4], 'little')
          for cls in [VOPD, VOP3P, VOP3, VDS, VFLAT, VBUFFER, VIMAGE, VSAMPLE, SMEM, VEXPORT, SOP1, SOPC, SOPP, SOPK, VOPC, VOP1, SOP2, VOP2, VDSDIR]:
            if cls._encoding is not None:
              bf, val = cls._encoding
              if ((word >> bf.lo) & bf.mask()) == val:
                fmt = cls
                break
          if fmt is None:
            decoded_instrs.append((ki, offset, None, None, None, False, "no format"))
            offset += 4
            continue

        base_size = fmt._size()
        if len(remaining) < base_size: break

        try:
          decoded = fmt.from_bytes(remaining)
          size = decoded.size()
          orig_bytes = remaining[:size]
          reencoded = decoded.to_bytes()
          our_disasm = decoded.disasm()
          decode_ok = reencoded == orig_bytes
          decode_err = None if decode_ok else f"orig={orig_bytes.hex()} reenc={reencoded.hex()}"
          decoded_instrs.append((ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err))
        except Exception as e:
          decoded_instrs.append((ki, offset, remaining[:base_size], None, None, False, str(e)))
          size = base_size
        offset += size

    # Collect disasm strings for batched LLVM calls
    asm_test_instrs: list[tuple[int, str]] = []
    for idx, (ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err) in enumerate(decoded_instrs):
      if our_disasm is None: continue
      if our_disasm.startswith('op_') or re.search(r', \d+, \d+, \d+,', our_disasm): continue
      asm_test_instrs.append((idx, our_disasm))

    # Batch compile for asm test
    asm_llvm_results = compile_asm_batch([d for _, d in asm_test_instrs], self.mcpu)
    asm_llvm_map = {idx: result for (idx, _), result in zip(asm_test_instrs, asm_llvm_results)}

    # Batch compile+disasm for disasm comparison test
    disasm_llvm_results = compile_and_disasm_batch([d for _, d in asm_test_instrs], self.mcpu)
    disasm_llvm_map = {idx: result for (idx, _), result in zip(asm_test_instrs, disasm_llvm_results)}

    # Evaluate results
    decode_passed, decode_failed, decode_skipped = 0, 0, 0
    asm_passed, asm_failed, asm_skipped = 0, 0, 0
    disasm_passed, disasm_failed, disasm_skipped = 0, 0, 0
    decode_failures, asm_failures, disasm_failures = [], [], []

    for idx, (ki, offset, orig_bytes, decoded, our_disasm, decode_ok, decode_err) in enumerate(decoded_instrs):
      if decode_ok: decode_passed += 1
      elif decode_err == "no format": decode_skipped += 1
      else:
        decode_failed += 1
        decode_failures.append(f"K{ki}@{offset}: {our_disasm}: {decode_err}")

      if our_disasm is None:
        asm_skipped += 1
        disasm_skipped += 1
      elif idx in asm_llvm_map:
        llvm_bytes = asm_llvm_map[idx]
        try:
          our_bytes = asm(our_disasm).to_bytes()
          if our_bytes[:len(llvm_bytes)] == llvm_bytes: asm_passed += 1
          else:
            asm_failed += 1
            asm_failures.append(f"K{ki}@{offset}: '{our_disasm}': ours={our_bytes[:len(llvm_bytes)].hex()} llvm={llvm_bytes.hex()}")
        except Exception:
          asm_skipped += 1

        if idx in disasm_llvm_map:
          if our_disasm == disasm_llvm_map[idx]: disasm_passed += 1
          else:
            disasm_failed += 1
            disasm_failures.append(f"K{ki}@{offset}: ours='{our_disasm}' llvm='{disasm_llvm_map[idx]}'")
        else:
          disasm_skipped += 1
      else:
        asm_skipped += 1
        disasm_skipped += 1

    print(f"{self.arch.upper()} decode roundtrip: {decode_passed} passed, {decode_failed} failed, {decode_skipped} skipped")
    print(f"{self.arch.upper()} asm vs llvm: {asm_passed} passed, {asm_failed} failed, {asm_skipped} skipped")
    print(f"{self.arch.upper()} disasm vs llvm: {disasm_passed} passed, {disasm_failed} failed, {disasm_skipped} skipped")
    self.assertEqual(decode_failed, 0, f"Decode failures:\n" + "\n".join(decode_failures[:20]))
    self.assertEqual(asm_failed, 0, f"Asm failures:\n" + "\n".join(asm_failures[:20]))

class TestRoundtripRDNA3(TestRoundtripBase):
  """Roundtrip tests for RDNA3 (gfx1100)."""
  mcpu, arch = 'gfx1100', 'rdna3'

  def test_neg(self): self._test_kernel_roundtrip(lambda T: -T([1.0, -2.0, 3.0, -4.0]))
  def test_relu(self): self._test_kernel_roundtrip(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())
  def test_exp(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).exp())
  def test_log(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 3.0]).log())
  def test_sin(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).sin())
  def test_sqrt(self): self._test_kernel_roundtrip(lambda T: T([1.0, 4.0, 9.0]).sqrt())
  def test_recip(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 4.0]).reciprocal())
  def test_add(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0]) + T([3.0, 4.0]))
  def test_sub(self): self._test_kernel_roundtrip(lambda T: T([5.0, 6.0]) - T([1.0, 2.0]))
  def test_mul(self): self._test_kernel_roundtrip(lambda T: T([2.0, 3.0]) * T([4.0, 5.0]))
  def test_div(self): self._test_kernel_roundtrip(lambda T: T([10.0, 20.0]) / T([2.0, 4.0]))
  def test_max_binary(self): self._test_kernel_roundtrip(lambda T: T([1.0, 5.0]).maximum(T([3.0, 2.0])))
  def test_sum_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).sum())
  def test_max_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).max())
  def test_mean_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(32).mean())
  def test_gemm_4x4(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4) @ T.empty(4, 4))
  def test_gemv(self): self._test_kernel_roundtrip(lambda T: T.empty(1, 16) @ T.empty(16, 16))
  def test_softmax(self): self._test_kernel_roundtrip(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel_roundtrip(lambda T: T.empty(8, 8).layernorm())
  def test_contiguous(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel_roundtrip(lambda T: (T.empty(16) + 1).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 1).expand(4, 4).contiguous())
  def test_cast_int(self): self._test_kernel_roundtrip(lambda T: T.empty(16).int().float())
  def test_cast_half(self): self._test_kernel_roundtrip(lambda T: T.empty(16).half().float())
  def test_cmp_lt(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_where(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) > 0).where(T.empty(64), T.empty(64)))
  def test_fma(self): self._test_kernel_roundtrip(lambda T: (T([1.0, 2.0]) * T([3.0, 4.0]) + T([5.0, 6.0])))

@unittest.skipUnless(os.environ.get("TEST_RDNA4"), "RDNA4 roundtrip tests require TEST_RDNA4=1 and gfx1200 hardware")
class TestRoundtripRDNA4(TestRoundtripBase):
  """Roundtrip tests for RDNA4 (gfx1200)."""
  mcpu, arch = 'gfx1200', 'rdna4'

  def test_neg(self): self._test_kernel_roundtrip(lambda T: -T([1.0, -2.0, 3.0, -4.0]))
  def test_relu(self): self._test_kernel_roundtrip(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())
  def test_exp(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).exp())
  def test_log(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 3.0]).log())
  def test_sin(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).sin())
  def test_sqrt(self): self._test_kernel_roundtrip(lambda T: T([1.0, 4.0, 9.0]).sqrt())
  def test_recip(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 4.0]).reciprocal())
  def test_add(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0]) + T([3.0, 4.0]))
  def test_sub(self): self._test_kernel_roundtrip(lambda T: T([5.0, 6.0]) - T([1.0, 2.0]))
  def test_mul(self): self._test_kernel_roundtrip(lambda T: T([2.0, 3.0]) * T([4.0, 5.0]))
  def test_div(self): self._test_kernel_roundtrip(lambda T: T([10.0, 20.0]) / T([2.0, 4.0]))
  def test_max_binary(self): self._test_kernel_roundtrip(lambda T: T([1.0, 5.0]).maximum(T([3.0, 2.0])))
  def test_sum_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).sum())
  def test_max_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).max())
  def test_mean_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(32).mean())
  def test_gemm_4x4(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4) @ T.empty(4, 4))
  def test_gemv(self): self._test_kernel_roundtrip(lambda T: T.empty(1, 16) @ T.empty(16, 16))
  def test_softmax(self): self._test_kernel_roundtrip(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel_roundtrip(lambda T: T.empty(8, 8).layernorm())
  def test_contiguous(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel_roundtrip(lambda T: (T.empty(16) + 1).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 1).expand(4, 4).contiguous())
  def test_cast_int(self): self._test_kernel_roundtrip(lambda T: T.empty(16).int().float())
  def test_cast_half(self): self._test_kernel_roundtrip(lambda T: T.empty(16).half().float())
  def test_cmp_lt(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_where(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) > 0).where(T.empty(64), T.empty(64)))
  def test_fma(self): self._test_kernel_roundtrip(lambda T: (T([1.0, 2.0]) * T([3.0, 4.0]) + T([5.0, 6.0])))

# Keep old class name for backwards compatibility
TestTinygradKernelRoundtrip = TestRoundtripRDNA3

if __name__ == "__main__":
  unittest.main()
