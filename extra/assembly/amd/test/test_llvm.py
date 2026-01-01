#!/usr/bin/env python3
"""Test RDNA3/RDNA4 assembler/disassembler against LLVM test vectors."""
import unittest, re, subprocess
from tinygrad.helpers import fetch
from extra.assembly.amd.test.helpers import get_llvm_mc

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

RDNA3_TEST_FILES = {
  'sop1': 'gfx11_asm_sop1.s', 'sop2': 'gfx11_asm_sop2.s', 'sopp': 'gfx11_asm_sopp.s', 'sopk': 'gfx11_asm_sopk.s', 'sopc': 'gfx11_asm_sopc.s',
  'vop1': 'gfx11_asm_vop1.s', 'vop2': 'gfx11_asm_vop2.s', 'vopc': 'gfx11_asm_vopc.s', 'vop3': 'gfx11_asm_vop3.s', 'vop3p': 'gfx11_asm_vop3p.s',
  'vinterp': 'gfx11_asm_vinterp.s', 'vopd': 'gfx11_asm_vopd.s', 'vopcx': 'gfx11_asm_vopcx.s',
  'vop3_from_vop1': 'gfx11_asm_vop3_from_vop1.s', 'vop3_from_vop2': 'gfx11_asm_vop3_from_vop2.s',
  'vop3_from_vopc': 'gfx11_asm_vop3_from_vopc.s', 'vop3_from_vopcx': 'gfx11_asm_vop3_from_vopcx.s',
  'ds': 'gfx11_asm_ds.s', 'smem': 'gfx11_asm_smem.s', 'flat': 'gfx11_asm_flat.s',
  'mubuf': 'gfx11_asm_mubuf.s', 'mtbuf': 'gfx11_asm_mtbuf.s', 'mimg': 'gfx11_asm_mimg.s', 'ldsdir': 'gfx11_asm_ldsdir.s',
  'exp': 'gfx11_asm_exp.s', 'wmma': 'gfx11_asm_wmma.s',
  'vop3_features': 'gfx11_asm_vop3_features.s', 'vop3p_features': 'gfx11_asm_vop3p_features.s', 'vopd_features': 'gfx11_asm_vopd_features.s',
  'vop3_alias': 'gfx11_asm_vop3_alias.s', 'vop3p_alias': 'gfx11_asm_vop3p_alias.s', 'vopc_alias': 'gfx11_asm_vopc_alias.s',
  'vopcx_alias': 'gfx11_asm_vopcx_alias.s', 'vinterp_alias': 'gfx11_asm_vinterp_alias.s',
  'smem_alias': 'gfx11_asm_smem_alias.s', 'mubuf_alias': 'gfx11_asm_mubuf_alias.s', 'mtbuf_alias': 'gfx11_asm_mtbuf_alias.s',
}

RDNA4_TEST_FILES = {
  'sop1': 'gfx12_asm_sop1.s', 'sop2': 'gfx12_asm_sop2.s', 'sop2_alias': 'gfx12_asm_sop2_alias.s',
  'sopp': 'gfx12_asm_sopp.s', 'sopk': 'gfx12_asm_sopk.s', 'sopk_alias': 'gfx12_asm_sopk_alias.s', 'sopc': 'gfx12_asm_sopc.s',
  'vop1': 'gfx12_asm_vop1.s', 'vop2': 'gfx12_asm_vop2.s', 'vop2_aliases': 'gfx12_asm_vop2_aliases.s',
  'vopc': 'gfx12_asm_vopc.s', 'vopcx': 'gfx12_asm_vopcx.s',
  'vop3': 'gfx12_asm_vop3.s', 'vop3_aliases': 'gfx12_asm_vop3_aliases.s', 'vop3c': 'gfx12_asm_vop3c.s', 'vop3cx': 'gfx12_asm_vop3cx.s',
  'vop3p': 'gfx12_asm_vop3p.s', 'vop3p_aliases': 'gfx12_asm_vop3p_aliases.s', 'vop3p_features': 'gfx12_asm_vop3p_features.s',
  'vopd': 'gfx12_asm_vopd.s', 'vopd_features': 'gfx12_asm_vopd_features.s',
  'vop3_from_vop1': 'gfx12_asm_vop3_from_vop1.s', 'vop3_from_vop2': 'gfx12_asm_vop3_from_vop2.s',
  'ds': 'gfx12_asm_ds.s', 'ds_alias': 'gfx12_asm_ds_alias.s', 'smem': 'gfx12_asm_smem.s',
  'vflat': 'gfx12_asm_vflat.s', 'vflat_alias': 'gfx12_asm_vflat_alias.s',
  'vscratch': 'gfx12_asm_vflat.s', 'vscratch_alias': 'gfx12_asm_vflat_alias.s',  # scratch instructions in vflat files
  'vbuffer_mubuf': 'gfx12_asm_vbuffer_mubuf.s', 'vbuffer_mubuf_alias': 'gfx12_asm_vbuffer_mubuf_alias.s',
  'vbuffer_mtbuf': 'gfx12_asm_vbuffer_mtbuf.s', 'vbuffer_mtbuf_alias': 'gfx12_asm_vbuffer_mtbuf_alias.s',
  'vimage': 'gfx12_asm_vimage.s', 'vimage_alias': 'gfx12_asm_vimage_alias.s', 'vsample': 'gfx12_asm_vsample.s',
  'vdsdir': 'gfx12_asm_vdsdir.s', 'vdsdir_alias': 'gfx12_asm_vdsdir_alias.s',
  'exp': 'gfx12_asm_exp.s', 'wmma_w32': 'gfx12_asm_wmma_w32.s', 'wmma_w64': 'gfx12_asm_wmma_w64.s',
  'global_load_tr': 'gfx12_asm_global_load_tr.s',
  # NOTE: 'features' (gfx12_asm_features.s) tests DPP instruction variants which require separate format decoders
}

def parse_llvm_tests(text: str, gfx_prefix: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  pattern = rf'(?:{gfx_prefix}|W32|W64)[^:]*:.*?encoding:\s*\[(.*?)\]'
  pattern2 = rf'(?:{gfx_prefix}|W32|W64)[^:]*:\s*\[(0x[0-9a-fA-F,x\s]+)\]'
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text: continue
    for j in range(i, min(i + 3, len(lines))):
      if m := re.search(pattern, lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      elif m := re.search(pattern2, lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      else: continue
      if hex_bytes:
        try: tests.append((asm_text, bytes.fromhex(hex_bytes)))
        except ValueError: pass
      break
  return tests

def compile_asm_batch(instrs: list[str], mcpu: str, mattr: str = '+real-true16,+wavefrontsize32') -> list[bytes]:
  """Compile multiple instructions with a single llvm-mc call."""
  if not instrs: return []
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', f'-mattr={mattr}', '-show-encoding'],
                          input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True, timeout=30)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
  results = []
  for line in result.stdout.split('\n'):
    if 'encoding:' not in line: continue
    enc = line.split('encoding:')[1].strip()
    if enc.startswith('[') and enc.endswith(']'):
      results.append(bytes.fromhex(enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')))
  if len(results) != len(instrs): raise RuntimeError(f"expected {len(instrs)} encodings, got {len(results)}")
  return results

def matches_encoding(data: bytes, fmt) -> bool:
  """Check if instruction bytes match format's expected encoding bits."""
  if not hasattr(fmt, '_encoding') or fmt._encoding is None: return True
  bf, expected = fmt._encoding
  val = int.from_bytes(data[:fmt._size()], 'little')
  return ((val >> bf.lo) & bf.mask()) == expected

class TestLLVMBase(unittest.TestCase):
  """Base class for LLVM assembler tests."""
  tests: dict[str, list[tuple[str, bytes]]] = {}
  formats: dict[str, type] = {}
  gfx_prefix: str = ""
  mcpu: str = ""
  arch_name: str = ""

  @classmethod
  def _load_tests(cls, test_files: dict[str, str]):
    for name, filename in test_files.items():
      try:
        data = fetch(f"{LLVM_BASE}/{filename}").read_bytes()
        cls.tests[name] = parse_llvm_tests(data.decode('utf-8', errors='ignore'), cls.gfx_prefix)
      except Exception as e:
        print(f"Warning: couldn't fetch {filename}: {e}")
        cls.tests[name] = []

  def _test_disasm(self, name: str):
    """Test decoding instructions and verify disassembly produces correct bytes."""
    if name not in self.tests or not self.tests[name]: self.skipTest(f"No test data for {name}")
    fmt_cls = self.formats.get(name)
    if fmt_cls is None: self.skipTest(f"No format class for {name}")

    # Determine wave size from test name (w64 = wave64, otherwise wave32)
    wave_size = 64 if 'w64' in name else 32
    mattr = f'+real-true16,+wavefrontsize{wave_size}'

    to_test: list[tuple[str, bytes, str | None, str | None]] = []
    for asm_text, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size(): continue
      if not matches_encoding(data, fmt_cls): continue
      try:
        decoded = fmt_cls.from_bytes(data)
        if decoded.to_bytes()[:len(data)] != data:
          to_test.append((asm_text, data, None, "decode roundtrip failed"))
          continue
        to_test.append((asm_text, data, decoded.disasm(wave_size), None))
      except Exception as e:
        to_test.append((asm_text, data, None, f"exception: {e}"))

    disasm_strs = [(i, t[2]) for i, t in enumerate(to_test) if t[2] is not None]
    llvm_map = {}
    if disasm_strs:
      llvm_results = compile_asm_batch([s for _, s in disasm_strs], self.mcpu, mattr)
      llvm_map = {i: llvm_results[j] for j, (i, _) in enumerate(disasm_strs)}

    passed, failed, failures = 0, 0, []
    for idx, (asm_text, data, disasm_str, error) in enumerate(to_test):
      if error:
        failed += 1; failures.append(f"{error} for {data.hex()}")
      elif disasm_str is not None and idx in llvm_map:
        llvm_bytes = llvm_map[idx]
        if llvm_bytes == data: passed += 1
        else: failed += 1; failures.append(f"'{disasm_str}': expected={data.hex()} got={llvm_bytes.hex()}")

    print(f"{self.arch_name} {name.upper()} disasm: {passed} passed, {failed} failed")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertGreater(passed, 0, f"No tests passed for {name}")

class TestLLVMRDNA3(TestLLVMBase):
  """Test RDNA3 assembler against LLVM test vectors."""
  gfx_prefix, mcpu, arch_name = "GFX11", "gfx1100", "RDNA3"

  @classmethod
  def setUpClass(cls):
    from extra.assembly.amd.autogen.rdna3.ins import SOP1, SOP2, SOPC, SOPK, SOPP, VOP1, VOP2, VOP3, VOP3P, VOPC, VOPD, VINTERP, DS, SMEM, FLAT, MUBUF, MTBUF, MIMG, LDSDIR, EXP
    cls.formats = {
      'sop1': SOP1, 'sop2': SOP2, 'sopc': SOPC, 'sopk': SOPK, 'sopp': SOPP,
      'vop1': VOP1, 'vop2': VOP2, 'vopc': VOPC, 'vopcx': VOPC, 'vop3': VOP3, 'vop3p': VOP3P,
      'vinterp': VINTERP, 'vopd': VOPD, 'ds': DS, 'smem': SMEM, 'flat': FLAT,
      'mubuf': MUBUF, 'mtbuf': MTBUF, 'mimg': MIMG, 'wmma': VOP3P, 'ldsdir': LDSDIR, 'exp': EXP,
      'vop3_from_vop1': VOP3, 'vop3_from_vop2': VOP3, 'vop3_from_vopc': VOP3, 'vop3_from_vopcx': VOP3,
      'vop3_features': VOP3, 'vop3p_features': VOP3P, 'vopd_features': VOPD,
      'vop3_alias': VOP3, 'vop3p_alias': VOP3P, 'vopc_alias': VOPC, 'vopcx_alias': VOPC,
      'vinterp_alias': VINTERP, 'smem_alias': SMEM, 'mubuf_alias': MUBUF, 'mtbuf_alias': MTBUF,
    }
    cls._load_tests(RDNA3_TEST_FILES)

class TestLLVMRDNA4(TestLLVMBase):
  """Test RDNA4 assembler against LLVM test vectors."""
  gfx_prefix, mcpu, arch_name = "GFX12", "gfx1200", "RDNA4"

  @classmethod
  def setUpClass(cls):
    import extra.assembly.amd.autogen.rdna4.ins as rdna4
    get = lambda n: getattr(rdna4, n, None)
    cls.formats = {
      'sop1': get('SOP1'), 'sop2': get('SOP2'), 'sop2_alias': get('SOP2'), 'sopc': get('SOPC'),
      'sopk': get('SOPK'), 'sopk_alias': get('SOPK'), 'sopp': get('SOPP'),
      'vop1': get('VOP1'), 'vop2': get('VOP2'), 'vop2_aliases': get('VOP2'), 'vopc': get('VOPC'), 'vopcx': get('VOPC'),
      'vop3': get('VOP3'), 'vop3_aliases': get('VOP3'), 'vop3c': get('VOP3'), 'vop3cx': get('VOP3'),
      'vop3p': get('VOP3P'), 'vop3p_aliases': get('VOP3P'), 'vop3p_features': get('VOP3P'),
      'vopd': get('VOPD'), 'vopd_features': get('VOPD'),
      'vop3_from_vop1': get('VOP3'), 'vop3_from_vop2': get('VOP3'),
      'ds': get('VDS'), 'ds_alias': get('VDS'), 'smem': get('SMEM'), 'vinterp': get('VINTERP'), 'exp': get('VEXPORT'),
      'vbuffer_mubuf': get('VBUFFER'), 'vbuffer_mubuf_alias': get('VBUFFER'),
      'vbuffer_mtbuf': get('VBUFFER'), 'vbuffer_mtbuf_alias': get('VBUFFER'),
      'vdsdir': get('VDSDIR'), 'vdsdir_alias': get('VDSDIR'),
      'vflat': get('VFLAT'), 'vflat_alias': get('VFLAT'),
      'vscratch': get('VSCRATCH'), 'vscratch_alias': get('VSCRATCH'),
      'vimage': get('VIMAGE'), 'vimage_alias': get('VIMAGE'), 'vsample': get('VSAMPLE'),
      'wmma_w32': get('VOP3P'), 'wmma_w64': get('VOP3P'),
      'global_load_tr': get('VGLOBAL'),
    }
    cls._load_tests(RDNA4_TEST_FILES)

# Generate test methods dynamically
def _make_test(name):
  def test(self): self._test_disasm(name)
  return test

for name in RDNA3_TEST_FILES: setattr(TestLLVMRDNA3, f'test_{name}_disasm', _make_test(name))
for name in RDNA4_TEST_FILES: setattr(TestLLVMRDNA4, f'test_{name}_disasm', _make_test(name))

if __name__ == "__main__": unittest.main()
