import unittest
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.device import CompileError
from tinygrad.runtime.autogen.amdgpu_kd import KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET as CO_OFFSET
from extra.assembly.amd.elf import create_elf
from extra.assembly.amd.test.helpers import TARGET_TO_ARCH

def assert_rodata_eq(cmp:bytes, ref:bytes):
  _, ref_sections, __ = elf_loader(ref)
  _, cmp_sections, __ = elf_loader(cmp)
  s_ref = bytearray(next(s.content for s in ref_sections if s.name == ".rodata"))
  s_cmp = bytearray(next(s.content for s in cmp_sections if s.name == ".rodata"))
  # zero out kernel_code_entry_byte_offset (8 bytes), our ELF layout is different from LLVM
  s_ref[CO_OFFSET:CO_OFFSET+8] = s_cmp[CO_OFFSET:CO_OFFSET+8] = b'\x00' * 8
  assert s_ref == s_cmp, f"{s_cmp.hex()} != {s_ref.hex()}"

class TestRodata(unittest.TestCase):

  def simple_test(self, target:str, **kwargs):
    arch = TARGET_TO_ARCH[target]
    if arch == "cdna": from extra.assembly.amd.autogen.cdna.ins import s_nop, s_endpgm
    else: from extra.assembly.amd.autogen.rdna3.ins import s_nop, s_endpgm

    hsa = {'kernarg_size':8, 'user_sgpr_kernarg_segment_ptr':1, 'next_free_vgpr':4, 'next_free_sgpr':92, 'user_sgpr_count':2}
    for k,v in kwargs.items(): hsa[k] = v

    wavefront_size = 64 if arch == "cdna" else 32
    if arch != "cdna": hsa['wavefront_size32'] = 1

    insts = [s_nop(i) for i in range(1, 10)]+[s_endpgm()]
    prg = b"".join(inst.to_bytes() for inst in insts)
    our_lib = create_elf(prg, hsa, arch)

    # LLVM requires a YAML style boilerplate section to create the ELF
    src = '\n'.join([
      '\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{target}"',
      '\t.protected\tkernel', '\t.globl\tkernel', '\t.p2align\t8', '\t.type\tkernel,@function', 'kernel:',
      *[inst.disasm() for inst in insts],
      '\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0', '\t.amdhsa_kernel kernel',
      *[f'\t\t.amdhsa_{k} {v}' for k, v in hsa.items()],
      '\t.end_amdhsa_kernel', '\t.text', '.Lfunc_end0:', '\t.size\tkernel, .Lfunc_end0-kernel',
      '\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:',
      f'    .group_segment_fixed_size: {hsa.get("group_segment_fixed_size", 0)}', '    .kernarg_segment_align: 8',
      '    .kernarg_segment_size: 0', '    .max_flat_workgroup_size: 128', '    .name: kernel',
      '    .private_segment_fixed_size: 0', f'    .sgpr_count: {hsa.get("next_free_sgpr", 0)}', '    .symbol: kernel.kd',
      f'    .vgpr_count: {hsa["next_free_vgpr"]}', f'    .wavefront_size: {wavefront_size}', f'amdhsa.target: amdgcn-amd-amdhsa--{target}',
      'amdhsa.version:', '  - 1', '  - 2', '...', '\t.end_amdgpu_metadata'])
    llvm_lib = HIPCompiler(target).compile(src)

    assert_rodata_eq(our_lib, llvm_lib)

  def test_rdna(self):
    self.simple_test("gfx1100")
    for i in range(256): self.simple_test("gfx1100", next_free_vgpr=i)

  def test_rdna4(self): self.simple_test("gfx1200")

  def test_cdna(self):
    # llvm docs for sgpr granule are inconsistent with actual behavior, match the real values
    self.simple_test("gfx942", accum_offset=4, next_free_sgpr=0)
    for i in range(1, 103): self.simple_test("gfx942", accum_offset=4, next_free_sgpr=i)
    for i in range(1, 64):
      self.simple_test("gfx942", accum_offset=i*4, next_free_vgpr=i*4+24)

  def test_vgpr_out_of_range(self):
    with self.assertRaises(CompileError):
      self.simple_test("gfx1100", next_free_vgpr=513)

  def test_accum_offset_min(self):
    with self.assertRaises(CompileError):
      self.simple_test("gfx950", accum_offset=0)

if __name__ == "__main__":
  unittest.main()
