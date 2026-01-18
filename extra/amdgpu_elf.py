# minimal amdgpu elf packer
import ctypes
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc

def align_up(x:int, a:int) -> int: return (x + (a - 1)) & ~(a - 1)

def put(dst:bytearray, off:int, data:bytes) -> None:
  end = off + len(data)
  if end > len(dst): raise ValueError("write past end of buffer")
  dst[off:end] = data

def pack_hsaco(prg:bytes, kd:dict) -> bytes:
  text = prg + b'\x00' * ((hsa.AMD_ISA_ALIGN_BYTES - len(prg) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES)
  text_offset = align_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
  rodata_offset = text_offset + len(text)

  # ** pack rodata object
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
  desc.group_segment_fixed_size = kd.get('group_segment_fixed_size', 0)
  desc.private_segment_fixed_size = kd.get('private_segment_fixed_size', 0)
  desc.kernarg_size = kd.get('kernarg_size', 0)
  desc.kernel_code_entry_byte_offset = text_offset-rodata_offset
  # rsrc1
  vgpr_granule = max(0, (kd['next_free_vgpr'] + 7) // 8 - 1)
  reserved1 = (kd.get('workgroup_processor_mode', 1) << 3) | (kd.get('memory_ordered', 1) << 4)
  desc.compute_pgm_rsrc1 = (vgpr_granule << hsa.AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT |
                            3 << hsa.AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT |
                            reserved1 << hsa.AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT)
  # rsrc2
  desc.compute_pgm_rsrc2 = (kd.get('user_sgpr_count', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT |
                            kd.get('system_sgpr_workgroup_id_x', 1) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT |
                            kd.get('system_sgpr_workgroup_id_y', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT |
                            kd.get('system_sgpr_workgroup_id_z', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT)
  # rsrc3, only gfx90a uses this
  amdhsa_accum_offset = kd.get('amdhsa_accum_offset', 0) & amdgpu_kd.COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET
  desc.compute_pgm_rsrc3 = (amdhsa_accum_offset << amdgpu_kd.COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET_SHIFT)
  # code properties, different for every arch
  desc.kernel_code_properties = (kd.get('user_sgpr_kernarg_segment_ptr', 0) << hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT |
                                 kd.get('uses_dynamic_stack', 0) << hsa.AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT |
                                 kd.get('wavefront_size32', 1) << hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT)
  rodata = bytes(desc)

  # ** pack elf sections
  sh_names:list[int] = []
  strtab = bytearray(b"\x00")
  for name in [".text", ".rodata", ".strtab"]:
    sh_names.append(len(strtab))
    strtab += name.encode("ascii") + b"\x00"

  rodata_offset = align_up(text_offset+(text_size:=len(text)), hsa.AMD_KERNEL_CODE_ALIGN_BYTES)
  strtab_offset = rodata_offset+(rodata_size:=len(rodata))
  shdr_offset   = strtab_offset+(strtab_size:=len(strtab))

  sections = [(libc.SHT_PROGBITS, libc.SHF_ALLOC | libc.SHF_EXECINSTR, text_offset, text_offset, text_size),
              (libc.SHT_PROGBITS, libc.SHF_ALLOC, rodata_offset, rodata_offset, rodata_size),
              (libc.SHT_STRTAB, 0, 0, strtab_offset, strtab_size)]
  shdrs = (libc.Elf64_Shdr * len(sections))()
  for i,s in enumerate(sections): shdrs[i] = libc.Elf64_Shdr(sh_names[i], *s)

  ehdr = libc.Elf64_Ehdr()
  ehdr.e_shoff, ehdr.e_shnum, ehdr.e_shstrndx = shdr_offset, len(sections), 2

  elf = bytearray(shdr_offset + ctypes.sizeof(shdrs))
  put(elf, 0, bytes(ehdr))
  put(elf, text_offset, text)
  put(elf, rodata_offset, rodata)
  put(elf, strtab_offset, strtab)
  put(elf, shdr_offset, bytes(shdrs))
  return bytes(elf)
