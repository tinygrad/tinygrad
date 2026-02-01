# minimal amdgpu elf packer
import ctypes
from tinygrad.helpers import ceildiv, round_up
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc

def put(dst:bytearray, off:int, data:bytes) -> None:
  end = off + len(data)
  if end > len(dst): raise ValueError("write past end of buffer")
  dst[off:end] = data

def pack_hsaco(prg:bytes, kd:dict, arch:str) -> bytes:
  is_cdna, is_rdna4 = arch == "cdna", arch == "rdna4"
  text_offset = round_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
  rodata_offset = text_offset + len(prg)

  # ** pack rodata object
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
  desc.group_segment_fixed_size = kd.get('group_segment_fixed_size', 0)
  desc.private_segment_fixed_size = kd.get('private_segment_fixed_size', 0)
  desc.kernarg_size = kd.get('kernarg_size', 0)
  desc.kernel_code_entry_byte_offset = text_offset-rodata_offset
  # rsrc1
  vgpr_granule = max(0, (kd['next_free_vgpr'] + 7) // 8 - 1)
  # CDNA: add 6 for VCC(2) + FLAT_SCRATCH(2) + XNACK_MASK(2)
  # next_free_sgpr is unused in RDNA
  sgpr_granule = max(0, ceildiv(kd['next_free_sgpr'] + 6, 8) - 1) if is_cdna else 0
  desc.compute_pgm_rsrc1 = (vgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT |
                            sgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT |
                            kd.get('float_round_mode_32', 0) << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32_SHIFT |
                            kd.get('float_round_mode_16_64', 0) << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64_SHIFT |
                            kd.get('float_denorm_mode_32', 0) << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32_SHIFT |
                            kd.get('float_denorm_mode_16_64', 3) << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64_SHIFT |
                            kd.get('dx10_clamp', 0 if is_rdna4 else 1) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP_SHIFT |
                            kd.get('ieee_mode', 0 if is_rdna4 else 1) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE_SHIFT |
                            kd.get('fp16_overflow', 0) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL_SHIFT |
                            (0 if is_cdna else kd.get('workgroup_processor_mode', 1)) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE_SHIFT |
                            (0 if is_cdna else kd.get('memory_ordered', 1)) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED_SHIFT |
                            (0 if is_cdna else kd.get('forward_progress', 0)) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_FWD_PROGRESS_SHIFT)
  # rsrc2
  desc.compute_pgm_rsrc2 = (kd.get('enable_private_segment', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT |
                            kd.get('user_sgpr_count', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT |
                            kd.get('system_sgpr_workgroup_id_x', 1) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT |
                            kd.get('system_sgpr_workgroup_id_y', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT |
                            kd.get('system_sgpr_workgroup_id_z', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT |
                            kd.get('system_sgpr_workgroup_info', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO_SHIFT |
                            kd.get('system_vgpr_workitem_id', 0) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT)
  # rsrc3
  if is_cdna:
    accum_offset = kd.get('accum_offset', 0)
    amdhsa_accum_offset = ((accum_offset // 4) - 1) & amdgpu_kd.COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET if accum_offset else 0
    desc.compute_pgm_rsrc3 = amdhsa_accum_offset << amdgpu_kd.COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET_SHIFT
  else:
    desc.compute_pgm_rsrc3 = kd.get('shared_vgpr_count', 0) << amdgpu_kd.COMPUTE_PGM_RSRC3_GFX10_GFX11_SHARED_VGPR_COUNT_SHIFT
  # kernel code properties
  desc.kernel_code_properties = (kd.get('user_sgpr_dispatch_ptr', 0) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR_SHIFT |
                                 kd.get('user_sgpr_queue_ptr', 0) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR_SHIFT |
                                 kd.get('user_sgpr_kernarg_segment_ptr', 0) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT |
                                 kd.get('user_sgpr_dispatch_id', 0) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID_SHIFT |
                                 kd.get('user_sgpr_private_segment_size',0) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT |
                                 kd.get('wavefront_size32', 0 if is_cdna else 1) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT |
                                 kd.get('uses_dynamic_stack', 0) << amdgpu_kd.KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT)
  rodata = bytes(desc)

  # ** pack elf sections
  sh_names:list[int] = []
  strtab = bytearray(b"\x00")
  for name in [".text", ".rodata", ".strtab"]:
    sh_names.append(len(strtab))
    strtab += name.encode("ascii") + b"\x00"

  rodata_offset = round_up(text_offset+(text_size:=len(prg)), hsa.AMD_KERNEL_CODE_ALIGN_BYTES)
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
  put(elf, text_offset, prg)
  put(elf, rodata_offset, rodata)
  put(elf, strtab_offset, strtab)
  put(elf, shdr_offset, bytes(shdrs))
  return bytes(elf)
