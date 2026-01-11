import os
os.environ["AMD"] = "1"
os.environ["DEBUG"] = "2"

from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo

from extra.assembly.amd.autogen.rdna4.ins import *

import ctypes
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc

def pack_kernel_descriptor(text_offset:int, kd:dict) -> bytes:
  # Pack compute_pgm_rsrc1 using hsa constants
  vgpr_granule = max(0, (kd['next_free_vgpr'] + 7) // 8 - 1)
  reserved1 = (kd.get('workgroup_processor_mode', 1) << 3) | (kd.get('memory_ordered', 1) << 4)
  compute_pgm_rsrc1 = (vgpr_granule << hsa.AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT |
                       3 << hsa.AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT |
                       reserved1 << hsa.AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT)

  # Pack compute_pgm_rsrc2 using hsa constants
  compute_pgm_rsrc2 = (kd.get('user_sgpr_count', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT |
                       kd.get('system_sgpr_workgroup_id_x', 1) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT |
                       kd.get('system_sgpr_workgroup_id_y', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT |
                       kd.get('system_sgpr_workgroup_id_z', 0) << hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT)

  # Pack kernel_code_properties using hsa constants
  kernel_code_properties = (kd.get('user_sgpr_kernarg_segment_ptr', 0) << hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT |
                            kd.get('uses_dynamic_stack', 0) << hsa.AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT |
                            kd['wavefront_size32'] << hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT)

  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
  desc.group_segment_fixed_size = kd.get('group_segment_fixed_size', 0)
  desc.private_segment_fixed_size = kd.get('private_segment_fixed_size', 0)
  desc.kernarg_size = kd.get('kernarg_size', 0)
  desc.kernel_code_entry_byte_offset = text_offset
  desc.compute_pgm_rsrc1 = compute_pgm_rsrc1
  desc.compute_pgm_rsrc2 = compute_pgm_rsrc2
  desc.kernel_code_properties = kernel_code_properties
  return bytes(desc)

def align_up(x:int, a:int) -> int: return (x + (a - 1)) & ~(a - 1)

def put(dst:bytearray, off:int, data:bytes) -> None:
  end = off + len(data)
  if end > len(dst): raise ValueError("write past end of buffer")
  dst[off:end] = data

def pack_elf(text:bytes, rodata:bytes) -> bytes:
  sh_names:list[int] = []
  strtab = bytearray(b"\x00")
  for name in [".text", ".rodata", ".strtab"]:
    sh_names.append(len(strtab))
    strtab += name.encode("ascii") + b"\x00"

  text_offset   = align_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
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

def build_hsaco(insts:list[Inst], kd:dict) -> bytes:
  text = b"".join(i.to_bytes() for i in insts)
  text_padded = text + b'\x00' * ((hsa.AMD_ISA_ALIGN_BYTES - len(text) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES)
  text_offset = align_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
  rodata_offset = text_offset + len(text_padded)
  return pack_elf(text_padded, pack_kernel_descriptor(text_offset-rodata_offset, kd))

if __name__ == "__main__":
  kd = {"group_segment_fixed_size":0, "private_segment_fixed_size":0, "kernarg_size":8, "next_free_vgpr":10, "next_free_sgpr":10, "memory_ordered":1,
        "system_sgpr_workgroup_id_x":1, "system_sgpr_workgroup_id_y":0, "system_sgpr_workgroup_id_z":0, "wavefront_size32":1, "forward_progress":0,
        "user_sgpr_kernarg_segment_ptr":1, "user_sgpr_count":2, "workgroup_processor_mode":1, "uses_dynamic_stack":0}

  RSRC_OOB_DISABLE    = 1 << 29
  RSRC_FORMAT_NONZERO = 1 << 17
  RSRC_RAW_UNBOUNDED  = RSRC_OOB_DISABLE | RSRC_FORMAT_NONZERO

  class Kernel:
    def __init__(self): self.insts = []
    def emit(self, s:Inst): self.insts.append(s)

  def custom_rdna4(C):
    lidx = UOp.special(1, "lidx0")
    gidx = UOp.special(1, "gidx0")

    k = Kernel()
    k.emit(s_load_b64(sdata=s[4:5], sbase=s[0:1], ioffset=0x0, soffset=NULL))
    k.emit(s_wait_kmcnt(0x0))
    k.emit(s_mov_b32(s[6], 1))
    k.emit(s_mov_b32(s[7], RSRC_RAW_UNBOUNDED))
    k.emit(v_mov_b32_e32(v[1], 2.))
    k.emit(buffer_store_b32(v[1], soffset=NULL, ioffset=0x0, rsrc=s[4:7], format=1))
    k.emit(s_endpgm())

    lib = build_hsaco(k.insts, kd)
    src = "\n".join(i.disasm() for i in k.insts)

    sink = UOp.sink(C.base, lidx, gidx, arg=KernelInfo(name="test"))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                                 UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=())

  C = Tensor([0.,]).realize()
  C = Tensor.custom_kernel(C, fxn=custom_rdna4)[0].realize()
  print(C.numpy())
  assert C.item() == 2
  print("ASM passed")
