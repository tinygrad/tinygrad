import ctypes
from tinygrad.runtime.autogen import llvm
from tinygrad.runtime.support.elf import elf_loader

# to pass NULL to callbacks
llvm.LLVMCreateDisasmCPUFeatures.argtypes = llvm.LLVMCreateDisasmCPUFeatures.argtypes[:5] + [ctypes.c_void_p, ctypes.c_void_p]

def comgr_get_address_table(lib:bytes) -> dict[int, tuple[str, int]]:
  cpu = "gfx1100"

  llvm.LLVMInitializeAMDGPUTargetInfo()
  llvm.LLVMInitializeAMDGPUTargetMC()
  llvm.LLVMInitializeAMDGPUAsmParser()
  llvm.LLVMInitializeAMDGPUDisassembler()
  ctx = llvm.LLVMCreateDisasmCPUFeatures("amdgcn-amd-amdhsa".encode(), cpu.encode(), "".encode(), None, 0, None, None)

  image, sections, relocs = elf_loader(lib)
  text = next((sh.header for sh in sections if sh.name == ".text"), -1)
  off, sz = text.sh_addr, text.sh_size

  addr_table:dict[int, tuple[str, int]] = {}
  out = ctypes.create_string_buffer(128)
  cur_off = off
  while cur_off < sz + off:
    view = (ctypes.c_ubyte * ((sz + off) - cur_off)).from_buffer_copy(memoryview(image)[cur_off:])
    instr_sz = llvm.LLVMDisasmInstruction(ctx, view, ctypes.c_uint64(len(view)), ctypes.c_uint64(0), out, ctypes.c_size_t(128))
    addr_table[cur_off] = (out.value.decode("utf-8", "replace").strip(), instr_sz)
    cur_off += instr_sz
  return addr_table
