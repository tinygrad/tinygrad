import ctypes
from dataclasses import dataclass
import tinygrad.runtime.autogen.comgr as comgr
from tinygrad.runtime.support.compiler_amd import check

@dataclass
class InstrCtx:
  pc:int=0
  inst:str=""

@comgr.amd_comgr_create_disassembly_info.argtypes[2]
def instr_cb(text, user_data):
  c = ctypes.cast(user_data, ctypes.POINTER(ctypes.py_object)).contents.value
  c.inst = ctypes.string_at(text).decode("utf-8","replace").strip()
  return comgr.AMD_COMGR_STATUS_SUCCESS

# nop callback
@comgr.amd_comgr_create_disassembly_info.argtypes[3]
def addr_cb(*args): return comgr.AMD_COMGR_STATUS_SUCCESS

def comgr_get_address_table(lib:bytes) -> dict[int, tuple[str, int]]:
  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_EXECUTABLE, ctypes.byref(data_src:=comgr.amd_comgr_data_t())))
  lib_buf = ctypes.create_string_buffer(lib, len(lib))
  check(comgr.amd_comgr_set_data(data_src, len(lib), lib_buf))
  check(comgr.amd_comgr_get_data_isa_name(data_src, isa_sz:=ctypes.c_size_t(128), isa:=(ctypes.c_char*isa_sz.value)()))

  @comgr.amd_comgr_create_disassembly_info.argtypes[1]
  def memory_cb(from_addr, to, size, _):
    base, buf_len = ctypes.addressof(lib_buf), len(lib_buf)
    start = int(from_addr) - base
    if start < 0 or start >= buf_len: return 0
    ctypes.memmove(to, base + start, n:=min(int(size), buf_len - start))
    return n

  info_src = comgr.amd_comgr_disassembly_info_t()
  check(comgr.amd_comgr_create_disassembly_info(ctypes.cast(isa, ctypes.POINTER(ctypes.c_char)), memory_cb, instr_cb, addr_cb, info_src))

  @comgr.amd_comgr_iterate_symbols.argtypes[1]
  def sym_callback(sym, udata):
    check(comgr.amd_comgr_symbol_get_info(sym, comgr.AMD_COMGR_SYMBOL_INFO_TYPE, ctypes.byref(sym_type:=ctypes.c_int())))
    if sym_type.value != comgr.AMD_COMGR_SYMBOL_TYPE_FUNC: return comgr.AMD_COMGR_STATUS_SUCCESS
    check(comgr.amd_comgr_symbol_get_info(sym, comgr.AMD_COMGR_SYMBOL_INFO_VALUE, ctypes.byref(vaddr:=ctypes.c_uint64())))
    check(comgr.amd_comgr_symbol_get_info(sym, comgr.AMD_COMGR_SYMBOL_INFO_SIZE, ctypes.byref(size:=ctypes.c_uint64())))
    check(comgr.amd_comgr_map_elf_virtual_address_to_code_object_offset(data_src, vaddr.value, ctypes.byref(offset:=ctypes.c_uint64()),
                                                                        ctypes.byref(ctypes.c_uint64()), ctypes.byref(nobits:=ctypes.c_bool())))
    check(nobits.value)
    base = ctypes.addressof(lib_buf)
    pc = base + offset.value
    end = pc + size.value
    addr_table = ctypes.cast(udata, ctypes.POINTER(ctypes.py_object)).contents.value
    instr_ref = ctypes.py_object(ctx:=InstrCtx())
    instr_ptr = ctypes.cast(ctypes.pointer(instr_ref), ctypes.c_void_p)
    while pc < end:
      size_read = ctypes.c_uint64(0)
      ctx.pc = pc
      st = comgr.amd_comgr_disassemble_instruction(info_src, ctypes.c_uint64(pc), instr_ptr, ctypes.byref(size_read))
      if st == comgr.AMD_COMGR_STATUS_SUCCESS and size_read.value:
        rel = (pc - base) - offset.value
        addr_table[vaddr.value + rel] = (ctx.inst, int(size_read.value))
        pc += size_read.value
      else: # don't inf loop if comgr fails
        b = ctypes.c_ubyte.from_buffer(lib_buf, pc - base).value
        addr_table[vaddr.value + (pc - base - offset.value)] = (f"DISASSEMBLER ISSUE 0x{b:02x}", 1)
        pc += 1
    return comgr.AMD_COMGR_STATUS_SUCCESS
  addr_table:dict[int, tuple[str, int]] = {}
  check(comgr.amd_comgr_iterate_symbols(data_src, sym_callback, ctypes.cast(ctypes.pointer(ctypes.py_object(addr_table)), ctypes.c_void_p)))
  return addr_table
