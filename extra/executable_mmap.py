from mmap import mmap, PROT_EXEC, PROT_WRITE, MAP_PRIVATE
import ctypes
import sys
from hexdump import hexdump

if sys.platform == "darwin":
    from extra.clang_parsers import MyMachO
    from mmap import MAP_JIT 
    libc = ctypes.CDLL("libc.dylib")
else:
    from extra.clang_parsers import ELFParser

def allocate_executable_memory(data, name):
    if sys.platform == "darwin":
        libc.pthread_jit_write_protect_np(0)
        mem = mmap(-1, len(data), flags=MAP_PRIVATE | MAP_JIT, prot=PROT_WRITE | PROT_EXEC)
        mem.write(data)
        libc.pthread_jit_write_protect_np(1)
    else:
        mem = mmap(-1, len(data), flags=MAP_PRIVATE, prot=PROT_WRITE | PROT_EXEC)
        mem.write(data)

    if sys.platform == "darwin":
        macho = MyMachO(data)
        offset, symbol_table = macho.extract_offset_and_symbols()
        symbol_addr = symbol_table["_"+name] + offset
    else: 
        elfparser = ELFParser(data)
        offset = elfparser.section_headers[".text"][0]
        symbol_addr = elfparser.symbol_table[name]["value"] + offset
    
    base = ctypes.addressof(ctypes.c_char.from_buffer(mem))
    func_type = ctypes.CFUNCTYPE(ctypes.c_int)
    fn = func_type(base + symbol_addr)

    return fn, mem
