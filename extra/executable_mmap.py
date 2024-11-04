from mmap import mmap, PROT_EXEC, PROT_WRITE
from mmap import MAP_JIT, MAP_PRIVATE
from extra.macho import extract_offset_and_symbols
import ctypes

libc = ctypes.CDLL("libc.dylib")

def allocate_executable_memory(data, name):
    libc.pthread_jit_write_protect_np(0)
    mem = mmap(-1, len(data), flags=MAP_PRIVATE | MAP_JIT, prot=PROT_WRITE | PROT_EXEC)
    mem.write(data)
    libc.pthread_jit_write_protect_np(1)

    offset, symbol_table = extract_offset_and_symbols(data)    
    base = ctypes.addressof(ctypes.c_char.from_buffer(mem)) + offset

    functions = {} 
    func_type = ctypes.CFUNCTYPE(ctypes.c_int)
    for symbol_name, symbol_addr in symbol_table:
        functions[symbol_name[1:]] = func_type(base + symbol_addr)

    return functions[name], mem