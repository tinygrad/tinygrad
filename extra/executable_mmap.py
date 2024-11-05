from mmap import mmap, PROT_EXEC, PROT_WRITE, PROT_READ, MAP_PRIVATE, MAP_ANONYMOUS
import ctypes
import sys

if sys.platform == "darwin":
  from extra.clang_parsers import MyMachO
  from mmap import MAP_JIT
  libc = ctypes.CDLL("libc.dylib")
else:
  from extra.clang_parsers import ELFParser

def allocate_executable_memory(data, name):
  print(name)
  if sys.platform == "darwin":
    macho = MyMachO(data)
    offset, symbol_table = macho.extract_offset_and_symbols()
    symbol_addr = symbol_table["_"+name] + offset
  else:
    elfparser = ELFParser(data)
    offset, sec_size  = elfparser.section_headers[".text"]
    symbol_offset = elfparser.symbol_table[name]["value"]
    symbol_size = elfparser.symbol_table[name]["size"]
    symbol_addr = symbol_offset + offset
    func_data = data[symbol_addr:symbol_addr+symbol_size]
    func_data = func_data.replace(b"\x00\x00\x00\x94", b"\x1f\x20\x03\xd5")

  if sys.platform == "darwin":
    libc.pthread_jit_write_protect_np(0)
    mem = mmap(-1, len(data) * 2, flags=MAP_PRIVATE | MAP_JIT, prot=PROT_WRITE | PROT_EXEC)
    mem.write(data)
    libc.pthread_jit_write_protect_np(1)
  else:
    mem = mmap(-1, len(func_data), flags=MAP_PRIVATE | MAP_ANONYMOUS, prot=PROT_WRITE | PROT_EXEC | PROT_READ)
    mem.write(func_data)

  base = ctypes.addressof(ctypes.c_char.from_buffer(mem))
  func_type = ctypes.CFUNCTYPE(ctypes.c_int)
  fn = func_type(base)

  return fn, mem
