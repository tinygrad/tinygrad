import ctypes
from tinygrad.renderer.amd import InstDecodeError, decode_inst

def run_asm(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, args_ptr:int, rsrc2:int=0x19c,
            scratch_size:int=0, arch:str="rdna3", user_data:list[int]|None=None) -> int:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  while offset < len(code):
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError: break
    if offset + inst.size() > len(code): raise RuntimeError(f"truncated instruction at {offset:#x}")
    offset += inst.size()
  return 0
