import struct, ctypes, platform
from tinygrad.helpers import round_up, DEBUG, OSX
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE, PAGESIZE
from typing import Dict, List, Tuple, Callable

def addrof(fn): return ctypes.cast(fn, ctypes.c_void_p).value
def mkfn(addr:int, ret=None, *args): return ctypes.cast(addr, ctypes.CFUNCTYPE(ret, *args))
def mkuint32(addr:int): return ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint32)).contents
def mkstub(absaddr: int) -> bytes:
  if platform.machine() == "arm64":
    return b'\x50\x00\x00\x58\x00\x02\x1f\xd6' + struct.pack("<Q", absaddr) # ldr x16, +0x8; br x16; .bytes absaddr (x16 is scratch)
  elif platform.machine() == "x86_64":
    return b'\x48\xB8' + struct.pack("<Q", absaddr) + b'\xFF\xE0' # mov rax, absaddr; jmp rax (rax is func return value so we can use it here)
  else: raise NotImplementedError(f"architecture {platform.machine()} isn't supported")

ALIGNMENT = 32 # standartized alignment to simplify everything
STUB_SZ = len(mkstub(0))
JIT_PAGESIZE = 4*1024*1024 # 4MB. To avoid frequent reallocations
MAP_JIT = 0x0800 if OSX else 0x0

libc = ctypes.CDLL(None)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t)

class ElfLoader:
  def __init__(self):
    self.cursor, self.left = 0, 0
    self.external_symbols = {
      'sin': addrof(libc.sin), 'sinf': addrof(libc.sinf),
    }
  def alloc(self, size: int, ensure_only: bool=False) -> int:
    size = round_up(size, ALIGNMENT)
    if self.left < size:
      toalloc = round_up(max(size, JIT_PAGESIZE), PAGESIZE)
      self.cursor, self.left = libc.mmap(None, toalloc, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0), toalloc
    if ensure_only: size = 0
    self.left, self.cursor = self.left - size, self.cursor + size
    return self.cursor-size
  def load_elf(self, blob: bytes) -> Dict[str, Callable]:
    assert blob[0:7] == b"\x7FELF\x02\x01\x01", "invalid magic (little endian 64 bit ELF v1 expected)"
    e_type, e_machine, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum, _ = struct.unpack("<2HI3QI6H", blob[16:64])
    assert e_machine in {0x3e, 0xb7} and e_type == 0x01, "relocatable object file for aarch64 or x86 expected"
    # parse segments for easier later use
    alloc_size, nstubs = 0, 0
    progbits: Dict[int, bytes] = {}
    symtab: Dict[int, Tuple[int, List[Tuple[int, int, int, int]]]] = {} # (sh_link, List[st_name, st_info, st_shndx, st_value])
    strtab: Dict[int, bytes] = {}
    rela: Dict[int, Tuple[int, int, List[Tuple[int, int, int, int]]]] = {} # (sh_link, sh_info, List[r_offset, r_type, r_sym, r_addend])
    for i in range(e_shnum):
      off = e_shoff+i*e_shentsize
      sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack("<2I4Q2I2Q", blob[off:off+64])
      assert sh_addr == 0, f"segments with non-zero load addresses are not supported by this loader"
      assert sh_addralign <= ALIGNMENT, f"for simplicity alignment is hardcoded to be {ALIGNMENT}, but segment {i} asks for {sh_addralign}"
      if sh_type == 0x1: # SHT_PROGBITS
        if sh_flags & 0x2 == 0x2:
          alloc_size += round_up(sh_size, ALIGNMENT)
          progbits[i] = blob[sh_offset:sh_offset+sh_size] # assuming all is RX, shouldn't be anything writable
        elif DEBUG >= 2: print(f'skipped non-runtime progbits section {sh_name=} {sh_type=:#x} {sh_flags=:#x}') # stuff like .comment
      elif sh_type == 0x2: # SHT_SYMTAB
        _, syms = symtab[i] = (sh_link,[struct.unpack("<IbxHQ8x", blob[sh_offset+i:sh_offset+i+24]) for i in range(0, sh_size, sh_entsize)])
        for _, st_info, _, _ in syms:
          if st_info == 0x10: nstubs += 1
      elif sh_type == 0x3: # SHT_STRTAB
        strtab[i] = blob[sh_offset:sh_offset+sh_size]
      elif sh_type == 0x4: # SHT_RELA
        rela[i] = (sh_link,sh_info,[struct.unpack("<QIIq", blob[sh_offset+i:sh_offset+i+24]) for i in range(0, sh_size, sh_entsize)])
    alloc_size = round_up(nstubs*STUB_SZ, ALIGNMENT) + round_up(alloc_size, ALIGNMENT)
    # allocate and copy runtime sections
    loaded: Dict[int, int] = {}
    self.alloc(alloc_size, ensure_only=True)
    plt_start, plt_off = self.alloc(nstubs*STUB_SZ), 0
    if OSX: libc.pthread_jit_write_protect_np(False)
    for idx,blob in progbits.items():
      loaded[idx] = section = self.alloc(len(blob))
      ctypes.memmove(section, blob, len(blob))
    # resolve symbol names and addresses
    snames: Dict[Tuple[int, int], str] = {}
    saddrs: Dict[Tuple[int, int], int] = {}
    exports: Dict[str, Callable] = {}
    for i,(sh_link,syms) in symtab.items():
      cur_strtab = strtab[sh_link]
      for j,(st_name,st_info,st_shndx,st_value) in enumerate(syms):
        snames[(i,j)] = cname = cur_strtab[st_name:cur_strtab.find(b'\x00',st_name)].decode()
        if (load_base:=loaded.get(st_shndx, None)) is not None:
          saddrs[(i,j)] = load_base+st_value
          if st_info == 0x12: exports[cname] = mkfn(load_base+st_value, ctypes.c_uint64)
        elif st_info == 0x10:
          ctypes.memmove(plt_start+plt_off, mkstub(self.external_symbols[cname]), STUB_SZ)
          saddrs[(i,j)], plt_off = plt_start+plt_off, plt_off + STUB_SZ
    # patch relocations
    for sh_link,sh_info,relas in rela.values():
      base = loaded[sh_info]
      for r_offset,r_type,r_sym,r_addend in relas:
        tgt, ploc = saddrs[(sh_link,r_sym)]+r_addend, base+r_offset
        rel = tgt - ploc
        if r_type in {0x2, 0x4}: # x86 stuff
          assert abs(rel)<2**31-1, "relocation out of bounds"
          mkuint32(ploc).value = 2**32+rel if rel < 0 else rel
        elif r_type == 0x113: # R_AARCH64_ADR_PREL_PG_HI21
          tgt_pg, ploc_pg = tgt >> 12, ploc >> 12
          assert (tgt_pg-ploc_pg)>>21 == 0, f"adr out of bounds - {tgt_pg=:#x} {ploc_pg=:#x}"
          lo, hi = (tgt_pg-ploc_pg)&0b11,(tgt_pg-ploc_pg)>>2
          mkuint32(ploc).value |= lo<<29 | hi<<5
        elif r_type == 0x115:  # R_AARCH64_ADD_ABS_LO12_NC
          mkuint32(ploc).value |= (tgt&0xFFF)<<10
        elif r_type in {0x11a, 0x11b}: # R_AARCH64_CALL26
          assert abs(rel)&0b11 == 0 and abs(rel)&0xfc000000 == 0, f"bad jump/call - {rel=:#x}"
          mkuint32(ploc).value |= 2**26+(rel>>2) if rel < 0 else (rel>>2)
        elif r_type == 0x11c: # R_AARCH64_LDST16_ABS_LO12_NC
          assert tgt&0b1 == 0, f"unaligned 16-bit load {tgt=:#x}"
          mkuint32(ploc).value |= (tgt&0xFFF)<<9
        elif r_type == 0x11d: # R_AARCH64_LDST32_ABS_LO12_NC
          assert tgt&0b11 == 0, f"unaligned 32-bit load {tgt=:#x}"
          mkuint32(ploc).value |= (tgt&0xFFF)<<8
        elif r_type == 0x11e: # R_AARCH64_LDST64_ABS_LO12_NC
          assert tgt&0b111 == 0, f"unaligned 64-bit load {tgt=:#x}"
          mkuint32(ploc).value |= (tgt&0xFFF)<<7
        elif r_type == 0x12b: # R_AARCH64_LDST128_ABS_LO12_NC
          assert tgt&0b1111 == 0, f"unaligned 128-bit load {tgt=:#x}"
          mkuint32(ploc).value |= (tgt&0xFFF)<<6
        else: raise NotImplementedError(f"Encountered unknown relocation type {r_type:#x}")
    # clear instruction cache, on macos switch page to RX
    if OSX:
      libc.pthread_jit_write_protect_np(True)
      libc.sys_icache_invalidate(ctypes.c_void_p(plt_start), ctypes.c_size_t(alloc_size))
    else:
      # FIXME: linux has clear cache syscall but there isn't a libc wrapper and number changes for x86/arm - maybe there is an easier way?
      #raise RuntimeError("linux is not supported yet")
      pass
    return exports