import struct, pickle
from tinygrad.helpers import DEBUG
from typing import Dict, List, Tuple

def patchuint32(blob: bytearray, ploc: int, new: int):
  blob[ploc:ploc+4] = struct.pack("<I", struct.unpack("<I", blob[ploc:ploc+4])[0] | new)

def fixup_relocations(blob: bytes) -> bytes:
  ret = bytearray()
  assert blob[0:7] == b"\x7FELF\x02\x01\x01", "invalid magic (little endian 64 bit ELF v1 expected)"
  e_type, e_machine, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum, _ = struct.unpack("<2HI3QI6H", blob[16:64])
  assert e_machine in {0x3e, 0xb7} and e_type == 0x01, "relocatable object file for aarch64 or x86 expected"
  loaded: Dict[int, int] = {}
  symtab: Dict[int, Tuple[int, List[Tuple[int, int, int, int]]]] = {} # (sh_link, List[st_name, st_info, st_shndx, st_value])
  strtab: Dict[int, bytes] = {}
  rela: Dict[int, Tuple[int, int, List[Tuple[int, int, int, int]]]] = {} # (sh_link, sh_info, List[r_offset, r_type, r_sym, r_addend])
  for i in range(e_shnum):
    off = e_shoff+i*e_shentsize
    sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack("<2I4Q2I2Q", blob[off:off+64])
    assert sh_addr == 0, f"segments with non-zero load addresses are not supported by this loader"
    if sh_type == 0x1: # SHT_PROGBITS
      if sh_flags & 0x2 == 0x2:
        if (extra := len(ret) % sh_addralign) > 0: ret += b'\x00' * (sh_addralign-extra)
        loaded[i] = len(ret)
        ret += blob[sh_offset:sh_offset+sh_size] # assuming all is RX, shouldn't be anything writable
      elif DEBUG >= 2: print(f'skipped non-runtime progbits section {sh_name=} {sh_type=:#x} {sh_flags=:#x}') # stuff like .comment
    elif sh_type == 0x2: # SHT_SYMTAB
      symtab[i] = (sh_link,[struct.unpack("<IbxHQ8x", blob[sh_offset+i:sh_offset+i+24]) for i in range(0, sh_size, sh_entsize)])
    elif sh_type == 0x3: # SHT_STRTAB
      strtab[i] = blob[sh_offset:sh_offset+sh_size]
    elif sh_type == 0x4: # SHT_RELA
      rela[i] = (sh_link,sh_info,[struct.unpack("<QIIq", blob[sh_offset+i:sh_offset+i+24]) for i in range(0, sh_size, sh_entsize)])
  snames: Dict[Tuple[int, int], str] = {}
  saddrs: Dict[Tuple[int, int], int] = {}
  exports: Dict[str, int] = {}
  for i,(sh_link,syms) in symtab.items():
    cur_strtab = strtab[sh_link]
    for j,(st_name,st_info,st_shndx,st_value) in enumerate(syms):
      snames[(i,j)] = cname = cur_strtab[st_name:cur_strtab.find(b'\x00',st_name)].decode()
      if (load_base:=loaded.get(st_shndx, None)) is not None:
        saddrs[(i,j)] = load_base+st_value
        if st_info == 0x12: exports[cname] = load_base+st_value
  for sh_link,sh_info,relas in rela.values():
    base = loaded[sh_info]
    for r_offset,r_type,r_sym,r_addend in relas:
      tgt, ploc = saddrs[(sh_link,r_sym)]+r_addend, base+r_offset
      rel = tgt - ploc
      if r_type in {0x2, 0x4}: # x86 stuff
        assert abs(rel)<2**31-1, "relocation out of bounds"
        patchuint32(ret, ploc, 2**32+rel if rel < 0 else rel)
      elif r_type == 0x113: # R_AARCH64_ADR_PREL_PG_HI21
        tgt_pg, ploc_pg = tgt >> 12, ploc >> 12
        assert (tgt_pg-ploc_pg)>>21 == 0, f"adr out of bounds - {tgt_pg=:#x} {ploc_pg=:#x}"
        lo, hi = (tgt_pg-ploc_pg)&0b11,(tgt_pg-ploc_pg)>>2
        patchuint32(ret, ploc, lo<<29 | hi<<5)
      elif r_type == 0x115:  # R_AARCH64_ADD_ABS_LO12_NC
        patchuint32(ret, ploc, (tgt&0xFFF)<<10)
      elif r_type in {0x11a, 0x11b}: # R_AARCH64_CALL26
        assert abs(rel)&0b11 == 0 and abs(rel)&0xfc000000 == 0, f"bad jump/call - {rel=:#x}"
        patchuint32(ret, ploc, 2**26+(rel>>2) if rel < 0 else (rel>>2))
      elif r_type == 0x11c: # R_AARCH64_LDST16_ABS_LO12_NC
        assert tgt&0b1 == 0, f"unaligned 16-bit load {tgt=:#x}"
        patchuint32(ret, ploc, (tgt&0xFFF)<<9)
      elif r_type == 0x11d: # R_AARCH64_LDST32_ABS_LO12_NC
        assert tgt&0b11 == 0, f"unaligned 32-bit load {tgt=:#x}"
        patchuint32(ret, ploc, (tgt&0xFFF)<<8)
      elif r_type == 0x11e: # R_AARCH64_LDST64_ABS_LO12_NC
        assert tgt&0b111 == 0, f"unaligned 64-bit load {tgt=:#x}"
        patchuint32(ret, ploc, (tgt&0xFFF)<<7)
      elif r_type == 0x12b: # R_AARCH64_LDST128_ABS_LO12_NC
        assert tgt&0b1111 == 0, f"unaligned 128-bit load {tgt=:#x}"
        patchuint32(ret, ploc, (tgt&0xFFF)<<6)
      else: raise NotImplementedError(f"Encountered unknown relocation type {r_type:#x}")
  return pickle.dumps((exports, bytes(ret)))