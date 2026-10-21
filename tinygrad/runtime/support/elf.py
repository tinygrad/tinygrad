import struct, ctypes
from dataclasses import dataclass
from tinygrad.helpers import getbits, i2u, unwrap
from tinygrad.runtime.autogen import libc

@dataclass(frozen=True)
class ElfSection: name:str; header:libc.Elf64_Shdr|libc.Elf32_Shdr; content:bytes # noqa: E702

def link_sym(sym:str, libs:list[ctypes.CDLL]) -> int:
  for lib in libs:
    try: return unwrap(ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value)
    except (OSError, AttributeError, TypeError): pass
  raise RuntimeError(f'Attempting to relocate against an undefined symbol {sym}')

def _strtab(blob:bytes, idx:int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

def elf_sections(blob:bytes) -> tuple[str, list[ElfSection]]:
  assert blob[:4] == libc.ELFMAG.encode(), "blob is not an ELF, missing magic bytes"
  ecls = {libc.ELFCLASS32: "Elf32", libc.ELFCLASS64: "Elf64"}[blob[libc.EI_CLASS]]

  header = getattr(libc, f"{ecls}_Ehdr").from_buffer_copy(blob)
  section_headers = (getattr(libc, f"{ecls}_Shdr") * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]
  return ecls, sections

def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)

def elf_symbols(ecls:str, sections:list[ElfSection]) -> list[tuple[str, libc.Elf64_Sym|libc.Elf32_Sym]]:
  symsec = next((sh for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB), None)
  if symsec is None: return []
  strings = sections[symsec.header.sh_link].content
  return [(_strtab(strings, sym.st_name), sym) for sym in _to_carray(symsec, getattr(libc, f"{ecls}_Sym"))]

def elf_loader(blob:bytes, force_section_align:int=1, link_libs:list[ctypes.CDLL]|None=None,
               exports:dict[int, int]|None=None, symbols:dict[str, int]|None=None) -> tuple[memoryview, list[ElfSection], list[tuple]]:
  ecls, sections = elf_sections(blob)

  rel = [(sh, _to_carray(sh, getattr(libc, f"{ecls}_Rel"))) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, _to_carray(sh, getattr(libc, f"{ecls}_Rela"))) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = elf_symbols(ecls, sections)
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  # Prealloc image for all fixed addresses.
  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0: image[sh.header.sh_addr:sh.header.sh_addr+sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align:=max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  if exports is not None:
    for idx, (_, sym) in enumerate(symtab):
      if 0 < sym.st_shndx < len(sections) and sym.st_info >> 4 == libc.STB_GLOBAL and sym.st_info & 15 == libc.STT_FUNC:
        exports[idx] = sections[sym.st_shndx].header.sh_addr + sym.st_value

  # Relocations
  relocs = []
  for sh, c_rels in rel + rela:
    if sections[sh.header.sh_info].name == ".eh_frame": continue
    target_image_off = sections[sh.header.sh_info].header.sh_addr
    rels = [(r.r_offset, getattr(libc, f"{ecls.upper()}_R_SYM")(r.r_info), getattr(libc, f"{ecls.upper()}_R_TYPE")(r.r_info),
             getattr(r, "r_addend", 0)) for r in c_rels]
    for roff, idx, rtype, raddend in rels:
      name, sym = symtab[idx]
      value = (((symbols or {}).get(name) or link_sym(name, link_libs or [])) if sym.st_shndx == 0
               else sections[sym.st_shndx].header.sh_addr + sym.st_value)
      relocs.append((target_image_off + roff, value, rtype, raddend))

  return memoryview(image), sections, relocs

def jit_loader(obj: bytes, base:int=0, link_libs:list[ctypes.CDLL]|None=None,
               exports:dict[int, int]|None=None, symbols:dict[str, int]|None=None) -> bytes:
  image_, _, relocs = elf_loader(obj, link_libs=link_libs, exports=exports, symbols=symbols)
  if exports is not None: exports.update({idx:base+offset for idx,offset in exports.items()})
  image = bytearray(image_)

  def relocate(instr: int, base: int, ploc: int, tgt: int, r_type: int, addend: int):
    nonlocal image
    match r_type:
      # https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.95.pdf
      case libc.R_X86_64_PC32: return i2u(32, tgt-ploc)
      case libc.R_X86_64_PLT32:
        if -2**31 <= tgt-ploc-base < 2**31: return i2u(32, tgt-ploc-base)
        # ext lib address is out of rel32 range, create a trampoline: JMP [RIP+0] .quad target
        image += struct.pack("<HIQ", 0x25FF, 0, tgt-addend)
        return i2u(32, len(image)-14+addend-ploc)
      # https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst for definitions of relocations
      # https://www.scs.stanford.edu/~zyedidia/arm64/index.html for instruction encodings
      case libc.R_AARCH64_ADR_PREL_PG_HI21:
        rel_pg = (tgt & ~0xFFF) - (ploc & ~0xFFF)
        return instr | (getbits(rel_pg, 12, 13) << 29) | (getbits(rel_pg, 14, 32) << 5)
      case libc.R_AARCH64_ADD_ABS_LO12_NC: return instr | (getbits(tgt, 0, 11) << 10)
      case libc.R_AARCH64_LDST16_ABS_LO12_NC: return instr | (getbits(tgt, 1, 11) << 10)
      case libc.R_AARCH64_LDST32_ABS_LO12_NC: return instr | (getbits(tgt, 2, 11) << 10)
      case libc.R_AARCH64_LDST64_ABS_LO12_NC: return instr | (getbits(tgt, 3, 11) << 10)
      case libc.R_AARCH64_LDST128_ABS_LO12_NC: return instr | (getbits(tgt, 4, 11) << 10)
      case libc.R_AARCH64_CALL26 | libc.R_AARCH64_JUMP26:
        if -(2**25) <= tgt-ploc-base and tgt-ploc-base <= (2**25 - 1) * 4: return instr | getbits(tgt-ploc-base, 2, 27)
        # create trampoline:         LDR x17, 8  BR x17
        image += bytes(-len(image) % 4) + struct.pack("<IIQ", 0x58000051, 0xD61F0220, tgt)
        return instr | getbits(len(image)-ploc-16, 2, 27)
    raise NotImplementedError(f"Encountered unknown relocation type {r_type}")

  # This is needed because we have an object file, not a .so that has all internal references (like loads of constants from .rodata) resolved.
  for ploc,tgt,r_type,r_addend in relocs:
    image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], base, ploc, tgt+r_addend, r_type, r_addend))
  return bytes(image)
