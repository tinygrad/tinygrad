import struct, sys, tinygrad.runtime.autogen.libc as libc
from dataclasses import dataclass
from tinygrad.helpers import getbits, i2u

@dataclass(frozen=True)
class ElfSection: name:str; header:libc.Elf64_Shdr; content:bytes # noqa: E702

def elf_loader(blob:bytes, force_section_align:int=1) -> tuple[memoryview, list[ElfSection], list[tuple]]:
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  section_headers = (libc.Elf64_Shdr * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, libc.Elf64_Rel)) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, libc.Elf64_Rela)) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = [_to_carray(sh, libc.Elf64_Sym) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB][0]
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  # Prealloc image for all fixed addresses.
  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0: image[sh.header.sh_addr:sh.header.sh_addr+sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align:=max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  # Relocations
  relocs = []
  for sh, trgt_sh_name, c_rels in rel + rela:
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, symtab[libc.ELF64_R_SYM(r.r_info)], libc.ELF64_R_TYPE(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    relocs += [(target_image_off + roff, sections[sym.st_shndx].header.sh_addr + sym.st_value, rtype, raddend) for roff, sym, rtype, raddend in rels]

  return memoryview(image), sections, relocs

def relocate(instr: int, ploc: int, tgt: int, r_type: int):
  match r_type:
    # https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.95.pdf
    case libc.R_X86_64_PC32: return i2u(32, tgt-ploc)
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
  raise NotImplementedError(f"Encountered unknown relocation type {r_type}")

def coff_loader(data):
  if len(data) < 20: raise ValueError("Invalid COFF file: Header too small.")
  file_header = struct.unpack('<HHIIIHH', data[:20])
  num_sections = file_header[1]
  size_optional_header = file_header[5]
  sections_offset = 20 + size_optional_header
  symtab_ptr = file_header[3]
  num_symbols = file_header[4]
  strtab_start = None
  strtab_size = None
  if symtab_ptr != 0 and num_symbols != 0:
    symtab_size = num_symbols * 18
    strtab_start = symtab_ptr + symtab_size
    if strtab_start + 4 <= len(data):
      strtab_size = struct.unpack('<I', data[strtab_start:strtab_start+4])[0]
  for i in range(num_sections):
    section_start = sections_offset + i * 40
    if section_start + 40 > len(data):
      break
    section_header = data[section_start:section_start+40]
    fields = struct.unpack('<8sIIIIIIHHI', section_header)
    name_bytes, _vsize, _vaddr, size_raw, ptr_raw = fields[:5]
    name_str = name_bytes.decode('ascii', errors='ignore').split('\x00')[0]
    if name_str.startswith('/') and strtab_start and strtab_size:
      try:
        offset = int(name_str[1:])
        str_offset = strtab_start + 4 + offset
        if str_offset >= len(data):
          continue
        end = str_offset
        while end < len(data) and data[end] != 0:
          end += 1
        section_name = data[str_offset:end].decode('ascii', errors='ignore')
      except ValueError:
        continue
    else:
      section_name = name_str.strip('\x00')
    if section_name == '.text':
      if ptr_raw + size_raw > len(data):
        raise ValueError(".text section data exceeds file bounds.")
      return data[ptr_raw:ptr_raw + size_raw]
  raise ValueError(".text section not found in COFF file.")

def jit_loader(obj: bytes) -> bytes:
  if sys.platform == "win32":
    return coff_loader(obj)
  else:
    image, _, relocs = elf_loader(obj)
    # This is needed because we have an object file, not a .so that has all internal references (like loads of constants from .rodata) resolved.
    for ploc,tgt,r_type,r_addend in relocs:
      image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], ploc, tgt+r_addend, r_type))
    return bytes(image)
