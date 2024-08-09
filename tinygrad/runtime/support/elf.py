from __future__ import annotations
from typing import Tuple, List, Any, Union
from dataclasses import dataclass
import tinygrad.runtime.autogen.libc as libc

@dataclass(frozen=True)
class Elf64Section: name:str; header:libc.Elf64_Shdr; content:bytes # noqa: E702

@dataclass(frozen=True)
class Elf32Section: name:str; header:libc.Elf32_Shdr; content:bytes # noqa: E702

def elf_loader(blob:bytes, force_section_align:int=1, elf32:bool=False) -> Tuple[memoryview, Union[List[Elf32Section], List[Elf64Section]], Any]:
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

  ElfSection = Elf32Section if elf32 else Elf64Section
  Elf_Ehdr, Elf_Shdr = (libc.Elf32_Ehdr, libc.Elf32_Shdr) if elf32 else (libc.Elf64_Ehdr, libc.Elf64_Shdr)
  Elf_Rel, Elf_Rela, Elf_Sym = (libc.Elf32_Rel, libc.Elf32_Rela, libc.Elf32_Sym) if elf32 else (libc.Elf64_Rel, libc.Elf64_Rela, libc.Elf64_Sym)
  ELF_R_SYM, ELF_R_TYPE = (libc.ELF32_R_SYM, libc.ELF32_R_TYPE) if elf32 else (libc.ELF64_R_SYM, libc.ELF64_R_TYPE)

  header = Elf_Ehdr.from_buffer_copy(blob)
  section_headers = (Elf_Shdr * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, Elf_Rel)) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, Elf_Rela)) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = [_to_carray(sh, Elf_Sym) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB][0]
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
    rels = [(r.r_offset, symtab[ELF_R_SYM(r.r_info)], ELF_R_TYPE(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    relocs += [(target_image_off + roff, sections[sym.st_shndx].header.sh_addr + sym.st_value, rtype, raddend) for roff, sym, rtype, raddend in rels]

  return memoryview(image), sections, relocs
