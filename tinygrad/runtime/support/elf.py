from __future__ import annotations
import os, ctypes, contextlib, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess, time, array
from typing import Tuple, List, Any, cast, Union, Dict
import tinygrad.runtime.autogen.libc as libc

def _elf_parse_names(cstr_tabs):
  lookup_dict, prev_len = {}, 0
  for word in cstr_tabs.split(b'\0'):
    lookup_dict[prev_len] = word.decode('utf-8')
    prev_len += len(word) + 1
  return lookup_dict

def elf_loader(blob:bytes, force_section_align:int=1):
  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  section_headers = (libc.Elf64_Shdr * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  section_names = _elf_parse_names(blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size])
  sections = {section_names[sh.sh_name]: (i, sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for i, sh in enumerate(section_headers)}
  strtab = _elf_parse_names(sections['.strtab'][2])
  symtab = (libc.Elf64_Sym * (sections['.symtab'][1].sh_size // sections['.symtab'][1].sh_entsize)).from_buffer_copy(sections['.symtab'][2])

  # Prealloc image for all fixed addresses.
  image = bytearray(max([shdr.sh_addr+shdr.sh_size for _,shdr,_ in sections.values() if shdr.sh_type == libc.SHT_PROGBITS and shdr.sh_addr != 0]+[0]))
  for section_name, (section_idx, section_header, section_content) in sections.items():
    if section_header.sh_type == libc.SHT_PROGBITS:
      if section_header.sh_addr != 0: image[section_header.sh_addr:section_header.sh_addr+section_header.sh_size] = section_content
      else:
        align = max(section_header.sh_addralign, force_section_align)
        if (extrapad:=len(image) % align) != 0: image += bytearray(align - extrapad)
        section_header.sh_addr = len(image)
        image += section_content

  # Relocations
  relocs = []
  for section_name, (section_idx, section_header, section_content) in sections.items():
    if section_header.sh_type == libc.SHT_REL or section_header.sh_type == libc.SHT_RELA:
      typ_lookup = {libc.SHT_REL: libc.Elf64_Rel, libc.SHT_RELA: libc.Elf64_Rela}
      rels = (typ_lookup[section_header.sh_type] * (section_header.sh_size // section_header.sh_entsize)).from_buffer_copy(section_content)
      for r in rels:
        rel_sym = symtab[libc.ELF64_R_SYM(r.r_info)]
        target_sh_off = sections[section_name[4 if section_header.sh_type == libc.SHT_REL else 5:]][1].sh_addr + r.r_offset
        rel_sh_off = sections[section_names[section_headers[rel_sym.st_shndx].sh_name]][1].sh_addr + rel_sym.st_value
        relocs.append((target_sh_off, rel_sh_off, libc.ELF64_R_TYPE(r.r_info), getattr(r, "r_addend", 0)))

  return memoryview(image), sections, relocs
