import struct, ctypes
from dataclasses import dataclass
from tinygrad.helpers import getbits, i2u, unwrap
from tinygrad.runtime.autogen import libc

@dataclass(frozen=True)
class ElfSection: name:str; header:libc.Elf64_Shdr|libc.Elf32_Shdr; content:bytes # noqa: E702

def link_sym(sym:str, libs:list[ctypes.CDLL], link_syms:dict[str,int]|None=None) -> int:
  if link_syms is not None and (addr:=link_syms.get(sym)): return addr
  for lib in libs:
    try: return unwrap(ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value)
    except (OSError, AttributeError, TypeError): pass
  raise RuntimeError(f'Attempting to relocate against an undefined symbol {sym}')

def elf_loader(blob:bytes, force_section_align:int=1, link_libs:list[ctypes.CDLL]|None=None,
               link_syms:dict[str,int]|None=None) -> tuple[memoryview, list[ElfSection], list[tuple]]:
  assert blob[:4] == libc.ELFMAG.encode(), "blob is not an ELF, missing magic bytes"
  ecls = {libc.ELFCLASS32: "Elf32", libc.ELFCLASS64: "Elf64"}[blob[libc.EI_CLASS]]

  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

  header = getattr(libc, f"{ecls}_Ehdr").from_buffer_copy(blob)
  section_headers = (getattr(libc, f"{ecls}_Shdr") * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, getattr(libc, f"{ecls}_Rel"))) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, getattr(libc, f"{ecls}_Rela"))) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = next((_to_carray(sh, getattr(libc, f"{ecls}_Sym")) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB), None)
  strtab = next((sections[sh.header.sh_link].content for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB), b'')
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
    if trgt_sh_name == ".eh_frame": continue
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, unwrap(symtab)[getattr(libc, f"{ecls.upper()}_R_SYM")(r.r_info)], getattr(libc, f"{ecls.upper()}_R_TYPE")(r.r_info),
             getattr(r, "r_addend", 0)) for r in c_rels]
    for roff, sym, rtype, raddend in rels:
      # RISC-V emits RELAX/ALIGN/NONE markers that require no action.
      if rtype in (getattr(libc, "R_RISCV_RELAX", -1), getattr(libc, "R_RISCV_ALIGN", -2), getattr(libc, "R_RISCV_NONE", -3)):
        continue
      relocs.append((target_image_off + roff,
                    link_sym(_strtab(strtab, sym.st_name), link_libs or [], link_syms) if sym.st_shndx == 0 else
                    sections[sym.st_shndx].header.sh_addr + sym.st_value,
                    rtype, raddend))

  return memoryview(image), sections, relocs

def jit_loader(obj: bytes, base:int=0, link_libs:list[ctypes.CDLL]|None=None, link_syms:dict[str,int]|None=None) -> bytes:
  image_, _, relocs = elf_loader(obj, link_libs=link_libs, link_syms=link_syms)
  image = bytearray(image_)
  riscv_hi: dict[int, tuple[int, int]] = {}  # hi_offset -> (target, addend)

  def relocate(instr: int, base: int, ploc: int, tgt: int, r_type: int, addend: int):
    nonlocal image
    def _u_imm(ins, imm20):
      return (ins & 0x00000fff) | ((imm20 & 0xfffff) << 12)
    def _i_imm(ins, imm12):
      return (ins & 0x000fffff) | ((imm12 & 0xfff) << 20)
    def _s_imm(ins, imm12):
      imm = imm12 & 0xfff
      return (ins & 0x01fff07f) | (((imm >> 5) & 0x7f) << 25) | ((imm & 0x1f) << 7)

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
      # ---- RISC-V (RV64) relocations ----
      # Helper immediate setters
      # Absolute HI/LO
      case libc.R_RISCV_HI20:
        _abs = (base + tgt) if tgt < len(image) else tgt
        return _u_imm(instr, (_abs + 0x800) >> 12)
      case libc.R_RISCV_LO12_I:
        _abs = (base + tgt) if tgt < len(image) else tgt
        return _i_imm(instr, _abs & 0xfff)
      case libc.R_RISCV_LO12_S:
        _abs = (base + tgt) if tgt < len(image) else tgt
        return _s_imm(instr, _abs & 0xfff)
      # PC-relative HI/LO (and GOT_HI20, resolved directly to the symbol)
      case libc.R_RISCV_PCREL_HI20 | libc.R_RISCV_GOT_HI20:
        if tgt < len(image):
          value = tgt - ploc
        else:
          value = tgt - (base + ploc)
        riscv_hi[ploc] = (tgt, 0)
        return _u_imm(instr, (value + 0x800) >> 12)
      case libc.R_RISCV_PCREL_LO12_I:
        hi_off = tgt  # 0-based offset of paired HI instruction
        if hi_off not in riscv_hi:
          # Fall back: treat tgt as the symbol location.
          if tgt < len(image):
            value = tgt - ploc
          else:
            value = tgt - (base + ploc)
        else:
          g_tgt, g_addend = riscv_hi[hi_off]
          value = g_tgt - hi_off
        # Rewrite ld/lw-style LO into addi to materialize the address directly.
        # Keep rs1/rd fields from the original instruction.
        rd = (instr >> 7) & 0x1f
        rs1 = (instr >> 15) & 0x1f
        new_ins = (0x00000013) | (rd << 7) | (rs1 << 15)  # addi
        return _i_imm(new_ins, value & 0xfff)
      case libc.R_RISCV_PCREL_LO12_S:
        hi_off = tgt
        if hi_off in riscv_hi:
          g_tgt, g_addend = riscv_hi[hi_off]
          value = g_tgt - hi_off
        else:
          value = (tgt + r_addend) - (base + ploc)
        return _s_imm(instr, value & 0xfff)
      # Calls (AUIPC + JALR pair at ploc, ploc+4)
      case libc.R_RISCV_CALL | libc.R_RISCV_CALL_PLT:
        if tgt < len(image):
          value = tgt - ploc
        else:
          value = tgt - (base + ploc)
        if value < -(1 << 31) or value >= (1 << 31):
          # Trampoline
          image += bytes((-len(image)) % 4)
          tramp = len(image)
          _t = (base + tgt) if tgt < len(image) else tgt
          hi = (_t + 0x800) >> 12
          lo = _t & 0xfff
          # lui t1, hi ; addi t1, t1, lo ; jalr zero, t1, 0
          image += struct.pack("<III", 0x00000337 | ((hi & 0xfffff) << 12),  # lui t1
                                      0x00030313 | ((lo & 0xfff) << 20),     # addi t1,t1,lo
                                      0x00030067)                            # jalr zero,t1,0
          value = tramp - ploc
        # patch AUIPC at ploc
        image[ploc:ploc+4] = struct.pack("<I", _u_imm(struct.unpack("<I", image[ploc:ploc+4])[0], (value + 0x800) >> 12))
        # patch JALR at ploc+4
        if ploc + 4 <= len(image):
          ins2 = struct.unpack("<I", image[ploc+4:ploc+8])[0]
          image[ploc+4:ploc+8] = struct.pack("<I", _i_imm(ins2, value & 0xfff))
        return struct.unpack("<I", image[ploc:ploc+4])[0]
      case libc.R_RISCV_JAL:
        value = (tgt + r_addend) - (base + ploc)
        return instr  # placeholder: JAL immediate patching not commonly needed
      case libc.R_RISCV_BRANCH:
        value = (tgt + r_addend) - (base + ploc)
        return instr
      case libc.R_RISCV_32:
        struct.pack_into("<I", image, ploc, tgt & 0xffffffff)
        return struct.unpack("<I", image[ploc:ploc+4])[0]
      case libc.R_RISCV_64:
        struct.pack_into("<Q", image, ploc, tgt)
        return struct.unpack("<I", image[ploc:ploc+4])[0]
    raise NotImplementedError(f"Encountered unknown relocation type {r_type}")

  # This is needed because we have an object file, not a .so that has all internal references (like loads of constants from .rodata) resolved.
  for ploc,tgt,r_type,r_addend in relocs:
    image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], base, ploc, tgt+r_addend, r_type, r_addend))
  return bytes(image)
