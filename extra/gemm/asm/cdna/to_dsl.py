import pathlib
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.assembly.amd.decode import decode_inst

src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", (pathlib.Path(__file__).parent/"gemm.s").read_text())
lib = HIPCompiler("gfx950").compile(src)
image, sections, _ = elf_loader(lib)
text = next((sh for sh in sections if sh.name == ".text"), None)
text_off, text_size = text.header.sh_addr, text.header.sh_size
offset = text_off
while offset < text_off + text_size:
  inst = decode_inst(image[offset:], "cdna")
  print(inst.disasm())
  offset += inst.size()
