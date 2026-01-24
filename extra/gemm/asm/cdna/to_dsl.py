import pathlib
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.cdna.ins import *

src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", (pathlib.Path(__file__).parent/"gemm.s").read_text())
lib = HIPCompiler("gfx950").compile(src)
image, sections, _ = elf_loader(lib)
text = next((sh for sh in sections if sh.name == ".text"), None)
text_off, text_size = text.header.sh_addr, text.header.sh_size
offset = text_off
py_dsl:list[str] = []
while offset < text_off + text_size:
  inst = decode_inst(image[offset:], "cdna")
  py_dsl.append(repr(inst)+",")
  offset += inst.size()

pre = "from extra.assembly.amd.autogen.cdna.ins import *\n\n"
with open(pathlib.Path(__file__).parent/"gemm.py", "w") as f: f.write(pre+"insts=["+"\n".join(py_dsl)+"]")

py_txt = (pathlib.Path(__file__).parent/"gemm.py").read_text()
exec(py_txt)

# roundtrip test: verify re-encoding matches original bytes
print(f"Testing roundtrip for {len(insts)} instructions...")
offset = text_off
passed, failed = 0, 0
failures:list[str] = []
for i, inst in enumerate(insts):
  orig_bytes = image[offset:offset+inst.size()]
  reencoded = inst.to_bytes()
  if reencoded == orig_bytes:
    passed += 1
  else:
    failed += 1
    failures.append(f"  [{i}] {inst.disasm()}: orig={orig_bytes.hex()} reenc={reencoded.hex()}")
  offset += inst.size()

print(f"Roundtrip: {passed} passed, {failed} failed")
if failures:
  print("Failures:")
  for f in failures[:20]: print(f)
  if len(failures) > 20: print(f"  ... and {len(failures) - 20} more")
  raise AssertionError(f"{failed} instructions failed roundtrip")
