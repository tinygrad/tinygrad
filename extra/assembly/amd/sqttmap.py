# maps trace packets to instructions.
from extra.assembly.amd.sqtt import decode, print_packets
from extra.assembly.amd.decode import decode_inst
from tinygrad.runtime.support.elf import elf_loader

def disasm_llvm(lib:bytes):
  from tinygrad.viz.serve import llvm_disasm
  return llvm_disasm(target, lib)

def map_insts(data:bytes, lib:bytes):
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found"
  text_off, text_size = text.header.sh_addr, text.header.sh_size

  pc_table:dict[int, str] = {}

  # disassemble all instructions in .text section
  offset = text_off
  while offset < text_off + text_size:
    inst = decode_inst(image[offset:])
    pc_table[offset] = inst
    offset += inst.size()

  # compare llvm
  llvm_pc_table = disasm_llvm(lib)
  assert len(pc_table) == len(llvm_pc_table)
  assert list(pc_table) == list(llvm_pc_table)

if __name__ == "__main__":
  import sys, pickle
  if len(sys.argv) < 2:
    print("Usage: python sqttmap.py <pkl_file>")
    sys.exit(1)
  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.name:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  target = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device.startswith("AMD"))).props["gfx_target_version"]
  e = sqtt_events[1]
  prg = kern_events[e.kern]
  map_insts(e.blob, prg.lib)
