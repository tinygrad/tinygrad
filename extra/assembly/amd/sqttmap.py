# maps trace packets to instructions.
from extra.assembly.amd.sqtt import decode, print_packets
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.decode import decode_inst
from tinygrad.runtime.support.elf import elf_loader

def get_rocprof_output(sqtt, prg, target):
  from tinygrad.viz.serve import llvm_disasm
  from extra.sqtt.roc import decode as roc_decode
  disasm = {addr+prg.base:inst_disasm for addr, inst_disasm in llvm_disasm(target, prg.lib).items()}
  rctx = roc_decode([sqtt], {prg.name:disasm})
  return rctx.inst_execs[(sqtt.kern, sqtt.exec_tag)], {k:v[0] for k,v in disasm.items()}

def map_insts(data:bytes, lib:bytes):
  # rocprof to compare
  #rwaves, rpc_table = get_rocprof_output(e, prg, target)
  #for inst in rwaves[0].unpack_insts(): print(rpc_table[inst.pc])

  # get pc table
  pc_table:dict[int, Inst] = {}
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found"
  text_off, text_size = text.header.sh_addr, text.header.sh_size
  offset = text_off
  while offset < text_off + text_size:
    inst = decode_inst(image[offset:])
    pc_table[offset] = inst
    offset += inst.size()

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
