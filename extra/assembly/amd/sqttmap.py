# maps trace packets to instructions.
from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND
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
  rwaves, rpc_table = get_rocprof_output(e, prg, target)

  # map pcs to insts
  pc_map:dict[int, Inst] = {}
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found"
  text_off, text_size = text.header.sh_addr, text.header.sh_size
  offset = text_off
  while offset < text_off + text_size:
    inst = decode_inst(image[offset:])
    pc_map[offset-text_off] = inst
    offset += inst.size()

  rwaves_iter, rwaves_base = {}, {}
  wave_pc:dict[int, int] = {}
  for p in decode(data):
    if isinstance(p, WAVESTART):
      wave_pc[p.wave] = 0
      rwaves_iter[p.wave] = rwaves[p.wave].unpack_insts()
      rwaves_base[p.wave] = next(rwaves[p.wave].unpack_insts()).pc
    if isinstance(p, (INST, VALUINST, IMMEDIATE)) and "OTHER_" not in type(p).__name__:
      inst = pc_map[wave_pc[p.wave]]
      if (ref:=next(rwaves_iter[p.wave], None)) is None and inst.disasm() in {"s_endpgm", "s_code_end"}: break
      assert ref.pc-rwaves_base[p.wave] == wave_pc[p.wave]
      #print(inst.disasm(), rpc_table[ref.pc])
      wave_pc[p.wave] += inst.size()

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
