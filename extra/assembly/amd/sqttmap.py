# maps trace packets to instructions.
from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND, InstOp
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP
from extra.assembly.amd.autogen.rdna3.enum import SOPPOp
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
  rwaves = {w.wave_id:w for w in rwaves}

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
  simd_sel = (0, 0)
  for p in decode(data):
    if isinstance(p, WAVESTART) and (p.cu, p.simd) == simd_sel:
      assert p.wave not in wave_pc
      wave_pc[p.wave] = 0
      rwaves_iter[p.wave] = rwaves[p.wave].unpack_insts()
      rwaves_base[p.wave] = next(rwaves[p.wave].unpack_insts()).pc
    if isinstance(p, WAVEEND) and (p.cu, p.simd) == simd_sel:
      wave_pc.pop(p.wave)
    if isinstance(p, INST) and ("OTHER_" not in p.op.name):
      inst = pc_map[pc:=wave_pc[p.wave]]
      # s_delay_alu doesn't get a packet
      if isinstance(inst, SOPP) and inst.op is SOPPOp.S_DELAY_ALU: continue
      if isinstance(inst, SOPP) and (inst.op is SOPPOp.S_BRANCH or (inst.op.name.startswith("S_CBRANCH") and p.op is InstOp.JUMP)):
        x = inst.simm16 & 0xffff
        wave_pc[p.wave] += inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        wave_pc[p.wave] += inst.size()

      print(f"{pc:012X} {inst.disasm()} wave={p.wave}")
      if (ref:=next(rwaves_iter[p.wave], None)) is None and inst.disasm() in {"s_endpgm", "s_code_end"}: break
      rpc = ref.pc-rwaves_base[p.wave]
      assert rpc == pc, f"{rpc}:{rpc_table[ref.pc]} != {pc}:{inst.disasm()}"

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
  for e in sqtt_events:
    if not e.itrace: continue
    print("------", e.kern)
    prg = kern_events[e.kern]
    map_insts(e.blob, prg.lib)
