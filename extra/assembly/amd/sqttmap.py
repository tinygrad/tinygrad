# maps trace packets to instructions.
from typing import Iterator
from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND, InstOp, PacketType
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP
from extra.assembly.amd.autogen.rdna3.enum import SOPPOp
from tinygrad.runtime.support.elf import elf_loader

def map_insts(data:bytes, lib:bytes) -> Iterator[tuple[PacketType, int, Inst]]:
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

  wave_pc:dict[int, int] = {}
  simd_sel = (0, 0)
  for p in decode(data):
    if isinstance(p, WAVESTART) and (p.cu, p.simd) == simd_sel:
      assert p.wave not in wave_pc
      wave_pc[p.wave] = 0
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

      yield (p, pc, inst)

def test_rocprof_inst_traces_match(sqtt, prg, target):
  from tinygrad.viz.serve import llvm_disasm
  from extra.sqtt.roc import decode as roc_decode
  disasm = {addr+prg.base:inst_disasm for addr, inst_disasm in llvm_disasm(target, prg.lib).items()}
  rctx = roc_decode([sqtt], {prg.name:disasm})
  rwaves = rctx.inst_execs[(sqtt.kern, sqtt.exec_tag)]
  rwaves_iter = {w.wave_id:w.unpack_insts() for w in rwaves}
  rwaves_base = {w.wave_id:next(w.unpack_insts()).pc for w in rwaves}

  for pkt, pc, inst in map_insts(sqtt.blob, prg.lib):
    rocprof_pc = next(rwaves_iter[pkt.wave]).pc
    ref_pc = rocprof_pc-rwaves_base[pkt.wave]
    assert ref_pc == pc, f"pc mismatch {ref_pc}:{disasm[rocprof_pc][0]} != {pc}:{inst.disasm()}"

if __name__ == "__main__":
  import sys, pickle
  from tinygrad.helpers import temp
  fp = temp("profile.pkl", append_user=True) if len(sys.argv) < 2 else sys.argv[1]
  with open(fp, "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.name:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  target = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device.startswith("AMD"))).props["gfx_target_version"]
  for e in sqtt_events:
    if not e.itrace or e.se != 1: continue
    print("------", e.kern)
    test_rocprof_inst_traces_match(e, kern_events[e.kern], target)
