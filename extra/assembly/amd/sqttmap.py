# maps trace packets to instructions.
from typing import Iterator
from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND, InstOp, PacketType
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP, s_endpgm
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
  def wave_focus(p) -> bool: return getattr(p, "cu", 0) == 0 and getattr(p, "simd", 0) == 0 and getattr(p, "wave", None) == 0
  for p in decode(data):
    if not wave_focus(p): continue
    print_packets([p])
    if isinstance(p, WAVESTART):
      assert p.wave not in wave_pc
      wave_pc[p.wave] = 0
      continue
    if isinstance(p, WAVEEND): break
    pc = wave_pc[p.wave]
    inst = pc_map[pc]
    wave_pc[p.wave] += inst.size()
    yield (p, pc, inst)

def test_rocprof_inst_traces_match(sqtt, prg, target):
  from tinygrad.viz.serve import llvm_disasm
  from extra.sqtt.roc import decode as roc_decode, InstExec
  disasm = {addr+prg.base:inst_disasm for addr, inst_disasm in llvm_disasm(target, prg.lib).items()}
  rctx = roc_decode([sqtt], {prg.name:disasm})
  rwaves = rctx.inst_execs[(sqtt.kern, sqtt.exec_tag)]
  rwaves_iter:dict[int, list[Iterator[InstExec]]] = {}
  for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(w.unpack_insts())
  rwaves_base = next(iter(disasm))
  wave_n:dict[int, int] = {}

  total_waves = sum(len(v) for v in rwaves_iter.values())
  insts = 0

  for pkt, pc, inst in map_insts(sqtt.blob, prg.lib):
    rocprof_inst = next(rwaves_iter[pkt.wave][0])
    ref_pc = rocprof_inst.pc-rwaves_base
    if inst.disasm() == "s_endpgm":
      wave_n[pkt.wave] = wave_n.get(pkt.wave, 0)+1
      last = list(rwaves_iter[pkt.wave].pop(0))
      try:
        assert len(last) == 1 and disasm[last[0].pc][0] == "s_endpgm"
      except AssertionError as e:
        print("-- not in pkt")
        for l in last: print(l)
        raise e
    else:
      assert pkt._time == rocprof_inst.time+rocprof_inst.stall
      assert ref_pc == pc, f"pc mismatch {ref_pc}:{disasm[rocprof_inst.pc][0]} != {pc}:{inst.disasm()} {pkt._time} {rocprof_inst.time+rocprof_inst.stall}"
      insts += 1

  print(f"passed for {insts} instructions across {total_waves} waves scheduled on {len(rwaves_iter)} wave units")

if __name__ == "__main__":
  import sys, pickle
  from tinygrad.helpers import temp
  fp = temp("profile.pkl", append_user=True) if len(sys.argv) < 2 else sys.argv[1]
  with open(fp, "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.name:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  target = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device.startswith("AMD"))).props["gfx_target_version"]
  e =sqtt_events[1]
  kern_events[e.kern]
  test_rocprof_inst_traces_match(e, kern_events[e.kern], target)
