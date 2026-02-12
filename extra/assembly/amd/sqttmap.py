# maps SQTT trace packets to instructions.
from dataclasses import dataclass
from typing import Iterator

from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND, InstOp, PacketType, IMMEDIATE_MASK
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP, s_endpgm
from extra.assembly.amd.autogen.rdna3.enum import SOPPOp

@dataclass(frozen=True)
class InstructionInfo:
  pc: int
  wave: int
  inst: Inst

def map_insts(data:bytes, lib:bytes, target:int) -> Iterator[tuple[PacketType, InstructionInfo|None]]:
  """maps SQTT packets to instructions, yields (packet, instruction_info or None)"""
  # map pcs to insts
  from tinygrad.viz.serve import amd_decode
  pc_map = amd_decode(lib, target)

  wave_pc:dict[int, int] = {}
  # only processing packets on one [CU, SIMD] unit
  def simd_select(p) -> bool: return getattr(p, "cu", 0) == 0 and getattr(p, "simd", 0) == 0
  for p in decode(data):
    if not simd_select(p): continue
    if isinstance(p, WAVESTART):
      assert p.wave not in wave_pc, "only one inflight wave per unit"
      wave_pc[p.wave] = next(iter(pc_map))
      continue
    if isinstance(p, WAVEEND):
      pc = wave_pc.pop(p.wave)
      yield (p, InstructionInfo(pc, p.wave, s_endpgm()))
      continue
    # skip OTHER_ instructions, they don't belong to this unit
    if isinstance(p, INST) and p.op.name.startswith("OTHER_"): continue
    if isinstance(p, IMMEDIATE_MASK):
      # immediate mask may yield multiple times per packet
      for wave in range(16):
        if p.mask & (1 << wave):
          inst = pc_map[pc:=wave_pc[wave]]
          # can this assert be more strict?
          assert isinstance(inst, SOPP), f"IMMEDIATE_MASK packet must map to SOPP, got {inst}"
          wave_pc[wave] += inst.size()
          yield (p, InstructionInfo(pc, wave, inst))
      continue
    if isinstance(p, (VALUINST, INST, IMMEDIATE)):
      inst = pc_map[pc:=wave_pc[p.wave]]
      # s_delay_alu doesn't get a packet?
      if isinstance(inst, SOPP) and inst.op in {SOPPOp.S_DELAY_ALU}:
        wave_pc[p.wave] += inst.size()
        inst = pc_map[pc:=wave_pc[p.wave]]
      # identify a branch instruction, only used for asserts
      is_branch = isinstance(inst, SOPP) and "BRANCH" in inst.op_name
      if is_branch: assert isinstance(p, INST) and p.op in {InstOp.JUMP_NO, InstOp.JUMP}, f"branch can only be folowed by jump packets, got {p}"
      # JUMP handling
      if isinstance(p, INST) and p.op is InstOp.JUMP:
        assert is_branch, f"JUMP packet must map to a branch instruction, got {inst}"
        x = inst.simm16 & 0xffff
        wave_pc[p.wave] += inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        if is_branch: assert inst.op != SOPPOp.S_BRANCH, f"S_BRANCH must have a JUMP packet, got {p}"
        wave_pc[p.wave] += inst.size()
      yield (p, InstructionInfo(pc, p.wave, inst))
      continue
    # for all other packets (VMEMEXEC, ALUEXEC, etc.), yield with None
    yield (p, None)

