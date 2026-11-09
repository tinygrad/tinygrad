# SQTT for the emulator on gfx1100: sqtt_model times the emulator's instruction streams and this packs the cycles into packets
from tinygrad.renderer.amd.dsl import Inst, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, SOPP
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp
from tinygrad.renderer.amd.sqtt import PacketType, InstOp, AluSrc, MemSrc, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, VALUINST, \
                                       IMMEDIATE, IMMEDIATE_MASK, ALUEXEC, VMEMEXEC, TS_DELTA_OR_MARK
from test.mockgpu.amd.sqtt_model import Step, Op, emulate, ds_words, EXEC

def new(cls:type[PacketType], **fields) -> PacketType:
  raw = cls.encoding.set(0)
  for k, v in fields.items(): raw = cls._fields[k].set(raw, v)  # type: ignore[attr-defined]
  return cls.from_raw(raw)

def encode(packets:list[PacketType]) -> bytes:
  nibbles = [(p._raw >> (4*i)) & 0xF for p in packets for i in range(p._size_nibbles)]  # type: ignore[attr-defined]
  # decode emits a packet once 16 nibbles from its start are buffered, so 14 zeros finish a 2-nibble packet and 16 would decode as NOPs
  nibbles += [0] * (14 + len(nibbles) % 2)
  return bytes(lo | hi << 4 for lo, hi in zip(nibbles[::2], nibbles[1::2]))

def wave_steps(insts:dict[int, Inst], trace:list[int]) -> list[Step]:
  steps, i, delays = list[Step](), 1, dict[int, list[int]]()
  while i <= trace[0]:
    inst = insts[trace[i]]
    lanes = trace[i+2:i+34] if isinstance(inst, DS) else None
    # s_delay_alu has no packet: instid0 applies to the next instruction, instid1 to the one instskip after it
    if isinstance(inst, SOPP) and inst.op is SOPPOp.S_DELAY_ALU:
      if inst.simm16 & 0xf: delays.setdefault(len(steps), []).append(inst.simm16 & 0xf)
      if inst.simm16 >> 7 & 0xf: delays.setdefault(len(steps) + (inst.simm16 >> 4 & 7), []).append(inst.simm16 >> 7 & 0xf)
    elif not (isinstance(inst, SOPP) and inst.op is SOPPOp.S_ENDPGM):
      steps.append(Step(trace[i], inst, trace[i+1], lanes, delays.pop(len(steps), [])))
    i += 2 + (32 if lanes is not None else 0)
  return steps

def packet(s:Step, kind:str, taken:bool) -> tuple[type[PacketType], dict]:
  o = Op(s.inst)
  if isinstance(s.inst, SOPP) and s.inst.op is SOPPOp.S_SENDMSG: return INST, {"op":InstOp.MESSAGE}
  if kind in ("IMM", "WAIT", "NOP", "SLEEP", "CLAUSE"): return IMMEDIATE, {}
  if kind in ("SALU", "VALU") and EXEC in o.dsts: op = InstOp.SALU_WR_EXEC if kind == "SALU" else InstOp.VALU1_WR_EXEC
  elif kind in ("VALU", "TRANS") and (o.f64 or o.passes > 1): op = InstOp.VALUB_16 if o.f64 else InstOp.VALUB_4 if o.passes == 4 else InstOp.VALUB_2
  elif kind in ("VALU", "WMMA"): return VALUINST, {}
  elif kind == "LDS": op = InstOp.LDS_RD if o.load else InstOp(InstOp.LDS_WR_1.value + o.words)
  elif kind == "VMEM":
    # no SGPR base is one op later, a store one more per dword
    null = getattr(getattr(s.inst, "saddr", None), "offset", NULL.offset) == NULL.offset
    op = InstOp(InstOp.SGMEM_WR_2.value + ds_words(s.inst) - 1 + null) if o.store else InstOp.SGMEM_RD_2 if null else InstOp.SGMEM_RD_1
  elif kind == "BRANCH": op = InstOp.JUMP if taken else InstOp.JUMP_NO
  else: op = {"TRANS":InstOp.VALUT_4, "BARRIER":InstOp.BARRIER, "SMEM":InstOp.SMEM_RD}.get(kind, InstOp.SALU)
  return INST, {"op":op}

def sqtt_blob(insts:dict[int, Inst], traces:list[list[int]], simd:int) -> bytes:
  steps = [wave_steps(insts, t) for t in traces]
  # the model numbers a CU's SIMDs 0 and 1: hw SIMDs 0 and 2
  timed = emulate(steps, simds=[k % 2 for k in range(len(steps))])
  events: list[tuple[int, type[PacketType], dict]] = []
  masks: dict[int, list[int]] = {}
  alu: dict[int, int] = {}
  for k, (stamps, end, kinds, execs, _) in enumerate(timed):
    hw, slot = 2 * (k % 2), k // 2
    events += [(k, WAVESTART, {"simd":hw, "wave":slot}), (end, WAVEEND, {"simd":hw, "wave":slot})]
    if hw != simd: continue
    for j, (s, t, kind, x) in enumerate(zip(steps[k], stamps, kinds, execs)):
      cls, fields = packet(s, kind, j+1 < len(steps[k]) and steps[k][j+1].pc != s.pc + s.inst.size())
      if cls is IMMEDIATE: masks.setdefault(t, []).append(slot)
      else: events.append((t, cls, {"wave":slot, **fields}))
      if x is not None and kind == "LDS": events.append((x, VMEMEXEC, {"src":MemSrc.LDS}))
      # a VALU and an SALU op that execute in one cycle share an ALUEXEC
      elif x is not None: alu[x] = alu.get(x, 0) | (AluSrc.SALU.value if kind == "SALU" else AluSrc.VALU.value)
  events += [(t, ALUEXEC, {"src":AluSrc(src)}) for t, src in alu.items()]
  for t, m in masks.items():
    events.append((t, IMMEDIATE, {"wave":m[0]}) if len(m) == 1 else (t, IMMEDIATE_MASK, {"mask":sum(1 << x for x in m)}))
  packets, now = [new(LAYOUT_HEADER, layout=3, simd=simd, sel_a=6)], 0
  for t, cls, fields in sorted(events, key=lambda e: (e[0], e[1] is WAVEEND)):
    if (gap:=t - now) > cls.delta.mask:  # type: ignore[attr-defined]
      packets.append(new(TS_DELTA_OR_MARK, delta=gap))
      gap = 0
    packets.append(new(cls, delta=gap, **fields))
    now = t
  return encode(packets)
