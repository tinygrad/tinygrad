# SQTT for the emulator on gfx1100. sqtt_model times the instructions the emulator ran, this packs them
# SQTT_MODEL_TRACE=1 prints what each instruction waited on
from tinygrad.renderer.amd.dsl import Inst, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, SOPP
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp, DSOp
from tinygrad.renderer.amd.sqtt import PacketType, InstOp, AluSrc, MemSrc, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, VALUINST, \
                                       IMMEDIATE, IMMEDIATE_MASK, ALUEXEC, VMEMEXEC, TS_DELTA_OR_MARK
from tinygrad.helpers import getenv
from tinygrad.viz.serve import amd_decode
from test.mockgpu.amd.sqtt_model import Op, CU, Unit, EXEC, NEVER, report

DS_LOADS = {op for op in DSOp if op.name.startswith("DS_LOAD")} | {DSOp.DS_SWIZZLE_B32}
DS_TWO_ADDR = {op for op in DSOp if "2ADDR" in op.name}

def new(cls:type[PacketType], **fields) -> PacketType:
  raw = cls.encoding.set(0)
  for k, v in fields.items(): raw = getattr(cls, k).set(raw, v)
  return cls.from_raw(raw)

def encode(packets:list[PacketType]) -> bytes:
  nibbles = [(p._raw >> (4*i)) & 0xF for p in packets for i in range(p._size_nibbles)]  # type: ignore[attr-defined]
  # 14 not 16, decode needs 16 nibbles buffered and 16 zeros would come out as NOPs
  nibbles += [0] * (14 + len(nibbles) % 2)
  return bytes(lo | hi << 4 for lo, hi in zip(nibbles[::2], nibbles[1::2]))

def wave_ops(insts:dict[int, Inst], trace) -> list[Op]:
  if trace[0] >= len(trace): raise RuntimeError("the trace does not fit in the SQTT buffer, raise SQTT_BUFFER_SIZE")
  ops: list[Op] = []
  i = 1
  while i <= trace[0]:
    inst = insts[trace[i]]
    addrs = list(trace[i+2:i+2+32]) if isinstance(inst, DS) else []
    if not (isinstance(inst, SOPP) and inst.op is SOPPOp.S_ENDPGM): ops.append(Op(trace[i], inst, trace[i+1], addrs))
    i += 2 + len(addrs)
  return ops

def words(inst) -> int: return next((inst.op_regs[f] for f in ("vdst", "data0", "data") if f in inst.op_regs), 1)

def packet(x:Op, taken:bool) -> tuple[type[PacketType], dict]:
  inst, u = x.inst, x.unit
  if u is Unit.BRANCH: op = InstOp.JUMP if taken else InstOp.JUMP_NO
  elif u is Unit.BARRIER: op = InstOp.BARRIER
  elif u is Unit.SMEM: op = InstOp.SMEM_RD
  elif u is Unit.OTHER and inst.op is SOPPOp.S_SENDMSG: op = InstOp.MESSAGE
  elif u is Unit.VMEM:
    # SGMEM_WR_n and LDS_WR_n count the address and the data dwords
    n = 2 if inst.saddr.offset == NULL.offset else 1  # no SGPR base
    op = InstOp[f"SGMEM_RD_{n}" if x.writes else f"SGMEM_WR_{n + words(inst)}"]
  elif x.stamp == NEVER: return IMMEDIATE, {}
  elif u is Unit.LDS: op = InstOp.LDS_RD if inst.op in DS_LOADS else InstOp[f"LDS_WR_{1 + words(inst) * (2 if inst.op in DS_TWO_ADDR else 1)}"]
  elif EXEC in x.writes: op = InstOp.SALU_WR_EXEC if u is Unit.SALU else InstOp.VALU1_WR_EXEC
  elif u is Unit.SALU: op = InstOp.SALU
  elif u is Unit.TRANS: op = InstOp.VALUT_4
  elif x.f64: op = InstOp.VALUB_16
  else: return VALUINST, {}
  return INST, {"op":op}

def sqtt_blob(code:bytes, traces:list[memoryview], simd:int) -> bytes:
  # waves alternate between the CU's two SIMDs, hw SIMD 0 and 2
  insts, simds = amd_decode(code, "rdna3"), [k % 2 for k in range(len(traces))]
  cu = CU([wave_ops(insts, trace) for trace in traces], simds)
  cu.run()
  if getenv("SQTT_MODEL_TRACE"): report(cu)
  events: list[tuple[int, type[PacketType], dict]] = []
  masks: dict[int, list[int]] = {}
  alu: dict[int, int] = {}
  for k, w in enumerate(cu.waves):
    hw, slot = 2 * simds[k], k // 2
    # hw starts the waves a cycle apart
    events += [(k, WAVESTART, {"simd":hw, "wave":slot}), (w.end(), WAVEEND, {"simd":hw, "wave":slot})]
    if hw != simd: continue
    for j, x in enumerate(w.ops):
      cls, fields = packet(x, j+1 < len(w.ops) and w.ops[j+1].pc != x.pc + x.inst.size())
      if cls is IMMEDIATE: masks.setdefault(x.t, []).append(slot)
      else: events.append((x.t, cls, {"wave":slot, **fields}))
      if x.stamp == NEVER: continue
      if x.unit is Unit.LDS: events.append((x.stamp, VMEMEXEC, {"src":MemSrc.LDS}))
      else: alu[x.stamp] = alu.get(x.stamp, 0) | (AluSrc.SALU if x.unit is Unit.SALU else AluSrc.VALU).value
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
