# SQTT packet builder and nibble-stream encoder for the RDNA3 mockgpu emulator.
from __future__ import annotations
from typing import Sequence
from tinygrad.renderer.amd.dsl import Inst, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, FLAT, SOPP
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp, DSOp
from tinygrad.renderer.amd.sqtt import (PacketType, InstOp, AluSrc, MemSrc, LAYOUT_HEADER, WAVESTART, WAVEEND, WAVERDY,
                                        INST, VALUINST, IMMEDIATE, IMMEDIATE_MASK, ALUEXEC, VMEMEXEC, TS_DELTA_OR_MARK, TS_DELTA_SHORT)
from tinygrad.viz.serve import amd_decode
from test.mockgpu.amd.sqtt_sim import MicroOp, RDNA3CU, Unit, EXEC, NEVER, VALUB4_OPS, VALUB2_NAMES

SHADER_CLK_HZ = 2_482_000_000
RT_CLK_HZ = 100_000_000
RT_PERIOD = 4096

DS_LOADS = {op for op in DSOp if op.name.startswith("DS_LOAD")} | {DSOp.DS_SWIZZLE_B32}
DS_ONE_CYC_WR = {DSOp.DS_APPEND, DSOp.DS_CONSUME, DSOp.DS_STORE_ADDTID_B32}
DS_TWO_ADDR = {op for op in DSOp if "2ADDR" in op.name}

def make_packet(cls: type[PacketType], time: int = 0, **fields) -> PacketType:
  raw = cls.encoding.set(0, None)
  for k, v in fields.items(): raw = cls._fields[k].set(raw, v)
  return cls.from_raw(raw, time)

def encode_packets(packets: list[PacketType]) -> bytes:
  nibbles = [(p._raw >> (4 * i)) & 0xF for p in packets for i in range(p._size_nibbles)]
  # 14 trailing zero nibbles flush a final 2-nibble packet through decode()'s 16-nibble lookahead without adding a NOP
  nibbles += [0] * (14 + len(nibbles) % 2)
  return bytes(lo | (hi << 4) for lo, hi in zip(nibbles[::2], nibbles[1::2]))

def _op_words(inst: Inst) -> int:
  return next((inst.op_regs[f] for f in ("vdst", "data0", "data") if f in inst.op_regs), 1)

def _parse_wave_trace(insts: dict[int, Inst], trace: memoryview | list[int]) -> list[MicroOp]:
  if trace[0] >= len(trace): raise RuntimeError("SQTT wave trace overflowed buffer; increase SQTT_BUFFER_SIZE")
  ops: list[MicroOp] = []
  i, n = 1, int(trace[0])
  while i <= n:
    inst = insts[int(trace[i])]
    has_addrs = isinstance(inst, DS) and hasattr(inst, "addr") and getattr(inst.addr, "offset", 0) >= 256
    addrs = [int(x) for x in trace[i + 2:i + 34]] if has_addrs else []
    if not (isinstance(inst, SOPP) and inst.op in (SOPPOp.S_ENDPGM, SOPPOp.S_ENDPGM_SAVED)):
      ops.append(MicroOp(int(trace[i]), inst, int(trace[i + 1]), addrs))
    i += 2 + (32 if has_addrs else 0)
  return ops

def _inst_packet(x: MicroOp, taken: bool) -> tuple[type[PacketType], dict, int]:
  inst, u = x.inst, x.unit
  if u is Unit.BRANCH: return INST, {"op": InstOp.JUMP if taken else InstOp.JUMP_NO}, 1
  if u is Unit.CALL: return INST, {"op": InstOp.CALL}, 1
  if u is Unit.BARRIER: return INST, {"op": InstOp.BARRIER}, 1
  if u is Unit.SMEM: return INST, {"op": InstOp.SMEM_RD}, 1
  if u is Unit.MSG: return INST, {"op": InstOp.MESSAGE}, 1
  if u is Unit.VMEM:
    w = _op_words(inst)
    if isinstance(inst, FLAT):
      op = InstOp.FLAT_RD_2 if x.writes else InstOp(min(InstOp.FLAT_WR_6.value, InstOp.FLAT_WR_3.value + w - 1))
      dur = 2 if x.writes else op.value - InstOp.FLAT_WR_3.value + 3
      return INST, {"op": op}, dur
    null_saddr = getattr(getattr(inst, "saddr", None), "offset", NULL.offset) == NULL.offset
    if x.writes: return INST, {"op": InstOp.SGMEM_RD_2 if null_saddr else InstOp.SGMEM_RD_1}, 2 if null_saddr else 1
    op = InstOp(min(InstOp.SGMEM_WR_6.value, InstOp.SGMEM_WR_2.value + int(null_saddr) + w - 1))
    return INST, {"op": op}, op.value - InstOp.SGMEM_WR_2.value + 2
  if not x.sent: return IMMEDIATE, {}, 1
  if u is Unit.LDS:
    if inst.op in DS_LOADS: return INST, {"op": InstOp.LDS_RD}, 1
    if inst.op in DS_ONE_CYC_WR: return INST, {"op": InstOp.LDS_WR_1}, 1
    dur = min(5, 1 + _op_words(inst) * (2 if inst.op in DS_TWO_ADDR else 1))
    return INST, {"op": InstOp[f"LDS_WR_{dur}"]}, dur
  if EXEC in x.writes: return INST, {"op": InstOp.SALU_WR_EXEC if u is Unit.SALU else InstOp.VALU1_WR_EXEC}, 1
  if u is Unit.SALU: return INST, {"op": InstOp.SALU}, 1
  if u is Unit.TRANS: return INST, {"op": InstOp.VALUT_4}, 4
  if getattr(inst, "op_name", "").startswith("V_INTERP"): return INST, {"op": InstOp.VINTERP}, 1
  if x.f64: return INST, {"op": InstOp.VALUB_16}, 16
  op_enum = getattr(inst, "op", None)
  if op_enum in VALUB4_OPS: return INST, {"op": InstOp.VALUB_4}, 4
  if getattr(op_enum, "name", "") in VALUB2_NAMES: return INST, {"op": InstOp.VALUB_2}, 2
  return VALUINST, {}, 1

def build_se_sqtt(code: bytes, wave_traces: Sequence[memoryview | list[int]] | None, total_wgs_on_se: int, n_waves_per_wg: int,
                  itrace: bool, simd_sel: int) -> bytes:
  events: list[tuple[int, int, type[PacketType], dict]] = []
  seq = 0
  def emit(t: int, cls: type[PacketType], **kw):
    nonlocal seq
    events.append((t, seq, cls, kw))
    seq += 1

  rt_base, t0 = 10_000_000_000, 100
  emit(t0, TS_DELTA_OR_MARK, rt=1, pl=0, delta=rt_base)
  max_t = t0 + RT_PERIOD

  if total_wgs_on_se > 0 and wave_traces:
    insts = amd_decode(code, "rdna3")
    # Simulate WGP0 / CU0 (two SIMDs: 0 and 2, or 1 and 3 if simd_sel is odd)
    cu_base = simd_sel & 1
    wgp0_ops = [_parse_wave_trace(insts, tr) for tr in wave_traces]
    simds = [k % 2 for k in range(len(wgp0_ops))]
    base_clk = 1000

    if itrace:
      cu = RDNA3CU(wgp0_ops, simds)
      cu.run()
      imm_masks: dict[int, list[int]] = {}
      alu_stamps: dict[int, int] = {}
      traced_slots: list[int] = []

      for k, w in enumerate(cu.waves):
        hw_simd, slot = cu_base + 2 * simds[k], k // 2
        emit(base_clk + k, WAVESTART, sa=0, simd=hw_simd, wgp=0, wave=slot, id7=k)
        end_t = base_clk + w.end()
        emit(end_t, WAVEEND, sa=0, simd=hw_simd, wgp=0, wave=slot)
        max_t = max(max_t, end_t)

        if hw_simd == simd_sel:
          traced_slots.append(slot)
          for j, x in enumerate(w.ops):
            taken = j + 1 < len(w.ops) and w.ops[j + 1].pc != x.pc + x.inst.size()
            cls, fields, _ = _inst_packet(x, taken)
            it = base_clk + x.t
            if cls is IMMEDIATE: imm_masks.setdefault(it, []).append(slot)
            else: emit(it, cls, wave=slot, **fields)
            if x.stamp != NEVER:
              st = base_clk + x.stamp
              if x.unit is Unit.LDS: emit(st, VMEMEXEC, src=MemSrc.LDS)
              elif x.unit is Unit.VMEM: emit(st, VMEMEXEC, src=MemSrc.VMEM)
              else: alu_stamps[st] = alu_stamps.get(st, 0) | (AluSrc.SALU.value if x.unit in (Unit.SALU, Unit.CALL, Unit.SMEM) else AluSrc.VALU.value)
        elif hw_simd == (simd_sel ^ 2):
          # Paired SIMD in the same CU emits OTHER_LDS_* / OTHER_VMEM_* and *_ALT exec packets
          for x in w.ops:
            if not x.sent or x.stamp == NEVER: continue
            _, _, dur = _inst_packet(x, False)
            it, st = base_clk + x.t, base_clk + x.stamp
            if x.unit is Unit.LDS:
              emit(it, INST, wave=slot, op=InstOp(min(0x54, 0x50 + dur - 1)))
              emit(st, VMEMEXEC, src=MemSrc.LDS_ALT)
            elif x.unit is Unit.VMEM:
              op_base = 0x55 if isinstance(x.inst, FLAT) else 0x5a
              op_max = 0x59 if isinstance(x.inst, FLAT) else 0x66
              emit(it, INST, wave=slot, op=InstOp(min(op_max, op_base + max(1, dur - (1 if isinstance(x.inst, FLAT) else 0)) - 1)))
              emit(st, VMEMEXEC, src=MemSrc.VMEM_ALT)

      for it, slots in sorted(imm_masks.items()):
        if len(slots) == 1: emit(it, IMMEDIATE, wave=slots[0])
        else: emit(it, IMMEDIATE_MASK, mask=sum(1 << s for s in slots))
      for st, src in sorted(alu_stamps.items()):
        emit(st, ALUEXEC, src=AluSrc(src))
      if traced_slots and cu.barrier.release_cycles:
        wmask = sum(1 << s for s in traced_slots)
        for rc in cu.barrier.release_cycles:
          emit(base_clk + rc, WAVERDY, mask=wmask)
    else:
      for k in range(len(wgp0_ops)):
        hw_simd, slot = cu_base + 2 * simds[k], k // 2
        st, et = base_clk + k, base_clk + k + max(40, len(wgp0_ops[k]) * 4)
        emit(st, WAVESTART, sa=0, simd=hw_simd, wgp=0, wave=slot, id7=k)
        emit(et, WAVEEND, sa=0, simd=hw_simd, wgp=0, wave=slot)
        max_t = max(max_t, et)

    # Emit coarse WAVESTART/WAVEEND for additional workgroups assigned to non-0 WGPs on this SE
    span = max(40, max_t - base_clk)
    for wg_idx in range(1, total_wgs_on_se):
      wgp, sa = wg_idx % 4, (wg_idx // 4) % 2
      if sa == 0 and wgp == 0: wgp = 1
      for lw in range(n_waves_per_wg):
        hw_simd, slot = 2 * (lw % 2), (wg_idx // 8) * ((n_waves_per_wg + 1) // 2) + lw // 2
        st = base_clk + wg_idx * 8 + lw
        emit(st, WAVESTART, sa=sa, simd=hw_simd, wgp=wgp, wave=slot & 0xF, id7=(wg_idx * n_waves_per_wg + lw) & 0x3FFF)
        emit(st + span, WAVEEND, sa=sa, simd=hw_simd, wgp=wgp, wave=slot & 0xF)
        max_t = max(max_t, st + span)

  # Realtime clock markers (100 MHz) every 4096 shader cycles so VIZ sqtt_timeline derives Shader Clock frequency
  t_mark = t0 + RT_PERIOD
  while t_mark <= max_t + RT_PERIOD:
    emit(t_mark, TS_DELTA_OR_MARK, rt=1, pl=0, delta=rt_base + (t_mark - t0) * RT_CLK_HZ // SHADER_CLK_HZ)
    t_mark += RT_PERIOD

  # Sort chronologically (WAVEEND after same-cycle exec/inst packets) and pack with delta bridges
  events.sort(key=lambda e: (e[0], 1 if e[2] is WAVEEND else 0, e[1]))
  packets: list[PacketType] = [make_packet(LAYOUT_HEADER, 0, layout=3, simd=simd_sel, sel_a=6)]
  now = 0
  for t, _, cls, fields in events:
    gap = max(0, t - now)
    if cls is TS_DELTA_OR_MARK and fields.get("rt"):
      if gap > 0:
        packets.append(make_packet(TS_DELTA_SHORT, t, delta=gap - 4) if 4 <= gap <= 19 else make_packet(TS_DELTA_OR_MARK, t, delta=gap))
        now = t
      packets.append(make_packet(cls, now, **fields))
      continue
    delta_f = getattr(cls, "delta", None)
    max_d = delta_f.mask if delta_f is not None else 0
    if gap > max_d:
      rem = gap - max_d
      if 4 <= rem <= 19:
        packets.append(make_packet(TS_DELTA_SHORT, now + rem, delta=rem - 4))
        gap = max_d
      else:
        packets.append(make_packet(TS_DELTA_OR_MARK, t, delta=gap))
        gap = 0
    now = max(now, t)
    packets.append(make_packet(cls, now, delta=gap, **fields))
  return encode_packets(packets)
