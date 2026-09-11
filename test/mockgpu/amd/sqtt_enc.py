# SQTT trace encoder for the emulator (the decoder lives in tinygrad/renderer/amd/sqtt.py).
# run_asm emits packets inline as instructions execute; finished traces end up in emu.sqtt_traces.
from __future__ import annotations
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.renderer.amd.sqtt import (_build_decode_tables, PACKET_TYPES_RDNA3, PacketType, InstOp,
                                        LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, TS_DELTA_OR_MARK)

_NIB_COUNTS = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _emit_nibbles(nibbles: list[int], pkt_cls: type[PacketType], **kwargs):
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  nibbles.extend((raw >> (i * 4)) & 0xF for i in range(_NIB_COUNTS[pkt_cls]))

def make_encoder():
  """Build an SQTT trace encoder for the emulator. Returns (emit, finish, finalize)."""
  from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3
  from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4
  from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
  from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
  from tinygrad.runtime.autogen.amd.cdna import ins as irc
  import re

  def _kinds(*names: str) -> tuple[type[Inst], ...]:
    return tuple(getattr(m, n) for m in (ir3, ir4, irc) for n in names if hasattr(m, n))
  _SOPP, _SMEM, _DS = _kinds('SOPP'), _kinds('SMEM'), _kinds('DS')
  _GLOBAL, _FLAT, _SCRATCH = _kinds('GLOBAL', 'VGLOBAL'), _kinds('FLAT', 'VFLAT'), _kinds('SCRATCH', 'VSCRATCH')
  _VALU = _kinds('VOP1', 'VOP2', 'VOP3', 'VOP3P', 'VOP3PX2', 'VOPC', 'VOPD', 'VOP3SD', 'VOP3_SDST', 'VOP1_SDST')

  # SOPP classification sets
  _SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value, SOPPOp3.S_DELAY_ALU.value}
  _SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                     SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value, SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
  for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
              SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
              SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
    _SOPP_IMMEDIATE.add(_op.value)
  _SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
  if hasattr(SOPPOp4, 'S_BARRIER_WAIT'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
  if hasattr(SOPPOp4, 'S_BARRIER_LEAVE'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
  _SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                  SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                  SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}

  # VALU sub-classification patterns
  _VALUT_4_RE = re.compile(r'V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_')
  _VALUB_2_RE = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64')
  _VALUB_4_RE = re.compile(r'V_MAD_(U|I)64')
  _VALUB_16_RE = re.compile(r'V_\w+_F64')

  def _valu_op(op_name: str) -> InstOp|None:
    if 'CMPX' in op_name: return InstOp.VALU1_WR_EXEC
    if _VALUB_2_RE.search(op_name): return InstOp.VALUB_2
    if _VALUB_4_RE.search(op_name): return InstOp.VALUB_4
    if _VALUB_16_RE.search(op_name): return InstOp.VALUB_16
    if _VALUT_4_RE.search(op_name): return InstOp.VALUT_4
    return None

  def _mem_op(t: type[Inst], op_name: str) -> InstOp:
    is_store = "STORE" in op_name
    if issubclass(t, _DS): return InstOp.LDS_WR_2 if is_store else InstOp.LDS_RD
    if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1
    if issubclass(t, _FLAT) or issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
    return InstOp.SALU

  nibbles: list[int] = []
  started: set[int] = set()
  time = 0
  _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)

  def packet(cls: type[PacketType], timestamp: int|None, **kwargs):
    nonlocal time
    target = time + 1 if timestamp is None else timestamp
    if type(target) is not int or target < time: raise ValueError("SQTT timestamps must be nondecreasing integers")
    delta = target - time
    # Long gaps need a timestamp packet: instruction packets have only 2-3 delta bits.
    if delta > cls.__dict__['delta'].mask:
      if delta > TS_DELTA_OR_MARK.delta.mask: raise ValueError("SQTT timestamp gap exceeds the 36-bit packet range")
      _emit_nibbles(nibbles, TS_DELTA_OR_MARK, delta=delta)
      delta = 0
    _emit_nibbles(nibbles, cls, delta=delta, **kwargs)
    time = target

  def emit(wave_id: int, inst: Inst, branch_taken: bool|None, timestamp: int|None = None):
    """Emit an instruction at an optional absolute trace tick; omitted ticks preserve legacy ordering."""
    if timestamp is not None and (type(timestamp) is not int or timestamp < time):
      raise ValueError("SQTT timestamps must be nondecreasing integers")
    w = wave_id & 0x1F
    if wave_id not in started:
      packet(WAVESTART, timestamp, simd=0, wgp=0, wave=w, id7=wave_id)
      started.add(wave_id)
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, 'op') else 0, inst.op.name if hasattr(inst, 'op') else ""
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP:
        # A skipped instruction can still advance modeled time, without an instruction packet.
        if timestamp is not None and timestamp > time: packet(TS_DELTA_OR_MARK, timestamp)
        return
      if inst_op in _SOPP_IMMEDIATE: packet(IMMEDIATE, timestamp, wave=w)
      elif inst_op in _SOPP_BARRIER: packet(INST, timestamp, wave=w, op=InstOp.BARRIER)
      elif inst_op in _SOPP_BRANCH: packet(INST, timestamp, wave=w, op=InstOp.JUMP if branch_taken else InstOp.JUMP_NO)
      else: packet(INST, timestamp, wave=w, op=InstOp.SALU)
    elif issubclass(inst_type, _VALU):
      if (op := _valu_op(op_name)) is None: packet(VALUINST, timestamp, wave=w)
      else: packet(INST, timestamp, wave=w, op=op)
    elif issubclass(inst_type, _SMEM): packet(INST, timestamp, wave=w, op=InstOp.SMEM_RD)
    else: packet(INST, timestamp, wave=w, op=_mem_op(inst_type, op_name))

  def finish(wave_id: int, timestamp: int|None = None):
    """Emit WAVEEND for a completed wave."""
    if wave_id in started:
      packet(WAVEEND, timestamp, simd=0, wgp=0, wave=wave_id & 0x1F)
      started.remove(wave_id)

  def finalize() -> bytes:
    """Pad and return the encoded SQTT blob."""
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return bytes(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4) for i in range(0, len(nibbles), 2))

  return emit, finish, finalize
