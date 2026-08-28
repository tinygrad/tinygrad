from __future__ import annotations
import struct, time
from tinygrad.helpers import to_mv, getenv
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.emu import A630Emu, write_u32, write_u64, read_u32

DEBUG_QCOM = getenv("MOCKQCOM_DEBUG", 0)

class A630GPU:
  def __init__(self, check_range=None):
    self.regs: dict[int, int] = {}
    self.always_on = 0
    self.shader = b""
    self.constants = b""
    self.local_size = (1, 1, 1)
    self.global_size = (1, 1, 1)
    self.pending: list[bytes] = []
    self.emu = A630Emu(check_range)

  def _u64(self, lo:int, hi:int) -> int: return (hi << 32) | lo

  def execute_ib(self, gpuaddr:int, size:int) -> None:
    self.pending.append(bytes(to_mv(gpuaddr, size)))
    self.resume()

  def resume(self) -> None:
    while self.pending:
      if dat := self._execute(self.pending[0]):
        self.pending[0] = dat
        return
      self.pending.pop(0)

  def _execute(self, dat:bytes) -> bytes:
    ptr = 0
    while ptr + 4 <= len(dat):
      cmd = struct.unpack_from("I", dat, ptr)[0]
      if (cmd >> 24) == 0x70:
        opcode, n = (cmd >> 16) & 0x7F, cmd & 0x3FFF
        vals = struct.unpack_from("I" * n, dat, ptr + 4) if n else ()
        if self._pkt7(opcode, vals) is False: return dat[ptr:]
        ptr += 4 + 4 * n
      elif (cmd >> 28) == 0x4:
        offset, n = (cmd >> 8) & 0x7FFFF, cmd & 0x7F
        vals = struct.unpack_from("I" * n, dat, ptr + 4) if n else ()
        for i, v in enumerate(vals): self.regs[offset + i] = v
        self._pkt4(offset, vals)
        ptr += 4 + 4 * n
      else:
        if DEBUG_QCOM: print(f"qcom emu unk pkt {cmd:#x} at {ptr:#x}")
        ptr += 4
    return b""

  def _pkt4(self, offset:int, vals:tuple[int, ...]) -> None:
    # SP_CS_NDRANGE_0: kerneldim + local sizes, then global dims
    if offset == mesa.REG_A6XX_SP_CS_NDRANGE_0 and len(vals) >= 7:
      locx = ((vals[0] & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT) + 1
      locy = ((vals[0] & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT) + 1
      locz = ((vals[0] & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT) + 1
      self.local_size = (locx, locy, locz)
      self.global_size = (max(1, vals[1]), max(1, vals[3]), max(1, vals[5]))

  def _pkt7(self, opcode:int, vals:tuple[int, ...]) -> bool|None:
    if opcode == mesa.CP_WAIT_FOR_IDLE: return
    if opcode == mesa.CP_WAIT_MEM_WRITES: return
    if opcode == mesa.CP_SET_MARKER: return
    if opcode == mesa.CP_NOP: return

    if opcode == mesa.CP_EVENT_WRITE and len(vals) >= 1:
      event = vals[0] & 0xFF
      if event == mesa.CACHE_FLUSH_TS and len(vals) >= 4:
        addr = self._u64(vals[1], vals[2])
        write_u32(addr, vals[3])
        if DEBUG_QCOM: print(f"EVENT_WRITE CACHE_FLUSH_TS {addr:#x}={vals[3]}")
      return

    if opcode == mesa.CP_REG_TO_MEM and len(vals) >= 3:
      self.always_on = max(self.always_on + 1, time.perf_counter_ns() * 19_200_000 // 1_000_000_000)
      dest = self._u64(vals[1], vals[2])
      write_u64(dest, self.always_on)
      return

    if opcode == mesa.CP_WAIT_REG_MEM and len(vals) >= 5:
      addr = self._u64(vals[1], vals[2])
      ref, mask = vals[3], vals[4]
      # WRITE_GE: *addr >= ref. Signals are written by EVENT_WRITE before wait.
      cur = read_u32(addr) & mask
      if DEBUG_QCOM and cur < (ref & mask): print(f"WAIT_REG_MEM {addr:#x} {cur:#x} < {ref:#x}")
      return cur >= (ref & mask)

    if opcode == mesa.CP_LOAD_STATE6_FRAG and len(vals) >= 3:
      state_type = (vals[0] >> 14) & 0x3
      state_block = (vals[0] >> 18) & 0xF
      num_unit = vals[0] >> 22
      src = self._u64(vals[1], vals[2])
      if state_block == 13:  # SB6_CS_SHADER
        if state_type == 0:  # ST_SHADER
          n = num_unit * 128
          self.shader = bytes(to_mv(src, n))
        elif state_type == 1:  # ST_CONSTANTS
          n = num_unit * 4
          self.constants = bytes(to_mv(src, n))
      return

    if opcode in (mesa.CP_RUN_OPENCL, mesa.CP_EXEC_CS):
      gs = self.global_size
      if opcode == mesa.CP_EXEC_CS and len(vals) >= 4:
        gs = (vals[1] or 1, vals[2] or 1, vals[3] or 1)
      cfg = self.regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0xfcfcfcfc)
      lid = (cfg & mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT
      wgid = (cfg & mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT
      if lid != 0xfc:
        if lid % 4: raise ValueError(f"unaligned local ID register {lid:#x}")
        lid //= 4
      if wgid != 0xfc:
        if wgid % 4: raise ValueError(f"unaligned workgroup ID register {wgid:#x}")
        wgid //= 4
      self.emu.execute_cs(self.shader, self.constants, gs, self.local_size, local_id_register=lid, workgroup_id_register=wgid)
      return
