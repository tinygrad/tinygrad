import time
import numpy as np
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom import emu

def field(val:int, name:str) -> int: return (val & getattr(mesa, f"{name}__MASK")) >> getattr(mesa, f"{name}__SHIFT")
def u64(lo:int, hi:int) -> int: return lo | (hi << 32)

class QCOMGPU:
  def __init__(self):
    self.regs:dict[int, int] = {}
    self.consts, self.shader = np.zeros(4096, np.uint32), b""
    self.error:Exception|None = None
    self.pending:list[list[int]] = []

  def submit(self, addr:int, ndwords:int):
    self.pending.append(list(to_mv(addr, ndwords * 4).cast("I")))
    self.progress()

  def progress(self, drain=False): # drain runs only the signal writes left after an error, so later submits don't hang on their waits
    while self.pending and (drain or self.error is None):
      words = self.pending[0]
      while words:
        hdr = words[0]
        if hdr >> 28 == mesa.CP_TYPE4_PKT >> 28:
          reg, cnt = (hdr >> 8) & 0x3FFFF, hdr & 0x7F
          for k in range(cnt): self.regs[reg + k] = words[1 + k]
        elif hdr >> 28 == mesa.CP_TYPE7_PKT >> 28:
          cnt, op = hdr & 0x3FFF, (hdr >> 16) & 0x7F
          try:
            if (not drain or op == mesa.CP_EVENT_WRITE) and not self._exec_pkt7(op, words[1:1+cnt]): return
          except Exception as e:
            self.error = e
            del words[:1 + cnt]
            return
        else:
          self.error = RuntimeError(f"unknown packet header {hdr:#x}")
          break
        del words[:1 + cnt]
      self.pending.pop(0)

  def _exec_pkt7(self, op:int, d:list[int]) -> bool: # False when the CP has to wait
    if op in (mesa.CP_SET_MARKER, mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES): return True
    if op == mesa.CP_EVENT_WRITE:
      if (d[0] & 0xFF) == mesa.CACHE_FLUSH_TS: to_mv(u64(d[1], d[2]), 4).cast("I")[0] = d[3]
    elif op == mesa.CP_WAIT_REG_MEM: return (to_mv(u64(d[1], d[2]), 4).cast("I")[0] & d[4]) >= d[3]
    elif op == mesa.CP_REG_TO_MEM: to_mv(u64(d[1], d[2]), 8).cast("Q")[0] = time.perf_counter_ns() * 192 // 10000 # 19.2MHz ticks
    elif op == mesa.CP_LOAD_STATE6_FRAG: self._load_state(d)
    elif op == mesa.CP_EXEC_CS: self._exec_cs(d[1:4])
    elif op == mesa.CP_RUN_OPENCL: self._exec_cs([self.regs[mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X + k] for k in range(3)])
    else: raise RuntimeError(f"unsupported pkt7 opcode {op:#x}")
    return True

  def _load_state(self, d:list[int]):
    typ, block, num, off = [field(d[0], f"CP_LOAD_STATE6_0_{f}") for f in ("STATE_TYPE", "STATE_BLOCK", "NUM_UNIT", "DST_OFF")]
    addr = u64(d[1], d[2])
    if (typ, block) == (mesa.ST_CONSTANTS, mesa.SB6_CS_SHADER): self.consts[off*4:(off+num)*4] = np.frombuffer(to_mv(addr, num * 16), np.uint32)
    elif (typ, block) == (mesa.ST_SHADER, mesa.SB6_CS_SHADER): self.shader = bytes(to_mv(addr, num * 128))

  def _reg64(self, r:int) -> int: return u64(self.regs[r], self.regs[r + 1])

  def _images(self, addr:int, num:int, tex:bool) -> list[emu.Image]:
    ret = []
    for k in range(num):
      d = to_mv(addr + k * 64, 64).cast("I")
      fmt = field(d[0], "A6XX_TEX_CONST_0_FMT")
      if fmt not in (mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT): raise RuntimeError(f"unsupported image format {fmt}")
      swiz_ok = all(field(d[0], f"A6XX_TEX_CONST_0_SWIZ_{c}") == getattr(mesa, f"A6XX_TEX_{c}") for c in "XYZW")
      if field(d[2], "A6XX_TEX_CONST_2_TYPE") != mesa.A6XX_TEX_2D or (tex and not swiz_ok):
        raise RuntimeError(f"unsupported image descriptor {d[0]:#x} {d[2]:#x}")
      ret.append(emu.Image(u64(d[4], d[5]), field(d[1], "A6XX_TEX_CONST_1_WIDTH"), field(d[1], "A6XX_TEX_CONST_1_HEIGHT"),
                           field(d[2], "A6XX_TEX_CONST_2_PITCH"), np.dtype(np.float32 if fmt == mesa.FMT6_32_32_32_32_FLOAT else np.float16)))
    return ret

  def _exec_cs(self, groups:list[int]):
    nd, cfg, mode = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0], self.regs[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0], self.regs[mesa.REG_A6XX_SP_MODE_CNTL]
    demote = bool(mode & mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE)
    if field(mode, "A6XX_SP_MODE_CNTL_ISAMMODE") != (mesa.ISAMMODE_GL if demote else mesa.ISAMMODE_CL):
      raise RuntimeError(f"unsupported SP_MODE_CNTL {mode:#x}")
    if self.regs[mesa.REG_A6XX_SP_CS_CNTL_0] & mesa.A6XX_SP_CS_CNTL_0_MERGEDREGS: raise RuntimeError("merged registers are not emulated")
    local = tuple(field(nd, f"A6XX_SP_CS_NDRANGE_0_LOCALSIZE{c}") + 1 for c in "XYZ")
    lmem_size = (field(self.regs[mesa.REG_A6XX_SP_CS_CNTL_1], "A6XX_SP_CS_CNTL_1_SHARED_SIZE") + 1) * 1024
    pvt_size = field(self.regs[mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM], "A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM") * 512 or (0 if demote else 512)
    ntex, nuav = (field(self.regs[mesa.REG_A6XX_SP_CS_CONFIG], f"A6XX_SP_CS_CONFIG_{f}") for f in ("NTEX", "NUAV"))
    textures = self._images(self._reg64(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE), ntex, tex=True) if ntex else []
    ibos = self._images(self._reg64(mesa.REG_A6XX_SP_CS_UAV_BASE), nuav, tex=False) if nuav else []
    emu.run(emu.Dispatch(self.shader, self.consts, local, tuple(groups), field(cfg, "A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID"),
                         field(cfg, "A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID"), lmem_size, pvt_size,
                         textures, ibos, demote, self.regs[mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET]))
