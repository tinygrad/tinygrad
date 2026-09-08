import time
from test.mockgpu.gpu import VirtGPU
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa

class QCOMGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.regs: dict[int, int] = {}
    self.const_addr = 0
    self.ibs: list[QCOMExecutor] = []

  def submit_ib(self, gpuaddr: int, size: int):
    self.ibs.append(QCOMExecutor(self, gpuaddr, size))

  def execute(self):
    while self.ibs and self.ibs[0].execute(): self.ibs.pop(0)

class QCOMExecutor:
  def __init__(self, gpu: QCOMGPU, base: int, size: int):
    # snapshot IB so bump-allocator wrap / later bind patches can't clobber in-flight work
    self._buf = bytearray(to_mv(base, size))
    self.gpu, self.ib, self.ptr, self.end = gpu, memoryview(self._buf).cast("I"), 0, size // 4
    self.pending_err: BaseException|None = None

  def _next(self) -> int:
    x = self.ib[self.ptr]
    self.ptr += 1
    return x

  def execute(self):
    while self.ptr < self.end:
      hdr = self._next()
      typ = hdr >> 28
      if typ == 7:
        op, n = (hdr >> 16) & 0x7F, hdr & 0x3FFF
        payload = [self._next() for _ in range(n)]
        if not self._exec7(op, payload): return False
      elif typ == 4:
        reg, n = (hdr >> 8) & 0x3FFFF, hdr & 0x7F
        for i in range(n): self.gpu.regs[reg + i] = self._next()
      else:
        raise RuntimeError(f"unknown PM4 type {typ:#x} hdr={hdr:#x}")
    if self.pending_err is not None: raise self.pending_err
    return True

  def _wait_reg_mem(self, p:list[int]) -> bool:
    addr = p[1] | (p[2] << 32)
    ref, mask = p[3], p[4]
    mval = to_mv(addr, 4).cast("I")[0] & mask
    fn = p[0] & 0x7
    if fn == mesa.WRITE_GE: ok = mval >= ref
    elif fn == mesa.WRITE_EQ: ok = mval == ref
    elif fn == mesa.WRITE_NE: ok = mval != ref
    elif fn == mesa.WRITE_ALWAYS: ok = True
    else: raise RuntimeError(f"unsupported wait function {fn}")
    if not ok: self.ptr -= len(p) + 1
    return ok

  def _exec7(self, op: int, p: list[int]):
    if op in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER):
      return True
    if op == mesa.CP_LOAD_STATE6_FRAG:
      if ((p[0] >> 14) & 3) == mesa.ST_CONSTANTS:
        self.gpu.const_addr = p[1] | (p[2] << 32)
      return True
    if op == mesa.CP_EVENT_WRITE:
      if len(p) >= 4:
        to_mv(p[1] | (p[2] << 32), 8).cast("Q")[0] = p[3]
      return True
    if op == mesa.CP_WAIT_REG_MEM:
      return self._wait_reg_mem(p)
    if op == mesa.CP_REG_TO_MEM:
      to_mv(p[1] | (p[2] << 32), 8).cast("Q")[0] = int(time.perf_counter() * 19.2e6)
      return True
    if op == mesa.CP_RUN_OPENCL:
      raise RuntimeError("CP_RUN_OPENCL (QCOMCL) not in this path")
    if op == mesa.CP_EXEC_CS:
      r = self.gpu.regs
      base = r[mesa.REG_A6XX_SP_CS_BASE] | (r.get(mesa.REG_A6XX_SP_CS_BASE + 1, 0) << 32)
      sz = r[mesa.REG_A6XX_SP_CS_INSTR_SIZE] * 128
      bs = (r.get(mesa.REG_A6XX_SP_CS_CNTL_0, 0) & mesa.A6XX_SP_CS_CNTL_0_BRANCHSTACK__MASK) >> mesa.A6XX_SP_CS_CNTL_0_BRANCHSTACK__SHIFT
      try:
        from test.mockgpu.qcom.mine_emu_wave import run_shader
        run_shader(base, sz, self.gpu.const_addr, r, bs)
      except Exception as e: self.pending_err = e
      return True
    raise RuntimeError(f"unhandled pkt7 op {op:#x}")
