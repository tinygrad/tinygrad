import time
from tinygrad.runtime.autogen import mesa
from tinygrad.helpers import to_mv
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.emu import IR3Emulator

# a630, matching the chip_id QCOMCompiler compiles for
CHIP_ID = 0x6030001

def data64(lo:int, hi:int) -> int: return (hi << 32) | lo

def fld(val:int, name:str) -> int: return (val & getattr(mesa, f"{name}__MASK")) >> getattr(mesa, f"{name}__SHIFT")

class QCOMGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.regs: dict[int, int] = {}
    self.mapped_ranges: set[tuple[int, int]] = set()
    self.const_base: int = 0

  def map_range(self, vaddr, size): self.mapped_ranges.add((vaddr, size))
  def unmap_range(self, vaddr, size): self.mapped_ranges.discard((vaddr, size))

  def _always_on_counter(self) -> int: return int(time.perf_counter() * 19200000)  # 19.2MHz, matches QCOMSignal timestamp_divider

  def execute(self, gpuaddr:int, size:int):
    q, i = to_mv(gpuaddr, size).cast('I'), 0
    while i < len(q):
      hdr = q[i]
      i += 1
      if (hdr & 0xF0000000) == mesa.CP_TYPE4_PKT:
        reg, cnt = (hdr >> 8) & 0x3FFFF, hdr & 0x7F
        for j in range(cnt): self.regs[reg + j] = q[i + j]
      elif (hdr & 0xF0000000) == mesa.CP_TYPE7_PKT:
        cnt, op = hdr & 0x3FFF, (hdr >> 16) & 0x7F
        self._exec_pkt7(op, [q[i + j] for j in range(cnt)])
      else: raise RuntimeError(f"unknown pm4 packet header {hdr:#x} at {gpuaddr + i * 4:#x}")
      i += cnt

  def _exec_pkt7(self, op:int, vals:list[int]):
    # everything below runs synchronously and in order, so idle/flush waits are no-ops
    if op in {mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER}: pass
    elif op == mesa.CP_EVENT_WRITE:
      if (event:=fld(vals[0], 'CP_EVENT_WRITE_0_EVENT')) == mesa.CACHE_FLUSH_TS: to_mv(data64(vals[1], vals[2]), 4).cast('I')[0] = vals[3]
      elif event != mesa.CACHE_INVALIDATE: raise RuntimeError(f"unsupported CP_EVENT_WRITE event {event}")
    elif op == mesa.CP_REG_TO_MEM:
      reg, is64 = fld(vals[0], 'CP_REG_TO_MEM_0_REG'), bool(vals[0] & mesa.CP_REG_TO_MEM_0_64B)
      val = self._always_on_counter() if reg == mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER else self.regs.get(reg, 0)
      to_mv(data64(vals[1], vals[2]), 8 if is64 else 4).cast('Q' if is64 else 'I')[0] = val if is64 else val & 0xFFFFFFFF
    elif op == mesa.CP_WAIT_REG_MEM:
      if (poll:=fld(vals[0], 'CP_WAIT_REG_MEM_0_POLL')) != mesa.POLL_MEMORY: raise RuntimeError(f"unsupported CP_WAIT_REG_MEM poll {poll}")
      if (fn:=fld(vals[0], 'CP_WAIT_REG_MEM_0_FUNCTION')) != mesa.WRITE_GE: raise RuntimeError(f"unsupported CP_WAIT_REG_MEM function {fn}")
      # in-order execution means the value is already there, so verify rather than spin
      got, ref, mask = to_mv(data64(vals[1], vals[2]), 4).cast('I')[0], fld(vals[3], 'CP_WAIT_REG_MEM_3_REF'), fld(vals[4], 'CP_WAIT_REG_MEM_4_MASK')
      if (got & mask) < (ref & mask): raise RuntimeError(f"CP_WAIT_REG_MEM would hang: {got:#x} < {ref:#x}")
    elif op == mesa.CP_LOAD_STATE6_FRAG:
      blk, styp = fld(vals[0], 'CP_LOAD_STATE6_0_STATE_BLOCK'), fld(vals[0], 'CP_LOAD_STATE6_0_STATE_TYPE')
      # the constant file for the compute shader is just the kernel argument buffer, loaded indirectly
      if blk == mesa.SB6_CS_SHADER and styp == mesa.ST_CONSTANTS: self.const_base = data64(vals[1], vals[2])
      elif blk == mesa.SB6_CS_SHADER and styp == mesa.ST_SHADER: pass  # shader address also comes from SP_CS_CNTL_0
      elif blk in {mesa.SB6_CS_TEX}: raise NotImplementedError("texture/sampler state is not supported yet")
      elif styp == mesa.ST6_UAV: raise NotImplementedError("image (UAV) state is not supported yet")
      else: raise RuntimeError(f"unsupported CP_LOAD_STATE6 block={blk} type={styp}")
    elif op in {mesa.CP_EXEC_CS, mesa.CP_RUN_OPENCL}: self._dispatch()
    else: raise RuntimeError(f"unsupported pkt7 opcode {op:#x}")

  def _dispatch(self):
    nd = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0]
    local = tuple(fld(nd, f'A6XX_SP_CS_NDRANGE_0_LOCALSIZE{c}') + 1 for c in "XYZ")
    # NDRANGE_1/3/5 hold global_size*local_size; NDRANGE_9..11 hold the workgroup counts
    threads = tuple(self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0 + 1 + 2 * i] for i in range(3))
    groups = tuple(self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0 + 9 + i] for i in range(3))
    prg_offset = self.regs[mesa.REG_A6XX_SP_CS_CNTL_0 + 3]
    shader = data64(self.regs[mesa.REG_A6XX_SP_CS_CNTL_0 + 4], self.regs[mesa.REG_A6XX_SP_CS_CNTL_0 + 5]) + prg_offset * 8
    if self.const_base == 0: raise RuntimeError("dispatch without a loaded constant buffer")
    for gz in range(groups[2]):
      for gy in range(groups[1]):
        for gx in range(groups[0]):
          shared = bytearray(64 << 10)   # one local-memory allocation per workgroup
          gid, running = (gx, gy, gz), []
          for lz in range(local[2]):
            for ly in range(local[1]):
              for lx in range(local[0]):
                lid = (lx, ly, lz)
                if any(g * l + i >= t for g, l, i, t in zip(gid, local, lid, threads)): continue
                emu = IR3Emulator(self.const_base, sorted(self.mapped_ranges), shared)
                # CL-mode register preload, established by probing what the QCOM CL compiler reads
                for i in range(3):
                  emu.r[0 * 4 + i] = lid[i]                 # r0.x/y/z: local invocation id
                  emu.r[51 * 4 + i] = gid[i] * local[i]     # r51.x/y/z: global id base
                  emu.r[48 * 4 + i] = local[i]              # r48.x/y/z: local size
                emu.r[51 * 4 + 3], emu.r[52 * 4 + 0], emu.r[52 * 4 + 1] = gid[0], gid[1], gid[2]
                running.append((lid, emu.run_steps(shader)))
          # Threads of a workgroup run one barrier interval at a time: each is advanced until it
          # hits `bar` or ends, and only then does the next interval start for any of them.
          while running:
            arrived = []
            for lid, thread in running:
              if (pc:=next(thread, None)) is not None: arrived.append((lid, thread, pc))
            if arrived and len(arrived) != len(running):
              lid, _, pc = arrived[0]
              raise RuntimeError(f"thread {lid} reached the barrier at pc {pc} but only "
                                 f"{len(arrived)} of {len(running)} threads did")
            running = [(lid, thread) for lid, thread, _ in arrived]
