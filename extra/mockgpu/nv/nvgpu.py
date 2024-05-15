import ctypes, ctypes.util, time
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
from enum import Enum, auto
from extra.mockgpu.gpu import VirtGPU
from tinygrad.helpers import to_mv, init_c_struct_t

def make_qmd_struct_type():
  fields = []
  bits = [(name,dt) for name,dt in nv_gpu.__dict__.items() if name.startswith("NVC6C0_QMDV03_00") and isinstance(dt, tuple)]
  bits += [(name+f"_{i}",dt(i)) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith("NVC6C0_QMDV03_00") and callable(dt)]
  bits = sorted(bits, key=lambda x: x[1][1])
  for i,(name, data) in enumerate(bits):
    if i > 0 and (gap:=(data[1] - bits[i-1][1][0] - 1)) != 0:  fields.append((f"_reserved{i}", ctypes.c_uint32, gap))
    fields.append((name.replace("NVC6C0_QMDV03_00_", "").lower(), ctypes.c_uint32, data[0]-data[1]+1))
  return init_c_struct_t(tuple(fields))
qmd_struct_t = make_qmd_struct_type()
assert ctypes.sizeof(qmd_struct_t) == 0x40 * 4

gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501

class SchedResult(Enum): CONT = auto(); YIELD = auto() # noqa: E702

class GPFIFO:
  def __init__(self, token, base, entries_cnt):
    self.token, self.base, self.entries_cnt = token, base, entries_cnt
    self.gpfifo = to_mv(self.base, self.entries_cnt * 8).cast("Q")
    self.ctrl = nv_gpu.AmpereAControlGPFifo.from_address(self.base + self.entries_cnt * 8)
    self.state = {}

    # Exec state
    self.buf = None
    self.off = 0

  def _next_dword(self):
    x = self.buf[self.off]
    self.off += 1
    return x

  def _next_header(self):
    header = self._next_dword()
    typ = (header >> 28) & 0b111
    size = (header >> 16) & 0xFFF
    subc = (header >> 13) & 0x7
    mthd = (header & 0x1FFF) << 2
    return typ, size, subc, mthd

  def _state(self, reg): return self.state[reg]
  def _state64(self, reg): return (self.state[reg] << 32) + self.state[reg + 4]
  def _state64_le(self, reg): return (self.state[reg + 4] << 32) + self.state[reg]
  
  def execute(self):
    while self.ctrl.GPGet != self.ctrl.GPPut:
      entry = self.gpfifo[self.ctrl.GPGet]
      ptr = ((entry >> 2) & 0xfffffffff) << 2
      sz = ((entry >> 42) & 0x1fffff) << 2
      # print("gpfifo", hex(ptr), sz // 4, entry, self.ctrl.GPGet, self.ctrl.GPPut, self.off)
      buf = to_mv(ptr, sz).cast("I")
      # print("comp rcv", [x for x in buf])
      finished = self.execute_buf(buf, sz // 4)
      if not finished: 
        # print("early exi")
        return
      # print("gpfifio saveexi")
      self.off = 0
      self.ctrl.GPGet = (self.ctrl.GPGet + 1) % self.entries_cnt

  def execute_cmd(self, cmd) -> SchedResult:
    # print("execute_cmd", cmd, nv_gpu.NVC56F_SEM_EXECUTE, nv_gpu.NVC6C0_LAUNCH_DMA, nv_gpu.NVC6C0_LOAD_INLINE_DATA)
    if cmd == nv_gpu.NVC56F_SEM_EXECUTE: return self._exec_signal()
    elif cmd == nv_gpu.NVC6C0_LAUNCH_DMA: return self._exec_nvc6c0_dma()
    elif cmd == nv_gpu.NVC6B5_LAUNCH_DMA: return self._exec_nvc6b5_dma()
    elif cmd == 0x0320: return self._exec_load_inline_qmd() # NVC6C0_LOAD_INLINE_QMD_DATA
    else: 
      self.state[cmd] = self._next_dword() # just state update
      # if cmd == nv_gpu.NVC56F_SEM_ADDR_LO: print("set lo sig", cmd, self.state[cmd])
      # elif cmd == nv_gpu.NVC56F_SEM_ADDR_HI: print("set hi sig", cmd, self.state[cmd])
    return SchedResult.CONT

  def execute_buf(self, buf, entries):
    self.buf = buf
    # print("init buf", self.off, entries)
    # assert self.off == 0
    
    # from hexdump import hexdump
    # hexdump(self.buf)

    while self.off < entries:
      init_off = self.off
      typ, size, subc, mthd = self._next_header()
      cmd_end_off = self.off + size
      # print("hrd", mthd, size, self.off)
      # assert mthd != 0 or (mthd == 0 and size == 0x1)

      while self.off < cmd_end_off:
        res = self.execute_cmd(mthd)
        if res == SchedResult.YIELD:
          # assert False
          # print("abort running, need to wait")
          self.off = init_off # just revert to the header
          return False
        mthd += 4
    return True

  def execute_qmd(self, qmd_addr):
    qmd = qmd_struct_t.from_address(qmd_addr)
    # print("exec", hex(qmd_addr))

    prg_addr = qmd.program_address_lower + (qmd.program_address_upper << 32)
    const0 = to_mv(qmd.constant_buffer_addr_lower_0 + (qmd.constant_buffer_addr_upper_0 << 32), 0x160).cast('I')
    args_cnt, vals_cnt = const0[0], const0[1]
    # print([x for x in const0])
    # print("cost0", hex(qmd.constant_buffer_addr_lower_0 + (qmd.constant_buffer_addr_upper_0 << 32)))
    # print(args_cnt, vals_cnt)
    args_addr = qmd.constant_buffer_addr_lower_0 + (qmd.constant_buffer_addr_upper_0 << 32) + 0x160
    args = to_mv(args_addr, args_cnt*8).cast('Q')
    vals = to_mv(args_addr + args_cnt*8, vals_cnt*4).cast('I')
    ocelot_args = [ctypes.cast(args[i], ctypes.c_void_p) for i in range(args_cnt)] + [ctypes.cast(vals[i], ctypes.c_void_p) for i in range(vals_cnt)]
    gx, gy, gz = qmd.cta_raster_width, qmd.cta_raster_height, qmd.cta_raster_depth
    lx, ly, lz = qmd.cta_thread_dimension0, qmd.cta_thread_dimension1, qmd.cta_thread_dimension2
    # print(hex(prg_addr), args_cnt + vals_cnt, ocelot_args)
    # print([x for x in to_mv(prg_addr, 0x100)])
    # prg_addr.append("0x0")
    assert vals_cnt == 0
    gpuocelot_lib.ptx_run(ctypes.cast(prg_addr, ctypes.c_char_p), args_cnt + vals_cnt, (ctypes.c_void_p * len(ocelot_args))(*ocelot_args), lx, ly, lz, gx, gy, gz, 0)

  def _exec_signal(self) -> SchedResult:
    signal = self._state64_le(nv_gpu.NVC56F_SEM_ADDR_LO)
    val = self._state64_le(nv_gpu.NVC56F_SEM_PAYLOAD_LO)
    flags = self._next_dword()
    typ = (flags >> 0) & 0b111
    # print(typ)
    if typ == 1:
      # print("signal sig", hex(signal), val)
      # print("exec signal")
      to_mv(signal, 8).cast('Q')[0] = val
    elif typ == 3:
      # print("signal wait", hex(signal),val)
      mval = to_mv(signal, 8).cast('Q')[0]
      # print("YELD", mval >= val)
      assert mval >= val
      return SchedResult.CONT if mval >= val else SchedResult.YIELD
    else: raise RuntimeError(f"Unsupported type={typ} in exec wait/signal")

  def _exec_load_inline_qmd(self):
    # print('_exec_load_inline_qmd')
    qmd_addr = self._state64(nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A) << 8
    assert qmd_addr != 0x0, f"invalid qmd address {qmd_addr}"
    qmd_data = [self._next_dword() for _ in range(0x40)]
    cdata = (ctypes.c_uint32 * len(qmd_data))(*qmd_data)
    # print("qmd", hex(qmd_addr))
    ctypes.memmove(qmd_addr, cdata, 0x40 * 4)
    self.execute_qmd(qmd_addr)

  def _exec_nvc6c0_dma(self):
    # print('_exec_nvc6c0_dma')
    addr = self._state64(nv_gpu.NVC6C0_OFFSET_OUT_UPPER)
    sz = self._state(nv_gpu.NVC6C0_LINE_LENGTH_IN)
    lanes = self._state(nv_gpu.NVC6C0_LINE_COUNT)
    assert lanes == 1, f"unsupported lanes > 1 in _exec_nvc6c0_dma: {lanes}"
    flags = self._next_dword()
    assert flags == 0x41, f"unsupported flags in _exec_nvc6c0_dma: {flags}"
    typ, dsize, subc, mthd = self._next_header()
    assert typ == 6 and mthd == nv_gpu.NVC6C0_LOAD_INLINE_DATA, f"Expected inline data not found after nvc6c0_dma, {typ=} {mthd=}"
    copy_data = [self._next_dword() for _ in range(dsize)]
    assert len(copy_data) * 4 == sz, f"different copy sizes in _exec_nvc6c0_dma: {len(copy_data) * 4} != {sz}"
    # print("inline", hex(addr), copy_data)
    cdata = (ctypes.c_uint32 * len(copy_data))(*copy_data)
    # print(cdata[0])
    ctypes.memmove(addr, cdata, sz)
    # print([x for x in to_mv(addr, 0x160).cast('I')])
    # print(to_mv(addr, 8).cast('I')[0])

  def _exec_nvc6b5_dma(self):
    # print('_exec_nvc6b5_dma')
    src = self._state64(nv_gpu.NVC6B5_OFFSET_IN_UPPER)
    dst = self._state64(nv_gpu.NVC6B5_OFFSET_OUT_UPPER)
    sz = self._state(nv_gpu.NVC6B5_LINE_LENGTH_IN)
    flags = self._next_dword()
    assert flags == 0x182, f"unsupported flags in _exec_nvc6b5_dma: {flags}"
    ctypes.memmove(dst, src, sz)

class NVGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.mapped_ranges = set()
    self.queues = []

  def map_range(self, vaddr, size): self.mapped_ranges.add((vaddr, size))
  def unmap_range(self, vaddr, size): self.mapped_ranges.remove((vaddr, size))
  def add_gpfifo(self, base, entries_count):
    self.queues.append(GPFIFO(token:=len(self.queues), base, entries_count))
    return token
  def gpu_uuid(self, sz=16): return self.gpuid.to_bytes(sz, byteorder='big', signed=False)