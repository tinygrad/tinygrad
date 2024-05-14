import ctypes, time
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
from extra.mockgpu.gpu import VirtGPU
from tinygrad.helpers import to_mv, init_c_struct_t

class GPFIFO:
  def __init__(self, token, base, entries_cnt):
    self.token, self.base, self.entries_cnt = token, base, entries_cnt
    self.gpfifo = to_mv(self.base, self.entries_cnt * 8).cast("Q")
    self.ctrl = nv_gpu.AmpereAControlGPFifo.from_address(self.base + self.entries_cnt * 8)
    self.off = 0

  def _skip(self, cnt): self.off += cnt
  def _next_dword(self, buf):
    x = buf[self.off]
    self.off += 1
    return x

  def _next_qword(self, buf):
    x = buf[self.off] << 32 + buf[self.off + 1]
    self.off += 2
    return x

  def _next_qword_le(self, buf):
    x = buf[self.off] + buf[self.off + 1] << 32
    self.off += 2
    return x
  
  def execute(self):
    while self.ctrl.GPGet < self.ctrl.GPPut:
      entry = self.gpfifo[self.ctrl.GPGet % self.entries_cnt]
      ptr = ((entry >> 2) & 0x7ffffffff) << 2
      sz = ((entry >> 42) & 0x1fffff) << 2
      print(hex(ptr), sz // 4, entry, self.ctrl.GPGet, self.ctrl.GPPut)
      finished = self.execute_buf(to_mv(ptr, sz).cast("I"), sz // 4)
      if not finished: return
      self.off = 0
      self.ctrl.GPGet += 1

  def execute_buf(self, buf, entries):
    while self.off < entries:
      cont = True
      header = self._next_dword(buf)
      typ = (header >> 28) & 0xF
      size = (header >> 16) & 0xFFF
      subc = (header >> 13) & 0x7
      mthd = (header & 0x1FFF) << 2

      print("exec", mthd, size, nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A)

      if mthd == nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI: self._skip(size)
      elif mthd == nv_gpu.NVC6C0_SET_OBJECT: self._skip(size)
      elif mthd == nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A: self._skip(size)
      elif mthd == nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A: self._skip(size)
      elif mthd == nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A: self._skip(size)
      elif mthd == nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A: self._skip(size)
      elif mthd == nv_gpu.NVC56F_SEM_ADDR_LO: cont = self._exec_signal(buf, size)
      elif mthd == nv_gpu.NVC56F_NON_STALL_INTERRUPT: self._skip(size)
      elif mthd == nv_gpu.NVC56F_NON_STALL_INTERRUPT: self._skip(size)
      else: raise RuntimeError(f"Unknown cmd in buffer {mthd}")
      if not cont: return False
    return True

  def _exec_signal(self, buf, size):
    assert size == 5
    signal = self._next_qword_le(buf)
    val = self._next_qword_le(buf)
    flags = self._next_dword(buf)
    typ = (flags >> 0) & 0b111
    if typ == 1: to_mv(signal, 8).cast('Q')[0] = val
    elif typ == 3:
      mval = to_mv(signal, 8).cast('Q')[0]
      if mval >= val: return True
      self.off -= 6 # revert, not pass signal for now
      return False
    else: raise RuntimeError(f"Unsupported type={typ} in exec wait/signal")

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