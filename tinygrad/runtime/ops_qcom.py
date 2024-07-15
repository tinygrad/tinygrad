import os
import os, ctypes, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess, time, array
from types import SimpleNamespace
from typing import Tuple, List, Any
import time

from tinygrad.device import HCQCompatCompiled, HCQCompatAllocator, HCQCompatAllocRes, Compiler, CompileError, BufferOptions
import tinygrad.runtime.autogen.kgsl as kgsl
import tinygrad.runtime.autogen.adreno as adreno
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice
from tinygrad.helpers import getenv, from_mv, mv_address, init_c_struct_t, to_mv, round_up, to_char_p_p, DEBUG, prod, PROFILE
import tinygrad.runtime.autogen.libc as libc

# if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl # noqa: F401

def data64_le(data): return (data & 0xFFFFFFFF, (data >> 32) & 0xFFFFFFFF)

def parity(val: int): return (~0x6996 >> ((val ^ (val >> 16) ^ (val >> 8) ^ (val >> 4)) & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int):
  return adreno.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int):
  return adreno.CP_TYPE4_PKT | cnt & 0x3FFF | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

class QcomCompiler(CLCompiler):
  def __init__(self, device:str=""): super().__init__(CLDevice(device), 'compile_qcom')

MAP_FIXED = 0x10
class QcomDevice:
  signals_page:Any = None
  signals_pool: List[Any] = []

  def __init__(self, device:str=""):
    self.fd = os.open('/dev/kgsl-3d0', os.O_RDWR)
    QcomDevice.signals_page = self._gpu_alloc(16 * 65536, flags=0xC0A00, map_to_cpu=True, uncached=True)
    QcomDevice.signals_pool = [to_mv(self.signals_page.va_addr + off, 16).cast("Q") for off in range(0, self.signals_page.size, 16)]
    cr = kgsl.struct_kgsl_drawctxt_create(flags=(2<<20) | 0x10 | 0x2)
    self._ioctl(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, cr)
    self.context_id = cr.drawctxt_id

  def ctx(self):
    cr = kgsl.struct_kgsl_drawctxt_create(flags=(2<<20) | 0x10 | 0x2)
    self._ioctl(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, cr)
    self.context_id = cr.drawctxt_id
    return self.context_id

  def _ioctl(self, nr, arg):
    ret = fcntl.ioctl(self.fd, (3 << 30) | (ctypes.sizeof(arg) & 0x1FFF) << 16 | 0x9 << 8 | (nr & 0xFF), arg)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return ret

  def _gpu_alloc(self, size:int, flags:int=0, map_to_cpu=False, uncached=False):
    size = round_up(size, align:=(2 << 20))
    flags |= ((align << kgsl.KGSL_MEMALIGN_SHIFT) & kgsl.KGSL_MEMALIGN_MASK)
    if uncached: flags |= ((kgsl.KGSL_CACHEMODE_UNCACHED << kgsl.KGSL_CACHEMODE_SHIFT) & kgsl.KGSL_CACHEMODE_MASK)

    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=size, flags=flags)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    info = kgsl.struct_kgsl_gpuobj_info(id=alloc.id)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_INFO, info)

    if map_to_cpu: libc.mmap(info.gpuaddr, info.va_len, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, self.fd, info.id * 0x1000)

    return SimpleNamespace(va_addr=info.gpuaddr, size=info.va_len, info=info)

  def _gpu_free(self, opaque):
    free = kgsl.struct_kgsl_gpuobj_free(id=opaque.info.id)
    self._ioctl(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free)

  @classmethod
  def _read_signal(self, sig): return sig[0]

  @classmethod
  def _read_timestamp(self, sig): return sig[1]

  @classmethod
  def _set_signal(self, sig, value): sig[0] = value

  @classmethod
  def _alloc_signal(self, value=0, **kwargs) -> memoryview:
    self._set_signal(sig := self.signals_pool.pop(), value)
    return sig

  @classmethod
  def _free_signal(self, sig): self.signals_pool.append(sig)

  @classmethod
  def _wait_signal(self, signal, value=0, timeout=10000):
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if signal[0] >= value: return
    raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

  def _gpu2cpu_time(self, gpu_time, is_copy): return self.cpu_start_time + (gpu_time - self.gpu_start_time) / 1e3

  def synchronize(self):
    self._wait_signal(self.timeline_signal, self.timeline_value - 1)

    if self.timeline_value > (1 << 63): self._wrap_timeline_signal()
    if PROFILE: self._prof_process_events()

class QcomAllocator(HCQCompatAllocator):
  def __init__(self, device:QcomDevice): super().__init__(device)

  def _alloc(self, size:int, options:BufferOptions) -> HCQCompatAllocRes:
    # TODO(vpachkov): host?
    return self.device._gpu_alloc(size, map_to_cpu=options.cpu_access)
  
  def _free(self, opaque, options:BufferOptions):
    # TODO(vpachkov): host?
    return self.device._gpu_free(opaque)

class HWCommandQueue():
  def __init__(self): self.q = []

  def push(self, opcode=None, reg=None, vals = []):
    if opcode: self.q += [pkt7_hdr(opcode, len(vals)), *vals]
    if reg: self.q += [pkt4_hdr(reg, len(vals)), *vals]

  def signal(self, signal, value=0):
    self.push(opcode=adreno.CP_EVENT_WRITE7, vals=[adreno.CACHE_FLUSH_TS | (adreno.EV_WRITE_USER_64B << adreno.CP_EVENT_WRITE7_0_WRITE_SRC__SHIFT), *data64_le(mv_address(signal)), *data64_le(value)])
    self.push(opcode=adreno.CP_EVENT_WRITE7, vals=[adreno.CACHE_INVALIDATE])

  def timestamp(self, signal):
    self.push(opcode=adreno.CP_REG_TO_MEM, vals=[0x40000980,  *data64_le(mv_address(signal))])
    self.push(opcode=adreno.CP_EVENT_WRITE7, vals=[adreno.CACHE_INVALIDATE])

  def submit(self, device: QcomDevice):
    # TODO(vpachkov): split objs based on cmd stream size
    obj = kgsl.struct_kgsl_command_object()
    cmdbytes = array.array('I', self.q)
    alloc = device._gpu_alloc(len(cmdbytes) * 4, 0xC0A00, map_to_cpu=True)
    ctypes.memmove(alloc.va_addr, mv_address(memoryview(cmdbytes)), len(cmdbytes) * 4)

    obj.gpuaddr = alloc.va_addr
    obj.size = len(cmdbytes) * 4
    obj.flags = 0x00000001

    submit_req = kgsl.struct_kgsl_gpu_command()
    submit_req.flags = 0x0
    submit_req.cmdlist = ctypes.addressof(obj)
    submit_req.cmdsize = ctypes.sizeof(kgsl.struct_kgsl_command_object)
    submit_req.numcmds = 1
    #TODO: not recreating context sometimes results to EDEADLK
    submit_req.context_id = device.ctx()

    device._ioctl(0x4A, submit_req)
    self.q = []

if __name__ == '__main__':
  device = QcomDevice()
  alloc = device._gpu_alloc(0x1000, map_to_cpu=True)
  ptr = to_mv(alloc.va_addr, alloc.size).cast("I")
  ptr[0] = 1
  print(ptr[0])
  device._gpu_free(alloc)

  sig = device._alloc_signal()
  queue = HWCommandQueue()

  lib = QcomCompiler('').compile('''
__kernel void E_72_9_8_16_4(__global float* data0) {
  int gidx1 = get_group_id(0); /* 9 */
  int lidx2 = get_local_id(0); /* 8 */
  int gidx0 = get_group_id(1); /* 72 */
  int lidx3 = get_local_id(1); /* 16 */
  int alu0 = (gidx1*64);
  int alu1 = (lidx2*576);
  int alu2 = (lidx3*4);
  int alu3 = (((gidx0*569)%577)+alu0+(alu1%577)+alu2);
  float alu4 = (((alu3%577)<1)?1.0f:0.0f);
  float alu5 = ((((alu3+1)%577)<1)?1.0f:0.0f);
  float alu6 = ((((alu3+2)%577)<1)?1.0f:0.0f);
  float alu7 = ((((alu3+3)%577)<1)?1.0f:0.0f);
  *((__global float4*)(data0+(gidx0*4608)+alu0+alu1+alu2)) = (float4)((alu4+alu4),(alu5+alu5),(alu6+alu6),(alu7+alu7));
}
''')
  
  print(lib)

  for i in range(1, 10):
    queue.signal(sig, i)
    queue.submit(device)
    device._wait_signal(sig, i)
    print(sig[0])
