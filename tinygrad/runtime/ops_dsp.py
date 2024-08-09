from __future__ import annotations
from typing import Tuple, List
import ctypes, fcntl, os, mmap, tempfile
from tinygrad.device import BufferOptions, LRUAllocator, Compiled
from tinygrad.helpers import from_mv, getenv, DEBUG
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.dtype import DType
from tinygrad.codegen.uops import UOp

from tinygrad.runtime.autogen import libc, ion, msm_ion, adsprpc
ION_IOC_ALLOC = 0
ION_IOC_FREE = 1
ION_IOC_SHARE = 4

if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))

class DSPProgram:
  def __init__(self, name:str, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      output_file.write(lib)
      output_file.flush()
      self.handle = ctypes.c_int64(-1)
      fp = f"file:///{output_file.name}?entry&_modver=1.0&_dom=cdsp"
      adsp.remote_handle64_open(ctypes.create_string_buffer(fp.encode()), ctypes.byref(self.handle))
      assert self.handle.value != -1, "load failed"
      if DEBUG >= 6: os.system(f"llvm-objdump -d {output_file.name}")

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    assert len(vals) == 0
    assert len(bufs) < 16
    pra = (adsprpc.union_remote_arg64 * (2+len(bufs)))()
    test = (ctypes.c_int32 * len(bufs))()
    time_est = ctypes.c_uint64(0)
    pra[0].buf.pv = ctypes.addressof(test)
    pra[0].buf.len = 4 * len(bufs)
    pra[1].buf.pv = ctypes.addressof(time_est)
    pra[1].buf.len = 8
    for i,b in enumerate(bufs):
      test[i] = b[1]
      pra[i+2].dma.fd = b[2].fd
      pra[i+2].dma.len = b[1]
    ret = adsp.remote_handle64_invoke(self.handle, (1<<24) | (1<<16) | (1<<8) | (len(bufs)<<4), pra)
    assert ret == 0, f"invoke returned {ret}"
    return time_est.value / 1e6

def ion_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord(ion.ION_IOC_MAGIC) & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class DSPAllocator(LRUAllocator):
  def __init__(self):
    self.ion_fd = os.open("/dev/ion", os.O_RDWR | os.O_CLOEXEC)
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    arg3 = ion.struct_ion_allocation_data(len=size, align=0x200, heap_id_mask=1<<msm_ion.ION_SYSTEM_HEAP_ID, flags=ion.ION_FLAG_CACHED)
    ion_iowr(self.ion_fd, ION_IOC_ALLOC, arg3)
    ion_iowr(self.ion_fd, ION_IOC_SHARE, arg2:=ion.struct_ion_fd_data(handle=arg3.handle))
    res = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, arg2.fd, 0)
    return (res, size, arg2)

  def _free(self, opaque, options:BufferOptions):
    libc.munmap(opaque[0], opaque[1])
    os.close(opaque[2].fd)
    ion_iowr(self.ion_fd, ION_IOC_FREE, ion.struct_ion_handle_data(handle=opaque[2].handle))

  def copyin(self, dest, src:memoryview): ctypes.memmove(dest[0], from_mv(src), dest[1])
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src[0], src[1])

class DSPRenderer(CStyleLanguage):
  device = "DSP"
  supports_float4 = False
  has_local = False
  buffer_suffix = " restrict __attribute__((align_value(128)))"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    ret = super().render_kernel(function_name, kernel, bufs, uops, prefix)
    prefix = ['#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })', 'typedef int bool;']
    msrc = ['typedef struct { int fd; unsigned int offset; } remote_dma_handle;',
            'typedef struct { void *pv; unsigned int len; } remote_buf;',
            'unsigned long long HAP_perf_get_time_us(void);'
            'typedef union { remote_buf buf; remote_dma_handle dma; } remote_arg;',
            'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);',
            'int HAP_munmap(void *addr, int len);',
            'int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'if (sc>>24 == 1) {']
    msrc += [f'  void *buf_{i} = HAP_mmap(0, ((int*)pra[0].buf.pv)[{i}], 3, 0, pra[{i+2}].dma.fd, 0);' for i in range(len(bufs))]
    msrc.append("  unsigned long long start = HAP_perf_get_time_us();")
    msrc.append(f"  {function_name}({', '.join([f'buf_{i}' for i in range(len(bufs))])});")
    msrc.append("  *(unsigned long long *)(pra[1].buf.pv) = HAP_perf_get_time_us() - start;")
    msrc += [f'  HAP_munmap(buf_{i}, ((int*)pra[0].buf.pv)[{i}]);' for i in range(len(bufs))]
    msrc.append("}")
    msrc.append("return 0;")
    msrc.append("}")
    return '\n'.join(prefix) + '\n' + ret + '\n' + '\n'.join(msrc)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    compiler = ClangCompiler("compile_dsp", args=["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib"])
    super().__init__(device, DSPAllocator(), DSPRenderer(), compiler, DSPProgram)
