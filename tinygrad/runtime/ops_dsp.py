from __future__ import annotations
from typing import Tuple, List
import ctypes, fcntl, os, mmap
from tinygrad.device import BufferOptions, LRUAllocator, Compiled
from tinygrad.runtime.autogen import libc
from tinygrad.helpers import from_mv, getenv
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.dtype import DType
from tinygrad.codegen.uops import UOp
import extra.dsp.ion as ion
import extra.dsp.msm_ion as msm_ion
import extra.dsp.adsprpc as adsprpc
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))

class DSPProgram:
  def __init__(self, name:str, lib:bytes):
    with open("/tmp/swag.so", "wb") as f: f.write(lib)
    self.handle = ctypes.c_int64(-1)
    adsp.remote_handle64_open(ctypes.create_string_buffer(b"file:////tmp/swag.so?entry&_modver=1.0&_dom=cdsp"), ctypes.byref(self.handle))

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    assert len(vals) == 0
    assert len(bufs) < 16
    pra = (adsprpc.union_remote_arg64 * len(bufs))()
    for i,b in enumerate(bufs):
      pra[i].dma.fd = b[2].fd
      pra[i].dma.len = b[1]
    ret = adsp.remote_handle64_invoke(self.handle, (1<<24) | (len(bufs)<<4), pra)
    assert ret == 0, f"invoke returned {ret}"

ION_IOC_ALLOC = 0
ION_IOC_SHARE = 4

def ion_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord(ion.ION_IOC_MAGIC) & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class DSPAllocator(LRUAllocator):
  def __init__(self):
    self.ion_fd = os.open("/dev/ion", os.O_RDWR | os.O_CLOEXEC)
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    arg3 = ion.struct_ion_allocation_data(len=size, align=0x10, heap_id_mask=1<<msm_ion.ION_SYSTEM_HEAP_ID, flags=ion.ION_FLAG_CACHED)
    ion_iowr(self.ion_fd, ION_IOC_ALLOC, arg3)
    arg2 = ion.struct_ion_fd_data(handle=arg3.handle)
    ion_iowr(self.ion_fd, ION_IOC_SHARE, arg2)
    res = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, arg2.fd, 0)
    return (res, size, arg2)

  def copyin(self, dest, src:memoryview): ctypes.memmove(dest[0], from_mv(src), dest[1])
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src[0], src[1])

class DSPRenderer(CStyleLanguage):
  device = "DSP"
  has_local = False

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    ret = super().render_kernel(function_name, kernel, bufs, uops, prefix)
    msrc = ['typedef struct { int fd; unsigned int offset; } remote_dma_handle;',
            'typedef struct { void *pv; unsigned int len; } remote_buf;',
            'typedef union { remote_buf buf; remote_dma_handle dma; } remote_arg;',
            'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);',
            'int HAP_munmap(void *addr, int len);',
            'int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'if (sc>>24 == 1) {']

    for i,b in enumerate(bufs):
      msrc.append(f'  void *buf_{i} = HAP_mmap(0, 12, 3, 0, pra[{i}].dma.fd, 0);')
    msrc.append(f"  {function_name}({', '.join([f'buf_{i}' for i in range(len(bufs))])});")
    for i,b in enumerate(bufs):
      msrc.append(f'  HAP_munmap(buf_{i}, 12);')
    msrc.append("}")
    msrc.append("return 0;")
    msrc.append("}")

    return ret + '\n' + '\n'.join(msrc)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    compiler = ClangCompiler("compile_dsp", args=["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib"])
    super().__init__(device, DSPAllocator(), DSPRenderer(), compiler, DSPProgram)
