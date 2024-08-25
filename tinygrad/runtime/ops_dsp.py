from __future__ import annotations
from typing import Tuple, List
import ctypes, os, mmap, tempfile, functools, array
from tinygrad.device import BufferOptions, Compiled, Allocator
from tinygrad.helpers import from_mv, getenv, DEBUG, mv_address, to_mv
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.renderer.cstyle import DSPRenderer
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))

class DSPProgram:
  def __init__(self, device:DSPDevice, name:str, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=False) as output_file:
      output_file.write(lib)
      output_file.flush()
    if DEBUG >= 6: os.system(f"llvm-objdump -mhvx -d {output_file.name}")
    self.device, self.lib, self.unique_filepath = device, lib, output_file.name

  def __del__(self):
    self.device._unregister_lib(self.unique_filepath)
    os.remove(self.unique_filepath)

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    if len(bufs) >= 16: raise RuntimeError(f"Too many buffers to execute with: {len(bufs)}")

    pra = (qcom_dsp.union_remote_arg64 * (2 + len(bufs)))()

    var_vals_mv = memoryview(bytearray((len(bufs) + len(vals)) * 4))
    var_vals_mv.cast('i')[:] = array.array('i', tuple(b.size for b in bufs) + vals)

    timer = memoryview(bytearray(8)).cast('Q')

    pra[0].buf.pv, pra[0].buf.len = mv_address(var_vals_mv), var_vals_mv.nbytes
    pra[1].buf.pv, pra[1].buf.len = mv_address(timer), timer.nbytes
    for i,b in enumerate(bufs): pra[i+2].dma.fd, pra[i+2].dma.len = b.share_info.fd, b.size

    handle = self.device._ensure_lib_opened(self.unique_filepath)
    if (ret:=adsp.remote_handle64_invoke(handle, (2<<24) | (1<<16) | (1<<8) | len(bufs), pra)) != 0:
      raise RuntimeError(f"Failed to execute on DSP: {ret}")
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info:Any): self.va_addr, self.size, self.share_info = va_addr, size, share_info

class DSPAllocator(Allocator):
  def __init__(self, device:DSPDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    b = qcom_dsp.ION_IOC_ALLOC(self.device.ion_fd, len=size, align=0x200, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED)
    share_info = qcom_dsp.ION_IOC_SHARE(self.device.ion_fd, handle=b.handle)
    va_addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, share_info.fd, 0)
    return DSPBuffer(va_addr, size, share_info)

  def _free(self, opaque:DSPBuffer, options:BufferOptions):
    libc.munmap(opaque.va_addr, opaque.size)
    os.close(opaque.share_info.fd)
    qcom_dsp.ION_IOC_FREE(self.device.ion_fd, handle=opaque.share_info.handle)

  def as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)
  def copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
    self.loaded_libs, self.lru_lib_order = {}, []

    compiler_args = ["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib", "-mhvx=v65", "-mhvx-length=128B"]
    super().__init__(device, DSPAllocator(self), DSPRenderer(), ClangCompiler("compile_dsp", args=compiler_args), functools.partial(DSPProgram, self))

  def _close_lib(self, libname):
    if (ret:=adsp.remote_handle64_close(self.loaded_libs[libname])) != 0: raise RuntimeError(f"Failed to close library {libname}: {ret}")
    self.loaded_libs.pop(libname)
    self.lru_lib_order.remove(libname)

  def _unregister_lib(self, libname):
    if (h:=self.loaded_libs.get(libname)) is not None: self._close_lib(libname)

  def _ensure_lib_opened(self, libname):
    if libname not in self.loaded_libs:
      if len(self.lru_lib_order) > 100: self._close_lib(self.lru_lib_order[0])
      bname = ctypes.create_string_buffer(f"file:///{libname}?entry&_modver=1.0&_dom=cdsp".encode())
      if (r:=adsp.remote_handle64_open(bname, ctypes.byref(h:=ctypes.c_int64()))) != 0: raise RuntimeError(f"Failed to load library: {libname}, {r}")
      self.loaded_libs[libname] = h
    else: self.lru_lib_order.remove(libname)

    self.lru_lib_order.append(libname)
    return self.loaded_libs[libname]
