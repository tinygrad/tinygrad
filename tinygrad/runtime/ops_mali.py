from __future__ import annotations
import base64, ctypes, os, struct
from tinygrad.device import Compiled, BufferStorage, BufferSpec, HostAllocator, Program
from tinygrad.dtype import AddrSpace, DType
from tinygrad.helpers import from_mv, suppress_finalizing, is_image_shape, unwrap
from tinygrad.renderer import TinyELF
from tinygrad.renderer.nir import MALIRenderer
from tinygrad.runtime.autogen import tinymali as mali


def check(value, msg="MALI"):
  if value is None or (isinstance(value, int) and value == 0): raise RuntimeError(f"{msg} failed")
  return value

def check_ret(ret:int, msg="MALI"):
  if ret != 0: raise RuntimeError(f"{msg}: {os.strerror(-ret) if ret < 0 else ret}")

class MALIAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host or options.cpu_access: return super()._alloc(size, options)
    return BufferStorage(check(mali.tm_buffer_create(self.dev.ctx, size), "MALI buffer allocation"))

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None: mali.tm_buffer_destroy(self.dev.ctx, storage.buf)

  def _copyin(self, dest:int, src:memoryview):
    check_ret(mali.tm_buffer_write(self.dev.ctx, dest, 0, from_mv(src), src.nbytes), "MALI copyin")

  def _copyout(self, dest:memoryview, src:int):
    check_ret(mali.tm_buffer_read(self.dev.ctx, src, 0, from_mv(dest), dest.nbytes), "MALI copyout")

  def _offset(self, buf:int, size:int, offset:int) -> int:
    return check(mali.tm_buffer_offset(self.dev.ctx, buf, offset), "MALI buffer view")

class MALIProgram(Program['MALIDevice']):
  def __init__(self, dev:MALIDevice, obj:TinyELF, links:dict[str,MALIProgram]):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    src = base64.b64decode(obj.lib)
    self.kernel = check(mali.tm_compile(dev.ctx, src, len(src)), f"MALI compile {obj.name}")
    self.args: list[tuple[int, DType, bool]] = []
    cursor = 0
    for i, (off, dt) in enumerate(TinyELF.iter_sig(obj.signature)):
      self.args.append((off, dt, obj.signature[i][1] is not AddrSpace.ALU))
      cursor = off + dt.itemsize
    self.arg_size = cursor
    self.offsets = [off for off, dt, is_buffer in self.args if is_buffer]
    self.image_info = [v for _, _, dt, shape in obj.signature
                       for v in ((shape[1], shape[0], dt.itemsize, 1) if is_image_shape(shape) else (0, 0, 0, 0))]

  @suppress_finalizing
  def __del__(self):
    if hasattr(self, "kernel"): mali.tm_kernel_destroy(self.dev.ctx, self.kernel)

  def __call__(self, *raw_args, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    args = ctypes.create_string_buffer(self.arg_size)
    buffers = (ctypes.c_void_p * len(self.offsets))()
    scalar_args: list[int] = []
    for value, (off, dt, is_buffer) in zip(raw_args, self.args):
      if is_buffer:
        buffers[len(scalar_args)] = value
        scalar_args.append(off)
      else: struct.pack_into(unwrap(dt.fmt), args, off, value)
    offsets = (ctypes.c_uint32 * len(scalar_args))(*scalar_args)
    image_info = (ctypes.c_uint32 * len(self.image_info))(*self.image_info)
    grid = (ctypes.c_uint32 * 3)(*global_size)
    block = (ctypes.c_uint32 * 3)(*local_size)
    check_ret(mali.tm_launch(self.dev.ctx, self.kernel, buffers, offsets, image_info, len(buffers), args, self.arg_size, grid, block, 0), self.name)
    if wait: self.dev.synchronize()

class MALIDevice(Compiled):
  renderers = [MALIRenderer]

  def __init__(self, device:str=""):
    self.ctx = check(mali.tm_create(b"/dev/dri/renderD128"), "MALI panthor device")
    super().__init__(device, MALIAllocator(self), MALIProgram, arch="g610")

  @property
  def has_copy_queue(self) -> bool: return False

  def synchronize(self, timeout:int|None=None):
    check_ret(mali.tm_synchronize(self.ctx), "MALI synchronize")

  def finalize(self):
    super().finalize()
    if self.ctx is not None:
      mali.tm_destroy(self.ctx)
      self.ctx = None
