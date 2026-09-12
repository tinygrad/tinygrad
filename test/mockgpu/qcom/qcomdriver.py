import ctypes, functools, mmap, os, traceback
from tinygrad.runtime.autogen import kgsl, libc
from tinygrad.runtime.support import hcq2
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc, TextFileDesc, DirFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU

A630_CHIP_ID = 0x06030001

def _ioctl_nr(ioctl:functools.partial) -> int: return ioctl.args[2]

# QCOM's HCQ submit path calls libc.ioctl from generated host code instead of
# FileIOInterface.ioctl. Keep one process-wide bridge so multiple virtual QCOM
# devices don't accidentally chain callbacks through each other.
_real_ioctl = libc.dll.ioctl
_qcom_drivers:list['QCOMDriver'] = []
def _dispatch_ioctl(fd, request, argp):
  for d in _qcom_drivers:
    if fd not in d.fds: continue
    if os.getenv("QCOM_TRACE"): print(f"mock kgsl ioctl fd={fd} req={request:#x} argp={int(argp or 0):#x}", flush=True)
    try: return int(d.ioctl(request, int(argp or 0)) or 0)
    except BaseException as e: # never unwind a Python exception through ctypes callback machinery
      d.last_callback_error = e
      if os.getenv("QCOM_TRACE"):
        print(f"mock kgsl ioctl failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
      return -1
  return int(_real_ioctl(fd, request, argp))
_mock_ioctl = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)(_dispatch_ioctl)

# cfunc_buf can already be cached when a generated CPU HCQ submit is built.
# Rewrite the cached function pointer too, instead of relying only on the DLL
# attribute replacement. This keeps the normal CPU HCQ path viable for MockQCOM.
_orig_cfunc_buf = hcq2.cfunc_buf
@functools.cache
def _cfunc_buf(lib:str, name:str):
  b = _orig_cfunc_buf(lib, name)
  if lib == "libc" and name == "ioctl": b.host.view(fmt='Q')[0] = ctypes.cast(_mock_ioctl, ctypes.c_void_p).value
  return b

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd:int, driver:'QCOMDriver'):
    super().__init__(fd)
    self.driver = driver
    driver.fds.add(fd)

  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def close(self, fd):
    self.driver.fds.discard(fd)
    return 0

  def mmap(self, start, sz, prot, flags, fd, offset):
    obj_id = offset // 0x1000
    if obj_id not in self.driver.objects: raise RuntimeError(f"mmap for unknown KGSL object {obj_id}")
    addr = libc.mmap(start, sz, prot, (flags & ~mmap.MAP_SHARED) | mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if not addr or addr == ctypes.c_void_p(-1).value: raise OSError(ctypes.get_errno(), "mock KGSL mmap failed")
    self.driver.objects[obj_id]["addr"] = int(addr)
    self.driver.gpu.map_range(int(addr), sz)
    return int(addr)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    # The generated QCOM HCQ submit calls libc.ioctl directly, which the Python
    # HCQ runtime cannot service (it hangs on the QCOM synchronization op), so
    # MockQCOM must run on the CPU HCQ runtime.  An explicit env var still wins.
    if "HCQ_RUNTIME_DEV" not in os.environ: hcq2.HCQ_RUNTIME_DEV.value = "CPU"
    self.tracked_files = [
      VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self)),
      VirtFile('/sys/class/kgsl', functools.partial(DirFileDesc, child_names=['kgsl-3d0'])),
      VirtFile('/sys/class/kgsl/kgsl-3d0', functools.partial(DirFileDesc, child_names=['idle_timer'])),
      VirtFile('/sys/class/kgsl/kgsl-3d0/idle_timer', functools.partial(TextFileDesc, text='10\n')),
    ]
    self.next_fd, self.next_ctx, self.next_obj, self.timestamp = 1 << 30, 1, 1, 0
    self.objects:dict[int, dict[str, int]] = {}
    self.user_maps:dict[int, list[int]] = {}
    self.fds:set[int] = set()
    self.gpu = QCOMGPU(0)
    self.last_callback_error:BaseException|None = None

    if self not in _qcom_drivers: _qcom_drivers.append(self)
    if hcq2.cfunc_buf is not _cfunc_buf: hcq2.cfunc_buf = _cfunc_buf

  def _alloc_fd(self):
    fd, self.next_fd = self.next_fd, self.next_fd + 1
    return fd

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def ioctl(self, request:int, argp:int):
    nr = request & 0xff
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      st_ctx = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      st_ctx.drawctxt_id, self.next_ctx = self.next_ctx, self.next_ctx + 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      st_prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if st_prop.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"unsupported KGSL property {st_prop.type}")
      info = kgsl.struct_kgsl_devinfo.from_address(int(st_prop.value))
      info.device_id, info.chip_id, info.mmu_enabled = 0, A630_CHIP_ID, 1
      info.gpu_id, info.gmem_sizebytes = 630, 1024 * 1024
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      st_alloc = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      st_alloc.id, self.next_obj = self.next_obj, self.next_obj + 1
      st_alloc.mmapsize = max(st_alloc.mmapsize, st_alloc.size)
      self.objects[st_alloc.id] = {"size": int(st_alloc.mmapsize), "flags": int(st_alloc.flags), "addr": 0}
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      freed = self.objects.pop(kgsl.struct_kgsl_gpuobj_free.from_address(argp).id, None)
      if freed is not None and freed["addr"]: self.gpu.unmap_range(freed["addr"], freed["size"])
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      st_map = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      st_map.gpuaddr = st_map.hostptr
      self.gpu.map_range(int(st_map.hostptr), int(st_map.len))
      self.user_maps.setdefault(int(st_map.gpuaddr), []).append(int(st_map.len))
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE):
      st_free = kgsl.struct_kgsl_sharedmem_free.from_address(argp)
      if sizes:=self.user_maps.get(addr:=int(st_free.gpuaddr)):
        self.gpu.unmap_range(addr, sizes.pop())
        if not sizes: self.user_maps.pop(addr)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      st_cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      if os.getenv("QCOM_TRACE"):
        print(f"KGSL_GPU_COMMAND cmdlist={int(st_cmd.cmdlist):#x} cmdsize={st_cmd.cmdsize} numcmds={st_cmd.numcmds} "
              f"context={st_cmd.context_id} objlist={int(st_cmd.objlist):#x} numobjs={st_cmd.numobjs}", flush=True)
      if os.getenv("QCOM_MOCK_NOEXEC"): return 0
      for i in range(st_cmd.numcmds):
        cmd = kgsl.struct_kgsl_command_object.from_address(st_cmd.cmdlist + i * st_cmd.cmdsize)
        if os.getenv("QCOM_TRACE"):
          print(f"  command[{i}] gpuaddr={int(cmd.gpuaddr):#x} offset={int(cmd.offset):#x} size={int(cmd.size):#x} flags={cmd.flags:#x}", flush=True)
        self.gpu.submit_ib(int(cmd.gpuaddr + cmd.offset), int(cmd.size))
      self.gpu.execute()
      self.timestamp += 1
      st_cmd.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
      st_ts = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp)
      st_ts.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID): pass
    else: raise NotImplementedError(f"unsupported KGSL ioctl {nr:#x}")
    return 0
