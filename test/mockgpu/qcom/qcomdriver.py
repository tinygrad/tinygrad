import ctypes, functools, mmap, os, struct
from tinygrad.runtime.autogen import kgsl, libc, mesa
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import A6XXEmulator

def _ioctl_nr(ioctl:functools.partial) -> int: return ioctl.args[2]

kgsl_ioctl_info = {
  _ioctl_nr(ioctl): (name, ioctl.args[3]) for name, ioctl in vars(kgsl).items()
  if name.startswith("IOCTL_KGSL_") and isinstance(ioctl, functools.partial)
}

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.mmap(start, sz, prot, flags, offset)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files = [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))]
    self.next_fd, self.next_context, self.next_object, self.timestamp = 1 << 30, 1, 1, 0
    self.objects:dict[int, tuple[int, int]] = {}
    self.gpu = A6XXEmulator()
    # QCOM submits through an ioctl in the HCQ host program instead of a queue doorbell.
    cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
    def _ioctl_bridge(fd, request, argp):
      if os.getenv("QCOM_TRACE"): print(f"ioctl fd={fd} request={request:#x} argp={argp:#x}")
      return self.ioctl(request, argp)
    self._ioctl_cb = cb_type(_ioctl_bridge)
    self._ioctl_cb.__name__, self._ioctl_cb.__module__ = "ioctl", "libc"
    libc.dll.ioctl = self._ioctl_cb

  def open(self, name, flags, mode, virtfile):
    fd, self.next_fd = self.next_fd, self.next_fd + 1
    return virtfile.fdcls(fd)

  def mmap(self, start, sz, prot, flags, offset):
    obj_id = offset // 0x1000
    if obj_id not in self.objects: raise RuntimeError(f"mmap for unknown KGSL object {obj_id}")
    addr = libc.mmap(start, sz, prot, (flags & ~mmap.MAP_SHARED) | mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.objects[obj_id] = (addr, sz)
    self.gpu.map_range(addr, sz)
    return addr

  @staticmethod
  def _write32(addr:int, value:int): ctypes.c_uint32.from_address(addr).value = value

  def _execute_command_buffer(self, addr:int, size:int):
    words = struct.unpack_from(f"{size // 4}I", ctypes.string_at(addr, size))
    pos = 0
    while pos < len(words):
      hdr = words[pos]
      if hdr & 0xf0000000 == mesa.CP_TYPE7_PKT:
        count, opcode = hdr & 0x3fff, (hdr >> 16) & 0x7f
        vals, pos = words[pos+1:pos+1+count], pos+1+count
        if os.getenv("QCOM_TRACE"): print(f"CP7 {opcode:#x} {list(map(hex, vals))}")
        if opcode == mesa.CP_EVENT_WRITE and len(vals) >= 4:
          self._write32(vals[1] | vals[2] << 32, vals[3])
        elif opcode == mesa.CP_MEM_WRITE and len(vals) >= 3:
          self._write32(vals[0] | vals[1] << 32, vals[2])
        elif opcode == mesa.CP_REG_TO_MEM and len(vals) >= 3:
          self._write32(vals[1] | vals[2] << 32, 0)
        elif opcode == mesa.CP_LOAD_STATE6_FRAG and len(vals) >= 3:
          state_type, num_unit = (vals[0] >> 14) & 0x3, vals[0] >> 22
          addr = vals[1] | vals[2] << 32
          if state_type == mesa.ST_CONSTANTS: self.gpu.load_constants(addr, num_unit * 16)
          elif state_type == mesa.ST_SHADER:
            self.gpu.load_shader(addr, num_unit * 128)
            if int(os.getenv("QCOM_TRACE", "0")) > 1:
              from tinygrad.runtime.support.compiler_mesa import disas_adreno
              disas_adreno(self.gpu.shader)
        elif opcode == mesa.CP_EXEC_CS and len(vals) >= 4:
          self.gpu.exec_cs((vals[1], vals[2], vals[3]))
      elif hdr & 0xf0000000 == mesa.CP_TYPE4_PKT:
        if os.getenv("QCOM_TRACE"): print(f"CP4 {(hdr >> 8) & 0x3ffff:#x} {list(map(hex, words[pos+1:pos+1+(hdr & 0x7f)]))}")
        count, base = hdr & 0x7f, (hdr >> 8) & 0x3ffff
        self.gpu.write_regs(base, words[pos+1:pos+1+count])
        pos += 1 + count
      else: raise RuntimeError(f"invalid A6XX packet header {hdr:#x} at dword {pos}")

  def ioctl(self, request, argp):
    nr = request & 0xff
    if nr not in kgsl_ioctl_info: raise RuntimeError(f"unknown KGSL ioctl {nr:#x}")
    name, struct_type = kgsl_ioctl_info[nr]
    req = struct_type.from_address(argp)

    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      req.drawctxt_id, self.next_context = self.next_context, self.next_context + 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      if req.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"unsupported KGSL property {req.type}")
      info = kgsl.struct_kgsl_devinfo.from_address(req.value)
      info.device_id, info.chip_id, info.mmu_enabled = 0, 0x06030001, 1
      info.gpu_id, info.gmem_sizebytes = 0, 1024 * 1024
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      req.id, self.next_object = self.next_object, self.next_object + 1
      req.mmapsize = max(req.mmapsize, req.size)
      self.objects[req.id] = (0, req.mmapsize)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      self.objects.pop(req.id, None)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      req.gpuaddr = req.hostptr
      self.gpu.map_range(req.hostptr, req.len)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      for i in range(req.numcmds):
        obj = kgsl.struct_kgsl_command_object.from_address(req.cmdlist + i * req.cmdsize)
        self._execute_command_buffer(obj.gpuaddr + obj.offset, obj.size)
      self.timestamp += 1
      req.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
      req.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID): pass
    else: raise NotImplementedError(f"unsupported KGSL ioctl {nr:#x} {name}")
    return 0
