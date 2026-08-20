import ctypes, functools, mmap, typing
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU
from test.mockgpu.qcom.emu import execute_dispatch

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp):
    return self.driver.kgsl_ioctl(request, argp)

  def mmap(self, st, sz, prot, flags, fd, off):
    return self.driver.kgsl_mmap(st, sz, prot, flags, off)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.next_fd = 1 << 28
    self.tracked_files.append(VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self)))
    self.next_obj, self.gpuobjs = 1, {}
    self.gpuaddrs = {}
    self.next_ctx, self.contexts = 1, set()
    self.gpu = QCOMGPU(self.execute_shader, self.check_gpu_range)
    self.timestamps = {}

  def check_gpu_range(self, address, size):
    if size <= 0: raise ValueError(f'invalid KGSL range size {size}')
    mapped = any(base <= address and address + size <= base + self.gpuobjs[obj_id]
                 for obj_id, base in self.gpuaddrs.items())
    if not mapped: raise ValueError(f'unmapped KGSL range {address:#x}+{size:#x}')

  def execute_shader(self, code, grid_size, local_size, _regs, _state):
    lanes = local_size[0] * local_size[1] * local_size[2]
    return execute_dispatch(code, grid_size, local_size, self.gpu.local_id_register(), self.gpu.constant_regs(lanes),
                            check_range=self.check_gpu_range, workgroup_id_register=self.gpu.workgroup_id_register())

  def kgsl_ioctl(self, request, argp):
    nr = request & 0xff
    if nr == 0x02:
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f'unsupported KGSL property {prop.type:#x}')
      if prop.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError('KGSL device info buffer is too small')
      kgsl.struct_kgsl_devinfo.from_address(typing.cast(int, prop.value)).chip_id = 0x06030000
      return 0
    if nr == 0x45:
      alloc = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      alloc.id = self.next_obj
      self.next_obj += 1
      alloc.mmapsize = alloc.size
      self.gpuobjs[alloc.id] = alloc.size
      return 0
    if nr == 0x13:
      ctx = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      ctx.drawctxt_id = self.next_ctx
      self.contexts.add(self.next_ctx)
      self.timestamps[self.next_ctx] = 0
      self.next_ctx += 1
      return 0
    if nr == 0x32:
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise NotImplementedError(f'unsupported KGSL property {prop.type:#x}')
      if prop.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_device_constraint): raise ValueError('KGSL power constraint is too small')
      constraint = kgsl.struct_kgsl_device_constraint.from_address(typing.cast(int, prop.value))
      if constraint.context_id not in self.contexts: raise ValueError(f'invalid KGSL context {constraint.context_id}')
      if constraint.type != kgsl.KGSL_CONSTRAINT_PWRLEVEL: raise NotImplementedError(f'unsupported KGSL constraint {constraint.type:#x}')
      level = kgsl.struct_kgsl_device_constraint_pwrlevel.from_address(typing.cast(int, constraint.data))
      if level.level != kgsl.KGSL_CONSTRAINT_PWR_MAX: raise NotImplementedError(f'unsupported KGSL power level {level.level}')
      return 0
    if nr == 0x07:
      wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp)
      if wait.context_id not in self.contexts: raise ValueError(f'invalid KGSL context {wait.context_id}')
      if wait.timestamp > self.timestamps[wait.context_id]: raise TimeoutError(f'incomplete KGSL timestamp {wait.timestamp}')
      return 0
    if nr == 0x46:
      free = kgsl.struct_kgsl_gpuobj_free.from_address(argp)
      if free.id not in self.gpuobjs: raise ValueError(f'invalid KGSL object {free.id}')
      self.gpuobjs.pop(free.id)
      self.gpuaddrs.pop(free.id, None)
      return 0
    if nr == 0x4a:
      cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      if cmd.context_id not in self.contexts: raise ValueError(f'invalid KGSL context {cmd.context_id}')
      if cmd.numcmds != 1 or cmd.cmdsize != ctypes.sizeof(kgsl.struct_kgsl_command_object):
        raise NotImplementedError('unsupported KGSL command list')
      obj = kgsl.struct_kgsl_command_object.from_address(cmd.cmdlist)
      if obj.flags != kgsl.KGSL_CMDLIST_IB or obj.size % 4: raise ValueError('invalid KGSL command object')
      self.check_gpu_range(obj.gpuaddr, obj.size)
      self.gpu.execute((ctypes.c_uint32 * (obj.size // 4)).from_address(obj.gpuaddr))
      self.timestamps[cmd.context_id] += 1
      cmd.timestamp = self.timestamps[cmd.context_id]
      return 0
    raise NotImplementedError(f'unsupported KGSL ioctl {nr:#x}')

  def kgsl_mmap(self, st, sz, prot, flags, off):
    if off % 0x1000: raise ValueError('KGSL mmap offset is not page aligned')
    obj_id = off // 0x1000
    if self.gpuobjs.get(obj_id) != sz: raise ValueError(f'invalid KGSL object mapping {obj_id}')
    addr = libc.mmap(st, sz, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
    self.gpuaddrs[obj_id] = addr
    return addr

  def open(self, name, flags, mode, virtfile):
    fd = self.next_fd
    self.next_fd += 1
    return virtfile.fdcls(fd)
