import ctypes, mmap, pytest
from tinygrad.runtime.autogen import kgsl, libc, mesa
from tinygrad.runtime.ops_qcom import pkt7_hdr
from test.mockgpu.qcom.qcomdriver import QCOMDriver

def test_open():
  driver = QCOMDriver()
  virtfile = driver.tracked_files[0]
  assert virtfile.path == '/dev/kgsl-3d0'
  assert driver.open(virtfile.path, 0, 0, virtfile).fd == 1 << 28
  assert driver.open(virtfile.path, 0, 0, virtfile).fd == (1 << 28) + 1

def test_unknown_ioctl_fails():
  driver = QCOMDriver()
  virtfile = driver.tracked_files[0]
  fd = driver.open(virtfile.path, 0, 0, virtfile)
  with pytest.raises(NotImplementedError):
    fd.ioctl(fd.fd, 0, 0)

def test_device_info():
  driver = QCOMDriver()
  info = kgsl.struct_kgsl_devinfo()
  prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_DEVICE_INFO,
    value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
  driver.kgsl_ioctl(2, ctypes.addressof(prop))
  assert info.chip_id == 0x06030000 # a630

def test_gpuobj_alloc():
  driver = QCOMDriver()
  alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
  driver.kgsl_ioctl(0x45, ctypes.addressof(alloc))
  assert alloc.id == 1
  assert alloc.mmapsize == 0x1000

def test_gpuobj_mmap():
  driver = QCOMDriver()
  alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
  driver.kgsl_ioctl(0x45, ctypes.addressof(alloc))
  virtfile = driver.tracked_files[0]
  fd = driver.open(virtfile.path, 0, 0, virtfile)
  addr = fd.mmap(0, alloc.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd.fd, alloc.id * 0x1000)
  assert addr not in (0, -1)
  libc.munmap(addr, alloc.mmapsize)

def test_context_create():
  driver = QCOMDriver()
  first, second = kgsl.struct_kgsl_drawctxt_create(), kgsl.struct_kgsl_drawctxt_create()
  driver.kgsl_ioctl(0x13, ctypes.addressof(first))
  driver.kgsl_ioctl(0x13, ctypes.addressof(second))
  assert (first.drawctxt_id, second.drawctxt_id) == (1, 2)

def test_power_constraint():
  driver = QCOMDriver()
  ctx = kgsl.struct_kgsl_drawctxt_create()
  driver.kgsl_ioctl(0x13, ctypes.addressof(ctx))
  level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=kgsl.KGSL_CONSTRAINT_PWR_MAX)
  constraint = kgsl.struct_kgsl_device_constraint(type=kgsl.KGSL_CONSTRAINT_PWRLEVEL, context_id=ctx.drawctxt_id,
    data=ctypes.addressof(level), size=ctypes.sizeof(level))
  prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
    value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
  assert driver.kgsl_ioctl(0x32, ctypes.addressof(prop)) == 0

def test_gpu_command():
  driver = QCOMDriver()
  ctx = kgsl.struct_kgsl_drawctxt_create()
  driver.kgsl_ioctl(0x13, ctypes.addressof(ctx))
  alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
  driver.kgsl_ioctl(0x45, ctypes.addressof(alloc))
  addr = driver.kgsl_mmap(0, alloc.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)
  ctypes.c_uint32.from_address(addr).value = pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0)
  obj = kgsl.struct_kgsl_command_object(gpuaddr=addr, size=4, flags=kgsl.KGSL_CMDLIST_IB)
  cmd = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1, context_id=ctx.drawctxt_id)
  driver.kgsl_ioctl(0x4a, ctypes.addressof(cmd))
  assert cmd.timestamp == 1
  libc.munmap(addr, alloc.mmapsize)

def test_waittimestamp():
  driver = QCOMDriver()
  ctx = kgsl.struct_kgsl_drawctxt_create()
  driver.kgsl_ioctl(0x13, ctypes.addressof(ctx))
  driver.timestamps[ctx.drawctxt_id] = 2
  wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=ctx.drawctxt_id, timestamp=2, timeout=0xffffffff)
  assert driver.kgsl_ioctl(0x07, ctypes.addressof(wait)) == 0

def test_gpuobj_free():
  driver = QCOMDriver()
  alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
  driver.kgsl_ioctl(0x45, ctypes.addressof(alloc))
  driver.gpuaddrs[alloc.id] = 0x1000
  free = kgsl.struct_kgsl_gpuobj_free(id=alloc.id)
  assert driver.kgsl_ioctl(0x46, ctypes.addressof(free)) == 0
  assert alloc.id not in driver.gpuobjs
  assert alloc.id not in driver.gpuaddrs

def test_gpu_range():
  driver = QCOMDriver()
  alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
  driver.kgsl_ioctl(0x45, ctypes.addressof(alloc))
  addr = driver.kgsl_mmap(0, alloc.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

  driver.check_gpu_range(addr, 4)
  driver.check_gpu_range(addr + 0xffc, 4)
  with pytest.raises(ValueError):
    driver.check_gpu_range(addr, 0)
  with pytest.raises(ValueError):
    driver.check_gpu_range(addr + 0xffd, 4)

  libc.munmap(addr, alloc.mmapsize)
