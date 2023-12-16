import ctypes, ctypes.util, struct, fcntl
from hexdump import hexdump
from tinygrad.runtime.ops_gpu import CLDevice, CLAllocator

import pathlib, sys
sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
hdr = (pathlib.Path(__file__).parent.parent.parent / "extra/qcom_gpu_driver/msm_kgsl.h").read_text()
hdr = [x for x in hdr.replace("\\\n","").split("\n") if x.startswith("#define")]
#print('\n'.join(hdr))

IOCTL_KGSL_GPUOBJ_ALLOC = 0x45
from extra.qcom_gpu_driver.msm_kgsl import struct_kgsl_gpuobj_alloc

# https://github.com/ensc/dietlibc/blob/master/include/sys/aarch64-ioctl.h

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def callback(fd, request, argp):
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF

  if nr == IOCTL_KGSL_GPUOBJ_ALLOC:
    s = ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(struct_kgsl_gpuobj_alloc)).contents
    for field_name, field_type in s._fields_:
      print(f"  {field_name:20s}: 0x{getattr(s, field_name):X}")

  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  print(f"ioctl({fd=}, (dir:{idir}, size:0x{size:3X}, type:{itype:d}, nr:0x{nr:2X}), {argp=:X}) = {ret=}")
  return ret

# AARCH64 trampoline to callback
tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
tramp += struct.pack("Q", ctypes.cast(ctypes.byref(callback), ctypes.POINTER(ctypes.c_ulong)).contents.value)

# get real ioctl address
libc = ctypes.CDLL(ctypes.util.find_library("libc"))
ioctl_address = ctypes.cast(ctypes.byref(libc.ioctl), ctypes.POINTER(ctypes.c_ulong))

# hook ioctl
ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
assert ret == 0
libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

print("***** init device")
dev = CLDevice()
print("***** alloc")
alloc = CLAllocator(dev)
alloc._alloc(16)
alloc._alloc(0x2000)
print("***** done")
