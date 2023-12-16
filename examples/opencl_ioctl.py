import ctypes, ctypes.util, struct, fcntl
from hexdump import hexdump
from tinygrad.runtime.ops_gpu import CLDevice, CLAllocator

# https://github.com/ensc/dietlibc/blob/master/include/sys/aarch64-ioctl.h

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def callback(fd, request, argp):
  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  print(f"ioctl({fd=}, {request=:X}, {argp=:X}) = {ret=}")
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
print("***** done")
