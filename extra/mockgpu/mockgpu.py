import ctypes, ctypes.util, struct, platform, pathlib, re, time, os, builtins
from extra.mockgpu.kdriver import AMDDriver
from tinygrad.helpers import from_mv
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

processor = platform.processor()
OPEN_SYSCALL = {"aarch64": None, "x86_64": 2}[processor]
OPENAT_SYSCALL = {"aarch64": 56, "x86_64": 257}[processor]
READ_SYSCALL = {"aarch64": 63, "x86_64": 0}[processor]
IOCTL_SYSCALL = {"aarch64": 29, "x86_64": 16}[processor]
MMAP_SYSCALL = {"aarch64": 222, "x86_64": 9}[processor]

def install_hook(c_function, python_function):
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  if processor == "aarch64":
    # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
    # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
    # 0x0000000000000008:  00 02 1F D6    br  x16
    tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
    tramp += struct.pack("Q", python_function_addr)
  elif processor == "x86_64":
    # 0x0000000000000000:  49 B8 aa aa aa aa aa aa aa aa    movabs r8, <address>
    # 0x000000000000000a:  41 FF E0                         jmp    r8
    # tramp = b"\x49\xB8" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE0"
    # push r9
    # push r9
    # mov r9, 0x1122334455667788
    # mov [rsp+8], r9
    # pop r9
    # ret
    tramp = b"\x41\x51\x41\x51\x49\xB9" + struct.pack("Q", python_function_addr) + b"\x4C\x89\x4C\x24\x08\x41\x59\xC3"
  else:
    raise Exception(f"processor {processor} not supported")

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

drivers = [AMDDriver()]

libc_open = None
libc_ioctl = None

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_ulong)
def _open(name, flags, mode):
  for d in drivers:
    if any(re.match(x, name.decode()) for x in d.files):
      x = d.open(name, flags, mode)
      return x

  return libc.syscall(OPEN_SYSCALL, name, flags, mode)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def _ioctl(fd, request, argp):
  for d in drivers:
    if any(fd == x for x in d.fds):
      return d.ioctl(fd, request, argp)

  return libc.syscall(IOCTL_SYSCALL, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
def _read(fd, buf, sz):
  for d in drivers:
    if any(fd == x for x in d.fds):
      return d.read(fd, request, argp)

  return libc.syscall(READ_SYSCALL, ctypes.c_int(fd), ctypes.c_void_p(buf), ctypes.c_size_t(sz))

# libc_mmap = None
def _mmap(start, sz, prot, flags, fd, offset):
  for d in drivers:
    if any(fd == x for x in d.fds):
      return d.mmap(start, sz, prot, flags, fd, offset)

  return libc.mmap(start, sz, prot, flags, fd, offset)

def _munmap(buf, sz):
  return libc.munmap(buf, sz)

orignal_memoryview = builtins.memoryview
class WatchedMemoryView:
  def __init__(self, data, rcb, wcb):
    self.mv = orignal_memoryview(data)
    self.rcb, self.wcb = rcb, wcb

  def __getitem__(self, index):
    self.rcb(self.mv, index)
    return self.mv[index]

  def __setitem__(self, index, value):
    self.mv[index] = value
    print(index, value)
    self.wcb(self.mv, index)

  def cast(self, new_type, **kwargs): 
    self.mv = self.mv.cast(new_type, **kwargs)
    return self

  @property
  def nbytes(self): return self.mv.nbytes
  def __len__(self): return len(self.mv)
  def __repr__(self): return repr(self.mv)

def _memoryview(mem):
  if isinstance(mem, int) or isinstance(mem, ctypes.Array):
    addr = ctypes.addressof(mem) if isinstance(mem, ctypes.Array) else mem
    for d in drivers:
      for st,en,rcb,wcb in d.watched_addresses:
        if st <= addr <= en: return WatchedMemoryView(mem, rcb, wcb)
  return orignal_memoryview(mem)

install_hook(libc.open, _open)
install_hook(libc.ioctl, _ioctl)
builtins.memoryview = _memoryview
