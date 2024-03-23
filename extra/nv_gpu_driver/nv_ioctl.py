# type: ignore
import ctypes, ctypes.util, struct, platform, pathlib, re, time, os
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
processor = platform.processor()
IOCTL_SYSCALL = {"aarch64": 0x1d, "x86_64":16}[processor]

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

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
    tramp = b"\x49\xB8" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE0"
  else:
    raise Exception(f"processor {processor} not supported")

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

# *** ioctl lib end ***
import extra.nv_gpu_driver.esc_ioctl as ESC
import extra.nv_gpu_driver.ctrl_ioctl as CTRL
import extra.nv_gpu_driver.class_ioctl as CLASS
nvescs = {getattr(ESC, x):x for x in dir(ESC) if x.startswith("NV_ESC")}
nvcmds = {getattr(CTRL, x):(x, getattr(CTRL, "struct_"+x+"_PARAMS", getattr(CTRL, "struct_"+x.replace("_CMD_", "_")+"_PARAMS", None))) for x in dir(CTRL) if \
          x.startswith("NV") and x[6:].startswith("_CTRL_") and isinstance(getattr(CTRL, x), int)}
nvclasses = {getattr(CLASS, x):x for x in dir(CLASS) if isinstance(getattr(CLASS, x), int)}

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  st = time.perf_counter()
  ret = libc.syscall(IOCTL_SYSCALL, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  et = time.perf_counter()-st
  fn = os.readlink(f"/proc/self/fd/{fd}")
  #print(f"ioctl {request:8x} {fn:20s}")
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  if itype == ord(ESC.NV_IOCTL_MAGIC):
    if nr == ESC.NV_ESC_RM_CONTROL:
      s = get_struct(argp, ESC.NVOS54_PARAMETERS)
      if s.cmd in nvcmds:
        name, struc = nvcmds[s.cmd]
        if struc is not None:
          ss = get_struct(s.params, struc)
          print("NV_ESC_RM_CONTROL ", name, format_struct(ss))
        else:
          print("NV_ESC_RM_CONTROL ", name)
      else:
        print("unhandled cmd", hex(s.cmd))
      #format_struct(s)
      #print(f"{(st-start)*1000:7.2f} ms +{et*1000.:7.2f} ms : {ret:2d} = {name:40s}", ' '.join(format_struct(s)))
    elif nr == ESC.NV_ESC_RM_ALLOC:
      s = get_struct(argp, ESC.NVOS21_PARAMETERS)
      print(f"NV_ESC_RM_ALLOC    class: {nvclasses[s.hClass]:30s}")
    elif nr == ESC.NV_ESC_RM_MAP_MEMORY:
      # nv_ioctl_nvos33_parameters_with_fd
      s = get_struct(argp, ESC.NVOS33_PARAMETERS)
      print(f"NV_ESC_RM_MAP_MEMORY   {s.pLinearAddress:x}")
    elif nr in nvescs:
      print(nvescs[nr])
    else:
      print("unhandled NR", nr)
  #print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=}", os.readlink(f"/proc/self/fd/{fd}") if fd >= 0 else "")
  return ret

install_hook(libc.ioctl, ioctl)


# IOCTL=1 PTX=1 CUDA=1 python3 test/test_ops.py TestOps.test_tiny_add