import os, ctypes, pathlib, re, fcntl, functools, mmap, time
from tinygrad.helpers import to_mv, getenv
from extra.nv_gpu_driver import nv_ioctl
from extra.nv_gpu_driver import esc_ioctl as nvesc
from extra.nv_gpu_driver import class_ioctl as nvcls
from extra.nv_gpu_driver import ctrl_ioctl as nvctrl
from extra.nv_gpu_driver import uvm_ioctl as nvuvm
from hexdump import hexdump

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
MAP_NORESERVE = 0x4000
MAP_FIXED = 0x10

def _IOWR(type, nr, size):
  return (3 << 30) | (size & 0x1FFF) << 16 | (type & 0xFF) << 8 | (nr & 0xFF)

def rm_alloc(fd, clss, root, parant, params):
  made = nvesc.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def rm_control(fd, cmd, client, obj, params):
  made = nvesc.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, params=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)), paramsSize=ctypes.sizeof(params))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_CONTROL, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def uvm_ioctl(fd, cmd, params):
  ret = fcntl.ioctl(fd, cmd, params)
  if ret != 0: raise RuntimeError(f"ioctl (uvm_control) returned {ret}")
  if params.rmStatus != 0: raise RuntimeError(f"ioctl (uvm_control) returned {params.rmStatus}")

if __name__ == "__main__":
  device_id = 0
  fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
  fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
  fd_dev0 = os.open(f"/dev/nvidia{device_id}", os.O_RDWR | os.O_CLOEXEC)

  root = rm_alloc(fd_ctl, nvesc.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew

  device_params = nvcls.NV0080_ALLOC_PARAMETERS(deviceId=0x0, hClientShare=root, vaMode=nvesc.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
  device = rm_alloc(fd_ctl, nvcls.NV01_DEVICE_0, root, root, device_params).hObjectNew
  subdevice = rm_alloc(fd_ctl, nvcls.NV20_SUBDEVICE_0, root, device, None).hObjectNew
  usermode = rm_alloc(fd_ctl, nvcls.TURING_USERMODE_A, root, subdevice, None).hObjectNew

  vaspace_params = nvesc.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
    flags=nvesc.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING|nvesc.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
  vaspace = rm_alloc(fd_ctl, nvcls.FERMI_VASPACE_A, root, device, vaspace_params).hObjectNew

  gpu_uuid_params = nvctrl.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nvctrl.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
  rm_control(fd_ctl, nvctrl.NV2080_CTRL_CMD_GPU_GET_GID_INFO, root, subdevice, gpu_uuid_params)
  gpu_uuid = (ctypes.c_ubyte*16)()
  for i in range(16): gpu_uuid[i] = gpu_uuid_params.data[i]
  
  # register uvm
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_INITIALIZE), nvuvm.UVM_INITIALIZE_PARAMS())

  register_gpu = nvuvm.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=nvuvm.struct_nv_uuid(uuid=gpu_uuid))
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_REGISTER_GPU[2]), register_gpu)

  # create_group = nvuvm.UVM_CREATE_RANGE_GROUP_PARAMS(rangeGroupId=0)
  # uvm_ioctl(fd_uvm, int(nvuvm.UVM_CREATE_RANGE_GROUP[2]), create_group)

  register_vaspace = nvuvm.UVM_REGISTER_GPU_VASPACE_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=gpu_uuid), rmCtrlFd=fd_ctl, hClient=root, hVaSpace=vaspace)
  uvm_ioctl(fd_uvm, int(nvuvm.UVM_REGISTER_GPU_VASPACE[2]), register_vaspace)


