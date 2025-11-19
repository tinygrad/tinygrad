import ctypes, queue
from tinygrad.runtime.ops_cuda import CUDADevice
from tinygrad.runtime.autogen import cuda, nvcuvid
from tinygrad.helpers import DEBUG, init_c_var, getenv

from PIL import Image
from tinygrad import dtypes
from tinygrad.tensor import Tensor

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo

import cv2, numpy as np
from tinygrad.device import Device
from pathlib import Path

from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, nv_gpu, NVKIface, nv_iowr
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED

class FakeNV:
  def __init__(self):
    self.iface = NVKIface(self, 0)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())

    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL,
      nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(offset=0, limit=562949953421311, hVASpace=0))

    self.usermode = self.iface.rm_alloc(self.subdevice, nv_gpu.VOLTA_USERMODE_A)

    # self.sysmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_SYSTEM,
    #   nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.virtmem, type=0, flags=114945, attr=713031680, attr2=67108873,
    #                                      format=6, size=163840, alignment=4096, offset=0, limit=163839))

    # self.notifier = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_ALLOC_MEMORY,
    #   nv_gpu.NV_NOTIFIER_ALLOC_PARAMETERS(hObjectError=self.sysmem.meta.hMemory, size=4096, count=1))

    gpfifo_mem = self.nv_virt_alloc(4832034816, 196608)

    # AMPERE_CHANNEL_GPFIFO_A
    self.vid_gpfifo = self.iface.rm_alloc(self.nvdevice, nv_gpu.AMPERE_CHANNEL_GPFIFO_A,
      nv_gpu.NV_CHANNEL_ALLOC_PARAMS(hObjectError=2147483663, ))
    

  def nv_virt_alloc(self, off, size):
    NVKIface.host_object_enumerator += 1

    virt = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV50_MEMORY_VIRTUAL,
      nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=1886745448, type=6, flags=573440, width=0, height=0, pitch=0, attr=369098752, attr2=5, format=6,
       comprCovg=0, zcullCovg=0, rangeLo=4294967296, rangeHi=562949953421311, size=size, alignment=65536, offset=off, limit=size-1, 
       address=None, ctagOffset=0, hVASpace=0, internalflags=0, tag=0, numaNode=0))

    made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.iface.root, hObjectParent=self.dev.nvdevice, flags=flags,
      hClass=nv_gpu.NV01_MEMORY_SYSTEM, pMemory=0, limit=size-1), fd=self.iface.fd_ctl.fd)
    nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

    hex(print(made.pMemory))

    # self.iface.fd_ctl.mmap(0x0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, 0)


if __name__ == "__main__":
  dev = FakeNV()

