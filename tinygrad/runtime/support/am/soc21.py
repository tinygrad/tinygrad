import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_smu_v13_0_0, amdgpu_nbio_4_3_0
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

class SOC21_IP:
  def __init__(self, adev): self.adev = adev

  def init(self):
    print("SOC21 init")
    self.adev.regRCC_DEV0_EPF2_STRAP2.write(0x03132000)
    self.adev.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN.write(0x1)
