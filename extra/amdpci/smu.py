import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0_offset, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

class SMU_IP:
  def __init__(self, adev):
    self.adev = adev

  def smu_cmn_send_smc_msg_with_param(self):
    pass
  
  def mode1_reset():
    pass