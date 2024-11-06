import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_smu_v13_0_0, amdgpu_nbio_4_3_0
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

mmMP1_SMN_C2PMSG_82 = 0x0292
mmMP1_SMN_C2PMSG_90 = 0x029a
mmMP1_SMN_C2PMSG_75 = 0x028b
mmMP1_SMN_C2PMSG_53 = 0x0275
mmMP1_SMN_C2PMSG_54 = 0x0276
mmMP1_SMN_C2PMSG_66 = 0x0282

class SOC21_IP:
  def __init__(self, adev):
    self.adev = adev

    self.param_reg = self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_82
    self.msg_reg = self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_66
    self.resp_reg = self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_90

  def init(self):
    print("SOC21 init")
    self.adev.wreg_ip("NBIO", 0, amdgpu_nbio_4_3_0.regRCC_DEV0_EPF2_STRAP2, amdgpu_nbio_4_3_0.regRCC_DEV0_EPF2_STRAP2_BASE_IDX, 0x03132000)
    self.adev.wreg_ip("NBIO", 0, amdgpu_nbio_4_3_0.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, amdgpu_nbio_4_3_0.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN_BASE_IDX, 0x1)
