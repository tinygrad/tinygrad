import os, ctypes, time
from tinygrad.runtime.autogen import amdgpu_smu_v13_0_0
from tinygrad.runtime.support.am.amring import AMRegister

mmMP1_SMN_C2PMSG_82 = 0x0292
mmMP1_SMN_C2PMSG_90 = 0x029a
mmMP1_SMN_C2PMSG_75 = 0x028b
mmMP1_SMN_C2PMSG_53 = 0x0275
mmMP1_SMN_C2PMSG_54 = 0x0276
mmMP1_SMN_C2PMSG_66 = 0x0282

class SMU_IP:
  def __init__(self, adev):
    self.adev = adev

    # self.driver_table_pm = self.adev.mm.palloc(0x4000)
    # self.tool_table_pm = self.adev.mm.palloc(0x19000)

    self.param_reg = AMRegister(self.adev, self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_82)
    self.msg_reg = AMRegister(self.adev, self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_66)
    self.resp_reg = AMRegister(self.adev, self.adev.ip_base("MP1", 0, 0) + mmMP1_SMN_C2PMSG_90)

  def smu_cmn_poll_stat(self): self.adev.wait_reg(self.resp_reg, mask=0xFFFFFFFF, value=1)

  def smu_cmn_send_msg(self, msg, param=0):
    self.resp_reg.write(0)
    self.param_reg.write(param)
    self.msg_reg.write(msg)

  def smu_cmn_send_smc_msg_with_param(self, msg, param, poll=True, read_back_arg=False):
    if poll: self.smu_cmn_poll_stat()

    self.smu_cmn_send_msg(msg, param)
    self.smu_cmn_poll_stat()

    return self.adev.rreg(self.param_reg) if read_back_arg else None

  def mode1_reset(self): self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_Mode1Reset, 0, poll=True)

  # def smu_start_smc_engine(self):
  #   pass

  # def smu_set_driver_table_location(self):
  #   self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetDriverDramAddrHigh, self.driver_table_pm.mc_addr() >> 32, poll=True)
  #   self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetDriverDramAddrLow, self.driver_table_pm.mc_addr() & 0xFFFFFFFF, poll=True)

  # def smu_set_tool_table_location(self):
  #   self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetToolsDramAddrHigh, self.tool_table_pm.mc_addr() >> 32, poll=True)
  #   self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetToolsDramAddrLow, self.tool_table_pm.mc_addr() & 0xFFFFFFFF, poll=True)

  # def smu_setup_pptable(self):
  #   pass

  def smu_run_btc(self):
    self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_RunDcBtc, 0, poll=True)

  def smu_system_features_control(self, enable):
    cmd = amdgpu_smu_v13_0_0.PPSMC_MSG_EnableAllSmuFeatures if enable else amdgpu_smu_v13_0_0.PPSMC_MSG_DisableAllSmuFeatures
    self.smu_cmn_send_smc_msg_with_param(cmd, 0, poll=True)

  def smu_smc_hw_setup(self):
    # self.smu_set_driver_table_location()
    # self.smu_set_tool_table_location()

    self.smu_run_btc()
    self.smu_system_features_control(True)

  def smu_set_power_profile(self, enabled):
    self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetWorkloadMask, 0x20 if enabled else 0x1, poll=True)

    # low = [0x000001F4, 0x00020060, 0x000101F4, 0x00050201, 0x00070201, 0x00040201, 0x00060201, 0x00030259]
    # hi = [0x00000C94, 0x000204E1, 0x000105DC, 0x00050B76, 0x00070B76, 0x00040898, 0x00060898, 0x000308FD]
    # hi = [0x000009B2, 0x000204E1, 0x000105DC, 0x00050B76, 0x00070B76, 0x00040898, 0x00060898, 0x000308FD]
    for clck in [0x00000C94, 0x000204E1, 0x000105DC, 0x00050B76, 0x00070B76, 0x00040898, 0x00060898, 0x000308FD]:
      self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetSoftMinByFreq, clck, poll=False)
      self.smu_cmn_send_smc_msg_with_param(amdgpu_smu_v13_0_0.PPSMC_MSG_SetSoftMaxByFreq, clck, poll=False)

  def init(self):
    # print("SMU init")
    self.smu_smc_hw_setup()
    self.smu_set_power_profile(enabled=True)
