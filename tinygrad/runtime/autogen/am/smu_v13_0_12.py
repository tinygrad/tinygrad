# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
PPSMC_Result: TypeAlias = Annotated[int, ctypes.c_uint32]
PPSMC_MSG: TypeAlias = Annotated[int, ctypes.c_uint32]
class FEATURE_LIST_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
FEATURE_DATA_CALCULATION = FEATURE_LIST_e.define('FEATURE_DATA_CALCULATION', 0)
FEATURE_DPM_FCLK = FEATURE_LIST_e.define('FEATURE_DPM_FCLK', 1)
FEATURE_DPM_GFXCLK = FEATURE_LIST_e.define('FEATURE_DPM_GFXCLK', 2)
FEATURE_DPM_LCLK = FEATURE_LIST_e.define('FEATURE_DPM_LCLK', 3)
FEATURE_DPM_SOCCLK = FEATURE_LIST_e.define('FEATURE_DPM_SOCCLK', 4)
FEATURE_DPM_UCLK = FEATURE_LIST_e.define('FEATURE_DPM_UCLK', 5)
FEATURE_DPM_VCN = FEATURE_LIST_e.define('FEATURE_DPM_VCN', 6)
FEATURE_DPM_XGMI = FEATURE_LIST_e.define('FEATURE_DPM_XGMI', 7)
FEATURE_DS_FCLK = FEATURE_LIST_e.define('FEATURE_DS_FCLK', 8)
FEATURE_DS_GFXCLK = FEATURE_LIST_e.define('FEATURE_DS_GFXCLK', 9)
FEATURE_DS_LCLK = FEATURE_LIST_e.define('FEATURE_DS_LCLK', 10)
FEATURE_DS_MP0CLK = FEATURE_LIST_e.define('FEATURE_DS_MP0CLK', 11)
FEATURE_DS_MP1CLK = FEATURE_LIST_e.define('FEATURE_DS_MP1CLK', 12)
FEATURE_DS_MPIOCLK = FEATURE_LIST_e.define('FEATURE_DS_MPIOCLK', 13)
FEATURE_DS_SOCCLK = FEATURE_LIST_e.define('FEATURE_DS_SOCCLK', 14)
FEATURE_DS_VCN = FEATURE_LIST_e.define('FEATURE_DS_VCN', 15)
FEATURE_APCC_DFLL = FEATURE_LIST_e.define('FEATURE_APCC_DFLL', 16)
FEATURE_APCC_PLUS = FEATURE_LIST_e.define('FEATURE_APCC_PLUS', 17)
FEATURE_PPT = FEATURE_LIST_e.define('FEATURE_PPT', 18)
FEATURE_TDC = FEATURE_LIST_e.define('FEATURE_TDC', 19)
FEATURE_THERMAL = FEATURE_LIST_e.define('FEATURE_THERMAL', 20)
FEATURE_SOC_PCC = FEATURE_LIST_e.define('FEATURE_SOC_PCC', 21)
FEATURE_PROCHOT = FEATURE_LIST_e.define('FEATURE_PROCHOT', 22)
FEATURE_FDD_AID_HBM = FEATURE_LIST_e.define('FEATURE_FDD_AID_HBM', 23)
FEATURE_FDD_AID_SOC = FEATURE_LIST_e.define('FEATURE_FDD_AID_SOC', 24)
FEATURE_FDD_XCD_EDC = FEATURE_LIST_e.define('FEATURE_FDD_XCD_EDC', 25)
FEATURE_FDD_XCD_XVMIN = FEATURE_LIST_e.define('FEATURE_FDD_XCD_XVMIN', 26)
FEATURE_FW_CTF = FEATURE_LIST_e.define('FEATURE_FW_CTF', 27)
FEATURE_SMU_CG = FEATURE_LIST_e.define('FEATURE_SMU_CG', 28)
FEATURE_PSI7 = FEATURE_LIST_e.define('FEATURE_PSI7', 29)
FEATURE_XGMI_PER_LINK_PWR_DOWN = FEATURE_LIST_e.define('FEATURE_XGMI_PER_LINK_PWR_DOWN', 30)
FEATURE_SOC_DC_RTC = FEATURE_LIST_e.define('FEATURE_SOC_DC_RTC', 31)
FEATURE_GFX_DC_RTC = FEATURE_LIST_e.define('FEATURE_GFX_DC_RTC', 32)
FEATURE_DVM_MIN_PSM = FEATURE_LIST_e.define('FEATURE_DVM_MIN_PSM', 33)
FEATURE_PRC = FEATURE_LIST_e.define('FEATURE_PRC', 34)
FEATURE_PSM_SQ_THROTTLER = FEATURE_LIST_e.define('FEATURE_PSM_SQ_THROTTLER', 35)
FEATURE_PIT = FEATURE_LIST_e.define('FEATURE_PIT', 36)
FEATURE_DVO = FEATURE_LIST_e.define('FEATURE_DVO', 37)
FEATURE_XVMINORPSM_CLKSTOP_DS = FEATURE_LIST_e.define('FEATURE_XVMINORPSM_CLKSTOP_DS', 38)
FEATURE_GLOBAL_DPM = FEATURE_LIST_e.define('FEATURE_GLOBAL_DPM', 39)
FEATURE_HROM_EN = FEATURE_LIST_e.define('FEATURE_HROM_EN', 40)
NUM_FEATURES = FEATURE_LIST_e.define('NUM_FEATURES', 41)

class PCIE_LINK_SPEED_INDEX_TABLE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PCIE_LINK_SPEED_INDEX_TABLE_RESERVED = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_RESERVED', 0)
PCIE_LINK_SPEED_INDEX_TABLE_GEN1 = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_GEN1', 1)
PCIE_LINK_SPEED_INDEX_TABLE_GEN2 = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_GEN2', 2)
PCIE_LINK_SPEED_INDEX_TABLE_GEN3 = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_GEN3', 3)
PCIE_LINK_SPEED_INDEX_TABLE_GEN4 = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_GEN4', 4)
PCIE_LINK_SPEED_INDEX_TABLE_GEN5 = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_GEN5', 5)
PCIE_LINK_SPEED_INDEX_TABLE_COUNT = PCIE_LINK_SPEED_INDEX_TABLE_e.define('PCIE_LINK_SPEED_INDEX_TABLE_COUNT', 6)

class GFX_GUARDBAND_OFFSET_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
GFX_GUARDBAND_OFFSET_0 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_0', 0)
GFX_GUARDBAND_OFFSET_1 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_1', 1)
GFX_GUARDBAND_OFFSET_2 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_2', 2)
GFX_GUARDBAND_OFFSET_3 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_3', 3)
GFX_GUARDBAND_OFFSET_4 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_4', 4)
GFX_GUARDBAND_OFFSET_5 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_5', 5)
GFX_GUARDBAND_OFFSET_6 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_6', 6)
GFX_GUARDBAND_OFFSET_7 = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_7', 7)
GFX_GUARDBAND_OFFSET_COUNT = GFX_GUARDBAND_OFFSET_e.define('GFX_GUARDBAND_OFFSET_COUNT', 8)

class GFX_DVM_MARGIN_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
GFX_DVM_MARGINHI_0 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_0', 0)
GFX_DVM_MARGINHI_1 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_1', 1)
GFX_DVM_MARGINHI_2 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_2', 2)
GFX_DVM_MARGINHI_3 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_3', 3)
GFX_DVM_MARGINHI_4 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_4', 4)
GFX_DVM_MARGINHI_5 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_5', 5)
GFX_DVM_MARGINHI_6 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_6', 6)
GFX_DVM_MARGINHI_7 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINHI_7', 7)
GFX_DVM_MARGINLO_0 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_0', 8)
GFX_DVM_MARGINLO_1 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_1', 9)
GFX_DVM_MARGINLO_2 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_2', 10)
GFX_DVM_MARGINLO_3 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_3', 11)
GFX_DVM_MARGINLO_4 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_4', 12)
GFX_DVM_MARGINLO_5 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_5', 13)
GFX_DVM_MARGINLO_6 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_6', 14)
GFX_DVM_MARGINLO_7 = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGINLO_7', 15)
GFX_DVM_MARGIN_COUNT = GFX_DVM_MARGIN_e.define('GFX_DVM_MARGIN_COUNT', 16)

class SYSTEM_TEMP_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SYSTEM_TEMP_UBB_FPGA = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_FPGA', 0)
SYSTEM_TEMP_UBB_FRONT = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_FRONT', 1)
SYSTEM_TEMP_UBB_BACK = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_BACK', 2)
SYSTEM_TEMP_UBB_OAM7 = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_OAM7', 3)
SYSTEM_TEMP_UBB_IBC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_IBC', 4)
SYSTEM_TEMP_UBB_UFPGA = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_UFPGA', 5)
SYSTEM_TEMP_UBB_OAM1 = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_OAM1', 6)
SYSTEM_TEMP_OAM_0_1_HSC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_0_1_HSC', 7)
SYSTEM_TEMP_OAM_2_3_HSC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_2_3_HSC', 8)
SYSTEM_TEMP_OAM_4_5_HSC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_4_5_HSC', 9)
SYSTEM_TEMP_OAM_6_7_HSC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_6_7_HSC', 10)
SYSTEM_TEMP_UBB_FPGA_0V72_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_FPGA_0V72_VR', 11)
SYSTEM_TEMP_UBB_FPGA_3V3_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_UBB_FPGA_3V3_VR', 12)
SYSTEM_TEMP_RETIMER_0_1_2_3_1V2_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_0_1_2_3_1V2_VR', 13)
SYSTEM_TEMP_RETIMER_4_5_6_7_1V2_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_4_5_6_7_1V2_VR', 14)
SYSTEM_TEMP_RETIMER_0_1_0V9_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_0_1_0V9_VR', 15)
SYSTEM_TEMP_RETIMER_4_5_0V9_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_4_5_0V9_VR', 16)
SYSTEM_TEMP_RETIMER_2_3_0V9_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_2_3_0V9_VR', 17)
SYSTEM_TEMP_RETIMER_6_7_0V9_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_RETIMER_6_7_0V9_VR', 18)
SYSTEM_TEMP_OAM_0_1_2_3_3V3_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_0_1_2_3_3V3_VR', 19)
SYSTEM_TEMP_OAM_4_5_6_7_3V3_VR = SYSTEM_TEMP_e.define('SYSTEM_TEMP_OAM_4_5_6_7_3V3_VR', 20)
SYSTEM_TEMP_IBC_HSC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_IBC_HSC', 21)
SYSTEM_TEMP_IBC = SYSTEM_TEMP_e.define('SYSTEM_TEMP_IBC', 22)
SYSTEM_TEMP_MAX_ENTRIES = SYSTEM_TEMP_e.define('SYSTEM_TEMP_MAX_ENTRIES', 32)

class NODE_TEMP_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
NODE_TEMP_RETIMER = NODE_TEMP_e.define('NODE_TEMP_RETIMER', 0)
NODE_TEMP_IBC_TEMP = NODE_TEMP_e.define('NODE_TEMP_IBC_TEMP', 1)
NODE_TEMP_IBC_2_TEMP = NODE_TEMP_e.define('NODE_TEMP_IBC_2_TEMP', 2)
NODE_TEMP_VDD18_VR_TEMP = NODE_TEMP_e.define('NODE_TEMP_VDD18_VR_TEMP', 3)
NODE_TEMP_04_HBM_B_VR_TEMP = NODE_TEMP_e.define('NODE_TEMP_04_HBM_B_VR_TEMP', 4)
NODE_TEMP_04_HBM_D_VR_TEMP = NODE_TEMP_e.define('NODE_TEMP_04_HBM_D_VR_TEMP', 5)
NODE_TEMP_MAX_TEMP_ENTRIES = NODE_TEMP_e.define('NODE_TEMP_MAX_TEMP_ENTRIES', 12)

class SVI_TEMP_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SVI_VDDCR_VDD0_TEMP = SVI_TEMP_e.define('SVI_VDDCR_VDD0_TEMP', 0)
SVI_VDDCR_VDD1_TEMP = SVI_TEMP_e.define('SVI_VDDCR_VDD1_TEMP', 1)
SVI_VDDCR_VDD2_TEMP = SVI_TEMP_e.define('SVI_VDDCR_VDD2_TEMP', 2)
SVI_VDDCR_VDD3_TEMP = SVI_TEMP_e.define('SVI_VDDCR_VDD3_TEMP', 3)
SVI_VDDCR_SOC_A_TEMP = SVI_TEMP_e.define('SVI_VDDCR_SOC_A_TEMP', 4)
SVI_VDDCR_SOC_C_TEMP = SVI_TEMP_e.define('SVI_VDDCR_SOC_C_TEMP', 5)
SVI_VDDCR_SOCIO_A_TEMP = SVI_TEMP_e.define('SVI_VDDCR_SOCIO_A_TEMP', 6)
SVI_VDDCR_SOCIO_C_TEMP = SVI_TEMP_e.define('SVI_VDDCR_SOCIO_C_TEMP', 7)
SVI_VDD_085_HBM_TEMP = SVI_TEMP_e.define('SVI_VDD_085_HBM_TEMP', 8)
SVI_VDDCR_11_HBM_B_TEMP = SVI_TEMP_e.define('SVI_VDDCR_11_HBM_B_TEMP', 9)
SVI_VDDCR_11_HBM_D_TEMP = SVI_TEMP_e.define('SVI_VDDCR_11_HBM_D_TEMP', 10)
SVI_VDD_USR_TEMP = SVI_TEMP_e.define('SVI_VDD_USR_TEMP', 11)
SVI_VDDIO_11_E32_TEMP = SVI_TEMP_e.define('SVI_VDDIO_11_E32_TEMP', 12)
SVI_MAX_TEMP_ENTRIES = SVI_TEMP_e.define('SVI_MAX_TEMP_ENTRIES', 13)

@c.record
class MetricsTable_t(c.Struct):
  SIZE = 1284
  AccumulationCounter: Annotated[uint64_t, 0]
  MaxSocketTemperature: Annotated[uint32_t, 8]
  MaxVrTemperature: Annotated[uint32_t, 12]
  MaxHbmTemperature: Annotated[uint32_t, 16]
  MaxSocketTemperatureAcc: Annotated[uint64_t, 20]
  MaxVrTemperatureAcc: Annotated[uint64_t, 28]
  MaxHbmTemperatureAcc: Annotated[uint64_t, 36]
  SocketPowerLimit: Annotated[uint32_t, 44]
  SocketPower: Annotated[uint32_t, 48]
  Timestamp: Annotated[uint64_t, 52]
  SocketEnergyAcc: Annotated[uint64_t, 60]
  XcdEnergyAcc: Annotated[uint64_t, 68]
  AidEnergyAcc: Annotated[uint64_t, 76]
  HbmEnergyAcc: Annotated[uint64_t, 84]
  GfxclkFrequencyLimit: Annotated[uint32_t, 92]
  FclkFrequency: Annotated[uint32_t, 96]
  UclkFrequency: Annotated[uint32_t, 100]
  SocclkFrequency: Annotated[c.Array[uint32_t, Literal[4]], 104]
  VclkFrequency: Annotated[c.Array[uint32_t, Literal[4]], 120]
  DclkFrequency: Annotated[c.Array[uint32_t, Literal[4]], 136]
  LclkFrequency: Annotated[c.Array[uint32_t, Literal[4]], 152]
  GfxclkFrequencyAcc: Annotated[c.Array[uint64_t, Literal[8]], 168]
  MaxLclkDpmRange: Annotated[uint32_t, 232]
  MinLclkDpmRange: Annotated[uint32_t, 236]
  XgmiWidth: Annotated[uint32_t, 240]
  XgmiBitrate: Annotated[uint32_t, 244]
  XgmiReadBandwidthAcc: Annotated[c.Array[uint64_t, Literal[8]], 248]
  XgmiWriteBandwidthAcc: Annotated[c.Array[uint64_t, Literal[8]], 312]
  SocketGfxBusy: Annotated[uint32_t, 376]
  DramBandwidthUtilization: Annotated[uint32_t, 380]
  SocketGfxBusyAcc: Annotated[uint64_t, 384]
  DramBandwidthAcc: Annotated[uint64_t, 392]
  MaxDramBandwidth: Annotated[uint32_t, 400]
  DramBandwidthUtilizationAcc: Annotated[uint64_t, 404]
  PcieBandwidthAcc: Annotated[c.Array[uint64_t, Literal[4]], 412]
  ProchotResidencyAcc: Annotated[uint32_t, 444]
  PptResidencyAcc: Annotated[uint32_t, 448]
  SocketThmResidencyAcc: Annotated[uint32_t, 452]
  VrThmResidencyAcc: Annotated[uint32_t, 456]
  HbmThmResidencyAcc: Annotated[uint32_t, 460]
  GfxLockXCDMak: Annotated[uint32_t, 464]
  GfxclkFrequency: Annotated[c.Array[uint32_t, Literal[8]], 468]
  XgmiReadDataSizeAcc: Annotated[c.Array[uint64_t, Literal[8]], 500]
  XgmiWriteDataSizeAcc: Annotated[c.Array[uint64_t, Literal[8]], 564]
  PcieBandwidth: Annotated[c.Array[uint32_t, Literal[4]], 628]
  PCIeL0ToRecoveryCountAcc: Annotated[uint32_t, 644]
  PCIenReplayAAcc: Annotated[uint32_t, 648]
  PCIenReplayARolloverCountAcc: Annotated[uint32_t, 652]
  PCIeNAKSentCountAcc: Annotated[uint32_t, 656]
  PCIeNAKReceivedCountAcc: Annotated[uint32_t, 660]
  VcnBusy: Annotated[c.Array[uint32_t, Literal[4]], 664]
  JpegBusy: Annotated[c.Array[uint32_t, Literal[40]], 680]
  PCIeLinkSpeed: Annotated[uint32_t, 840]
  PCIeLinkWidth: Annotated[uint32_t, 844]
  GfxBusy: Annotated[c.Array[uint32_t, Literal[8]], 848]
  GfxBusyAcc: Annotated[c.Array[uint64_t, Literal[8]], 880]
  PCIeOtherEndRecoveryAcc: Annotated[uint32_t, 944]
  GfxclkBelowHostLimitPptAcc: Annotated[c.Array[uint64_t, Literal[8]], 948]
  GfxclkBelowHostLimitThmAcc: Annotated[c.Array[uint64_t, Literal[8]], 1012]
  GfxclkBelowHostLimitTotalAcc: Annotated[c.Array[uint64_t, Literal[8]], 1076]
  GfxclkLowUtilizationAcc: Annotated[c.Array[uint64_t, Literal[8]], 1140]
  AidTemperature: Annotated[c.Array[uint32_t, Literal[4]], 1204]
  XcdTemperature: Annotated[c.Array[uint32_t, Literal[8]], 1220]
  HbmTemperature: Annotated[c.Array[uint32_t, Literal[8]], 1252]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class SystemMetricsTable_t(c.Struct):
  SIZE = 152
  AccumulationCounter: Annotated[uint64_t, 0]
  LabelVersion: Annotated[uint16_t, 8]
  NodeIdentifier: Annotated[uint16_t, 10]
  SystemTemperatures: Annotated[c.Array[int16_t, Literal[32]], 12]
  NodeTemperatures: Annotated[c.Array[int16_t, Literal[12]], 76]
  VrTemperatures: Annotated[c.Array[int16_t, Literal[13]], 100]
  spare: Annotated[c.Array[int16_t, Literal[7]], 126]
  NodePowerLimit: Annotated[uint32_t, 140]
  NodePower: Annotated[uint32_t, 144]
  GlobalPPTResidencyAcc: Annotated[uint32_t, 148]
uint16_t: TypeAlias = Annotated[int, ctypes.c_uint16]
int16_t: TypeAlias = Annotated[int, ctypes.c_int16]
@c.record
class VfMetricsTable_t(c.Struct):
  SIZE = 56
  AccumulationCounter: Annotated[uint32_t, 0]
  InstGfxclk_TargFreq: Annotated[uint32_t, 4]
  AccGfxclk_TargFreq: Annotated[uint64_t, 8]
  AccGfxRsmuDpm_Busy: Annotated[uint64_t, 16]
  AccGfxclkBelowHostLimitPpt: Annotated[uint64_t, 24]
  AccGfxclkBelowHostLimitThm: Annotated[uint64_t, 32]
  AccGfxclkBelowHostLimitTotal: Annotated[uint64_t, 40]
  AccGfxclkLowUtilization: Annotated[uint64_t, 48]
@c.record
class FRUProductInfo_t(c.Struct):
  SIZE = 168
  ModelNumber: Annotated[c.Array[uint8_t, Literal[20]], 0]
  Name: Annotated[c.Array[uint8_t, Literal[64]], 20]
  Serial: Annotated[c.Array[uint8_t, Literal[20]], 84]
  ManufacturerName: Annotated[c.Array[uint8_t, Literal[32]], 104]
  FruId: Annotated[c.Array[uint8_t, Literal[32]], 136]
uint8_t: TypeAlias = Annotated[int, ctypes.c_ubyte]
@c.record
class StaticMetricsTable_t(c.Struct):
  SIZE = 408
  ProductInfo: Annotated[FRUProductInfo_t, 0]
  MaxSocketPowerLimit: Annotated[uint32_t, 168]
  MaxGfxclkFrequency: Annotated[uint32_t, 172]
  MinGfxclkFrequency: Annotated[uint32_t, 176]
  FclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 180]
  UclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 196]
  SocclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 212]
  VclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 228]
  DclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 244]
  LclkFrequencyTable: Annotated[c.Array[uint32_t, Literal[4]], 260]
  PublicSerialNumber_AID: Annotated[c.Array[uint64_t, Literal[4]], 276]
  PublicSerialNumber_XCD: Annotated[c.Array[uint64_t, Literal[8]], 308]
  MaxXgmiWidth: Annotated[uint32_t, 372]
  MaxXgmiBitrate: Annotated[uint32_t, 376]
  InputTelemetryVoltageInmV: Annotated[uint32_t, 380]
  pldmVersion: Annotated[c.Array[uint32_t, Literal[2]], 384]
  MaxNodePowerLimit: Annotated[uint32_t, 392]
  PPT1Max: Annotated[uint32_t, 396]
  PPT1Min: Annotated[uint32_t, 400]
  PPT1Default: Annotated[uint32_t, 404]
class I2cControllerPort_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CONTROLLER_PORT_0 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_0', 0)
I2C_CONTROLLER_PORT_1 = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_1', 1)
I2C_CONTROLLER_PORT_COUNT = I2cControllerPort_e.define('I2C_CONTROLLER_PORT_COUNT', 2)

class I2cSpeed_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
UNSUPPORTED_1 = I2cSpeed_e.define('UNSUPPORTED_1', 0)
I2C_SPEED_STANDARD_100K = I2cSpeed_e.define('I2C_SPEED_STANDARD_100K', 1)
I2C_SPEED_FAST_400K = I2cSpeed_e.define('I2C_SPEED_FAST_400K', 2)
I2C_SPEED_FAST_PLUS_1M = I2cSpeed_e.define('I2C_SPEED_FAST_PLUS_1M', 3)
UNSUPPORTED_2 = I2cSpeed_e.define('UNSUPPORTED_2', 4)
UNSUPPORTED_3 = I2cSpeed_e.define('UNSUPPORTED_3', 5)
I2C_SPEED_COUNT = I2cSpeed_e.define('I2C_SPEED_COUNT', 6)

class I2cCmdType_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
I2C_CMD_READ = I2cCmdType_e.define('I2C_CMD_READ', 0)
I2C_CMD_WRITE = I2cCmdType_e.define('I2C_CMD_WRITE', 1)
I2C_CMD_COUNT = I2cCmdType_e.define('I2C_CMD_COUNT', 2)

class ERR_CODE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
CODE_DAGB0 = ERR_CODE_e.define('CODE_DAGB0', 0)
CODE_EA0 = ERR_CODE_e.define('CODE_EA0', 5)
CODE_UTCL2_ROUTER = ERR_CODE_e.define('CODE_UTCL2_ROUTER', 10)
CODE_VML2 = ERR_CODE_e.define('CODE_VML2', 11)
CODE_VML2_WALKER = ERR_CODE_e.define('CODE_VML2_WALKER', 12)
CODE_MMCANE = ERR_CODE_e.define('CODE_MMCANE', 13)
CODE_VIDD = ERR_CODE_e.define('CODE_VIDD', 14)
CODE_VIDV = ERR_CODE_e.define('CODE_VIDV', 15)
CODE_JPEG0S = ERR_CODE_e.define('CODE_JPEG0S', 16)
CODE_JPEG0D = ERR_CODE_e.define('CODE_JPEG0D', 17)
CODE_JPEG1S = ERR_CODE_e.define('CODE_JPEG1S', 18)
CODE_JPEG1D = ERR_CODE_e.define('CODE_JPEG1D', 19)
CODE_JPEG2S = ERR_CODE_e.define('CODE_JPEG2S', 20)
CODE_JPEG2D = ERR_CODE_e.define('CODE_JPEG2D', 21)
CODE_JPEG3S = ERR_CODE_e.define('CODE_JPEG3S', 22)
CODE_JPEG3D = ERR_CODE_e.define('CODE_JPEG3D', 23)
CODE_JPEG4S = ERR_CODE_e.define('CODE_JPEG4S', 24)
CODE_JPEG4D = ERR_CODE_e.define('CODE_JPEG4D', 25)
CODE_JPEG5S = ERR_CODE_e.define('CODE_JPEG5S', 26)
CODE_JPEG5D = ERR_CODE_e.define('CODE_JPEG5D', 27)
CODE_JPEG6S = ERR_CODE_e.define('CODE_JPEG6S', 28)
CODE_JPEG6D = ERR_CODE_e.define('CODE_JPEG6D', 29)
CODE_JPEG7S = ERR_CODE_e.define('CODE_JPEG7S', 30)
CODE_JPEG7D = ERR_CODE_e.define('CODE_JPEG7D', 31)
CODE_MMSCHD = ERR_CODE_e.define('CODE_MMSCHD', 32)
CODE_SDMA0 = ERR_CODE_e.define('CODE_SDMA0', 33)
CODE_SDMA1 = ERR_CODE_e.define('CODE_SDMA1', 34)
CODE_SDMA2 = ERR_CODE_e.define('CODE_SDMA2', 35)
CODE_SDMA3 = ERR_CODE_e.define('CODE_SDMA3', 36)
CODE_HDP = ERR_CODE_e.define('CODE_HDP', 37)
CODE_ATHUB = ERR_CODE_e.define('CODE_ATHUB', 38)
CODE_IH = ERR_CODE_e.define('CODE_IH', 39)
CODE_XHUB_POISON = ERR_CODE_e.define('CODE_XHUB_POISON', 40)
CODE_SMN_SLVERR = ERR_CODE_e.define('CODE_SMN_SLVERR', 40)
CODE_WDT = ERR_CODE_e.define('CODE_WDT', 41)
CODE_UNKNOWN = ERR_CODE_e.define('CODE_UNKNOWN', 42)
CODE_COUNT = ERR_CODE_e.define('CODE_COUNT', 43)

class GC_ERROR_CODE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
SH_FED_CODE = GC_ERROR_CODE_e.define('SH_FED_CODE', 0)
GCEA_CODE = GC_ERROR_CODE_e.define('GCEA_CODE', 1)
SQ_CODE = GC_ERROR_CODE_e.define('SQ_CODE', 2)
LDS_CODE = GC_ERROR_CODE_e.define('LDS_CODE', 3)
GDS_CODE = GC_ERROR_CODE_e.define('GDS_CODE', 4)
SP0_CODE = GC_ERROR_CODE_e.define('SP0_CODE', 5)
SP1_CODE = GC_ERROR_CODE_e.define('SP1_CODE', 6)
TCC_CODE = GC_ERROR_CODE_e.define('TCC_CODE', 7)
TCA_CODE = GC_ERROR_CODE_e.define('TCA_CODE', 8)
TCX_CODE = GC_ERROR_CODE_e.define('TCX_CODE', 9)
CPC_CODE = GC_ERROR_CODE_e.define('CPC_CODE', 10)
CPF_CODE = GC_ERROR_CODE_e.define('CPF_CODE', 11)
CPG_CODE = GC_ERROR_CODE_e.define('CPG_CODE', 12)
SPI_CODE = GC_ERROR_CODE_e.define('SPI_CODE', 13)
RLC_CODE = GC_ERROR_CODE_e.define('RLC_CODE', 14)
SQC_CODE = GC_ERROR_CODE_e.define('SQC_CODE', 15)
TA_CODE = GC_ERROR_CODE_e.define('TA_CODE', 16)
TD_CODE = GC_ERROR_CODE_e.define('TD_CODE', 17)
TCP_CODE = GC_ERROR_CODE_e.define('TCP_CODE', 18)
TCI_CODE = GC_ERROR_CODE_e.define('TCI_CODE', 19)
GC_ROUTER_CODE = GC_ERROR_CODE_e.define('GC_ROUTER_CODE', 20)
VML2_CODE = GC_ERROR_CODE_e.define('VML2_CODE', 21)
VML2_WALKER_CODE = GC_ERROR_CODE_e.define('VML2_WALKER_CODE', 22)
ATCL2_CODE = GC_ERROR_CODE_e.define('ATCL2_CODE', 23)
GC_CANE_CODE = GC_ERROR_CODE_e.define('GC_CANE_CODE', 24)
MP5_CODE_SMN_SLVERR = GC_ERROR_CODE_e.define('MP5_CODE_SMN_SLVERR', 40)
MP5_CODE_UNKNOWN = GC_ERROR_CODE_e.define('MP5_CODE_UNKNOWN', 42)

@c.record
class SwI2cCmd_t(c.Struct):
  SIZE = 2
  ReadWriteData: Annotated[uint8_t, 0]
  CmdConfig: Annotated[uint8_t, 1]
@c.record
class SwI2cRequest_t(c.Struct):
  SIZE = 52
  I2CcontrollerPort: Annotated[uint8_t, 0]
  I2CSpeed: Annotated[uint8_t, 1]
  SlaveAddress: Annotated[uint8_t, 2]
  NumCmds: Annotated[uint8_t, 3]
  SwI2cCmds: Annotated[c.Array[SwI2cCmd_t, Literal[24]], 4]
@c.record
class SwI2cRequestExternal_t(c.Struct):
  SIZE = 116
  SwI2cRequest: Annotated[SwI2cRequest_t, 0]
  Spare: Annotated[c.Array[uint32_t, Literal[8]], 52]
  MmHubPadding: Annotated[c.Array[uint32_t, Literal[8]], 84]
class PPCLK_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
PPCLK_VCLK = PPCLK_e.define('PPCLK_VCLK', 0)
PPCLK_DCLK = PPCLK_e.define('PPCLK_DCLK', 1)
PPCLK_SOCCLK = PPCLK_e.define('PPCLK_SOCCLK', 2)
PPCLK_UCLK = PPCLK_e.define('PPCLK_UCLK', 3)
PPCLK_FCLK = PPCLK_e.define('PPCLK_FCLK', 4)
PPCLK_LCLK = PPCLK_e.define('PPCLK_LCLK', 5)
PPCLK_COUNT = PPCLK_e.define('PPCLK_COUNT', 6)

class GpioIntPolarity_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
GPIO_INT_POLARITY_ACTIVE_LOW = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_LOW', 0)
GPIO_INT_POLARITY_ACTIVE_HIGH = GpioIntPolarity_e.define('GPIO_INT_POLARITY_ACTIVE_HIGH', 1)

class UCLK_DPM_MODE_e(Annotated[int, ctypes.c_uint32], c.Enum): pass
UCLK_DPM_MODE_BANDWIDTH = UCLK_DPM_MODE_e.define('UCLK_DPM_MODE_BANDWIDTH', 0)
UCLK_DPM_MODE_LATENCY = UCLK_DPM_MODE_e.define('UCLK_DPM_MODE_LATENCY', 1)

@c.record
class AvfsDebugTableAid_t(c.Struct):
  SIZE = 360
  avgPsmCount: Annotated[c.Array[uint16_t, Literal[30]], 0]
  minPsmCount: Annotated[c.Array[uint16_t, Literal[30]], 60]
  avgPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[30]], 120]
  minPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[30]], 240]
@c.record
class AvfsDebugTableXcd_t(c.Struct):
  SIZE = 360
  avgPsmCount: Annotated[c.Array[uint16_t, Literal[30]], 0]
  minPsmCount: Annotated[c.Array[uint16_t, Literal[30]], 60]
  avgPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[30]], 120]
  minPsmVoltage: Annotated[c.Array[Annotated[float, ctypes.c_float], Literal[30]], 240]
@c.record
class struct_smu_hw_power_state(c.Struct):
  SIZE = 4
  magic: Annotated[Annotated[int, ctypes.c_uint32], 0]
class struct_smu_power_state(ctypes.Structure): pass
class enum_smu_state_ui_label(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_STATE_UI_LABEL_NONE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_NONE', 0)
SMU_STATE_UI_LABEL_BATTERY = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BATTERY', 1)
SMU_STATE_UI_TABEL_MIDDLE_LOW = enum_smu_state_ui_label.define('SMU_STATE_UI_TABEL_MIDDLE_LOW', 2)
SMU_STATE_UI_LABEL_BALLANCED = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BALLANCED', 3)
SMU_STATE_UI_LABEL_MIDDLE_HIGHT = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_MIDDLE_HIGHT', 4)
SMU_STATE_UI_LABEL_PERFORMANCE = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_PERFORMANCE', 5)
SMU_STATE_UI_LABEL_BACO = enum_smu_state_ui_label.define('SMU_STATE_UI_LABEL_BACO', 6)

class enum_smu_state_classification_flag(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_STATE_CLASSIFICATION_FLAG_BOOT = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_BOOT', 1)
SMU_STATE_CLASSIFICATION_FLAG_THERMAL = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_THERMAL', 2)
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE', 4)
SMU_STATE_CLASSIFICATION_FLAG_RESET = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_RESET', 8)
SMU_STATE_CLASSIFICATION_FLAG_FORCED = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_FORCED', 16)
SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE', 32)
SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE', 64)
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE', 128)
SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE', 256)
SMU_STATE_CLASSIFICATION_FLAG_UVD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD', 512)
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW', 1024)
SMU_STATE_CLASSIFICATION_FLAG_ACPI = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_ACPI', 2048)
SMU_STATE_CLASSIFICATION_FLAG_HD2 = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_HD2', 4096)
SMU_STATE_CLASSIFICATION_FLAG_UVD_HD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_HD', 8192)
SMU_STATE_CLASSIFICATION_FLAG_UVD_SD = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_SD', 16384)
SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE', 32768)
SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE', 65536)
SMU_STATE_CLASSIFICATION_FLAG_BACO = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_BACO', 131072)
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2 = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2', 262144)
SMU_STATE_CLASSIFICATION_FLAG_ULV = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_ULV', 524288)
SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC = enum_smu_state_classification_flag.define('SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC', 1048576)

@c.record
class struct_smu_state_classification_block(c.Struct):
  SIZE = 16
  ui_label: Annotated[enum_smu_state_ui_label, 0]
  flags: Annotated[enum_smu_state_classification_flag, 4]
  bios_index: Annotated[Annotated[int, ctypes.c_int32], 8]
  temporary_state: Annotated[Annotated[bool, ctypes.c_bool], 12]
  to_be_deleted: Annotated[Annotated[bool, ctypes.c_bool], 13]
@c.record
class struct_smu_state_pcie_block(c.Struct):
  SIZE = 4
  lanes: Annotated[Annotated[int, ctypes.c_uint32], 0]
class enum_smu_refreshrate_source(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_REFRESHRATE_SOURCE_EDID = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EDID', 0)
SMU_REFRESHRATE_SOURCE_EXPLICIT = enum_smu_refreshrate_source.define('SMU_REFRESHRATE_SOURCE_EXPLICIT', 1)

@c.record
class struct_smu_state_display_block(c.Struct):
  SIZE = 20
  disable_frame_modulation: Annotated[Annotated[bool, ctypes.c_bool], 0]
  limit_refreshrate: Annotated[Annotated[bool, ctypes.c_bool], 1]
  refreshrate_source: Annotated[enum_smu_refreshrate_source, 4]
  explicit_refreshrate: Annotated[Annotated[int, ctypes.c_int32], 8]
  edid_refreshrate_index: Annotated[Annotated[int, ctypes.c_int32], 12]
  enable_vari_bright: Annotated[Annotated[bool, ctypes.c_bool], 16]
@c.record
class struct_smu_state_memory_block(c.Struct):
  SIZE = 5
  dll_off: Annotated[Annotated[bool, ctypes.c_bool], 0]
  m3arb: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  unused: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 2]
@c.record
class struct_smu_state_software_algorithm_block(c.Struct):
  SIZE = 2
  disable_load_balancing: Annotated[Annotated[bool, ctypes.c_bool], 0]
  enable_sleep_for_timestamps: Annotated[Annotated[bool, ctypes.c_bool], 1]
@c.record
class struct_smu_temperature_range(c.Struct):
  SIZE = 44
  min: Annotated[Annotated[int, ctypes.c_int32], 0]
  max: Annotated[Annotated[int, ctypes.c_int32], 4]
  edge_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 8]
  hotspot_min: Annotated[Annotated[int, ctypes.c_int32], 12]
  hotspot_crit_max: Annotated[Annotated[int, ctypes.c_int32], 16]
  hotspot_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 20]
  mem_min: Annotated[Annotated[int, ctypes.c_int32], 24]
  mem_crit_max: Annotated[Annotated[int, ctypes.c_int32], 28]
  mem_emergency_max: Annotated[Annotated[int, ctypes.c_int32], 32]
  software_shutdown_temp: Annotated[Annotated[int, ctypes.c_int32], 36]
  software_shutdown_temp_offset: Annotated[Annotated[int, ctypes.c_int32], 40]
@c.record
class struct_smu_state_validation_block(c.Struct):
  SIZE = 3
  single_display_only: Annotated[Annotated[bool, ctypes.c_bool], 0]
  disallow_on_dc: Annotated[Annotated[bool, ctypes.c_bool], 1]
  supported_power_levels: Annotated[Annotated[int, ctypes.c_ubyte], 2]
@c.record
class struct_smu_uvd_clocks(c.Struct):
  SIZE = 8
  vclk: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dclk: Annotated[Annotated[int, ctypes.c_uint32], 4]
class enum_smu_power_src_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_POWER_SOURCE_AC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_AC', 0)
SMU_POWER_SOURCE_DC = enum_smu_power_src_type.define('SMU_POWER_SOURCE_DC', 1)
SMU_POWER_SOURCE_COUNT = enum_smu_power_src_type.define('SMU_POWER_SOURCE_COUNT', 2)

class enum_smu_ppt_limit_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_DEFAULT_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_DEFAULT_PPT_LIMIT', 0)
SMU_FAST_PPT_LIMIT = enum_smu_ppt_limit_type.define('SMU_FAST_PPT_LIMIT', 1)

class enum_smu_ppt_limit_level(Annotated[int, ctypes.c_int32], c.Enum): pass
SMU_PPT_LIMIT_MIN = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MIN', -1)
SMU_PPT_LIMIT_CURRENT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_CURRENT', 0)
SMU_PPT_LIMIT_DEFAULT = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_DEFAULT', 1)
SMU_PPT_LIMIT_MAX = enum_smu_ppt_limit_level.define('SMU_PPT_LIMIT_MAX', 2)

class enum_smu_memory_pool_size(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_MEMORY_POOL_SIZE_ZERO = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_ZERO', 0)
SMU_MEMORY_POOL_SIZE_256_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_256_MB', 268435456)
SMU_MEMORY_POOL_SIZE_512_MB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_512_MB', 536870912)
SMU_MEMORY_POOL_SIZE_1_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_1_GB', 1073741824)
SMU_MEMORY_POOL_SIZE_2_GB = enum_smu_memory_pool_size.define('SMU_MEMORY_POOL_SIZE_2_GB', 2147483648)

class enum_smu_clk_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_GFXCLK = enum_smu_clk_type.define('SMU_GFXCLK', 0)
SMU_VCLK = enum_smu_clk_type.define('SMU_VCLK', 1)
SMU_DCLK = enum_smu_clk_type.define('SMU_DCLK', 2)
SMU_VCLK1 = enum_smu_clk_type.define('SMU_VCLK1', 3)
SMU_DCLK1 = enum_smu_clk_type.define('SMU_DCLK1', 4)
SMU_ECLK = enum_smu_clk_type.define('SMU_ECLK', 5)
SMU_SOCCLK = enum_smu_clk_type.define('SMU_SOCCLK', 6)
SMU_UCLK = enum_smu_clk_type.define('SMU_UCLK', 7)
SMU_DCEFCLK = enum_smu_clk_type.define('SMU_DCEFCLK', 8)
SMU_DISPCLK = enum_smu_clk_type.define('SMU_DISPCLK', 9)
SMU_PIXCLK = enum_smu_clk_type.define('SMU_PIXCLK', 10)
SMU_PHYCLK = enum_smu_clk_type.define('SMU_PHYCLK', 11)
SMU_FCLK = enum_smu_clk_type.define('SMU_FCLK', 12)
SMU_SCLK = enum_smu_clk_type.define('SMU_SCLK', 13)
SMU_MCLK = enum_smu_clk_type.define('SMU_MCLK', 14)
SMU_PCIE = enum_smu_clk_type.define('SMU_PCIE', 15)
SMU_LCLK = enum_smu_clk_type.define('SMU_LCLK', 16)
SMU_OD_CCLK = enum_smu_clk_type.define('SMU_OD_CCLK', 17)
SMU_OD_SCLK = enum_smu_clk_type.define('SMU_OD_SCLK', 18)
SMU_OD_MCLK = enum_smu_clk_type.define('SMU_OD_MCLK', 19)
SMU_OD_VDDC_CURVE = enum_smu_clk_type.define('SMU_OD_VDDC_CURVE', 20)
SMU_OD_RANGE = enum_smu_clk_type.define('SMU_OD_RANGE', 21)
SMU_OD_VDDGFX_OFFSET = enum_smu_clk_type.define('SMU_OD_VDDGFX_OFFSET', 22)
SMU_OD_FAN_CURVE = enum_smu_clk_type.define('SMU_OD_FAN_CURVE', 23)
SMU_OD_ACOUSTIC_LIMIT = enum_smu_clk_type.define('SMU_OD_ACOUSTIC_LIMIT', 24)
SMU_OD_ACOUSTIC_TARGET = enum_smu_clk_type.define('SMU_OD_ACOUSTIC_TARGET', 25)
SMU_OD_FAN_TARGET_TEMPERATURE = enum_smu_clk_type.define('SMU_OD_FAN_TARGET_TEMPERATURE', 26)
SMU_OD_FAN_MINIMUM_PWM = enum_smu_clk_type.define('SMU_OD_FAN_MINIMUM_PWM', 27)
SMU_CLK_COUNT = enum_smu_clk_type.define('SMU_CLK_COUNT', 28)

@c.record
class struct_smu_user_dpm_profile(c.Struct):
  SIZE = 140
  fan_mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  power_limit: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fan_speed_pwm: Annotated[Annotated[int, ctypes.c_uint32], 8]
  fan_speed_rpm: Annotated[Annotated[int, ctypes.c_uint32], 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  user_od: Annotated[Annotated[int, ctypes.c_uint32], 20]
  clk_mask: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[28]], 24]
  clk_dependency: Annotated[Annotated[int, ctypes.c_uint32], 136]
@c.record
class struct_smu_table(c.Struct):
  SIZE = 48
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  align: Annotated[Annotated[int, ctypes.c_uint32], 8]
  domain: Annotated[Annotated[int, ctypes.c_ubyte], 12]
  mc_address: Annotated[Annotated[int, ctypes.c_uint64], 16]
  cpu_addr: Annotated[ctypes.c_void_p, 24]
  bo: Annotated[c.POINTER[struct_amdgpu_bo], 32]
  version: Annotated[Annotated[int, ctypes.c_uint32], 40]
class struct_amdgpu_bo(ctypes.Structure): pass
class enum_smu_perf_level_designation(Annotated[int, ctypes.c_uint32], c.Enum): pass
PERF_LEVEL_ACTIVITY = enum_smu_perf_level_designation.define('PERF_LEVEL_ACTIVITY', 0)
PERF_LEVEL_POWER_CONTAINMENT = enum_smu_perf_level_designation.define('PERF_LEVEL_POWER_CONTAINMENT', 1)

@c.record
class struct_smu_performance_level(c.Struct):
  SIZE = 24
  core_clock: Annotated[Annotated[int, ctypes.c_uint32], 0]
  memory_clock: Annotated[Annotated[int, ctypes.c_uint32], 4]
  vddc: Annotated[Annotated[int, ctypes.c_uint32], 8]
  vddci: Annotated[Annotated[int, ctypes.c_uint32], 12]
  non_local_mem_freq: Annotated[Annotated[int, ctypes.c_uint32], 16]
  non_local_mem_width: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_smu_clock_info(c.Struct):
  SIZE = 24
  min_mem_clk: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_mem_clk: Annotated[Annotated[int, ctypes.c_uint32], 4]
  min_eng_clk: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_eng_clk: Annotated[Annotated[int, ctypes.c_uint32], 12]
  min_bus_bandwidth: Annotated[Annotated[int, ctypes.c_uint32], 16]
  max_bus_bandwidth: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_smu_bios_boot_up_values(c.Struct):
  SIZE = 68
  revision: Annotated[Annotated[int, ctypes.c_uint32], 0]
  gfxclk: Annotated[Annotated[int, ctypes.c_uint32], 4]
  uclk: Annotated[Annotated[int, ctypes.c_uint32], 8]
  socclk: Annotated[Annotated[int, ctypes.c_uint32], 12]
  dcefclk: Annotated[Annotated[int, ctypes.c_uint32], 16]
  eclk: Annotated[Annotated[int, ctypes.c_uint32], 20]
  vclk: Annotated[Annotated[int, ctypes.c_uint32], 24]
  dclk: Annotated[Annotated[int, ctypes.c_uint32], 28]
  vddc: Annotated[Annotated[int, ctypes.c_uint16], 32]
  vddci: Annotated[Annotated[int, ctypes.c_uint16], 34]
  mvddc: Annotated[Annotated[int, ctypes.c_uint16], 36]
  vdd_gfx: Annotated[Annotated[int, ctypes.c_uint16], 38]
  cooling_id: Annotated[Annotated[int, ctypes.c_ubyte], 40]
  pp_table_id: Annotated[Annotated[int, ctypes.c_uint32], 44]
  format_revision: Annotated[Annotated[int, ctypes.c_uint32], 48]
  content_revision: Annotated[Annotated[int, ctypes.c_uint32], 52]
  fclk: Annotated[Annotated[int, ctypes.c_uint32], 56]
  lclk: Annotated[Annotated[int, ctypes.c_uint32], 60]
  firmware_caps: Annotated[Annotated[int, ctypes.c_uint32], 64]
class enum_smu_table_id(Annotated[int, ctypes.c_uint32], c.Enum): pass
SMU_TABLE_PPTABLE = enum_smu_table_id.define('SMU_TABLE_PPTABLE', 0)
SMU_TABLE_WATERMARKS = enum_smu_table_id.define('SMU_TABLE_WATERMARKS', 1)
SMU_TABLE_CUSTOM_DPM = enum_smu_table_id.define('SMU_TABLE_CUSTOM_DPM', 2)
SMU_TABLE_DPMCLOCKS = enum_smu_table_id.define('SMU_TABLE_DPMCLOCKS', 3)
SMU_TABLE_AVFS = enum_smu_table_id.define('SMU_TABLE_AVFS', 4)
SMU_TABLE_AVFS_PSM_DEBUG = enum_smu_table_id.define('SMU_TABLE_AVFS_PSM_DEBUG', 5)
SMU_TABLE_AVFS_FUSE_OVERRIDE = enum_smu_table_id.define('SMU_TABLE_AVFS_FUSE_OVERRIDE', 6)
SMU_TABLE_PMSTATUSLOG = enum_smu_table_id.define('SMU_TABLE_PMSTATUSLOG', 7)
SMU_TABLE_SMU_METRICS = enum_smu_table_id.define('SMU_TABLE_SMU_METRICS', 8)
SMU_TABLE_DRIVER_SMU_CONFIG = enum_smu_table_id.define('SMU_TABLE_DRIVER_SMU_CONFIG', 9)
SMU_TABLE_ACTIVITY_MONITOR_COEFF = enum_smu_table_id.define('SMU_TABLE_ACTIVITY_MONITOR_COEFF', 10)
SMU_TABLE_OVERDRIVE = enum_smu_table_id.define('SMU_TABLE_OVERDRIVE', 11)
SMU_TABLE_I2C_COMMANDS = enum_smu_table_id.define('SMU_TABLE_I2C_COMMANDS', 12)
SMU_TABLE_PACE = enum_smu_table_id.define('SMU_TABLE_PACE', 13)
SMU_TABLE_ECCINFO = enum_smu_table_id.define('SMU_TABLE_ECCINFO', 14)
SMU_TABLE_COMBO_PPTABLE = enum_smu_table_id.define('SMU_TABLE_COMBO_PPTABLE', 15)
SMU_TABLE_WIFIBAND = enum_smu_table_id.define('SMU_TABLE_WIFIBAND', 16)
SMU_TABLE_COUNT = enum_smu_table_id.define('SMU_TABLE_COUNT', 17)

c.init_records()
PPSMC_Result_OK = 0x1 # type: ignore
PPSMC_Result_Failed = 0xFF # type: ignore
PPSMC_Result_UnknownCmd = 0xFE # type: ignore
PPSMC_Result_CmdRejectedPrereq = 0xFD # type: ignore
PPSMC_Result_CmdRejectedBusy = 0xFC # type: ignore
PPSMC_MSG_TestMessage = 0x1 # type: ignore
PPSMC_MSG_GetSmuVersion = 0x2 # type: ignore
PPSMC_MSG_GfxDriverReset = 0x3 # type: ignore
PPSMC_MSG_GetDriverIfVersion = 0x4 # type: ignore
PPSMC_MSG_EnableAllSmuFeatures = 0x5 # type: ignore
PPSMC_MSG_DisableAllSmuFeatures = 0x6 # type: ignore
PPSMC_MSG_RequestI2cTransaction = 0x7 # type: ignore
PPSMC_MSG_GetMetricsVersion = 0x8 # type: ignore
PPSMC_MSG_GetMetricsTable = 0x9 # type: ignore
PPSMC_MSG_GetEccInfoTable = 0xA # type: ignore
PPSMC_MSG_GetEnabledSmuFeaturesLow = 0xB # type: ignore
PPSMC_MSG_GetEnabledSmuFeaturesHigh = 0xC # type: ignore
PPSMC_MSG_SetDriverDramAddrHigh = 0xD # type: ignore
PPSMC_MSG_SetDriverDramAddrLow = 0xE # type: ignore
PPSMC_MSG_SetToolsDramAddrHigh = 0xF # type: ignore
PPSMC_MSG_SetToolsDramAddrLow = 0x10 # type: ignore
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x11 # type: ignore
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x12 # type: ignore
PPSMC_MSG_SetSoftMinByFreq = 0x13 # type: ignore
PPSMC_MSG_SetSoftMaxByFreq = 0x14 # type: ignore
PPSMC_MSG_GetMinDpmFreq = 0x15 # type: ignore
PPSMC_MSG_GetMaxDpmFreq = 0x16 # type: ignore
PPSMC_MSG_GetDpmFreqByIndex = 0x17 # type: ignore
PPSMC_MSG_SetPptLimit = 0x18 # type: ignore
PPSMC_MSG_GetPptLimit = 0x19 # type: ignore
PPSMC_MSG_DramLogSetDramAddrHigh = 0x1A # type: ignore
PPSMC_MSG_DramLogSetDramAddrLow = 0x1B # type: ignore
PPSMC_MSG_DramLogSetDramSize = 0x1C # type: ignore
PPSMC_MSG_GetDebugData = 0x1D # type: ignore
PPSMC_MSG_HeavySBR = 0x1E # type: ignore
PPSMC_MSG_SetNumBadHbmPagesRetired = 0x1F # type: ignore
PPSMC_MSG_DFCstateControl = 0x20 # type: ignore
PPSMC_MSG_GetGmiPwrDnHyst = 0x21 # type: ignore
PPSMC_MSG_SetGmiPwrDnHyst = 0x22 # type: ignore
PPSMC_MSG_GmiPwrDnControl = 0x23 # type: ignore
PPSMC_MSG_EnterGfxoff = 0x24 # type: ignore
PPSMC_MSG_ExitGfxoff = 0x25 # type: ignore
PPSMC_MSG_EnableDeterminism = 0x26 # type: ignore
PPSMC_MSG_DisableDeterminism = 0x27 # type: ignore
PPSMC_MSG_DumpSTBtoDram = 0x28 # type: ignore
PPSMC_MSG_STBtoDramLogSetDramAddrHigh = 0x29 # type: ignore
PPSMC_MSG_STBtoDramLogSetDramAddrLow = 0x2A # type: ignore
PPSMC_MSG_STBtoDramLogSetDramSize = 0x2B # type: ignore
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrHigh = 0x2C # type: ignore
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrLow = 0x2D # type: ignore
PPSMC_MSG_GfxDriverResetRecovery = 0x2E # type: ignore
PPSMC_MSG_TriggerVFFLR = 0x2F # type: ignore
PPSMC_MSG_SetSoftMinGfxClk = 0x30 # type: ignore
PPSMC_MSG_SetSoftMaxGfxClk = 0x31 # type: ignore
PPSMC_MSG_GetMinGfxDpmFreq = 0x32 # type: ignore
PPSMC_MSG_GetMaxGfxDpmFreq = 0x33 # type: ignore
PPSMC_MSG_PrepareForDriverUnload = 0x34 # type: ignore
PPSMC_MSG_ReadThrottlerLimit = 0x35 # type: ignore
PPSMC_MSG_QueryValidMcaCount = 0x36 # type: ignore
PPSMC_MSG_McaBankDumpDW = 0x37 # type: ignore
PPSMC_MSG_GetCTFLimit = 0x38 # type: ignore
PPSMC_MSG_ClearMcaOnRead = 0x39 # type: ignore
PPSMC_MSG_QueryValidMcaCeCount = 0x3A # type: ignore
PPSMC_MSG_McaBankCeDumpDW = 0x3B # type: ignore
PPSMC_MSG_SelectPLPDMode = 0x40 # type: ignore
PPSMC_MSG_PmLogReadSample = 0x41 # type: ignore
PPSMC_MSG_PmLogGetTableVersion = 0x42 # type: ignore
PPSMC_MSG_RmaDueToBadPageThreshold = 0x43 # type: ignore
PPSMC_MSG_SetThrottlingPolicy = 0x44 # type: ignore
PPSMC_MSG_SetPhaseDetectCSBWThreshold = 0x45 # type: ignore
PPSMC_MSG_SetPhaseDetectFreqHigh = 0x46 # type: ignore
PPSMC_MSG_SetPhaseDetectFreqLow = 0x47 # type: ignore
PPSMC_MSG_SetPhaseDetectDownHysterisis = 0x48 # type: ignore
PPSMC_MSG_SetPhaseDetectAlphaX1e6 = 0x49 # type: ignore
PPSMC_MSG_SetPhaseDetectOnOff = 0x4A # type: ignore
PPSMC_MSG_GetPhaseDetectResidency = 0x4B # type: ignore
PPSMC_MSG_UpdatePccWaitDecMaxStr = 0x4C # type: ignore
PPSMC_MSG_ResetSDMA = 0x4D # type: ignore
PPSMC_MSG_GetRasTableVersion = 0x4E # type: ignore
PPSMC_MSG_GetBadPageCount = 0x50 # type: ignore
PPSMC_MSG_GetBadPageMcaAddress = 0x51 # type: ignore
PPSMC_MSG_SetTimestamp = 0x53 # type: ignore
PPSMC_MSG_SetTimestampHi = 0x54 # type: ignore
PPSMC_MSG_GetTimestamp = 0x55 # type: ignore
PPSMC_MSG_GetBadPageIpIdLoHi = 0x57 # type: ignore
PPSMC_MSG_EraseRasTable = 0x58 # type: ignore
PPSMC_MSG_GetStaticMetricsTable = 0x59 # type: ignore
PPSMC_MSG_ResetVfArbitersByIndex = 0x5A # type: ignore
PPSMC_MSG_GetSystemMetricsTable = 0x5C # type: ignore
PPSMC_MSG_GetSystemMetricsVersion = 0x5D # type: ignore
PPSMC_MSG_ResetVCN = 0x5E # type: ignore
PPSMC_MSG_SetFastPptLimit = 0x5F # type: ignore
PPSMC_MSG_GetFastPptLimit = 0x60 # type: ignore
PPSMC_Message_Count = 0x61 # type: ignore
PPSMC_RESET_TYPE_DRIVER_MODE_1_RESET = 0x1 # type: ignore
PPSMC_RESET_TYPE_DRIVER_MODE_2_RESET = 0x2 # type: ignore
PPSMC_RESET_TYPE_DRIVER_MODE_3_RESET = 0x3 # type: ignore
PPSMC_THROTTLING_LIMIT_TYPE_SOCKET = 0x1 # type: ignore
PPSMC_THROTTLING_LIMIT_TYPE_HBM = 0x2 # type: ignore
PPSMC_AID_THM_TYPE = 0x1 # type: ignore
PPSMC_CCD_THM_TYPE = 0x2 # type: ignore
PPSMC_XCD_THM_TYPE = 0x3 # type: ignore
PPSMC_HBM_THM_TYPE = 0x4 # type: ignore
PPSMC_PLPD_MODE_DEFAULT = 0x1 # type: ignore
PPSMC_PLPD_MODE_OPTIMIZED = 0x2 # type: ignore
NUM_VCLK_DPM_LEVELS = 4 # type: ignore
NUM_DCLK_DPM_LEVELS = 4 # type: ignore
NUM_SOCCLK_DPM_LEVELS = 4 # type: ignore
NUM_LCLK_DPM_LEVELS = 4 # type: ignore
NUM_UCLK_DPM_LEVELS = 4 # type: ignore
NUM_FCLK_DPM_LEVELS = 4 # type: ignore
NUM_XGMI_DPM_LEVELS = 2 # type: ignore
NUM_CXL_BITRATES = 4 # type: ignore
NUM_PCIE_BITRATES = 4 # type: ignore
NUM_XGMI_BITRATES = 4 # type: ignore
NUM_XGMI_WIDTHS = 3 # type: ignore
NUM_TDP_GROUPS = 4 # type: ignore
NUM_SOC_P2S_TABLES = 6 # type: ignore
NUM_GFX_P2S_TABLES = 8 # type: ignore
NUM_PSM_DIDT_THRESHOLDS = 3 # type: ignore
NUM_XVMIN_VMIN_THRESHOLDS = 3 # type: ignore
PRODUCT_MODEL_NUMBER_LEN = 20 # type: ignore
PRODUCT_NAME_LEN = 64 # type: ignore
PRODUCT_SERIAL_LEN = 20 # type: ignore
PRODUCT_MANUFACTURER_NAME_LEN = 32 # type: ignore
PRODUCT_FRU_ID_LEN = 32 # type: ignore
SMU_METRICS_TABLE_VERSION = 0x15 # type: ignore
SMU_SYSTEM_METRICS_TABLE_VERSION = 0x1 # type: ignore
SMU_VF_METRICS_TABLE_MASK = (1 << 31) # type: ignore
SMU_VF_METRICS_TABLE_VERSION = (0x6 | SMU_VF_METRICS_TABLE_MASK) # type: ignore
SMU13_0_6_DRIVER_IF_VERSION = 0x08042024 # type: ignore
NUM_I2C_CONTROLLERS = 8 # type: ignore
I2C_CONTROLLER_ENABLED = 1 # type: ignore
I2C_CONTROLLER_DISABLED = 0 # type: ignore
MAX_SW_I2C_COMMANDS = 24 # type: ignore
CMDCONFIG_STOP_BIT = 0 # type: ignore
CMDCONFIG_RESTART_BIT = 1 # type: ignore
CMDCONFIG_READWRITE_BIT = 2 # type: ignore
CMDCONFIG_STOP_MASK = (1 << CMDCONFIG_STOP_BIT) # type: ignore
CMDCONFIG_RESTART_MASK = (1 << CMDCONFIG_RESTART_BIT) # type: ignore
CMDCONFIG_READWRITE_MASK = (1 << CMDCONFIG_READWRITE_BIT) # type: ignore
IH_INTERRUPT_ID_TO_DRIVER = 0xFE # type: ignore
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7 # type: ignore
THROTTLER_PROCHOT_BIT = 0 # type: ignore
THROTTLER_PPT_BIT = 1 # type: ignore
THROTTLER_THERMAL_SOCKET_BIT = 2 # type: ignore
THROTTLER_THERMAL_VR_BIT = 3 # type: ignore
THROTTLER_THERMAL_HBM_BIT = 4 # type: ignore
ClearMcaOnRead_UE_FLAG_MASK = 0x1 # type: ignore
ClearMcaOnRead_CE_POLL_MASK = 0x2 # type: ignore
int32_t = int # type: ignore
SMU_THERMAL_MINIMUM_ALERT_TEMP = 0 # type: ignore
SMU_THERMAL_MAXIMUM_ALERT_TEMP = 255 # type: ignore
SMU_TEMPERATURE_UNITS_PER_CENTIGRADES = 1000 # type: ignore
SMU_FW_NAME_LEN = 0x24 # type: ignore
SMU_DPM_USER_PROFILE_RESTORE = (1 << 0) # type: ignore
SMU_CUSTOM_FAN_SPEED_RPM = (1 << 1) # type: ignore
SMU_CUSTOM_FAN_SPEED_PWM = (1 << 2) # type: ignore
SMU_THROTTLER_PPT0_BIT = 0 # type: ignore
SMU_THROTTLER_PPT1_BIT = 1 # type: ignore
SMU_THROTTLER_PPT2_BIT = 2 # type: ignore
SMU_THROTTLER_PPT3_BIT = 3 # type: ignore
SMU_THROTTLER_SPL_BIT = 4 # type: ignore
SMU_THROTTLER_FPPT_BIT = 5 # type: ignore
SMU_THROTTLER_SPPT_BIT = 6 # type: ignore
SMU_THROTTLER_SPPT_APU_BIT = 7 # type: ignore
SMU_THROTTLER_TDC_GFX_BIT = 16 # type: ignore
SMU_THROTTLER_TDC_SOC_BIT = 17 # type: ignore
SMU_THROTTLER_TDC_MEM_BIT = 18 # type: ignore
SMU_THROTTLER_TDC_VDD_BIT = 19 # type: ignore
SMU_THROTTLER_TDC_CVIP_BIT = 20 # type: ignore
SMU_THROTTLER_EDC_CPU_BIT = 21 # type: ignore
SMU_THROTTLER_EDC_GFX_BIT = 22 # type: ignore
SMU_THROTTLER_APCC_BIT = 23 # type: ignore
SMU_THROTTLER_TEMP_GPU_BIT = 32 # type: ignore
SMU_THROTTLER_TEMP_CORE_BIT = 33 # type: ignore
SMU_THROTTLER_TEMP_MEM_BIT = 34 # type: ignore
SMU_THROTTLER_TEMP_EDGE_BIT = 35 # type: ignore
SMU_THROTTLER_TEMP_HOTSPOT_BIT = 36 # type: ignore
SMU_THROTTLER_TEMP_SOC_BIT = 37 # type: ignore
SMU_THROTTLER_TEMP_VR_GFX_BIT = 38 # type: ignore
SMU_THROTTLER_TEMP_VR_SOC_BIT = 39 # type: ignore
SMU_THROTTLER_TEMP_VR_MEM0_BIT = 40 # type: ignore
SMU_THROTTLER_TEMP_VR_MEM1_BIT = 41 # type: ignore
SMU_THROTTLER_TEMP_LIQUID0_BIT = 42 # type: ignore
SMU_THROTTLER_TEMP_LIQUID1_BIT = 43 # type: ignore
SMU_THROTTLER_VRHOT0_BIT = 44 # type: ignore
SMU_THROTTLER_VRHOT1_BIT = 45 # type: ignore
SMU_THROTTLER_PROCHOT_CPU_BIT = 46 # type: ignore
SMU_THROTTLER_PROCHOT_GFX_BIT = 47 # type: ignore
SMU_THROTTLER_PPM_BIT = 56 # type: ignore
SMU_THROTTLER_FIT_BIT = 57 # type: ignore