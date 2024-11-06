# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes




SMU_V13_0_0_PPSMC_H = True # macro
PPSMC_VERSION = 0x1 # macro
DEBUGSMC_VERSION = 0x1 # macro
PPSMC_Result_OK = 0x1 # macro
PPSMC_Result_Failed = 0xFF # macro
PPSMC_Result_UnknownCmd = 0xFE # macro
PPSMC_Result_CmdRejectedPrereq = 0xFD # macro
PPSMC_Result_CmdRejectedBusy = 0xFC # macro
PPSMC_MSG_TestMessage = 0x1 # macro
PPSMC_MSG_GetSmuVersion = 0x2 # macro
PPSMC_MSG_GetDriverIfVersion = 0x3 # macro
PPSMC_MSG_SetAllowedFeaturesMaskLow = 0x4 # macro
PPSMC_MSG_SetAllowedFeaturesMaskHigh = 0x5 # macro
PPSMC_MSG_EnableAllSmuFeatures = 0x6 # macro
PPSMC_MSG_DisableAllSmuFeatures = 0x7 # macro
PPSMC_MSG_EnableSmuFeaturesLow = 0x8 # macro
PPSMC_MSG_EnableSmuFeaturesHigh = 0x9 # macro
PPSMC_MSG_DisableSmuFeaturesLow = 0xA # macro
PPSMC_MSG_DisableSmuFeaturesHigh = 0xB # macro
PPSMC_MSG_GetRunningSmuFeaturesLow = 0xC # macro
PPSMC_MSG_GetRunningSmuFeaturesHigh = 0xD # macro
PPSMC_MSG_SetDriverDramAddrHigh = 0xE # macro
PPSMC_MSG_SetDriverDramAddrLow = 0xF # macro
PPSMC_MSG_SetToolsDramAddrHigh = 0x10 # macro
PPSMC_MSG_SetToolsDramAddrLow = 0x11 # macro
PPSMC_MSG_TransferTableSmu2Dram = 0x12 # macro
PPSMC_MSG_TransferTableDram2Smu = 0x13 # macro
PPSMC_MSG_UseDefaultPPTable = 0x14 # macro
PPSMC_MSG_EnterBaco = 0x15 # macro
PPSMC_MSG_ExitBaco = 0x16 # macro
PPSMC_MSG_ArmD3 = 0x17 # macro
PPSMC_MSG_BacoAudioD3PME = 0x18 # macro
PPSMC_MSG_SetSoftMinByFreq = 0x19 # macro
PPSMC_MSG_SetSoftMaxByFreq = 0x1A # macro
PPSMC_MSG_SetHardMinByFreq = 0x1B # macro
PPSMC_MSG_SetHardMaxByFreq = 0x1C # macro
PPSMC_MSG_GetMinDpmFreq = 0x1D # macro
PPSMC_MSG_GetMaxDpmFreq = 0x1E # macro
PPSMC_MSG_GetDpmFreqByIndex = 0x1F # macro
PPSMC_MSG_OverridePcieParameters = 0x20 # macro
PPSMC_MSG_DramLogSetDramAddrHigh = 0x21 # macro
PPSMC_MSG_DramLogSetDramAddrLow = 0x22 # macro
PPSMC_MSG_DramLogSetDramSize = 0x23 # macro
PPSMC_MSG_SetWorkloadMask = 0x24 # macro
PPSMC_MSG_GetVoltageByDpm = 0x25 # macro
PPSMC_MSG_SetVideoFps = 0x26 # macro
PPSMC_MSG_GetDcModeMaxDpmFreq = 0x27 # macro
PPSMC_MSG_AllowGfxOff = 0x28 # macro
PPSMC_MSG_DisallowGfxOff = 0x29 # macro
PPSMC_MSG_PowerUpVcn = 0x2A # macro
PPSMC_MSG_PowerDownVcn = 0x2B # macro
PPSMC_MSG_PowerUpJpeg = 0x2C # macro
PPSMC_MSG_PowerDownJpeg = 0x2D # macro
PPSMC_MSG_PrepareMp1ForUnload = 0x2E # macro
PPSMC_MSG_Mode1Reset = 0x2F # macro
PPSMC_MSG_Mode2Reset = 0x4F # macro
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x30 # macro
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x31 # macro
PPSMC_MSG_SetPptLimit = 0x32 # macro
PPSMC_MSG_GetPptLimit = 0x33 # macro
PPSMC_MSG_ReenableAcDcInterrupt = 0x34 # macro
PPSMC_MSG_NotifyPowerSource = 0x35 # macro
PPSMC_MSG_RunDcBtc = 0x36 # macro
PPSMC_MSG_GetDebugData = 0x37 # macro
PPSMC_MSG_SetTemperatureInputSelect = 0x38 # macro
PPSMC_MSG_SetFwDstatesMask = 0x39 # macro
PPSMC_MSG_SetThrottlerMask = 0x3A # macro
PPSMC_MSG_SetExternalClientDfCstateAllow = 0x3B # macro
PPSMC_MSG_SetMGpuFanBoostLimitRpm = 0x3C # macro
PPSMC_MSG_DumpSTBtoDram = 0x3D # macro
PPSMC_MSG_STBtoDramLogSetDramAddrHigh = 0x3E # macro
PPSMC_MSG_STBtoDramLogSetDramAddrLow = 0x3F # macro
PPSMC_MSG_STBtoDramLogSetDramSize = 0x40 # macro
PPSMC_MSG_SetGpoAllow = 0x41 # macro
PPSMC_MSG_AllowGfxDcs = 0x42 # macro
PPSMC_MSG_DisallowGfxDcs = 0x43 # macro
PPSMC_MSG_EnableAudioStutterWA = 0x44 # macro
PPSMC_MSG_PowerUpUmsch = 0x45 # macro
PPSMC_MSG_PowerDownUmsch = 0x46 # macro
PPSMC_MSG_SetDcsArch = 0x47 # macro
PPSMC_MSG_TriggerVFFLR = 0x48 # macro
PPSMC_MSG_SetNumBadMemoryPagesRetired = 0x49 # macro
PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel = 0x4A # macro
PPSMC_MSG_SetPriorityDeltaGain = 0x4B # macro
PPSMC_MSG_AllowIHHostInterrupt = 0x4C # macro
PPSMC_MSG_DALNotPresent = 0x4E # macro
PPSMC_MSG_EnableUCLKShadow = 0x51 # macro
PPSMC_Message_Count = 0x52 # macro
DEBUGSMC_MSG_TestMessage = 0x1 # macro
DEBUGSMC_MSG_GetDebugData = 0x2 # macro
DEBUGSMC_MSG_DebugDumpExit = 0x3 # macro
DEBUGSMC_Message_Count = 0x4 # macro
__all__ = \
    ['DEBUGSMC_MSG_DebugDumpExit', 'DEBUGSMC_MSG_GetDebugData',
    'DEBUGSMC_MSG_TestMessage', 'DEBUGSMC_Message_Count',
    'DEBUGSMC_VERSION', 'PPSMC_MSG_AllowGfxDcs',
    'PPSMC_MSG_AllowGfxOff', 'PPSMC_MSG_AllowIHHostInterrupt',
    'PPSMC_MSG_ArmD3', 'PPSMC_MSG_BacoAudioD3PME',
    'PPSMC_MSG_DALNotPresent', 'PPSMC_MSG_DisableAllSmuFeatures',
    'PPSMC_MSG_DisableSmuFeaturesHigh',
    'PPSMC_MSG_DisableSmuFeaturesLow', 'PPSMC_MSG_DisallowGfxDcs',
    'PPSMC_MSG_DisallowGfxOff', 'PPSMC_MSG_DramLogSetDramAddrHigh',
    'PPSMC_MSG_DramLogSetDramAddrLow', 'PPSMC_MSG_DramLogSetDramSize',
    'PPSMC_MSG_DumpSTBtoDram', 'PPSMC_MSG_EnableAllSmuFeatures',
    'PPSMC_MSG_EnableAudioStutterWA',
    'PPSMC_MSG_EnableSmuFeaturesHigh',
    'PPSMC_MSG_EnableSmuFeaturesLow', 'PPSMC_MSG_EnableUCLKShadow',
    'PPSMC_MSG_EnterBaco', 'PPSMC_MSG_ExitBaco',
    'PPSMC_MSG_GetDcModeMaxDpmFreq', 'PPSMC_MSG_GetDebugData',
    'PPSMC_MSG_GetDpmFreqByIndex', 'PPSMC_MSG_GetDriverIfVersion',
    'PPSMC_MSG_GetMaxDpmFreq', 'PPSMC_MSG_GetMinDpmFreq',
    'PPSMC_MSG_GetPptLimit', 'PPSMC_MSG_GetRunningSmuFeaturesHigh',
    'PPSMC_MSG_GetRunningSmuFeaturesLow', 'PPSMC_MSG_GetSmuVersion',
    'PPSMC_MSG_GetVoltageByDpm', 'PPSMC_MSG_Mode1Reset',
    'PPSMC_MSG_Mode2Reset', 'PPSMC_MSG_NotifyPowerSource',
    'PPSMC_MSG_OverridePcieParameters', 'PPSMC_MSG_PowerDownJpeg',
    'PPSMC_MSG_PowerDownUmsch', 'PPSMC_MSG_PowerDownVcn',
    'PPSMC_MSG_PowerUpJpeg', 'PPSMC_MSG_PowerUpUmsch',
    'PPSMC_MSG_PowerUpVcn', 'PPSMC_MSG_PrepareMp1ForUnload',
    'PPSMC_MSG_ReenableAcDcInterrupt', 'PPSMC_MSG_RunDcBtc',
    'PPSMC_MSG_STBtoDramLogSetDramAddrHigh',
    'PPSMC_MSG_STBtoDramLogSetDramAddrLow',
    'PPSMC_MSG_STBtoDramLogSetDramSize',
    'PPSMC_MSG_SetAllowedFeaturesMaskHigh',
    'PPSMC_MSG_SetAllowedFeaturesMaskLow',
    'PPSMC_MSG_SetBadMemoryPagesRetiredFlagsPerChannel',
    'PPSMC_MSG_SetDcsArch', 'PPSMC_MSG_SetDriverDramAddrHigh',
    'PPSMC_MSG_SetDriverDramAddrLow',
    'PPSMC_MSG_SetExternalClientDfCstateAllow',
    'PPSMC_MSG_SetFwDstatesMask', 'PPSMC_MSG_SetGpoAllow',
    'PPSMC_MSG_SetHardMaxByFreq', 'PPSMC_MSG_SetHardMinByFreq',
    'PPSMC_MSG_SetMGpuFanBoostLimitRpm',
    'PPSMC_MSG_SetNumBadMemoryPagesRetired', 'PPSMC_MSG_SetPptLimit',
    'PPSMC_MSG_SetPriorityDeltaGain', 'PPSMC_MSG_SetSoftMaxByFreq',
    'PPSMC_MSG_SetSoftMinByFreq',
    'PPSMC_MSG_SetSystemVirtualDramAddrHigh',
    'PPSMC_MSG_SetSystemVirtualDramAddrLow',
    'PPSMC_MSG_SetTemperatureInputSelect',
    'PPSMC_MSG_SetThrottlerMask', 'PPSMC_MSG_SetToolsDramAddrHigh',
    'PPSMC_MSG_SetToolsDramAddrLow', 'PPSMC_MSG_SetVideoFps',
    'PPSMC_MSG_SetWorkloadMask', 'PPSMC_MSG_TestMessage',
    'PPSMC_MSG_TransferTableDram2Smu',
    'PPSMC_MSG_TransferTableSmu2Dram', 'PPSMC_MSG_TriggerVFFLR',
    'PPSMC_MSG_UseDefaultPPTable', 'PPSMC_Message_Count',
    'PPSMC_Result_CmdRejectedBusy', 'PPSMC_Result_CmdRejectedPrereq',
    'PPSMC_Result_Failed', 'PPSMC_Result_OK',
    'PPSMC_Result_UnknownCmd', 'PPSMC_VERSION', 'SMU_V13_0_0_PPSMC_H']
