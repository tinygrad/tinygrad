# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-include', 'stdint.h']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



SMU_13_0_6_PPSMC_H = True # macro
PPSMC_Result_OK = 0x1 # macro
PPSMC_Result_Failed = 0xFF # macro
PPSMC_Result_UnknownCmd = 0xFE # macro
PPSMC_Result_CmdRejectedPrereq = 0xFD # macro
PPSMC_Result_CmdRejectedBusy = 0xFC # macro
PPSMC_MSG_TestMessage = 0x1 # macro
PPSMC_MSG_GetSmuVersion = 0x2 # macro
PPSMC_MSG_GfxDriverReset = 0x3 # macro
PPSMC_MSG_GetDriverIfVersion = 0x4 # macro
PPSMC_MSG_EnableAllSmuFeatures = 0x5 # macro
PPSMC_MSG_DisableAllSmuFeatures = 0x6 # macro
PPSMC_MSG_RequestI2cTransaction = 0x7 # macro
PPSMC_MSG_GetMetricsVersion = 0x8 # macro
PPSMC_MSG_GetMetricsTable = 0x9 # macro
PPSMC_MSG_GetEccInfoTable = 0xA # macro
PPSMC_MSG_GetEnabledSmuFeaturesLow = 0xB # macro
PPSMC_MSG_GetEnabledSmuFeaturesHigh = 0xC # macro
PPSMC_MSG_SetDriverDramAddrHigh = 0xD # macro
PPSMC_MSG_SetDriverDramAddrLow = 0xE # macro
PPSMC_MSG_SetToolsDramAddrHigh = 0xF # macro
PPSMC_MSG_SetToolsDramAddrLow = 0x10 # macro
PPSMC_MSG_SetSystemVirtualDramAddrHigh = 0x11 # macro
PPSMC_MSG_SetSystemVirtualDramAddrLow = 0x12 # macro
PPSMC_MSG_SetSoftMinByFreq = 0x13 # macro
PPSMC_MSG_SetSoftMaxByFreq = 0x14 # macro
PPSMC_MSG_GetMinDpmFreq = 0x15 # macro
PPSMC_MSG_GetMaxDpmFreq = 0x16 # macro
PPSMC_MSG_GetDpmFreqByIndex = 0x17 # macro
PPSMC_MSG_SetPptLimit = 0x18 # macro
PPSMC_MSG_GetPptLimit = 0x19 # macro
PPSMC_MSG_DramLogSetDramAddrHigh = 0x1A # macro
PPSMC_MSG_DramLogSetDramAddrLow = 0x1B # macro
PPSMC_MSG_DramLogSetDramSize = 0x1C # macro
PPSMC_MSG_GetDebugData = 0x1D # macro
PPSMC_MSG_HeavySBR = 0x1E # macro
PPSMC_MSG_SetNumBadHbmPagesRetired = 0x1F # macro
PPSMC_MSG_DFCstateControl = 0x20 # macro
PPSMC_MSG_GetGmiPwrDnHyst = 0x21 # macro
PPSMC_MSG_SetGmiPwrDnHyst = 0x22 # macro
PPSMC_MSG_GmiPwrDnControl = 0x23 # macro
PPSMC_MSG_EnterGfxoff = 0x24 # macro
PPSMC_MSG_ExitGfxoff = 0x25 # macro
PPSMC_MSG_EnableDeterminism = 0x26 # macro
PPSMC_MSG_DisableDeterminism = 0x27 # macro
PPSMC_MSG_DumpSTBtoDram = 0x28 # macro
PPSMC_MSG_STBtoDramLogSetDramAddrHigh = 0x29 # macro
PPSMC_MSG_STBtoDramLogSetDramAddrLow = 0x2A # macro
PPSMC_MSG_STBtoDramLogSetDramSize = 0x2B # macro
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrHigh = 0x2C # macro
PPSMC_MSG_SetSystemVirtualSTBtoDramAddrLow = 0x2D # macro
PPSMC_MSG_GfxDriverResetRecovery = 0x2E # macro
PPSMC_MSG_TriggerVFFLR = 0x2F # macro
PPSMC_MSG_SetSoftMinGfxClk = 0x30 # macro
PPSMC_MSG_SetSoftMaxGfxClk = 0x31 # macro
PPSMC_MSG_GetMinGfxDpmFreq = 0x32 # macro
PPSMC_MSG_GetMaxGfxDpmFreq = 0x33 # macro
PPSMC_MSG_PrepareForDriverUnload = 0x34 # macro
PPSMC_MSG_ReadThrottlerLimit = 0x35 # macro
PPSMC_MSG_QueryValidMcaCount = 0x36 # macro
PPSMC_MSG_McaBankDumpDW = 0x37 # macro
PPSMC_MSG_GetCTFLimit = 0x38 # macro
PPSMC_MSG_ClearMcaOnRead = 0x39 # macro
PPSMC_MSG_QueryValidMcaCeCount = 0x3A # macro
PPSMC_MSG_McaBankCeDumpDW = 0x3B # macro
PPSMC_MSG_SelectPLPDMode = 0x40 # macro
PPSMC_MSG_RmaDueToBadPageThreshold = 0x43 # macro
PPSMC_MSG_SetThrottlingPolicy = 0x44 # macro
PPSMC_MSG_SetPhsDetWRbwThreshold = 0x45 # macro
PPSMC_MSG_SetPhsDetWRbwFreqHigh = 0x46 # macro
PPSMC_MSG_SetPhsDetWRbwFreqLow = 0x47 # macro
PPSMC_MSG_SetPhsDetWRbwHystDown = 0x48 # macro
PPSMC_MSG_SetPhsDetWRbwAlpha = 0x49 # macro
PPSMC_MSG_SetPhsDetOnOff = 0x4A # macro
PPSMC_MSG_GetPhsDetResidency = 0x4B # macro
PPSMC_MSG_ResetSDMA = 0x4D # macro
PPSMC_MSG_GetStaticMetricsTable = 0x59 # macro
PPSMC_Message_Count = 0x5A # macro
PPSMC_RESET_TYPE_DRIVER_MODE_1_RESET = 0x1 # macro
PPSMC_RESET_TYPE_DRIVER_MODE_2_RESET = 0x2 # macro
PPSMC_RESET_TYPE_DRIVER_MODE_3_RESET = 0x3 # macro
PPSMC_THROTTLING_LIMIT_TYPE_SOCKET = 0x1 # macro
PPSMC_THROTTLING_LIMIT_TYPE_HBM = 0x2 # macro
PPSMC_AID_THM_TYPE = 0x1 # macro
PPSMC_CCD_THM_TYPE = 0x2 # macro
PPSMC_XCD_THM_TYPE = 0x3 # macro
PPSMC_HBM_THM_TYPE = 0x4 # macro
PPSMC_PLPD_MODE_DEFAULT = 0x1 # macro
PPSMC_PLPD_MODE_OPTIMIZED = 0x2 # macro
PPSMC_Result = ctypes.c_uint32
PPSMC_MSG = ctypes.c_uint32
SMU_13_0_6_DRIVER_IF_H = True # macro
SMU13_0_6_DRIVER_IF_VERSION = 0x08042024 # macro
NUM_I2C_CONTROLLERS = 8 # macro
I2C_CONTROLLER_ENABLED = 1 # macro
I2C_CONTROLLER_DISABLED = 0 # macro
MAX_SW_I2C_COMMANDS = 24 # macro
CMDCONFIG_STOP_BIT = 0 # macro
CMDCONFIG_RESTART_BIT = 1 # macro
CMDCONFIG_READWRITE_BIT = 2 # macro
CMDCONFIG_STOP_MASK = (1<<0) # macro
CMDCONFIG_RESTART_MASK = (1<<1) # macro
CMDCONFIG_READWRITE_MASK = (1<<2) # macro
IH_INTERRUPT_ID_TO_DRIVER = 0xFE # macro
IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING = 0x7 # macro
THROTTLER_PROCHOT_BIT = 0 # macro
THROTTLER_PPT_BIT = 1 # macro
THROTTLER_THERMAL_SOCKET_BIT = 2 # macro
THROTTLER_THERMAL_VR_BIT = 3 # macro
THROTTLER_THERMAL_HBM_BIT = 4 # macro
ClearMcaOnRead_UE_FLAG_MASK = 0x1 # macro
ClearMcaOnRead_CE_POLL_MASK = 0x2 # macro

# values for enumeration 'c__EA_I2cControllerPort_e'
c__EA_I2cControllerPort_e__enumvalues = {
    0: 'I2C_CONTROLLER_PORT_0',
    1: 'I2C_CONTROLLER_PORT_1',
    2: 'I2C_CONTROLLER_PORT_COUNT',
}
I2C_CONTROLLER_PORT_0 = 0
I2C_CONTROLLER_PORT_1 = 1
I2C_CONTROLLER_PORT_COUNT = 2
c__EA_I2cControllerPort_e = ctypes.c_uint32 # enum
I2cControllerPort_e = c__EA_I2cControllerPort_e
I2cControllerPort_e__enumvalues = c__EA_I2cControllerPort_e__enumvalues

# values for enumeration 'c__EA_I2cSpeed_e'
c__EA_I2cSpeed_e__enumvalues = {
    0: 'UNSUPPORTED_1',
    1: 'I2C_SPEED_STANDARD_100K',
    2: 'I2C_SPEED_FAST_400K',
    3: 'I2C_SPEED_FAST_PLUS_1M',
    4: 'UNSUPPORTED_2',
    5: 'UNSUPPORTED_3',
    6: 'I2C_SPEED_COUNT',
}
UNSUPPORTED_1 = 0
I2C_SPEED_STANDARD_100K = 1
I2C_SPEED_FAST_400K = 2
I2C_SPEED_FAST_PLUS_1M = 3
UNSUPPORTED_2 = 4
UNSUPPORTED_3 = 5
I2C_SPEED_COUNT = 6
c__EA_I2cSpeed_e = ctypes.c_uint32 # enum
I2cSpeed_e = c__EA_I2cSpeed_e
I2cSpeed_e__enumvalues = c__EA_I2cSpeed_e__enumvalues

# values for enumeration 'c__EA_I2cCmdType_e'
c__EA_I2cCmdType_e__enumvalues = {
    0: 'I2C_CMD_READ',
    1: 'I2C_CMD_WRITE',
    2: 'I2C_CMD_COUNT',
}
I2C_CMD_READ = 0
I2C_CMD_WRITE = 1
I2C_CMD_COUNT = 2
c__EA_I2cCmdType_e = ctypes.c_uint32 # enum
I2cCmdType_e = c__EA_I2cCmdType_e
I2cCmdType_e__enumvalues = c__EA_I2cCmdType_e__enumvalues

# values for enumeration 'c__EA_ERR_CODE_e'
c__EA_ERR_CODE_e__enumvalues = {
    0: 'CODE_DAGB0',
    5: 'CODE_EA0',
    10: 'CODE_UTCL2_ROUTER',
    11: 'CODE_VML2',
    12: 'CODE_VML2_WALKER',
    13: 'CODE_MMCANE',
    14: 'CODE_VIDD',
    15: 'CODE_VIDV',
    16: 'CODE_JPEG0S',
    17: 'CODE_JPEG0D',
    18: 'CODE_JPEG1S',
    19: 'CODE_JPEG1D',
    20: 'CODE_JPEG2S',
    21: 'CODE_JPEG2D',
    22: 'CODE_JPEG3S',
    23: 'CODE_JPEG3D',
    24: 'CODE_JPEG4S',
    25: 'CODE_JPEG4D',
    26: 'CODE_JPEG5S',
    27: 'CODE_JPEG5D',
    28: 'CODE_JPEG6S',
    29: 'CODE_JPEG6D',
    30: 'CODE_JPEG7S',
    31: 'CODE_JPEG7D',
    32: 'CODE_MMSCHD',
    33: 'CODE_SDMA0',
    34: 'CODE_SDMA1',
    35: 'CODE_SDMA2',
    36: 'CODE_SDMA3',
    37: 'CODE_HDP',
    38: 'CODE_ATHUB',
    39: 'CODE_IH',
    40: 'CODE_XHUB_POISON',
    40: 'CODE_SMN_SLVERR',
    41: 'CODE_WDT',
    42: 'CODE_UNKNOWN',
    43: 'CODE_COUNT',
}
CODE_DAGB0 = 0
CODE_EA0 = 5
CODE_UTCL2_ROUTER = 10
CODE_VML2 = 11
CODE_VML2_WALKER = 12
CODE_MMCANE = 13
CODE_VIDD = 14
CODE_VIDV = 15
CODE_JPEG0S = 16
CODE_JPEG0D = 17
CODE_JPEG1S = 18
CODE_JPEG1D = 19
CODE_JPEG2S = 20
CODE_JPEG2D = 21
CODE_JPEG3S = 22
CODE_JPEG3D = 23
CODE_JPEG4S = 24
CODE_JPEG4D = 25
CODE_JPEG5S = 26
CODE_JPEG5D = 27
CODE_JPEG6S = 28
CODE_JPEG6D = 29
CODE_JPEG7S = 30
CODE_JPEG7D = 31
CODE_MMSCHD = 32
CODE_SDMA0 = 33
CODE_SDMA1 = 34
CODE_SDMA2 = 35
CODE_SDMA3 = 36
CODE_HDP = 37
CODE_ATHUB = 38
CODE_IH = 39
CODE_XHUB_POISON = 40
CODE_SMN_SLVERR = 40
CODE_WDT = 41
CODE_UNKNOWN = 42
CODE_COUNT = 43
c__EA_ERR_CODE_e = ctypes.c_uint32 # enum
ERR_CODE_e = c__EA_ERR_CODE_e
ERR_CODE_e__enumvalues = c__EA_ERR_CODE_e__enumvalues

# values for enumeration 'c__EA_GC_ERROR_CODE_e'
c__EA_GC_ERROR_CODE_e__enumvalues = {
    0: 'SH_FED_CODE',
    1: 'GCEA_CODE',
    2: 'SQ_CODE',
    3: 'LDS_CODE',
    4: 'GDS_CODE',
    5: 'SP0_CODE',
    6: 'SP1_CODE',
    7: 'TCC_CODE',
    8: 'TCA_CODE',
    9: 'TCX_CODE',
    10: 'CPC_CODE',
    11: 'CPF_CODE',
    12: 'CPG_CODE',
    13: 'SPI_CODE',
    14: 'RLC_CODE',
    15: 'SQC_CODE',
    16: 'TA_CODE',
    17: 'TD_CODE',
    18: 'TCP_CODE',
    19: 'TCI_CODE',
    20: 'GC_ROUTER_CODE',
    21: 'VML2_CODE',
    22: 'VML2_WALKER_CODE',
    23: 'ATCL2_CODE',
    24: 'GC_CANE_CODE',
    40: 'MP5_CODE_SMN_SLVERR',
    42: 'MP5_CODE_UNKNOWN',
}
SH_FED_CODE = 0
GCEA_CODE = 1
SQ_CODE = 2
LDS_CODE = 3
GDS_CODE = 4
SP0_CODE = 5
SP1_CODE = 6
TCC_CODE = 7
TCA_CODE = 8
TCX_CODE = 9
CPC_CODE = 10
CPF_CODE = 11
CPG_CODE = 12
SPI_CODE = 13
RLC_CODE = 14
SQC_CODE = 15
TA_CODE = 16
TD_CODE = 17
TCP_CODE = 18
TCI_CODE = 19
GC_ROUTER_CODE = 20
VML2_CODE = 21
VML2_WALKER_CODE = 22
ATCL2_CODE = 23
GC_CANE_CODE = 24
MP5_CODE_SMN_SLVERR = 40
MP5_CODE_UNKNOWN = 42
c__EA_GC_ERROR_CODE_e = ctypes.c_uint32 # enum
GC_ERROR_CODE_e = c__EA_GC_ERROR_CODE_e
GC_ERROR_CODE_e__enumvalues = c__EA_GC_ERROR_CODE_e__enumvalues
class struct_c__SA_SwI2cCmd_t(Structure):
    pass

struct_c__SA_SwI2cCmd_t._pack_ = 1 # source:False
struct_c__SA_SwI2cCmd_t._fields_ = [
    ('ReadWriteData', ctypes.c_ubyte),
    ('CmdConfig', ctypes.c_ubyte),
]

SwI2cCmd_t = struct_c__SA_SwI2cCmd_t
class struct_c__SA_SwI2cRequest_t(Structure):
    pass

struct_c__SA_SwI2cRequest_t._pack_ = 1 # source:False
struct_c__SA_SwI2cRequest_t._fields_ = [
    ('I2CcontrollerPort', ctypes.c_ubyte),
    ('I2CSpeed', ctypes.c_ubyte),
    ('SlaveAddress', ctypes.c_ubyte),
    ('NumCmds', ctypes.c_ubyte),
    ('SwI2cCmds', struct_c__SA_SwI2cCmd_t * 24),
]

SwI2cRequest_t = struct_c__SA_SwI2cRequest_t
class struct_c__SA_SwI2cRequestExternal_t(Structure):
    pass

struct_c__SA_SwI2cRequestExternal_t._pack_ = 1 # source:False
struct_c__SA_SwI2cRequestExternal_t._fields_ = [
    ('SwI2cRequest', SwI2cRequest_t),
    ('Spare', ctypes.c_uint32 * 8),
    ('MmHubPadding', ctypes.c_uint32 * 8),
]

SwI2cRequestExternal_t = struct_c__SA_SwI2cRequestExternal_t

# values for enumeration 'c__EA_PPCLK_e'
c__EA_PPCLK_e__enumvalues = {
    0: 'PPCLK_VCLK',
    1: 'PPCLK_DCLK',
    2: 'PPCLK_SOCCLK',
    3: 'PPCLK_UCLK',
    4: 'PPCLK_FCLK',
    5: 'PPCLK_LCLK',
    6: 'PPCLK_COUNT',
}
PPCLK_VCLK = 0
PPCLK_DCLK = 1
PPCLK_SOCCLK = 2
PPCLK_UCLK = 3
PPCLK_FCLK = 4
PPCLK_LCLK = 5
PPCLK_COUNT = 6
c__EA_PPCLK_e = ctypes.c_uint32 # enum
PPCLK_e = c__EA_PPCLK_e
PPCLK_e__enumvalues = c__EA_PPCLK_e__enumvalues

# values for enumeration 'c__EA_GpioIntPolarity_e'
c__EA_GpioIntPolarity_e__enumvalues = {
    0: 'GPIO_INT_POLARITY_ACTIVE_LOW',
    1: 'GPIO_INT_POLARITY_ACTIVE_HIGH',
}
GPIO_INT_POLARITY_ACTIVE_LOW = 0
GPIO_INT_POLARITY_ACTIVE_HIGH = 1
c__EA_GpioIntPolarity_e = ctypes.c_uint32 # enum
GpioIntPolarity_e = c__EA_GpioIntPolarity_e
GpioIntPolarity_e__enumvalues = c__EA_GpioIntPolarity_e__enumvalues

# values for enumeration 'c__EA_UCLK_DPM_MODE_e'
c__EA_UCLK_DPM_MODE_e__enumvalues = {
    0: 'UCLK_DPM_MODE_BANDWIDTH',
    1: 'UCLK_DPM_MODE_LATENCY',
}
UCLK_DPM_MODE_BANDWIDTH = 0
UCLK_DPM_MODE_LATENCY = 1
c__EA_UCLK_DPM_MODE_e = ctypes.c_uint32 # enum
UCLK_DPM_MODE_e = c__EA_UCLK_DPM_MODE_e
UCLK_DPM_MODE_e__enumvalues = c__EA_UCLK_DPM_MODE_e__enumvalues
class struct_c__SA_AvfsDebugTableAid_t(Structure):
    pass

struct_c__SA_AvfsDebugTableAid_t._pack_ = 1 # source:False
struct_c__SA_AvfsDebugTableAid_t._fields_ = [
    ('avgPsmCount', ctypes.c_uint16 * 30),
    ('minPsmCount', ctypes.c_uint16 * 30),
    ('avgPsmVoltage', ctypes.c_float * 30),
    ('minPsmVoltage', ctypes.c_float * 30),
]

AvfsDebugTableAid_t = struct_c__SA_AvfsDebugTableAid_t
class struct_c__SA_AvfsDebugTableXcd_t(Structure):
    pass

struct_c__SA_AvfsDebugTableXcd_t._pack_ = 1 # source:False
struct_c__SA_AvfsDebugTableXcd_t._fields_ = [
    ('avgPsmCount', ctypes.c_uint16 * 30),
    ('minPsmCount', ctypes.c_uint16 * 30),
    ('avgPsmVoltage', ctypes.c_float * 30),
    ('minPsmVoltage', ctypes.c_float * 30),
]

AvfsDebugTableXcd_t = struct_c__SA_AvfsDebugTableXcd_t
__AMDGPU_SMU_H__ = True # macro
int32_t = True # macro
uint32_t = True # macro
int8_t = True # macro
uint8_t = True # macro
uint16_t = True # macro
int16_t = True # macro
uint64_t = True # macro
bool = True # macro
u32 = True # macro
SMU_THERMAL_MINIMUM_ALERT_TEMP = 0 # macro
SMU_THERMAL_MAXIMUM_ALERT_TEMP = 255 # macro
SMU_TEMPERATURE_UNITS_PER_CENTIGRADES = 1000 # macro
SMU_FW_NAME_LEN = 0x24 # macro
SMU_DPM_USER_PROFILE_RESTORE = (1<<0) # macro
SMU_CUSTOM_FAN_SPEED_RPM = (1<<1) # macro
SMU_CUSTOM_FAN_SPEED_PWM = (1<<2) # macro
SMU_THROTTLER_PPT0_BIT = 0 # macro
SMU_THROTTLER_PPT1_BIT = 1 # macro
SMU_THROTTLER_PPT2_BIT = 2 # macro
SMU_THROTTLER_PPT3_BIT = 3 # macro
SMU_THROTTLER_SPL_BIT = 4 # macro
SMU_THROTTLER_FPPT_BIT = 5 # macro
SMU_THROTTLER_SPPT_BIT = 6 # macro
SMU_THROTTLER_SPPT_APU_BIT = 7 # macro
SMU_THROTTLER_TDC_GFX_BIT = 16 # macro
SMU_THROTTLER_TDC_SOC_BIT = 17 # macro
SMU_THROTTLER_TDC_MEM_BIT = 18 # macro
SMU_THROTTLER_TDC_VDD_BIT = 19 # macro
SMU_THROTTLER_TDC_CVIP_BIT = 20 # macro
SMU_THROTTLER_EDC_CPU_BIT = 21 # macro
SMU_THROTTLER_EDC_GFX_BIT = 22 # macro
SMU_THROTTLER_APCC_BIT = 23 # macro
SMU_THROTTLER_TEMP_GPU_BIT = 32 # macro
SMU_THROTTLER_TEMP_CORE_BIT = 33 # macro
SMU_THROTTLER_TEMP_MEM_BIT = 34 # macro
SMU_THROTTLER_TEMP_EDGE_BIT = 35 # macro
SMU_THROTTLER_TEMP_HOTSPOT_BIT = 36 # macro
SMU_THROTTLER_TEMP_SOC_BIT = 37 # macro
SMU_THROTTLER_TEMP_VR_GFX_BIT = 38 # macro
SMU_THROTTLER_TEMP_VR_SOC_BIT = 39 # macro
SMU_THROTTLER_TEMP_VR_MEM0_BIT = 40 # macro
SMU_THROTTLER_TEMP_VR_MEM1_BIT = 41 # macro
SMU_THROTTLER_TEMP_LIQUID0_BIT = 42 # macro
SMU_THROTTLER_TEMP_LIQUID1_BIT = 43 # macro
SMU_THROTTLER_VRHOT0_BIT = 44 # macro
SMU_THROTTLER_VRHOT1_BIT = 45 # macro
SMU_THROTTLER_PROCHOT_CPU_BIT = 46 # macro
SMU_THROTTLER_PROCHOT_GFX_BIT = 47 # macro
SMU_THROTTLER_PPM_BIT = 56 # macro
SMU_THROTTLER_FIT_BIT = 57 # macro
# def SMU_TABLE_INIT(tables, table_id, s, a, d):  # macro
#    return {tables[table_id].size=s;tables[table_id].align=a;tables[table_id].domain=d;}(0)
class struct_smu_hw_power_state(Structure):
    pass

struct_smu_hw_power_state._pack_ = 1 # source:False
struct_smu_hw_power_state._fields_ = [
    ('magic', ctypes.c_uint32),
]

class struct_smu_power_state(Structure):
    pass


# values for enumeration 'smu_state_ui_label'
smu_state_ui_label__enumvalues = {
    0: 'SMU_STATE_UI_LABEL_NONE',
    1: 'SMU_STATE_UI_LABEL_BATTERY',
    2: 'SMU_STATE_UI_TABEL_MIDDLE_LOW',
    3: 'SMU_STATE_UI_LABEL_BALLANCED',
    4: 'SMU_STATE_UI_LABEL_MIDDLE_HIGHT',
    5: 'SMU_STATE_UI_LABEL_PERFORMANCE',
    6: 'SMU_STATE_UI_LABEL_BACO',
}
SMU_STATE_UI_LABEL_NONE = 0
SMU_STATE_UI_LABEL_BATTERY = 1
SMU_STATE_UI_TABEL_MIDDLE_LOW = 2
SMU_STATE_UI_LABEL_BALLANCED = 3
SMU_STATE_UI_LABEL_MIDDLE_HIGHT = 4
SMU_STATE_UI_LABEL_PERFORMANCE = 5
SMU_STATE_UI_LABEL_BACO = 6
smu_state_ui_label = ctypes.c_uint32 # enum

# values for enumeration 'smu_state_classification_flag'
smu_state_classification_flag__enumvalues = {
    1: 'SMU_STATE_CLASSIFICATION_FLAG_BOOT',
    2: 'SMU_STATE_CLASSIFICATION_FLAG_THERMAL',
    4: 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE',
    8: 'SMU_STATE_CLASSIFICATION_FLAG_RESET',
    16: 'SMU_STATE_CLASSIFICATION_FLAG_FORCED',
    32: 'SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE',
    64: 'SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE',
    128: 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE',
    256: 'SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE',
    512: 'SMU_STATE_CLASSIFICATION_FLAG_UVD',
    1024: 'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW',
    2048: 'SMU_STATE_CLASSIFICATION_FLAG_ACPI',
    4096: 'SMU_STATE_CLASSIFICATION_FLAG_HD2',
    8192: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_HD',
    16384: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_SD',
    32768: 'SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE',
    65536: 'SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE',
    131072: 'SMU_STATE_CLASSIFICATION_FLAG_BACO',
    262144: 'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2',
    524288: 'SMU_STATE_CLASSIFICATION_FLAG_ULV',
    1048576: 'SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC',
}
SMU_STATE_CLASSIFICATION_FLAG_BOOT = 1
SMU_STATE_CLASSIFICATION_FLAG_THERMAL = 2
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE = 4
SMU_STATE_CLASSIFICATION_FLAG_RESET = 8
SMU_STATE_CLASSIFICATION_FLAG_FORCED = 16
SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE = 32
SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE = 64
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE = 128
SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE = 256
SMU_STATE_CLASSIFICATION_FLAG_UVD = 512
SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW = 1024
SMU_STATE_CLASSIFICATION_FLAG_ACPI = 2048
SMU_STATE_CLASSIFICATION_FLAG_HD2 = 4096
SMU_STATE_CLASSIFICATION_FLAG_UVD_HD = 8192
SMU_STATE_CLASSIFICATION_FLAG_UVD_SD = 16384
SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE = 32768
SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE = 65536
SMU_STATE_CLASSIFICATION_FLAG_BACO = 131072
SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2 = 262144
SMU_STATE_CLASSIFICATION_FLAG_ULV = 524288
SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC = 1048576
smu_state_classification_flag = ctypes.c_uint32 # enum
class struct_smu_state_classification_block(Structure):
    pass

struct_smu_state_classification_block._pack_ = 1 # source:False
struct_smu_state_classification_block._fields_ = [
    ('ui_label', smu_state_ui_label),
    ('flags', smu_state_classification_flag),
    ('bios_index', ctypes.c_int32),
    ('temporary_state', ctypes.c_bool),
    ('to_be_deleted', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_smu_state_pcie_block(Structure):
    pass

struct_smu_state_pcie_block._pack_ = 1 # source:False
struct_smu_state_pcie_block._fields_ = [
    ('lanes', ctypes.c_uint32),
]


# values for enumeration 'smu_refreshrate_source'
smu_refreshrate_source__enumvalues = {
    0: 'SMU_REFRESHRATE_SOURCE_EDID',
    1: 'SMU_REFRESHRATE_SOURCE_EXPLICIT',
}
SMU_REFRESHRATE_SOURCE_EDID = 0
SMU_REFRESHRATE_SOURCE_EXPLICIT = 1
smu_refreshrate_source = ctypes.c_uint32 # enum
class struct_smu_state_display_block(Structure):
    pass

struct_smu_state_display_block._pack_ = 1 # source:False
struct_smu_state_display_block._fields_ = [
    ('disable_frame_modulation', ctypes.c_bool),
    ('limit_refreshrate', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('refreshrate_source', smu_refreshrate_source),
    ('explicit_refreshrate', ctypes.c_int32),
    ('edid_refreshrate_index', ctypes.c_int32),
    ('enable_vari_bright', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

class struct_smu_state_memory_block(Structure):
    pass

struct_smu_state_memory_block._pack_ = 1 # source:False
struct_smu_state_memory_block._fields_ = [
    ('dll_off', ctypes.c_bool),
    ('m3arb', ctypes.c_ubyte),
    ('unused', ctypes.c_ubyte * 3),
]

class struct_smu_state_software_algorithm_block(Structure):
    pass

struct_smu_state_software_algorithm_block._pack_ = 1 # source:False
struct_smu_state_software_algorithm_block._fields_ = [
    ('disable_load_balancing', ctypes.c_bool),
    ('enable_sleep_for_timestamps', ctypes.c_bool),
]

class struct_smu_temperature_range(Structure):
    pass

struct_smu_temperature_range._pack_ = 1 # source:False
struct_smu_temperature_range._fields_ = [
    ('min', ctypes.c_int32),
    ('max', ctypes.c_int32),
    ('edge_emergency_max', ctypes.c_int32),
    ('hotspot_min', ctypes.c_int32),
    ('hotspot_crit_max', ctypes.c_int32),
    ('hotspot_emergency_max', ctypes.c_int32),
    ('mem_min', ctypes.c_int32),
    ('mem_crit_max', ctypes.c_int32),
    ('mem_emergency_max', ctypes.c_int32),
    ('software_shutdown_temp', ctypes.c_int32),
    ('software_shutdown_temp_offset', ctypes.c_int32),
]

class struct_smu_state_validation_block(Structure):
    pass

struct_smu_state_validation_block._pack_ = 1 # source:False
struct_smu_state_validation_block._fields_ = [
    ('single_display_only', ctypes.c_bool),
    ('disallow_on_dc', ctypes.c_bool),
    ('supported_power_levels', ctypes.c_ubyte),
]

class struct_smu_uvd_clocks(Structure):
    pass

struct_smu_uvd_clocks._pack_ = 1 # source:False
struct_smu_uvd_clocks._fields_ = [
    ('vclk', ctypes.c_uint32),
    ('dclk', ctypes.c_uint32),
]


# values for enumeration 'smu_power_src_type'
smu_power_src_type__enumvalues = {
    0: 'SMU_POWER_SOURCE_AC',
    1: 'SMU_POWER_SOURCE_DC',
    2: 'SMU_POWER_SOURCE_COUNT',
}
SMU_POWER_SOURCE_AC = 0
SMU_POWER_SOURCE_DC = 1
SMU_POWER_SOURCE_COUNT = 2
smu_power_src_type = ctypes.c_uint32 # enum

# values for enumeration 'smu_ppt_limit_type'
smu_ppt_limit_type__enumvalues = {
    0: 'SMU_DEFAULT_PPT_LIMIT',
    1: 'SMU_FAST_PPT_LIMIT',
}
SMU_DEFAULT_PPT_LIMIT = 0
SMU_FAST_PPT_LIMIT = 1
smu_ppt_limit_type = ctypes.c_uint32 # enum

# values for enumeration 'smu_ppt_limit_level'
smu_ppt_limit_level__enumvalues = {
    -1: 'SMU_PPT_LIMIT_MIN',
    0: 'SMU_PPT_LIMIT_CURRENT',
    1: 'SMU_PPT_LIMIT_DEFAULT',
    2: 'SMU_PPT_LIMIT_MAX',
}
SMU_PPT_LIMIT_MIN = -1
SMU_PPT_LIMIT_CURRENT = 0
SMU_PPT_LIMIT_DEFAULT = 1
SMU_PPT_LIMIT_MAX = 2
smu_ppt_limit_level = ctypes.c_int32 # enum

# values for enumeration 'smu_memory_pool_size'
smu_memory_pool_size__enumvalues = {
    0: 'SMU_MEMORY_POOL_SIZE_ZERO',
    268435456: 'SMU_MEMORY_POOL_SIZE_256_MB',
    536870912: 'SMU_MEMORY_POOL_SIZE_512_MB',
    1073741824: 'SMU_MEMORY_POOL_SIZE_1_GB',
    2147483648: 'SMU_MEMORY_POOL_SIZE_2_GB',
}
SMU_MEMORY_POOL_SIZE_ZERO = 0
SMU_MEMORY_POOL_SIZE_256_MB = 268435456
SMU_MEMORY_POOL_SIZE_512_MB = 536870912
SMU_MEMORY_POOL_SIZE_1_GB = 1073741824
SMU_MEMORY_POOL_SIZE_2_GB = 2147483648
smu_memory_pool_size = ctypes.c_uint32 # enum

# values for enumeration 'smu_clk_type'
smu_clk_type__enumvalues = {
    0: 'SMU_GFXCLK',
    1: 'SMU_VCLK',
    2: 'SMU_DCLK',
    3: 'SMU_VCLK1',
    4: 'SMU_DCLK1',
    5: 'SMU_ECLK',
    6: 'SMU_SOCCLK',
    7: 'SMU_UCLK',
    8: 'SMU_DCEFCLK',
    9: 'SMU_DISPCLK',
    10: 'SMU_PIXCLK',
    11: 'SMU_PHYCLK',
    12: 'SMU_FCLK',
    13: 'SMU_SCLK',
    14: 'SMU_MCLK',
    15: 'SMU_PCIE',
    16: 'SMU_LCLK',
    17: 'SMU_OD_CCLK',
    18: 'SMU_OD_SCLK',
    19: 'SMU_OD_MCLK',
    20: 'SMU_OD_VDDC_CURVE',
    21: 'SMU_OD_RANGE',
    22: 'SMU_OD_VDDGFX_OFFSET',
    23: 'SMU_OD_FAN_CURVE',
    24: 'SMU_OD_ACOUSTIC_LIMIT',
    25: 'SMU_OD_ACOUSTIC_TARGET',
    26: 'SMU_OD_FAN_TARGET_TEMPERATURE',
    27: 'SMU_OD_FAN_MINIMUM_PWM',
    28: 'SMU_CLK_COUNT',
}
SMU_GFXCLK = 0
SMU_VCLK = 1
SMU_DCLK = 2
SMU_VCLK1 = 3
SMU_DCLK1 = 4
SMU_ECLK = 5
SMU_SOCCLK = 6
SMU_UCLK = 7
SMU_DCEFCLK = 8
SMU_DISPCLK = 9
SMU_PIXCLK = 10
SMU_PHYCLK = 11
SMU_FCLK = 12
SMU_SCLK = 13
SMU_MCLK = 14
SMU_PCIE = 15
SMU_LCLK = 16
SMU_OD_CCLK = 17
SMU_OD_SCLK = 18
SMU_OD_MCLK = 19
SMU_OD_VDDC_CURVE = 20
SMU_OD_RANGE = 21
SMU_OD_VDDGFX_OFFSET = 22
SMU_OD_FAN_CURVE = 23
SMU_OD_ACOUSTIC_LIMIT = 24
SMU_OD_ACOUSTIC_TARGET = 25
SMU_OD_FAN_TARGET_TEMPERATURE = 26
SMU_OD_FAN_MINIMUM_PWM = 27
SMU_CLK_COUNT = 28
smu_clk_type = ctypes.c_uint32 # enum
class struct_smu_user_dpm_profile(Structure):
    pass

struct_smu_user_dpm_profile._pack_ = 1 # source:False
struct_smu_user_dpm_profile._fields_ = [
    ('fan_mode', ctypes.c_uint32),
    ('power_limit', ctypes.c_uint32),
    ('fan_speed_pwm', ctypes.c_uint32),
    ('fan_speed_rpm', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('user_od', ctypes.c_uint32),
    ('clk_mask', ctypes.c_uint32 * 28),
    ('clk_dependency', ctypes.c_uint32),
]

class struct_smu_table(Structure):
    pass

class struct_amdgpu_bo(Structure):
    pass

struct_smu_table._pack_ = 1 # source:False
struct_smu_table._fields_ = [
    ('size', ctypes.c_uint64),
    ('align', ctypes.c_uint32),
    ('domain', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('mc_address', ctypes.c_uint64),
    ('cpu_addr', ctypes.POINTER(None)),
    ('bo', ctypes.POINTER(struct_amdgpu_bo)),
    ('version', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]


# values for enumeration 'smu_perf_level_designation'
smu_perf_level_designation__enumvalues = {
    0: 'PERF_LEVEL_ACTIVITY',
    1: 'PERF_LEVEL_POWER_CONTAINMENT',
}
PERF_LEVEL_ACTIVITY = 0
PERF_LEVEL_POWER_CONTAINMENT = 1
smu_perf_level_designation = ctypes.c_uint32 # enum
class struct_smu_performance_level(Structure):
    pass

struct_smu_performance_level._pack_ = 1 # source:False
struct_smu_performance_level._fields_ = [
    ('core_clock', ctypes.c_uint32),
    ('memory_clock', ctypes.c_uint32),
    ('vddc', ctypes.c_uint32),
    ('vddci', ctypes.c_uint32),
    ('non_local_mem_freq', ctypes.c_uint32),
    ('non_local_mem_width', ctypes.c_uint32),
]

class struct_smu_clock_info(Structure):
    pass

struct_smu_clock_info._pack_ = 1 # source:False
struct_smu_clock_info._fields_ = [
    ('min_mem_clk', ctypes.c_uint32),
    ('max_mem_clk', ctypes.c_uint32),
    ('min_eng_clk', ctypes.c_uint32),
    ('max_eng_clk', ctypes.c_uint32),
    ('min_bus_bandwidth', ctypes.c_uint32),
    ('max_bus_bandwidth', ctypes.c_uint32),
]

class struct_smu_bios_boot_up_values(Structure):
    pass

struct_smu_bios_boot_up_values._pack_ = 1 # source:False
struct_smu_bios_boot_up_values._fields_ = [
    ('revision', ctypes.c_uint32),
    ('gfxclk', ctypes.c_uint32),
    ('uclk', ctypes.c_uint32),
    ('socclk', ctypes.c_uint32),
    ('dcefclk', ctypes.c_uint32),
    ('eclk', ctypes.c_uint32),
    ('vclk', ctypes.c_uint32),
    ('dclk', ctypes.c_uint32),
    ('vddc', ctypes.c_uint16),
    ('vddci', ctypes.c_uint16),
    ('mvddc', ctypes.c_uint16),
    ('vdd_gfx', ctypes.c_uint16),
    ('cooling_id', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('pp_table_id', ctypes.c_uint32),
    ('format_revision', ctypes.c_uint32),
    ('content_revision', ctypes.c_uint32),
    ('fclk', ctypes.c_uint32),
    ('lclk', ctypes.c_uint32),
    ('firmware_caps', ctypes.c_uint32),
]


# values for enumeration 'smu_table_id'
smu_table_id__enumvalues = {
    0: 'SMU_TABLE_PPTABLE',
    1: 'SMU_TABLE_WATERMARKS',
    2: 'SMU_TABLE_CUSTOM_DPM',
    3: 'SMU_TABLE_DPMCLOCKS',
    4: 'SMU_TABLE_AVFS',
    5: 'SMU_TABLE_AVFS_PSM_DEBUG',
    6: 'SMU_TABLE_AVFS_FUSE_OVERRIDE',
    7: 'SMU_TABLE_PMSTATUSLOG',
    8: 'SMU_TABLE_SMU_METRICS',
    9: 'SMU_TABLE_DRIVER_SMU_CONFIG',
    10: 'SMU_TABLE_ACTIVITY_MONITOR_COEFF',
    11: 'SMU_TABLE_OVERDRIVE',
    12: 'SMU_TABLE_I2C_COMMANDS',
    13: 'SMU_TABLE_PACE',
    14: 'SMU_TABLE_ECCINFO',
    15: 'SMU_TABLE_COMBO_PPTABLE',
    16: 'SMU_TABLE_WIFIBAND',
    17: 'SMU_TABLE_COUNT',
}
SMU_TABLE_PPTABLE = 0
SMU_TABLE_WATERMARKS = 1
SMU_TABLE_CUSTOM_DPM = 2
SMU_TABLE_DPMCLOCKS = 3
SMU_TABLE_AVFS = 4
SMU_TABLE_AVFS_PSM_DEBUG = 5
SMU_TABLE_AVFS_FUSE_OVERRIDE = 6
SMU_TABLE_PMSTATUSLOG = 7
SMU_TABLE_SMU_METRICS = 8
SMU_TABLE_DRIVER_SMU_CONFIG = 9
SMU_TABLE_ACTIVITY_MONITOR_COEFF = 10
SMU_TABLE_OVERDRIVE = 11
SMU_TABLE_I2C_COMMANDS = 12
SMU_TABLE_PACE = 13
SMU_TABLE_ECCINFO = 14
SMU_TABLE_COMBO_PPTABLE = 15
SMU_TABLE_WIFIBAND = 16
SMU_TABLE_COUNT = 17
smu_table_id = ctypes.c_uint32 # enum
__all__ = \
    ['ATCL2_CODE', 'AvfsDebugTableAid_t', 'AvfsDebugTableXcd_t',
    'CMDCONFIG_READWRITE_BIT', 'CMDCONFIG_READWRITE_MASK',
    'CMDCONFIG_RESTART_BIT', 'CMDCONFIG_RESTART_MASK',
    'CMDCONFIG_STOP_BIT', 'CMDCONFIG_STOP_MASK', 'CODE_ATHUB',
    'CODE_COUNT', 'CODE_DAGB0', 'CODE_EA0', 'CODE_HDP', 'CODE_IH',
    'CODE_JPEG0D', 'CODE_JPEG0S', 'CODE_JPEG1D', 'CODE_JPEG1S',
    'CODE_JPEG2D', 'CODE_JPEG2S', 'CODE_JPEG3D', 'CODE_JPEG3S',
    'CODE_JPEG4D', 'CODE_JPEG4S', 'CODE_JPEG5D', 'CODE_JPEG5S',
    'CODE_JPEG6D', 'CODE_JPEG6S', 'CODE_JPEG7D', 'CODE_JPEG7S',
    'CODE_MMCANE', 'CODE_MMSCHD', 'CODE_SDMA0', 'CODE_SDMA1',
    'CODE_SDMA2', 'CODE_SDMA3', 'CODE_SMN_SLVERR', 'CODE_UNKNOWN',
    'CODE_UTCL2_ROUTER', 'CODE_VIDD', 'CODE_VIDV', 'CODE_VML2',
    'CODE_VML2_WALKER', 'CODE_WDT', 'CODE_XHUB_POISON', 'CPC_CODE',
    'CPF_CODE', 'CPG_CODE', 'ClearMcaOnRead_CE_POLL_MASK',
    'ClearMcaOnRead_UE_FLAG_MASK', 'ERR_CODE_e',
    'ERR_CODE_e__enumvalues', 'GCEA_CODE', 'GC_CANE_CODE',
    'GC_ERROR_CODE_e', 'GC_ERROR_CODE_e__enumvalues',
    'GC_ROUTER_CODE', 'GDS_CODE', 'GPIO_INT_POLARITY_ACTIVE_HIGH',
    'GPIO_INT_POLARITY_ACTIVE_LOW', 'GpioIntPolarity_e',
    'GpioIntPolarity_e__enumvalues', 'I2C_CMD_COUNT', 'I2C_CMD_READ',
    'I2C_CMD_WRITE', 'I2C_CONTROLLER_DISABLED',
    'I2C_CONTROLLER_ENABLED', 'I2C_CONTROLLER_PORT_0',
    'I2C_CONTROLLER_PORT_1', 'I2C_CONTROLLER_PORT_COUNT',
    'I2C_SPEED_COUNT', 'I2C_SPEED_FAST_400K',
    'I2C_SPEED_FAST_PLUS_1M', 'I2C_SPEED_STANDARD_100K',
    'I2cCmdType_e', 'I2cCmdType_e__enumvalues', 'I2cControllerPort_e',
    'I2cControllerPort_e__enumvalues', 'I2cSpeed_e',
    'I2cSpeed_e__enumvalues',
    'IH_INTERRUPT_CONTEXT_ID_THERMAL_THROTTLING',
    'IH_INTERRUPT_ID_TO_DRIVER', 'LDS_CODE', 'MAX_SW_I2C_COMMANDS',
    'MP5_CODE_SMN_SLVERR', 'MP5_CODE_UNKNOWN', 'NUM_I2C_CONTROLLERS',
    'PERF_LEVEL_ACTIVITY', 'PERF_LEVEL_POWER_CONTAINMENT',
    'PPCLK_COUNT', 'PPCLK_DCLK', 'PPCLK_FCLK', 'PPCLK_LCLK',
    'PPCLK_SOCCLK', 'PPCLK_UCLK', 'PPCLK_VCLK', 'PPCLK_e',
    'PPCLK_e__enumvalues', 'PPSMC_AID_THM_TYPE', 'PPSMC_CCD_THM_TYPE',
    'PPSMC_HBM_THM_TYPE', 'PPSMC_MSG', 'PPSMC_MSG_ClearMcaOnRead',
    'PPSMC_MSG_DFCstateControl', 'PPSMC_MSG_DisableAllSmuFeatures',
    'PPSMC_MSG_DisableDeterminism',
    'PPSMC_MSG_DramLogSetDramAddrHigh',
    'PPSMC_MSG_DramLogSetDramAddrLow', 'PPSMC_MSG_DramLogSetDramSize',
    'PPSMC_MSG_DumpSTBtoDram', 'PPSMC_MSG_EnableAllSmuFeatures',
    'PPSMC_MSG_EnableDeterminism', 'PPSMC_MSG_EnterGfxoff',
    'PPSMC_MSG_ExitGfxoff', 'PPSMC_MSG_GetCTFLimit',
    'PPSMC_MSG_GetDebugData', 'PPSMC_MSG_GetDpmFreqByIndex',
    'PPSMC_MSG_GetDriverIfVersion', 'PPSMC_MSG_GetEccInfoTable',
    'PPSMC_MSG_GetEnabledSmuFeaturesHigh',
    'PPSMC_MSG_GetEnabledSmuFeaturesLow', 'PPSMC_MSG_GetGmiPwrDnHyst',
    'PPSMC_MSG_GetMaxDpmFreq', 'PPSMC_MSG_GetMaxGfxDpmFreq',
    'PPSMC_MSG_GetMetricsTable', 'PPSMC_MSG_GetMetricsVersion',
    'PPSMC_MSG_GetMinDpmFreq', 'PPSMC_MSG_GetMinGfxDpmFreq',
    'PPSMC_MSG_GetPhsDetResidency', 'PPSMC_MSG_GetPptLimit',
    'PPSMC_MSG_GetSmuVersion', 'PPSMC_MSG_GetStaticMetricsTable',
    'PPSMC_MSG_GfxDriverReset', 'PPSMC_MSG_GfxDriverResetRecovery',
    'PPSMC_MSG_GmiPwrDnControl', 'PPSMC_MSG_HeavySBR',
    'PPSMC_MSG_McaBankCeDumpDW', 'PPSMC_MSG_McaBankDumpDW',
    'PPSMC_MSG_PrepareForDriverUnload',
    'PPSMC_MSG_QueryValidMcaCeCount', 'PPSMC_MSG_QueryValidMcaCount',
    'PPSMC_MSG_ReadThrottlerLimit', 'PPSMC_MSG_RequestI2cTransaction',
    'PPSMC_MSG_ResetSDMA', 'PPSMC_MSG_RmaDueToBadPageThreshold',
    'PPSMC_MSG_STBtoDramLogSetDramAddrHigh',
    'PPSMC_MSG_STBtoDramLogSetDramAddrLow',
    'PPSMC_MSG_STBtoDramLogSetDramSize', 'PPSMC_MSG_SelectPLPDMode',
    'PPSMC_MSG_SetDriverDramAddrHigh',
    'PPSMC_MSG_SetDriverDramAddrLow', 'PPSMC_MSG_SetGmiPwrDnHyst',
    'PPSMC_MSG_SetNumBadHbmPagesRetired', 'PPSMC_MSG_SetPhsDetOnOff',
    'PPSMC_MSG_SetPhsDetWRbwAlpha', 'PPSMC_MSG_SetPhsDetWRbwFreqHigh',
    'PPSMC_MSG_SetPhsDetWRbwFreqLow',
    'PPSMC_MSG_SetPhsDetWRbwHystDown',
    'PPSMC_MSG_SetPhsDetWRbwThreshold', 'PPSMC_MSG_SetPptLimit',
    'PPSMC_MSG_SetSoftMaxByFreq', 'PPSMC_MSG_SetSoftMaxGfxClk',
    'PPSMC_MSG_SetSoftMinByFreq', 'PPSMC_MSG_SetSoftMinGfxClk',
    'PPSMC_MSG_SetSystemVirtualDramAddrHigh',
    'PPSMC_MSG_SetSystemVirtualDramAddrLow',
    'PPSMC_MSG_SetSystemVirtualSTBtoDramAddrHigh',
    'PPSMC_MSG_SetSystemVirtualSTBtoDramAddrLow',
    'PPSMC_MSG_SetThrottlingPolicy', 'PPSMC_MSG_SetToolsDramAddrHigh',
    'PPSMC_MSG_SetToolsDramAddrLow', 'PPSMC_MSG_TestMessage',
    'PPSMC_MSG_TriggerVFFLR', 'PPSMC_Message_Count',
    'PPSMC_PLPD_MODE_DEFAULT', 'PPSMC_PLPD_MODE_OPTIMIZED',
    'PPSMC_RESET_TYPE_DRIVER_MODE_1_RESET',
    'PPSMC_RESET_TYPE_DRIVER_MODE_2_RESET',
    'PPSMC_RESET_TYPE_DRIVER_MODE_3_RESET', 'PPSMC_Result',
    'PPSMC_Result_CmdRejectedBusy', 'PPSMC_Result_CmdRejectedPrereq',
    'PPSMC_Result_Failed', 'PPSMC_Result_OK',
    'PPSMC_Result_UnknownCmd', 'PPSMC_THROTTLING_LIMIT_TYPE_HBM',
    'PPSMC_THROTTLING_LIMIT_TYPE_SOCKET', 'PPSMC_XCD_THM_TYPE',
    'RLC_CODE', 'SH_FED_CODE', 'SMU13_0_6_DRIVER_IF_VERSION',
    'SMU_13_0_6_DRIVER_IF_H', 'SMU_13_0_6_PPSMC_H', 'SMU_CLK_COUNT',
    'SMU_CUSTOM_FAN_SPEED_PWM', 'SMU_CUSTOM_FAN_SPEED_RPM',
    'SMU_DCEFCLK', 'SMU_DCLK', 'SMU_DCLK1', 'SMU_DEFAULT_PPT_LIMIT',
    'SMU_DISPCLK', 'SMU_DPM_USER_PROFILE_RESTORE', 'SMU_ECLK',
    'SMU_FAST_PPT_LIMIT', 'SMU_FCLK', 'SMU_FW_NAME_LEN', 'SMU_GFXCLK',
    'SMU_LCLK', 'SMU_MCLK', 'SMU_MEMORY_POOL_SIZE_1_GB',
    'SMU_MEMORY_POOL_SIZE_256_MB', 'SMU_MEMORY_POOL_SIZE_2_GB',
    'SMU_MEMORY_POOL_SIZE_512_MB', 'SMU_MEMORY_POOL_SIZE_ZERO',
    'SMU_OD_ACOUSTIC_LIMIT', 'SMU_OD_ACOUSTIC_TARGET', 'SMU_OD_CCLK',
    'SMU_OD_FAN_CURVE', 'SMU_OD_FAN_MINIMUM_PWM',
    'SMU_OD_FAN_TARGET_TEMPERATURE', 'SMU_OD_MCLK', 'SMU_OD_RANGE',
    'SMU_OD_SCLK', 'SMU_OD_VDDC_CURVE', 'SMU_OD_VDDGFX_OFFSET',
    'SMU_PCIE', 'SMU_PHYCLK', 'SMU_PIXCLK', 'SMU_POWER_SOURCE_AC',
    'SMU_POWER_SOURCE_COUNT', 'SMU_POWER_SOURCE_DC',
    'SMU_PPT_LIMIT_CURRENT', 'SMU_PPT_LIMIT_DEFAULT',
    'SMU_PPT_LIMIT_MAX', 'SMU_PPT_LIMIT_MIN',
    'SMU_REFRESHRATE_SOURCE_EDID', 'SMU_REFRESHRATE_SOURCE_EXPLICIT',
    'SMU_SCLK', 'SMU_SOCCLK',
    'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE',
    'SMU_STATE_CLASSIFICATIN_FLAG_LIMITED_POWER_SOURCE2',
    'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_3D_PERFORMANCE_LOW',
    'SMU_STATE_CLASSIFICATION_FLAG_ACPI',
    'SMU_STATE_CLASSIFICATION_FLAG_AC_OVERDIRVER_TEMPLATE',
    'SMU_STATE_CLASSIFICATION_FLAG_BACO',
    'SMU_STATE_CLASSIFICATION_FLAG_BOOT',
    'SMU_STATE_CLASSIFICATION_FLAG_DC_OVERDIRVER_TEMPLATE',
    'SMU_STATE_CLASSIFICATION_FLAG_FORCED',
    'SMU_STATE_CLASSIFICATION_FLAG_HD2',
    'SMU_STATE_CLASSIFICATION_FLAG_RESET',
    'SMU_STATE_CLASSIFICATION_FLAG_THERMAL',
    'SMU_STATE_CLASSIFICATION_FLAG_ULV',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_2D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_3D_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_USER_DC_PERFORMANCE',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_HD',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_MVC',
    'SMU_STATE_CLASSIFICATION_FLAG_UVD_SD', 'SMU_STATE_UI_LABEL_BACO',
    'SMU_STATE_UI_LABEL_BALLANCED', 'SMU_STATE_UI_LABEL_BATTERY',
    'SMU_STATE_UI_LABEL_MIDDLE_HIGHT', 'SMU_STATE_UI_LABEL_NONE',
    'SMU_STATE_UI_LABEL_PERFORMANCE', 'SMU_STATE_UI_TABEL_MIDDLE_LOW',
    'SMU_TABLE_ACTIVITY_MONITOR_COEFF', 'SMU_TABLE_AVFS',
    'SMU_TABLE_AVFS_FUSE_OVERRIDE', 'SMU_TABLE_AVFS_PSM_DEBUG',
    'SMU_TABLE_COMBO_PPTABLE', 'SMU_TABLE_COUNT',
    'SMU_TABLE_CUSTOM_DPM', 'SMU_TABLE_DPMCLOCKS',
    'SMU_TABLE_DRIVER_SMU_CONFIG', 'SMU_TABLE_ECCINFO',
    'SMU_TABLE_I2C_COMMANDS', 'SMU_TABLE_OVERDRIVE', 'SMU_TABLE_PACE',
    'SMU_TABLE_PMSTATUSLOG', 'SMU_TABLE_PPTABLE',
    'SMU_TABLE_SMU_METRICS', 'SMU_TABLE_WATERMARKS',
    'SMU_TABLE_WIFIBAND', 'SMU_TEMPERATURE_UNITS_PER_CENTIGRADES',
    'SMU_THERMAL_MAXIMUM_ALERT_TEMP',
    'SMU_THERMAL_MINIMUM_ALERT_TEMP', 'SMU_THROTTLER_APCC_BIT',
    'SMU_THROTTLER_EDC_CPU_BIT', 'SMU_THROTTLER_EDC_GFX_BIT',
    'SMU_THROTTLER_FIT_BIT', 'SMU_THROTTLER_FPPT_BIT',
    'SMU_THROTTLER_PPM_BIT', 'SMU_THROTTLER_PPT0_BIT',
    'SMU_THROTTLER_PPT1_BIT', 'SMU_THROTTLER_PPT2_BIT',
    'SMU_THROTTLER_PPT3_BIT', 'SMU_THROTTLER_PROCHOT_CPU_BIT',
    'SMU_THROTTLER_PROCHOT_GFX_BIT', 'SMU_THROTTLER_SPL_BIT',
    'SMU_THROTTLER_SPPT_APU_BIT', 'SMU_THROTTLER_SPPT_BIT',
    'SMU_THROTTLER_TDC_CVIP_BIT', 'SMU_THROTTLER_TDC_GFX_BIT',
    'SMU_THROTTLER_TDC_MEM_BIT', 'SMU_THROTTLER_TDC_SOC_BIT',
    'SMU_THROTTLER_TDC_VDD_BIT', 'SMU_THROTTLER_TEMP_CORE_BIT',
    'SMU_THROTTLER_TEMP_EDGE_BIT', 'SMU_THROTTLER_TEMP_GPU_BIT',
    'SMU_THROTTLER_TEMP_HOTSPOT_BIT',
    'SMU_THROTTLER_TEMP_LIQUID0_BIT',
    'SMU_THROTTLER_TEMP_LIQUID1_BIT', 'SMU_THROTTLER_TEMP_MEM_BIT',
    'SMU_THROTTLER_TEMP_SOC_BIT', 'SMU_THROTTLER_TEMP_VR_GFX_BIT',
    'SMU_THROTTLER_TEMP_VR_MEM0_BIT',
    'SMU_THROTTLER_TEMP_VR_MEM1_BIT', 'SMU_THROTTLER_TEMP_VR_SOC_BIT',
    'SMU_THROTTLER_VRHOT0_BIT', 'SMU_THROTTLER_VRHOT1_BIT',
    'SMU_UCLK', 'SMU_VCLK', 'SMU_VCLK1', 'SP0_CODE', 'SP1_CODE',
    'SPI_CODE', 'SQC_CODE', 'SQ_CODE', 'SwI2cCmd_t',
    'SwI2cRequestExternal_t', 'SwI2cRequest_t', 'TA_CODE', 'TCA_CODE',
    'TCC_CODE', 'TCI_CODE', 'TCP_CODE', 'TCX_CODE', 'TD_CODE',
    'THROTTLER_PPT_BIT', 'THROTTLER_PROCHOT_BIT',
    'THROTTLER_THERMAL_HBM_BIT', 'THROTTLER_THERMAL_SOCKET_BIT',
    'THROTTLER_THERMAL_VR_BIT', 'UCLK_DPM_MODE_BANDWIDTH',
    'UCLK_DPM_MODE_LATENCY', 'UCLK_DPM_MODE_e',
    'UCLK_DPM_MODE_e__enumvalues', 'UNSUPPORTED_1', 'UNSUPPORTED_2',
    'UNSUPPORTED_3', 'VML2_CODE', 'VML2_WALKER_CODE',
    '__AMDGPU_SMU_H__', 'bool', 'c__EA_ERR_CODE_e',
    'c__EA_GC_ERROR_CODE_e', 'c__EA_GpioIntPolarity_e',
    'c__EA_I2cCmdType_e', 'c__EA_I2cControllerPort_e',
    'c__EA_I2cSpeed_e', 'c__EA_PPCLK_e', 'c__EA_UCLK_DPM_MODE_e',
    'int16_t', 'int32_t', 'int8_t', 'smu_clk_type',
    'smu_memory_pool_size', 'smu_perf_level_designation',
    'smu_power_src_type', 'smu_ppt_limit_level', 'smu_ppt_limit_type',
    'smu_refreshrate_source', 'smu_state_classification_flag',
    'smu_state_ui_label', 'smu_table_id', 'struct_amdgpu_bo',
    'struct_c__SA_AvfsDebugTableAid_t',
    'struct_c__SA_AvfsDebugTableXcd_t', 'struct_c__SA_SwI2cCmd_t',
    'struct_c__SA_SwI2cRequestExternal_t',
    'struct_c__SA_SwI2cRequest_t', 'struct_smu_bios_boot_up_values',
    'struct_smu_clock_info', 'struct_smu_hw_power_state',
    'struct_smu_performance_level', 'struct_smu_power_state',
    'struct_smu_state_classification_block',
    'struct_smu_state_display_block', 'struct_smu_state_memory_block',
    'struct_smu_state_pcie_block',
    'struct_smu_state_software_algorithm_block',
    'struct_smu_state_validation_block', 'struct_smu_table',
    'struct_smu_temperature_range', 'struct_smu_user_dpm_profile',
    'struct_smu_uvd_clocks', 'u32', 'uint16_t', 'uint32_t',
    'uint64_t', 'uint8_t']
