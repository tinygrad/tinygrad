# -*- coding: utf-8 -*-
#
# TARGET arch is: []
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





_PSP_TEE_GFX_IF_H_ = True # macro
PSP_GFX_CMD_BUF_VERSION = 0x00000001 # macro
GFX_CMD_STATUS_MASK = 0x0000FFFF # macro
GFX_CMD_ID_MASK = 0x000F0000 # macro
GFX_CMD_RESERVED_MASK = 0x7FF00000 # macro
GFX_CMD_RESPONSE_MASK = 0x80000000 # macro
C2PMSG_CMD_GFX_USB_PD_FW_VER = 0x2000000 # macro
GFX_FLAG_RESPONSE = 0x80000000 # macro
GFX_BUF_MAX_DESC = 64 # macro
FRAME_TYPE_DESTROY = 1 # macro
PSP_ERR_UNKNOWN_COMMAND = 0x00000100 # macro

# values for enumeration 'psp_gfx_crtl_cmd_id'
psp_gfx_crtl_cmd_id__enumvalues = {
    65536: 'GFX_CTRL_CMD_ID_INIT_RBI_RING',
    131072: 'GFX_CTRL_CMD_ID_INIT_GPCOM_RING',
    196608: 'GFX_CTRL_CMD_ID_DESTROY_RINGS',
    262144: 'GFX_CTRL_CMD_ID_CAN_INIT_RINGS',
    327680: 'GFX_CTRL_CMD_ID_ENABLE_INT',
    393216: 'GFX_CTRL_CMD_ID_DISABLE_INT',
    458752: 'GFX_CTRL_CMD_ID_MODE1_RST',
    524288: 'GFX_CTRL_CMD_ID_GBR_IH_SET',
    589824: 'GFX_CTRL_CMD_ID_CONSUME_CMD',
    786432: 'GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING',
    983040: 'GFX_CTRL_CMD_ID_MAX',
}
GFX_CTRL_CMD_ID_INIT_RBI_RING = 65536
GFX_CTRL_CMD_ID_INIT_GPCOM_RING = 131072
GFX_CTRL_CMD_ID_DESTROY_RINGS = 196608
GFX_CTRL_CMD_ID_CAN_INIT_RINGS = 262144
GFX_CTRL_CMD_ID_ENABLE_INT = 327680
GFX_CTRL_CMD_ID_DISABLE_INT = 393216
GFX_CTRL_CMD_ID_MODE1_RST = 458752
GFX_CTRL_CMD_ID_GBR_IH_SET = 524288
GFX_CTRL_CMD_ID_CONSUME_CMD = 589824
GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING = 786432
GFX_CTRL_CMD_ID_MAX = 983040
psp_gfx_crtl_cmd_id = ctypes.c_uint32 # enum
class struct_psp_gfx_ctrl(Structure):
    pass

struct_psp_gfx_ctrl._pack_ = 1 # source:False
struct_psp_gfx_ctrl._fields_ = [
    ('cmd_resp', ctypes.c_uint32),
    ('rbi_wptr', ctypes.c_uint32),
    ('rbi_rptr', ctypes.c_uint32),
    ('gpcom_wptr', ctypes.c_uint32),
    ('gpcom_rptr', ctypes.c_uint32),
    ('ring_addr_lo', ctypes.c_uint32),
    ('ring_addr_hi', ctypes.c_uint32),
    ('ring_buf_size', ctypes.c_uint32),
]


# values for enumeration 'psp_gfx_cmd_id'
psp_gfx_cmd_id__enumvalues = {
    1: 'GFX_CMD_ID_LOAD_TA',
    2: 'GFX_CMD_ID_UNLOAD_TA',
    3: 'GFX_CMD_ID_INVOKE_CMD',
    4: 'GFX_CMD_ID_LOAD_ASD',
    5: 'GFX_CMD_ID_SETUP_TMR',
    6: 'GFX_CMD_ID_LOAD_IP_FW',
    7: 'GFX_CMD_ID_DESTROY_TMR',
    8: 'GFX_CMD_ID_SAVE_RESTORE',
    9: 'GFX_CMD_ID_SETUP_VMR',
    10: 'GFX_CMD_ID_DESTROY_VMR',
    11: 'GFX_CMD_ID_PROG_REG',
    15: 'GFX_CMD_ID_GET_FW_ATTESTATION',
    32: 'GFX_CMD_ID_LOAD_TOC',
    33: 'GFX_CMD_ID_AUTOLOAD_RLC',
    34: 'GFX_CMD_ID_BOOT_CFG',
    39: 'GFX_CMD_ID_SRIOV_SPATIAL_PART',
}
GFX_CMD_ID_LOAD_TA = 1
GFX_CMD_ID_UNLOAD_TA = 2
GFX_CMD_ID_INVOKE_CMD = 3
GFX_CMD_ID_LOAD_ASD = 4
GFX_CMD_ID_SETUP_TMR = 5
GFX_CMD_ID_LOAD_IP_FW = 6
GFX_CMD_ID_DESTROY_TMR = 7
GFX_CMD_ID_SAVE_RESTORE = 8
GFX_CMD_ID_SETUP_VMR = 9
GFX_CMD_ID_DESTROY_VMR = 10
GFX_CMD_ID_PROG_REG = 11
GFX_CMD_ID_GET_FW_ATTESTATION = 15
GFX_CMD_ID_LOAD_TOC = 32
GFX_CMD_ID_AUTOLOAD_RLC = 33
GFX_CMD_ID_BOOT_CFG = 34
GFX_CMD_ID_SRIOV_SPATIAL_PART = 39
psp_gfx_cmd_id = ctypes.c_uint32 # enum

# values for enumeration 'psp_gfx_boot_config_cmd'
psp_gfx_boot_config_cmd__enumvalues = {
    1: 'BOOTCFG_CMD_SET',
    2: 'BOOTCFG_CMD_GET',
    3: 'BOOTCFG_CMD_INVALIDATE',
}
BOOTCFG_CMD_SET = 1
BOOTCFG_CMD_GET = 2
BOOTCFG_CMD_INVALIDATE = 3
psp_gfx_boot_config_cmd = ctypes.c_uint32 # enum

# values for enumeration 'psp_gfx_boot_config'
psp_gfx_boot_config__enumvalues = {
    1: 'BOOT_CONFIG_GECC',
}
BOOT_CONFIG_GECC = 1
psp_gfx_boot_config = ctypes.c_uint32 # enum
class struct_psp_gfx_cmd_load_ta(Structure):
    pass

struct_psp_gfx_cmd_load_ta._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_ta._fields_ = [
    ('app_phy_addr_lo', ctypes.c_uint32),
    ('app_phy_addr_hi', ctypes.c_uint32),
    ('app_len', ctypes.c_uint32),
    ('cmd_buf_phy_addr_lo', ctypes.c_uint32),
    ('cmd_buf_phy_addr_hi', ctypes.c_uint32),
    ('cmd_buf_len', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_unload_ta(Structure):
    pass

struct_psp_gfx_cmd_unload_ta._pack_ = 1 # source:False
struct_psp_gfx_cmd_unload_ta._fields_ = [
    ('session_id', ctypes.c_uint32),
]

class struct_psp_gfx_buf_desc(Structure):
    pass

struct_psp_gfx_buf_desc._pack_ = 1 # source:False
struct_psp_gfx_buf_desc._fields_ = [
    ('buf_phy_addr_lo', ctypes.c_uint32),
    ('buf_phy_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
]

class struct_psp_gfx_buf_list(Structure):
    pass

struct_psp_gfx_buf_list._pack_ = 1 # source:False
struct_psp_gfx_buf_list._fields_ = [
    ('num_desc', ctypes.c_uint32),
    ('total_size', ctypes.c_uint32),
    ('buf_desc', struct_psp_gfx_buf_desc * 64),
]

class struct_psp_gfx_cmd_invoke_cmd(Structure):
    pass

struct_psp_gfx_cmd_invoke_cmd._pack_ = 1 # source:False
struct_psp_gfx_cmd_invoke_cmd._fields_ = [
    ('session_id', ctypes.c_uint32),
    ('ta_cmd_id', ctypes.c_uint32),
    ('buf', struct_psp_gfx_buf_list),
]

class struct_psp_gfx_cmd_setup_tmr(Structure):
    pass

class union_psp_gfx_cmd_setup_tmr_0(Union):
    pass

class struct_psp_gfx_cmd_setup_tmr_0_bitfield(Structure):
    pass

struct_psp_gfx_cmd_setup_tmr_0_bitfield._pack_ = 1 # source:False
struct_psp_gfx_cmd_setup_tmr_0_bitfield._fields_ = [
    ('sriov_enabled', ctypes.c_uint32, 1),
    ('virt_phy_addr', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 30),
]

union_psp_gfx_cmd_setup_tmr_0._pack_ = 1 # source:False
union_psp_gfx_cmd_setup_tmr_0._fields_ = [
    ('bitfield', struct_psp_gfx_cmd_setup_tmr_0_bitfield),
    ('tmr_flags', ctypes.c_uint32),
]

struct_psp_gfx_cmd_setup_tmr._pack_ = 1 # source:False
struct_psp_gfx_cmd_setup_tmr._anonymous_ = ('_0',)
struct_psp_gfx_cmd_setup_tmr._fields_ = [
    ('buf_phy_addr_lo', ctypes.c_uint32),
    ('buf_phy_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
    ('_0', union_psp_gfx_cmd_setup_tmr_0),
    ('system_phy_addr_lo', ctypes.c_uint32),
    ('system_phy_addr_hi', ctypes.c_uint32),
]


# values for enumeration 'psp_gfx_fw_type'
psp_gfx_fw_type__enumvalues = {
    0: 'GFX_FW_TYPE_NONE',
    1: 'GFX_FW_TYPE_CP_ME',
    2: 'GFX_FW_TYPE_CP_PFP',
    3: 'GFX_FW_TYPE_CP_CE',
    4: 'GFX_FW_TYPE_CP_MEC',
    5: 'GFX_FW_TYPE_CP_MEC_ME1',
    6: 'GFX_FW_TYPE_CP_MEC_ME2',
    7: 'GFX_FW_TYPE_RLC_V',
    8: 'GFX_FW_TYPE_RLC_G',
    9: 'GFX_FW_TYPE_SDMA0',
    10: 'GFX_FW_TYPE_SDMA1',
    11: 'GFX_FW_TYPE_DMCU_ERAM',
    12: 'GFX_FW_TYPE_DMCU_ISR',
    13: 'GFX_FW_TYPE_VCN',
    14: 'GFX_FW_TYPE_UVD',
    15: 'GFX_FW_TYPE_VCE',
    16: 'GFX_FW_TYPE_ISP',
    17: 'GFX_FW_TYPE_ACP',
    18: 'GFX_FW_TYPE_SMU',
    19: 'GFX_FW_TYPE_MMSCH',
    20: 'GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM',
    21: 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM',
    22: 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL',
    23: 'GFX_FW_TYPE_UVD1',
    24: 'GFX_FW_TYPE_TOC',
    25: 'GFX_FW_TYPE_RLC_P',
    26: 'GFX_FW_TYPE_RLC_IRAM',
    27: 'GFX_FW_TYPE_GLOBAL_TAP_DELAYS',
    28: 'GFX_FW_TYPE_SE0_TAP_DELAYS',
    29: 'GFX_FW_TYPE_SE1_TAP_DELAYS',
    30: 'GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS',
    31: 'GFX_FW_TYPE_SDMA0_JT',
    32: 'GFX_FW_TYPE_SDMA1_JT',
    33: 'GFX_FW_TYPE_CP_MES',
    34: 'GFX_FW_TYPE_MES_STACK',
    35: 'GFX_FW_TYPE_RLC_SRM_DRAM_SR',
    36: 'GFX_FW_TYPE_RLCG_SCRATCH_SR',
    37: 'GFX_FW_TYPE_RLCP_SCRATCH_SR',
    38: 'GFX_FW_TYPE_RLCV_SCRATCH_SR',
    39: 'GFX_FW_TYPE_RLX6_DRAM_SR',
    40: 'GFX_FW_TYPE_SDMA0_PG_CONTEXT',
    41: 'GFX_FW_TYPE_SDMA1_PG_CONTEXT',
    42: 'GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM',
    43: 'GFX_FW_TYPE_SE0_MUX_SELECT_RAM',
    44: 'GFX_FW_TYPE_SE1_MUX_SELECT_RAM',
    45: 'GFX_FW_TYPE_ACCUM_CTRL_RAM',
    46: 'GFX_FW_TYPE_RLCP_CAM',
    47: 'GFX_FW_TYPE_RLC_SPP_CAM_EXT',
    48: 'GFX_FW_TYPE_RLC_DRAM_BOOT',
    49: 'GFX_FW_TYPE_VCN0_RAM',
    50: 'GFX_FW_TYPE_VCN1_RAM',
    51: 'GFX_FW_TYPE_DMUB',
    52: 'GFX_FW_TYPE_SDMA2',
    53: 'GFX_FW_TYPE_SDMA3',
    54: 'GFX_FW_TYPE_SDMA4',
    55: 'GFX_FW_TYPE_SDMA5',
    56: 'GFX_FW_TYPE_SDMA6',
    57: 'GFX_FW_TYPE_SDMA7',
    58: 'GFX_FW_TYPE_VCN1',
    62: 'GFX_FW_TYPE_CAP',
    65: 'GFX_FW_TYPE_SE2_TAP_DELAYS',
    66: 'GFX_FW_TYPE_SE3_TAP_DELAYS',
    67: 'GFX_FW_TYPE_REG_LIST',
    68: 'GFX_FW_TYPE_IMU_I',
    69: 'GFX_FW_TYPE_IMU_D',
    70: 'GFX_FW_TYPE_LSDMA',
    71: 'GFX_FW_TYPE_SDMA_UCODE_TH0',
    72: 'GFX_FW_TYPE_SDMA_UCODE_TH1',
    73: 'GFX_FW_TYPE_PPTABLE',
    74: 'GFX_FW_TYPE_DISCRETE_USB4',
    75: 'GFX_FW_TYPE_TA',
    76: 'GFX_FW_TYPE_RS64_MES',
    77: 'GFX_FW_TYPE_RS64_MES_STACK',
    78: 'GFX_FW_TYPE_RS64_KIQ',
    79: 'GFX_FW_TYPE_RS64_KIQ_STACK',
    80: 'GFX_FW_TYPE_ISP_DATA',
    81: 'GFX_FW_TYPE_CP_MES_KIQ',
    82: 'GFX_FW_TYPE_MES_KIQ_STACK',
    83: 'GFX_FW_TYPE_UMSCH_DATA',
    84: 'GFX_FW_TYPE_UMSCH_UCODE',
    85: 'GFX_FW_TYPE_UMSCH_CMD_BUFFER',
    86: 'GFX_FW_TYPE_USB_DP_COMBO_PHY',
    87: 'GFX_FW_TYPE_RS64_PFP',
    88: 'GFX_FW_TYPE_RS64_ME',
    89: 'GFX_FW_TYPE_RS64_MEC',
    90: 'GFX_FW_TYPE_RS64_PFP_P0_STACK',
    91: 'GFX_FW_TYPE_RS64_PFP_P1_STACK',
    92: 'GFX_FW_TYPE_RS64_ME_P0_STACK',
    93: 'GFX_FW_TYPE_RS64_ME_P1_STACK',
    94: 'GFX_FW_TYPE_RS64_MEC_P0_STACK',
    95: 'GFX_FW_TYPE_RS64_MEC_P1_STACK',
    96: 'GFX_FW_TYPE_RS64_MEC_P2_STACK',
    97: 'GFX_FW_TYPE_RS64_MEC_P3_STACK',
    100: 'GFX_FW_TYPE_VPEC_FW1',
    101: 'GFX_FW_TYPE_VPEC_FW2',
    102: 'GFX_FW_TYPE_VPE',
    128: 'GFX_FW_TYPE_JPEG_RAM',
    129: 'GFX_FW_TYPE_P2S_TABLE',
    130: 'GFX_FW_TYPE_MAX',
}
GFX_FW_TYPE_NONE = 0
GFX_FW_TYPE_CP_ME = 1
GFX_FW_TYPE_CP_PFP = 2
GFX_FW_TYPE_CP_CE = 3
GFX_FW_TYPE_CP_MEC = 4
GFX_FW_TYPE_CP_MEC_ME1 = 5
GFX_FW_TYPE_CP_MEC_ME2 = 6
GFX_FW_TYPE_RLC_V = 7
GFX_FW_TYPE_RLC_G = 8
GFX_FW_TYPE_SDMA0 = 9
GFX_FW_TYPE_SDMA1 = 10
GFX_FW_TYPE_DMCU_ERAM = 11
GFX_FW_TYPE_DMCU_ISR = 12
GFX_FW_TYPE_VCN = 13
GFX_FW_TYPE_UVD = 14
GFX_FW_TYPE_VCE = 15
GFX_FW_TYPE_ISP = 16
GFX_FW_TYPE_ACP = 17
GFX_FW_TYPE_SMU = 18
GFX_FW_TYPE_MMSCH = 19
GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM = 20
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM = 21
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL = 22
GFX_FW_TYPE_UVD1 = 23
GFX_FW_TYPE_TOC = 24
GFX_FW_TYPE_RLC_P = 25
GFX_FW_TYPE_RLC_IRAM = 26
GFX_FW_TYPE_GLOBAL_TAP_DELAYS = 27
GFX_FW_TYPE_SE0_TAP_DELAYS = 28
GFX_FW_TYPE_SE1_TAP_DELAYS = 29
GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS = 30
GFX_FW_TYPE_SDMA0_JT = 31
GFX_FW_TYPE_SDMA1_JT = 32
GFX_FW_TYPE_CP_MES = 33
GFX_FW_TYPE_MES_STACK = 34
GFX_FW_TYPE_RLC_SRM_DRAM_SR = 35
GFX_FW_TYPE_RLCG_SCRATCH_SR = 36
GFX_FW_TYPE_RLCP_SCRATCH_SR = 37
GFX_FW_TYPE_RLCV_SCRATCH_SR = 38
GFX_FW_TYPE_RLX6_DRAM_SR = 39
GFX_FW_TYPE_SDMA0_PG_CONTEXT = 40
GFX_FW_TYPE_SDMA1_PG_CONTEXT = 41
GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM = 42
GFX_FW_TYPE_SE0_MUX_SELECT_RAM = 43
GFX_FW_TYPE_SE1_MUX_SELECT_RAM = 44
GFX_FW_TYPE_ACCUM_CTRL_RAM = 45
GFX_FW_TYPE_RLCP_CAM = 46
GFX_FW_TYPE_RLC_SPP_CAM_EXT = 47
GFX_FW_TYPE_RLC_DRAM_BOOT = 48
GFX_FW_TYPE_VCN0_RAM = 49
GFX_FW_TYPE_VCN1_RAM = 50
GFX_FW_TYPE_DMUB = 51
GFX_FW_TYPE_SDMA2 = 52
GFX_FW_TYPE_SDMA3 = 53
GFX_FW_TYPE_SDMA4 = 54
GFX_FW_TYPE_SDMA5 = 55
GFX_FW_TYPE_SDMA6 = 56
GFX_FW_TYPE_SDMA7 = 57
GFX_FW_TYPE_VCN1 = 58
GFX_FW_TYPE_CAP = 62
GFX_FW_TYPE_SE2_TAP_DELAYS = 65
GFX_FW_TYPE_SE3_TAP_DELAYS = 66
GFX_FW_TYPE_REG_LIST = 67
GFX_FW_TYPE_IMU_I = 68
GFX_FW_TYPE_IMU_D = 69
GFX_FW_TYPE_LSDMA = 70
GFX_FW_TYPE_SDMA_UCODE_TH0 = 71
GFX_FW_TYPE_SDMA_UCODE_TH1 = 72
GFX_FW_TYPE_PPTABLE = 73
GFX_FW_TYPE_DISCRETE_USB4 = 74
GFX_FW_TYPE_TA = 75
GFX_FW_TYPE_RS64_MES = 76
GFX_FW_TYPE_RS64_MES_STACK = 77
GFX_FW_TYPE_RS64_KIQ = 78
GFX_FW_TYPE_RS64_KIQ_STACK = 79
GFX_FW_TYPE_ISP_DATA = 80
GFX_FW_TYPE_CP_MES_KIQ = 81
GFX_FW_TYPE_MES_KIQ_STACK = 82
GFX_FW_TYPE_UMSCH_DATA = 83
GFX_FW_TYPE_UMSCH_UCODE = 84
GFX_FW_TYPE_UMSCH_CMD_BUFFER = 85
GFX_FW_TYPE_USB_DP_COMBO_PHY = 86
GFX_FW_TYPE_RS64_PFP = 87
GFX_FW_TYPE_RS64_ME = 88
GFX_FW_TYPE_RS64_MEC = 89
GFX_FW_TYPE_RS64_PFP_P0_STACK = 90
GFX_FW_TYPE_RS64_PFP_P1_STACK = 91
GFX_FW_TYPE_RS64_ME_P0_STACK = 92
GFX_FW_TYPE_RS64_ME_P1_STACK = 93
GFX_FW_TYPE_RS64_MEC_P0_STACK = 94
GFX_FW_TYPE_RS64_MEC_P1_STACK = 95
GFX_FW_TYPE_RS64_MEC_P2_STACK = 96
GFX_FW_TYPE_RS64_MEC_P3_STACK = 97
GFX_FW_TYPE_VPEC_FW1 = 100
GFX_FW_TYPE_VPEC_FW2 = 101
GFX_FW_TYPE_VPE = 102
GFX_FW_TYPE_JPEG_RAM = 128
GFX_FW_TYPE_P2S_TABLE = 129
GFX_FW_TYPE_MAX = 130
psp_gfx_fw_type = ctypes.c_uint32 # enum
class struct_psp_gfx_cmd_load_ip_fw(Structure):
    pass

struct_psp_gfx_cmd_load_ip_fw._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_ip_fw._fields_ = [
    ('fw_phy_addr_lo', ctypes.c_uint32),
    ('fw_phy_addr_hi', ctypes.c_uint32),
    ('fw_size', ctypes.c_uint32),
    ('fw_type', psp_gfx_fw_type),
]

class struct_psp_gfx_cmd_save_restore_ip_fw(Structure):
    pass

struct_psp_gfx_cmd_save_restore_ip_fw._pack_ = 1 # source:False
struct_psp_gfx_cmd_save_restore_ip_fw._fields_ = [
    ('save_fw', ctypes.c_uint32),
    ('save_restore_addr_lo', ctypes.c_uint32),
    ('save_restore_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
    ('fw_type', psp_gfx_fw_type),
]

class struct_psp_gfx_cmd_reg_prog(Structure):
    pass

struct_psp_gfx_cmd_reg_prog._pack_ = 1 # source:False
struct_psp_gfx_cmd_reg_prog._fields_ = [
    ('reg_value', ctypes.c_uint32),
    ('reg_id', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_load_toc(Structure):
    pass

struct_psp_gfx_cmd_load_toc._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_toc._fields_ = [
    ('toc_phy_addr_lo', ctypes.c_uint32),
    ('toc_phy_addr_hi', ctypes.c_uint32),
    ('toc_size', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_boot_cfg(Structure):
    pass

struct_psp_gfx_cmd_boot_cfg._pack_ = 1 # source:False
struct_psp_gfx_cmd_boot_cfg._fields_ = [
    ('timestamp', ctypes.c_uint32),
    ('sub_cmd', psp_gfx_boot_config_cmd),
    ('boot_config', ctypes.c_uint32),
    ('boot_config_valid', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_sriov_spatial_part(Structure):
    pass

struct_psp_gfx_cmd_sriov_spatial_part._pack_ = 1 # source:False
struct_psp_gfx_cmd_sriov_spatial_part._fields_ = [
    ('mode', ctypes.c_uint32),
    ('override_ips', ctypes.c_uint32),
    ('override_xcds_avail', ctypes.c_uint32),
    ('override_this_aid', ctypes.c_uint32),
]

class union_psp_gfx_commands(Union):
    pass

union_psp_gfx_commands._pack_ = 1 # source:False
union_psp_gfx_commands._fields_ = [
    ('cmd_load_ta', struct_psp_gfx_cmd_load_ta),
    ('cmd_unload_ta', struct_psp_gfx_cmd_unload_ta),
    ('cmd_invoke_cmd', struct_psp_gfx_cmd_invoke_cmd),
    ('cmd_setup_tmr', struct_psp_gfx_cmd_setup_tmr),
    ('cmd_load_ip_fw', struct_psp_gfx_cmd_load_ip_fw),
    ('cmd_save_restore_ip_fw', struct_psp_gfx_cmd_save_restore_ip_fw),
    ('cmd_setup_reg_prog', struct_psp_gfx_cmd_reg_prog),
    ('cmd_setup_vmr', struct_psp_gfx_cmd_setup_tmr),
    ('cmd_load_toc', struct_psp_gfx_cmd_load_toc),
    ('boot_cfg', struct_psp_gfx_cmd_boot_cfg),
    ('cmd_spatial_part', struct_psp_gfx_cmd_sriov_spatial_part),
    ('PADDING_0', ctypes.c_ubyte * 768),
]

class struct_psp_gfx_uresp_reserved(Structure):
    pass

struct_psp_gfx_uresp_reserved._pack_ = 1 # source:False
struct_psp_gfx_uresp_reserved._fields_ = [
    ('reserved', ctypes.c_uint32 * 8),
]

class struct_psp_gfx_uresp_fwar_db_info(Structure):
    pass

struct_psp_gfx_uresp_fwar_db_info._pack_ = 1 # source:False
struct_psp_gfx_uresp_fwar_db_info._fields_ = [
    ('fwar_db_addr_lo', ctypes.c_uint32),
    ('fwar_db_addr_hi', ctypes.c_uint32),
]

class struct_psp_gfx_uresp_bootcfg(Structure):
    pass

struct_psp_gfx_uresp_bootcfg._pack_ = 1 # source:False
struct_psp_gfx_uresp_bootcfg._fields_ = [
    ('boot_cfg', ctypes.c_uint32),
]

class union_psp_gfx_uresp(Union):
    pass

union_psp_gfx_uresp._pack_ = 1 # source:False
union_psp_gfx_uresp._fields_ = [
    ('reserved', struct_psp_gfx_uresp_reserved),
    ('boot_cfg', struct_psp_gfx_uresp_bootcfg),
    ('fwar_db_info', struct_psp_gfx_uresp_fwar_db_info),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

class struct_psp_gfx_resp(Structure):
    pass

struct_psp_gfx_resp._pack_ = 1 # source:False
struct_psp_gfx_resp._fields_ = [
    ('status', ctypes.c_uint32),
    ('session_id', ctypes.c_uint32),
    ('fw_addr_lo', ctypes.c_uint32),
    ('fw_addr_hi', ctypes.c_uint32),
    ('tmr_size', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 11),
    ('uresp', union_psp_gfx_uresp),
]

class struct_psp_gfx_cmd_resp(Structure):
    pass

struct_psp_gfx_cmd_resp._pack_ = 1 # source:False
struct_psp_gfx_cmd_resp._fields_ = [
    ('buf_size', ctypes.c_uint32),
    ('buf_version', ctypes.c_uint32),
    ('cmd_id', ctypes.c_uint32),
    ('resp_buf_addr_lo', ctypes.c_uint32),
    ('resp_buf_addr_hi', ctypes.c_uint32),
    ('resp_offset', ctypes.c_uint32),
    ('resp_buf_size', ctypes.c_uint32),
    ('cmd', union_psp_gfx_commands),
    ('reserved_1', ctypes.c_ubyte * 52),
    ('resp', struct_psp_gfx_resp),
    ('reserved_2', ctypes.c_ubyte * 64),
]

class struct_psp_gfx_rb_frame(Structure):
    pass

struct_psp_gfx_rb_frame._pack_ = 1 # source:False
struct_psp_gfx_rb_frame._fields_ = [
    ('cmd_buf_addr_lo', ctypes.c_uint32),
    ('cmd_buf_addr_hi', ctypes.c_uint32),
    ('cmd_buf_size', ctypes.c_uint32),
    ('fence_addr_lo', ctypes.c_uint32),
    ('fence_addr_hi', ctypes.c_uint32),
    ('fence_value', ctypes.c_uint32),
    ('sid_lo', ctypes.c_uint32),
    ('sid_hi', ctypes.c_uint32),
    ('vmid', ctypes.c_ubyte),
    ('frame_type', ctypes.c_ubyte),
    ('reserved1', ctypes.c_ubyte * 2),
    ('reserved2', ctypes.c_uint32 * 7),
]


# values for enumeration 'tee_error_code'
tee_error_code__enumvalues = {
    0: 'TEE_SUCCESS',
    4294901770: 'TEE_ERROR_NOT_SUPPORTED',
}
TEE_SUCCESS = 0
TEE_ERROR_NOT_SUPPORTED = 4294901770
tee_error_code = ctypes.c_uint32 # enum
__all__ = \
    ['BOOTCFG_CMD_GET', 'BOOTCFG_CMD_INVALIDATE', 'BOOTCFG_CMD_SET',
    'BOOT_CONFIG_GECC', 'C2PMSG_CMD_GFX_USB_PD_FW_VER',
    'FRAME_TYPE_DESTROY', 'GFX_BUF_MAX_DESC',
    'GFX_CMD_ID_AUTOLOAD_RLC', 'GFX_CMD_ID_BOOT_CFG',
    'GFX_CMD_ID_DESTROY_TMR', 'GFX_CMD_ID_DESTROY_VMR',
    'GFX_CMD_ID_GET_FW_ATTESTATION', 'GFX_CMD_ID_INVOKE_CMD',
    'GFX_CMD_ID_LOAD_ASD', 'GFX_CMD_ID_LOAD_IP_FW',
    'GFX_CMD_ID_LOAD_TA', 'GFX_CMD_ID_LOAD_TOC', 'GFX_CMD_ID_MASK',
    'GFX_CMD_ID_PROG_REG', 'GFX_CMD_ID_SAVE_RESTORE',
    'GFX_CMD_ID_SETUP_TMR', 'GFX_CMD_ID_SETUP_VMR',
    'GFX_CMD_ID_SRIOV_SPATIAL_PART', 'GFX_CMD_ID_UNLOAD_TA',
    'GFX_CMD_RESERVED_MASK', 'GFX_CMD_RESPONSE_MASK',
    'GFX_CMD_STATUS_MASK', 'GFX_CTRL_CMD_ID_CAN_INIT_RINGS',
    'GFX_CTRL_CMD_ID_CONSUME_CMD',
    'GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING',
    'GFX_CTRL_CMD_ID_DESTROY_RINGS', 'GFX_CTRL_CMD_ID_DISABLE_INT',
    'GFX_CTRL_CMD_ID_ENABLE_INT', 'GFX_CTRL_CMD_ID_GBR_IH_SET',
    'GFX_CTRL_CMD_ID_INIT_GPCOM_RING',
    'GFX_CTRL_CMD_ID_INIT_RBI_RING', 'GFX_CTRL_CMD_ID_MAX',
    'GFX_CTRL_CMD_ID_MODE1_RST', 'GFX_FLAG_RESPONSE',
    'GFX_FW_TYPE_ACCUM_CTRL_RAM', 'GFX_FW_TYPE_ACP',
    'GFX_FW_TYPE_CAP', 'GFX_FW_TYPE_CP_CE', 'GFX_FW_TYPE_CP_ME',
    'GFX_FW_TYPE_CP_MEC', 'GFX_FW_TYPE_CP_MEC_ME1',
    'GFX_FW_TYPE_CP_MEC_ME2', 'GFX_FW_TYPE_CP_MES',
    'GFX_FW_TYPE_CP_MES_KIQ', 'GFX_FW_TYPE_CP_PFP',
    'GFX_FW_TYPE_DISCRETE_USB4', 'GFX_FW_TYPE_DMCU_ERAM',
    'GFX_FW_TYPE_DMCU_ISR', 'GFX_FW_TYPE_DMUB',
    'GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM',
    'GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS',
    'GFX_FW_TYPE_GLOBAL_TAP_DELAYS', 'GFX_FW_TYPE_IMU_D',
    'GFX_FW_TYPE_IMU_I', 'GFX_FW_TYPE_ISP', 'GFX_FW_TYPE_ISP_DATA',
    'GFX_FW_TYPE_JPEG_RAM', 'GFX_FW_TYPE_LSDMA', 'GFX_FW_TYPE_MAX',
    'GFX_FW_TYPE_MES_KIQ_STACK', 'GFX_FW_TYPE_MES_STACK',
    'GFX_FW_TYPE_MMSCH', 'GFX_FW_TYPE_NONE', 'GFX_FW_TYPE_P2S_TABLE',
    'GFX_FW_TYPE_PPTABLE', 'GFX_FW_TYPE_REG_LIST',
    'GFX_FW_TYPE_RLCG_SCRATCH_SR', 'GFX_FW_TYPE_RLCP_CAM',
    'GFX_FW_TYPE_RLCP_SCRATCH_SR', 'GFX_FW_TYPE_RLCV_SCRATCH_SR',
    'GFX_FW_TYPE_RLC_DRAM_BOOT', 'GFX_FW_TYPE_RLC_G',
    'GFX_FW_TYPE_RLC_IRAM', 'GFX_FW_TYPE_RLC_P',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM',
    'GFX_FW_TYPE_RLC_SPP_CAM_EXT', 'GFX_FW_TYPE_RLC_SRM_DRAM_SR',
    'GFX_FW_TYPE_RLC_V', 'GFX_FW_TYPE_RLX6_DRAM_SR',
    'GFX_FW_TYPE_RS64_KIQ', 'GFX_FW_TYPE_RS64_KIQ_STACK',
    'GFX_FW_TYPE_RS64_ME', 'GFX_FW_TYPE_RS64_MEC',
    'GFX_FW_TYPE_RS64_MEC_P0_STACK', 'GFX_FW_TYPE_RS64_MEC_P1_STACK',
    'GFX_FW_TYPE_RS64_MEC_P2_STACK', 'GFX_FW_TYPE_RS64_MEC_P3_STACK',
    'GFX_FW_TYPE_RS64_MES', 'GFX_FW_TYPE_RS64_MES_STACK',
    'GFX_FW_TYPE_RS64_ME_P0_STACK', 'GFX_FW_TYPE_RS64_ME_P1_STACK',
    'GFX_FW_TYPE_RS64_PFP', 'GFX_FW_TYPE_RS64_PFP_P0_STACK',
    'GFX_FW_TYPE_RS64_PFP_P1_STACK', 'GFX_FW_TYPE_SDMA0',
    'GFX_FW_TYPE_SDMA0_JT', 'GFX_FW_TYPE_SDMA0_PG_CONTEXT',
    'GFX_FW_TYPE_SDMA1', 'GFX_FW_TYPE_SDMA1_JT',
    'GFX_FW_TYPE_SDMA1_PG_CONTEXT', 'GFX_FW_TYPE_SDMA2',
    'GFX_FW_TYPE_SDMA3', 'GFX_FW_TYPE_SDMA4', 'GFX_FW_TYPE_SDMA5',
    'GFX_FW_TYPE_SDMA6', 'GFX_FW_TYPE_SDMA7',
    'GFX_FW_TYPE_SDMA_UCODE_TH0', 'GFX_FW_TYPE_SDMA_UCODE_TH1',
    'GFX_FW_TYPE_SE0_MUX_SELECT_RAM', 'GFX_FW_TYPE_SE0_TAP_DELAYS',
    'GFX_FW_TYPE_SE1_MUX_SELECT_RAM', 'GFX_FW_TYPE_SE1_TAP_DELAYS',
    'GFX_FW_TYPE_SE2_TAP_DELAYS', 'GFX_FW_TYPE_SE3_TAP_DELAYS',
    'GFX_FW_TYPE_SMU', 'GFX_FW_TYPE_TA', 'GFX_FW_TYPE_TOC',
    'GFX_FW_TYPE_UMSCH_CMD_BUFFER', 'GFX_FW_TYPE_UMSCH_DATA',
    'GFX_FW_TYPE_UMSCH_UCODE', 'GFX_FW_TYPE_USB_DP_COMBO_PHY',
    'GFX_FW_TYPE_UVD', 'GFX_FW_TYPE_UVD1', 'GFX_FW_TYPE_VCE',
    'GFX_FW_TYPE_VCN', 'GFX_FW_TYPE_VCN0_RAM', 'GFX_FW_TYPE_VCN1',
    'GFX_FW_TYPE_VCN1_RAM', 'GFX_FW_TYPE_VPE', 'GFX_FW_TYPE_VPEC_FW1',
    'GFX_FW_TYPE_VPEC_FW2', 'PSP_ERR_UNKNOWN_COMMAND',
    'PSP_GFX_CMD_BUF_VERSION', 'TEE_ERROR_NOT_SUPPORTED',
    'TEE_SUCCESS', '_PSP_TEE_GFX_IF_H_', 'psp_gfx_boot_config',
    'psp_gfx_boot_config_cmd', 'psp_gfx_cmd_id',
    'psp_gfx_crtl_cmd_id', 'psp_gfx_fw_type',
    'struct_psp_gfx_buf_desc', 'struct_psp_gfx_buf_list',
    'struct_psp_gfx_cmd_boot_cfg', 'struct_psp_gfx_cmd_invoke_cmd',
    'struct_psp_gfx_cmd_load_ip_fw', 'struct_psp_gfx_cmd_load_ta',
    'struct_psp_gfx_cmd_load_toc', 'struct_psp_gfx_cmd_reg_prog',
    'struct_psp_gfx_cmd_resp',
    'struct_psp_gfx_cmd_save_restore_ip_fw',
    'struct_psp_gfx_cmd_setup_tmr',
    'struct_psp_gfx_cmd_setup_tmr_0_bitfield',
    'struct_psp_gfx_cmd_sriov_spatial_part',
    'struct_psp_gfx_cmd_unload_ta', 'struct_psp_gfx_ctrl',
    'struct_psp_gfx_rb_frame', 'struct_psp_gfx_resp',
    'struct_psp_gfx_uresp_bootcfg',
    'struct_psp_gfx_uresp_fwar_db_info',
    'struct_psp_gfx_uresp_reserved', 'tee_error_code',
    'union_psp_gfx_cmd_setup_tmr_0', 'union_psp_gfx_commands',
    'union_psp_gfx_uresp']
