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



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



__AMDGPU_UCODE_H__ = True # macro
AMDGPU_SDMA0_UCODE_LOADED = 0x00000001 # macro
AMDGPU_SDMA1_UCODE_LOADED = 0x00000002 # macro
AMDGPU_CPCE_UCODE_LOADED = 0x00000004 # macro
AMDGPU_CPPFP_UCODE_LOADED = 0x00000008 # macro
AMDGPU_CPME_UCODE_LOADED = 0x00000010 # macro
AMDGPU_CPMEC1_UCODE_LOADED = 0x00000020 # macro
AMDGPU_CPMEC2_UCODE_LOADED = 0x00000040 # macro
AMDGPU_CPRLC_UCODE_LOADED = 0x00000100 # macro
class struct_common_firmware_header(Structure):
    pass

struct_common_firmware_header._pack_ = 1 # source:False
struct_common_firmware_header._fields_ = [
    ('size_bytes', ctypes.c_uint32),
    ('header_size_bytes', ctypes.c_uint32),
    ('header_version_major', ctypes.c_uint16),
    ('header_version_minor', ctypes.c_uint16),
    ('ip_version_major', ctypes.c_uint16),
    ('ip_version_minor', ctypes.c_uint16),
    ('ucode_version', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
    ('ucode_array_offset_bytes', ctypes.c_uint32),
    ('crc32', ctypes.c_uint32),
]

class struct_mc_firmware_header_v1_0(Structure):
    pass

struct_mc_firmware_header_v1_0._pack_ = 1 # source:False
struct_mc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('io_debug_size_bytes', ctypes.c_uint32),
    ('io_debug_array_offset_bytes', ctypes.c_uint32),
]

class struct_smc_firmware_header_v1_0(Structure):
    pass

struct_smc_firmware_header_v1_0._pack_ = 1 # source:False
struct_smc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_start_addr', ctypes.c_uint32),
]

class struct_smc_firmware_header_v2_0(Structure):
    pass

struct_smc_firmware_header_v2_0._pack_ = 1 # source:False
struct_smc_firmware_header_v2_0._fields_ = [
    ('v1_0', struct_smc_firmware_header_v1_0),
    ('ppt_offset_bytes', ctypes.c_uint32),
    ('ppt_size_bytes', ctypes.c_uint32),
]

class struct_smc_soft_pptable_entry(Structure):
    pass

struct_smc_soft_pptable_entry._pack_ = 1 # source:False
struct_smc_soft_pptable_entry._fields_ = [
    ('id', ctypes.c_uint32),
    ('ppt_offset_bytes', ctypes.c_uint32),
    ('ppt_size_bytes', ctypes.c_uint32),
]

class struct_smc_firmware_header_v2_1(Structure):
    pass

struct_smc_firmware_header_v2_1._pack_ = 1 # source:False
struct_smc_firmware_header_v2_1._fields_ = [
    ('v1_0', struct_smc_firmware_header_v1_0),
    ('pptable_count', ctypes.c_uint32),
    ('pptable_entry_offset', ctypes.c_uint32),
]

class struct_psp_fw_legacy_bin_desc(Structure):
    pass

struct_psp_fw_legacy_bin_desc._pack_ = 1 # source:False
struct_psp_fw_legacy_bin_desc._fields_ = [
    ('fw_version', ctypes.c_uint32),
    ('offset_bytes', ctypes.c_uint32),
    ('size_bytes', ctypes.c_uint32),
]

class struct_psp_firmware_header_v1_0(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_common_firmware_header),
    ('sos', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_1(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_0', struct_psp_firmware_header_v1_0),
    ('toc', struct_psp_fw_legacy_bin_desc),
    ('kdb', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_0', struct_psp_firmware_header_v1_0),
    ('res', struct_psp_fw_legacy_bin_desc),
    ('kdb', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_3(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_1', struct_psp_firmware_header_v1_1),
    ('spl', struct_psp_fw_legacy_bin_desc),
    ('rl', struct_psp_fw_legacy_bin_desc),
    ('sys_drv_aux', struct_psp_fw_legacy_bin_desc),
    ('sos_aux', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_fw_bin_desc(Structure):
    pass

struct_psp_fw_bin_desc._pack_ = 1 # source:False
struct_psp_fw_bin_desc._fields_ = [
    ('fw_type', ctypes.c_uint32),
    ('fw_version', ctypes.c_uint32),
    ('offset_bytes', ctypes.c_uint32),
    ('size_bytes', ctypes.c_uint32),
]

# UCODE_MAX_PSP_PACKAGING = (((ctypes.sizeof(amdgpu_firmware_header)-ctypes.sizeof(struct_common_firmware_header)-4)/ctypes.sizeof(struct_psp_fw_bin_desc))*2) # macro

# values for enumeration 'psp_fw_type'
psp_fw_type__enumvalues = {
    0: 'PSP_FW_TYPE_UNKOWN',
    1: 'PSP_FW_TYPE_PSP_SOS',
    2: 'PSP_FW_TYPE_PSP_SYS_DRV',
    3: 'PSP_FW_TYPE_PSP_KDB',
    4: 'PSP_FW_TYPE_PSP_TOC',
    5: 'PSP_FW_TYPE_PSP_SPL',
    6: 'PSP_FW_TYPE_PSP_RL',
    7: 'PSP_FW_TYPE_PSP_SOC_DRV',
    8: 'PSP_FW_TYPE_PSP_INTF_DRV',
    9: 'PSP_FW_TYPE_PSP_DBG_DRV',
    10: 'PSP_FW_TYPE_PSP_RAS_DRV',
    11: 'PSP_FW_TYPE_PSP_IPKEYMGR_DRV',
    12: 'PSP_FW_TYPE_MAX_INDEX',
}
PSP_FW_TYPE_UNKOWN = 0
PSP_FW_TYPE_PSP_SOS = 1
PSP_FW_TYPE_PSP_SYS_DRV = 2
PSP_FW_TYPE_PSP_KDB = 3
PSP_FW_TYPE_PSP_TOC = 4
PSP_FW_TYPE_PSP_SPL = 5
PSP_FW_TYPE_PSP_RL = 6
PSP_FW_TYPE_PSP_SOC_DRV = 7
PSP_FW_TYPE_PSP_INTF_DRV = 8
PSP_FW_TYPE_PSP_DBG_DRV = 9
PSP_FW_TYPE_PSP_RAS_DRV = 10
PSP_FW_TYPE_PSP_IPKEYMGR_DRV = 11
PSP_FW_TYPE_MAX_INDEX = 12
psp_fw_type = ctypes.c_uint32 # enum
class struct_psp_firmware_header_v2_0(Structure):
    pass

struct_psp_firmware_header_v2_0._pack_ = 1 # source:False
struct_psp_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('psp_fw_bin_count', ctypes.c_uint32),
    ('psp_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_psp_firmware_header_v2_1(Structure):
    pass

struct_psp_firmware_header_v2_1._pack_ = 1 # source:False
struct_psp_firmware_header_v2_1._fields_ = [
    ('header', struct_common_firmware_header),
    ('psp_fw_bin_count', ctypes.c_uint32),
    ('psp_aux_fw_bin_index', ctypes.c_uint32),
    ('psp_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_ta_firmware_header_v1_0(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_common_firmware_header),
    ('xgmi', struct_psp_fw_legacy_bin_desc),
    ('ras', struct_psp_fw_legacy_bin_desc),
    ('hdcp', struct_psp_fw_legacy_bin_desc),
    ('dtm', struct_psp_fw_legacy_bin_desc),
    ('securedisplay', struct_psp_fw_legacy_bin_desc),
     ]


# values for enumeration 'ta_fw_type'
ta_fw_type__enumvalues = {
    0: 'TA_FW_TYPE_UNKOWN',
    1: 'TA_FW_TYPE_PSP_ASD',
    2: 'TA_FW_TYPE_PSP_XGMI',
    3: 'TA_FW_TYPE_PSP_RAS',
    4: 'TA_FW_TYPE_PSP_HDCP',
    5: 'TA_FW_TYPE_PSP_DTM',
    6: 'TA_FW_TYPE_PSP_RAP',
    7: 'TA_FW_TYPE_PSP_SECUREDISPLAY',
    8: 'TA_FW_TYPE_MAX_INDEX',
}
TA_FW_TYPE_UNKOWN = 0
TA_FW_TYPE_PSP_ASD = 1
TA_FW_TYPE_PSP_XGMI = 2
TA_FW_TYPE_PSP_RAS = 3
TA_FW_TYPE_PSP_HDCP = 4
TA_FW_TYPE_PSP_DTM = 5
TA_FW_TYPE_PSP_RAP = 6
TA_FW_TYPE_PSP_SECUREDISPLAY = 7
TA_FW_TYPE_MAX_INDEX = 8
ta_fw_type = ctypes.c_uint32 # enum
class struct_ta_firmware_header_v2_0(Structure):
    pass

struct_ta_firmware_header_v2_0._pack_ = 1 # source:False
struct_ta_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ta_fw_bin_count', ctypes.c_uint32),
    ('ta_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_gfx_firmware_header_v1_0(Structure):
    pass

struct_gfx_firmware_header_v1_0._pack_ = 1 # source:False
struct_gfx_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
]

class struct_gfx_firmware_header_v2_0(Structure):
    pass

struct_gfx_firmware_header_v2_0._pack_ = 1 # source:False
struct_gfx_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
    ('ucode_offset_bytes', ctypes.c_uint32),
    ('data_size_bytes', ctypes.c_uint32),
    ('data_offset_bytes', ctypes.c_uint32),
    ('ucode_start_addr_lo', ctypes.c_uint32),
    ('ucode_start_addr_hi', ctypes.c_uint32),
]

class struct_mes_firmware_header_v1_0(Structure):
    pass

struct_mes_firmware_header_v1_0._pack_ = 1 # source:False
struct_mes_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('mes_ucode_version', ctypes.c_uint32),
    ('mes_ucode_size_bytes', ctypes.c_uint32),
    ('mes_ucode_offset_bytes', ctypes.c_uint32),
    ('mes_ucode_data_version', ctypes.c_uint32),
    ('mes_ucode_data_size_bytes', ctypes.c_uint32),
    ('mes_ucode_data_offset_bytes', ctypes.c_uint32),
    ('mes_uc_start_addr_lo', ctypes.c_uint32),
    ('mes_uc_start_addr_hi', ctypes.c_uint32),
    ('mes_data_start_addr_lo', ctypes.c_uint32),
    ('mes_data_start_addr_hi', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v1_0(Structure):
    pass

struct_rlc_firmware_header_v1_0._pack_ = 1 # source:False
struct_rlc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('save_and_restore_offset', ctypes.c_uint32),
    ('clear_state_descriptor_offset', ctypes.c_uint32),
    ('avail_scratch_ram_locations', ctypes.c_uint32),
    ('master_pkt_description_offset', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_0(Structure):
    pass

struct_rlc_firmware_header_v2_0._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
    ('save_and_restore_offset', ctypes.c_uint32),
    ('clear_state_descriptor_offset', ctypes.c_uint32),
    ('avail_scratch_ram_locations', ctypes.c_uint32),
    ('reg_restore_list_size', ctypes.c_uint32),
    ('reg_list_format_start', ctypes.c_uint32),
    ('reg_list_format_separate_start', ctypes.c_uint32),
    ('starting_offsets_start', ctypes.c_uint32),
    ('reg_list_format_size_bytes', ctypes.c_uint32),
    ('reg_list_format_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_size_bytes', ctypes.c_uint32),
    ('reg_list_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_format_separate_size_bytes', ctypes.c_uint32),
    ('reg_list_format_separate_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_separate_size_bytes', ctypes.c_uint32),
    ('reg_list_separate_array_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_1(Structure):
    pass

struct_rlc_firmware_header_v2_1._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_1._fields_ = [
    ('v2_0', struct_rlc_firmware_header_v2_0),
    ('reg_list_format_direct_reg_list_length', ctypes.c_uint32),
    ('save_restore_list_cntl_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_cntl_feature_ver', ctypes.c_uint32),
    ('save_restore_list_cntl_size_bytes', ctypes.c_uint32),
    ('save_restore_list_cntl_offset_bytes', ctypes.c_uint32),
    ('save_restore_list_gpm_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_gpm_feature_ver', ctypes.c_uint32),
    ('save_restore_list_gpm_size_bytes', ctypes.c_uint32),
    ('save_restore_list_gpm_offset_bytes', ctypes.c_uint32),
    ('save_restore_list_srm_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_srm_feature_ver', ctypes.c_uint32),
    ('save_restore_list_srm_size_bytes', ctypes.c_uint32),
    ('save_restore_list_srm_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_2(Structure):
    pass

struct_rlc_firmware_header_v2_2._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_2._fields_ = [
    ('v2_1', struct_rlc_firmware_header_v2_1),
    ('rlc_iram_ucode_size_bytes', ctypes.c_uint32),
    ('rlc_iram_ucode_offset_bytes', ctypes.c_uint32),
    ('rlc_dram_ucode_size_bytes', ctypes.c_uint32),
    ('rlc_dram_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_3(Structure):
    pass

struct_rlc_firmware_header_v2_3._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_3._fields_ = [
    ('v2_2', struct_rlc_firmware_header_v2_2),
    ('rlcp_ucode_version', ctypes.c_uint32),
    ('rlcp_ucode_feature_version', ctypes.c_uint32),
    ('rlcp_ucode_size_bytes', ctypes.c_uint32),
    ('rlcp_ucode_offset_bytes', ctypes.c_uint32),
    ('rlcv_ucode_version', ctypes.c_uint32),
    ('rlcv_ucode_feature_version', ctypes.c_uint32),
    ('rlcv_ucode_size_bytes', ctypes.c_uint32),
    ('rlcv_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_4(Structure):
    pass

struct_rlc_firmware_header_v2_4._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_4._fields_ = [
    ('v2_3', struct_rlc_firmware_header_v2_3),
    ('global_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('global_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se0_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se0_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se1_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se1_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se2_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se2_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se3_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se3_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v1_0(Structure):
    pass

struct_sdma_firmware_header_v1_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_change_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v1_1(Structure):
    pass

struct_sdma_firmware_header_v1_1._pack_ = 1 # source:False
struct_sdma_firmware_header_v1_1._fields_ = [
    ('v1_0', struct_sdma_firmware_header_v1_0),
    ('digest_size', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v2_0(Structure):
    pass

struct_sdma_firmware_header_v2_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ctx_ucode_size_bytes', ctypes.c_uint32),
    ('ctx_jt_offset', ctypes.c_uint32),
    ('ctx_jt_size', ctypes.c_uint32),
    ('ctl_ucode_offset', ctypes.c_uint32),
    ('ctl_ucode_size_bytes', ctypes.c_uint32),
    ('ctl_jt_offset', ctypes.c_uint32),
    ('ctl_jt_size', ctypes.c_uint32),
]

class struct_vpe_firmware_header_v1_0(Structure):
    pass

struct_vpe_firmware_header_v1_0._pack_ = 1 # source:False
struct_vpe_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ctx_ucode_size_bytes', ctypes.c_uint32),
    ('ctx_jt_offset', ctypes.c_uint32),
    ('ctx_jt_size', ctypes.c_uint32),
    ('ctl_ucode_offset', ctypes.c_uint32),
    ('ctl_ucode_size_bytes', ctypes.c_uint32),
    ('ctl_jt_offset', ctypes.c_uint32),
    ('ctl_jt_size', ctypes.c_uint32),
]

class struct_umsch_mm_firmware_header_v1_0(Structure):
    pass

struct_umsch_mm_firmware_header_v1_0._pack_ = 1 # source:False
struct_umsch_mm_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('umsch_mm_ucode_version', ctypes.c_uint32),
    ('umsch_mm_ucode_size_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_offset_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_data_version', ctypes.c_uint32),
    ('umsch_mm_ucode_data_size_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_data_offset_bytes', ctypes.c_uint32),
    ('umsch_mm_irq_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_irq_start_addr_hi', ctypes.c_uint32),
    ('umsch_mm_uc_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_uc_start_addr_hi', ctypes.c_uint32),
    ('umsch_mm_data_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_data_start_addr_hi', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v3_0(Structure):
    pass

struct_sdma_firmware_header_v3_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v3_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_offset_bytes', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
]

class struct_gpu_info_firmware_v1_0(Structure):
    pass

struct_gpu_info_firmware_v1_0._pack_ = 1 # source:False
struct_gpu_info_firmware_v1_0._fields_ = [
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_cu_per_sh', ctypes.c_uint32),
    ('gc_num_sh_per_se', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_tccs', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
]

class struct_gpu_info_firmware_v1_1(Structure):
    pass

struct_gpu_info_firmware_v1_1._pack_ = 1 # source:False
struct_gpu_info_firmware_v1_1._fields_ = [
    ('v1_0', struct_gpu_info_firmware_v1_0),
    ('num_sc_per_sh', ctypes.c_uint32),
    ('num_packer_per_sc', ctypes.c_uint32),
]

class struct_gpu_info_firmware_v1_2(Structure):
    pass

class struct_gpu_info_soc_bounding_box_v1_0(Structure):
    pass

class struct_gpu_info_voltage_scaling_v1_0(Structure):
    pass

struct_gpu_info_voltage_scaling_v1_0._pack_ = 1 # source:False
struct_gpu_info_voltage_scaling_v1_0._fields_ = [
    ('state', ctypes.c_uint32),
    ('dscclk_mhz', ctypes.c_uint32),
    ('dcfclk_mhz', ctypes.c_uint32),
    ('socclk_mhz', ctypes.c_uint32),
    ('dram_speed_mts', ctypes.c_uint32),
    ('fabricclk_mhz', ctypes.c_uint32),
    ('dispclk_mhz', ctypes.c_uint32),
    ('phyclk_mhz', ctypes.c_uint32),
    ('dppclk_mhz', ctypes.c_uint32),
]

struct_gpu_info_soc_bounding_box_v1_0._pack_ = 1 # source:False
struct_gpu_info_soc_bounding_box_v1_0._fields_ = [
    ('sr_exit_time_us', ctypes.c_uint32),
    ('sr_enter_plus_exit_time_us', ctypes.c_uint32),
    ('urgent_latency_us', ctypes.c_uint32),
    ('urgent_latency_pixel_data_only_us', ctypes.c_uint32),
    ('urgent_latency_pixel_mixed_with_vm_data_us', ctypes.c_uint32),
    ('urgent_latency_vm_data_only_us', ctypes.c_uint32),
    ('writeback_latency_us', ctypes.c_uint32),
    ('ideal_dram_bw_after_urgent_percent', ctypes.c_uint32),
    ('pct_ideal_dram_sdp_bw_after_urgent_pixel_only', ctypes.c_uint32),
    ('pct_ideal_dram_sdp_bw_after_urgent_pixel_and_vm', ctypes.c_uint32),
    ('pct_ideal_dram_sdp_bw_after_urgent_vm_only', ctypes.c_uint32),
    ('max_avg_sdp_bw_use_normal_percent', ctypes.c_uint32),
    ('max_avg_dram_bw_use_normal_percent', ctypes.c_uint32),
    ('max_request_size_bytes', ctypes.c_uint32),
    ('downspread_percent', ctypes.c_uint32),
    ('dram_page_open_time_ns', ctypes.c_uint32),
    ('dram_rw_turnaround_time_ns', ctypes.c_uint32),
    ('dram_return_buffer_per_channel_bytes', ctypes.c_uint32),
    ('dram_channel_width_bytes', ctypes.c_uint32),
    ('fabric_datapath_to_dcn_data_return_bytes', ctypes.c_uint32),
    ('dcn_downspread_percent', ctypes.c_uint32),
    ('dispclk_dppclk_vco_speed_mhz', ctypes.c_uint32),
    ('dfs_vco_period_ps', ctypes.c_uint32),
    ('urgent_out_of_order_return_per_channel_pixel_only_bytes', ctypes.c_uint32),
    ('urgent_out_of_order_return_per_channel_pixel_and_vm_bytes', ctypes.c_uint32),
    ('urgent_out_of_order_return_per_channel_vm_only_bytes', ctypes.c_uint32),
    ('round_trip_ping_latency_dcfclk_cycles', ctypes.c_uint32),
    ('urgent_out_of_order_return_per_channel_bytes', ctypes.c_uint32),
    ('channel_interleave_bytes', ctypes.c_uint32),
    ('num_banks', ctypes.c_uint32),
    ('num_chans', ctypes.c_uint32),
    ('vmm_page_size_bytes', ctypes.c_uint32),
    ('dram_clock_change_latency_us', ctypes.c_uint32),
    ('writeback_dram_clock_change_latency_us', ctypes.c_uint32),
    ('return_bus_width_bytes', ctypes.c_uint32),
    ('voltage_override', ctypes.c_uint32),
    ('xfc_bus_transport_time_us', ctypes.c_uint32),
    ('xfc_xbuf_latency_tolerance_us', ctypes.c_uint32),
    ('use_urgent_burst_bw', ctypes.c_uint32),
    ('num_states', ctypes.c_uint32),
    ('clock_limits', struct_gpu_info_voltage_scaling_v1_0 * 8),
]

struct_gpu_info_firmware_v1_2._pack_ = 1 # source:False
struct_gpu_info_firmware_v1_2._fields_ = [
    ('v1_1', struct_gpu_info_firmware_v1_1),
    ('soc_bounding_box', struct_gpu_info_soc_bounding_box_v1_0),
]

class struct_gpu_info_firmware_header_v1_0(Structure):
    pass

struct_gpu_info_firmware_header_v1_0._pack_ = 1 # source:False
struct_gpu_info_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
]

class struct_dmcu_firmware_header_v1_0(Structure):
    pass

struct_dmcu_firmware_header_v1_0._pack_ = 1 # source:False
struct_dmcu_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('intv_offset_bytes', ctypes.c_uint32),
    ('intv_size_bytes', ctypes.c_uint32),
]

class struct_dmcub_firmware_header_v1_0(Structure):
    pass

struct_dmcub_firmware_header_v1_0._pack_ = 1 # source:False
struct_dmcub_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('inst_const_bytes', ctypes.c_uint32),
    ('bss_data_bytes', ctypes.c_uint32),
]

class struct_imu_firmware_header_v1_0(Structure):
    pass

struct_imu_firmware_header_v1_0._pack_ = 1 # source:False
struct_imu_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('imu_iram_ucode_size_bytes', ctypes.c_uint32),
    ('imu_iram_ucode_offset_bytes', ctypes.c_uint32),
    ('imu_dram_ucode_size_bytes', ctypes.c_uint32),
    ('imu_dram_ucode_offset_bytes', ctypes.c_uint32),
]

class union_amdgpu_firmware_header(Union):
    pass

union_amdgpu_firmware_header._pack_ = 1 # source:False
union_amdgpu_firmware_header._fields_ = [
    ('common', struct_common_firmware_header),
    ('mc', struct_mc_firmware_header_v1_0),
    ('smc', struct_smc_firmware_header_v1_0),
    ('smc_v2_0', struct_smc_firmware_header_v2_0),
    ('psp', struct_psp_firmware_header_v1_0),
    ('psp_v1_1', struct_psp_firmware_header_v1_1),
    ('psp_v1_3', struct_psp_firmware_header_v1_3),
    ('psp_v2_0', struct_psp_firmware_header_v2_0),
    ('psp_v2_1', struct_psp_firmware_header_v2_0),
    ('ta', struct_ta_firmware_header_v1_0),
    ('ta_v2_0', struct_ta_firmware_header_v2_0),
    ('gfx', struct_gfx_firmware_header_v1_0),
    ('gfx_v2_0', struct_gfx_firmware_header_v2_0),
    ('rlc', struct_rlc_firmware_header_v1_0),
    ('rlc_v2_0', struct_rlc_firmware_header_v2_0),
    ('rlc_v2_1', struct_rlc_firmware_header_v2_1),
    ('rlc_v2_2', struct_rlc_firmware_header_v2_2),
    ('rlc_v2_3', struct_rlc_firmware_header_v2_3),
    ('rlc_v2_4', struct_rlc_firmware_header_v2_4),
    ('sdma', struct_sdma_firmware_header_v1_0),
    ('sdma_v1_1', struct_sdma_firmware_header_v1_1),
    ('sdma_v2_0', struct_sdma_firmware_header_v2_0),
    ('sdma_v3_0', struct_sdma_firmware_header_v3_0),
    ('gpu_info', struct_gpu_info_firmware_header_v1_0),
    ('dmcu', struct_dmcu_firmware_header_v1_0),
    ('dmcub', struct_dmcub_firmware_header_v1_0),
    ('imu', struct_imu_firmware_header_v1_0),
    ('raw', ctypes.c_ubyte * 256),
]


# values for enumeration 'AMDGPU_UCODE_ID'
AMDGPU_UCODE_ID__enumvalues = {
    0: 'AMDGPU_UCODE_ID_CAP',
    1: 'AMDGPU_UCODE_ID_SDMA0',
    2: 'AMDGPU_UCODE_ID_SDMA1',
    3: 'AMDGPU_UCODE_ID_SDMA2',
    4: 'AMDGPU_UCODE_ID_SDMA3',
    5: 'AMDGPU_UCODE_ID_SDMA4',
    6: 'AMDGPU_UCODE_ID_SDMA5',
    7: 'AMDGPU_UCODE_ID_SDMA6',
    8: 'AMDGPU_UCODE_ID_SDMA7',
    9: 'AMDGPU_UCODE_ID_SDMA_UCODE_TH0',
    10: 'AMDGPU_UCODE_ID_SDMA_UCODE_TH1',
    11: 'AMDGPU_UCODE_ID_SDMA_RS64',
    12: 'AMDGPU_UCODE_ID_CP_CE',
    13: 'AMDGPU_UCODE_ID_CP_PFP',
    14: 'AMDGPU_UCODE_ID_CP_ME',
    15: 'AMDGPU_UCODE_ID_CP_RS64_PFP',
    16: 'AMDGPU_UCODE_ID_CP_RS64_ME',
    17: 'AMDGPU_UCODE_ID_CP_RS64_MEC',
    18: 'AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK',
    19: 'AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK',
    20: 'AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK',
    21: 'AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK',
    22: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK',
    23: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK',
    24: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK',
    25: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK',
    26: 'AMDGPU_UCODE_ID_CP_MEC1',
    27: 'AMDGPU_UCODE_ID_CP_MEC1_JT',
    28: 'AMDGPU_UCODE_ID_CP_MEC2',
    29: 'AMDGPU_UCODE_ID_CP_MEC2_JT',
    30: 'AMDGPU_UCODE_ID_CP_MES',
    31: 'AMDGPU_UCODE_ID_CP_MES_DATA',
    32: 'AMDGPU_UCODE_ID_CP_MES1',
    33: 'AMDGPU_UCODE_ID_CP_MES1_DATA',
    34: 'AMDGPU_UCODE_ID_IMU_I',
    35: 'AMDGPU_UCODE_ID_IMU_D',
    36: 'AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS',
    37: 'AMDGPU_UCODE_ID_SE0_TAP_DELAYS',
    38: 'AMDGPU_UCODE_ID_SE1_TAP_DELAYS',
    39: 'AMDGPU_UCODE_ID_SE2_TAP_DELAYS',
    40: 'AMDGPU_UCODE_ID_SE3_TAP_DELAYS',
    41: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL',
    42: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM',
    43: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM',
    44: 'AMDGPU_UCODE_ID_RLC_IRAM',
    45: 'AMDGPU_UCODE_ID_RLC_DRAM',
    46: 'AMDGPU_UCODE_ID_RLC_P',
    47: 'AMDGPU_UCODE_ID_RLC_V',
    48: 'AMDGPU_UCODE_ID_RLC_G',
    49: 'AMDGPU_UCODE_ID_STORAGE',
    50: 'AMDGPU_UCODE_ID_SMC',
    51: 'AMDGPU_UCODE_ID_PPTABLE',
    52: 'AMDGPU_UCODE_ID_UVD',
    53: 'AMDGPU_UCODE_ID_UVD1',
    54: 'AMDGPU_UCODE_ID_VCE',
    55: 'AMDGPU_UCODE_ID_VCN',
    56: 'AMDGPU_UCODE_ID_VCN1',
    57: 'AMDGPU_UCODE_ID_DMCU_ERAM',
    58: 'AMDGPU_UCODE_ID_DMCU_INTV',
    59: 'AMDGPU_UCODE_ID_VCN0_RAM',
    60: 'AMDGPU_UCODE_ID_VCN1_RAM',
    61: 'AMDGPU_UCODE_ID_DMCUB',
    62: 'AMDGPU_UCODE_ID_VPE_CTX',
    63: 'AMDGPU_UCODE_ID_VPE_CTL',
    64: 'AMDGPU_UCODE_ID_VPE',
    65: 'AMDGPU_UCODE_ID_UMSCH_MM_UCODE',
    66: 'AMDGPU_UCODE_ID_UMSCH_MM_DATA',
    67: 'AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER',
    68: 'AMDGPU_UCODE_ID_P2S_TABLE',
    69: 'AMDGPU_UCODE_ID_JPEG_RAM',
    70: 'AMDGPU_UCODE_ID_ISP',
    71: 'AMDGPU_UCODE_ID_MAXIMUM',
}
AMDGPU_UCODE_ID_CAP = 0
AMDGPU_UCODE_ID_SDMA0 = 1
AMDGPU_UCODE_ID_SDMA1 = 2
AMDGPU_UCODE_ID_SDMA2 = 3
AMDGPU_UCODE_ID_SDMA3 = 4
AMDGPU_UCODE_ID_SDMA4 = 5
AMDGPU_UCODE_ID_SDMA5 = 6
AMDGPU_UCODE_ID_SDMA6 = 7
AMDGPU_UCODE_ID_SDMA7 = 8
AMDGPU_UCODE_ID_SDMA_UCODE_TH0 = 9
AMDGPU_UCODE_ID_SDMA_UCODE_TH1 = 10
AMDGPU_UCODE_ID_SDMA_RS64 = 11
AMDGPU_UCODE_ID_CP_CE = 12
AMDGPU_UCODE_ID_CP_PFP = 13
AMDGPU_UCODE_ID_CP_ME = 14
AMDGPU_UCODE_ID_CP_RS64_PFP = 15
AMDGPU_UCODE_ID_CP_RS64_ME = 16
AMDGPU_UCODE_ID_CP_RS64_MEC = 17
AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK = 18
AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK = 19
AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK = 20
AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK = 21
AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK = 22
AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK = 23
AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK = 24
AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK = 25
AMDGPU_UCODE_ID_CP_MEC1 = 26
AMDGPU_UCODE_ID_CP_MEC1_JT = 27
AMDGPU_UCODE_ID_CP_MEC2 = 28
AMDGPU_UCODE_ID_CP_MEC2_JT = 29
AMDGPU_UCODE_ID_CP_MES = 30
AMDGPU_UCODE_ID_CP_MES_DATA = 31
AMDGPU_UCODE_ID_CP_MES1 = 32
AMDGPU_UCODE_ID_CP_MES1_DATA = 33
AMDGPU_UCODE_ID_IMU_I = 34
AMDGPU_UCODE_ID_IMU_D = 35
AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS = 36
AMDGPU_UCODE_ID_SE0_TAP_DELAYS = 37
AMDGPU_UCODE_ID_SE1_TAP_DELAYS = 38
AMDGPU_UCODE_ID_SE2_TAP_DELAYS = 39
AMDGPU_UCODE_ID_SE3_TAP_DELAYS = 40
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL = 41
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM = 42
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM = 43
AMDGPU_UCODE_ID_RLC_IRAM = 44
AMDGPU_UCODE_ID_RLC_DRAM = 45
AMDGPU_UCODE_ID_RLC_P = 46
AMDGPU_UCODE_ID_RLC_V = 47
AMDGPU_UCODE_ID_RLC_G = 48
AMDGPU_UCODE_ID_STORAGE = 49
AMDGPU_UCODE_ID_SMC = 50
AMDGPU_UCODE_ID_PPTABLE = 51
AMDGPU_UCODE_ID_UVD = 52
AMDGPU_UCODE_ID_UVD1 = 53
AMDGPU_UCODE_ID_VCE = 54
AMDGPU_UCODE_ID_VCN = 55
AMDGPU_UCODE_ID_VCN1 = 56
AMDGPU_UCODE_ID_DMCU_ERAM = 57
AMDGPU_UCODE_ID_DMCU_INTV = 58
AMDGPU_UCODE_ID_VCN0_RAM = 59
AMDGPU_UCODE_ID_VCN1_RAM = 60
AMDGPU_UCODE_ID_DMCUB = 61
AMDGPU_UCODE_ID_VPE_CTX = 62
AMDGPU_UCODE_ID_VPE_CTL = 63
AMDGPU_UCODE_ID_VPE = 64
AMDGPU_UCODE_ID_UMSCH_MM_UCODE = 65
AMDGPU_UCODE_ID_UMSCH_MM_DATA = 66
AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER = 67
AMDGPU_UCODE_ID_P2S_TABLE = 68
AMDGPU_UCODE_ID_JPEG_RAM = 69
AMDGPU_UCODE_ID_ISP = 70
AMDGPU_UCODE_ID_MAXIMUM = 71
AMDGPU_UCODE_ID = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_UCODE_STATUS'
AMDGPU_UCODE_STATUS__enumvalues = {
    0: 'AMDGPU_UCODE_STATUS_INVALID',
    1: 'AMDGPU_UCODE_STATUS_NOT_LOADED',
    2: 'AMDGPU_UCODE_STATUS_LOADED',
}
AMDGPU_UCODE_STATUS_INVALID = 0
AMDGPU_UCODE_STATUS_NOT_LOADED = 1
AMDGPU_UCODE_STATUS_LOADED = 2
AMDGPU_UCODE_STATUS = ctypes.c_uint32 # enum

# values for enumeration 'amdgpu_firmware_load_type'
amdgpu_firmware_load_type__enumvalues = {
    0: 'AMDGPU_FW_LOAD_DIRECT',
    1: 'AMDGPU_FW_LOAD_PSP',
    2: 'AMDGPU_FW_LOAD_SMU',
    3: 'AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO',
}
AMDGPU_FW_LOAD_DIRECT = 0
AMDGPU_FW_LOAD_PSP = 1
AMDGPU_FW_LOAD_SMU = 2
AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO = 3
amdgpu_firmware_load_type = ctypes.c_uint32 # enum
class struct_amdgpu_firmware_info(Structure):
    pass

class struct_firmware(Structure):
    pass

struct_amdgpu_firmware_info._pack_ = 1 # source:False
struct_amdgpu_firmware_info._fields_ = [
    ('ucode_id', AMDGPU_UCODE_ID),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fw', ctypes.POINTER(struct_firmware)),
    ('mc_addr', ctypes.c_uint64),
    ('kaddr', ctypes.POINTER(None)),
    ('ucode_size', ctypes.c_uint32),
    ('tmr_mc_addr_lo', ctypes.c_uint32),
    ('tmr_mc_addr_hi', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

__AMDGPU_RING_H__ = True # macro
# uint32_t = True # macro
# uint8_t = True # macro
# uint16_t = True # macro
# uint = True # macro
# u32 = True # macro
# u8 = True # macro
# u16 = True # macro
# u64 = True # macro
# bool = True # macro
AMDGPU_MAX_RINGS = 124 # macro
AMDGPU_MAX_HWIP_RINGS = 64 # macro
AMDGPU_MAX_GFX_RINGS = 2 # macro
AMDGPU_MAX_SW_GFX_RINGS = 2 # macro
AMDGPU_MAX_COMPUTE_RINGS = 8 # macro
AMDGPU_MAX_VCE_RINGS = 3 # macro
AMDGPU_MAX_UVD_ENC_RINGS = 2 # macro
AMDGPU_MAX_VPE_RINGS = 2 # macro
# AMDGPU_FENCE_OWNER_UNDEFINED = ((void*)0) # macro
# AMDGPU_FENCE_OWNER_VM = ((void*)1) # macro
# AMDGPU_FENCE_OWNER_KFD = ((void*)2) # macro
AMDGPU_FENCE_FLAG_64BIT = (1<<0) # macro
AMDGPU_FENCE_FLAG_INT = (1<<1) # macro
AMDGPU_FENCE_FLAG_TC_WB_ONLY = (1<<2) # macro
AMDGPU_FENCE_FLAG_EXEC = (1<<3) # macro
AMDGPU_IB_POOL_SIZE = (1024*1024) # macro

# values for enumeration 'amdgpu_ring_priority_level'
amdgpu_ring_priority_level__enumvalues = {
    0: 'AMDGPU_RING_PRIO_0',
    1: 'AMDGPU_RING_PRIO_1',
    1: 'AMDGPU_RING_PRIO_DEFAULT',
    2: 'AMDGPU_RING_PRIO_2',
    3: 'AMDGPU_RING_PRIO_MAX',
}
AMDGPU_RING_PRIO_0 = 0
AMDGPU_RING_PRIO_1 = 1
AMDGPU_RING_PRIO_DEFAULT = 1
AMDGPU_RING_PRIO_2 = 2
AMDGPU_RING_PRIO_MAX = 3
amdgpu_ring_priority_level = ctypes.c_uint32 # enum

# values for enumeration 'amdgpu_ib_pool_type'
amdgpu_ib_pool_type__enumvalues = {
    0: 'AMDGPU_IB_POOL_DELAYED',
    1: 'AMDGPU_IB_POOL_IMMEDIATE',
    2: 'AMDGPU_IB_POOL_DIRECT',
    3: 'AMDGPU_IB_POOL_MAX',
}
AMDGPU_IB_POOL_DELAYED = 0
AMDGPU_IB_POOL_IMMEDIATE = 1
AMDGPU_IB_POOL_DIRECT = 2
AMDGPU_IB_POOL_MAX = 3
amdgpu_ib_pool_type = ctypes.c_uint32 # enum
class struct_amdgpu_ring(Structure):
    pass

class struct_amdgpu_bo(Structure):
    pass

struct_amdgpu_ring._pack_ = 1 # source:False
struct_amdgpu_ring._fields_ = [
    ('ring', ctypes.POINTER(ctypes.c_uint32)),
    ('rptr_offs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('rptr_gpu_addr', ctypes.c_uint64),
    ('rptr_cpu_addr', ctypes.POINTER(ctypes.c_uint32)),
    ('wptr', ctypes.c_uint64),
    ('wptr_old', ctypes.c_uint64),
    ('ring_size', ctypes.c_uint32),
    ('max_dw', ctypes.c_uint32),
    ('count_dw', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('gpu_addr', ctypes.c_uint64),
    ('ptr_mask', ctypes.c_uint64),
    ('buf_mask', ctypes.c_uint32),
    ('idx', ctypes.c_uint32),
    ('xcc_id', ctypes.c_uint32),
    ('xcp_id', ctypes.c_uint32),
    ('me', ctypes.c_uint32),
    ('pipe', ctypes.c_uint32),
    ('queue', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('mqd_obj', ctypes.POINTER(struct_amdgpu_bo)),
    ('mqd_gpu_addr', ctypes.c_uint64),
    ('mqd_ptr', ctypes.POINTER(None)),
    ('mqd_size', ctypes.c_uint32),
    ('PADDING_3', ctypes.c_ubyte * 4),
    ('eop_gpu_addr', ctypes.c_uint64),
    ('doorbell_index', ctypes.c_uint32),
    ('use_doorbell', ctypes.c_ubyte),
    ('use_pollmem', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 2),
    ('wptr_offs', ctypes.c_uint32),
    ('PADDING_5', ctypes.c_ubyte * 4),
    ('wptr_gpu_addr', ctypes.c_uint64),
    ('wptr_cpu_addr', ctypes.POINTER(ctypes.c_uint32)),
    ('fence_offs', ctypes.c_uint32),
    ('PADDING_6', ctypes.c_ubyte * 4),
    ('fence_gpu_addr', ctypes.c_uint64),
    ('fence_cpu_addr', ctypes.POINTER(ctypes.c_uint32)),
    ('current_ctx', ctypes.c_uint64),
    ('name', ctypes.c_char * 16),
    ('trail_seq', ctypes.c_uint32),
    ('trail_fence_offs', ctypes.c_uint32),
    ('trail_fence_gpu_addr', ctypes.c_uint64),
    ('trail_fence_cpu_addr', ctypes.POINTER(ctypes.c_uint32)),
    ('cond_exe_offs', ctypes.c_uint32),
    ('PADDING_7', ctypes.c_ubyte * 4),
    ('cond_exe_gpu_addr', ctypes.c_uint64),
    ('cond_exe_cpu_addr', ctypes.POINTER(ctypes.c_uint32)),
    ('set_q_mode_offs', ctypes.c_uint32),
    ('PADDING_8', ctypes.c_ubyte * 4),
    ('set_q_mode_ptr', ctypes.POINTER(ctypes.c_uint32)),
    ('set_q_mode_token', ctypes.c_uint64),
    ('vm_hub', ctypes.c_uint32),
    ('vm_inv_eng', ctypes.c_uint32),
    ('has_compute_vm_bug', ctypes.c_ubyte),
    ('no_scheduler', ctypes.c_ubyte),
    ('PADDING_9', ctypes.c_ubyte * 2),
    ('hw_prio', ctypes.c_int32),
    ('num_hw_submission', ctypes.c_uint32),
    ('is_mes_queue', ctypes.c_ubyte),
    ('PADDING_10', ctypes.c_ubyte * 3),
    ('hw_queue_id', ctypes.c_uint32),
    ('is_sw_ring', ctypes.c_ubyte),
    ('PADDING_11', ctypes.c_ubyte * 3),
    ('entry_index', ctypes.c_uint32),
    ('PADDING_12', ctypes.c_ubyte * 4),
]

V11_STRUCTS_H_ = True # macro
class struct_v11_gfx_mqd(Structure):
    pass

struct_v11_gfx_mqd._pack_ = 1 # source:False
struct_v11_gfx_mqd._fields_ = [
    ('shadow_base_lo', ctypes.c_uint32),
    ('shadow_base_hi', ctypes.c_uint32),
    ('gds_bkup_base_lo', ctypes.c_uint32),
    ('gds_bkup_base_hi', ctypes.c_uint32),
    ('fw_work_area_base_lo', ctypes.c_uint32),
    ('fw_work_area_base_hi', ctypes.c_uint32),
    ('shadow_initialized', ctypes.c_uint32),
    ('ib_vmid', ctypes.c_uint32),
    ('reserved_8', ctypes.c_uint32),
    ('reserved_9', ctypes.c_uint32),
    ('reserved_10', ctypes.c_uint32),
    ('reserved_11', ctypes.c_uint32),
    ('reserved_12', ctypes.c_uint32),
    ('reserved_13', ctypes.c_uint32),
    ('reserved_14', ctypes.c_uint32),
    ('reserved_15', ctypes.c_uint32),
    ('reserved_16', ctypes.c_uint32),
    ('reserved_17', ctypes.c_uint32),
    ('reserved_18', ctypes.c_uint32),
    ('reserved_19', ctypes.c_uint32),
    ('reserved_20', ctypes.c_uint32),
    ('reserved_21', ctypes.c_uint32),
    ('reserved_22', ctypes.c_uint32),
    ('reserved_23', ctypes.c_uint32),
    ('reserved_24', ctypes.c_uint32),
    ('reserved_25', ctypes.c_uint32),
    ('reserved_26', ctypes.c_uint32),
    ('reserved_27', ctypes.c_uint32),
    ('reserved_28', ctypes.c_uint32),
    ('reserved_29', ctypes.c_uint32),
    ('reserved_30', ctypes.c_uint32),
    ('reserved_31', ctypes.c_uint32),
    ('reserved_32', ctypes.c_uint32),
    ('reserved_33', ctypes.c_uint32),
    ('reserved_34', ctypes.c_uint32),
    ('reserved_35', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('reserved_37', ctypes.c_uint32),
    ('reserved_38', ctypes.c_uint32),
    ('reserved_39', ctypes.c_uint32),
    ('reserved_40', ctypes.c_uint32),
    ('reserved_41', ctypes.c_uint32),
    ('reserved_42', ctypes.c_uint32),
    ('reserved_43', ctypes.c_uint32),
    ('reserved_44', ctypes.c_uint32),
    ('reserved_45', ctypes.c_uint32),
    ('reserved_46', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('checksum_lo', ctypes.c_uint32),
    ('checksum_hi', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('cp_mqd_query_wave_count', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_offset', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('control_buf_addr_lo', ctypes.c_uint32),
    ('control_buf_addr_hi', ctypes.c_uint32),
    ('disable_queue', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_active', ctypes.c_uint32),
    ('cp_gfx_hqd_vmid', ctypes.c_uint32),
    ('reserved_131', ctypes.c_uint32),
    ('reserved_132', ctypes.c_uint32),
    ('cp_gfx_hqd_queue_priority', ctypes.c_uint32),
    ('cp_gfx_hqd_quantum', ctypes.c_uint32),
    ('cp_gfx_hqd_base', ctypes.c_uint32),
    ('cp_gfx_hqd_base_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr_hi', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_rb_doorbell_control', ctypes.c_uint32),
    ('cp_gfx_hqd_offset', ctypes.c_uint32),
    ('cp_gfx_hqd_cntl', ctypes.c_uint32),
    ('reserved_146', ctypes.c_uint32),
    ('reserved_147', ctypes.c_uint32),
    ('cp_gfx_hqd_csmd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr_hi', ctypes.c_uint32),
    ('reserved_151', ctypes.c_uint32),
    ('reserved_152', ctypes.c_uint32),
    ('reserved_153', ctypes.c_uint32),
    ('reserved_154', ctypes.c_uint32),
    ('reserved_155', ctypes.c_uint32),
    ('cp_gfx_hqd_mapped', ctypes.c_uint32),
    ('cp_gfx_hqd_que_mgr_control', ctypes.c_uint32),
    ('reserved_158', ctypes.c_uint32),
    ('reserved_159', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_status0', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_control0', ctypes.c_uint32),
    ('cp_gfx_mqd_control', ctypes.c_uint32),
    ('reserved_163', ctypes.c_uint32),
    ('reserved_164', ctypes.c_uint32),
    ('reserved_165', ctypes.c_uint32),
    ('reserved_166', ctypes.c_uint32),
    ('reserved_167', ctypes.c_uint32),
    ('reserved_168', ctypes.c_uint32),
    ('reserved_169', ctypes.c_uint32),
    ('cp_num_prim_needed_count0_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count0_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count1_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count1_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count2_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count2_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count3_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count3_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count0_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count0_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count1_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count1_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count2_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count2_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count3_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count3_hi', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('mp1_smn_fps_cnt', ctypes.c_uint32),
    ('sq_thread_trace_buf0_base', ctypes.c_uint32),
    ('sq_thread_trace_buf0_size', ctypes.c_uint32),
    ('sq_thread_trace_buf1_base', ctypes.c_uint32),
    ('sq_thread_trace_buf1_size', ctypes.c_uint32),
    ('sq_thread_trace_wptr', ctypes.c_uint32),
    ('sq_thread_trace_mask', ctypes.c_uint32),
    ('sq_thread_trace_token_mask', ctypes.c_uint32),
    ('sq_thread_trace_ctrl', ctypes.c_uint32),
    ('sq_thread_trace_status', ctypes.c_uint32),
    ('sq_thread_trace_dropped_cntr', ctypes.c_uint32),
    ('sq_thread_trace_finish_done_debug', ctypes.c_uint32),
    ('sq_thread_trace_gfx_draw_cntr', ctypes.c_uint32),
    ('sq_thread_trace_gfx_marker_cntr', ctypes.c_uint32),
    ('sq_thread_trace_hp3d_draw_cntr', ctypes.c_uint32),
    ('sq_thread_trace_hp3d_marker_cntr', ctypes.c_uint32),
    ('reserved_206', ctypes.c_uint32),
    ('reserved_207', ctypes.c_uint32),
    ('cp_sc_psinvoc_count0_lo', ctypes.c_uint32),
    ('cp_sc_psinvoc_count0_hi', ctypes.c_uint32),
    ('cp_pa_cprim_count_lo', ctypes.c_uint32),
    ('cp_pa_cprim_count_hi', ctypes.c_uint32),
    ('cp_pa_cinvoc_count_lo', ctypes.c_uint32),
    ('cp_pa_cinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_vsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_vsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_gsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_gsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_gsprim_count_lo', ctypes.c_uint32),
    ('cp_vgt_gsprim_count_hi', ctypes.c_uint32),
    ('cp_vgt_iaprim_count_lo', ctypes.c_uint32),
    ('cp_vgt_iaprim_count_hi', ctypes.c_uint32),
    ('cp_vgt_iavert_count_lo', ctypes.c_uint32),
    ('cp_vgt_iavert_count_hi', ctypes.c_uint32),
    ('cp_vgt_hsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_hsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_dsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_dsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_csinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_csinvoc_count_hi', ctypes.c_uint32),
    ('reserved_230', ctypes.c_uint32),
    ('reserved_231', ctypes.c_uint32),
    ('reserved_232', ctypes.c_uint32),
    ('reserved_233', ctypes.c_uint32),
    ('reserved_234', ctypes.c_uint32),
    ('reserved_235', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('reserved_240', ctypes.c_uint32),
    ('reserved_241', ctypes.c_uint32),
    ('reserved_242', ctypes.c_uint32),
    ('reserved_243', ctypes.c_uint32),
    ('reserved_244', ctypes.c_uint32),
    ('reserved_245', ctypes.c_uint32),
    ('reserved_246', ctypes.c_uint32),
    ('reserved_247', ctypes.c_uint32),
    ('reserved_248', ctypes.c_uint32),
    ('reserved_249', ctypes.c_uint32),
    ('reserved_250', ctypes.c_uint32),
    ('reserved_251', ctypes.c_uint32),
    ('reserved_252', ctypes.c_uint32),
    ('reserved_253', ctypes.c_uint32),
    ('reserved_254', ctypes.c_uint32),
    ('reserved_255', ctypes.c_uint32),
    ('reserved_256', ctypes.c_uint32),
    ('reserved_257', ctypes.c_uint32),
    ('reserved_258', ctypes.c_uint32),
    ('reserved_259', ctypes.c_uint32),
    ('reserved_260', ctypes.c_uint32),
    ('reserved_261', ctypes.c_uint32),
    ('reserved_262', ctypes.c_uint32),
    ('reserved_263', ctypes.c_uint32),
    ('reserved_264', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_0', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_1', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_2', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_3', ctypes.c_uint32),
    ('reserved_272', ctypes.c_uint32),
    ('reserved_273', ctypes.c_uint32),
    ('reserved_274', ctypes.c_uint32),
    ('reserved_275', ctypes.c_uint32),
    ('vgt_dma_max_size', ctypes.c_uint32),
    ('vgt_dma_num_instances', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('it_set_base_ib_addr_lo', ctypes.c_uint32),
    ('it_set_base_ib_addr_hi', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_ps', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_vs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_gs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_hs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_ps', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_vs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_gs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_hs', ctypes.c_uint32),
    ('db_occlusion_count0_low_00', ctypes.c_uint32),
    ('db_occlusion_count0_hi_00', ctypes.c_uint32),
    ('db_occlusion_count1_low_00', ctypes.c_uint32),
    ('db_occlusion_count1_hi_00', ctypes.c_uint32),
    ('db_occlusion_count2_low_00', ctypes.c_uint32),
    ('db_occlusion_count2_hi_00', ctypes.c_uint32),
    ('db_occlusion_count3_low_00', ctypes.c_uint32),
    ('db_occlusion_count3_hi_00', ctypes.c_uint32),
    ('db_occlusion_count0_low_01', ctypes.c_uint32),
    ('db_occlusion_count0_hi_01', ctypes.c_uint32),
    ('db_occlusion_count1_low_01', ctypes.c_uint32),
    ('db_occlusion_count1_hi_01', ctypes.c_uint32),
    ('db_occlusion_count2_low_01', ctypes.c_uint32),
    ('db_occlusion_count2_hi_01', ctypes.c_uint32),
    ('db_occlusion_count3_low_01', ctypes.c_uint32),
    ('db_occlusion_count3_hi_01', ctypes.c_uint32),
    ('db_occlusion_count0_low_02', ctypes.c_uint32),
    ('db_occlusion_count0_hi_02', ctypes.c_uint32),
    ('db_occlusion_count1_low_02', ctypes.c_uint32),
    ('db_occlusion_count1_hi_02', ctypes.c_uint32),
    ('db_occlusion_count2_low_02', ctypes.c_uint32),
    ('db_occlusion_count2_hi_02', ctypes.c_uint32),
    ('db_occlusion_count3_low_02', ctypes.c_uint32),
    ('db_occlusion_count3_hi_02', ctypes.c_uint32),
    ('db_occlusion_count0_low_03', ctypes.c_uint32),
    ('db_occlusion_count0_hi_03', ctypes.c_uint32),
    ('db_occlusion_count1_low_03', ctypes.c_uint32),
    ('db_occlusion_count1_hi_03', ctypes.c_uint32),
    ('db_occlusion_count2_low_03', ctypes.c_uint32),
    ('db_occlusion_count2_hi_03', ctypes.c_uint32),
    ('db_occlusion_count3_low_03', ctypes.c_uint32),
    ('db_occlusion_count3_hi_03', ctypes.c_uint32),
    ('db_occlusion_count0_low_04', ctypes.c_uint32),
    ('db_occlusion_count0_hi_04', ctypes.c_uint32),
    ('db_occlusion_count1_low_04', ctypes.c_uint32),
    ('db_occlusion_count1_hi_04', ctypes.c_uint32),
    ('db_occlusion_count2_low_04', ctypes.c_uint32),
    ('db_occlusion_count2_hi_04', ctypes.c_uint32),
    ('db_occlusion_count3_low_04', ctypes.c_uint32),
    ('db_occlusion_count3_hi_04', ctypes.c_uint32),
    ('db_occlusion_count0_low_05', ctypes.c_uint32),
    ('db_occlusion_count0_hi_05', ctypes.c_uint32),
    ('db_occlusion_count1_low_05', ctypes.c_uint32),
    ('db_occlusion_count1_hi_05', ctypes.c_uint32),
    ('db_occlusion_count2_low_05', ctypes.c_uint32),
    ('db_occlusion_count2_hi_05', ctypes.c_uint32),
    ('db_occlusion_count3_low_05', ctypes.c_uint32),
    ('db_occlusion_count3_hi_05', ctypes.c_uint32),
    ('db_occlusion_count0_low_06', ctypes.c_uint32),
    ('db_occlusion_count0_hi_06', ctypes.c_uint32),
    ('db_occlusion_count1_low_06', ctypes.c_uint32),
    ('db_occlusion_count1_hi_06', ctypes.c_uint32),
    ('db_occlusion_count2_low_06', ctypes.c_uint32),
    ('db_occlusion_count2_hi_06', ctypes.c_uint32),
    ('db_occlusion_count3_low_06', ctypes.c_uint32),
    ('db_occlusion_count3_hi_06', ctypes.c_uint32),
    ('db_occlusion_count0_low_07', ctypes.c_uint32),
    ('db_occlusion_count0_hi_07', ctypes.c_uint32),
    ('db_occlusion_count1_low_07', ctypes.c_uint32),
    ('db_occlusion_count1_hi_07', ctypes.c_uint32),
    ('db_occlusion_count2_low_07', ctypes.c_uint32),
    ('db_occlusion_count2_hi_07', ctypes.c_uint32),
    ('db_occlusion_count3_low_07', ctypes.c_uint32),
    ('db_occlusion_count3_hi_07', ctypes.c_uint32),
    ('db_occlusion_count0_low_10', ctypes.c_uint32),
    ('db_occlusion_count0_hi_10', ctypes.c_uint32),
    ('db_occlusion_count1_low_10', ctypes.c_uint32),
    ('db_occlusion_count1_hi_10', ctypes.c_uint32),
    ('db_occlusion_count2_low_10', ctypes.c_uint32),
    ('db_occlusion_count2_hi_10', ctypes.c_uint32),
    ('db_occlusion_count3_low_10', ctypes.c_uint32),
    ('db_occlusion_count3_hi_10', ctypes.c_uint32),
    ('db_occlusion_count0_low_11', ctypes.c_uint32),
    ('db_occlusion_count0_hi_11', ctypes.c_uint32),
    ('db_occlusion_count1_low_11', ctypes.c_uint32),
    ('db_occlusion_count1_hi_11', ctypes.c_uint32),
    ('db_occlusion_count2_low_11', ctypes.c_uint32),
    ('db_occlusion_count2_hi_11', ctypes.c_uint32),
    ('db_occlusion_count3_low_11', ctypes.c_uint32),
    ('db_occlusion_count3_hi_11', ctypes.c_uint32),
    ('db_occlusion_count0_low_12', ctypes.c_uint32),
    ('db_occlusion_count0_hi_12', ctypes.c_uint32),
    ('db_occlusion_count1_low_12', ctypes.c_uint32),
    ('db_occlusion_count1_hi_12', ctypes.c_uint32),
    ('db_occlusion_count2_low_12', ctypes.c_uint32),
    ('db_occlusion_count2_hi_12', ctypes.c_uint32),
    ('db_occlusion_count3_low_12', ctypes.c_uint32),
    ('db_occlusion_count3_hi_12', ctypes.c_uint32),
    ('db_occlusion_count0_low_13', ctypes.c_uint32),
    ('db_occlusion_count0_hi_13', ctypes.c_uint32),
    ('db_occlusion_count1_low_13', ctypes.c_uint32),
    ('db_occlusion_count1_hi_13', ctypes.c_uint32),
    ('db_occlusion_count2_low_13', ctypes.c_uint32),
    ('db_occlusion_count2_hi_13', ctypes.c_uint32),
    ('db_occlusion_count3_low_13', ctypes.c_uint32),
    ('db_occlusion_count3_hi_13', ctypes.c_uint32),
    ('db_occlusion_count0_low_14', ctypes.c_uint32),
    ('db_occlusion_count0_hi_14', ctypes.c_uint32),
    ('db_occlusion_count1_low_14', ctypes.c_uint32),
    ('db_occlusion_count1_hi_14', ctypes.c_uint32),
    ('db_occlusion_count2_low_14', ctypes.c_uint32),
    ('db_occlusion_count2_hi_14', ctypes.c_uint32),
    ('db_occlusion_count3_low_14', ctypes.c_uint32),
    ('db_occlusion_count3_hi_14', ctypes.c_uint32),
    ('db_occlusion_count0_low_15', ctypes.c_uint32),
    ('db_occlusion_count0_hi_15', ctypes.c_uint32),
    ('db_occlusion_count1_low_15', ctypes.c_uint32),
    ('db_occlusion_count1_hi_15', ctypes.c_uint32),
    ('db_occlusion_count2_low_15', ctypes.c_uint32),
    ('db_occlusion_count2_hi_15', ctypes.c_uint32),
    ('db_occlusion_count3_low_15', ctypes.c_uint32),
    ('db_occlusion_count3_hi_15', ctypes.c_uint32),
    ('db_occlusion_count0_low_16', ctypes.c_uint32),
    ('db_occlusion_count0_hi_16', ctypes.c_uint32),
    ('db_occlusion_count1_low_16', ctypes.c_uint32),
    ('db_occlusion_count1_hi_16', ctypes.c_uint32),
    ('db_occlusion_count2_low_16', ctypes.c_uint32),
    ('db_occlusion_count2_hi_16', ctypes.c_uint32),
    ('db_occlusion_count3_low_16', ctypes.c_uint32),
    ('db_occlusion_count3_hi_16', ctypes.c_uint32),
    ('db_occlusion_count0_low_17', ctypes.c_uint32),
    ('db_occlusion_count0_hi_17', ctypes.c_uint32),
    ('db_occlusion_count1_low_17', ctypes.c_uint32),
    ('db_occlusion_count1_hi_17', ctypes.c_uint32),
    ('db_occlusion_count2_low_17', ctypes.c_uint32),
    ('db_occlusion_count2_hi_17', ctypes.c_uint32),
    ('db_occlusion_count3_low_17', ctypes.c_uint32),
    ('db_occlusion_count3_hi_17', ctypes.c_uint32),
    ('reserved_492', ctypes.c_uint32),
    ('reserved_493', ctypes.c_uint32),
    ('reserved_494', ctypes.c_uint32),
    ('reserved_495', ctypes.c_uint32),
    ('reserved_496', ctypes.c_uint32),
    ('reserved_497', ctypes.c_uint32),
    ('reserved_498', ctypes.c_uint32),
    ('reserved_499', ctypes.c_uint32),
    ('reserved_500', ctypes.c_uint32),
    ('reserved_501', ctypes.c_uint32),
    ('reserved_502', ctypes.c_uint32),
    ('reserved_503', ctypes.c_uint32),
    ('reserved_504', ctypes.c_uint32),
    ('reserved_505', ctypes.c_uint32),
    ('reserved_506', ctypes.c_uint32),
    ('reserved_507', ctypes.c_uint32),
    ('reserved_508', ctypes.c_uint32),
    ('reserved_509', ctypes.c_uint32),
    ('reserved_510', ctypes.c_uint32),
    ('reserved_511', ctypes.c_uint32),
]

class struct_v11_sdma_mqd(Structure):
    pass

struct_v11_sdma_mqd._pack_ = 1 # source:False
struct_v11_sdma_mqd._fields_ = [
    ('sdmax_rlcx_rb_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_ib_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_ib_offset', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_lo', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_ib_size', ctypes.c_uint32),
    ('sdmax_rlcx_skip_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_context_status', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_log', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_offset', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_sched_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_sub_remain', ctypes.c_uint32),
    ('sdmax_rlcx_preempt', ctypes.c_uint32),
    ('sdmax_rlcx_dummy_reg', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_rb_aql_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_minor_ptr_update', ctypes.c_uint32),
    ('sdmax_rlcx_rb_preempt', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data0', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data1', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data2', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data3', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data4', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data5', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data6', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data7', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data8', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data9', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data10', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_f32_dbg0', ctypes.c_uint32),
    ('sdmax_rlcx_f32_dbg1', ctypes.c_uint32),
    ('reserved_45', ctypes.c_uint32),
    ('reserved_46', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('reserved_86', ctypes.c_uint32),
    ('reserved_87', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('reserved_92', ctypes.c_uint32),
    ('reserved_93', ctypes.c_uint32),
    ('reserved_94', ctypes.c_uint32),
    ('reserved_95', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('reserved_104', ctypes.c_uint32),
    ('reserved_105', ctypes.c_uint32),
    ('reserved_106', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('sdma_engine_id', ctypes.c_uint32),
    ('sdma_queue_id', ctypes.c_uint32),
]

class struct_v11_compute_mqd(Structure):
    pass

struct_v11_compute_mqd._pack_ = 1 # source:False
struct_v11_compute_mqd._fields_ = [
    ('header', ctypes.c_uint32),
    ('compute_dispatch_initiator', ctypes.c_uint32),
    ('compute_dim_x', ctypes.c_uint32),
    ('compute_dim_y', ctypes.c_uint32),
    ('compute_dim_z', ctypes.c_uint32),
    ('compute_start_x', ctypes.c_uint32),
    ('compute_start_y', ctypes.c_uint32),
    ('compute_start_z', ctypes.c_uint32),
    ('compute_num_thread_x', ctypes.c_uint32),
    ('compute_num_thread_y', ctypes.c_uint32),
    ('compute_num_thread_z', ctypes.c_uint32),
    ('compute_pipelinestat_enable', ctypes.c_uint32),
    ('compute_perfcount_enable', ctypes.c_uint32),
    ('compute_pgm_lo', ctypes.c_uint32),
    ('compute_pgm_hi', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_lo', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_hi', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_lo', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_hi', ctypes.c_uint32),
    ('compute_pgm_rsrc1', ctypes.c_uint32),
    ('compute_pgm_rsrc2', ctypes.c_uint32),
    ('compute_vmid', ctypes.c_uint32),
    ('compute_resource_limits', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se0', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se1', ctypes.c_uint32),
    ('compute_tmpring_size', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se2', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se3', ctypes.c_uint32),
    ('compute_restart_x', ctypes.c_uint32),
    ('compute_restart_y', ctypes.c_uint32),
    ('compute_restart_z', ctypes.c_uint32),
    ('compute_thread_trace_enable', ctypes.c_uint32),
    ('compute_misc_reserved', ctypes.c_uint32),
    ('compute_dispatch_id', ctypes.c_uint32),
    ('compute_threadgroup_id', ctypes.c_uint32),
    ('compute_req_ctrl', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('compute_user_accum_0', ctypes.c_uint32),
    ('compute_user_accum_1', ctypes.c_uint32),
    ('compute_user_accum_2', ctypes.c_uint32),
    ('compute_user_accum_3', ctypes.c_uint32),
    ('compute_pgm_rsrc3', ctypes.c_uint32),
    ('compute_ddid_index', ctypes.c_uint32),
    ('compute_shader_chksum', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se4', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se5', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se6', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se7', ctypes.c_uint32),
    ('compute_dispatch_interleave', ctypes.c_uint32),
    ('compute_relaunch', ctypes.c_uint32),
    ('compute_wave_restore_addr_lo', ctypes.c_uint32),
    ('compute_wave_restore_addr_hi', ctypes.c_uint32),
    ('compute_wave_restore_control', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('compute_user_data_0', ctypes.c_uint32),
    ('compute_user_data_1', ctypes.c_uint32),
    ('compute_user_data_2', ctypes.c_uint32),
    ('compute_user_data_3', ctypes.c_uint32),
    ('compute_user_data_4', ctypes.c_uint32),
    ('compute_user_data_5', ctypes.c_uint32),
    ('compute_user_data_6', ctypes.c_uint32),
    ('compute_user_data_7', ctypes.c_uint32),
    ('compute_user_data_8', ctypes.c_uint32),
    ('compute_user_data_9', ctypes.c_uint32),
    ('compute_user_data_10', ctypes.c_uint32),
    ('compute_user_data_11', ctypes.c_uint32),
    ('compute_user_data_12', ctypes.c_uint32),
    ('compute_user_data_13', ctypes.c_uint32),
    ('compute_user_data_14', ctypes.c_uint32),
    ('compute_user_data_15', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_lo', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_hi', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_wf_count', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_rptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_wptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_ib_rptr', ctypes.c_uint32),
    ('cp_mqd_readindex_lo', ctypes.c_uint32),
    ('cp_mqd_readindex_hi', ctypes.c_uint32),
    ('cp_mqd_save_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_save_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_hi', ctypes.c_uint32),
    ('disable_queue', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt0', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt1', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt2', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt3', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('cp_pq_exe_status_lo', ctypes.c_uint32),
    ('cp_pq_exe_status_hi', ctypes.c_uint32),
    ('cp_packet_id_lo', ctypes.c_uint32),
    ('cp_packet_id_hi', ctypes.c_uint32),
    ('cp_packet_exe_status_lo', ctypes.c_uint32),
    ('cp_packet_exe_status_hi', ctypes.c_uint32),
    ('gds_save_base_addr_lo', ctypes.c_uint32),
    ('gds_save_base_addr_hi', ctypes.c_uint32),
    ('gds_save_mask_lo', ctypes.c_uint32),
    ('gds_save_mask_hi', ctypes.c_uint32),
    ('ctx_save_base_addr_lo', ctypes.c_uint32),
    ('ctx_save_base_addr_hi', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr_lo', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_active', ctypes.c_uint32),
    ('cp_hqd_vmid', ctypes.c_uint32),
    ('cp_hqd_persistent_state', ctypes.c_uint32),
    ('cp_hqd_pipe_priority', ctypes.c_uint32),
    ('cp_hqd_queue_priority', ctypes.c_uint32),
    ('cp_hqd_quantum', ctypes.c_uint32),
    ('cp_hqd_pq_base_lo', ctypes.c_uint32),
    ('cp_hqd_pq_base_hi', ctypes.c_uint32),
    ('cp_hqd_pq_rptr', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_doorbell_control', ctypes.c_uint32),
    ('reserved_144', ctypes.c_uint32),
    ('cp_hqd_pq_control', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ib_rptr', ctypes.c_uint32),
    ('cp_hqd_ib_control', ctypes.c_uint32),
    ('cp_hqd_iq_timer', ctypes.c_uint32),
    ('cp_hqd_iq_rptr', ctypes.c_uint32),
    ('cp_hqd_dequeue_request', ctypes.c_uint32),
    ('cp_hqd_dma_offload', ctypes.c_uint32),
    ('cp_hqd_sema_cmd', ctypes.c_uint32),
    ('cp_hqd_msg_type', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_hi', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_hi', ctypes.c_uint32),
    ('cp_hqd_hq_status0', ctypes.c_uint32),
    ('cp_hqd_hq_control0', ctypes.c_uint32),
    ('cp_mqd_control', ctypes.c_uint32),
    ('cp_hqd_hq_status1', ctypes.c_uint32),
    ('cp_hqd_hq_control1', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_eop_control', ctypes.c_uint32),
    ('cp_hqd_eop_rptr', ctypes.c_uint32),
    ('cp_hqd_eop_wptr', ctypes.c_uint32),
    ('cp_hqd_eop_done_events', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ctx_save_control', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_offset', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_size', ctypes.c_uint32),
    ('cp_hqd_wg_state_offset', ctypes.c_uint32),
    ('cp_hqd_ctx_save_size', ctypes.c_uint32),
    ('cp_hqd_gds_resource_state', ctypes.c_uint32),
    ('cp_hqd_error', ctypes.c_uint32),
    ('cp_hqd_eop_wptr_mem', ctypes.c_uint32),
    ('cp_hqd_aql_control', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_hi', ctypes.c_uint32),
    ('reserved_184', ctypes.c_uint32),
    ('reserved_185', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('reserved_190', ctypes.c_uint32),
    ('reserved_191', ctypes.c_uint32),
    ('iqtimer_pkt_header', ctypes.c_uint32),
    ('iqtimer_pkt_dw0', ctypes.c_uint32),
    ('iqtimer_pkt_dw1', ctypes.c_uint32),
    ('iqtimer_pkt_dw2', ctypes.c_uint32),
    ('iqtimer_pkt_dw3', ctypes.c_uint32),
    ('iqtimer_pkt_dw4', ctypes.c_uint32),
    ('iqtimer_pkt_dw5', ctypes.c_uint32),
    ('iqtimer_pkt_dw6', ctypes.c_uint32),
    ('iqtimer_pkt_dw7', ctypes.c_uint32),
    ('iqtimer_pkt_dw8', ctypes.c_uint32),
    ('iqtimer_pkt_dw9', ctypes.c_uint32),
    ('iqtimer_pkt_dw10', ctypes.c_uint32),
    ('iqtimer_pkt_dw11', ctypes.c_uint32),
    ('iqtimer_pkt_dw12', ctypes.c_uint32),
    ('iqtimer_pkt_dw13', ctypes.c_uint32),
    ('iqtimer_pkt_dw14', ctypes.c_uint32),
    ('iqtimer_pkt_dw15', ctypes.c_uint32),
    ('iqtimer_pkt_dw16', ctypes.c_uint32),
    ('iqtimer_pkt_dw17', ctypes.c_uint32),
    ('iqtimer_pkt_dw18', ctypes.c_uint32),
    ('iqtimer_pkt_dw19', ctypes.c_uint32),
    ('iqtimer_pkt_dw20', ctypes.c_uint32),
    ('iqtimer_pkt_dw21', ctypes.c_uint32),
    ('iqtimer_pkt_dw22', ctypes.c_uint32),
    ('iqtimer_pkt_dw23', ctypes.c_uint32),
    ('iqtimer_pkt_dw24', ctypes.c_uint32),
    ('iqtimer_pkt_dw25', ctypes.c_uint32),
    ('iqtimer_pkt_dw26', ctypes.c_uint32),
    ('iqtimer_pkt_dw27', ctypes.c_uint32),
    ('iqtimer_pkt_dw28', ctypes.c_uint32),
    ('iqtimer_pkt_dw29', ctypes.c_uint32),
    ('iqtimer_pkt_dw30', ctypes.c_uint32),
    ('iqtimer_pkt_dw31', ctypes.c_uint32),
    ('reserved_225', ctypes.c_uint32),
    ('reserved_226', ctypes.c_uint32),
    ('reserved_227', ctypes.c_uint32),
    ('set_resources_header', ctypes.c_uint32),
    ('set_resources_dw1', ctypes.c_uint32),
    ('set_resources_dw2', ctypes.c_uint32),
    ('set_resources_dw3', ctypes.c_uint32),
    ('set_resources_dw4', ctypes.c_uint32),
    ('set_resources_dw5', ctypes.c_uint32),
    ('set_resources_dw6', ctypes.c_uint32),
    ('set_resources_dw7', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('queue_doorbell_id0', ctypes.c_uint32),
    ('queue_doorbell_id1', ctypes.c_uint32),
    ('queue_doorbell_id2', ctypes.c_uint32),
    ('queue_doorbell_id3', ctypes.c_uint32),
    ('queue_doorbell_id4', ctypes.c_uint32),
    ('queue_doorbell_id5', ctypes.c_uint32),
    ('queue_doorbell_id6', ctypes.c_uint32),
    ('queue_doorbell_id7', ctypes.c_uint32),
    ('queue_doorbell_id8', ctypes.c_uint32),
    ('queue_doorbell_id9', ctypes.c_uint32),
    ('queue_doorbell_id10', ctypes.c_uint32),
    ('queue_doorbell_id11', ctypes.c_uint32),
    ('queue_doorbell_id12', ctypes.c_uint32),
    ('queue_doorbell_id13', ctypes.c_uint32),
    ('queue_doorbell_id14', ctypes.c_uint32),
    ('queue_doorbell_id15', ctypes.c_uint32),
    ('control_buf_addr_lo', ctypes.c_uint32),
    ('control_buf_addr_hi', ctypes.c_uint32),
    ('control_buf_wptr_lo', ctypes.c_uint32),
    ('control_buf_wptr_hi', ctypes.c_uint32),
    ('control_buf_dptr_lo', ctypes.c_uint32),
    ('control_buf_dptr_hi', ctypes.c_uint32),
    ('control_buf_num_entries', ctypes.c_uint32),
    ('draw_ring_addr_lo', ctypes.c_uint32),
    ('draw_ring_addr_hi', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('reserved_268', ctypes.c_uint32),
    ('reserved_269', ctypes.c_uint32),
    ('reserved_270', ctypes.c_uint32),
    ('reserved_271', ctypes.c_uint32),
    ('reserved_272', ctypes.c_uint32),
    ('reserved_273', ctypes.c_uint32),
    ('reserved_274', ctypes.c_uint32),
    ('reserved_275', ctypes.c_uint32),
    ('reserved_276', ctypes.c_uint32),
    ('reserved_277', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('reserved_288', ctypes.c_uint32),
    ('reserved_289', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('reserved_356', ctypes.c_uint32),
    ('reserved_357', ctypes.c_uint32),
    ('reserved_358', ctypes.c_uint32),
    ('reserved_359', ctypes.c_uint32),
    ('reserved_360', ctypes.c_uint32),
    ('reserved_361', ctypes.c_uint32),
    ('reserved_362', ctypes.c_uint32),
    ('reserved_363', ctypes.c_uint32),
    ('reserved_364', ctypes.c_uint32),
    ('reserved_365', ctypes.c_uint32),
    ('reserved_366', ctypes.c_uint32),
    ('reserved_367', ctypes.c_uint32),
    ('reserved_368', ctypes.c_uint32),
    ('reserved_369', ctypes.c_uint32),
    ('reserved_370', ctypes.c_uint32),
    ('reserved_371', ctypes.c_uint32),
    ('reserved_372', ctypes.c_uint32),
    ('reserved_373', ctypes.c_uint32),
    ('reserved_374', ctypes.c_uint32),
    ('reserved_375', ctypes.c_uint32),
    ('reserved_376', ctypes.c_uint32),
    ('reserved_377', ctypes.c_uint32),
    ('reserved_378', ctypes.c_uint32),
    ('reserved_379', ctypes.c_uint32),
    ('reserved_380', ctypes.c_uint32),
    ('reserved_381', ctypes.c_uint32),
    ('reserved_382', ctypes.c_uint32),
    ('reserved_383', ctypes.c_uint32),
    ('reserved_384', ctypes.c_uint32),
    ('reserved_385', ctypes.c_uint32),
    ('reserved_386', ctypes.c_uint32),
    ('reserved_387', ctypes.c_uint32),
    ('reserved_388', ctypes.c_uint32),
    ('reserved_389', ctypes.c_uint32),
    ('reserved_390', ctypes.c_uint32),
    ('reserved_391', ctypes.c_uint32),
    ('reserved_392', ctypes.c_uint32),
    ('reserved_393', ctypes.c_uint32),
    ('reserved_394', ctypes.c_uint32),
    ('reserved_395', ctypes.c_uint32),
    ('reserved_396', ctypes.c_uint32),
    ('reserved_397', ctypes.c_uint32),
    ('reserved_398', ctypes.c_uint32),
    ('reserved_399', ctypes.c_uint32),
    ('reserved_400', ctypes.c_uint32),
    ('reserved_401', ctypes.c_uint32),
    ('reserved_402', ctypes.c_uint32),
    ('reserved_403', ctypes.c_uint32),
    ('reserved_404', ctypes.c_uint32),
    ('reserved_405', ctypes.c_uint32),
    ('reserved_406', ctypes.c_uint32),
    ('reserved_407', ctypes.c_uint32),
    ('reserved_408', ctypes.c_uint32),
    ('reserved_409', ctypes.c_uint32),
    ('reserved_410', ctypes.c_uint32),
    ('reserved_411', ctypes.c_uint32),
    ('reserved_412', ctypes.c_uint32),
    ('reserved_413', ctypes.c_uint32),
    ('reserved_414', ctypes.c_uint32),
    ('reserved_415', ctypes.c_uint32),
    ('reserved_416', ctypes.c_uint32),
    ('reserved_417', ctypes.c_uint32),
    ('reserved_418', ctypes.c_uint32),
    ('reserved_419', ctypes.c_uint32),
    ('reserved_420', ctypes.c_uint32),
    ('reserved_421', ctypes.c_uint32),
    ('reserved_422', ctypes.c_uint32),
    ('reserved_423', ctypes.c_uint32),
    ('reserved_424', ctypes.c_uint32),
    ('reserved_425', ctypes.c_uint32),
    ('reserved_426', ctypes.c_uint32),
    ('reserved_427', ctypes.c_uint32),
    ('reserved_428', ctypes.c_uint32),
    ('reserved_429', ctypes.c_uint32),
    ('reserved_430', ctypes.c_uint32),
    ('reserved_431', ctypes.c_uint32),
    ('reserved_432', ctypes.c_uint32),
    ('reserved_433', ctypes.c_uint32),
    ('reserved_434', ctypes.c_uint32),
    ('reserved_435', ctypes.c_uint32),
    ('reserved_436', ctypes.c_uint32),
    ('reserved_437', ctypes.c_uint32),
    ('reserved_438', ctypes.c_uint32),
    ('reserved_439', ctypes.c_uint32),
    ('reserved_440', ctypes.c_uint32),
    ('reserved_441', ctypes.c_uint32),
    ('reserved_442', ctypes.c_uint32),
    ('reserved_443', ctypes.c_uint32),
    ('reserved_444', ctypes.c_uint32),
    ('reserved_445', ctypes.c_uint32),
    ('reserved_446', ctypes.c_uint32),
    ('reserved_447', ctypes.c_uint32),
    ('gws_0_val', ctypes.c_uint32),
    ('gws_1_val', ctypes.c_uint32),
    ('gws_2_val', ctypes.c_uint32),
    ('gws_3_val', ctypes.c_uint32),
    ('gws_4_val', ctypes.c_uint32),
    ('gws_5_val', ctypes.c_uint32),
    ('gws_6_val', ctypes.c_uint32),
    ('gws_7_val', ctypes.c_uint32),
    ('gws_8_val', ctypes.c_uint32),
    ('gws_9_val', ctypes.c_uint32),
    ('gws_10_val', ctypes.c_uint32),
    ('gws_11_val', ctypes.c_uint32),
    ('gws_12_val', ctypes.c_uint32),
    ('gws_13_val', ctypes.c_uint32),
    ('gws_14_val', ctypes.c_uint32),
    ('gws_15_val', ctypes.c_uint32),
    ('gws_16_val', ctypes.c_uint32),
    ('gws_17_val', ctypes.c_uint32),
    ('gws_18_val', ctypes.c_uint32),
    ('gws_19_val', ctypes.c_uint32),
    ('gws_20_val', ctypes.c_uint32),
    ('gws_21_val', ctypes.c_uint32),
    ('gws_22_val', ctypes.c_uint32),
    ('gws_23_val', ctypes.c_uint32),
    ('gws_24_val', ctypes.c_uint32),
    ('gws_25_val', ctypes.c_uint32),
    ('gws_26_val', ctypes.c_uint32),
    ('gws_27_val', ctypes.c_uint32),
    ('gws_28_val', ctypes.c_uint32),
    ('gws_29_val', ctypes.c_uint32),
    ('gws_30_val', ctypes.c_uint32),
    ('gws_31_val', ctypes.c_uint32),
    ('gws_32_val', ctypes.c_uint32),
    ('gws_33_val', ctypes.c_uint32),
    ('gws_34_val', ctypes.c_uint32),
    ('gws_35_val', ctypes.c_uint32),
    ('gws_36_val', ctypes.c_uint32),
    ('gws_37_val', ctypes.c_uint32),
    ('gws_38_val', ctypes.c_uint32),
    ('gws_39_val', ctypes.c_uint32),
    ('gws_40_val', ctypes.c_uint32),
    ('gws_41_val', ctypes.c_uint32),
    ('gws_42_val', ctypes.c_uint32),
    ('gws_43_val', ctypes.c_uint32),
    ('gws_44_val', ctypes.c_uint32),
    ('gws_45_val', ctypes.c_uint32),
    ('gws_46_val', ctypes.c_uint32),
    ('gws_47_val', ctypes.c_uint32),
    ('gws_48_val', ctypes.c_uint32),
    ('gws_49_val', ctypes.c_uint32),
    ('gws_50_val', ctypes.c_uint32),
    ('gws_51_val', ctypes.c_uint32),
    ('gws_52_val', ctypes.c_uint32),
    ('gws_53_val', ctypes.c_uint32),
    ('gws_54_val', ctypes.c_uint32),
    ('gws_55_val', ctypes.c_uint32),
    ('gws_56_val', ctypes.c_uint32),
    ('gws_57_val', ctypes.c_uint32),
    ('gws_58_val', ctypes.c_uint32),
    ('gws_59_val', ctypes.c_uint32),
    ('gws_60_val', ctypes.c_uint32),
    ('gws_61_val', ctypes.c_uint32),
    ('gws_62_val', ctypes.c_uint32),
    ('gws_63_val', ctypes.c_uint32),
]

__AMDGPU_VM_H__ = True # macro
AMDGPU_VM_MAX_UPDATE_SIZE = 0x3FFFF # macro
# def AMDGPU_VM_PTE_COUNT(adev):  # macro
#    return (1<<(adev)->vm_manager.block_size)  
AMDGPU_PTE_VALID = (1<<0) # macro
AMDGPU_PTE_SYSTEM = (1<<1) # macro
AMDGPU_PTE_SNOOPED = (1<<2) # macro
AMDGPU_PTE_TMZ = (1<<3) # macro
AMDGPU_PTE_EXECUTABLE = (1<<4) # macro
AMDGPU_PTE_READABLE = (1<<5) # macro
AMDGPU_PTE_WRITEABLE = (1<<6) # macro
def AMDGPU_PTE_FRAG(x):  # macro
   return ((x&0x1f)<<7)  
AMDGPU_PTE_PRT = (1<<51) # macro
AMDGPU_PDE_PTE = (1<<54) # macro
AMDGPU_PTE_LOG = (1<<55) # macro
AMDGPU_PTE_TF = (1<<56) # macro
AMDGPU_PTE_NOALLOC = (1<<58) # macro
def AMDGPU_PDE_BFS(a):  # macro
   return ((int)(a)<<59)  
AMDGPU_VM_NORETRY_FLAGS = ((1<<4)|(1<<54)|(1<<56)) # macro
AMDGPU_VM_NORETRY_FLAGS_TF = ((1<<0)|(1<<1)|(1<<51)) # macro
def AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype):  # macro
   return ((int)(mtype)<<57)  
AMDGPU_PTE_MTYPE_VG10_MASK = AMDGPU_PTE_MTYPE_VG10_SHIFT ( 3 ) # macro
def AMDGPU_PTE_MTYPE_VG10(flags, mtype):  # macro
   return (((int)(flags)&(~AMDGPU_PTE_MTYPE_VG10_SHIFT(3)))|AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype))  
AMDGPU_MTYPE_NC = 0 # macro
AMDGPU_MTYPE_CC = 2 # macro
# AMDGPU_PTE_DEFAULT_ATC = ((1<<1)|(1<<2)|(1<<4)|(1<<5)|(1<<6)|AMDGPU_PTE_MTYPE_VG10(2)) # macro
def AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype):  # macro
   return ((int)(mtype)<<48)  
AMDGPU_PTE_MTYPE_NV10_MASK = AMDGPU_PTE_MTYPE_NV10_SHIFT ( 7 ) # macro
def AMDGPU_PTE_MTYPE_NV10(flags, mtype):  # macro
   return (((int)(flags)&(~AMDGPU_PTE_MTYPE_NV10_SHIFT(7)))|AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype))  
AMDGPU_PTE_PRT_GFX12 = (1<<56) # macro
# def AMDGPU_PTE_PRT_FLAG(adev):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?(1<<56):(1<<51))  
def AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype):  # macro
   return ((int)(mtype)<<54)  
AMDGPU_PTE_MTYPE_GFX12_MASK = AMDGPU_PTE_MTYPE_GFX12_SHIFT ( 3 ) # macro
def AMDGPU_PTE_MTYPE_GFX12(flags, mtype):  # macro
   return (((int)(flags)&(~AMDGPU_PTE_MTYPE_GFX12_SHIFT(3)))|AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype))  
AMDGPU_PTE_IS_PTE = (1<<63) # macro
def AMDGPU_PDE_BFS_GFX12(a):  # macro
   return ((int)((a)&0x1f)<<58)  
# def AMDGPU_PDE_BFS_FLAG(adev, a):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?AMDGPU_PDE_BFS_GFX12(a):AMDGPU_PDE_BFS(a))  
AMDGPU_PDE_PTE_GFX12 = (1<<63) # macro
# def AMDGPU_PDE_PTE_FLAG(adev):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?(1<<63):(1<<54))  
AMDGPU_VM_FAULT_STOP_NEVER = 0 # macro
AMDGPU_VM_FAULT_STOP_FIRST = 1 # macro
AMDGPU_VM_FAULT_STOP_ALWAYS = 2 # macro
AMDGPU_VM_RESERVED_VRAM = (8<<20) # macro
AMDGPU_MAX_VMHUBS = 13 # macro
AMDGPU_GFXHUB_START = 0 # macro
AMDGPU_MMHUB0_START = 8 # macro
AMDGPU_MMHUB1_START = 12 # macro
def AMDGPU_GFXHUB(x):  # macro
   return (0+(x))  
def AMDGPU_MMHUB0(x):  # macro
   return (8+(x))  
def AMDGPU_MMHUB1(x):  # macro
   return (12+(x))  
def AMDGPU_IS_GFXHUB(x):  # macro
   return ((x)>=0 and (x)<8)  
def AMDGPU_IS_MMHUB0(x):  # macro
   return ((x)>=8 and (x)<12)  
def AMDGPU_IS_MMHUB1(x):  # macro
   return ((x)>=12 and (x)<13)  
AMDGPU_VA_RESERVED_CSA_SIZE = (2<<20) # macro
# def AMDGPU_VA_RESERVED_CSA_START(adev):  # macro
#    return (((adev)->vm_manager.max_pfn<<AMDGPU_GPU_PAGE_SHIFT)-(2<<20))  
AMDGPU_VA_RESERVED_SEQ64_SIZE = (2<<20) # macro
def AMDGPU_VA_RESERVED_SEQ64_START(adev):  # macro
   return (AMDGPU_VA_RESERVED_CSA_START(adev)-(2<<20))  
AMDGPU_VA_RESERVED_TRAP_SIZE = (2<<12) # macro
def AMDGPU_VA_RESERVED_TRAP_START(adev):  # macro
   return (AMDGPU_VA_RESERVED_SEQ64_START(adev)-(2<<12))  
AMDGPU_VA_RESERVED_BOTTOM = (1<<16) # macro
AMDGPU_VA_RESERVED_TOP = ((2<<12)+(2<<20)+(2<<20)) # macro
AMDGPU_VM_USE_CPU_FOR_GFX = (1<<0) # macro
AMDGPU_VM_USE_CPU_FOR_COMPUTE = (1<<1) # macro

# values for enumeration 'amdgpu_vm_level'
amdgpu_vm_level__enumvalues = {
    0: 'AMDGPU_VM_PDB2',
    1: 'AMDGPU_VM_PDB1',
    2: 'AMDGPU_VM_PDB0',
    3: 'AMDGPU_VM_PTB',
}
AMDGPU_VM_PDB2 = 0
AMDGPU_VM_PDB1 = 1
AMDGPU_VM_PDB0 = 2
AMDGPU_VM_PTB = 3
amdgpu_vm_level = ctypes.c_uint32 # enum
__all__ = \
    ['AMDGPU_CPCE_UCODE_LOADED', 'AMDGPU_CPMEC1_UCODE_LOADED',
    'AMDGPU_CPMEC2_UCODE_LOADED', 'AMDGPU_CPME_UCODE_LOADED',
    'AMDGPU_CPPFP_UCODE_LOADED', 'AMDGPU_CPRLC_UCODE_LOADED',
    'AMDGPU_FENCE_FLAG_64BIT', 'AMDGPU_FENCE_FLAG_EXEC',
    'AMDGPU_FENCE_FLAG_INT', 'AMDGPU_FENCE_FLAG_TC_WB_ONLY',
    'AMDGPU_FW_LOAD_DIRECT', 'AMDGPU_FW_LOAD_PSP',
    'AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO', 'AMDGPU_FW_LOAD_SMU',
    'AMDGPU_GFXHUB_START', 'AMDGPU_IB_POOL_DELAYED',
    'AMDGPU_IB_POOL_DIRECT', 'AMDGPU_IB_POOL_IMMEDIATE',
    'AMDGPU_IB_POOL_MAX', 'AMDGPU_IB_POOL_SIZE',
    'AMDGPU_MAX_COMPUTE_RINGS', 'AMDGPU_MAX_GFX_RINGS',
    'AMDGPU_MAX_HWIP_RINGS', 'AMDGPU_MAX_RINGS',
    'AMDGPU_MAX_SW_GFX_RINGS', 'AMDGPU_MAX_UVD_ENC_RINGS',
    'AMDGPU_MAX_VCE_RINGS', 'AMDGPU_MAX_VMHUBS',
    'AMDGPU_MAX_VPE_RINGS', 'AMDGPU_MMHUB0_START',
    'AMDGPU_MMHUB1_START', 'AMDGPU_MTYPE_CC', 'AMDGPU_MTYPE_NC',
    'AMDGPU_PDE_PTE', 'AMDGPU_PDE_PTE_GFX12',
    'AMDGPU_PTE_DEFAULT_ATC', 'AMDGPU_PTE_EXECUTABLE',
    'AMDGPU_PTE_IS_PTE', 'AMDGPU_PTE_LOG',
    'AMDGPU_PTE_MTYPE_GFX12_MASK', 'AMDGPU_PTE_MTYPE_NV10_MASK',
    'AMDGPU_PTE_MTYPE_VG10_MASK', 'AMDGPU_PTE_NOALLOC',
    'AMDGPU_PTE_PRT', 'AMDGPU_PTE_PRT_GFX12', 'AMDGPU_PTE_READABLE',
    'AMDGPU_PTE_SNOOPED', 'AMDGPU_PTE_SYSTEM', 'AMDGPU_PTE_TF',
    'AMDGPU_PTE_TMZ', 'AMDGPU_PTE_VALID', 'AMDGPU_PTE_WRITEABLE',
    'AMDGPU_RING_PRIO_0', 'AMDGPU_RING_PRIO_1', 'AMDGPU_RING_PRIO_2',
    'AMDGPU_RING_PRIO_DEFAULT', 'AMDGPU_RING_PRIO_MAX',
    'AMDGPU_SDMA0_UCODE_LOADED', 'AMDGPU_SDMA1_UCODE_LOADED',
    'AMDGPU_UCODE_ID', 'AMDGPU_UCODE_ID_CAP', 'AMDGPU_UCODE_ID_CP_CE',
    'AMDGPU_UCODE_ID_CP_ME', 'AMDGPU_UCODE_ID_CP_MEC1',
    'AMDGPU_UCODE_ID_CP_MEC1_JT', 'AMDGPU_UCODE_ID_CP_MEC2',
    'AMDGPU_UCODE_ID_CP_MEC2_JT', 'AMDGPU_UCODE_ID_CP_MES',
    'AMDGPU_UCODE_ID_CP_MES1', 'AMDGPU_UCODE_ID_CP_MES1_DATA',
    'AMDGPU_UCODE_ID_CP_MES_DATA', 'AMDGPU_UCODE_ID_CP_PFP',
    'AMDGPU_UCODE_ID_CP_RS64_ME', 'AMDGPU_UCODE_ID_CP_RS64_MEC',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_PFP',
    'AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK', 'AMDGPU_UCODE_ID_DMCUB',
    'AMDGPU_UCODE_ID_DMCU_ERAM', 'AMDGPU_UCODE_ID_DMCU_INTV',
    'AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS', 'AMDGPU_UCODE_ID_IMU_D',
    'AMDGPU_UCODE_ID_IMU_I', 'AMDGPU_UCODE_ID_ISP',
    'AMDGPU_UCODE_ID_JPEG_RAM', 'AMDGPU_UCODE_ID_MAXIMUM',
    'AMDGPU_UCODE_ID_P2S_TABLE', 'AMDGPU_UCODE_ID_PPTABLE',
    'AMDGPU_UCODE_ID_RLC_DRAM', 'AMDGPU_UCODE_ID_RLC_G',
    'AMDGPU_UCODE_ID_RLC_IRAM', 'AMDGPU_UCODE_ID_RLC_P',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM',
    'AMDGPU_UCODE_ID_RLC_V', 'AMDGPU_UCODE_ID_SDMA0',
    'AMDGPU_UCODE_ID_SDMA1', 'AMDGPU_UCODE_ID_SDMA2',
    'AMDGPU_UCODE_ID_SDMA3', 'AMDGPU_UCODE_ID_SDMA4',
    'AMDGPU_UCODE_ID_SDMA5', 'AMDGPU_UCODE_ID_SDMA6',
    'AMDGPU_UCODE_ID_SDMA7', 'AMDGPU_UCODE_ID_SDMA_RS64',
    'AMDGPU_UCODE_ID_SDMA_UCODE_TH0',
    'AMDGPU_UCODE_ID_SDMA_UCODE_TH1',
    'AMDGPU_UCODE_ID_SE0_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE1_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE2_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE3_TAP_DELAYS', 'AMDGPU_UCODE_ID_SMC',
    'AMDGPU_UCODE_ID_STORAGE', 'AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER',
    'AMDGPU_UCODE_ID_UMSCH_MM_DATA', 'AMDGPU_UCODE_ID_UMSCH_MM_UCODE',
    'AMDGPU_UCODE_ID_UVD', 'AMDGPU_UCODE_ID_UVD1',
    'AMDGPU_UCODE_ID_VCE', 'AMDGPU_UCODE_ID_VCN',
    'AMDGPU_UCODE_ID_VCN0_RAM', 'AMDGPU_UCODE_ID_VCN1',
    'AMDGPU_UCODE_ID_VCN1_RAM', 'AMDGPU_UCODE_ID_VPE',
    'AMDGPU_UCODE_ID_VPE_CTL', 'AMDGPU_UCODE_ID_VPE_CTX',
    'AMDGPU_UCODE_STATUS', 'AMDGPU_UCODE_STATUS_INVALID',
    'AMDGPU_UCODE_STATUS_LOADED', 'AMDGPU_UCODE_STATUS_NOT_LOADED',
    'AMDGPU_VA_RESERVED_BOTTOM', 'AMDGPU_VA_RESERVED_CSA_SIZE',
    'AMDGPU_VA_RESERVED_SEQ64_SIZE', 'AMDGPU_VA_RESERVED_TOP',
    'AMDGPU_VA_RESERVED_TRAP_SIZE', 'AMDGPU_VM_FAULT_STOP_ALWAYS',
    'AMDGPU_VM_FAULT_STOP_FIRST', 'AMDGPU_VM_FAULT_STOP_NEVER',
    'AMDGPU_VM_MAX_UPDATE_SIZE', 'AMDGPU_VM_NORETRY_FLAGS',
    'AMDGPU_VM_NORETRY_FLAGS_TF', 'AMDGPU_VM_PDB0', 'AMDGPU_VM_PDB1',
    'AMDGPU_VM_PDB2', 'AMDGPU_VM_PTB', 'AMDGPU_VM_RESERVED_VRAM',
    'AMDGPU_VM_USE_CPU_FOR_COMPUTE', 'AMDGPU_VM_USE_CPU_FOR_GFX',
    'PSP_FW_TYPE_MAX_INDEX', 'PSP_FW_TYPE_PSP_DBG_DRV',
    'PSP_FW_TYPE_PSP_INTF_DRV', 'PSP_FW_TYPE_PSP_IPKEYMGR_DRV',
    'PSP_FW_TYPE_PSP_KDB', 'PSP_FW_TYPE_PSP_RAS_DRV',
    'PSP_FW_TYPE_PSP_RL', 'PSP_FW_TYPE_PSP_SOC_DRV',
    'PSP_FW_TYPE_PSP_SOS', 'PSP_FW_TYPE_PSP_SPL',
    'PSP_FW_TYPE_PSP_SYS_DRV', 'PSP_FW_TYPE_PSP_TOC',
    'PSP_FW_TYPE_UNKOWN', 'TA_FW_TYPE_MAX_INDEX',
    'TA_FW_TYPE_PSP_ASD', 'TA_FW_TYPE_PSP_DTM', 'TA_FW_TYPE_PSP_HDCP',
    'TA_FW_TYPE_PSP_RAP', 'TA_FW_TYPE_PSP_RAS',
    'TA_FW_TYPE_PSP_SECUREDISPLAY', 'TA_FW_TYPE_PSP_XGMI',
    'TA_FW_TYPE_UNKOWN', 'V11_STRUCTS_H_', '__AMDGPU_RING_H__',
    '__AMDGPU_UCODE_H__', '__AMDGPU_VM_H__',
    'amdgpu_firmware_load_type', 'amdgpu_ib_pool_type',
    'amdgpu_ring_priority_level', 'amdgpu_vm_level', 'bool',
    'psp_fw_type', 'struct_amdgpu_bo', 'struct_amdgpu_firmware_info',
    'struct_amdgpu_ring', 'struct_common_firmware_header',
    'struct_dmcu_firmware_header_v1_0',
    'struct_dmcub_firmware_header_v1_0', 'struct_firmware',
    'struct_gfx_firmware_header_v1_0',
    'struct_gfx_firmware_header_v2_0',
    'struct_gpu_info_firmware_header_v1_0',
    'struct_gpu_info_firmware_v1_0', 'struct_gpu_info_firmware_v1_1',
    'struct_gpu_info_firmware_v1_2',
    'struct_gpu_info_soc_bounding_box_v1_0',
    'struct_gpu_info_voltage_scaling_v1_0',
    'struct_imu_firmware_header_v1_0',
    'struct_mc_firmware_header_v1_0',
    'struct_mes_firmware_header_v1_0',
    'struct_psp_firmware_header_v1_0',
    'struct_psp_firmware_header_v1_1',
    'struct_psp_firmware_header_v1_2',
    'struct_psp_firmware_header_v1_3',
    'struct_psp_firmware_header_v2_0',
    'struct_psp_firmware_header_v2_1', 'struct_psp_fw_bin_desc',
    'struct_psp_fw_legacy_bin_desc',
    'struct_rlc_firmware_header_v1_0',
    'struct_rlc_firmware_header_v2_0',
    'struct_rlc_firmware_header_v2_1',
    'struct_rlc_firmware_header_v2_2',
    'struct_rlc_firmware_header_v2_3',
    'struct_rlc_firmware_header_v2_4',
    'struct_sdma_firmware_header_v1_0',
    'struct_sdma_firmware_header_v1_1',
    'struct_sdma_firmware_header_v2_0',
    'struct_sdma_firmware_header_v3_0',
    'struct_smc_firmware_header_v1_0',
    'struct_smc_firmware_header_v2_0',
    'struct_smc_firmware_header_v2_1',
    'struct_smc_soft_pptable_entry', 'struct_ta_firmware_header_v1_0',
    'struct_ta_firmware_header_v2_0',
    'struct_umsch_mm_firmware_header_v1_0', 'struct_v11_compute_mqd',
    'struct_v11_gfx_mqd', 'struct_v11_sdma_mqd',
    'struct_vpe_firmware_header_v1_0', 'ta_fw_type', 'u16', 'u32',
    'u64', 'u8', 'uint16_t', 'uint32_t', 'uint', 'uint8_t',
    'union_amdgpu_firmware_header']
