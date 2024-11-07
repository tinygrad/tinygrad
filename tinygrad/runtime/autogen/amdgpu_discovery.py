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





_DISCOVERY_H_ = True # macro
uint32_t = True # macro
uint8_t = True # macro
uint16_t = True # macro
uint64_t = True # macro
u32 = True # macro
u8 = True # macro
u16 = True # macro
u64 = True # macro
bool = True # macro
PSP_HEADER_SIZE = 256 # macro
BINARY_SIGNATURE = 0x28211407 # macro
DISCOVERY_TABLE_SIGNATURE = 0x53445049 # macro
GC_TABLE_ID = 0x4347 # macro
HARVEST_TABLE_SIGNATURE = 0x56524148 # macro
VCN_INFO_TABLE_ID = 0x004E4356 # macro
MALL_INFO_TABLE_ID = 0x4C4C414D # macro
NPS_INFO_TABLE_ID = 0x0053504E # macro
VCN_INFO_TABLE_MAX_NUM_INSTANCES = 4 # macro
NPS_INFO_TABLE_MAX_NUM_INSTANCES = 12 # macro
HWIP_MAX_INSTANCE = 44 # macro
HW_ID_MAX = 300 # macro
MP1_HWID = 1 # macro
MP2_HWID = 2 # macro
THM_HWID = 3 # macro
SMUIO_HWID = 4 # macro
FUSE_HWID = 5 # macro
CLKA_HWID = 6 # macro
PWR_HWID = 10 # macro
GC_HWID = 11 # macro
UVD_HWID = 12 # macro
VCN_HWID = 12 # macro
AUDIO_AZ_HWID = 13 # macro
ACP_HWID = 14 # macro
DCI_HWID = 15 # macro
DMU_HWID = 271 # macro
DCO_HWID = 16 # macro
DIO_HWID = 272 # macro
XDMA_HWID = 17 # macro
DCEAZ_HWID = 18 # macro
DAZ_HWID = 274 # macro
SDPMUX_HWID = 19 # macro
NTB_HWID = 20 # macro
VPE_HWID = 21 # macro
IOHC_HWID = 24 # macro
L2IMU_HWID = 28 # macro
VCE_HWID = 32 # macro
MMHUB_HWID = 34 # macro
ATHUB_HWID = 35 # macro
DBGU_NBIO_HWID = 36 # macro
DFX_HWID = 37 # macro
DBGU0_HWID = 38 # macro
DBGU1_HWID = 39 # macro
OSSSYS_HWID = 40 # macro
HDP_HWID = 41 # macro
SDMA0_HWID = 42 # macro
SDMA1_HWID = 43 # macro
ISP_HWID = 44 # macro
DBGU_IO_HWID = 45 # macro
DF_HWID = 46 # macro
CLKB_HWID = 47 # macro
FCH_HWID = 48 # macro
DFX_DAP_HWID = 49 # macro
L1IMU_PCIE_HWID = 50 # macro
L1IMU_NBIF_HWID = 51 # macro
L1IMU_IOAGR_HWID = 52 # macro
L1IMU3_HWID = 53 # macro
L1IMU4_HWID = 54 # macro
L1IMU5_HWID = 55 # macro
L1IMU6_HWID = 56 # macro
L1IMU7_HWID = 57 # macro
L1IMU8_HWID = 58 # macro
L1IMU9_HWID = 59 # macro
L1IMU10_HWID = 60 # macro
L1IMU11_HWID = 61 # macro
L1IMU12_HWID = 62 # macro
L1IMU13_HWID = 63 # macro
L1IMU14_HWID = 64 # macro
L1IMU15_HWID = 65 # macro
WAFLC_HWID = 66 # macro
FCH_USB_PD_HWID = 67 # macro
SDMA2_HWID = 68 # macro
SDMA3_HWID = 69 # macro
PCIE_HWID = 70 # macro
PCS_HWID = 80 # macro
DDCL_HWID = 89 # macro
SST_HWID = 90 # macro
LSDMA_HWID = 91 # macro
IOAGR_HWID = 100 # macro
NBIF_HWID = 108 # macro
IOAPIC_HWID = 124 # macro
SYSTEMHUB_HWID = 128 # macro
NTBCCP_HWID = 144 # macro
UMC_HWID = 150 # macro
SATA_HWID = 168 # macro
USB_HWID = 170 # macro
CCXSEC_HWID = 176 # macro
XGMI_HWID = 200 # macro
XGBE_HWID = 216 # macro
MP0_HWID = 255 # macro

# values for enumeration 'c__EA_table'
c__EA_table__enumvalues = {
    0: 'IP_DISCOVERY',
    1: 'GC',
    2: 'HARVEST_INFO',
    3: 'VCN_INFO',
    4: 'MALL_INFO',
    5: 'NPS_INFO',
    6: 'TOTAL_TABLES',
}
IP_DISCOVERY = 0
GC = 1
HARVEST_INFO = 2
VCN_INFO = 3
MALL_INFO = 4
NPS_INFO = 5
TOTAL_TABLES = 6
c__EA_table = ctypes.c_uint32 # enum
table = c__EA_table
table__enumvalues = c__EA_table__enumvalues
class struct_table_info(Structure):
    pass

struct_table_info._pack_ = 1 # source:False
struct_table_info._fields_ = [
    ('offset', ctypes.c_uint16),
    ('checksum', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('padding', ctypes.c_uint16),
]

table_info = struct_table_info
class struct_binary_header(Structure):
    pass

struct_binary_header._pack_ = 1 # source:False
struct_binary_header._fields_ = [
    ('binary_signature', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('binary_checksum', ctypes.c_uint16),
    ('binary_size', ctypes.c_uint16),
    ('table_list', struct_table_info * 6),
]

binary_header = struct_binary_header
class struct_die_info(Structure):
    pass

struct_die_info._pack_ = 1 # source:False
struct_die_info._fields_ = [
    ('die_id', ctypes.c_uint16),
    ('die_offset', ctypes.c_uint16),
]

die_info = struct_die_info
class struct_ip_discovery_header(Structure):
    pass

class union_ip_discovery_header_0(Union):
    pass

class struct_ip_discovery_header_0_0(Structure):
    pass

struct_ip_discovery_header_0_0._pack_ = 1 # source:False
struct_ip_discovery_header_0_0._fields_ = [
    ('base_addr_64_bit', ctypes.c_ubyte, 1),
    ('reserved', ctypes.c_ubyte, 7),
    ('reserved2', ctypes.c_ubyte, 8),
]

union_ip_discovery_header_0._pack_ = 1 # source:False
union_ip_discovery_header_0._anonymous_ = ('_0',)
union_ip_discovery_header_0._fields_ = [
    ('padding', ctypes.c_uint16 * 1),
    ('_0', struct_ip_discovery_header_0_0),
]

struct_ip_discovery_header._pack_ = 1 # source:False
struct_ip_discovery_header._anonymous_ = ('_0',)
struct_ip_discovery_header._fields_ = [
    ('signature', ctypes.c_uint32),
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('id', ctypes.c_uint32),
    ('num_dies', ctypes.c_uint16),
    ('die_info', struct_die_info * 16),
    ('_0', union_ip_discovery_header_0),
]

ip_discovery_header = struct_ip_discovery_header
class struct_ip(Structure):
    pass

struct_ip._pack_ = 1 # source:False
struct_ip._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('number_instance', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
    ('harvest', ctypes.c_ubyte, 4),
    ('reserved', ctypes.c_ubyte, 4),
    ('base_address', ctypes.c_uint32 * 1),
]

ip = struct_ip
class struct_ip_v3(Structure):
    pass

struct_ip_v3._pack_ = 1 # source:False
struct_ip_v3._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('instance_number', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
    ('sub_revision', ctypes.c_ubyte, 4),
    ('variant', ctypes.c_ubyte, 4),
    ('base_address', ctypes.c_uint32 * 1),
]

ip_v3 = struct_ip_v3
class struct_ip_v4(Structure):
    pass

struct_ip_v4._pack_ = 1 # source:False
struct_ip_v4._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('instance_number', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
    ('base_address_64', ctypes.c_uint64 * 1),
]

ip_v4 = struct_ip_v4
class struct_die_header(Structure):
    pass

struct_die_header._pack_ = 1 # source:False
struct_die_header._fields_ = [
    ('die_id', ctypes.c_uint16),
    ('num_ips', ctypes.c_uint16),
]

die_header = struct_die_header
class struct_ip_structure(Structure):
    pass

class struct_die(Structure):
    pass

class union_die_0(Union):
    pass

union_die_0._pack_ = 1 # source:False
union_die_0._fields_ = [
    ('ip_list', ctypes.POINTER(struct_ip)),
    ('ip_v3_list', ctypes.POINTER(struct_ip_v3)),
    ('ip_v4_list', ctypes.POINTER(struct_ip_v4)),
]

struct_die._pack_ = 1 # source:False
struct_die._anonymous_ = ('_0',)
struct_die._fields_ = [
    ('die_header', ctypes.POINTER(struct_die_header)),
    ('_0', union_die_0),
]

struct_ip_structure._pack_ = 1 # source:False
struct_ip_structure._fields_ = [
    ('header', ctypes.POINTER(struct_ip_discovery_header)),
    ('die', struct_die),
]

ip_structure = struct_ip_structure
class struct_gpu_info_header(Structure):
    pass

struct_gpu_info_header._pack_ = 1 # source:False
struct_gpu_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size', ctypes.c_uint32),
]

class struct_gc_info_v1_0(Structure):
    pass

struct_gc_info_v1_0._pack_ = 1 # source:False
struct_gc_info_v1_0._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
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
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
]

class struct_gc_info_v1_1(Structure):
    pass

struct_gc_info_v1_1._pack_ = 1 # source:False
struct_gc_info_v1_1._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
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
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
    ('gc_num_tcp_per_sa', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_tcps', ctypes.c_uint32),
]

class struct_gc_info_v1_2(Structure):
    pass

struct_gc_info_v1_2._pack_ = 1 # source:False
struct_gc_info_v1_2._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
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
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
    ('gc_num_tcp_per_sa', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_tcps', ctypes.c_uint32),
    ('gc_num_tcp_per_wpg', ctypes.c_uint32),
    ('gc_tcp_l1_size', ctypes.c_uint32),
    ('gc_num_sqc_per_wgp', ctypes.c_uint32),
    ('gc_l1_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_l1_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_gl1c_per_sa', ctypes.c_uint32),
    ('gc_gl1c_size_per_instance', ctypes.c_uint32),
    ('gc_gl2c_per_gpu', ctypes.c_uint32),
]

class struct_gc_info_v2_0(Structure):
    pass

struct_gc_info_v2_0._pack_ = 1 # source:False
struct_gc_info_v2_0._fields_ = [
    ('header', struct_gpu_info_header),
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
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
]

class struct_gc_info_v2_1(Structure):
    pass

struct_gc_info_v2_1._pack_ = 1 # source:False
struct_gc_info_v2_1._fields_ = [
    ('header', struct_gpu_info_header),
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
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_tcp_per_sh', ctypes.c_uint32),
    ('gc_tcp_size_per_cu', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_cu_per_sqc', ctypes.c_uint32),
    ('gc_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_scalar_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_tcc_size', ctypes.c_uint32),
]

class struct_harvest_info_header(Structure):
    pass

struct_harvest_info_header._pack_ = 1 # source:False
struct_harvest_info_header._fields_ = [
    ('signature', ctypes.c_uint32),
    ('version', ctypes.c_uint32),
]

harvest_info_header = struct_harvest_info_header
class struct_harvest_info(Structure):
    pass

struct_harvest_info._pack_ = 1 # source:False
struct_harvest_info._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('number_instance', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

harvest_info = struct_harvest_info
class struct_harvest_table(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', harvest_info_header),
    ('list', struct_harvest_info * 32),
     ]

harvest_table = struct_harvest_table
class struct_mall_info_header(Structure):
    pass

struct_mall_info_header._pack_ = 1 # source:False
struct_mall_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_mall_info_v1_0(Structure):
    pass

struct_mall_info_v1_0._pack_ = 1 # source:False
struct_mall_info_v1_0._fields_ = [
    ('header', struct_mall_info_header),
    ('mall_size_per_m', ctypes.c_uint32),
    ('m_s_present', ctypes.c_uint32),
    ('m_half_use', ctypes.c_uint32),
    ('m_mall_config', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 5),
]

class struct_mall_info_v2_0(Structure):
    pass

struct_mall_info_v2_0._pack_ = 1 # source:False
struct_mall_info_v2_0._fields_ = [
    ('header', struct_mall_info_header),
    ('mall_size_per_umc', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 8),
]

class struct_vcn_info_header(Structure):
    pass

struct_vcn_info_header._pack_ = 1 # source:False
struct_vcn_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_vcn_instance_info_v1_0(Structure):
    pass

class union__fuse_data(Union):
    pass

class struct__fuse_data_bits(Structure):
    pass

struct__fuse_data_bits._pack_ = 1 # source:False
struct__fuse_data_bits._fields_ = [
    ('av1_disabled', ctypes.c_uint32, 1),
    ('vp9_disabled', ctypes.c_uint32, 1),
    ('hevc_disabled', ctypes.c_uint32, 1),
    ('h264_disabled', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 28),
]

union__fuse_data._pack_ = 1 # source:False
union__fuse_data._fields_ = [
    ('bits', struct__fuse_data_bits),
    ('all_bits', ctypes.c_uint32),
]

struct_vcn_instance_info_v1_0._pack_ = 1 # source:False
struct_vcn_instance_info_v1_0._fields_ = [
    ('instance_num', ctypes.c_uint32),
    ('fuse_data', union__fuse_data),
    ('reserved', ctypes.c_uint32 * 2),
]

class struct_vcn_info_v1_0(Structure):
    pass

struct_vcn_info_v1_0._pack_ = 1 # source:False
struct_vcn_info_v1_0._fields_ = [
    ('header', struct_vcn_info_header),
    ('num_of_instances', ctypes.c_uint32),
    ('instance_info', struct_vcn_instance_info_v1_0 * 4),
    ('reserved', ctypes.c_uint32 * 4),
]

class struct_nps_info_header(Structure):
    pass

struct_nps_info_header._pack_ = 1 # source:False
struct_nps_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_nps_instance_info_v1_0(Structure):
    pass

struct_nps_instance_info_v1_0._pack_ = 1 # source:False
struct_nps_instance_info_v1_0._fields_ = [
    ('base_address', ctypes.c_uint64),
    ('limit_address', ctypes.c_uint64),
]

class struct_nps_info_v1_0(Structure):
    pass

struct_nps_info_v1_0._pack_ = 1 # source:False
struct_nps_info_v1_0._fields_ = [
    ('header', struct_nps_info_header),
    ('nps_type', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('instance_info', struct_nps_instance_info_v1_0 * 12),
]


# values for enumeration 'amd_hw_ip_block_type'
amd_hw_ip_block_type__enumvalues = {
    1: 'GC_HWIP',
    2: 'HDP_HWIP',
    3: 'SDMA0_HWIP',
    4: 'SDMA1_HWIP',
    5: 'SDMA2_HWIP',
    6: 'SDMA3_HWIP',
    7: 'SDMA4_HWIP',
    8: 'SDMA5_HWIP',
    9: 'SDMA6_HWIP',
    10: 'SDMA7_HWIP',
    11: 'LSDMA_HWIP',
    12: 'MMHUB_HWIP',
    13: 'ATHUB_HWIP',
    14: 'NBIO_HWIP',
    15: 'MP0_HWIP',
    16: 'MP1_HWIP',
    17: 'UVD_HWIP',
    17: 'VCN_HWIP',
    17: 'JPEG_HWIP',
    18: 'VCN1_HWIP',
    19: 'VCE_HWIP',
    20: 'VPE_HWIP',
    21: 'DF_HWIP',
    22: 'DCE_HWIP',
    23: 'OSSSYS_HWIP',
    24: 'SMUIO_HWIP',
    25: 'PWR_HWIP',
    26: 'NBIF_HWIP',
    27: 'THM_HWIP',
    28: 'CLK_HWIP',
    29: 'UMC_HWIP',
    30: 'RSMU_HWIP',
    31: 'XGMI_HWIP',
    32: 'DCI_HWIP',
    33: 'PCIE_HWIP',
    34: 'ISP_HWIP',
    35: 'MAX_HWIP',
}
GC_HWIP = 1
HDP_HWIP = 2
SDMA0_HWIP = 3
SDMA1_HWIP = 4
SDMA2_HWIP = 5
SDMA3_HWIP = 6
SDMA4_HWIP = 7
SDMA5_HWIP = 8
SDMA6_HWIP = 9
SDMA7_HWIP = 10
LSDMA_HWIP = 11
MMHUB_HWIP = 12
ATHUB_HWIP = 13
NBIO_HWIP = 14
MP0_HWIP = 15
MP1_HWIP = 16
UVD_HWIP = 17
VCN_HWIP = 17
JPEG_HWIP = 17
VCN1_HWIP = 18
VCE_HWIP = 19
VPE_HWIP = 20
DF_HWIP = 21
DCE_HWIP = 22
OSSSYS_HWIP = 23
SMUIO_HWIP = 24
PWR_HWIP = 25
NBIF_HWIP = 26
THM_HWIP = 27
CLK_HWIP = 28
UMC_HWIP = 29
RSMU_HWIP = 30
XGMI_HWIP = 31
DCI_HWIP = 32
PCIE_HWIP = 33
ISP_HWIP = 34
MAX_HWIP = 35
amd_hw_ip_block_type = ctypes.c_uint32 # enum
hw_id_map = [['GC_HWIP', '11'],['HDP_HWIP', '41'],['SDMA0_HWIP', '42'],['SDMA1_HWIP', '43'],['SDMA2_HWIP', '68'],['SDMA3_HWIP', '69'],['LSDMA_HWIP', '91'],['MMHUB_HWIP', '34'],['ATHUB_HWIP', '35'],['NBIO_HWIP', '108'],['MP0_HWIP', '255'],['MP1_HWIP', '1'],['UVD_HWIP', '12'],['VCE_HWIP', '32'],['DF_HWIP', '46'],['DCE_HWIP', '271'],['OSSSYS_HWIP', '40'],['SMUIO_HWIP', '4'],['PWR_HWIP', '10'],['NBIF_HWIP', '108'],['THM_HWIP', '3'],['CLK_HWIP', '6'],['UMC_HWIP', '150'],['XGMI_HWIP', '200'],['DCI_HWIP', '15'],['PCIE_HWIP', '70'],['VPE_HWIP', '21'],['ISP_HWIP', '44']] # Variable ctypes.c_int32 * 35
__all__ = \
    ['ACP_HWID', 'ATHUB_HWID', 'ATHUB_HWIP', 'AUDIO_AZ_HWID',
    'BINARY_SIGNATURE', 'CCXSEC_HWID', 'CLKA_HWID', 'CLKB_HWID',
    'CLK_HWIP', 'DAZ_HWID', 'DBGU0_HWID', 'DBGU1_HWID',
    'DBGU_IO_HWID', 'DBGU_NBIO_HWID', 'DCEAZ_HWID', 'DCE_HWIP',
    'DCI_HWID', 'DCI_HWIP', 'DCO_HWID', 'DDCL_HWID', 'DFX_DAP_HWID',
    'DFX_HWID', 'DF_HWID', 'DF_HWIP', 'DIO_HWID',
    'DISCOVERY_TABLE_SIGNATURE', 'DMU_HWID', 'FCH_HWID',
    'FCH_USB_PD_HWID', 'FUSE_HWID', 'GC', 'GC_HWID', 'GC_HWIP',
    'GC_TABLE_ID', 'HARVEST_INFO', 'HARVEST_TABLE_SIGNATURE',
    'HDP_HWID', 'HDP_HWIP', 'HWIP_MAX_INSTANCE', 'HW_ID_MAX',
    'IOAGR_HWID', 'IOAPIC_HWID', 'IOHC_HWID', 'IP_DISCOVERY',
    'ISP_HWID', 'ISP_HWIP', 'JPEG_HWIP', 'L1IMU10_HWID',
    'L1IMU11_HWID', 'L1IMU12_HWID', 'L1IMU13_HWID', 'L1IMU14_HWID',
    'L1IMU15_HWID', 'L1IMU3_HWID', 'L1IMU4_HWID', 'L1IMU5_HWID',
    'L1IMU6_HWID', 'L1IMU7_HWID', 'L1IMU8_HWID', 'L1IMU9_HWID',
    'L1IMU_IOAGR_HWID', 'L1IMU_NBIF_HWID', 'L1IMU_PCIE_HWID',
    'L2IMU_HWID', 'LSDMA_HWID', 'LSDMA_HWIP', 'MALL_INFO',
    'MALL_INFO_TABLE_ID', 'MAX_HWIP', 'MMHUB_HWID', 'MMHUB_HWIP',
    'MP0_HWID', 'MP0_HWIP', 'MP1_HWID', 'MP1_HWIP', 'MP2_HWID',
    'NBIF_HWID', 'NBIF_HWIP', 'NBIO_HWIP', 'NPS_INFO',
    'NPS_INFO_TABLE_ID', 'NPS_INFO_TABLE_MAX_NUM_INSTANCES',
    'NTBCCP_HWID', 'NTB_HWID', 'OSSSYS_HWID', 'OSSSYS_HWIP',
    'PCIE_HWID', 'PCIE_HWIP', 'PCS_HWID', 'PSP_HEADER_SIZE',
    'PWR_HWID', 'PWR_HWIP', 'RSMU_HWIP', 'SATA_HWID', 'SDMA0_HWID',
    'SDMA0_HWIP', 'SDMA1_HWID', 'SDMA1_HWIP', 'SDMA2_HWID',
    'SDMA2_HWIP', 'SDMA3_HWID', 'SDMA3_HWIP', 'SDMA4_HWIP',
    'SDMA5_HWIP', 'SDMA6_HWIP', 'SDMA7_HWIP', 'SDPMUX_HWID',
    'SMUIO_HWID', 'SMUIO_HWIP', 'SST_HWID', 'SYSTEMHUB_HWID',
    'THM_HWID', 'THM_HWIP', 'TOTAL_TABLES', 'UMC_HWID', 'UMC_HWIP',
    'USB_HWID', 'UVD_HWID', 'UVD_HWIP', 'VCE_HWID', 'VCE_HWIP',
    'VCN1_HWIP', 'VCN_HWID', 'VCN_HWIP', 'VCN_INFO',
    'VCN_INFO_TABLE_ID', 'VCN_INFO_TABLE_MAX_NUM_INSTANCES',
    'VPE_HWID', 'VPE_HWIP', 'WAFLC_HWID', 'XDMA_HWID', 'XGBE_HWID',
    'XGMI_HWID', 'XGMI_HWIP', '_DISCOVERY_H_', 'amd_hw_ip_block_type',
    'binary_header', 'bool', 'c__EA_table', 'die_header', 'die_info',
    'harvest_info', 'harvest_info_header', 'harvest_table',
    'hw_id_map', 'ip', 'ip_discovery_header', 'ip_structure', 'ip_v3',
    'ip_v4', 'struct__fuse_data_bits', 'struct_binary_header',
    'struct_die', 'struct_die_header', 'struct_die_info',
    'struct_gc_info_v1_0', 'struct_gc_info_v1_1',
    'struct_gc_info_v1_2', 'struct_gc_info_v2_0',
    'struct_gc_info_v2_1', 'struct_gpu_info_header',
    'struct_harvest_info', 'struct_harvest_info_header',
    'struct_harvest_table', 'struct_ip', 'struct_ip_discovery_header',
    'struct_ip_discovery_header_0_0', 'struct_ip_structure',
    'struct_ip_v3', 'struct_ip_v4', 'struct_mall_info_header',
    'struct_mall_info_v1_0', 'struct_mall_info_v2_0',
    'struct_nps_info_header', 'struct_nps_info_v1_0',
    'struct_nps_instance_info_v1_0', 'struct_table_info',
    'struct_vcn_info_header', 'struct_vcn_info_v1_0',
    'struct_vcn_instance_info_v1_0', 'table', 'table__enumvalues',
    'table_info', 'u16', 'u32', 'u64', 'u8', 'uint16_t', 'uint32_t',
    'uint64_t', 'uint8_t', 'union__fuse_data', 'union_die_0',
    'union_ip_discovery_header_0']
