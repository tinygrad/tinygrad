# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/src/common/sdk/nvidia/inc/']
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





NV01_DEVICE_0 = (0x00000080) # macro
# NV0080_MAX_DEVICES = NV_MAX_DEVICES # macro
NV0080_ALLOC_PARAMETERS_MESSAGE_ID = (0x0080) # macro
class struct_NV0080_ALLOC_PARAMETERS(Structure):
    pass

struct_NV0080_ALLOC_PARAMETERS._pack_ = 1 # source:False
struct_NV0080_ALLOC_PARAMETERS._fields_ = [
    ('deviceId', ctypes.c_uint32),
    ('hClientShare', ctypes.c_uint32),
    ('hTargetClient', ctypes.c_uint32),
    ('hTargetDevice', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vaSpaceSize', ctypes.c_uint64),
    ('vaStartInternal', ctypes.c_uint64),
    ('vaLimitInternal', ctypes.c_uint64),
    ('vaMode', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

NV0080_ALLOC_PARAMETERS = struct_NV0080_ALLOC_PARAMETERS
_cl2080_notification_h_ = True # macro
NV2080_NOTIFIERS_SW = (0) # macro
NV2080_NOTIFIERS_HOTPLUG = (1) # macro
NV2080_NOTIFIERS_POWER_CONNECTOR = (2) # macro
NV2080_NOTIFIERS_THERMAL_SW = (3) # macro
NV2080_NOTIFIERS_THERMAL_HW = (4) # macro
NV2080_NOTIFIERS_FULL_SCREEN_CHANGE = (5) # macro
NV2080_NOTIFIERS_EVENTBUFFER = (6) # macro
NV2080_NOTIFIERS_DP_IRQ = (7) # macro
NV2080_NOTIFIERS_GR_DEBUG_INTR = (8) # macro
NV2080_NOTIFIERS_PMU_EVENT = (9) # macro
NV2080_NOTIFIERS_PMU_COMMAND = (10) # macro
NV2080_NOTIFIERS_TIMER = (11) # macro
NV2080_NOTIFIERS_GRAPHICS = (12) # macro
NV2080_NOTIFIERS_PPP = (13) # macro
NV2080_NOTIFIERS_VLD = (14) # macro
NV2080_NOTIFIERS_NVDEC0 = (14) # macro
NV2080_NOTIFIERS_NVDEC1 = (15) # macro
NV2080_NOTIFIERS_NVDEC2 = (16) # macro
NV2080_NOTIFIERS_NVDEC3 = (17) # macro
NV2080_NOTIFIERS_NVDEC4 = (18) # macro
NV2080_NOTIFIERS_NVDEC5 = (19) # macro
NV2080_NOTIFIERS_NVDEC6 = (20) # macro
NV2080_NOTIFIERS_NVDEC7 = (21) # macro
NV2080_NOTIFIERS_PDEC = (22) # macro
NV2080_NOTIFIERS_CE0 = (23) # macro
NV2080_NOTIFIERS_CE1 = (24) # macro
NV2080_NOTIFIERS_CE2 = (25) # macro
NV2080_NOTIFIERS_CE3 = (26) # macro
NV2080_NOTIFIERS_CE4 = (27) # macro
NV2080_NOTIFIERS_CE5 = (28) # macro
NV2080_NOTIFIERS_CE6 = (29) # macro
NV2080_NOTIFIERS_CE7 = (30) # macro
NV2080_NOTIFIERS_CE8 = (31) # macro
NV2080_NOTIFIERS_CE9 = (32) # macro
NV2080_NOTIFIERS_PSTATE_CHANGE = (33) # macro
NV2080_NOTIFIERS_HDCP_STATUS_CHANGE = (34) # macro
NV2080_NOTIFIERS_FIFO_EVENT_MTHD = (35) # macro
NV2080_NOTIFIERS_PRIV_RING_HANG = (36) # macro
NV2080_NOTIFIERS_RC_ERROR = (37) # macro
NV2080_NOTIFIERS_MSENC = (38) # macro
NV2080_NOTIFIERS_NVENC0 = (38) # macro
NV2080_NOTIFIERS_NVENC1 = (39) # macro
NV2080_NOTIFIERS_NVENC2 = (40) # macro
NV2080_NOTIFIERS_UNUSED_0 = (41) # macro
NV2080_NOTIFIERS_ACPI_NOTIFY = (42) # macro
NV2080_NOTIFIERS_COOLER_DIAG_ZONE = (43) # macro
NV2080_NOTIFIERS_THERMAL_DIAG_ZONE = (44) # macro
NV2080_NOTIFIERS_AUDIO_HDCP_REQUEST = (45) # macro
NV2080_NOTIFIERS_WORKLOAD_MODULATION_CHANGE = (46) # macro
NV2080_NOTIFIERS_GPIO_0_RISING_INTERRUPT = (47) # macro
NV2080_NOTIFIERS_GPIO_1_RISING_INTERRUPT = (48) # macro
NV2080_NOTIFIERS_GPIO_2_RISING_INTERRUPT = (49) # macro
NV2080_NOTIFIERS_GPIO_3_RISING_INTERRUPT = (50) # macro
NV2080_NOTIFIERS_GPIO_4_RISING_INTERRUPT = (51) # macro
NV2080_NOTIFIERS_GPIO_5_RISING_INTERRUPT = (52) # macro
NV2080_NOTIFIERS_GPIO_6_RISING_INTERRUPT = (53) # macro
NV2080_NOTIFIERS_GPIO_7_RISING_INTERRUPT = (54) # macro
NV2080_NOTIFIERS_GPIO_8_RISING_INTERRUPT = (55) # macro
NV2080_NOTIFIERS_GPIO_9_RISING_INTERRUPT = (56) # macro
NV2080_NOTIFIERS_GPIO_10_RISING_INTERRUPT = (57) # macro
NV2080_NOTIFIERS_GPIO_11_RISING_INTERRUPT = (58) # macro
NV2080_NOTIFIERS_GPIO_12_RISING_INTERRUPT = (59) # macro
NV2080_NOTIFIERS_GPIO_13_RISING_INTERRUPT = (60) # macro
NV2080_NOTIFIERS_GPIO_14_RISING_INTERRUPT = (61) # macro
NV2080_NOTIFIERS_GPIO_15_RISING_INTERRUPT = (62) # macro
NV2080_NOTIFIERS_GPIO_16_RISING_INTERRUPT = (63) # macro
NV2080_NOTIFIERS_GPIO_17_RISING_INTERRUPT = (64) # macro
NV2080_NOTIFIERS_GPIO_18_RISING_INTERRUPT = (65) # macro
NV2080_NOTIFIERS_GPIO_19_RISING_INTERRUPT = (66) # macro
NV2080_NOTIFIERS_GPIO_20_RISING_INTERRUPT = (67) # macro
NV2080_NOTIFIERS_GPIO_21_RISING_INTERRUPT = (68) # macro
NV2080_NOTIFIERS_GPIO_22_RISING_INTERRUPT = (69) # macro
NV2080_NOTIFIERS_GPIO_23_RISING_INTERRUPT = (70) # macro
NV2080_NOTIFIERS_GPIO_24_RISING_INTERRUPT = (71) # macro
NV2080_NOTIFIERS_GPIO_25_RISING_INTERRUPT = (72) # macro
NV2080_NOTIFIERS_GPIO_26_RISING_INTERRUPT = (73) # macro
NV2080_NOTIFIERS_GPIO_27_RISING_INTERRUPT = (74) # macro
NV2080_NOTIFIERS_GPIO_28_RISING_INTERRUPT = (75) # macro
NV2080_NOTIFIERS_GPIO_29_RISING_INTERRUPT = (76) # macro
NV2080_NOTIFIERS_GPIO_30_RISING_INTERRUPT = (77) # macro
NV2080_NOTIFIERS_GPIO_31_RISING_INTERRUPT = (78) # macro
NV2080_NOTIFIERS_GPIO_0_FALLING_INTERRUPT = (79) # macro
NV2080_NOTIFIERS_GPIO_1_FALLING_INTERRUPT = (80) # macro
NV2080_NOTIFIERS_GPIO_2_FALLING_INTERRUPT = (81) # macro
NV2080_NOTIFIERS_GPIO_3_FALLING_INTERRUPT = (82) # macro
NV2080_NOTIFIERS_GPIO_4_FALLING_INTERRUPT = (83) # macro
NV2080_NOTIFIERS_GPIO_5_FALLING_INTERRUPT = (84) # macro
NV2080_NOTIFIERS_GPIO_6_FALLING_INTERRUPT = (85) # macro
NV2080_NOTIFIERS_GPIO_7_FALLING_INTERRUPT = (86) # macro
NV2080_NOTIFIERS_GPIO_8_FALLING_INTERRUPT = (87) # macro
NV2080_NOTIFIERS_GPIO_9_FALLING_INTERRUPT = (88) # macro
NV2080_NOTIFIERS_GPIO_10_FALLING_INTERRUPT = (89) # macro
NV2080_NOTIFIERS_GPIO_11_FALLING_INTERRUPT = (90) # macro
NV2080_NOTIFIERS_GPIO_12_FALLING_INTERRUPT = (91) # macro
NV2080_NOTIFIERS_GPIO_13_FALLING_INTERRUPT = (92) # macro
NV2080_NOTIFIERS_GPIO_14_FALLING_INTERRUPT = (93) # macro
NV2080_NOTIFIERS_GPIO_15_FALLING_INTERRUPT = (94) # macro
NV2080_NOTIFIERS_GPIO_16_FALLING_INTERRUPT = (95) # macro
NV2080_NOTIFIERS_GPIO_17_FALLING_INTERRUPT = (96) # macro
NV2080_NOTIFIERS_GPIO_18_FALLING_INTERRUPT = (97) # macro
NV2080_NOTIFIERS_GPIO_19_FALLING_INTERRUPT = (98) # macro
NV2080_NOTIFIERS_GPIO_20_FALLING_INTERRUPT = (99) # macro
NV2080_NOTIFIERS_GPIO_21_FALLING_INTERRUPT = (100) # macro
NV2080_NOTIFIERS_GPIO_22_FALLING_INTERRUPT = (101) # macro
NV2080_NOTIFIERS_GPIO_23_FALLING_INTERRUPT = (102) # macro
NV2080_NOTIFIERS_GPIO_24_FALLING_INTERRUPT = (103) # macro
NV2080_NOTIFIERS_GPIO_25_FALLING_INTERRUPT = (104) # macro
NV2080_NOTIFIERS_GPIO_26_FALLING_INTERRUPT = (105) # macro
NV2080_NOTIFIERS_GPIO_27_FALLING_INTERRUPT = (106) # macro
NV2080_NOTIFIERS_GPIO_28_FALLING_INTERRUPT = (107) # macro
NV2080_NOTIFIERS_GPIO_29_FALLING_INTERRUPT = (108) # macro
NV2080_NOTIFIERS_GPIO_30_FALLING_INTERRUPT = (109) # macro
NV2080_NOTIFIERS_GPIO_31_FALLING_INTERRUPT = (110) # macro
NV2080_NOTIFIERS_ECC_SBE = (111) # macro
NV2080_NOTIFIERS_ECC_DBE = (112) # macro
NV2080_NOTIFIERS_STEREO_EMITTER_DETECTION = (113) # macro
NV2080_NOTIFIERS_GC5_GPU_READY = (114) # macro
NV2080_NOTIFIERS_SEC2 = (115) # macro
NV2080_NOTIFIERS_GC6_REFCOUNT_INC = (116) # macro
NV2080_NOTIFIERS_GC6_REFCOUNT_DEC = (117) # macro
NV2080_NOTIFIERS_POWER_EVENT = (118) # macro
NV2080_NOTIFIERS_CLOCKS_CHANGE = (119) # macro
NV2080_NOTIFIERS_HOTPLUG_PROCESSING_COMPLETE = (120) # macro
NV2080_NOTIFIERS_PHYSICAL_PAGE_FAULT = (121) # macro
NV2080_NOTIFIERS_RESERVED122 = (122) # macro
NV2080_NOTIFIERS_NVLINK_ERROR_FATAL = (123) # macro
NV2080_NOTIFIERS_PRIV_REG_ACCESS_FAULT = (124) # macro
NV2080_NOTIFIERS_NVLINK_ERROR_RECOVERY_REQUIRED = (125) # macro
NV2080_NOTIFIERS_NVJPG = (126) # macro
NV2080_NOTIFIERS_NVJPEG0 = (126) # macro
NV2080_NOTIFIERS_NVJPEG1 = (127) # macro
NV2080_NOTIFIERS_NVJPEG2 = (128) # macro
NV2080_NOTIFIERS_NVJPEG3 = (129) # macro
NV2080_NOTIFIERS_NVJPEG4 = (130) # macro
NV2080_NOTIFIERS_NVJPEG5 = (131) # macro
NV2080_NOTIFIERS_NVJPEG6 = (132) # macro
NV2080_NOTIFIERS_NVJPEG7 = (133) # macro
NV2080_NOTIFIERS_RUNLIST_AND_ENG_IDLE = (134) # macro
NV2080_NOTIFIERS_RUNLIST_ACQUIRE = (135) # macro
NV2080_NOTIFIERS_RUNLIST_ACQUIRE_AND_ENG_IDLE = (136) # macro
NV2080_NOTIFIERS_RUNLIST_IDLE = (137) # macro
NV2080_NOTIFIERS_TSG_PREEMPT_COMPLETE = (138) # macro
NV2080_NOTIFIERS_RUNLIST_PREEMPT_COMPLETE = (139) # macro
NV2080_NOTIFIERS_CTXSW_TIMEOUT = (140) # macro
NV2080_NOTIFIERS_INFOROM_ECC_OBJECT_UPDATED = (141) # macro
NV2080_NOTIFIERS_NVTELEMETRY_REPORT_EVENT = (142) # macro
NV2080_NOTIFIERS_DSTATE_XUSB_PPC = (143) # macro
NV2080_NOTIFIERS_FECS_CTX_SWITCH = (144) # macro
NV2080_NOTIFIERS_XUSB_PPC_CONNECTED = (145) # macro
NV2080_NOTIFIERS_GR0 = (12) # macro
NV2080_NOTIFIERS_GR1 = (146) # macro
NV2080_NOTIFIERS_GR2 = (147) # macro
NV2080_NOTIFIERS_GR3 = (148) # macro
NV2080_NOTIFIERS_GR4 = (149) # macro
NV2080_NOTIFIERS_GR5 = (150) # macro
NV2080_NOTIFIERS_GR6 = (151) # macro
NV2080_NOTIFIERS_GR7 = (152) # macro
NV2080_NOTIFIERS_OFA = (153) # macro
NV2080_NOTIFIERS_OFA0 = (153) # macro
NV2080_NOTIFIERS_DSTATE_HDA = (154) # macro
NV2080_NOTIFIERS_POISON_ERROR_NON_FATAL = (155) # macro
NV2080_NOTIFIERS_POISON_ERROR_FATAL = (156) # macro
NV2080_NOTIFIERS_UCODE_RESET = (157) # macro
NV2080_NOTIFIERS_PLATFORM_POWER_MODE_CHANGE = (158) # macro
NV2080_NOTIFIERS_SMC_CONFIG_UPDATE = (159) # macro
NV2080_NOTIFIERS_INFOROM_RRL_OBJECT_UPDATED = (160) # macro
NV2080_NOTIFIERS_INFOROM_PBL_OBJECT_UPDATED = (161) # macro
NV2080_NOTIFIERS_LPWR_DIFR_PREFETCH_REQUEST = (162) # macro
NV2080_NOTIFIERS_SEC_FAULT_ERROR = (163) # macro
NV2080_NOTIFIERS_POSSIBLE_ERROR = (164) # macro
NV2080_NOTIFIERS_NVLINK_INFO_LINK_UP = (165) # macro
NV2080_NOTIFIERS_RESERVED166 = (166) # macro
NV2080_NOTIFIERS_RESERVED167 = (167) # macro
NV2080_NOTIFIERS_RESERVED168 = (168) # macro
NV2080_NOTIFIERS_RESERVED169 = (169) # macro
NV2080_NOTIFIERS_RESERVED170 = (170) # macro
NV2080_NOTIFIERS_RESERVED171 = (171) # macro
NV2080_NOTIFIERS_RESERVED172 = (172) # macro
NV2080_NOTIFIERS_RESERVED173 = (173) # macro
NV2080_NOTIFIERS_RESERVED174 = (174) # macro
NV2080_NOTIFIERS_RESERVED175 = (175) # macro
NV2080_NOTIFIERS_NVLINK_INFO_LINK_DOWN = (176) # macro
NV2080_NOTIFIERS_NVPCF_EVENTS = (177) # macro
NV2080_NOTIFIERS_HDMI_FRL_RETRAINING_REQUEST = (178) # macro
NV2080_NOTIFIERS_VRR_SET_TIMEOUT = (179) # macro
NV2080_NOTIFIERS_RESERVED180 = (180) # macro
NV2080_NOTIFIERS_AUX_POWER_EVENT = (181) # macro
NV2080_NOTIFIERS_AUX_POWER_STATE_CHANGE = (182) # macro
NV2080_NOTIFIERS_RESERVED_183 = (183) # macro
NV2080_NOTIFIERS_GSP_PERF_TRACE = (184) # macro
NV2080_NOTIFIERS_INBAND_RESPONSE = (185) # macro
NV2080_NOTIFIERS_RESERVED_186 = (186) # macro
NV2080_NOTIFIERS_MAXCOUNT = (187) # macro
# def NV2080_NOTIFIERS_GR(x):  # macro
#    return ((x==0)?((12)):((146)+(x-1)))  
# def NV2080_NOTIFIERS_GR_IDX(x):  # macro
#    return ((x)-(12))  
# def NV2080_NOTIFIER_TYPE_IS_GR(x):  # macro
#    return (((x)==(12))||(((x)>=(146))&&((x)<=(152))))  
# def NV2080_NOTIFIERS_CE(x):  # macro
#    return ((23)+(x))  
# def NV2080_NOTIFIERS_CE_IDX(x):  # macro
#    return ((x)-(23))  
# def NV2080_NOTIFIER_TYPE_IS_CE(x):  # macro
#    return (((x)>=(23))&&((x)<=(32)))  
# def NV2080_NOTIFIERS_NVENC(x):  # macro
#    return ((38)+(x))  
# def NV2080_NOTIFIERS_NVENC_IDX(x):  # macro
#    return ((x)-(38))  
# def NV2080_NOTIFIER_TYPE_IS_NVENC(x):  # macro
#    return (((x)>=(38))&&((x)<=(40)))  
# def NV2080_NOTIFIERS_NVDEC(x):  # macro
#    return ((14)+(x))  
# def NV2080_NOTIFIERS_NVDEC_IDX(x):  # macro
#    return ((x)-(14))  
# def NV2080_NOTIFIER_TYPE_IS_NVDEC(x):  # macro
#    return (((x)>=(14))&&((x)<=(21)))  
# def NV2080_NOTIFIERS_NVJPEG(x):  # macro
#    return ((126)+(x))  
# def NV2080_NOTIFIERS_NVJPEG_IDX(x):  # macro
#    return ((x)-(126))  
# def NV2080_NOTIFIER_TYPE_IS_NVJPEG(x):  # macro
#    return (((x)>=(126))&&((x)<=(133)))  
# def NV2080_NOTIFIERS_OFAn(x):  # macro
#    return ((x==0)?((153)):((187)))  
# def NV2080_NOTIFIERS_OFA_IDX(x):  # macro
#    return ((x==(153))?(0):(-1))  
# def NV2080_NOTIFIER_TYPE_IS_OFA(x):  # macro
#    return (((x)==(153)))  
# def NV2080_NOTIFIERS_GPIO_RISING_INTERRUPT(pin):  # macro
#    return ((47)+(pin))  
# def NV2080_NOTIFIERS_GPIO_FALLING_INTERRUPT(pin):  # macro
#    return ((79)+(pin))  
NV2080_SUBDEVICE_NOTIFICATION_STATUS_IN_PROGRESS = (0x8000) # macro
NV2080_SUBDEVICE_NOTIFICATION_STATUS_BAD_ARGUMENT = (0x4000) # macro
NV2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_INVALID_STATE = (0x2000) # macro
NV2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_STATE_IN_USE = (0x1000) # macro
NV2080_SUBDEVICE_NOTIFICATION_STATUS_DONE_SUCCESS = (0x0000) # macro
NV2080_ENGINE_TYPE_NULL = (0x00000000) # macro
NV2080_ENGINE_TYPE_GRAPHICS = (0x00000001) # macro
NV2080_ENGINE_TYPE_GR0 = (0x00000001) # macro
NV2080_ENGINE_TYPE_GR1 = (0x00000002) # macro
NV2080_ENGINE_TYPE_GR2 = (0x00000003) # macro
NV2080_ENGINE_TYPE_GR3 = (0x00000004) # macro
NV2080_ENGINE_TYPE_GR4 = (0x00000005) # macro
NV2080_ENGINE_TYPE_GR5 = (0x00000006) # macro
NV2080_ENGINE_TYPE_GR6 = (0x00000007) # macro
NV2080_ENGINE_TYPE_GR7 = (0x00000008) # macro
NV2080_ENGINE_TYPE_COPY0 = (0x00000009) # macro
NV2080_ENGINE_TYPE_COPY1 = (0x0000000a) # macro
NV2080_ENGINE_TYPE_COPY2 = (0x0000000b) # macro
NV2080_ENGINE_TYPE_COPY3 = (0x0000000c) # macro
NV2080_ENGINE_TYPE_COPY4 = (0x0000000d) # macro
NV2080_ENGINE_TYPE_COPY5 = (0x0000000e) # macro
NV2080_ENGINE_TYPE_COPY6 = (0x0000000f) # macro
NV2080_ENGINE_TYPE_COPY7 = (0x00000010) # macro
NV2080_ENGINE_TYPE_COPY8 = (0x00000011) # macro
NV2080_ENGINE_TYPE_COPY9 = (0x00000012) # macro
NV2080_ENGINE_TYPE_BSP = (0x00000013) # macro
NV2080_ENGINE_TYPE_NVDEC0 = (0x00000013) # macro
NV2080_ENGINE_TYPE_NVDEC1 = (0x00000014) # macro
NV2080_ENGINE_TYPE_NVDEC2 = (0x00000015) # macro
NV2080_ENGINE_TYPE_NVDEC3 = (0x00000016) # macro
NV2080_ENGINE_TYPE_NVDEC4 = (0x00000017) # macro
NV2080_ENGINE_TYPE_NVDEC5 = (0x00000018) # macro
NV2080_ENGINE_TYPE_NVDEC6 = (0x00000019) # macro
NV2080_ENGINE_TYPE_NVDEC7 = (0x0000001a) # macro
NV2080_ENGINE_TYPE_MSENC = (0x0000001b) # macro
NV2080_ENGINE_TYPE_NVENC0 = (0x0000001b) # macro
NV2080_ENGINE_TYPE_NVENC1 = (0x0000001c) # macro
NV2080_ENGINE_TYPE_NVENC2 = (0x0000001d) # macro
NV2080_ENGINE_TYPE_VP = (0x0000001e) # macro
NV2080_ENGINE_TYPE_ME = (0x0000001f) # macro
NV2080_ENGINE_TYPE_PPP = (0x00000020) # macro
NV2080_ENGINE_TYPE_MPEG = (0x00000021) # macro
NV2080_ENGINE_TYPE_SW = (0x00000022) # macro
NV2080_ENGINE_TYPE_CIPHER = (0x00000023) # macro
NV2080_ENGINE_TYPE_TSEC = (0x00000023) # macro
NV2080_ENGINE_TYPE_VIC = (0x00000024) # macro
NV2080_ENGINE_TYPE_MP = (0x00000025) # macro
NV2080_ENGINE_TYPE_SEC2 = (0x00000026) # macro
NV2080_ENGINE_TYPE_HOST = (0x00000027) # macro
NV2080_ENGINE_TYPE_DPU = (0x00000028) # macro
NV2080_ENGINE_TYPE_PMU = (0x00000029) # macro
NV2080_ENGINE_TYPE_FBFLCN = (0x0000002a) # macro
NV2080_ENGINE_TYPE_NVJPG = (0x0000002b) # macro
NV2080_ENGINE_TYPE_NVJPEG0 = (0x0000002b) # macro
NV2080_ENGINE_TYPE_NVJPEG1 = (0x0000002c) # macro
NV2080_ENGINE_TYPE_NVJPEG2 = (0x0000002d) # macro
NV2080_ENGINE_TYPE_NVJPEG3 = (0x0000002e) # macro
NV2080_ENGINE_TYPE_NVJPEG4 = (0x0000002f) # macro
NV2080_ENGINE_TYPE_NVJPEG5 = (0x00000030) # macro
NV2080_ENGINE_TYPE_NVJPEG6 = (0x00000031) # macro
NV2080_ENGINE_TYPE_NVJPEG7 = (0x00000032) # macro
NV2080_ENGINE_TYPE_OFA = (0x00000033) # macro
NV2080_ENGINE_TYPE_OFA0 = (0x00000033) # macro
NV2080_ENGINE_TYPE_RESERVED34 = (0x00000034) # macro
NV2080_ENGINE_TYPE_RESERVED35 = (0x00000035) # macro
NV2080_ENGINE_TYPE_RESERVED36 = (0x00000036) # macro
NV2080_ENGINE_TYPE_RESERVED37 = (0x00000037) # macro
NV2080_ENGINE_TYPE_RESERVED38 = (0x00000038) # macro
NV2080_ENGINE_TYPE_RESERVED39 = (0x00000039) # macro
NV2080_ENGINE_TYPE_RESERVED3a = (0x0000003a) # macro
NV2080_ENGINE_TYPE_RESERVED3b = (0x0000003b) # macro
NV2080_ENGINE_TYPE_RESERVED3c = (0x0000003c) # macro
NV2080_ENGINE_TYPE_RESERVED3d = (0x0000003d) # macro
NV2080_ENGINE_TYPE_RESERVED3e = (0x0000003e) # macro
NV2080_ENGINE_TYPE_RESERVED3f = (0x0000003f) # macro
NV2080_ENGINE_TYPE_LAST = (0x00000040) # macro
NV2080_ENGINE_TYPE_ALLENGINES = (0xffffffff) # macro
NV2080_ENGINE_TYPE_COPY_SIZE = 64 # macro
NV2080_ENGINE_TYPE_NVENC_SIZE = 3 # macro
NV2080_ENGINE_TYPE_NVJPEG_SIZE = 8 # macro
NV2080_ENGINE_TYPE_NVDEC_SIZE = 8 # macro
NV2080_ENGINE_TYPE_GR_SIZE = 8 # macro
NV2080_ENGINE_TYPE_OFA_SIZE = 1 # macro
# def NV2080_ENGINE_TYPE_COPY(i):  # macro
#    return ((0x00000009)+(i))  
# def NV2080_ENGINE_TYPE_IS_COPY(i):  # macro
#    return (((i)>=(0x00000009))&&((i)<=(0x00000012)))  
# def NV2080_ENGINE_TYPE_COPY_IDX(i):  # macro
#    return ((i)-(0x00000009))  
# def NV2080_ENGINE_TYPE_NVENC(i):  # macro
#    return ((0x0000001b)+(i))  
# def NV2080_ENGINE_TYPE_IS_NVENC(i):  # macro
#    return (((i)>=(0x0000001b))&&((i)<((0x0000001b)+(i))(3)))  
# def NV2080_ENGINE_TYPE_NVENC_IDX(i):  # macro
#    return ((i)-(0x0000001b))  
# def NV2080_ENGINE_TYPE_NVDEC(i):  # macro
#    return ((0x00000013)+(i))  
# def NV2080_ENGINE_TYPE_IS_NVDEC(i):  # macro
#    return (((i)>=(0x00000013))&&((i)<((0x00000013)+(i))(8)))  
# def NV2080_ENGINE_TYPE_NVDEC_IDX(i):  # macro
#    return ((i)-(0x00000013))  
# def NV2080_ENGINE_TYPE_NVJPEG(i):  # macro
#    return ((0x0000002b)+(i))  
# def NV2080_ENGINE_TYPE_IS_NVJPEG(i):  # macro
#    return (((i)>=(0x0000002b))&&((i)<((0x0000002b)+(i))(8)))  
# def NV2080_ENGINE_TYPE_NVJPEG_IDX(i):  # macro
#    return ((i)-(0x0000002b))  
# def NV2080_ENGINE_TYPE_GR(i):  # macro
#    return ((0x00000001)+(i))  
# def NV2080_ENGINE_TYPE_IS_GR(i):  # macro
#    return (((i)>=(0x00000001))&&((i)<((0x00000001)+(i))(8)))  
# def NV2080_ENGINE_TYPE_GR_IDX(i):  # macro
#    return ((i)-(0x00000001))  
# def NV2080_ENGINE_TYPE_OFAn(i):  # macro
#    return ((i==0)?((0x00000033)):((0x00000040)))  
# def NV2080_ENGINE_TYPE_IS_OFA(i):  # macro
#    return (((i)==(0x00000033)))  
# def NV2080_ENGINE_TYPE_OFA_IDX(i):  # macro
#    return ((i==(0x00000033))?(0):(-1))  
# def NV2080_ENGINE_TYPE_IS_VALID(i):  # macro
#    return (((i)>((0x00000000)))&&((i)<((0x00000040))))  
NV2080_CLIENT_TYPE_TEX = (0x00000001) # macro
NV2080_CLIENT_TYPE_COLOR = (0x00000002) # macro
NV2080_CLIENT_TYPE_DEPTH = (0x00000003) # macro
NV2080_CLIENT_TYPE_DA = (0x00000004) # macro
NV2080_CLIENT_TYPE_FE = (0x00000005) # macro
NV2080_CLIENT_TYPE_SCC = (0x00000006) # macro
NV2080_CLIENT_TYPE_WID = (0x00000007) # macro
NV2080_CLIENT_TYPE_MSVLD = (0x00000008) # macro
NV2080_CLIENT_TYPE_MSPDEC = (0x00000009) # macro
NV2080_CLIENT_TYPE_MSPPP = (0x0000000a) # macro
NV2080_CLIENT_TYPE_VIC = (0x0000000b) # macro
NV2080_CLIENT_TYPE_ALLCLIENTS = (0xffffffff) # macro
NV2080_GC5_EXIT_COMPLETE = (0x00000001) # macro
NV2080_GC5_ENTRY_ABORTED = (0x00000002) # macro
NV2080_PLATFORM_POWER_MODE_CHANGE_COMPLETION = (0x00000000) # macro
NV2080_PLATFORM_POWER_MODE_CHANGE_ACPI_NOTIFICATION = (0x00000001) # macro
NV2080_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT = (0x4000) # macro
# NV2080_TYPEDEF = Nv20Subdevice0 # macro
NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_INDEX = ['7', ':', '0'] # macro
NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_MASK = ['15', ':', '8'] # macro
NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_REASON = ['23', ':', '16'] # macro
class struct__cl2080_tag0(Structure):
    pass

struct__cl2080_tag0._pack_ = 1 # source:False
struct__cl2080_tag0._fields_ = [
    ('Reserved00', ctypes.c_uint32 * 1984),
]

Nv2080Typedef = struct__cl2080_tag0
Nv20Subdevice0 = struct__cl2080_tag0
class struct_Nv2080HdcpStatusChangeNotificationRec(Structure):
    pass

struct_Nv2080HdcpStatusChangeNotificationRec._pack_ = 1 # source:False
struct_Nv2080HdcpStatusChangeNotificationRec._fields_ = [
    ('displayId', ctypes.c_uint32),
    ('hdcpStatusChangeNotif', ctypes.c_uint32),
]

Nv2080HdcpStatusChangeNotification = struct_Nv2080HdcpStatusChangeNotificationRec
class struct_Nv2080PStateChangeNotificationRec(Structure):
    pass

class struct_Nv2080PStateChangeNotificationRec_timeStamp(Structure):
    pass

struct_Nv2080PStateChangeNotificationRec_timeStamp._pack_ = 1 # source:False
struct_Nv2080PStateChangeNotificationRec_timeStamp._fields_ = [
    ('nanoseconds', ctypes.c_uint32 * 2),
]

struct_Nv2080PStateChangeNotificationRec._pack_ = 1 # source:False
struct_Nv2080PStateChangeNotificationRec._fields_ = [
    ('timeStamp', struct_Nv2080PStateChangeNotificationRec_timeStamp),
    ('NewPstate', ctypes.c_uint32),
]

Nv2080PStateChangeNotification = struct_Nv2080PStateChangeNotificationRec
class struct_Nv2080ClocksChangeNotificationRec(Structure):
    pass

class struct_Nv2080ClocksChangeNotificationRec_timeStamp(Structure):
    pass

struct_Nv2080ClocksChangeNotificationRec_timeStamp._pack_ = 1 # source:False
struct_Nv2080ClocksChangeNotificationRec_timeStamp._fields_ = [
    ('nanoseconds', ctypes.c_uint32 * 2),
]

struct_Nv2080ClocksChangeNotificationRec._pack_ = 1 # source:False
struct_Nv2080ClocksChangeNotificationRec._fields_ = [
    ('timeStamp', struct_Nv2080ClocksChangeNotificationRec_timeStamp),
]

Nv2080ClocksChangeNotification = struct_Nv2080ClocksChangeNotificationRec
class struct_Nv2080WorkloadModulationChangeNotificationRec(Structure):
    pass

class struct_Nv2080WorkloadModulationChangeNotificationRec_timeStamp(Structure):
    pass

struct_Nv2080WorkloadModulationChangeNotificationRec_timeStamp._pack_ = 1 # source:False
struct_Nv2080WorkloadModulationChangeNotificationRec_timeStamp._fields_ = [
    ('nanoseconds', ctypes.c_uint32 * 2),
]

struct_Nv2080WorkloadModulationChangeNotificationRec._pack_ = 1 # source:False
struct_Nv2080WorkloadModulationChangeNotificationRec._fields_ = [
    ('timeStamp', struct_Nv2080WorkloadModulationChangeNotificationRec_timeStamp),
    ('WorkloadModulationEnabled', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

Nv2080WorkloadModulationChangeNotification = struct_Nv2080WorkloadModulationChangeNotificationRec
class struct_c__SA_Nv2080HotplugNotification(Structure):
    pass

struct_c__SA_Nv2080HotplugNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080HotplugNotification._fields_ = [
    ('plugDisplayMask', ctypes.c_uint32),
    ('unplugDisplayMask', ctypes.c_uint32),
]

Nv2080HotplugNotification = struct_c__SA_Nv2080HotplugNotification
class struct_c__SA_Nv2080PowerEventNotification(Structure):
    pass

struct_c__SA_Nv2080PowerEventNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080PowerEventNotification._fields_ = [
    ('bSwitchToAC', ctypes.c_ubyte),
    ('bGPUCapabilityChanged', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('displayMaskAffected', ctypes.c_uint32),
]

Nv2080PowerEventNotification = struct_c__SA_Nv2080PowerEventNotification
class struct_Nv2080DpIrqNotificationRec(Structure):
    pass

struct_Nv2080DpIrqNotificationRec._pack_ = 1 # source:False
struct_Nv2080DpIrqNotificationRec._fields_ = [
    ('displayId', ctypes.c_uint32),
]

Nv2080DpIrqNotification = struct_Nv2080DpIrqNotificationRec
class struct_Nv2080DstateXusbPpcNotificationRec(Structure):
    pass

struct_Nv2080DstateXusbPpcNotificationRec._pack_ = 1 # source:False
struct_Nv2080DstateXusbPpcNotificationRec._fields_ = [
    ('dstateXusb', ctypes.c_uint32),
    ('dstatePpc', ctypes.c_uint32),
]

Nv2080DstateXusbPpcNotification = struct_Nv2080DstateXusbPpcNotificationRec
class struct_Nv2080XusbPpcConnectStateNotificationRec(Structure):
    pass

struct_Nv2080XusbPpcConnectStateNotificationRec._pack_ = 1 # source:False
struct_Nv2080XusbPpcConnectStateNotificationRec._fields_ = [
    ('bConnected', ctypes.c_ubyte),
]

Nv2080XusbPpcConnectStateNotification = struct_Nv2080XusbPpcConnectStateNotificationRec
class struct_Nv2080ACPIEvent(Structure):
    pass

struct_Nv2080ACPIEvent._pack_ = 1 # source:False
struct_Nv2080ACPIEvent._fields_ = [
    ('event', ctypes.c_uint32),
]

Nv2080ACPIEvent = struct_Nv2080ACPIEvent
class struct__NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC(Structure):
    pass

struct__NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC._pack_ = 1 # source:False
struct__NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC._fields_ = [
    ('currentZone', ctypes.c_uint32),
]

NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC = struct__NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC
class struct__NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC(Structure):
    pass

struct__NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC._pack_ = 1 # source:False
struct__NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC._fields_ = [
    ('currentZone', ctypes.c_uint32),
]

NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC = struct__NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC
class struct_Nv2080AudioHdcpRequestRec(Structure):
    pass

struct_Nv2080AudioHdcpRequestRec._pack_ = 1 # source:False
struct_Nv2080AudioHdcpRequestRec._fields_ = [
    ('displayId', ctypes.c_uint32),
    ('requestedState', ctypes.c_uint32),
]

Nv2080AudioHdcpRequest = struct_Nv2080AudioHdcpRequestRec
class struct_Nv2080GC5GpuReadyParams(Structure):
    pass

struct_Nv2080GC5GpuReadyParams._pack_ = 1 # source:False
struct_Nv2080GC5GpuReadyParams._fields_ = [
    ('event', ctypes.c_uint32),
    ('sciIntr0', ctypes.c_uint32),
    ('sciIntr1', ctypes.c_uint32),
]

Nv2080GC5GpuReadyParams = struct_Nv2080GC5GpuReadyParams
class struct_c__SA_Nv2080PrivRegAccessFaultNotification(Structure):
    pass

struct_c__SA_Nv2080PrivRegAccessFaultNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080PrivRegAccessFaultNotification._fields_ = [
    ('errAddr', ctypes.c_uint32),
]

Nv2080PrivRegAccessFaultNotification = struct_c__SA_Nv2080PrivRegAccessFaultNotification
class struct_Nv2080DstateHdaCodecNotificationRec(Structure):
    pass

struct_Nv2080DstateHdaCodecNotificationRec._pack_ = 1 # source:False
struct_Nv2080DstateHdaCodecNotificationRec._fields_ = [
    ('dstateHdaCodec', ctypes.c_uint32),
]

Nv2080DstateHdaCodecNotification = struct_Nv2080DstateHdaCodecNotificationRec
class struct_Nv2080HdmiFrlRequestNotificationRec(Structure):
    pass

struct_Nv2080HdmiFrlRequestNotificationRec._pack_ = 1 # source:False
struct_Nv2080HdmiFrlRequestNotificationRec._fields_ = [
    ('displayId', ctypes.c_uint32),
]

Nv2080HdmiFrlRequestNotification = struct_Nv2080HdmiFrlRequestNotificationRec
class struct__NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS(Structure):
    pass

struct__NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS._pack_ = 1 # source:False
struct__NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS._fields_ = [
    ('platformPowerModeIndex', ctypes.c_ubyte),
    ('platformPowerModeMask', ctypes.c_ubyte),
    ('eventReason', ctypes.c_ubyte),
]

NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS = struct__NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS
class struct_c__SA_Nv2080QosIntrNotification(Structure):
    pass

struct_c__SA_Nv2080QosIntrNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080QosIntrNotification._fields_ = [
    ('engineType', ctypes.c_uint32),
]

Nv2080QosIntrNotification = struct_c__SA_Nv2080QosIntrNotification
class struct_c__SA_Nv2080EccDbeNotification(Structure):
    pass

struct_c__SA_Nv2080EccDbeNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080EccDbeNotification._fields_ = [
    ('physAddress', ctypes.c_uint64),
]

Nv2080EccDbeNotification = struct_c__SA_Nv2080EccDbeNotification
class struct_c__SA_Nv2080LpwrDifrPrefetchNotification(Structure):
    pass

struct_c__SA_Nv2080LpwrDifrPrefetchNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080LpwrDifrPrefetchNotification._fields_ = [
    ('l2CacheSize', ctypes.c_uint32),
]

Nv2080LpwrDifrPrefetchNotification = struct_c__SA_Nv2080LpwrDifrPrefetchNotification
class struct_c__SA_Nv2080NvlinkLnkChangeNotification(Structure):
    pass

struct_c__SA_Nv2080NvlinkLnkChangeNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080NvlinkLnkChangeNotification._fields_ = [
    ('GpuId', ctypes.c_uint32),
    ('linkId', ctypes.c_uint32),
]

Nv2080NvlinkLnkChangeNotification = struct_c__SA_Nv2080NvlinkLnkChangeNotification
class struct_c__SA_Nv2080VrrSetTimeoutNotification(Structure):
    pass

struct_c__SA_Nv2080VrrSetTimeoutNotification._pack_ = 1 # source:False
struct_c__SA_Nv2080VrrSetTimeoutNotification._fields_ = [
    ('head', ctypes.c_uint32),
]

Nv2080VrrSetTimeoutNotification = struct_c__SA_Nv2080VrrSetTimeoutNotification
_clc56f_h_ = True # macro
AMPERE_CHANNEL_GPFIFO_A = (0x0000c56f) # macro
# NVC56F_TYPEDEF = AMPERE_CHANNELChannelGPFifoA # macro
NVC56F_NUMBER_OF_SUBCHANNELS = (8) # macro
NVC56F_SET_OBJECT = (0x00000000) # macro
NVC56F_SET_OBJECT_NVCLASS = ['15', ':', '0'] # macro
NVC56F_SET_OBJECT_ENGINE = ['20', ':', '16'] # macro
NVC56F_SET_OBJECT_ENGINE_SW = 0x0000001f # macro
NVC56F_ILLEGAL = (0x00000004) # macro
NVC56F_ILLEGAL_HANDLE = ['31', ':', '0'] # macro
NVC56F_NOP = (0x00000008) # macro
NVC56F_NOP_HANDLE = ['31', ':', '0'] # macro
NVC56F_SEMAPHOREA = (0x00000010) # macro
NVC56F_SEMAPHOREA_OFFSET_UPPER = ['7', ':', '0'] # macro
NVC56F_SEMAPHOREB = (0x00000014) # macro
NVC56F_SEMAPHOREB_OFFSET_LOWER = ['31', ':', '2'] # macro
NVC56F_SEMAPHOREC = (0x00000018) # macro
NVC56F_SEMAPHOREC_PAYLOAD = ['31', ':', '0'] # macro
NVC56F_SEMAPHORED = (0x0000001C) # macro
NVC56F_SEMAPHORED_OPERATION = ['4', ':', '0'] # macro
NVC56F_SEMAPHORED_OPERATION_ACQUIRE = 0x00000001 # macro
NVC56F_SEMAPHORED_OPERATION_RELEASE = 0x00000002 # macro
NVC56F_SEMAPHORED_OPERATION_ACQ_GEQ = 0x00000004 # macro
NVC56F_SEMAPHORED_OPERATION_ACQ_AND = 0x00000008 # macro
NVC56F_SEMAPHORED_OPERATION_REDUCTION = 0x00000010 # macro
NVC56F_SEMAPHORED_ACQUIRE_SWITCH = ['12', ':', '12'] # macro
NVC56F_SEMAPHORED_ACQUIRE_SWITCH_DISABLED = 0x00000000 # macro
NVC56F_SEMAPHORED_ACQUIRE_SWITCH_ENABLED = 0x00000001 # macro
NVC56F_SEMAPHORED_RELEASE_WFI = ['20', ':', '20'] # macro
NVC56F_SEMAPHORED_RELEASE_WFI_EN = 0x00000000 # macro
NVC56F_SEMAPHORED_RELEASE_WFI_DIS = 0x00000001 # macro
NVC56F_SEMAPHORED_RELEASE_SIZE = ['24', ':', '24'] # macro
NVC56F_SEMAPHORED_RELEASE_SIZE_16BYTE = 0x00000000 # macro
NVC56F_SEMAPHORED_RELEASE_SIZE_4BYTE = 0x00000001 # macro
NVC56F_SEMAPHORED_REDUCTION = ['30', ':', '27'] # macro
NVC56F_SEMAPHORED_REDUCTION_MIN = 0x00000000 # macro
NVC56F_SEMAPHORED_REDUCTION_MAX = 0x00000001 # macro
NVC56F_SEMAPHORED_REDUCTION_XOR = 0x00000002 # macro
NVC56F_SEMAPHORED_REDUCTION_AND = 0x00000003 # macro
NVC56F_SEMAPHORED_REDUCTION_OR = 0x00000004 # macro
NVC56F_SEMAPHORED_REDUCTION_ADD = 0x00000005 # macro
NVC56F_SEMAPHORED_REDUCTION_INC = 0x00000006 # macro
NVC56F_SEMAPHORED_REDUCTION_DEC = 0x00000007 # macro
NVC56F_SEMAPHORED_FORMAT = ['31', ':', '31'] # macro
NVC56F_SEMAPHORED_FORMAT_SIGNED = 0x00000000 # macro
NVC56F_SEMAPHORED_FORMAT_UNSIGNED = 0x00000001 # macro
NVC56F_NON_STALL_INTERRUPT = (0x00000020) # macro
NVC56F_NON_STALL_INTERRUPT_HANDLE = ['31', ':', '0'] # macro
NVC56F_FB_FLUSH = (0x00000024) # macro
NVC56F_FB_FLUSH_HANDLE = ['31', ':', '0'] # macro
NVC56F_MEM_OP_A = (0x00000028) # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_TARGET_CLIENT_UNIT_ID = ['5', ':', '0'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVALIDATION_SIZE = ['5', ':', '0'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_TARGET_GPC_ID = ['10', ':', '6'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE = ['7', ':', '6'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_ALL_TLBS = 0 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_LINK_TLBS = 1 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_NON_LINK_TLBS = 2 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_RSVRVD = 3 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_MMU_ENGINE_ID = ['6', ':', '0'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR = ['11', ':', '11'] # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_EN = 0x00000001 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_DIS = 0x00000000 # macro
NVC56F_MEM_OP_A_TLB_INVALIDATE_TARGET_ADDR_LO = ['31', ':', '12'] # macro
NVC56F_MEM_OP_B = (0x0000002c) # macro
NVC56F_MEM_OP_B_TLB_INVALIDATE_TARGET_ADDR_HI = ['31', ':', '0'] # macro
NVC56F_MEM_OP_C = (0x00000030) # macro
NVC56F_MEM_OP_C_MEMBAR_TYPE = ['2', ':', '0'] # macro
NVC56F_MEM_OP_C_MEMBAR_TYPE_SYS_MEMBAR = 0x00000000 # macro
NVC56F_MEM_OP_C_MEMBAR_TYPE_MEMBAR = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB = ['0', ':', '0'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ONE = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ALL = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC = ['1', ':', '1'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC_ENABLE = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC_DISABLE = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY = ['4', ':', '2'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_NONE = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_START = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_START_ACK_ALL = 0x00000002 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_TARGETED = 0x00000003 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_GLOBAL = 0x00000004 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_VA_GLOBAL = 0x00000005 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE = ['6', ':', '5'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_NONE = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_GLOBALLY = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_INTRANODE = 0x00000002 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE = ['9', ':', '7'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_READ = 0 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_WRITE = 1 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_STRONG = 2 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_RSVRVD = 3 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_WEAK = 4 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_ALL = 5 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_WRITE_AND_ATOMIC = 6 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ALL = 7 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL = ['9', ':', '7'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_ALL = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_PTE_ONLY = 0x00000001 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE0 = 0x00000002 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE1 = 0x00000003 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE2 = 0x00000004 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE3 = 0x00000005 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE4 = 0x00000006 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE5 = 0x00000007 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE = ['11', ':', '10'] # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_VID_MEM = 0x00000000 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_SYS_MEM_COHERENT = 0x00000002 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_SYS_MEM_NONCOHERENT = 0x00000003 # macro
NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ADDR_LO = ['31', ':', '12'] # macro
NVC56F_MEM_OP_C_ACCESS_COUNTER_CLR_TARGETED_NOTIFY_TAG = ['19', ':', '0'] # macro
NVC56F_MEM_OP_D = (0x00000034) # macro
NVC56F_MEM_OP_D_TLB_INVALIDATE_PDB_ADDR_HI = ['26', ':', '0'] # macro
NVC56F_MEM_OP_D_OPERATION = ['31', ':', '27'] # macro
NVC56F_MEM_OP_D_OPERATION_MEMBAR = 0x00000005 # macro
NVC56F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE = 0x00000009 # macro
NVC56F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE_TARGETED = 0x0000000a # macro
NVC56F_MEM_OP_D_OPERATION_L2_PEERMEM_INVALIDATE = 0x0000000d # macro
NVC56F_MEM_OP_D_OPERATION_L2_SYSMEM_INVALIDATE = 0x0000000e # macro
NVC56F_MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES = 0x0000000e # macro
NVC56F_MEM_OP_D_OPERATION_L2_CLEAN_COMPTAGS = 0x0000000f # macro
NVC56F_MEM_OP_D_OPERATION_L2_FLUSH_DIRTY = 0x00000010 # macro
NVC56F_MEM_OP_D_OPERATION_L2_WAIT_FOR_SYS_PENDING_READS = 0x00000015 # macro
NVC56F_MEM_OP_D_OPERATION_ACCESS_COUNTER_CLR = 0x00000016 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE = ['1', ':', '0'] # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MIMC = 0x00000000 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MOMC = 0x00000001 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_ALL = 0x00000002 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_TARGETED = 0x00000003 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE = ['2', ':', '2'] # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE_MIMC = 0x00000000 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE_MOMC = 0x00000001 # macro
NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_BANK = ['6', ':', '3'] # macro
NVC56F_SET_REFERENCE = (0x00000050) # macro
NVC56F_SET_REFERENCE_COUNT = ['31', ':', '0'] # macro
NVC56F_SEM_ADDR_LO = (0x0000005c) # macro
NVC56F_SEM_ADDR_LO_OFFSET = ['31', ':', '2'] # macro
NVC56F_SEM_ADDR_HI = (0x00000060) # macro
NVC56F_SEM_ADDR_HI_OFFSET = ['7', ':', '0'] # macro
NVC56F_SEM_PAYLOAD_LO = (0x00000064) # macro
NVC56F_SEM_PAYLOAD_LO_PAYLOAD = ['31', ':', '0'] # macro
NVC56F_SEM_PAYLOAD_HI = (0x00000068) # macro
NVC56F_SEM_PAYLOAD_HI_PAYLOAD = ['31', ':', '0'] # macro
NVC56F_SEM_EXECUTE = (0x0000006c) # macro
NVC56F_SEM_EXECUTE_OPERATION = ['2', ':', '0'] # macro
NVC56F_SEM_EXECUTE_OPERATION_ACQUIRE = 0x00000000 # macro
NVC56F_SEM_EXECUTE_OPERATION_RELEASE = 0x00000001 # macro
NVC56F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ = 0x00000002 # macro
NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ = 0x00000003 # macro
NVC56F_SEM_EXECUTE_OPERATION_ACQ_AND = 0x00000004 # macro
NVC56F_SEM_EXECUTE_OPERATION_ACQ_NOR = 0x00000005 # macro
NVC56F_SEM_EXECUTE_OPERATION_REDUCTION = 0x00000006 # macro
NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG = ['12', ':', '12'] # macro
NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG_DIS = 0x00000000 # macro
NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG_EN = 0x00000001 # macro
NVC56F_SEM_EXECUTE_RELEASE_WFI = ['20', ':', '20'] # macro
NVC56F_SEM_EXECUTE_RELEASE_WFI_DIS = 0x00000000 # macro
NVC56F_SEM_EXECUTE_RELEASE_WFI_EN = 0x00000001 # macro
NVC56F_SEM_EXECUTE_PAYLOAD_SIZE = ['24', ':', '24'] # macro
NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_32BIT = 0x00000000 # macro
NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT = 0x00000001 # macro
NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP = ['25', ':', '25'] # macro
NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS = 0x00000000 # macro
NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN = 0x00000001 # macro
NVC56F_SEM_EXECUTE_REDUCTION = ['30', ':', '27'] # macro
NVC56F_SEM_EXECUTE_REDUCTION_IMIN = 0x00000000 # macro
NVC56F_SEM_EXECUTE_REDUCTION_IMAX = 0x00000001 # macro
NVC56F_SEM_EXECUTE_REDUCTION_IXOR = 0x00000002 # macro
NVC56F_SEM_EXECUTE_REDUCTION_IAND = 0x00000003 # macro
NVC56F_SEM_EXECUTE_REDUCTION_IOR = 0x00000004 # macro
NVC56F_SEM_EXECUTE_REDUCTION_IADD = 0x00000005 # macro
NVC56F_SEM_EXECUTE_REDUCTION_INC = 0x00000006 # macro
NVC56F_SEM_EXECUTE_REDUCTION_DEC = 0x00000007 # macro
NVC56F_SEM_EXECUTE_REDUCTION_FORMAT = ['31', ':', '31'] # macro
NVC56F_SEM_EXECUTE_REDUCTION_FORMAT_SIGNED = 0x00000000 # macro
NVC56F_SEM_EXECUTE_REDUCTION_FORMAT_UNSIGNED = 0x00000001 # macro
NVC56F_WFI = (0x00000078) # macro
NVC56F_WFI_SCOPE = ['0', ':', '0'] # macro
NVC56F_WFI_SCOPE_CURRENT_SCG_TYPE = 0x00000000 # macro
NVC56F_WFI_SCOPE_CURRENT_VEID = 0x00000000 # macro
NVC56F_WFI_SCOPE_ALL = 0x00000001 # macro
NVC56F_YIELD = (0x00000080) # macro
NVC56F_YIELD_OP = ['1', ':', '0'] # macro
NVC56F_YIELD_OP_NOP = 0x00000000 # macro
NVC56F_YIELD_OP_TSG = 0x00000003 # macro
NVC56F_CLEAR_FAULTED = (0x00000084) # macro
NVC56F_CLEAR_FAULTED_HANDLE = ['30', ':', '0'] # macro
NVC56F_CLEAR_FAULTED_TYPE = ['31', ':', '31'] # macro
NVC56F_CLEAR_FAULTED_TYPE_PBDMA_FAULTED = 0x00000000 # macro
NVC56F_CLEAR_FAULTED_TYPE_ENG_FAULTED = 0x00000001 # macro
NVC56F_GP_ENTRY__SIZE = 8 # macro
NVC56F_GP_ENTRY0_FETCH = ['0', ':', '0'] # macro
NVC56F_GP_ENTRY0_FETCH_UNCONDITIONAL = 0x00000000 # macro
NVC56F_GP_ENTRY0_FETCH_CONDITIONAL = 0x00000001 # macro
NVC56F_GP_ENTRY0_GET = ['31', ':', '2'] # macro
NVC56F_GP_ENTRY0_OPERAND = ['31', ':', '0'] # macro
NVC56F_GP_ENTRY1_GET_HI = ['7', ':', '0'] # macro
NVC56F_GP_ENTRY1_LEVEL = ['9', ':', '9'] # macro
NVC56F_GP_ENTRY1_LEVEL_MAIN = 0x00000000 # macro
NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE = 0x00000001 # macro
NVC56F_GP_ENTRY1_LENGTH = ['30', ':', '10'] # macro
NVC56F_GP_ENTRY1_SYNC = ['31', ':', '31'] # macro
NVC56F_GP_ENTRY1_SYNC_PROCEED = 0x00000000 # macro
NVC56F_GP_ENTRY1_SYNC_WAIT = 0x00000001 # macro
NVC56F_GP_ENTRY1_OPCODE = ['7', ':', '0'] # macro
NVC56F_GP_ENTRY1_OPCODE_NOP = 0x00000000 # macro
NVC56F_GP_ENTRY1_OPCODE_ILLEGAL = 0x00000001 # macro
NVC56F_GP_ENTRY1_OPCODE_GP_CRC = 0x00000002 # macro
NVC56F_GP_ENTRY1_OPCODE_PB_CRC = 0x00000003 # macro
NVC56F_DMA_METHOD_ADDRESS_OLD = ['12', ':', '2'] # macro
NVC56F_DMA_METHOD_ADDRESS = ['11', ':', '0'] # macro
NVC56F_DMA_SUBDEVICE_MASK = ['15', ':', '4'] # macro
NVC56F_DMA_METHOD_SUBCHANNEL = ['15', ':', '13'] # macro
NVC56F_DMA_TERT_OP = ['17', ':', '16'] # macro
NVC56F_DMA_TERT_OP_GRP0_INC_METHOD = (0x00000000) # macro
NVC56F_DMA_TERT_OP_GRP0_SET_SUB_DEV_MASK = (0x00000001) # macro
NVC56F_DMA_TERT_OP_GRP0_STORE_SUB_DEV_MASK = (0x00000002) # macro
NVC56F_DMA_TERT_OP_GRP0_USE_SUB_DEV_MASK = (0x00000003) # macro
NVC56F_DMA_TERT_OP_GRP2_NON_INC_METHOD = (0x00000000) # macro
NVC56F_DMA_METHOD_COUNT_OLD = ['28', ':', '18'] # macro
NVC56F_DMA_METHOD_COUNT = ['28', ':', '16'] # macro
NVC56F_DMA_IMMD_DATA = ['28', ':', '16'] # macro
NVC56F_DMA_SEC_OP = ['31', ':', '29'] # macro
NVC56F_DMA_SEC_OP_GRP0_USE_TERT = (0x00000000) # macro
NVC56F_DMA_SEC_OP_INC_METHOD = (0x00000001) # macro
NVC56F_DMA_SEC_OP_GRP2_USE_TERT = (0x00000002) # macro
NVC56F_DMA_SEC_OP_NON_INC_METHOD = (0x00000003) # macro
NVC56F_DMA_SEC_OP_IMMD_DATA_METHOD = (0x00000004) # macro
NVC56F_DMA_SEC_OP_ONE_INC = (0x00000005) # macro
NVC56F_DMA_SEC_OP_RESERVED6 = (0x00000006) # macro
NVC56F_DMA_SEC_OP_END_PB_SEGMENT = (0x00000007) # macro
NVC56F_DMA_INCR_ADDRESS = ['11', ':', '0'] # macro
NVC56F_DMA_INCR_SUBCHANNEL = ['15', ':', '13'] # macro
NVC56F_DMA_INCR_COUNT = ['28', ':', '16'] # macro
NVC56F_DMA_INCR_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_INCR_OPCODE_VALUE = (0x00000001) # macro
NVC56F_DMA_INCR_DATA = ['31', ':', '0'] # macro
NVC56F_DMA_NONINCR_ADDRESS = ['11', ':', '0'] # macro
NVC56F_DMA_NONINCR_SUBCHANNEL = ['15', ':', '13'] # macro
NVC56F_DMA_NONINCR_COUNT = ['28', ':', '16'] # macro
NVC56F_DMA_NONINCR_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_NONINCR_OPCODE_VALUE = (0x00000003) # macro
NVC56F_DMA_NONINCR_DATA = ['31', ':', '0'] # macro
NVC56F_DMA_ONEINCR_ADDRESS = ['11', ':', '0'] # macro
NVC56F_DMA_ONEINCR_SUBCHANNEL = ['15', ':', '13'] # macro
NVC56F_DMA_ONEINCR_COUNT = ['28', ':', '16'] # macro
NVC56F_DMA_ONEINCR_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_ONEINCR_OPCODE_VALUE = (0x00000005) # macro
NVC56F_DMA_ONEINCR_DATA = ['31', ':', '0'] # macro
NVC56F_DMA_NOP = (0x00000000) # macro
NVC56F_DMA_IMMD_ADDRESS = ['11', ':', '0'] # macro
NVC56F_DMA_IMMD_SUBCHANNEL = ['15', ':', '13'] # macro
NVC56F_DMA_IMMD_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_IMMD_OPCODE_VALUE = (0x00000004) # macro
NVC56F_DMA_SET_SUBDEVICE_MASK_VALUE = ['15', ':', '4'] # macro
NVC56F_DMA_SET_SUBDEVICE_MASK_OPCODE = ['31', ':', '16'] # macro
NVC56F_DMA_SET_SUBDEVICE_MASK_OPCODE_VALUE = (0x00000001) # macro
NVC56F_DMA_STORE_SUBDEVICE_MASK_VALUE = ['15', ':', '4'] # macro
NVC56F_DMA_STORE_SUBDEVICE_MASK_OPCODE = ['31', ':', '16'] # macro
NVC56F_DMA_STORE_SUBDEVICE_MASK_OPCODE_VALUE = (0x00000002) # macro
NVC56F_DMA_USE_SUBDEVICE_MASK_OPCODE = ['31', ':', '16'] # macro
NVC56F_DMA_USE_SUBDEVICE_MASK_OPCODE_VALUE = (0x00000003) # macro
NVC56F_DMA_ENDSEG_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_ENDSEG_OPCODE_VALUE = (0x00000007) # macro
NVC56F_DMA_ADDRESS = ['12', ':', '2'] # macro
NVC56F_DMA_SUBCH = ['15', ':', '13'] # macro
NVC56F_DMA_OPCODE3 = ['17', ':', '16'] # macro
NVC56F_DMA_OPCODE3_NONE = (0x00000000) # macro
NVC56F_DMA_COUNT = ['28', ':', '18'] # macro
NVC56F_DMA_OPCODE = ['31', ':', '29'] # macro
NVC56F_DMA_OPCODE_METHOD = (0x00000000) # macro
NVC56F_DMA_OPCODE_NONINC_METHOD = (0x00000002) # macro
NVC56F_DMA_DATA = ['31', ':', '0'] # macro
class struct_Nvc56fControl_struct(Structure):
    pass

struct_Nvc56fControl_struct._pack_ = 1 # source:False
struct_Nvc56fControl_struct._fields_ = [
    ('Ignored00', ctypes.c_uint32 * 16),
    ('Put', ctypes.c_uint32),
    ('Get', ctypes.c_uint32),
    ('Reference', ctypes.c_uint32),
    ('PutHi', ctypes.c_uint32),
    ('Ignored01', ctypes.c_uint32 * 2),
    ('TopLevelGet', ctypes.c_uint32),
    ('TopLevelGetHi', ctypes.c_uint32),
    ('GetHi', ctypes.c_uint32),
    ('Ignored02', ctypes.c_uint32 * 7),
    ('Ignored03', ctypes.c_uint32),
    ('Ignored04', ctypes.c_uint32 * 1),
    ('GPGet', ctypes.c_uint32),
    ('GPPut', ctypes.c_uint32),
    ('Ignored05', ctypes.c_uint32 * 92),
]

Nvc56fControl = struct_Nvc56fControl_struct
AmpereAControlGPFifo = struct_Nvc56fControl_struct
NV01_ROOT = (0x00000000) # macro
NV1_ROOT = (0x00000000) # macro
NV01_NULL_OBJECT = (0x00000000) # macro
NV1_NULL_OBJECT = (0x00000000) # macro
NV01_ROOT_NON_PRIV = (0x00000001) # macro
NV1_ROOT_NON_PRIV = (0x00000001) # macro
NV01_ROOT_CLIENT = (0x00000041) # macro
FABRIC_MANAGER_SESSION = (0x0000000f) # macro
NV0020_GPU_MANAGEMENT = (0x00000020) # macro
NV20_SUBDEVICE_0 = (0x00002080) # macro
NV2081_BINAPI = (0x00002081) # macro
NV2082_BINAPI_PRIVILEGED = (0x00002082) # macro
NV20_SUBDEVICE_DIAG = (0x0000208f) # macro
NV01_CONTEXT_DMA = (0x00000002) # macro
NV01_MEMORY_SYSTEM = (0x0000003e) # macro
NV1_MEMORY_SYSTEM = (0x0000003e) # macro
NV01_MEMORY_LOCAL_PRIVILEGED = (0x0000003f) # macro
NV1_MEMORY_LOCAL_PRIVILEGED = (0x0000003f) # macro
NV01_MEMORY_PRIVILEGED = (0x0000003f) # macro
NV1_MEMORY_PRIVILEGED = (0x0000003f) # macro
NV01_MEMORY_LOCAL_USER = (0x00000040) # macro
NV1_MEMORY_LOCAL_USER = (0x00000040) # macro
NV01_MEMORY_USER = (0x00000040) # macro
NV1_MEMORY_USER = (0x00000040) # macro
NV_MEMORY_EXTENDED_USER = (0x00000042) # macro
NV01_MEMORY_VIRTUAL = (0x00000070) # macro
NV01_MEMORY_SYSTEM_DYNAMIC = (0x00000070) # macro
NV1_MEMORY_SYSTEM_DYNAMIC = (0x00000070) # macro
NV_MEMORY_MAPPER = (0x000000fe) # macro
NV01_MEMORY_LOCAL_PHYSICAL = (0x000000c2) # macro
NV01_MEMORY_SYSTEM_OS_DESCRIPTOR = (0x00000071) # macro
NV01_MEMORY_DEVICELESS = (0x000090ce) # macro
NV01_MEMORY_FRAMEBUFFER_CONSOLE = (0x00000076) # macro
NV01_MEMORY_HW_RESOURCES = (0x000000b1) # macro
NV01_MEMORY_LIST_SYSTEM = (0x00000081) # macro
NV01_MEMORY_LIST_FBMEM = (0x00000082) # macro
NV01_MEMORY_LIST_OBJECT = (0x00000083) # macro
NV_IMEX_SESSION = (0x000000f1) # macro
NV01_MEMORY_FLA = (0x000000f3) # macro
NV_MEMORY_EXPORT = (0x000000e0) # macro
NV_CE_UTILS = (0x00000050) # macro
NV_MEMORY_FABRIC = (0x000000f8) # macro
NV_MEMORY_FABRIC_IMPORT_V2 = (0x000000f9) # macro
NV_MEMORY_FABRIC_IMPORTED_REF = (0x000000fb) # macro
FABRIC_VASPACE_A = (0x000000fc) # macro
NV_MEMORY_MULTICAST_FABRIC = (0x000000fd) # macro
IO_VASPACE_A = (0x000000f2) # macro
NV01_NULL = (0x00000030) # macro
NV1_NULL = (0x00000030) # macro
NV01_EVENT = (0x00000005) # macro
NV1_EVENT = (0x00000005) # macro
NV01_EVENT_KERNEL_CALLBACK = (0x00000078) # macro
NV1_EVENT_KERNEL_CALLBACK = (0x00000078) # macro
NV01_EVENT_OS_EVENT = (0x00000079) # macro
NV1_EVENT_OS_EVENT = (0x00000079) # macro
NV01_EVENT_WIN32_EVENT = (0x00000079) # macro
NV1_EVENT_WIN32_EVENT = (0x00000079) # macro
NV01_EVENT_KERNEL_CALLBACK_EX = (0x0000007e) # macro
NV1_EVENT_KERNEL_CALLBACK_EX = (0x0000007e) # macro
NV01_TIMER = (0x00000004) # macro
NV1_TIMER = (0x00000004) # macro
KERNEL_GRAPHICS_CONTEXT = (0x00000090) # macro
NV50_CHANNEL_GPFIFO = (0x0000506f) # macro
GF100_CHANNEL_GPFIFO = (0x0000906f) # macro
KEPLER_CHANNEL_GPFIFO_A = (0x0000a06f) # macro
UVM_CHANNEL_RETAINER = (0x0000c574) # macro
KEPLER_CHANNEL_GPFIFO_B = (0x0000a16f) # macro
MAXWELL_CHANNEL_GPFIFO_A = (0x0000b06f) # macro
PASCAL_CHANNEL_GPFIFO_A = (0x0000c06f) # macro
VOLTA_CHANNEL_GPFIFO_A = (0x0000c36f) # macro
TURING_CHANNEL_GPFIFO_A = (0x0000c46f) # macro
HOPPER_CHANNEL_GPFIFO_A = (0x0000c86f) # macro
NV04_SOFTWARE_TEST = (0x0000007d) # macro
NV4_SOFTWARE_TEST = (0x0000007d) # macro
NV30_GSYNC = (0x000030f1) # macro
VOLTA_USERMODE_A = (0x0000c361) # macro
TURING_USERMODE_A = (0x0000c461) # macro
AMPERE_USERMODE_A = (0x0000c561) # macro
HOPPER_USERMODE_A = (0x0000c661) # macro
NVC371_DISP_SF_USER = (0x0000c371) # macro
NVC372_DISPLAY_SW = (0x0000c372) # macro
NVC573_DISP_CAPABILITIES = (0x0000c573) # macro
NVC673_DISP_CAPABILITIES = (0x0000c673) # macro
NVC773_DISP_CAPABILITIES = (0x0000c773) # macro
NV04_DISPLAY_COMMON = (0x00000073) # macro
NV50_DEFERRED_API_CLASS = (0x00005080) # macro
MPS_COMPUTE = (0x0000900e) # macro
NVC570_DISPLAY = (0x0000c570) # macro
NVC57A_CURSOR_IMM_CHANNEL_PIO = (0x0000c57a) # macro
NVC57B_WINDOW_IMM_CHANNEL_DMA = (0x0000c57b) # macro
NVC57D_CORE_CHANNEL_DMA = (0x0000c57d) # macro
NVC57E_WINDOW_CHANNEL_DMA = (0x0000c57e) # macro
NVC670_DISPLAY = (0x0000c670) # macro
NVC671_DISP_SF_USER = (0x0000c671) # macro
NVC67A_CURSOR_IMM_CHANNEL_PIO = (0x0000c67a) # macro
NVC67B_WINDOW_IMM_CHANNEL_DMA = (0x0000c67b) # macro
NVC67D_CORE_CHANNEL_DMA = (0x0000c67d) # macro
NVC67E_WINDOW_CHANNEL_DMA = (0x0000c67e) # macro
NVC77F_ANY_CHANNEL_DMA = (0x0000c77f) # macro
NVC770_DISPLAY = (0x0000c770) # macro
NVC771_DISP_SF_USER = (0x0000c771) # macro
NVC77D_CORE_CHANNEL_DMA = (0x0000c77d) # macro
NV9010_VBLANK_CALLBACK = (0x00009010) # macro
GF100_PROFILER = (0x000090cc) # macro
MAXWELL_PROFILER = (0x0000b0cc) # macro
MAXWELL_PROFILER_DEVICE = (0x0000b2cc) # macro
GF100_SUBDEVICE_MASTER = (0x000090e6) # macro
GF100_SUBDEVICE_INFOROM = (0x000090e7) # macro
GF100_ZBC_CLEAR = (0x00009096) # macro
GF100_DISP_SW = (0x00009072) # macro
GF100_TIMED_SEMAPHORE_SW = (0x00009074) # macro
G84_PERFBUFFER = (0x0000844c) # macro
NV50_MEMORY_VIRTUAL = (0x000050a0) # macro
NV50_P2P = (0x0000503b) # macro
NV50_THIRD_PARTY_P2P = (0x0000503c) # macro
FERMI_TWOD_A = (0x0000902d) # macro
FERMI_VASPACE_A = (0x000090f1) # macro
HOPPER_SEC2_WORK_LAUNCH_A = (0x0000cba2) # macro
GF100_HDACODEC = (0x000090ec) # macro
NVB8B0_VIDEO_DECODER = (0x0000b8b0) # macro
NVC4B0_VIDEO_DECODER = (0x0000c4b0) # macro
NVC6B0_VIDEO_DECODER = (0x0000c6b0) # macro
NVC7B0_VIDEO_DECODER = (0x0000c7b0) # macro
NVC9B0_VIDEO_DECODER = (0x0000c9b0) # macro
NVC4B7_VIDEO_ENCODER = (0x0000c4b7) # macro
NVB4B7_VIDEO_ENCODER = (0x0000b4b7) # macro
NVC7B7_VIDEO_ENCODER = (0x0000c7b7) # macro
NVC9B7_VIDEO_ENCODER = (0x0000c9b7) # macro
NVB8D1_VIDEO_NVJPG = (0x0000b8d1) # macro
NVC4D1_VIDEO_NVJPG = (0x0000c4d1) # macro
NVC9D1_VIDEO_NVJPG = (0x0000c9d1) # macro
NVB8FA_VIDEO_OFA = (0x0000b8fa) # macro
NVC6FA_VIDEO_OFA = (0x0000c6fa) # macro
NVC7FA_VIDEO_OFA = (0x0000c7fa) # macro
NVC9FA_VIDEO_OFA = (0x0000c9fa) # macro
KEPLER_INLINE_TO_MEMORY_B = (0x0000a140) # macro
FERMI_CONTEXT_SHARE_A = (0x00009067) # macro
KEPLER_CHANNEL_GROUP_A = (0x0000a06c) # macro
PASCAL_DMA_COPY_A = (0x0000c0b5) # macro
TURING_DMA_COPY_A = (0x0000c5b5) # macro
AMPERE_DMA_COPY_A = (0x0000c6b5) # macro
AMPERE_DMA_COPY_B = (0x0000c7b5) # macro
HOPPER_DMA_COPY_A = (0x0000c8b5) # macro
MAXWELL_DMA_COPY_A = (0x0000b0b5) # macro
ACCESS_COUNTER_NOTIFY_BUFFER = (0x0000c365) # macro
MMU_FAULT_BUFFER = (0x0000c369) # macro
MMU_VIDMEM_ACCESS_BIT_BUFFER = (0x0000c763) # macro
TURING_A = (0x0000c597) # macro
TURING_COMPUTE_A = (0x0000c5c0) # macro
AMPERE_A = (0x0000c697) # macro
AMPERE_COMPUTE_A = (0x0000c6c0) # macro
AMPERE_B = (0x0000c797) # macro
AMPERE_COMPUTE_B = (0x0000c7c0) # macro
ADA_A = (0x0000c997) # macro
ADA_COMPUTE_A = (0x0000c9c0) # macro
AMPERE_SMC_PARTITION_REF = (0x0000c637) # macro
AMPERE_SMC_EXEC_PARTITION_REF = (0x0000c638) # macro
AMPERE_SMC_CONFIG_SESSION = (0x0000c639) # macro
NV0092_RG_LINE_CALLBACK = (0x00000092) # macro
AMPERE_SMC_MONITOR_SESSION = (0x0000c640) # macro
HOPPER_A = (0x0000cb97) # macro
HOPPER_COMPUTE_A = (0x0000cbc0) # macro
NV40_DEBUG_BUFFER = (0x000000db) # macro
RM_USER_SHARED_DATA = (0x000000de) # macro
GT200_DEBUGGER = (0x000083de) # macro
NV40_I2C = (0x0000402c) # macro
KEPLER_DEVICE_VGPU = (0x0000a080) # macro
NVA081_VGPU_CONFIG = (0x0000a081) # macro
NVA084_KERNEL_HOST_VGPU_DEVICE = (0x0000a084) # macro
NV0060_SYNC_GPU_BOOST = (0x00000060) # macro
GP100_UVM_SW = (0x0000c076) # macro
NVENC_SW_SESSION = (0x0000a0bc) # macro
NV_EVENT_BUFFER = (0x000090cd) # macro
NVFBC_SW_SESSION = (0x0000a0bd) # macro
NV_CONFIDENTIAL_COMPUTE = (0x0000cb33) # macro
NV_COUNTER_COLLECTION_UNIT = (0x0000cbca) # macro
NV_SEMAPHORE_SURFACE = (0x000000da) # macro
__all__ = \
    ['ACCESS_COUNTER_NOTIFY_BUFFER', 'ADA_A', 'ADA_COMPUTE_A',
    'AMPERE_A', 'AMPERE_B', 'AMPERE_CHANNEL_GPFIFO_A',
    'AMPERE_COMPUTE_A', 'AMPERE_COMPUTE_B', 'AMPERE_DMA_COPY_A',
    'AMPERE_DMA_COPY_B', 'AMPERE_SMC_CONFIG_SESSION',
    'AMPERE_SMC_EXEC_PARTITION_REF', 'AMPERE_SMC_MONITOR_SESSION',
    'AMPERE_SMC_PARTITION_REF', 'AMPERE_USERMODE_A',
    'AmpereAControlGPFifo', 'FABRIC_MANAGER_SESSION',
    'FABRIC_VASPACE_A', 'FERMI_CONTEXT_SHARE_A', 'FERMI_TWOD_A',
    'FERMI_VASPACE_A', 'G84_PERFBUFFER', 'GF100_CHANNEL_GPFIFO',
    'GF100_DISP_SW', 'GF100_HDACODEC', 'GF100_PROFILER',
    'GF100_SUBDEVICE_INFOROM', 'GF100_SUBDEVICE_MASTER',
    'GF100_TIMED_SEMAPHORE_SW', 'GF100_ZBC_CLEAR', 'GP100_UVM_SW',
    'GT200_DEBUGGER', 'HOPPER_A', 'HOPPER_CHANNEL_GPFIFO_A',
    'HOPPER_COMPUTE_A', 'HOPPER_DMA_COPY_A',
    'HOPPER_SEC2_WORK_LAUNCH_A', 'HOPPER_USERMODE_A', 'IO_VASPACE_A',
    'KEPLER_CHANNEL_GPFIFO_A', 'KEPLER_CHANNEL_GPFIFO_B',
    'KEPLER_CHANNEL_GROUP_A', 'KEPLER_DEVICE_VGPU',
    'KEPLER_INLINE_TO_MEMORY_B', 'KERNEL_GRAPHICS_CONTEXT',
    'MAXWELL_CHANNEL_GPFIFO_A', 'MAXWELL_DMA_COPY_A',
    'MAXWELL_PROFILER', 'MAXWELL_PROFILER_DEVICE', 'MMU_FAULT_BUFFER',
    'MMU_VIDMEM_ACCESS_BIT_BUFFER', 'MPS_COMPUTE',
    'NV0020_GPU_MANAGEMENT', 'NV0060_SYNC_GPU_BOOST',
    'NV0080_ALLOC_PARAMETERS', 'NV0080_ALLOC_PARAMETERS_MESSAGE_ID',
    'NV0092_RG_LINE_CALLBACK', 'NV01_CONTEXT_DMA', 'NV01_DEVICE_0',
    'NV01_EVENT', 'NV01_EVENT_KERNEL_CALLBACK',
    'NV01_EVENT_KERNEL_CALLBACK_EX', 'NV01_EVENT_OS_EVENT',
    'NV01_EVENT_WIN32_EVENT', 'NV01_MEMORY_DEVICELESS',
    'NV01_MEMORY_FLA', 'NV01_MEMORY_FRAMEBUFFER_CONSOLE',
    'NV01_MEMORY_HW_RESOURCES', 'NV01_MEMORY_LIST_FBMEM',
    'NV01_MEMORY_LIST_OBJECT', 'NV01_MEMORY_LIST_SYSTEM',
    'NV01_MEMORY_LOCAL_PHYSICAL', 'NV01_MEMORY_LOCAL_PRIVILEGED',
    'NV01_MEMORY_LOCAL_USER', 'NV01_MEMORY_PRIVILEGED',
    'NV01_MEMORY_SYSTEM', 'NV01_MEMORY_SYSTEM_DYNAMIC',
    'NV01_MEMORY_SYSTEM_OS_DESCRIPTOR', 'NV01_MEMORY_USER',
    'NV01_MEMORY_VIRTUAL', 'NV01_NULL', 'NV01_NULL_OBJECT',
    'NV01_ROOT', 'NV01_ROOT_CLIENT', 'NV01_ROOT_NON_PRIV',
    'NV01_TIMER', 'NV04_DISPLAY_COMMON', 'NV04_SOFTWARE_TEST',
    'NV1_EVENT', 'NV1_EVENT_KERNEL_CALLBACK',
    'NV1_EVENT_KERNEL_CALLBACK_EX', 'NV1_EVENT_OS_EVENT',
    'NV1_EVENT_WIN32_EVENT', 'NV1_MEMORY_LOCAL_PRIVILEGED',
    'NV1_MEMORY_LOCAL_USER', 'NV1_MEMORY_PRIVILEGED',
    'NV1_MEMORY_SYSTEM', 'NV1_MEMORY_SYSTEM_DYNAMIC',
    'NV1_MEMORY_USER', 'NV1_NULL', 'NV1_NULL_OBJECT', 'NV1_ROOT',
    'NV1_ROOT_NON_PRIV', 'NV1_TIMER', 'NV2080_CLIENT_TYPE_ALLCLIENTS',
    'NV2080_CLIENT_TYPE_COLOR', 'NV2080_CLIENT_TYPE_DA',
    'NV2080_CLIENT_TYPE_DEPTH', 'NV2080_CLIENT_TYPE_FE',
    'NV2080_CLIENT_TYPE_MSPDEC', 'NV2080_CLIENT_TYPE_MSPPP',
    'NV2080_CLIENT_TYPE_MSVLD', 'NV2080_CLIENT_TYPE_SCC',
    'NV2080_CLIENT_TYPE_TEX', 'NV2080_CLIENT_TYPE_VIC',
    'NV2080_CLIENT_TYPE_WID',
    'NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC',
    'NV2080_ENGINE_TYPE_ALLENGINES', 'NV2080_ENGINE_TYPE_BSP',
    'NV2080_ENGINE_TYPE_CIPHER', 'NV2080_ENGINE_TYPE_COPY0',
    'NV2080_ENGINE_TYPE_COPY1', 'NV2080_ENGINE_TYPE_COPY2',
    'NV2080_ENGINE_TYPE_COPY3', 'NV2080_ENGINE_TYPE_COPY4',
    'NV2080_ENGINE_TYPE_COPY5', 'NV2080_ENGINE_TYPE_COPY6',
    'NV2080_ENGINE_TYPE_COPY7', 'NV2080_ENGINE_TYPE_COPY8',
    'NV2080_ENGINE_TYPE_COPY9', 'NV2080_ENGINE_TYPE_COPY_SIZE',
    'NV2080_ENGINE_TYPE_DPU', 'NV2080_ENGINE_TYPE_FBFLCN',
    'NV2080_ENGINE_TYPE_GR0', 'NV2080_ENGINE_TYPE_GR1',
    'NV2080_ENGINE_TYPE_GR2', 'NV2080_ENGINE_TYPE_GR3',
    'NV2080_ENGINE_TYPE_GR4', 'NV2080_ENGINE_TYPE_GR5',
    'NV2080_ENGINE_TYPE_GR6', 'NV2080_ENGINE_TYPE_GR7',
    'NV2080_ENGINE_TYPE_GRAPHICS', 'NV2080_ENGINE_TYPE_GR_SIZE',
    'NV2080_ENGINE_TYPE_HOST', 'NV2080_ENGINE_TYPE_LAST',
    'NV2080_ENGINE_TYPE_ME', 'NV2080_ENGINE_TYPE_MP',
    'NV2080_ENGINE_TYPE_MPEG', 'NV2080_ENGINE_TYPE_MSENC',
    'NV2080_ENGINE_TYPE_NULL', 'NV2080_ENGINE_TYPE_NVDEC0',
    'NV2080_ENGINE_TYPE_NVDEC1', 'NV2080_ENGINE_TYPE_NVDEC2',
    'NV2080_ENGINE_TYPE_NVDEC3', 'NV2080_ENGINE_TYPE_NVDEC4',
    'NV2080_ENGINE_TYPE_NVDEC5', 'NV2080_ENGINE_TYPE_NVDEC6',
    'NV2080_ENGINE_TYPE_NVDEC7', 'NV2080_ENGINE_TYPE_NVDEC_SIZE',
    'NV2080_ENGINE_TYPE_NVENC0', 'NV2080_ENGINE_TYPE_NVENC1',
    'NV2080_ENGINE_TYPE_NVENC2', 'NV2080_ENGINE_TYPE_NVENC_SIZE',
    'NV2080_ENGINE_TYPE_NVJPEG0', 'NV2080_ENGINE_TYPE_NVJPEG1',
    'NV2080_ENGINE_TYPE_NVJPEG2', 'NV2080_ENGINE_TYPE_NVJPEG3',
    'NV2080_ENGINE_TYPE_NVJPEG4', 'NV2080_ENGINE_TYPE_NVJPEG5',
    'NV2080_ENGINE_TYPE_NVJPEG6', 'NV2080_ENGINE_TYPE_NVJPEG7',
    'NV2080_ENGINE_TYPE_NVJPEG_SIZE', 'NV2080_ENGINE_TYPE_NVJPG',
    'NV2080_ENGINE_TYPE_OFA', 'NV2080_ENGINE_TYPE_OFA0',
    'NV2080_ENGINE_TYPE_OFA_SIZE', 'NV2080_ENGINE_TYPE_PMU',
    'NV2080_ENGINE_TYPE_PPP', 'NV2080_ENGINE_TYPE_RESERVED34',
    'NV2080_ENGINE_TYPE_RESERVED35', 'NV2080_ENGINE_TYPE_RESERVED36',
    'NV2080_ENGINE_TYPE_RESERVED37', 'NV2080_ENGINE_TYPE_RESERVED38',
    'NV2080_ENGINE_TYPE_RESERVED39', 'NV2080_ENGINE_TYPE_RESERVED3a',
    'NV2080_ENGINE_TYPE_RESERVED3b', 'NV2080_ENGINE_TYPE_RESERVED3c',
    'NV2080_ENGINE_TYPE_RESERVED3d', 'NV2080_ENGINE_TYPE_RESERVED3e',
    'NV2080_ENGINE_TYPE_RESERVED3f', 'NV2080_ENGINE_TYPE_SEC2',
    'NV2080_ENGINE_TYPE_SW', 'NV2080_ENGINE_TYPE_TSEC',
    'NV2080_ENGINE_TYPE_VIC', 'NV2080_ENGINE_TYPE_VP',
    'NV2080_GC5_ENTRY_ABORTED', 'NV2080_GC5_EXIT_COMPLETE',
    'NV2080_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT',
    'NV2080_NOTIFIERS_ACPI_NOTIFY',
    'NV2080_NOTIFIERS_AUDIO_HDCP_REQUEST',
    'NV2080_NOTIFIERS_AUX_POWER_EVENT',
    'NV2080_NOTIFIERS_AUX_POWER_STATE_CHANGE', 'NV2080_NOTIFIERS_CE0',
    'NV2080_NOTIFIERS_CE1', 'NV2080_NOTIFIERS_CE2',
    'NV2080_NOTIFIERS_CE3', 'NV2080_NOTIFIERS_CE4',
    'NV2080_NOTIFIERS_CE5', 'NV2080_NOTIFIERS_CE6',
    'NV2080_NOTIFIERS_CE7', 'NV2080_NOTIFIERS_CE8',
    'NV2080_NOTIFIERS_CE9', 'NV2080_NOTIFIERS_CLOCKS_CHANGE',
    'NV2080_NOTIFIERS_COOLER_DIAG_ZONE',
    'NV2080_NOTIFIERS_CTXSW_TIMEOUT', 'NV2080_NOTIFIERS_DP_IRQ',
    'NV2080_NOTIFIERS_DSTATE_HDA', 'NV2080_NOTIFIERS_DSTATE_XUSB_PPC',
    'NV2080_NOTIFIERS_ECC_DBE', 'NV2080_NOTIFIERS_ECC_SBE',
    'NV2080_NOTIFIERS_EVENTBUFFER',
    'NV2080_NOTIFIERS_FECS_CTX_SWITCH',
    'NV2080_NOTIFIERS_FIFO_EVENT_MTHD',
    'NV2080_NOTIFIERS_FULL_SCREEN_CHANGE',
    'NV2080_NOTIFIERS_GC5_GPU_READY',
    'NV2080_NOTIFIERS_GC6_REFCOUNT_DEC',
    'NV2080_NOTIFIERS_GC6_REFCOUNT_INC',
    'NV2080_NOTIFIERS_GPIO_0_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_0_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_10_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_10_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_11_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_11_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_12_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_12_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_13_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_13_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_14_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_14_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_15_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_15_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_16_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_16_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_17_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_17_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_18_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_18_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_19_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_19_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_1_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_1_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_20_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_20_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_21_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_21_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_22_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_22_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_23_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_23_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_24_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_24_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_25_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_25_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_26_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_26_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_27_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_27_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_28_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_28_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_29_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_29_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_2_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_2_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_30_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_30_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_31_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_31_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_3_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_3_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_4_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_4_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_5_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_5_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_6_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_6_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_7_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_7_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_8_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_8_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_9_FALLING_INTERRUPT',
    'NV2080_NOTIFIERS_GPIO_9_RISING_INTERRUPT',
    'NV2080_NOTIFIERS_GR0', 'NV2080_NOTIFIERS_GR1',
    'NV2080_NOTIFIERS_GR2', 'NV2080_NOTIFIERS_GR3',
    'NV2080_NOTIFIERS_GR4', 'NV2080_NOTIFIERS_GR5',
    'NV2080_NOTIFIERS_GR6', 'NV2080_NOTIFIERS_GR7',
    'NV2080_NOTIFIERS_GRAPHICS', 'NV2080_NOTIFIERS_GR_DEBUG_INTR',
    'NV2080_NOTIFIERS_GSP_PERF_TRACE',
    'NV2080_NOTIFIERS_HDCP_STATUS_CHANGE',
    'NV2080_NOTIFIERS_HDMI_FRL_RETRAINING_REQUEST',
    'NV2080_NOTIFIERS_HOTPLUG',
    'NV2080_NOTIFIERS_HOTPLUG_PROCESSING_COMPLETE',
    'NV2080_NOTIFIERS_INBAND_RESPONSE',
    'NV2080_NOTIFIERS_INFOROM_ECC_OBJECT_UPDATED',
    'NV2080_NOTIFIERS_INFOROM_PBL_OBJECT_UPDATED',
    'NV2080_NOTIFIERS_INFOROM_RRL_OBJECT_UPDATED',
    'NV2080_NOTIFIERS_LPWR_DIFR_PREFETCH_REQUEST',
    'NV2080_NOTIFIERS_MAXCOUNT', 'NV2080_NOTIFIERS_MSENC',
    'NV2080_NOTIFIERS_NVDEC0', 'NV2080_NOTIFIERS_NVDEC1',
    'NV2080_NOTIFIERS_NVDEC2', 'NV2080_NOTIFIERS_NVDEC3',
    'NV2080_NOTIFIERS_NVDEC4', 'NV2080_NOTIFIERS_NVDEC5',
    'NV2080_NOTIFIERS_NVDEC6', 'NV2080_NOTIFIERS_NVDEC7',
    'NV2080_NOTIFIERS_NVENC0', 'NV2080_NOTIFIERS_NVENC1',
    'NV2080_NOTIFIERS_NVENC2', 'NV2080_NOTIFIERS_NVJPEG0',
    'NV2080_NOTIFIERS_NVJPEG1', 'NV2080_NOTIFIERS_NVJPEG2',
    'NV2080_NOTIFIERS_NVJPEG3', 'NV2080_NOTIFIERS_NVJPEG4',
    'NV2080_NOTIFIERS_NVJPEG5', 'NV2080_NOTIFIERS_NVJPEG6',
    'NV2080_NOTIFIERS_NVJPEG7', 'NV2080_NOTIFIERS_NVJPG',
    'NV2080_NOTIFIERS_NVLINK_ERROR_FATAL',
    'NV2080_NOTIFIERS_NVLINK_ERROR_RECOVERY_REQUIRED',
    'NV2080_NOTIFIERS_NVLINK_INFO_LINK_DOWN',
    'NV2080_NOTIFIERS_NVLINK_INFO_LINK_UP',
    'NV2080_NOTIFIERS_NVPCF_EVENTS',
    'NV2080_NOTIFIERS_NVTELEMETRY_REPORT_EVENT',
    'NV2080_NOTIFIERS_OFA', 'NV2080_NOTIFIERS_OFA0',
    'NV2080_NOTIFIERS_PDEC', 'NV2080_NOTIFIERS_PHYSICAL_PAGE_FAULT',
    'NV2080_NOTIFIERS_PLATFORM_POWER_MODE_CHANGE',
    'NV2080_NOTIFIERS_PMU_COMMAND', 'NV2080_NOTIFIERS_PMU_EVENT',
    'NV2080_NOTIFIERS_POISON_ERROR_FATAL',
    'NV2080_NOTIFIERS_POISON_ERROR_NON_FATAL',
    'NV2080_NOTIFIERS_POSSIBLE_ERROR',
    'NV2080_NOTIFIERS_POWER_CONNECTOR',
    'NV2080_NOTIFIERS_POWER_EVENT', 'NV2080_NOTIFIERS_PPP',
    'NV2080_NOTIFIERS_PRIV_REG_ACCESS_FAULT',
    'NV2080_NOTIFIERS_PRIV_RING_HANG',
    'NV2080_NOTIFIERS_PSTATE_CHANGE', 'NV2080_NOTIFIERS_RC_ERROR',
    'NV2080_NOTIFIERS_RESERVED122', 'NV2080_NOTIFIERS_RESERVED166',
    'NV2080_NOTIFIERS_RESERVED167', 'NV2080_NOTIFIERS_RESERVED168',
    'NV2080_NOTIFIERS_RESERVED169', 'NV2080_NOTIFIERS_RESERVED170',
    'NV2080_NOTIFIERS_RESERVED171', 'NV2080_NOTIFIERS_RESERVED172',
    'NV2080_NOTIFIERS_RESERVED173', 'NV2080_NOTIFIERS_RESERVED174',
    'NV2080_NOTIFIERS_RESERVED175', 'NV2080_NOTIFIERS_RESERVED180',
    'NV2080_NOTIFIERS_RESERVED_183', 'NV2080_NOTIFIERS_RESERVED_186',
    'NV2080_NOTIFIERS_RUNLIST_ACQUIRE',
    'NV2080_NOTIFIERS_RUNLIST_ACQUIRE_AND_ENG_IDLE',
    'NV2080_NOTIFIERS_RUNLIST_AND_ENG_IDLE',
    'NV2080_NOTIFIERS_RUNLIST_IDLE',
    'NV2080_NOTIFIERS_RUNLIST_PREEMPT_COMPLETE',
    'NV2080_NOTIFIERS_SEC2', 'NV2080_NOTIFIERS_SEC_FAULT_ERROR',
    'NV2080_NOTIFIERS_SMC_CONFIG_UPDATE',
    'NV2080_NOTIFIERS_STEREO_EMITTER_DETECTION',
    'NV2080_NOTIFIERS_SW', 'NV2080_NOTIFIERS_THERMAL_DIAG_ZONE',
    'NV2080_NOTIFIERS_THERMAL_HW', 'NV2080_NOTIFIERS_THERMAL_SW',
    'NV2080_NOTIFIERS_TIMER', 'NV2080_NOTIFIERS_TSG_PREEMPT_COMPLETE',
    'NV2080_NOTIFIERS_UCODE_RESET', 'NV2080_NOTIFIERS_UNUSED_0',
    'NV2080_NOTIFIERS_VLD', 'NV2080_NOTIFIERS_VRR_SET_TIMEOUT',
    'NV2080_NOTIFIERS_WORKLOAD_MODULATION_CHANGE',
    'NV2080_NOTIFIERS_XUSB_PPC_CONNECTED',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_ACPI_NOTIFICATION',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_COMPLETION',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_INDEX',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_MASK',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_INFO_REASON',
    'NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS',
    'NV2080_SUBDEVICE_NOTIFICATION_STATUS_BAD_ARGUMENT',
    'NV2080_SUBDEVICE_NOTIFICATION_STATUS_DONE_SUCCESS',
    'NV2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_INVALID_STATE',
    'NV2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_STATE_IN_USE',
    'NV2080_SUBDEVICE_NOTIFICATION_STATUS_IN_PROGRESS',
    'NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC', 'NV2081_BINAPI',
    'NV2082_BINAPI_PRIVILEGED', 'NV20_SUBDEVICE_0',
    'NV20_SUBDEVICE_DIAG', 'NV30_GSYNC', 'NV40_DEBUG_BUFFER',
    'NV40_I2C', 'NV4_SOFTWARE_TEST', 'NV50_CHANNEL_GPFIFO',
    'NV50_DEFERRED_API_CLASS', 'NV50_MEMORY_VIRTUAL', 'NV50_P2P',
    'NV50_THIRD_PARTY_P2P', 'NV9010_VBLANK_CALLBACK',
    'NVA081_VGPU_CONFIG', 'NVA084_KERNEL_HOST_VGPU_DEVICE',
    'NVB4B7_VIDEO_ENCODER', 'NVB8B0_VIDEO_DECODER',
    'NVB8D1_VIDEO_NVJPG', 'NVB8FA_VIDEO_OFA', 'NVC371_DISP_SF_USER',
    'NVC372_DISPLAY_SW', 'NVC4B0_VIDEO_DECODER',
    'NVC4B7_VIDEO_ENCODER', 'NVC4D1_VIDEO_NVJPG',
    'NVC56F_CLEAR_FAULTED', 'NVC56F_CLEAR_FAULTED_HANDLE',
    'NVC56F_CLEAR_FAULTED_TYPE',
    'NVC56F_CLEAR_FAULTED_TYPE_ENG_FAULTED',
    'NVC56F_CLEAR_FAULTED_TYPE_PBDMA_FAULTED', 'NVC56F_DMA_ADDRESS',
    'NVC56F_DMA_COUNT', 'NVC56F_DMA_DATA', 'NVC56F_DMA_ENDSEG_OPCODE',
    'NVC56F_DMA_ENDSEG_OPCODE_VALUE', 'NVC56F_DMA_IMMD_ADDRESS',
    'NVC56F_DMA_IMMD_DATA', 'NVC56F_DMA_IMMD_OPCODE',
    'NVC56F_DMA_IMMD_OPCODE_VALUE', 'NVC56F_DMA_IMMD_SUBCHANNEL',
    'NVC56F_DMA_INCR_ADDRESS', 'NVC56F_DMA_INCR_COUNT',
    'NVC56F_DMA_INCR_DATA', 'NVC56F_DMA_INCR_OPCODE',
    'NVC56F_DMA_INCR_OPCODE_VALUE', 'NVC56F_DMA_INCR_SUBCHANNEL',
    'NVC56F_DMA_METHOD_ADDRESS', 'NVC56F_DMA_METHOD_ADDRESS_OLD',
    'NVC56F_DMA_METHOD_COUNT', 'NVC56F_DMA_METHOD_COUNT_OLD',
    'NVC56F_DMA_METHOD_SUBCHANNEL', 'NVC56F_DMA_NONINCR_ADDRESS',
    'NVC56F_DMA_NONINCR_COUNT', 'NVC56F_DMA_NONINCR_DATA',
    'NVC56F_DMA_NONINCR_OPCODE', 'NVC56F_DMA_NONINCR_OPCODE_VALUE',
    'NVC56F_DMA_NONINCR_SUBCHANNEL', 'NVC56F_DMA_NOP',
    'NVC56F_DMA_ONEINCR_ADDRESS', 'NVC56F_DMA_ONEINCR_COUNT',
    'NVC56F_DMA_ONEINCR_DATA', 'NVC56F_DMA_ONEINCR_OPCODE',
    'NVC56F_DMA_ONEINCR_OPCODE_VALUE',
    'NVC56F_DMA_ONEINCR_SUBCHANNEL', 'NVC56F_DMA_OPCODE',
    'NVC56F_DMA_OPCODE3', 'NVC56F_DMA_OPCODE3_NONE',
    'NVC56F_DMA_OPCODE_METHOD', 'NVC56F_DMA_OPCODE_NONINC_METHOD',
    'NVC56F_DMA_SEC_OP', 'NVC56F_DMA_SEC_OP_END_PB_SEGMENT',
    'NVC56F_DMA_SEC_OP_GRP0_USE_TERT',
    'NVC56F_DMA_SEC_OP_GRP2_USE_TERT',
    'NVC56F_DMA_SEC_OP_IMMD_DATA_METHOD',
    'NVC56F_DMA_SEC_OP_INC_METHOD',
    'NVC56F_DMA_SEC_OP_NON_INC_METHOD', 'NVC56F_DMA_SEC_OP_ONE_INC',
    'NVC56F_DMA_SEC_OP_RESERVED6',
    'NVC56F_DMA_SET_SUBDEVICE_MASK_OPCODE',
    'NVC56F_DMA_SET_SUBDEVICE_MASK_OPCODE_VALUE',
    'NVC56F_DMA_SET_SUBDEVICE_MASK_VALUE',
    'NVC56F_DMA_STORE_SUBDEVICE_MASK_OPCODE',
    'NVC56F_DMA_STORE_SUBDEVICE_MASK_OPCODE_VALUE',
    'NVC56F_DMA_STORE_SUBDEVICE_MASK_VALUE', 'NVC56F_DMA_SUBCH',
    'NVC56F_DMA_SUBDEVICE_MASK', 'NVC56F_DMA_TERT_OP',
    'NVC56F_DMA_TERT_OP_GRP0_INC_METHOD',
    'NVC56F_DMA_TERT_OP_GRP0_SET_SUB_DEV_MASK',
    'NVC56F_DMA_TERT_OP_GRP0_STORE_SUB_DEV_MASK',
    'NVC56F_DMA_TERT_OP_GRP0_USE_SUB_DEV_MASK',
    'NVC56F_DMA_TERT_OP_GRP2_NON_INC_METHOD',
    'NVC56F_DMA_USE_SUBDEVICE_MASK_OPCODE',
    'NVC56F_DMA_USE_SUBDEVICE_MASK_OPCODE_VALUE', 'NVC56F_FB_FLUSH',
    'NVC56F_FB_FLUSH_HANDLE', 'NVC56F_GP_ENTRY0_FETCH',
    'NVC56F_GP_ENTRY0_FETCH_CONDITIONAL',
    'NVC56F_GP_ENTRY0_FETCH_UNCONDITIONAL', 'NVC56F_GP_ENTRY0_GET',
    'NVC56F_GP_ENTRY0_OPERAND', 'NVC56F_GP_ENTRY1_GET_HI',
    'NVC56F_GP_ENTRY1_LENGTH', 'NVC56F_GP_ENTRY1_LEVEL',
    'NVC56F_GP_ENTRY1_LEVEL_MAIN',
    'NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE', 'NVC56F_GP_ENTRY1_OPCODE',
    'NVC56F_GP_ENTRY1_OPCODE_GP_CRC',
    'NVC56F_GP_ENTRY1_OPCODE_ILLEGAL', 'NVC56F_GP_ENTRY1_OPCODE_NOP',
    'NVC56F_GP_ENTRY1_OPCODE_PB_CRC', 'NVC56F_GP_ENTRY1_SYNC',
    'NVC56F_GP_ENTRY1_SYNC_PROCEED', 'NVC56F_GP_ENTRY1_SYNC_WAIT',
    'NVC56F_GP_ENTRY__SIZE', 'NVC56F_ILLEGAL',
    'NVC56F_ILLEGAL_HANDLE', 'NVC56F_MEM_OP_A',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_MMU_ENGINE_ID',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_TARGET_CLIENT_UNIT_ID',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_CANCEL_TARGET_GPC_ID',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVALIDATION_SIZE',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_ALL_TLBS',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_LINK_TLBS',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_NON_LINK_TLBS',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_RSVRVD',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_DIS',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_EN',
    'NVC56F_MEM_OP_A_TLB_INVALIDATE_TARGET_ADDR_LO',
    'NVC56F_MEM_OP_B',
    'NVC56F_MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES',
    'NVC56F_MEM_OP_B_TLB_INVALIDATE_TARGET_ADDR_HI',
    'NVC56F_MEM_OP_C',
    'NVC56F_MEM_OP_C_ACCESS_COUNTER_CLR_TARGETED_NOTIFY_TAG',
    'NVC56F_MEM_OP_C_MEMBAR_TYPE',
    'NVC56F_MEM_OP_C_MEMBAR_TYPE_MEMBAR',
    'NVC56F_MEM_OP_C_MEMBAR_TYPE_SYS_MEMBAR',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ALL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_ALL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_STRONG',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_ATOMIC_WEAK',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_READ',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_RSVRVD',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_WRITE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACCESS_TYPE_VIRT_WRITE_AND_ATOMIC',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_GLOBALLY',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_INTRANODE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_NONE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC_DISABLE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_GPC_ENABLE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_ALL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_PTE_ONLY',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE0',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE1',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE2',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE3',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE4',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE5',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ADDR_LO',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ALL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_SYS_MEM_COHERENT',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_SYS_MEM_NONCOHERENT',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_VID_MEM',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_PDB_ONE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_GLOBAL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_TARGETED',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_VA_GLOBAL',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_NONE',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_START',
    'NVC56F_MEM_OP_C_TLB_INVALIDATE_REPLAY_START_ACK_ALL',
    'NVC56F_MEM_OP_D',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_BANK',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE_MIMC',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE_MOMC',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_ALL',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MIMC',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MOMC',
    'NVC56F_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_TARGETED',
    'NVC56F_MEM_OP_D_OPERATION',
    'NVC56F_MEM_OP_D_OPERATION_ACCESS_COUNTER_CLR',
    'NVC56F_MEM_OP_D_OPERATION_L2_CLEAN_COMPTAGS',
    'NVC56F_MEM_OP_D_OPERATION_L2_FLUSH_DIRTY',
    'NVC56F_MEM_OP_D_OPERATION_L2_PEERMEM_INVALIDATE',
    'NVC56F_MEM_OP_D_OPERATION_L2_SYSMEM_INVALIDATE',
    'NVC56F_MEM_OP_D_OPERATION_L2_WAIT_FOR_SYS_PENDING_READS',
    'NVC56F_MEM_OP_D_OPERATION_MEMBAR',
    'NVC56F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE',
    'NVC56F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE_TARGETED',
    'NVC56F_MEM_OP_D_TLB_INVALIDATE_PDB_ADDR_HI',
    'NVC56F_NON_STALL_INTERRUPT', 'NVC56F_NON_STALL_INTERRUPT_HANDLE',
    'NVC56F_NOP', 'NVC56F_NOP_HANDLE', 'NVC56F_NUMBER_OF_SUBCHANNELS',
    'NVC56F_SEMAPHOREA', 'NVC56F_SEMAPHOREA_OFFSET_UPPER',
    'NVC56F_SEMAPHOREB', 'NVC56F_SEMAPHOREB_OFFSET_LOWER',
    'NVC56F_SEMAPHOREC', 'NVC56F_SEMAPHOREC_PAYLOAD',
    'NVC56F_SEMAPHORED', 'NVC56F_SEMAPHORED_ACQUIRE_SWITCH',
    'NVC56F_SEMAPHORED_ACQUIRE_SWITCH_DISABLED',
    'NVC56F_SEMAPHORED_ACQUIRE_SWITCH_ENABLED',
    'NVC56F_SEMAPHORED_FORMAT', 'NVC56F_SEMAPHORED_FORMAT_SIGNED',
    'NVC56F_SEMAPHORED_FORMAT_UNSIGNED',
    'NVC56F_SEMAPHORED_OPERATION',
    'NVC56F_SEMAPHORED_OPERATION_ACQUIRE',
    'NVC56F_SEMAPHORED_OPERATION_ACQ_AND',
    'NVC56F_SEMAPHORED_OPERATION_ACQ_GEQ',
    'NVC56F_SEMAPHORED_OPERATION_REDUCTION',
    'NVC56F_SEMAPHORED_OPERATION_RELEASE',
    'NVC56F_SEMAPHORED_REDUCTION', 'NVC56F_SEMAPHORED_REDUCTION_ADD',
    'NVC56F_SEMAPHORED_REDUCTION_AND',
    'NVC56F_SEMAPHORED_REDUCTION_DEC',
    'NVC56F_SEMAPHORED_REDUCTION_INC',
    'NVC56F_SEMAPHORED_REDUCTION_MAX',
    'NVC56F_SEMAPHORED_REDUCTION_MIN',
    'NVC56F_SEMAPHORED_REDUCTION_OR',
    'NVC56F_SEMAPHORED_REDUCTION_XOR',
    'NVC56F_SEMAPHORED_RELEASE_SIZE',
    'NVC56F_SEMAPHORED_RELEASE_SIZE_16BYTE',
    'NVC56F_SEMAPHORED_RELEASE_SIZE_4BYTE',
    'NVC56F_SEMAPHORED_RELEASE_WFI',
    'NVC56F_SEMAPHORED_RELEASE_WFI_DIS',
    'NVC56F_SEMAPHORED_RELEASE_WFI_EN', 'NVC56F_SEM_ADDR_HI',
    'NVC56F_SEM_ADDR_HI_OFFSET', 'NVC56F_SEM_ADDR_LO',
    'NVC56F_SEM_ADDR_LO_OFFSET', 'NVC56F_SEM_EXECUTE',
    'NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG',
    'NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG_DIS',
    'NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG_EN',
    'NVC56F_SEM_EXECUTE_OPERATION',
    'NVC56F_SEM_EXECUTE_OPERATION_ACQUIRE',
    'NVC56F_SEM_EXECUTE_OPERATION_ACQ_AND',
    'NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ',
    'NVC56F_SEM_EXECUTE_OPERATION_ACQ_NOR',
    'NVC56F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ',
    'NVC56F_SEM_EXECUTE_OPERATION_REDUCTION',
    'NVC56F_SEM_EXECUTE_OPERATION_RELEASE',
    'NVC56F_SEM_EXECUTE_PAYLOAD_SIZE',
    'NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_32BIT',
    'NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT',
    'NVC56F_SEM_EXECUTE_REDUCTION',
    'NVC56F_SEM_EXECUTE_REDUCTION_DEC',
    'NVC56F_SEM_EXECUTE_REDUCTION_FORMAT',
    'NVC56F_SEM_EXECUTE_REDUCTION_FORMAT_SIGNED',
    'NVC56F_SEM_EXECUTE_REDUCTION_FORMAT_UNSIGNED',
    'NVC56F_SEM_EXECUTE_REDUCTION_IADD',
    'NVC56F_SEM_EXECUTE_REDUCTION_IAND',
    'NVC56F_SEM_EXECUTE_REDUCTION_IMAX',
    'NVC56F_SEM_EXECUTE_REDUCTION_IMIN',
    'NVC56F_SEM_EXECUTE_REDUCTION_INC',
    'NVC56F_SEM_EXECUTE_REDUCTION_IOR',
    'NVC56F_SEM_EXECUTE_REDUCTION_IXOR',
    'NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP',
    'NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS',
    'NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN',
    'NVC56F_SEM_EXECUTE_RELEASE_WFI',
    'NVC56F_SEM_EXECUTE_RELEASE_WFI_DIS',
    'NVC56F_SEM_EXECUTE_RELEASE_WFI_EN', 'NVC56F_SEM_PAYLOAD_HI',
    'NVC56F_SEM_PAYLOAD_HI_PAYLOAD', 'NVC56F_SEM_PAYLOAD_LO',
    'NVC56F_SEM_PAYLOAD_LO_PAYLOAD', 'NVC56F_SET_OBJECT',
    'NVC56F_SET_OBJECT_ENGINE', 'NVC56F_SET_OBJECT_ENGINE_SW',
    'NVC56F_SET_OBJECT_NVCLASS', 'NVC56F_SET_REFERENCE',
    'NVC56F_SET_REFERENCE_COUNT', 'NVC56F_WFI', 'NVC56F_WFI_SCOPE',
    'NVC56F_WFI_SCOPE_ALL', 'NVC56F_WFI_SCOPE_CURRENT_SCG_TYPE',
    'NVC56F_WFI_SCOPE_CURRENT_VEID', 'NVC56F_YIELD',
    'NVC56F_YIELD_OP', 'NVC56F_YIELD_OP_NOP', 'NVC56F_YIELD_OP_TSG',
    'NVC570_DISPLAY', 'NVC573_DISP_CAPABILITIES',
    'NVC57A_CURSOR_IMM_CHANNEL_PIO', 'NVC57B_WINDOW_IMM_CHANNEL_DMA',
    'NVC57D_CORE_CHANNEL_DMA', 'NVC57E_WINDOW_CHANNEL_DMA',
    'NVC670_DISPLAY', 'NVC671_DISP_SF_USER',
    'NVC673_DISP_CAPABILITIES', 'NVC67A_CURSOR_IMM_CHANNEL_PIO',
    'NVC67B_WINDOW_IMM_CHANNEL_DMA', 'NVC67D_CORE_CHANNEL_DMA',
    'NVC67E_WINDOW_CHANNEL_DMA', 'NVC6B0_VIDEO_DECODER',
    'NVC6FA_VIDEO_OFA', 'NVC770_DISPLAY', 'NVC771_DISP_SF_USER',
    'NVC773_DISP_CAPABILITIES', 'NVC77D_CORE_CHANNEL_DMA',
    'NVC77F_ANY_CHANNEL_DMA', 'NVC7B0_VIDEO_DECODER',
    'NVC7B7_VIDEO_ENCODER', 'NVC7FA_VIDEO_OFA',
    'NVC9B0_VIDEO_DECODER', 'NVC9B7_VIDEO_ENCODER',
    'NVC9D1_VIDEO_NVJPG', 'NVC9FA_VIDEO_OFA', 'NVENC_SW_SESSION',
    'NVFBC_SW_SESSION', 'NV_CE_UTILS', 'NV_CONFIDENTIAL_COMPUTE',
    'NV_COUNTER_COLLECTION_UNIT', 'NV_EVENT_BUFFER',
    'NV_IMEX_SESSION', 'NV_MEMORY_EXPORT', 'NV_MEMORY_EXTENDED_USER',
    'NV_MEMORY_FABRIC', 'NV_MEMORY_FABRIC_IMPORTED_REF',
    'NV_MEMORY_FABRIC_IMPORT_V2', 'NV_MEMORY_MAPPER',
    'NV_MEMORY_MULTICAST_FABRIC', 'NV_SEMAPHORE_SURFACE',
    'Nv2080ACPIEvent', 'Nv2080AudioHdcpRequest',
    'Nv2080ClocksChangeNotification', 'Nv2080DpIrqNotification',
    'Nv2080DstateHdaCodecNotification',
    'Nv2080DstateXusbPpcNotification', 'Nv2080EccDbeNotification',
    'Nv2080GC5GpuReadyParams', 'Nv2080HdcpStatusChangeNotification',
    'Nv2080HdmiFrlRequestNotification', 'Nv2080HotplugNotification',
    'Nv2080LpwrDifrPrefetchNotification',
    'Nv2080NvlinkLnkChangeNotification',
    'Nv2080PStateChangeNotification', 'Nv2080PowerEventNotification',
    'Nv2080PrivRegAccessFaultNotification',
    'Nv2080QosIntrNotification', 'Nv2080Typedef',
    'Nv2080VrrSetTimeoutNotification',
    'Nv2080WorkloadModulationChangeNotification',
    'Nv2080XusbPpcConnectStateNotification', 'Nv20Subdevice0',
    'Nvc56fControl', 'PASCAL_CHANNEL_GPFIFO_A', 'PASCAL_DMA_COPY_A',
    'RM_USER_SHARED_DATA', 'TURING_A', 'TURING_CHANNEL_GPFIFO_A',
    'TURING_COMPUTE_A', 'TURING_DMA_COPY_A', 'TURING_USERMODE_A',
    'UVM_CHANNEL_RETAINER', 'VOLTA_CHANNEL_GPFIFO_A',
    'VOLTA_USERMODE_A', '_cl2080_notification_h_', '_clc56f_h_',
    'struct_NV0080_ALLOC_PARAMETERS', 'struct_Nv2080ACPIEvent',
    'struct_Nv2080AudioHdcpRequestRec',
    'struct_Nv2080ClocksChangeNotificationRec',
    'struct_Nv2080ClocksChangeNotificationRec_timeStamp',
    'struct_Nv2080DpIrqNotificationRec',
    'struct_Nv2080DstateHdaCodecNotificationRec',
    'struct_Nv2080DstateXusbPpcNotificationRec',
    'struct_Nv2080GC5GpuReadyParams',
    'struct_Nv2080HdcpStatusChangeNotificationRec',
    'struct_Nv2080HdmiFrlRequestNotificationRec',
    'struct_Nv2080PStateChangeNotificationRec',
    'struct_Nv2080PStateChangeNotificationRec_timeStamp',
    'struct_Nv2080WorkloadModulationChangeNotificationRec',
    'struct_Nv2080WorkloadModulationChangeNotificationRec_timeStamp',
    'struct_Nv2080XusbPpcConnectStateNotificationRec',
    'struct_Nvc56fControl_struct',
    'struct__NV2080_COOLER_DIAG_ZONE_NOTIFICATION_REC',
    'struct__NV2080_PLATFORM_POWER_MODE_CHANGE_STATUS',
    'struct__NV2080_THERM_DIAG_ZONE_NOTIFICATION_REC',
    'struct__cl2080_tag0', 'struct_c__SA_Nv2080EccDbeNotification',
    'struct_c__SA_Nv2080HotplugNotification',
    'struct_c__SA_Nv2080LpwrDifrPrefetchNotification',
    'struct_c__SA_Nv2080NvlinkLnkChangeNotification',
    'struct_c__SA_Nv2080PowerEventNotification',
    'struct_c__SA_Nv2080PrivRegAccessFaultNotification',
    'struct_c__SA_Nv2080QosIntrNotification',
    'struct_c__SA_Nv2080VrrSetTimeoutNotification']
