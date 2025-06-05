# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DRPC_MESSAGE_STRUCTURES', '-include', '/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/nvtypes.h', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/generated', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/interface/', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc/kernel', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc/libraries', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/nvidia-uvm', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/unix/include', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/ctrl']
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

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





KERN_FSP_COT_PAYLOAD_H = True # macro
class struct_c__SA_MCTP_HEADER(Structure):
    pass

struct_c__SA_MCTP_HEADER._pack_ = 1 # source:False
struct_c__SA_MCTP_HEADER._fields_ = [
    ('constBlob', ctypes.c_uint32),
    ('msgType', ctypes.c_ubyte),
    ('vendorId', ctypes.c_uint16),
]

MCTP_HEADER = struct_c__SA_MCTP_HEADER
class struct_c__SA_NVDM_PAYLOAD_COT(Structure):
    pass

struct_c__SA_NVDM_PAYLOAD_COT._pack_ = 1 # source:False
struct_c__SA_NVDM_PAYLOAD_COT._fields_ = [
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('gspFmcSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemSize', ctypes.c_uint32),
    ('frtsVidmemOffset', ctypes.c_uint64),
    ('frtsVidmemSize', ctypes.c_uint32),
    ('hash384', ctypes.c_uint32 * 12),
    ('publicKey', ctypes.c_uint32 * 96),
    ('signature', ctypes.c_uint32 * 96),
    ('gspBootArgsSysmemOffset', ctypes.c_uint64),
]

NVDM_PAYLOAD_COT = struct_c__SA_NVDM_PAYLOAD_COT
GSPIFPUB_H = True # macro

# values for enumeration 'c__EA_GSP_DMA_TARGET'
c__EA_GSP_DMA_TARGET__enumvalues = {
    0: 'GSP_DMA_TARGET_LOCAL_FB',
    1: 'GSP_DMA_TARGET_COHERENT_SYSTEM',
    2: 'GSP_DMA_TARGET_NONCOHERENT_SYSTEM',
    3: 'GSP_DMA_TARGET_COUNT',
}
GSP_DMA_TARGET_LOCAL_FB = 0
GSP_DMA_TARGET_COHERENT_SYSTEM = 1
GSP_DMA_TARGET_NONCOHERENT_SYSTEM = 2
GSP_DMA_TARGET_COUNT = 3
c__EA_GSP_DMA_TARGET = ctypes.c_uint32 # enum
GSP_DMA_TARGET = c__EA_GSP_DMA_TARGET
GSP_DMA_TARGET__enumvalues = c__EA_GSP_DMA_TARGET__enumvalues
class struct_GSP_FMC_INIT_PARAMS(Structure):
    pass

struct_GSP_FMC_INIT_PARAMS._pack_ = 1 # source:False
struct_GSP_FMC_INIT_PARAMS._fields_ = [
    ('regkeys', ctypes.c_uint32),
]

GSP_FMC_INIT_PARAMS = struct_GSP_FMC_INIT_PARAMS
class struct_GSP_ACR_BOOT_GSP_RM_PARAMS(Structure):
    pass

struct_GSP_ACR_BOOT_GSP_RM_PARAMS._pack_ = 1 # source:False
struct_GSP_ACR_BOOT_GSP_RM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('gspRmDescSize', ctypes.c_uint32),
    ('gspRmDescOffset', ctypes.c_uint64),
    ('wprCarveoutOffset', ctypes.c_uint64),
    ('wprCarveoutSize', ctypes.c_uint32),
    ('bIsGspRmBoot', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

GSP_ACR_BOOT_GSP_RM_PARAMS = struct_GSP_ACR_BOOT_GSP_RM_PARAMS
class struct_GSP_RM_PARAMS(Structure):
    pass

struct_GSP_RM_PARAMS._pack_ = 1 # source:False
struct_GSP_RM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bootArgsOffset', ctypes.c_uint64),
]

GSP_RM_PARAMS = struct_GSP_RM_PARAMS
class struct_GSP_SPDM_PARAMS(Structure):
    pass

struct_GSP_SPDM_PARAMS._pack_ = 1 # source:False
struct_GSP_SPDM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('payloadBufferOffset', ctypes.c_uint64),
    ('payloadBufferSize', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

GSP_SPDM_PARAMS = struct_GSP_SPDM_PARAMS
class struct_GSP_FMC_BOOT_PARAMS(Structure):
    pass

struct_GSP_FMC_BOOT_PARAMS._pack_ = 1 # source:False
struct_GSP_FMC_BOOT_PARAMS._fields_ = [
    ('initParams', GSP_FMC_INIT_PARAMS),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bootGspRmParams', GSP_ACR_BOOT_GSP_RM_PARAMS),
    ('gspRmParams', GSP_RM_PARAMS),
    ('gspSpdmParams', GSP_SPDM_PARAMS),
]

GSP_FMC_BOOT_PARAMS = struct_GSP_FMC_BOOT_PARAMS
GSP_FW_WPR_META_H_ = True # macro
GSP_FW_WPR_META_VERIFIED = 0xa0a0a0a0a0a0a0a0 # macro
GSP_FW_WPR_META_REVISION = 1 # macro
GSP_FW_WPR_META_MAGIC = 0xdc3aae21371a60b3 # macro
GSP_FW_WPR_HEAP_FREE_REGION_COUNT = 128 # macro
GSP_FW_HEAP_FREE_LIST_MAGIC = 0x4845415046524545 # macro
# GSP_FW_FLAGS = 8 : 0 # macro
# GSP_FW_FLAGS_CLOCK_BOOST = NVBIT ( 0 ) # macro
# GSP_FW_FLAGS_RECOVERY_MARGIN_PRESENT = NVBIT ( 1 ) # macro
# GSP_FW_FLAGS_PPCIE_ENABLED = NVBIT ( 2 ) # macro
class struct_c__SA_GspFwWprMeta(Structure):
    pass

class union_c__SA_GspFwWprMeta_0(Union):
    pass

class struct_c__SA_GspFwWprMeta_0_0(Structure):
    pass

struct_c__SA_GspFwWprMeta_0_0._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_0_0._fields_ = [
    ('sysmemAddrOfSignature', ctypes.c_uint64),
    ('sizeOfSignature', ctypes.c_uint64),
]

class struct_c__SA_GspFwWprMeta_0_1(Structure):
    pass

struct_c__SA_GspFwWprMeta_0_1._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_0_1._fields_ = [
    ('gspFwHeapFreeListWprOffset', ctypes.c_uint32),
    ('unused0', ctypes.c_uint32),
    ('unused1', ctypes.c_uint64),
]

union_c__SA_GspFwWprMeta_0._pack_ = 1 # source:False
union_c__SA_GspFwWprMeta_0._anonymous_ = ('_0', '_1',)
union_c__SA_GspFwWprMeta_0._fields_ = [
    ('_0', struct_c__SA_GspFwWprMeta_0_0),
    ('_1', struct_c__SA_GspFwWprMeta_0_1),
]

class union_c__SA_GspFwWprMeta_1(Union):
    pass

class struct_c__SA_GspFwWprMeta_1_0(Structure):
    pass

struct_c__SA_GspFwWprMeta_1_0._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_1_0._fields_ = [
    ('partitionRpcAddr', ctypes.c_uint64),
    ('partitionRpcRequestOffset', ctypes.c_uint16),
    ('partitionRpcReplyOffset', ctypes.c_uint16),
    ('elfCodeOffset', ctypes.c_uint32),
    ('elfDataOffset', ctypes.c_uint32),
    ('elfCodeSize', ctypes.c_uint32),
    ('elfDataSize', ctypes.c_uint32),
    ('lsUcodeVersion', ctypes.c_uint32),
]

class struct_c__SA_GspFwWprMeta_1_1(Structure):
    pass

struct_c__SA_GspFwWprMeta_1_1._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_1_1._fields_ = [
    ('partitionRpcPadding', ctypes.c_uint32 * 4),
    ('sysmemAddrOfCrashReportQueue', ctypes.c_uint64),
    ('sizeOfCrashReportQueue', ctypes.c_uint32),
    ('lsUcodeVersionPadding', ctypes.c_uint32 * 1),
]

union_c__SA_GspFwWprMeta_1._pack_ = 1 # source:False
union_c__SA_GspFwWprMeta_1._anonymous_ = ('_0', '_1',)
union_c__SA_GspFwWprMeta_1._fields_ = [
    ('_0', struct_c__SA_GspFwWprMeta_1_0),
    ('_1', struct_c__SA_GspFwWprMeta_1_1),
]

struct_c__SA_GspFwWprMeta._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta._anonymous_ = ('_0', '_1',)
struct_c__SA_GspFwWprMeta._fields_ = [
    ('magic', ctypes.c_uint64),
    ('revision', ctypes.c_uint64),
    ('sysmemAddrOfRadix3Elf', ctypes.c_uint64),
    ('sizeOfRadix3Elf', ctypes.c_uint64),
    ('sysmemAddrOfBootloader', ctypes.c_uint64),
    ('sizeOfBootloader', ctypes.c_uint64),
    ('bootloaderCodeOffset', ctypes.c_uint64),
    ('bootloaderDataOffset', ctypes.c_uint64),
    ('bootloaderManifestOffset', ctypes.c_uint64),
    ('_0', union_c__SA_GspFwWprMeta_0),
    ('gspFwRsvdStart', ctypes.c_uint64),
    ('nonWprHeapOffset', ctypes.c_uint64),
    ('nonWprHeapSize', ctypes.c_uint64),
    ('gspFwWprStart', ctypes.c_uint64),
    ('gspFwHeapOffset', ctypes.c_uint64),
    ('gspFwHeapSize', ctypes.c_uint64),
    ('gspFwOffset', ctypes.c_uint64),
    ('bootBinOffset', ctypes.c_uint64),
    ('frtsOffset', ctypes.c_uint64),
    ('frtsSize', ctypes.c_uint64),
    ('gspFwWprEnd', ctypes.c_uint64),
    ('fbSize', ctypes.c_uint64),
    ('vgaWorkspaceOffset', ctypes.c_uint64),
    ('vgaWorkspaceSize', ctypes.c_uint64),
    ('bootCount', ctypes.c_uint64),
    ('_1', union_c__SA_GspFwWprMeta_1),
    ('gspFwHeapVfPartitionCount', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('padding', ctypes.c_ubyte * 2),
    ('pmuReservedSize', ctypes.c_uint32),
    ('verified', ctypes.c_uint64),
]

GspFwWprMeta = struct_c__SA_GspFwWprMeta
class struct_c__SA_GspFwHeapFreeRegion(Structure):
    pass

struct_c__SA_GspFwHeapFreeRegion._pack_ = 1 # source:False
struct_c__SA_GspFwHeapFreeRegion._fields_ = [
    ('offs', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

GspFwHeapFreeRegion = struct_c__SA_GspFwHeapFreeRegion
class struct_c__SA_GspFwHeapFreeList(Structure):
    pass

struct_c__SA_GspFwHeapFreeList._pack_ = 1 # source:False
struct_c__SA_GspFwHeapFreeList._fields_ = [
    ('magic', ctypes.c_uint64),
    ('nregions', ctypes.c_uint32),
    ('regions', struct_c__SA_GspFwHeapFreeRegion * 128),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

GspFwHeapFreeList = struct_c__SA_GspFwHeapFreeList
GSP_FW_SR_META_H_ = True # macro
GSP_FW_SR_META_MAGIC = 0x8a3bb9e6c6c39d93 # macro
GSP_FW_SR_META_REVISION = 2 # macro
GSP_FW_SR_META_INTERNAL_SIZE = 128 # macro
class struct_c__SA_GspFwSRMeta(Structure):
    pass

struct_c__SA_GspFwSRMeta._pack_ = 1 # source:False
struct_c__SA_GspFwSRMeta._fields_ = [
    ('magic', ctypes.c_uint64),
    ('revision', ctypes.c_uint64),
    ('sysmemAddrOfSuspendResumeData', ctypes.c_uint64),
    ('sizeOfSuspendResumeData', ctypes.c_uint64),
    ('internal', ctypes.c_uint32 * 32),
    ('flags', ctypes.c_uint32),
    ('subrevision', ctypes.c_uint32),
    ('padding', ctypes.c_uint32 * 22),
]

GspFwSRMeta = struct_c__SA_GspFwSRMeta
GSP_INIT_ARGS_H = True # macro
class struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS(Structure):
    pass

struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS._pack_ = 1 # source:False
struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS._fields_ = [
    ('sharedMemPhysAddr', ctypes.c_uint64),
    ('pageTableEntryCount', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cmdQueueOffset', ctypes.c_uint64),
    ('statQueueOffset', ctypes.c_uint64),
]

MESSAGE_QUEUE_INIT_ARGUMENTS = struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS
class struct_c__SA_GSP_SR_INIT_ARGUMENTS(Structure):
    pass

struct_c__SA_GSP_SR_INIT_ARGUMENTS._pack_ = 1 # source:False
struct_c__SA_GSP_SR_INIT_ARGUMENTS._fields_ = [
    ('oldLevel', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('bInPMTransition', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

GSP_SR_INIT_ARGUMENTS = struct_c__SA_GSP_SR_INIT_ARGUMENTS
class struct_c__SA_GSP_ARGUMENTS_CACHED(Structure):
    pass

class struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs(Structure):
    pass

struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs._pack_ = 1 # source:False
struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs._fields_ = [
    ('pa', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

struct_c__SA_GSP_ARGUMENTS_CACHED._pack_ = 1 # source:False
struct_c__SA_GSP_ARGUMENTS_CACHED._fields_ = [
    ('messageQueueInitArguments', MESSAGE_QUEUE_INIT_ARGUMENTS),
    ('srInitArguments', GSP_SR_INIT_ARGUMENTS),
    ('gpuInstance', ctypes.c_uint32),
    ('bDmemStack', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('profilerArgs', struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs),
]

GSP_ARGUMENTS_CACHED = struct_c__SA_GSP_ARGUMENTS_CACHED
LIBOS_INIT_H_ = True # macro
LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX = 4096 # macro
LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE = 4096 # macro
LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 = 12 # macro
LibosAddress = ctypes.c_uint64

# values for enumeration 'c__EA_LibosMemoryRegionKind'
c__EA_LibosMemoryRegionKind__enumvalues = {
    0: 'LIBOS_MEMORY_REGION_NONE',
    1: 'LIBOS_MEMORY_REGION_CONTIGUOUS',
    2: 'LIBOS_MEMORY_REGION_RADIX3',
}
LIBOS_MEMORY_REGION_NONE = 0
LIBOS_MEMORY_REGION_CONTIGUOUS = 1
LIBOS_MEMORY_REGION_RADIX3 = 2
c__EA_LibosMemoryRegionKind = ctypes.c_uint32 # enum
LibosMemoryRegionKind = c__EA_LibosMemoryRegionKind
LibosMemoryRegionKind__enumvalues = c__EA_LibosMemoryRegionKind__enumvalues

# values for enumeration 'c__EA_LibosMemoryRegionLoc'
c__EA_LibosMemoryRegionLoc__enumvalues = {
    0: 'LIBOS_MEMORY_REGION_LOC_NONE',
    1: 'LIBOS_MEMORY_REGION_LOC_SYSMEM',
    2: 'LIBOS_MEMORY_REGION_LOC_FB',
}
LIBOS_MEMORY_REGION_LOC_NONE = 0
LIBOS_MEMORY_REGION_LOC_SYSMEM = 1
LIBOS_MEMORY_REGION_LOC_FB = 2
c__EA_LibosMemoryRegionLoc = ctypes.c_uint32 # enum
LibosMemoryRegionLoc = c__EA_LibosMemoryRegionLoc
LibosMemoryRegionLoc__enumvalues = c__EA_LibosMemoryRegionLoc__enumvalues
class struct_c__SA_LibosMemoryRegionInitArgument(Structure):
    pass

struct_c__SA_LibosMemoryRegionInitArgument._pack_ = 1 # source:False
struct_c__SA_LibosMemoryRegionInitArgument._fields_ = [
    ('id8', ctypes.c_uint64),
    ('pa', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('kind', ctypes.c_ubyte),
    ('loc', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

LibosMemoryRegionInitArgument = struct_c__SA_LibosMemoryRegionInitArgument
RM_RISCV_UCODE_H = True # macro
class struct_c__SA_RM_RISCV_UCODE_DESC(Structure):
    pass

struct_c__SA_RM_RISCV_UCODE_DESC._pack_ = 1 # source:False
struct_c__SA_RM_RISCV_UCODE_DESC._fields_ = [
    ('version', ctypes.c_uint32),
    ('bootloaderOffset', ctypes.c_uint32),
    ('bootloaderSize', ctypes.c_uint32),
    ('bootloaderParamOffset', ctypes.c_uint32),
    ('bootloaderParamSize', ctypes.c_uint32),
    ('riscvElfOffset', ctypes.c_uint32),
    ('riscvElfSize', ctypes.c_uint32),
    ('appVersion', ctypes.c_uint32),
    ('manifestOffset', ctypes.c_uint32),
    ('manifestSize', ctypes.c_uint32),
    ('monitorDataOffset', ctypes.c_uint32),
    ('monitorDataSize', ctypes.c_uint32),
    ('monitorCodeOffset', ctypes.c_uint32),
    ('monitorCodeSize', ctypes.c_uint32),
    ('bIsMonitorEnabled', ctypes.c_uint32),
    ('swbromCodeOffset', ctypes.c_uint32),
    ('swbromCodeSize', ctypes.c_uint32),
    ('swbromDataOffset', ctypes.c_uint32),
    ('swbromDataSize', ctypes.c_uint32),
    ('fbReservedSize', ctypes.c_uint32),
    ('bSignedAsCode', ctypes.c_uint32),
]

RM_RISCV_UCODE_DESC = struct_c__SA_RM_RISCV_UCODE_DESC
MSGQ_PRIV_H = True # macro
MSGQ_VERSION = 0 # macro
class struct_c__SA_msgqTxHeader(Structure):
    pass

struct_c__SA_msgqTxHeader._pack_ = 1 # source:False
struct_c__SA_msgqTxHeader._fields_ = [
    ('version', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('msgSize', ctypes.c_uint32),
    ('msgCount', ctypes.c_uint32),
    ('writePtr', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('rxHdrOff', ctypes.c_uint32),
    ('entryOff', ctypes.c_uint32),
]

msgqTxHeader = struct_c__SA_msgqTxHeader
class struct_c__SA_msgqRxHeader(Structure):
    pass

struct_c__SA_msgqRxHeader._pack_ = 1 # source:False
struct_c__SA_msgqRxHeader._fields_ = [
    ('readPtr', ctypes.c_uint32),
]

msgqRxHeader = struct_c__SA_msgqRxHeader
class struct_c__SA_msgqMetadata(Structure):
    pass

struct_c__SA_msgqMetadata._pack_ = 1 # source:False
struct_c__SA_msgqMetadata._fields_ = [
    ('pOurTxHdr', ctypes.POINTER(struct_c__SA_msgqTxHeader)),
    ('pTheirTxHdr', ctypes.POINTER(struct_c__SA_msgqTxHeader)),
    ('pOurRxHdr', ctypes.POINTER(struct_c__SA_msgqRxHeader)),
    ('pTheirRxHdr', ctypes.POINTER(struct_c__SA_msgqRxHeader)),
    ('pOurEntries', ctypes.POINTER(ctypes.c_ubyte)),
    ('pTheirEntries', ctypes.POINTER(ctypes.c_ubyte)),
    ('pReadIncoming', ctypes.POINTER(ctypes.c_uint32)),
    ('pWriteIncoming', ctypes.POINTER(ctypes.c_uint32)),
    ('pReadOutgoing', ctypes.POINTER(ctypes.c_uint32)),
    ('pWriteOutgoing', ctypes.POINTER(ctypes.c_uint32)),
    ('tx', msgqTxHeader),
    ('txReadPtr', ctypes.c_uint32),
    ('txFree', ctypes.c_uint32),
    ('txLinked', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('rx', msgqTxHeader),
    ('rxReadPtr', ctypes.c_uint32),
    ('rxAvail', ctypes.c_uint32),
    ('rxLinked', ctypes.c_ubyte),
    ('rxSwapped', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('fcnNotify', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))),
    ('fcnNotifyArg', ctypes.POINTER(None)),
    ('fcnBackendRw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(None))),
    ('fcnBackendRwArg', ctypes.POINTER(None)),
    ('fcnInvalidate', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnFlush', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnZero', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnBarrier', ctypes.CFUNCTYPE(None)),
]

msgqMetadata = struct_c__SA_msgqMetadata
__vgpu_rpc_nv_headers_h__ = True # macro
MAX_GPC_COUNT = 32 # macro
VGPU_MAX_REGOPS_PER_RPC = 100 # macro
VGPU_RESERVED_HANDLE_BASE = 0xCAF3F000 # macro
VGPU_RESERVED_HANDLE_RANGE = 0x1000 # macro
# def VGPU_CALC_PARAM_OFFSET(prev_offset, prev_params):  # macro
#    return (prev_offset+NV_ALIGN_UP(ctypes.sizeof(prev_params),ctypes.sizeof(NvU32)))
# NV_VGPU_MSG_HEADER_VERSION_MAJOR = 31 : 24 # macro
# NV_VGPU_MSG_HEADER_VERSION_MINOR = 23 : 16 # macro
NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT = 0x00000003 # macro
NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT = 0x00000000 # macro
NV_VGPU_MSG_SIGNATURE_VALID = 0x43505256 # macro
# NV_VGPU_MSG_RESULT__RM = NV_ERR_GENERIC : 0x00000000 # macro
# NV_VGPU_MSG_RESULT_SUCCESS = NV_OK # macro
# NV_VGPU_MSG_RESULT_CARD_NOT_PRESENT = NV_ERR_CARD_NOT_PRESENT # macro
# NV_VGPU_MSG_RESULT_DUAL_LINK_INUSE = NV_ERR_DUAL_LINK_INUSE # macro
# NV_VGPU_MSG_RESULT_GENERIC = NV_ERR_GENERIC # macro
# NV_VGPU_MSG_RESULT_GPU_NOT_FULL_POWER = NV_ERR_GPU_NOT_FULL_POWER # macro
# NV_VGPU_MSG_RESULT_IN_USE = NV_ERR_IN_USE # macro
# NV_VGPU_MSG_RESULT_INSUFFICIENT_RESOURCES = NV_ERR_INSUFFICIENT_RESOURCES # macro
# NV_VGPU_MSG_RESULT_INVALID_ACCESS_TYPE = NV_ERR_INVALID_ACCESS_TYPE # macro
# NV_VGPU_MSG_RESULT_INVALID_ARGUMENT = NV_ERR_INVALID_ARGUMENT # macro
# NV_VGPU_MSG_RESULT_INVALID_BASE = NV_ERR_INVALID_BASE # macro
# NV_VGPU_MSG_RESULT_INVALID_CHANNEL = NV_ERR_INVALID_CHANNEL # macro
# NV_VGPU_MSG_RESULT_INVALID_CLASS = NV_ERR_INVALID_CLASS # macro
# NV_VGPU_MSG_RESULT_INVALID_CLIENT = NV_ERR_INVALID_CLIENT # macro
# NV_VGPU_MSG_RESULT_INVALID_COMMAND = NV_ERR_INVALID_COMMAND # macro
# NV_VGPU_MSG_RESULT_INVALID_DATA = NV_ERR_INVALID_DATA # macro
# NV_VGPU_MSG_RESULT_INVALID_DEVICE = NV_ERR_INVALID_DEVICE # macro
# NV_VGPU_MSG_RESULT_INVALID_DMA_SPECIFIER = NV_ERR_INVALID_DMA_SPECIFIER # macro
# NV_VGPU_MSG_RESULT_INVALID_EVENT = NV_ERR_INVALID_EVENT # macro
# NV_VGPU_MSG_RESULT_INVALID_FLAGS = NV_ERR_INVALID_FLAGS # macro
# NV_VGPU_MSG_RESULT_INVALID_FUNCTION = NV_ERR_INVALID_FUNCTION # macro
# NV_VGPU_MSG_RESULT_INVALID_HEAP = NV_ERR_INVALID_HEAP # macro
# NV_VGPU_MSG_RESULT_INVALID_INDEX = NV_ERR_INVALID_INDEX # macro
# NV_VGPU_MSG_RESULT_INVALID_LIMIT = NV_ERR_INVALID_LIMIT # macro
# NV_VGPU_MSG_RESULT_INVALID_METHOD = NV_ERR_INVALID_METHOD # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_BUFFER = NV_ERR_INVALID_OBJECT_BUFFER # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_ERROR = NV_ERR_INVALID_OBJECT # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_HANDLE = NV_ERR_INVALID_OBJECT_HANDLE # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_NEW = NV_ERR_INVALID_OBJECT_NEW # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_OLD = NV_ERR_INVALID_OBJECT_OLD # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_PARENT = NV_ERR_INVALID_OBJECT_PARENT # macro
# NV_VGPU_MSG_RESULT_INVALID_OFFSET = NV_ERR_INVALID_OFFSET # macro
# NV_VGPU_MSG_RESULT_INVALID_OWNER = NV_ERR_INVALID_OWNER # macro
# NV_VGPU_MSG_RESULT_INVALID_PARAM_STRUCT = NV_ERR_INVALID_PARAM_STRUCT # macro
# NV_VGPU_MSG_RESULT_INVALID_PARAMETER = NV_ERR_INVALID_PARAMETER # macro
# NV_VGPU_MSG_RESULT_INVALID_POINTER = NV_ERR_INVALID_POINTER # macro
# NV_VGPU_MSG_RESULT_INVALID_REGISTRY_KEY = NV_ERR_INVALID_REGISTRY_KEY # macro
# NV_VGPU_MSG_RESULT_INVALID_STATE = NV_ERR_INVALID_STATE # macro
# NV_VGPU_MSG_RESULT_INVALID_STRING_LENGTH = NV_ERR_INVALID_STRING_LENGTH # macro
# NV_VGPU_MSG_RESULT_INVALID_XLATE = NV_ERR_INVALID_XLATE # macro
# NV_VGPU_MSG_RESULT_IRQ_NOT_FIRING = NV_ERR_IRQ_NOT_FIRING # macro
# NV_VGPU_MSG_RESULT_MULTIPLE_MEMORY_TYPES = NV_ERR_MULTIPLE_MEMORY_TYPES # macro
# NV_VGPU_MSG_RESULT_NOT_SUPPORTED = NV_ERR_NOT_SUPPORTED # macro
# NV_VGPU_MSG_RESULT_OPERATING_SYSTEM = NV_ERR_OPERATING_SYSTEM # macro
# NV_VGPU_MSG_RESULT_PROTECTION_FAULT = NV_ERR_PROTECTION_FAULT # macro
# NV_VGPU_MSG_RESULT_TIMEOUT = NV_ERR_TIMEOUT # macro
# NV_VGPU_MSG_RESULT_TOO_MANY_PRIMARIES = NV_ERR_TOO_MANY_PRIMARIES # macro
# NV_VGPU_MSG_RESULT_IRQ_EDGE_TRIGGERED = NV_ERR_IRQ_EDGE_TRIGGERED # macro
# NV_VGPU_MSG_RESULT_GUEST_HOST_DRIVER_MISMATCH = NV_ERR_LIB_RM_VERSION_MISMATCH # macro
# NV_VGPU_MSG_RESULT__VMIOP = 0xFF00000a : 0xFF000000 # macro
NV_VGPU_MSG_RESULT_VMIOP_INVAL = 0xFF000001 # macro
NV_VGPU_MSG_RESULT_VMIOP_RESOURCE = 0xFF000002 # macro
NV_VGPU_MSG_RESULT_VMIOP_RANGE = 0xFF000003 # macro
NV_VGPU_MSG_RESULT_VMIOP_READ_ONLY = 0xFF000004 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_FOUND = 0xFF000005 # macro
NV_VGPU_MSG_RESULT_VMIOP_NO_ADDRESS_SPACE = 0xFF000006 # macro
NV_VGPU_MSG_RESULT_VMIOP_TIMEOUT = 0xFF000007 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_ALLOWED_IN_CALLBACK = 0xFF000008 # macro
NV_VGPU_MSG_RESULT_VMIOP_ECC_MISMATCH = 0xFF000009 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_SUPPORTED = 0xFF00000a # macro
# NV_VGPU_MSG_RESULT__RPC = 0xFF100009 : 0xFF100000 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_FUNCTION = 0xFF100001 # macro
NV_VGPU_MSG_RESULT_RPC_INVALID_MESSAGE_FORMAT = 0xFF100002 # macro
NV_VGPU_MSG_RESULT_RPC_HANDLE_NOT_FOUND = 0xFF100003 # macro
NV_VGPU_MSG_RESULT_RPC_HANDLE_EXISTS = 0xFF100004 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_RM_ERROR = 0xFF100005 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_VMIOP_ERROR = 0xFF100006 # macro
NV_VGPU_MSG_RESULT_RPC_RESERVED_HANDLE = 0xFF100007 # macro
NV_VGPU_MSG_RESULT_RPC_CUDA_PROFILING_DISABLED = 0xFF100008 # macro
NV_VGPU_MSG_RESULT_RPC_API_CONTROL_NOT_SUPPORTED = 0xFF100009 # macro
NV_VGPU_MSG_RESULT_RPC_PENDING = 0xFFFFFFFF # macro
NV_VGPU_MSG_UNION_INIT = 0x00000000 # macro
NV_VGPU_PTEDESC_INIT = 0x00000000 # macro
NV_VGPU_PTEDESC__PROD = 0x00000000 # macro
NV_VGPU_PTEDESC_IDR_NONE = 0x00000000 # macro
NV_VGPU_PTEDESC_IDR_SINGLE = 0x00000001 # macro
NV_VGPU_PTEDESC_IDR_DOUBLE = 0x00000002 # macro
NV_VGPU_PTEDESC_IDR_TRIPLE = 0x00000003 # macro
NV_VGPU_PTE_PAGE_SIZE = 0x1000 # macro
NV_VGPU_PTE_SIZE = 4 # macro
NV_VGPU_PTE_INDEX_SHIFT = 10 # macro
NV_VGPU_PTE_INDEX_MASK = 0x3FF # macro
NV_VGPU_PTE_64_PAGE_SIZE = 0x1000 # macro
NV_VGPU_PTE_64_SIZE = 8 # macro
NV_VGPU_PTE_64_INDEX_SHIFT = 9 # macro
NV_VGPU_PTE_64_INDEX_MASK = 0x1FF # macro
NV_VGPU_LOG_LEVEL_FATAL = 0x00000000 # macro
NV_VGPU_LOG_LEVEL_ERROR = 0x00000001 # macro
NV_VGPU_LOG_LEVEL_NOTICE = 0x00000002 # macro
NV_VGPU_LOG_LEVEL_STATUS = 0x00000003 # macro
NV_VGPU_LOG_LEVEL_DEBUG = 0x00000004 # macro
VGPU_RPC_GET_P2P_CAPS_V2_MAX_GPUS_SQUARED_PER_RPC = 512 # macro
GR_MAX_RPC_CTX_BUFFER_COUNT = 32 # macro
VGPU_RPC_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PER_RPC_v21_06 = 80 # macro

# values for enumeration 'c__EA_RPC_GR_BUFFER_TYPE'
c__EA_RPC_GR_BUFFER_TYPE__enumvalues = {
    0: 'RPC_GR_BUFFER_TYPE_GRAPHICS',
    1: 'RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL',
    2: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM',
    3: 'RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT',
    4: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH',
    5: 'RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB',
    6: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL',
    7: 'RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB',
    8: 'RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL',
    9: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL',
    10: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK',
    11: 'RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT',
    12: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP',
    13: 'RPC_GR_BUFFER_TYPE_GRAPHICS_MAX',
}
RPC_GR_BUFFER_TYPE_GRAPHICS = 0
RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL = 1
RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM = 2
RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT = 3
RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH = 4
RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB = 5
RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL = 6
RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB = 7
RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL = 8
RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL = 9
RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK = 10
RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT = 11
RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP = 12
RPC_GR_BUFFER_TYPE_GRAPHICS_MAX = 13
c__EA_RPC_GR_BUFFER_TYPE = ctypes.c_uint32 # enum
RPC_GR_BUFFER_TYPE = c__EA_RPC_GR_BUFFER_TYPE
RPC_GR_BUFFER_TYPE__enumvalues = c__EA_RPC_GR_BUFFER_TYPE__enumvalues

# values for enumeration 'c__EA_FECS_ERROR_EVENT_TYPE'
c__EA_FECS_ERROR_EVENT_TYPE__enumvalues = {
    0: 'FECS_ERROR_EVENT_TYPE_NONE',
    1: 'FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED',
    2: 'FECS_ERROR_EVENT_TYPE_BUFFER_FULL',
    3: 'FECS_ERROR_EVENT_TYPE_MAX',
}
FECS_ERROR_EVENT_TYPE_NONE = 0
FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED = 1
FECS_ERROR_EVENT_TYPE_BUFFER_FULL = 2
FECS_ERROR_EVENT_TYPE_MAX = 3
c__EA_FECS_ERROR_EVENT_TYPE = ctypes.c_uint32 # enum
FECS_ERROR_EVENT_TYPE = c__EA_FECS_ERROR_EVENT_TYPE
FECS_ERROR_EVENT_TYPE__enumvalues = c__EA_FECS_ERROR_EVENT_TYPE__enumvalues

# values for enumeration 'c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE'
c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues = {
    0: 'NV_RPC_UPDATE_PDE_BAR_1',
    1: 'NV_RPC_UPDATE_PDE_BAR_2',
    2: 'NV_RPC_UPDATE_PDE_BAR_INVALID',
}
NV_RPC_UPDATE_PDE_BAR_1 = 0
NV_RPC_UPDATE_PDE_BAR_2 = 1
NV_RPC_UPDATE_PDE_BAR_INVALID = 2
c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE = ctypes.c_uint32 # enum
NV_RPC_UPDATE_PDE_BAR_TYPE = c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE
NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues = c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues
class struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS(Structure):
    pass

struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS._pack_ = 1 # source:False
struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS._fields_ = [
    ('headIndex', ctypes.c_uint32),
    ('maxHResolution', ctypes.c_uint32),
    ('maxVResolution', ctypes.c_uint32),
]

VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS = struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS
class struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS(Structure):
    pass

struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS._pack_ = 1 # source:False
struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS._fields_ = [
    ('numHeads', ctypes.c_uint32),
    ('maxNumHeads', ctypes.c_uint32),
]

VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS = struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS

# values for enumeration 'c__EA_GPU_RECOVERY_EVENT_TYPE'
c__EA_GPU_RECOVERY_EVENT_TYPE__enumvalues = {
    0: 'GPU_RECOVERY_EVENT_TYPE_REFRESH',
    1: 'GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P',
    2: 'GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT',
}
GPU_RECOVERY_EVENT_TYPE_REFRESH = 0
GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P = 1
GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT = 2
c__EA_GPU_RECOVERY_EVENT_TYPE = ctypes.c_uint32 # enum
GPU_RECOVERY_EVENT_TYPE = c__EA_GPU_RECOVERY_EVENT_TYPE
GPU_RECOVERY_EVENT_TYPE__enumvalues = c__EA_GPU_RECOVERY_EVENT_TYPE__enumvalues
class struct_GSP_MSG_QUEUE_ELEMENT(Structure):
    pass

struct_GSP_MSG_QUEUE_ELEMENT._pack_ = 1 # source:False
struct_GSP_MSG_QUEUE_ELEMENT._fields_ = [
    ('authTagBuffer', ctypes.c_ubyte * 16),
    ('aadBuffer', ctypes.c_ubyte * 16),
    ('checkSum', ctypes.c_uint32),
    ('seqNum', ctypes.c_uint32),
    ('elemCount', ctypes.c_uint32),
    ('padding', ctypes.c_uint32),
]

GSP_MSG_QUEUE_ELEMENT = struct_GSP_MSG_QUEUE_ELEMENT
class union_rpc_message_rpc_union_field_v03_00(Union):
    pass

union_rpc_message_rpc_union_field_v03_00._pack_ = 1 # source:False
union_rpc_message_rpc_union_field_v03_00._fields_ = [
    ('spare', ctypes.c_uint32),
    ('cpuRmGfid', ctypes.c_uint32),
]

rpc_message_rpc_union_field_v03_00 = union_rpc_message_rpc_union_field_v03_00
rpc_message_rpc_union_field_v = union_rpc_message_rpc_union_field_v03_00
class struct_rpc_message_header_v03_00(Structure):
    pass

struct_rpc_message_header_v03_00._pack_ = 1 # source:False
struct_rpc_message_header_v03_00._fields_ = [
    ('header_version', ctypes.c_uint32),
    ('signature', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
    ('function', ctypes.c_uint32),
    ('rpc_result', ctypes.c_uint32),
    ('rpc_result_private', ctypes.c_uint32),
    ('sequence', ctypes.c_uint32),
    ('u', rpc_message_rpc_union_field_v),
]

rpc_message_header_v03_00 = struct_rpc_message_header_v03_00
rpc_message_header_v = struct_rpc_message_header_v03_00
GSP_STATIC_CONFIG_H = True # macro
MAX_DSM_SUPPORTED_FUNCS_RTN_LEN = 8 # macro
NV_ACPI_GENERIC_FUNC_COUNT = 8 # macro
REGISTRY_TABLE_ENTRY_TYPE_UNKNOWN = 0 # macro
REGISTRY_TABLE_ENTRY_TYPE_DWORD = 1 # macro
REGISTRY_TABLE_ENTRY_TYPE_BINARY = 2 # macro
REGISTRY_TABLE_ENTRY_TYPE_STRING = 3 # macro
MAX_GROUP_COUNT = 2 # macro
RM_ENGINE_TYPE_COPY_SIZE = 20 # macro
RM_ENGINE_TYPE_NVENC_SIZE = 4 # macro
RM_ENGINE_TYPE_NVJPEG_SIZE = 8 # macro
RM_ENGINE_TYPE_NVDEC_SIZE = 8 # macro
RM_ENGINE_TYPE_OFA_SIZE = 2 # macro
RM_ENGINE_TYPE_GR_SIZE = 8 # macro
NVGPU_ENGINE_CAPS_MASK_BITS = 32 # macro
# def NVGPU_GET_ENGINE_CAPS_MASK(caps, id):  # macro
#    return (caps[(id)/32]&NVBIT((id)%32))
# def NVGPU_SET_ENGINE_CAPS_MASK(caps, id):  # macro
#    return (caps[(id)/32]|=NVBIT((id)%32))
class struct_PACKED_REGISTRY_ENTRY(Structure):
    pass

struct_PACKED_REGISTRY_ENTRY._pack_ = 1 # source:False
struct_PACKED_REGISTRY_ENTRY._fields_ = [
    ('nameOffset', ctypes.c_uint32),
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('data', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

PACKED_REGISTRY_ENTRY = struct_PACKED_REGISTRY_ENTRY
class struct_PACKED_REGISTRY_TABLE(Structure):
    pass

struct_PACKED_REGISTRY_TABLE._pack_ = 1 # source:False
struct_PACKED_REGISTRY_TABLE._fields_ = [
    ('size', ctypes.c_uint32),
    ('numEntries', ctypes.c_uint32),
]

PACKED_REGISTRY_TABLE = struct_PACKED_REGISTRY_TABLE

# values for enumeration 'c__EA_DISPMUXSTATE'
c__EA_DISPMUXSTATE__enumvalues = {
    0: 'dispMuxState_None',
    1: 'dispMuxState_IntegratedGPU',
    2: 'dispMuxState_DiscreteGPU',
}
dispMuxState_None = 0
dispMuxState_IntegratedGPU = 1
dispMuxState_DiscreteGPU = 2
c__EA_DISPMUXSTATE = ctypes.c_uint32 # enum
DISPMUXSTATE = c__EA_DISPMUXSTATE
DISPMUXSTATE__enumvalues = c__EA_DISPMUXSTATE__enumvalues
class struct_c__SA_ACPI_DSM_CACHE(Structure):
    pass

struct_c__SA_ACPI_DSM_CACHE._pack_ = 1 # source:False
struct_c__SA_ACPI_DSM_CACHE._fields_ = [
    ('suppFuncStatus', ctypes.c_uint32),
    ('suppFuncs', ctypes.c_ubyte * 8),
    ('suppFuncsLen', ctypes.c_uint32),
    ('bArg3isInteger', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('callbackStatus', ctypes.c_uint32),
    ('callback', ctypes.c_uint32),
]

ACPI_DSM_CACHE = struct_c__SA_ACPI_DSM_CACHE
class struct_c__SA_ACPI_DATA(Structure):
    pass


# values for enumeration '_ACPI_DSM_FUNCTION'
_ACPI_DSM_FUNCTION__enumvalues = {
    0: 'ACPI_DSM_FUNCTION_NBSI',
    1: 'ACPI_DSM_FUNCTION_NVHG',
    2: 'ACPI_DSM_FUNCTION_MXM',
    3: 'ACPI_DSM_FUNCTION_NBCI',
    4: 'ACPI_DSM_FUNCTION_NVOP',
    5: 'ACPI_DSM_FUNCTION_PCFG',
    6: 'ACPI_DSM_FUNCTION_GPS_2X',
    7: 'ACPI_DSM_FUNCTION_JT',
    8: 'ACPI_DSM_FUNCTION_PEX',
    9: 'ACPI_DSM_FUNCTION_NVPCF_2X',
    10: 'ACPI_DSM_FUNCTION_GPS',
    11: 'ACPI_DSM_FUNCTION_NVPCF',
    12: 'ACPI_DSM_FUNCTION_COUNT',
    13: 'ACPI_DSM_FUNCTION_CURRENT',
    255: 'ACPI_DSM_FUNCTION_INVALID',
}
ACPI_DSM_FUNCTION_NBSI = 0
ACPI_DSM_FUNCTION_NVHG = 1
ACPI_DSM_FUNCTION_MXM = 2
ACPI_DSM_FUNCTION_NBCI = 3
ACPI_DSM_FUNCTION_NVOP = 4
ACPI_DSM_FUNCTION_PCFG = 5
ACPI_DSM_FUNCTION_GPS_2X = 6
ACPI_DSM_FUNCTION_JT = 7
ACPI_DSM_FUNCTION_PEX = 8
ACPI_DSM_FUNCTION_NVPCF_2X = 9
ACPI_DSM_FUNCTION_GPS = 10
ACPI_DSM_FUNCTION_NVPCF = 11
ACPI_DSM_FUNCTION_COUNT = 12
ACPI_DSM_FUNCTION_CURRENT = 13
ACPI_DSM_FUNCTION_INVALID = 255
_ACPI_DSM_FUNCTION = ctypes.c_uint32 # enum
struct_c__SA_ACPI_DATA._pack_ = 1 # source:False
struct_c__SA_ACPI_DATA._fields_ = [
    ('dsm', struct_c__SA_ACPI_DSM_CACHE * 12),
    ('dispStatusHotplugFunc', _ACPI_DSM_FUNCTION),
    ('dispStatusConfigFunc', _ACPI_DSM_FUNCTION),
    ('perfPostPowerStateFunc', _ACPI_DSM_FUNCTION),
    ('stereo3dStateActiveFunc', _ACPI_DSM_FUNCTION),
    ('dsmPlatCapsCache', ctypes.c_uint32 * 12),
    ('MDTLFeatureSupport', ctypes.c_uint32),
    ('dsmCurrentFunc', _ACPI_DSM_FUNCTION * 8),
    ('dsmCurrentSubFunc', ctypes.c_uint32 * 8),
    ('dsmCurrentFuncSupport', ctypes.c_uint32),
]

ACPI_DATA = struct_c__SA_ACPI_DATA
class struct_DOD_METHOD_DATA(Structure):
    pass

struct_DOD_METHOD_DATA._pack_ = 1 # source:False
struct_DOD_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('acpiIdListLen', ctypes.c_uint32),
    ('acpiIdList', ctypes.c_uint32 * 16),
]

DOD_METHOD_DATA = struct_DOD_METHOD_DATA
class struct_JT_METHOD_DATA(Structure):
    pass

struct_JT_METHOD_DATA._pack_ = 1 # source:False
struct_JT_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('jtCaps', ctypes.c_uint32),
    ('jtRevId', ctypes.c_uint16),
    ('bSBIOSCaps', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

JT_METHOD_DATA = struct_JT_METHOD_DATA
class struct_MUX_METHOD_DATA_ELEMENT(Structure):
    pass

struct_MUX_METHOD_DATA_ELEMENT._pack_ = 1 # source:False
struct_MUX_METHOD_DATA_ELEMENT._fields_ = [
    ('acpiId', ctypes.c_uint32),
    ('mode', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

MUX_METHOD_DATA_ELEMENT = struct_MUX_METHOD_DATA_ELEMENT
class struct_MUX_METHOD_DATA(Structure):
    pass

struct_MUX_METHOD_DATA._pack_ = 1 # source:False
struct_MUX_METHOD_DATA._fields_ = [
    ('tableLen', ctypes.c_uint32),
    ('acpiIdMuxModeTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
    ('acpiIdMuxPartTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
    ('acpiIdMuxStateTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
]

MUX_METHOD_DATA = struct_MUX_METHOD_DATA
class struct_CAPS_METHOD_DATA(Structure):
    pass

struct_CAPS_METHOD_DATA._pack_ = 1 # source:False
struct_CAPS_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('optimusCaps', ctypes.c_uint32),
]

CAPS_METHOD_DATA = struct_CAPS_METHOD_DATA
class struct_ACPI_METHOD_DATA(Structure):
    pass

struct_ACPI_METHOD_DATA._pack_ = 1 # source:False
struct_ACPI_METHOD_DATA._fields_ = [
    ('bValid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('dodMethodData', DOD_METHOD_DATA),
    ('jtMethodData', JT_METHOD_DATA),
    ('muxMethodData', MUX_METHOD_DATA),
    ('capsMethodData', CAPS_METHOD_DATA),
]

ACPI_METHOD_DATA = struct_ACPI_METHOD_DATA

# values for enumeration 'c__EA_RM_ENGINE_TYPE'
c__EA_RM_ENGINE_TYPE__enumvalues = {
    0: 'RM_ENGINE_TYPE_NULL',
    1: 'RM_ENGINE_TYPE_GR0',
    2: 'RM_ENGINE_TYPE_GR1',
    3: 'RM_ENGINE_TYPE_GR2',
    4: 'RM_ENGINE_TYPE_GR3',
    5: 'RM_ENGINE_TYPE_GR4',
    6: 'RM_ENGINE_TYPE_GR5',
    7: 'RM_ENGINE_TYPE_GR6',
    8: 'RM_ENGINE_TYPE_GR7',
    9: 'RM_ENGINE_TYPE_COPY0',
    10: 'RM_ENGINE_TYPE_COPY1',
    11: 'RM_ENGINE_TYPE_COPY2',
    12: 'RM_ENGINE_TYPE_COPY3',
    13: 'RM_ENGINE_TYPE_COPY4',
    14: 'RM_ENGINE_TYPE_COPY5',
    15: 'RM_ENGINE_TYPE_COPY6',
    16: 'RM_ENGINE_TYPE_COPY7',
    17: 'RM_ENGINE_TYPE_COPY8',
    18: 'RM_ENGINE_TYPE_COPY9',
    19: 'RM_ENGINE_TYPE_COPY10',
    20: 'RM_ENGINE_TYPE_COPY11',
    21: 'RM_ENGINE_TYPE_COPY12',
    22: 'RM_ENGINE_TYPE_COPY13',
    23: 'RM_ENGINE_TYPE_COPY14',
    24: 'RM_ENGINE_TYPE_COPY15',
    25: 'RM_ENGINE_TYPE_COPY16',
    26: 'RM_ENGINE_TYPE_COPY17',
    27: 'RM_ENGINE_TYPE_COPY18',
    28: 'RM_ENGINE_TYPE_COPY19',
    29: 'RM_ENGINE_TYPE_NVDEC0',
    30: 'RM_ENGINE_TYPE_NVDEC1',
    31: 'RM_ENGINE_TYPE_NVDEC2',
    32: 'RM_ENGINE_TYPE_NVDEC3',
    33: 'RM_ENGINE_TYPE_NVDEC4',
    34: 'RM_ENGINE_TYPE_NVDEC5',
    35: 'RM_ENGINE_TYPE_NVDEC6',
    36: 'RM_ENGINE_TYPE_NVDEC7',
    37: 'RM_ENGINE_TYPE_NVENC0',
    38: 'RM_ENGINE_TYPE_NVENC1',
    39: 'RM_ENGINE_TYPE_NVENC2',
    40: 'RM_ENGINE_TYPE_NVENC3',
    41: 'RM_ENGINE_TYPE_VP',
    42: 'RM_ENGINE_TYPE_ME',
    43: 'RM_ENGINE_TYPE_PPP',
    44: 'RM_ENGINE_TYPE_MPEG',
    45: 'RM_ENGINE_TYPE_SW',
    46: 'RM_ENGINE_TYPE_TSEC',
    47: 'RM_ENGINE_TYPE_VIC',
    48: 'RM_ENGINE_TYPE_MP',
    49: 'RM_ENGINE_TYPE_SEC2',
    50: 'RM_ENGINE_TYPE_HOST',
    51: 'RM_ENGINE_TYPE_DPU',
    52: 'RM_ENGINE_TYPE_PMU',
    53: 'RM_ENGINE_TYPE_FBFLCN',
    54: 'RM_ENGINE_TYPE_NVJPEG0',
    55: 'RM_ENGINE_TYPE_NVJPEG1',
    56: 'RM_ENGINE_TYPE_NVJPEG2',
    57: 'RM_ENGINE_TYPE_NVJPEG3',
    58: 'RM_ENGINE_TYPE_NVJPEG4',
    59: 'RM_ENGINE_TYPE_NVJPEG5',
    60: 'RM_ENGINE_TYPE_NVJPEG6',
    61: 'RM_ENGINE_TYPE_NVJPEG7',
    62: 'RM_ENGINE_TYPE_OFA0',
    63: 'RM_ENGINE_TYPE_OFA1',
    64: 'RM_ENGINE_TYPE_RESERVED40',
    65: 'RM_ENGINE_TYPE_RESERVED41',
    66: 'RM_ENGINE_TYPE_RESERVED42',
    67: 'RM_ENGINE_TYPE_RESERVED43',
    68: 'RM_ENGINE_TYPE_RESERVED44',
    69: 'RM_ENGINE_TYPE_RESERVED45',
    70: 'RM_ENGINE_TYPE_RESERVED46',
    71: 'RM_ENGINE_TYPE_RESERVED47',
    72: 'RM_ENGINE_TYPE_RESERVED48',
    73: 'RM_ENGINE_TYPE_RESERVED49',
    74: 'RM_ENGINE_TYPE_RESERVED4a',
    75: 'RM_ENGINE_TYPE_RESERVED4b',
    76: 'RM_ENGINE_TYPE_RESERVED4c',
    77: 'RM_ENGINE_TYPE_RESERVED4d',
    78: 'RM_ENGINE_TYPE_RESERVED4e',
    79: 'RM_ENGINE_TYPE_RESERVED4f',
    80: 'RM_ENGINE_TYPE_RESERVED50',
    81: 'RM_ENGINE_TYPE_RESERVED51',
    82: 'RM_ENGINE_TYPE_RESERVED52',
    83: 'RM_ENGINE_TYPE_RESERVED53',
    84: 'RM_ENGINE_TYPE_LAST',
}
RM_ENGINE_TYPE_NULL = 0
RM_ENGINE_TYPE_GR0 = 1
RM_ENGINE_TYPE_GR1 = 2
RM_ENGINE_TYPE_GR2 = 3
RM_ENGINE_TYPE_GR3 = 4
RM_ENGINE_TYPE_GR4 = 5
RM_ENGINE_TYPE_GR5 = 6
RM_ENGINE_TYPE_GR6 = 7
RM_ENGINE_TYPE_GR7 = 8
RM_ENGINE_TYPE_COPY0 = 9
RM_ENGINE_TYPE_COPY1 = 10
RM_ENGINE_TYPE_COPY2 = 11
RM_ENGINE_TYPE_COPY3 = 12
RM_ENGINE_TYPE_COPY4 = 13
RM_ENGINE_TYPE_COPY5 = 14
RM_ENGINE_TYPE_COPY6 = 15
RM_ENGINE_TYPE_COPY7 = 16
RM_ENGINE_TYPE_COPY8 = 17
RM_ENGINE_TYPE_COPY9 = 18
RM_ENGINE_TYPE_COPY10 = 19
RM_ENGINE_TYPE_COPY11 = 20
RM_ENGINE_TYPE_COPY12 = 21
RM_ENGINE_TYPE_COPY13 = 22
RM_ENGINE_TYPE_COPY14 = 23
RM_ENGINE_TYPE_COPY15 = 24
RM_ENGINE_TYPE_COPY16 = 25
RM_ENGINE_TYPE_COPY17 = 26
RM_ENGINE_TYPE_COPY18 = 27
RM_ENGINE_TYPE_COPY19 = 28
RM_ENGINE_TYPE_NVDEC0 = 29
RM_ENGINE_TYPE_NVDEC1 = 30
RM_ENGINE_TYPE_NVDEC2 = 31
RM_ENGINE_TYPE_NVDEC3 = 32
RM_ENGINE_TYPE_NVDEC4 = 33
RM_ENGINE_TYPE_NVDEC5 = 34
RM_ENGINE_TYPE_NVDEC6 = 35
RM_ENGINE_TYPE_NVDEC7 = 36
RM_ENGINE_TYPE_NVENC0 = 37
RM_ENGINE_TYPE_NVENC1 = 38
RM_ENGINE_TYPE_NVENC2 = 39
RM_ENGINE_TYPE_NVENC3 = 40
RM_ENGINE_TYPE_VP = 41
RM_ENGINE_TYPE_ME = 42
RM_ENGINE_TYPE_PPP = 43
RM_ENGINE_TYPE_MPEG = 44
RM_ENGINE_TYPE_SW = 45
RM_ENGINE_TYPE_TSEC = 46
RM_ENGINE_TYPE_VIC = 47
RM_ENGINE_TYPE_MP = 48
RM_ENGINE_TYPE_SEC2 = 49
RM_ENGINE_TYPE_HOST = 50
RM_ENGINE_TYPE_DPU = 51
RM_ENGINE_TYPE_PMU = 52
RM_ENGINE_TYPE_FBFLCN = 53
RM_ENGINE_TYPE_NVJPEG0 = 54
RM_ENGINE_TYPE_NVJPEG1 = 55
RM_ENGINE_TYPE_NVJPEG2 = 56
RM_ENGINE_TYPE_NVJPEG3 = 57
RM_ENGINE_TYPE_NVJPEG4 = 58
RM_ENGINE_TYPE_NVJPEG5 = 59
RM_ENGINE_TYPE_NVJPEG6 = 60
RM_ENGINE_TYPE_NVJPEG7 = 61
RM_ENGINE_TYPE_OFA0 = 62
RM_ENGINE_TYPE_OFA1 = 63
RM_ENGINE_TYPE_RESERVED40 = 64
RM_ENGINE_TYPE_RESERVED41 = 65
RM_ENGINE_TYPE_RESERVED42 = 66
RM_ENGINE_TYPE_RESERVED43 = 67
RM_ENGINE_TYPE_RESERVED44 = 68
RM_ENGINE_TYPE_RESERVED45 = 69
RM_ENGINE_TYPE_RESERVED46 = 70
RM_ENGINE_TYPE_RESERVED47 = 71
RM_ENGINE_TYPE_RESERVED48 = 72
RM_ENGINE_TYPE_RESERVED49 = 73
RM_ENGINE_TYPE_RESERVED4a = 74
RM_ENGINE_TYPE_RESERVED4b = 75
RM_ENGINE_TYPE_RESERVED4c = 76
RM_ENGINE_TYPE_RESERVED4d = 77
RM_ENGINE_TYPE_RESERVED4e = 78
RM_ENGINE_TYPE_RESERVED4f = 79
RM_ENGINE_TYPE_RESERVED50 = 80
RM_ENGINE_TYPE_RESERVED51 = 81
RM_ENGINE_TYPE_RESERVED52 = 82
RM_ENGINE_TYPE_RESERVED53 = 83
RM_ENGINE_TYPE_LAST = 84
c__EA_RM_ENGINE_TYPE = ctypes.c_uint32 # enum
RM_ENGINE_TYPE_GRAPHICS = RM_ENGINE_TYPE_GR0 # macro
RM_ENGINE_TYPE_BSP = RM_ENGINE_TYPE_NVDEC0 # macro
RM_ENGINE_TYPE_MSENC = RM_ENGINE_TYPE_NVENC0 # macro
RM_ENGINE_TYPE_CIPHER = RM_ENGINE_TYPE_TSEC # macro
RM_ENGINE_TYPE_NVJPG = RM_ENGINE_TYPE_NVJPEG0 # macro
NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX = ((RM_ENGINE_TYPE_LAST-1)/32+1) # macro
RM_ENGINE_TYPE = c__EA_RM_ENGINE_TYPE
RM_ENGINE_TYPE__enumvalues = c__EA_RM_ENGINE_TYPE__enumvalues
class struct_c__SA_BUSINFO(Structure):
    pass

struct_c__SA_BUSINFO._pack_ = 1 # source:False
struct_c__SA_BUSINFO._fields_ = [
    ('deviceID', ctypes.c_uint16),
    ('vendorID', ctypes.c_uint16),
    ('subdeviceID', ctypes.c_uint16),
    ('subvendorID', ctypes.c_uint16),
    ('revisionID', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

BUSINFO = struct_c__SA_BUSINFO
class struct_GSP_VF_INFO(Structure):
    pass

struct_GSP_VF_INFO._pack_ = 1 # source:False
struct_GSP_VF_INFO._fields_ = [
    ('totalVFs', ctypes.c_uint32),
    ('firstVFOffset', ctypes.c_uint32),
    ('FirstVFBar0Address', ctypes.c_uint64),
    ('FirstVFBar1Address', ctypes.c_uint64),
    ('FirstVFBar2Address', ctypes.c_uint64),
    ('b64bitBar0', ctypes.c_ubyte),
    ('b64bitBar1', ctypes.c_ubyte),
    ('b64bitBar2', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 5),
]

GSP_VF_INFO = struct_GSP_VF_INFO
class struct_c__SA_GSP_PCIE_CONFIG_REG(Structure):
    pass

struct_c__SA_GSP_PCIE_CONFIG_REG._pack_ = 1 # source:False
struct_c__SA_GSP_PCIE_CONFIG_REG._fields_ = [
    ('linkCap', ctypes.c_uint32),
]

GSP_PCIE_CONFIG_REG = struct_c__SA_GSP_PCIE_CONFIG_REG
class struct_c__SA_EcidManufacturingInfo(Structure):
    pass

struct_c__SA_EcidManufacturingInfo._pack_ = 1 # source:False
struct_c__SA_EcidManufacturingInfo._fields_ = [
    ('ecidLow', ctypes.c_uint32),
    ('ecidHigh', ctypes.c_uint32),
    ('ecidExtended', ctypes.c_uint32),
]

EcidManufacturingInfo = struct_c__SA_EcidManufacturingInfo
class struct_c__SA_FW_WPR_LAYOUT_OFFSET(Structure):
    pass

struct_c__SA_FW_WPR_LAYOUT_OFFSET._pack_ = 1 # source:False
struct_c__SA_FW_WPR_LAYOUT_OFFSET._fields_ = [
    ('nonWprHeapOffset', ctypes.c_uint64),
    ('frtsOffset', ctypes.c_uint64),
]

FW_WPR_LAYOUT_OFFSET = struct_c__SA_FW_WPR_LAYOUT_OFFSET
class struct_GspStaticConfigInfo_t(Structure):
    pass

class struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(Structure):
    pass

struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS._fields_ = [
    ('index', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 256),
]

class struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS(Structure):
    pass

struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS._fields_ = [
    ('BoardID', ctypes.c_uint32),
    ('chipSKU', ctypes.c_char * 9),
    ('chipSKUMod', ctypes.c_char * 5),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('skuConfigVersion', ctypes.c_uint32),
    ('project', ctypes.c_char * 5),
    ('projectSKU', ctypes.c_char * 5),
    ('CDP', ctypes.c_char * 6),
    ('projectSKUMod', ctypes.c_char * 2),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('businessCycle', ctypes.c_uint32),
]

class struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS(Structure):
    pass

class struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO(Structure):
    pass

struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO._fields_ = [
    ('base', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
    ('performance', ctypes.c_uint32),
    ('supportCompressed', ctypes.c_ubyte),
    ('supportISO', ctypes.c_ubyte),
    ('bProtected', ctypes.c_ubyte),
    ('blackList', ctypes.c_ubyte * 17),
]

struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS._fields_ = [
    ('numFBRegions', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fbRegion', struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO * 16),
]

class struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS(Structure):
    pass

struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS._pack_ = 1 # source:False
struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS._fields_ = [
    ('totalVFs', ctypes.c_uint32),
    ('firstVfOffset', ctypes.c_uint32),
    ('vfFeatureMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('FirstVFBar0Address', ctypes.c_uint64),
    ('FirstVFBar1Address', ctypes.c_uint64),
    ('FirstVFBar2Address', ctypes.c_uint64),
    ('bar0Size', ctypes.c_uint64),
    ('bar1Size', ctypes.c_uint64),
    ('bar2Size', ctypes.c_uint64),
    ('b64bitBar0', ctypes.c_ubyte),
    ('b64bitBar1', ctypes.c_ubyte),
    ('b64bitBar2', ctypes.c_ubyte),
    ('bSriovEnabled', ctypes.c_ubyte),
    ('bSriovHeavyEnabled', ctypes.c_ubyte),
    ('bEmulateVFBar0TlbInvalidationRegister', ctypes.c_ubyte),
    ('bClientRmAllocatedCtxBuffer', ctypes.c_ubyte),
    ('bNonPowerOf2ChannelCountSupported', ctypes.c_ubyte),
    ('bVfResizableBAR1Supported', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

struct_GspStaticConfigInfo_t._pack_ = 1 # source:False
struct_GspStaticConfigInfo_t._fields_ = [
    ('grCapsBits', ctypes.c_ubyte * 23),
    ('PADDING_0', ctypes.c_ubyte),
    ('gidInfo', struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS),
    ('SKUInfo', struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('fbRegionInfoParams', struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS),
    ('sriovCaps', struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS),
    ('sriovMaxGfid', ctypes.c_uint32),
    ('engineCaps', ctypes.c_uint32 * 3),
    ('poisonFuseEnabled', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 7),
    ('fb_length', ctypes.c_uint64),
    ('fbio_mask', ctypes.c_uint64),
    ('fb_bus_width', ctypes.c_uint32),
    ('fb_ram_type', ctypes.c_uint32),
    ('fbp_mask', ctypes.c_uint64),
    ('l2_cache_size', ctypes.c_uint32),
    ('gpuNameString', ctypes.c_ubyte * 64),
    ('gpuShortNameString', ctypes.c_ubyte * 64),
    ('gpuNameString_Unicode', ctypes.c_uint16 * 64),
    ('bGpuInternalSku', ctypes.c_ubyte),
    ('bIsQuadroGeneric', ctypes.c_ubyte),
    ('bIsQuadroAd', ctypes.c_ubyte),
    ('bIsNvidiaNvs', ctypes.c_ubyte),
    ('bIsVgx', ctypes.c_ubyte),
    ('bGeforceSmb', ctypes.c_ubyte),
    ('bIsTitan', ctypes.c_ubyte),
    ('bIsTesla', ctypes.c_ubyte),
    ('bIsMobile', ctypes.c_ubyte),
    ('bIsGc6Rtd3Allowed', ctypes.c_ubyte),
    ('bIsGc8Rtd3Allowed', ctypes.c_ubyte),
    ('bIsGcOffRtd3Allowed', ctypes.c_ubyte),
    ('bIsGcoffLegacyAllowed', ctypes.c_ubyte),
    ('bIsMigSupported', ctypes.c_ubyte),
    ('RTD3GC6TotalBoardPower', ctypes.c_uint16),
    ('RTD3GC6PerstDelay', ctypes.c_uint16),
    ('PADDING_3', ctypes.c_ubyte * 2),
    ('bar1PdeBase', ctypes.c_uint64),
    ('bar2PdeBase', ctypes.c_uint64),
    ('bVbiosValid', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 3),
    ('vbiosSubVendor', ctypes.c_uint32),
    ('vbiosSubDevice', ctypes.c_uint32),
    ('bPageRetirementSupported', ctypes.c_ubyte),
    ('bSplitVasBetweenServerClientRm', ctypes.c_ubyte),
    ('bClRootportNeedsNosnoopWAR', ctypes.c_ubyte),
    ('PADDING_5', ctypes.c_ubyte),
    ('displaylessMaxHeads', VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS),
    ('displaylessMaxResolution', VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS),
    ('PADDING_6', ctypes.c_ubyte * 4),
    ('displaylessMaxPixels', ctypes.c_uint64),
    ('hInternalClient', ctypes.c_uint32),
    ('hInternalDevice', ctypes.c_uint32),
    ('hInternalSubdevice', ctypes.c_uint32),
    ('bSelfHostedMode', ctypes.c_ubyte),
    ('bAtsSupported', ctypes.c_ubyte),
    ('bIsGpuUefi', ctypes.c_ubyte),
    ('bIsEfiInit', ctypes.c_ubyte),
    ('ecidInfo', struct_c__SA_EcidManufacturingInfo * 2),
    ('fwWprLayoutOffset', FW_WPR_LAYOUT_OFFSET),
]

GspStaticConfigInfo = struct_GspStaticConfigInfo_t
class struct_GspSystemInfo(Structure):
    pass

struct_GspSystemInfo._pack_ = 1 # source:False
struct_GspSystemInfo._fields_ = [
    ('gpuPhysAddr', ctypes.c_uint64),
    ('gpuPhysFbAddr', ctypes.c_uint64),
    ('gpuPhysInstAddr', ctypes.c_uint64),
    ('gpuPhysIoAddr', ctypes.c_uint64),
    ('nvDomainBusDeviceFunc', ctypes.c_uint64),
    ('simAccessBufPhysAddr', ctypes.c_uint64),
    ('notifyOpSharedSurfacePhysAddr', ctypes.c_uint64),
    ('pcieAtomicsOpMask', ctypes.c_uint64),
    ('consoleMemSize', ctypes.c_uint64),
    ('maxUserVa', ctypes.c_uint64),
    ('pciConfigMirrorBase', ctypes.c_uint32),
    ('pciConfigMirrorSize', ctypes.c_uint32),
    ('PCIDeviceID', ctypes.c_uint32),
    ('PCISubDeviceID', ctypes.c_uint32),
    ('PCIRevisionID', ctypes.c_uint32),
    ('pcieAtomicsCplDeviceCapMask', ctypes.c_uint32),
    ('oorArch', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('clPdbProperties', ctypes.c_uint64),
    ('Chipset', ctypes.c_uint32),
    ('bGpuBehindBridge', ctypes.c_ubyte),
    ('bFlrSupported', ctypes.c_ubyte),
    ('b64bBar0Supported', ctypes.c_ubyte),
    ('bMnocAvailable', ctypes.c_ubyte),
    ('chipsetL1ssEnable', ctypes.c_uint32),
    ('bUpstreamL0sUnsupported', ctypes.c_ubyte),
    ('bUpstreamL1Unsupported', ctypes.c_ubyte),
    ('bUpstreamL1PorSupported', ctypes.c_ubyte),
    ('bUpstreamL1PorMobileOnly', ctypes.c_ubyte),
    ('bSystemHasMux', ctypes.c_ubyte),
    ('upstreamAddressValid', ctypes.c_ubyte),
    ('FHBBusInfo', BUSINFO),
    ('chipsetIDInfo', BUSINFO),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('acpiMethodData', ACPI_METHOD_DATA),
    ('hypervisorType', ctypes.c_uint32),
    ('bIsPassthru', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 7),
    ('sysTimerOffsetNs', ctypes.c_uint64),
    ('gspVFInfo', GSP_VF_INFO),
    ('bIsPrimary', ctypes.c_ubyte),
    ('isGridBuild', ctypes.c_ubyte),
    ('PADDING_3', ctypes.c_ubyte * 2),
    ('pcieConfigReg', GSP_PCIE_CONFIG_REG),
    ('gridBuildCsp', ctypes.c_uint32),
    ('bPreserveVideoMemoryAllocations', ctypes.c_ubyte),
    ('bTdrEventSupported', ctypes.c_ubyte),
    ('bFeatureStretchVblankCapable', ctypes.c_ubyte),
    ('bEnableDynamicGranularityPageArrays', ctypes.c_ubyte),
    ('bClockBoostSupported', ctypes.c_ubyte),
    ('bRouteDispIntrsToCPU', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 6),
    ('hostPageSize', ctypes.c_uint64),
]

GspSystemInfo = struct_GspSystemInfo
__all__ = \
    ['ACPI_DATA', 'ACPI_DSM_CACHE', 'ACPI_DSM_FUNCTION_COUNT',
    'ACPI_DSM_FUNCTION_CURRENT', 'ACPI_DSM_FUNCTION_GPS',
    'ACPI_DSM_FUNCTION_GPS_2X', 'ACPI_DSM_FUNCTION_INVALID',
    'ACPI_DSM_FUNCTION_JT', 'ACPI_DSM_FUNCTION_MXM',
    'ACPI_DSM_FUNCTION_NBCI', 'ACPI_DSM_FUNCTION_NBSI',
    'ACPI_DSM_FUNCTION_NVHG', 'ACPI_DSM_FUNCTION_NVOP',
    'ACPI_DSM_FUNCTION_NVPCF', 'ACPI_DSM_FUNCTION_NVPCF_2X',
    'ACPI_DSM_FUNCTION_PCFG', 'ACPI_DSM_FUNCTION_PEX',
    'ACPI_METHOD_DATA', 'BUSINFO', 'CAPS_METHOD_DATA', 'DISPMUXSTATE',
    'DISPMUXSTATE__enumvalues', 'DOD_METHOD_DATA',
    'EcidManufacturingInfo', 'FECS_ERROR_EVENT_TYPE',
    'FECS_ERROR_EVENT_TYPE_BUFFER_FULL',
    'FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED',
    'FECS_ERROR_EVENT_TYPE_MAX', 'FECS_ERROR_EVENT_TYPE_NONE',
    'FECS_ERROR_EVENT_TYPE__enumvalues', 'FW_WPR_LAYOUT_OFFSET',
    'GPU_RECOVERY_EVENT_TYPE',
    'GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P',
    'GPU_RECOVERY_EVENT_TYPE_REFRESH',
    'GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT',
    'GPU_RECOVERY_EVENT_TYPE__enumvalues',
    'GR_MAX_RPC_CTX_BUFFER_COUNT', 'GSPIFPUB_H',
    'GSP_ACR_BOOT_GSP_RM_PARAMS', 'GSP_ARGUMENTS_CACHED',
    'GSP_DMA_TARGET', 'GSP_DMA_TARGET_COHERENT_SYSTEM',
    'GSP_DMA_TARGET_COUNT', 'GSP_DMA_TARGET_LOCAL_FB',
    'GSP_DMA_TARGET_NONCOHERENT_SYSTEM', 'GSP_DMA_TARGET__enumvalues',
    'GSP_FMC_BOOT_PARAMS', 'GSP_FMC_INIT_PARAMS',
    'GSP_FW_HEAP_FREE_LIST_MAGIC', 'GSP_FW_SR_META_H_',
    'GSP_FW_SR_META_INTERNAL_SIZE', 'GSP_FW_SR_META_MAGIC',
    'GSP_FW_SR_META_REVISION', 'GSP_FW_WPR_HEAP_FREE_REGION_COUNT',
    'GSP_FW_WPR_META_H_', 'GSP_FW_WPR_META_MAGIC',
    'GSP_FW_WPR_META_REVISION', 'GSP_FW_WPR_META_VERIFIED',
    'GSP_INIT_ARGS_H', 'GSP_MSG_QUEUE_ELEMENT', 'GSP_PCIE_CONFIG_REG',
    'GSP_RM_PARAMS', 'GSP_SPDM_PARAMS', 'GSP_SR_INIT_ARGUMENTS',
    'GSP_STATIC_CONFIG_H', 'GSP_VF_INFO', 'GspFwHeapFreeList',
    'GspFwHeapFreeRegion', 'GspFwSRMeta', 'GspFwWprMeta',
    'GspStaticConfigInfo', 'GspSystemInfo', 'JT_METHOD_DATA',
    'KERN_FSP_COT_PAYLOAD_H', 'LIBOS_INIT_H_',
    'LIBOS_MEMORY_REGION_CONTIGUOUS',
    'LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX',
    'LIBOS_MEMORY_REGION_LOC_FB', 'LIBOS_MEMORY_REGION_LOC_NONE',
    'LIBOS_MEMORY_REGION_LOC_SYSMEM', 'LIBOS_MEMORY_REGION_NONE',
    'LIBOS_MEMORY_REGION_RADIX3',
    'LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2',
    'LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE', 'LibosAddress',
    'LibosMemoryRegionInitArgument', 'LibosMemoryRegionKind',
    'LibosMemoryRegionKind__enumvalues', 'LibosMemoryRegionLoc',
    'LibosMemoryRegionLoc__enumvalues',
    'MAX_DSM_SUPPORTED_FUNCS_RTN_LEN', 'MAX_GPC_COUNT',
    'MAX_GROUP_COUNT', 'MCTP_HEADER', 'MESSAGE_QUEUE_INIT_ARGUMENTS',
    'MSGQ_PRIV_H', 'MSGQ_VERSION', 'MUX_METHOD_DATA',
    'MUX_METHOD_DATA_ELEMENT', 'NVDM_PAYLOAD_COT',
    'NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX', 'NVGPU_ENGINE_CAPS_MASK_BITS',
    'NV_ACPI_GENERIC_FUNC_COUNT', 'NV_RPC_UPDATE_PDE_BAR_1',
    'NV_RPC_UPDATE_PDE_BAR_2', 'NV_RPC_UPDATE_PDE_BAR_INVALID',
    'NV_RPC_UPDATE_PDE_BAR_TYPE',
    'NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues',
    'NV_VGPU_LOG_LEVEL_DEBUG', 'NV_VGPU_LOG_LEVEL_ERROR',
    'NV_VGPU_LOG_LEVEL_FATAL', 'NV_VGPU_LOG_LEVEL_NOTICE',
    'NV_VGPU_LOG_LEVEL_STATUS',
    'NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT',
    'NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT',
    'NV_VGPU_MSG_RESULT_RPC_API_CONTROL_NOT_SUPPORTED',
    'NV_VGPU_MSG_RESULT_RPC_CUDA_PROFILING_DISABLED',
    'NV_VGPU_MSG_RESULT_RPC_HANDLE_EXISTS',
    'NV_VGPU_MSG_RESULT_RPC_HANDLE_NOT_FOUND',
    'NV_VGPU_MSG_RESULT_RPC_INVALID_MESSAGE_FORMAT',
    'NV_VGPU_MSG_RESULT_RPC_PENDING',
    'NV_VGPU_MSG_RESULT_RPC_RESERVED_HANDLE',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_FUNCTION',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_RM_ERROR',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_VMIOP_ERROR',
    'NV_VGPU_MSG_RESULT_VMIOP_ECC_MISMATCH',
    'NV_VGPU_MSG_RESULT_VMIOP_INVAL',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_ALLOWED_IN_CALLBACK',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_FOUND',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_SUPPORTED',
    'NV_VGPU_MSG_RESULT_VMIOP_NO_ADDRESS_SPACE',
    'NV_VGPU_MSG_RESULT_VMIOP_RANGE',
    'NV_VGPU_MSG_RESULT_VMIOP_READ_ONLY',
    'NV_VGPU_MSG_RESULT_VMIOP_RESOURCE',
    'NV_VGPU_MSG_RESULT_VMIOP_TIMEOUT', 'NV_VGPU_MSG_SIGNATURE_VALID',
    'NV_VGPU_MSG_UNION_INIT', 'NV_VGPU_PTEDESC_IDR_DOUBLE',
    'NV_VGPU_PTEDESC_IDR_NONE', 'NV_VGPU_PTEDESC_IDR_SINGLE',
    'NV_VGPU_PTEDESC_IDR_TRIPLE', 'NV_VGPU_PTEDESC_INIT',
    'NV_VGPU_PTEDESC__PROD', 'NV_VGPU_PTE_64_INDEX_MASK',
    'NV_VGPU_PTE_64_INDEX_SHIFT', 'NV_VGPU_PTE_64_PAGE_SIZE',
    'NV_VGPU_PTE_64_SIZE', 'NV_VGPU_PTE_INDEX_MASK',
    'NV_VGPU_PTE_INDEX_SHIFT', 'NV_VGPU_PTE_PAGE_SIZE',
    'NV_VGPU_PTE_SIZE', 'PACKED_REGISTRY_ENTRY',
    'PACKED_REGISTRY_TABLE', 'REGISTRY_TABLE_ENTRY_TYPE_BINARY',
    'REGISTRY_TABLE_ENTRY_TYPE_DWORD',
    'REGISTRY_TABLE_ENTRY_TYPE_STRING',
    'REGISTRY_TABLE_ENTRY_TYPE_UNKNOWN', 'RM_ENGINE_TYPE',
    'RM_ENGINE_TYPE_BSP', 'RM_ENGINE_TYPE_CIPHER',
    'RM_ENGINE_TYPE_COPY0', 'RM_ENGINE_TYPE_COPY1',
    'RM_ENGINE_TYPE_COPY10', 'RM_ENGINE_TYPE_COPY11',
    'RM_ENGINE_TYPE_COPY12', 'RM_ENGINE_TYPE_COPY13',
    'RM_ENGINE_TYPE_COPY14', 'RM_ENGINE_TYPE_COPY15',
    'RM_ENGINE_TYPE_COPY16', 'RM_ENGINE_TYPE_COPY17',
    'RM_ENGINE_TYPE_COPY18', 'RM_ENGINE_TYPE_COPY19',
    'RM_ENGINE_TYPE_COPY2', 'RM_ENGINE_TYPE_COPY3',
    'RM_ENGINE_TYPE_COPY4', 'RM_ENGINE_TYPE_COPY5',
    'RM_ENGINE_TYPE_COPY6', 'RM_ENGINE_TYPE_COPY7',
    'RM_ENGINE_TYPE_COPY8', 'RM_ENGINE_TYPE_COPY9',
    'RM_ENGINE_TYPE_COPY_SIZE', 'RM_ENGINE_TYPE_DPU',
    'RM_ENGINE_TYPE_FBFLCN', 'RM_ENGINE_TYPE_GR0',
    'RM_ENGINE_TYPE_GR1', 'RM_ENGINE_TYPE_GR2', 'RM_ENGINE_TYPE_GR3',
    'RM_ENGINE_TYPE_GR4', 'RM_ENGINE_TYPE_GR5', 'RM_ENGINE_TYPE_GR6',
    'RM_ENGINE_TYPE_GR7', 'RM_ENGINE_TYPE_GRAPHICS',
    'RM_ENGINE_TYPE_GR_SIZE', 'RM_ENGINE_TYPE_HOST',
    'RM_ENGINE_TYPE_LAST', 'RM_ENGINE_TYPE_ME', 'RM_ENGINE_TYPE_MP',
    'RM_ENGINE_TYPE_MPEG', 'RM_ENGINE_TYPE_MSENC',
    'RM_ENGINE_TYPE_NULL', 'RM_ENGINE_TYPE_NVDEC0',
    'RM_ENGINE_TYPE_NVDEC1', 'RM_ENGINE_TYPE_NVDEC2',
    'RM_ENGINE_TYPE_NVDEC3', 'RM_ENGINE_TYPE_NVDEC4',
    'RM_ENGINE_TYPE_NVDEC5', 'RM_ENGINE_TYPE_NVDEC6',
    'RM_ENGINE_TYPE_NVDEC7', 'RM_ENGINE_TYPE_NVDEC_SIZE',
    'RM_ENGINE_TYPE_NVENC0', 'RM_ENGINE_TYPE_NVENC1',
    'RM_ENGINE_TYPE_NVENC2', 'RM_ENGINE_TYPE_NVENC3',
    'RM_ENGINE_TYPE_NVENC_SIZE', 'RM_ENGINE_TYPE_NVJPEG0',
    'RM_ENGINE_TYPE_NVJPEG1', 'RM_ENGINE_TYPE_NVJPEG2',
    'RM_ENGINE_TYPE_NVJPEG3', 'RM_ENGINE_TYPE_NVJPEG4',
    'RM_ENGINE_TYPE_NVJPEG5', 'RM_ENGINE_TYPE_NVJPEG6',
    'RM_ENGINE_TYPE_NVJPEG7', 'RM_ENGINE_TYPE_NVJPEG_SIZE',
    'RM_ENGINE_TYPE_NVJPG', 'RM_ENGINE_TYPE_OFA0',
    'RM_ENGINE_TYPE_OFA1', 'RM_ENGINE_TYPE_OFA_SIZE',
    'RM_ENGINE_TYPE_PMU', 'RM_ENGINE_TYPE_PPP',
    'RM_ENGINE_TYPE_RESERVED40', 'RM_ENGINE_TYPE_RESERVED41',
    'RM_ENGINE_TYPE_RESERVED42', 'RM_ENGINE_TYPE_RESERVED43',
    'RM_ENGINE_TYPE_RESERVED44', 'RM_ENGINE_TYPE_RESERVED45',
    'RM_ENGINE_TYPE_RESERVED46', 'RM_ENGINE_TYPE_RESERVED47',
    'RM_ENGINE_TYPE_RESERVED48', 'RM_ENGINE_TYPE_RESERVED49',
    'RM_ENGINE_TYPE_RESERVED4a', 'RM_ENGINE_TYPE_RESERVED4b',
    'RM_ENGINE_TYPE_RESERVED4c', 'RM_ENGINE_TYPE_RESERVED4d',
    'RM_ENGINE_TYPE_RESERVED4e', 'RM_ENGINE_TYPE_RESERVED4f',
    'RM_ENGINE_TYPE_RESERVED50', 'RM_ENGINE_TYPE_RESERVED51',
    'RM_ENGINE_TYPE_RESERVED52', 'RM_ENGINE_TYPE_RESERVED53',
    'RM_ENGINE_TYPE_SEC2', 'RM_ENGINE_TYPE_SW', 'RM_ENGINE_TYPE_TSEC',
    'RM_ENGINE_TYPE_VIC', 'RM_ENGINE_TYPE_VP',
    'RM_ENGINE_TYPE__enumvalues', 'RM_RISCV_UCODE_DESC',
    'RM_RISCV_UCODE_H', 'RPC_GR_BUFFER_TYPE',
    'RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT',
    'RPC_GR_BUFFER_TYPE_GRAPHICS',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_MAX',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL',
    'RPC_GR_BUFFER_TYPE__enumvalues', 'VGPU_MAX_REGOPS_PER_RPC',
    'VGPU_RESERVED_HANDLE_BASE', 'VGPU_RESERVED_HANDLE_RANGE',
    'VGPU_RPC_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PER_RPC_v21_06',
    'VGPU_RPC_GET_P2P_CAPS_V2_MAX_GPUS_SQUARED_PER_RPC',
    'VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS',
    'VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS', '_ACPI_DSM_FUNCTION',
    '__vgpu_rpc_nv_headers_h__', 'c__EA_DISPMUXSTATE',
    'c__EA_FECS_ERROR_EVENT_TYPE', 'c__EA_GPU_RECOVERY_EVENT_TYPE',
    'c__EA_GSP_DMA_TARGET', 'c__EA_LibosMemoryRegionKind',
    'c__EA_LibosMemoryRegionLoc', 'c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE',
    'c__EA_RM_ENGINE_TYPE', 'c__EA_RPC_GR_BUFFER_TYPE',
    'dispMuxState_DiscreteGPU', 'dispMuxState_IntegratedGPU',
    'dispMuxState_None', 'msgqMetadata', 'msgqRxHeader',
    'msgqTxHeader', 'rpc_message_header_v',
    'rpc_message_header_v03_00', 'rpc_message_rpc_union_field_v',
    'rpc_message_rpc_union_field_v03_00', 'struct_ACPI_METHOD_DATA',
    'struct_CAPS_METHOD_DATA', 'struct_DOD_METHOD_DATA',
    'struct_GSP_ACR_BOOT_GSP_RM_PARAMS', 'struct_GSP_FMC_BOOT_PARAMS',
    'struct_GSP_FMC_INIT_PARAMS', 'struct_GSP_MSG_QUEUE_ELEMENT',
    'struct_GSP_RM_PARAMS', 'struct_GSP_SPDM_PARAMS',
    'struct_GSP_VF_INFO', 'struct_GspStaticConfigInfo_t',
    'struct_GspSystemInfo', 'struct_JT_METHOD_DATA',
    'struct_MUX_METHOD_DATA', 'struct_MUX_METHOD_DATA_ELEMENT',
    'struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS',
    'struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS',
    'struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO',
    'struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS',
    'struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS',
    'struct_PACKED_REGISTRY_ENTRY', 'struct_PACKED_REGISTRY_TABLE',
    'struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS',
    'struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS',
    'struct_c__SA_ACPI_DATA', 'struct_c__SA_ACPI_DSM_CACHE',
    'struct_c__SA_BUSINFO', 'struct_c__SA_EcidManufacturingInfo',
    'struct_c__SA_FW_WPR_LAYOUT_OFFSET',
    'struct_c__SA_GSP_ARGUMENTS_CACHED',
    'struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs',
    'struct_c__SA_GSP_PCIE_CONFIG_REG',
    'struct_c__SA_GSP_SR_INIT_ARGUMENTS',
    'struct_c__SA_GspFwHeapFreeList',
    'struct_c__SA_GspFwHeapFreeRegion', 'struct_c__SA_GspFwSRMeta',
    'struct_c__SA_GspFwWprMeta', 'struct_c__SA_GspFwWprMeta_0_0',
    'struct_c__SA_GspFwWprMeta_0_1', 'struct_c__SA_GspFwWprMeta_1_0',
    'struct_c__SA_GspFwWprMeta_1_1',
    'struct_c__SA_LibosMemoryRegionInitArgument',
    'struct_c__SA_MCTP_HEADER',
    'struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS',
    'struct_c__SA_NVDM_PAYLOAD_COT',
    'struct_c__SA_RM_RISCV_UCODE_DESC', 'struct_c__SA_msgqMetadata',
    'struct_c__SA_msgqRxHeader', 'struct_c__SA_msgqTxHeader',
    'struct_rpc_message_header_v03_00', 'union_c__SA_GspFwWprMeta_0',
    'union_c__SA_GspFwWprMeta_1',
    'union_rpc_message_rpc_union_field_v03_00']
