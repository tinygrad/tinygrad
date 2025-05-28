# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-include', '/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/nvtypes.h', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/nvidia-uvm', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/unix/include', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/ctrl']
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
__all__ = \
    ['GSPIFPUB_H', 'GSP_ACR_BOOT_GSP_RM_PARAMS',
    'GSP_ARGUMENTS_CACHED', 'GSP_DMA_TARGET',
    'GSP_DMA_TARGET_COHERENT_SYSTEM', 'GSP_DMA_TARGET_COUNT',
    'GSP_DMA_TARGET_LOCAL_FB', 'GSP_DMA_TARGET_NONCOHERENT_SYSTEM',
    'GSP_DMA_TARGET__enumvalues', 'GSP_FMC_BOOT_PARAMS',
    'GSP_FMC_INIT_PARAMS', 'GSP_FW_HEAP_FREE_LIST_MAGIC',
    'GSP_FW_SR_META_H_', 'GSP_FW_SR_META_INTERNAL_SIZE',
    'GSP_FW_SR_META_MAGIC', 'GSP_FW_SR_META_REVISION',
    'GSP_FW_WPR_HEAP_FREE_REGION_COUNT', 'GSP_FW_WPR_META_H_',
    'GSP_FW_WPR_META_MAGIC', 'GSP_FW_WPR_META_REVISION',
    'GSP_FW_WPR_META_VERIFIED', 'GSP_INIT_ARGS_H', 'GSP_RM_PARAMS',
    'GSP_SPDM_PARAMS', 'GSP_SR_INIT_ARGUMENTS', 'GspFwHeapFreeList',
    'GspFwHeapFreeRegion', 'GspFwSRMeta', 'GspFwWprMeta',
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
    'LibosMemoryRegionLoc__enumvalues', 'MCTP_HEADER',
    'MESSAGE_QUEUE_INIT_ARGUMENTS', 'NVDM_PAYLOAD_COT',
    'RM_RISCV_UCODE_DESC', 'RM_RISCV_UCODE_H', 'c__EA_GSP_DMA_TARGET',
    'c__EA_LibosMemoryRegionKind', 'c__EA_LibosMemoryRegionLoc',
    'struct_GSP_ACR_BOOT_GSP_RM_PARAMS', 'struct_GSP_FMC_BOOT_PARAMS',
    'struct_GSP_FMC_INIT_PARAMS', 'struct_GSP_RM_PARAMS',
    'struct_GSP_SPDM_PARAMS', 'struct_c__SA_GSP_ARGUMENTS_CACHED',
    'struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs',
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
    'struct_c__SA_RM_RISCV_UCODE_DESC', 'union_c__SA_GspFwWprMeta_0',
    'union_c__SA_GspFwWprMeta_1']
