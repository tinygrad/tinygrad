# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os


_libraries = {}
_libraries['libhsakmt.so'] = ctypes.CDLL('/usr/local/lib/libhsakmt.so')
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

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')



# values for enumeration '_HSAKMT_STATUS'
_HSAKMT_STATUS__enumvalues = {
    0: 'HSAKMT_STATUS_SUCCESS',
    1: 'HSAKMT_STATUS_ERROR',
    2: 'HSAKMT_STATUS_DRIVER_MISMATCH',
    3: 'HSAKMT_STATUS_INVALID_PARAMETER',
    4: 'HSAKMT_STATUS_INVALID_HANDLE',
    5: 'HSAKMT_STATUS_INVALID_NODE_UNIT',
    6: 'HSAKMT_STATUS_NO_MEMORY',
    7: 'HSAKMT_STATUS_BUFFER_TOO_SMALL',
    10: 'HSAKMT_STATUS_NOT_IMPLEMENTED',
    11: 'HSAKMT_STATUS_NOT_SUPPORTED',
    12: 'HSAKMT_STATUS_UNAVAILABLE',
    13: 'HSAKMT_STATUS_OUT_OF_RESOURCES',
    20: 'HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED',
    21: 'HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR',
    22: 'HSAKMT_STATUS_KERNEL_ALREADY_OPENED',
    23: 'HSAKMT_STATUS_HSAMMU_UNAVAILABLE',
    30: 'HSAKMT_STATUS_WAIT_FAILURE',
    31: 'HSAKMT_STATUS_WAIT_TIMEOUT',
    35: 'HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED',
    36: 'HSAKMT_STATUS_MEMORY_NOT_REGISTERED',
    37: 'HSAKMT_STATUS_MEMORY_ALIGNMENT',
}
HSAKMT_STATUS_SUCCESS = 0
HSAKMT_STATUS_ERROR = 1
HSAKMT_STATUS_DRIVER_MISMATCH = 2
HSAKMT_STATUS_INVALID_PARAMETER = 3
HSAKMT_STATUS_INVALID_HANDLE = 4
HSAKMT_STATUS_INVALID_NODE_UNIT = 5
HSAKMT_STATUS_NO_MEMORY = 6
HSAKMT_STATUS_BUFFER_TOO_SMALL = 7
HSAKMT_STATUS_NOT_IMPLEMENTED = 10
HSAKMT_STATUS_NOT_SUPPORTED = 11
HSAKMT_STATUS_UNAVAILABLE = 12
HSAKMT_STATUS_OUT_OF_RESOURCES = 13
HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20
HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR = 21
HSAKMT_STATUS_KERNEL_ALREADY_OPENED = 22
HSAKMT_STATUS_HSAMMU_UNAVAILABLE = 23
HSAKMT_STATUS_WAIT_FAILURE = 30
HSAKMT_STATUS_WAIT_TIMEOUT = 31
HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED = 35
HSAKMT_STATUS_MEMORY_NOT_REGISTERED = 36
HSAKMT_STATUS_MEMORY_ALIGNMENT = 37
_HSAKMT_STATUS = ctypes.c_uint32 # enum
HSAKMT_STATUS = _HSAKMT_STATUS
HSAKMT_STATUS__enumvalues = _HSAKMT_STATUS__enumvalues
try:
    hsaKmtOpenKFD = _libraries['libhsakmt.so'].hsaKmtOpenKFD
    hsaKmtOpenKFD.restype = HSAKMT_STATUS
    hsaKmtOpenKFD.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtCloseKFD = _libraries['libhsakmt.so'].hsaKmtCloseKFD
    hsaKmtCloseKFD.restype = HSAKMT_STATUS
    hsaKmtCloseKFD.argtypes = []
except AttributeError:
    pass
class struct__HsaVersionInfo(Structure):
    pass

struct__HsaVersionInfo._pack_ = 1 # source:False
struct__HsaVersionInfo._fields_ = [
    ('KernelInterfaceMajorVersion', ctypes.c_uint32),
    ('KernelInterfaceMinorVersion', ctypes.c_uint32),
]

try:
    hsaKmtGetVersion = _libraries['libhsakmt.so'].hsaKmtGetVersion
    hsaKmtGetVersion.restype = HSAKMT_STATUS
    hsaKmtGetVersion.argtypes = [ctypes.POINTER(struct__HsaVersionInfo)]
except AttributeError:
    pass
class struct__HsaSystemProperties(Structure):
    pass

struct__HsaSystemProperties._pack_ = 1 # source:False
struct__HsaSystemProperties._fields_ = [
    ('NumNodes', ctypes.c_uint32),
    ('PlatformOem', ctypes.c_uint32),
    ('PlatformId', ctypes.c_uint32),
    ('PlatformRev', ctypes.c_uint32),
]

try:
    hsaKmtAcquireSystemProperties = _libraries['libhsakmt.so'].hsaKmtAcquireSystemProperties
    hsaKmtAcquireSystemProperties.restype = HSAKMT_STATUS
    hsaKmtAcquireSystemProperties.argtypes = [ctypes.POINTER(struct__HsaSystemProperties)]
except AttributeError:
    pass
try:
    hsaKmtReleaseSystemProperties = _libraries['libhsakmt.so'].hsaKmtReleaseSystemProperties
    hsaKmtReleaseSystemProperties.restype = HSAKMT_STATUS
    hsaKmtReleaseSystemProperties.argtypes = []
except AttributeError:
    pass
HSAuint32 = ctypes.c_uint32
class struct__HsaNodeProperties(Structure):
    pass

class union_c__UA_HSA_CAPABILITY(Union):
    pass

class struct_c__UA_HSA_CAPABILITY_ui32(Structure):
    pass

struct_c__UA_HSA_CAPABILITY_ui32._pack_ = 1 # source:False
struct_c__UA_HSA_CAPABILITY_ui32._fields_ = [
    ('HotPluggable', ctypes.c_uint32, 1),
    ('HSAMMUPresent', ctypes.c_uint32, 1),
    ('SharedWithGraphics', ctypes.c_uint32, 1),
    ('QueueSizePowerOfTwo', ctypes.c_uint32, 1),
    ('QueueSize32bit', ctypes.c_uint32, 1),
    ('QueueIdleEvent', ctypes.c_uint32, 1),
    ('VALimit', ctypes.c_uint32, 1),
    ('WatchPointsSupported', ctypes.c_uint32, 1),
    ('WatchPointsTotalBits', ctypes.c_uint32, 4),
    ('DoorbellType', ctypes.c_uint32, 2),
    ('AQLQueueDoubleMap', ctypes.c_uint32, 1),
    ('DebugTrapSupported', ctypes.c_uint32, 1),
    ('WaveLaunchTrapOverrideSupported', ctypes.c_uint32, 1),
    ('WaveLaunchModeSupported', ctypes.c_uint32, 1),
    ('PreciseMemoryOperationsSupported', ctypes.c_uint32, 1),
    ('DEPRECATED_SRAM_EDCSupport', ctypes.c_uint32, 1),
    ('Mem_EDCSupport', ctypes.c_uint32, 1),
    ('RASEventNotify', ctypes.c_uint32, 1),
    ('ASICRevision', ctypes.c_uint32, 4),
    ('SRAM_EDCSupport', ctypes.c_uint32, 1),
    ('SVMAPISupported', ctypes.c_uint32, 1),
    ('CoherentHostAccess', ctypes.c_uint32, 1),
    ('DebugSupportedFirmware', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 2),
]

union_c__UA_HSA_CAPABILITY._pack_ = 1 # source:False
union_c__UA_HSA_CAPABILITY._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_c__UA_HSA_CAPABILITY_ui32),
]

class union_c__UA_HSA_ENGINE_ID(Union):
    pass

class struct_c__UA_HSA_ENGINE_ID_ui32(Structure):
    pass

struct_c__UA_HSA_ENGINE_ID_ui32._pack_ = 1 # source:False
struct_c__UA_HSA_ENGINE_ID_ui32._fields_ = [
    ('uCode', ctypes.c_uint32, 10),
    ('Major', ctypes.c_uint32, 6),
    ('Minor', ctypes.c_uint32, 8),
    ('Stepping', ctypes.c_uint32, 8),
]

union_c__UA_HSA_ENGINE_ID._pack_ = 1 # source:False
union_c__UA_HSA_ENGINE_ID._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_c__UA_HSA_ENGINE_ID_ui32),
]

class union_c__UA_HSA_ENGINE_VERSION(Union):
    pass

class struct_c__UA_HSA_ENGINE_VERSION_0(Structure):
    pass

struct_c__UA_HSA_ENGINE_VERSION_0._pack_ = 1 # source:False
struct_c__UA_HSA_ENGINE_VERSION_0._fields_ = [
    ('uCodeSDMA', ctypes.c_uint32, 10),
    ('uCodeRes', ctypes.c_uint32, 10),
    ('Reserved', ctypes.c_uint32, 12),
]

union_c__UA_HSA_ENGINE_VERSION._pack_ = 1 # source:False
union_c__UA_HSA_ENGINE_VERSION._anonymous_ = ('_0',)
union_c__UA_HSA_ENGINE_VERSION._fields_ = [
    ('Value', ctypes.c_uint32),
    ('_0', struct_c__UA_HSA_ENGINE_VERSION_0),
]

class union_c__UA_HSA_DEBUG_PROPERTIES(Union):
    pass

class struct_c__UA_HSA_DEBUG_PROPERTIES_0(Structure):
    pass

struct_c__UA_HSA_DEBUG_PROPERTIES_0._pack_ = 1 # source:False
struct_c__UA_HSA_DEBUG_PROPERTIES_0._fields_ = [
    ('WatchAddrMaskLoBit', ctypes.c_uint64, 4),
    ('WatchAddrMaskHiBit', ctypes.c_uint64, 6),
    ('DispatchInfoAlwaysValid', ctypes.c_uint64, 1),
    ('AddressWatchpointShareKind', ctypes.c_uint64, 1),
    ('Reserved', ctypes.c_uint64, 52),
]

union_c__UA_HSA_DEBUG_PROPERTIES._pack_ = 1 # source:False
union_c__UA_HSA_DEBUG_PROPERTIES._anonymous_ = ('_0',)
union_c__UA_HSA_DEBUG_PROPERTIES._fields_ = [
    ('Value', ctypes.c_uint64),
    ('_0', struct_c__UA_HSA_DEBUG_PROPERTIES_0),
]

struct__HsaNodeProperties._pack_ = 1 # source:False
struct__HsaNodeProperties._fields_ = [
    ('NumCPUCores', ctypes.c_uint32),
    ('NumFComputeCores', ctypes.c_uint32),
    ('NumMemoryBanks', ctypes.c_uint32),
    ('NumCaches', ctypes.c_uint32),
    ('NumIOLinks', ctypes.c_uint32),
    ('CComputeIdLo', ctypes.c_uint32),
    ('FComputeIdLo', ctypes.c_uint32),
    ('Capability', union_c__UA_HSA_CAPABILITY),
    ('MaxWavesPerSIMD', ctypes.c_uint32),
    ('LDSSizeInKB', ctypes.c_uint32),
    ('GDSSizeInKB', ctypes.c_uint32),
    ('WaveFrontSize', ctypes.c_uint32),
    ('NumShaderBanks', ctypes.c_uint32),
    ('NumArrays', ctypes.c_uint32),
    ('NumCUPerArray', ctypes.c_uint32),
    ('NumSIMDPerCU', ctypes.c_uint32),
    ('MaxSlotsScratchCU', ctypes.c_uint32),
    ('EngineId', union_c__UA_HSA_ENGINE_ID),
    ('VendorId', ctypes.c_uint16),
    ('DeviceId', ctypes.c_uint16),
    ('LocationId', ctypes.c_uint32),
    ('LocalMemSize', ctypes.c_uint64),
    ('MaxEngineClockMhzFCompute', ctypes.c_uint32),
    ('MaxEngineClockMhzCCompute', ctypes.c_uint32),
    ('DrmRenderMinor', ctypes.c_int32),
    ('MarketingName', ctypes.c_uint16 * 64),
    ('AMDName', ctypes.c_ubyte * 64),
    ('uCodeEngineVersions', union_c__UA_HSA_ENGINE_VERSION),
    ('DebugProperties', union_c__UA_HSA_DEBUG_PROPERTIES),
    ('HiveID', ctypes.c_uint64),
    ('NumSdmaEngines', ctypes.c_uint32),
    ('NumSdmaXgmiEngines', ctypes.c_uint32),
    ('NumSdmaQueuesPerEngine', ctypes.c_ubyte),
    ('NumCpQueues', ctypes.c_ubyte),
    ('NumGws', ctypes.c_ubyte),
    ('Reserved2', ctypes.c_ubyte),
    ('Domain', ctypes.c_uint32),
    ('UniqueID', ctypes.c_uint64),
    ('VGPRSizePerCU', ctypes.c_uint32),
    ('SGPRSizePerCU', ctypes.c_uint32),
    ('NumXcc', ctypes.c_uint32),
    ('KFDGpuID', ctypes.c_uint32),
    ('FamilyID', ctypes.c_uint32),
]

try:
    hsaKmtGetNodeProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeProperties
    hsaKmtGetNodeProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeProperties.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaNodeProperties)]
except AttributeError:
    pass
class struct__HsaMemoryProperties(Structure):
    pass


# values for enumeration '_HSA_HEAPTYPE'
_HSA_HEAPTYPE__enumvalues = {
    0: 'HSA_HEAPTYPE_SYSTEM',
    1: 'HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC',
    2: 'HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE',
    3: 'HSA_HEAPTYPE_GPU_GDS',
    4: 'HSA_HEAPTYPE_GPU_LDS',
    5: 'HSA_HEAPTYPE_GPU_SCRATCH',
    6: 'HSA_HEAPTYPE_DEVICE_SVM',
    7: 'HSA_HEAPTYPE_MMIO_REMAP',
    8: 'HSA_HEAPTYPE_NUMHEAPTYPES',
    4294967295: 'HSA_HEAPTYPE_SIZE',
}
HSA_HEAPTYPE_SYSTEM = 0
HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC = 1
HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE = 2
HSA_HEAPTYPE_GPU_GDS = 3
HSA_HEAPTYPE_GPU_LDS = 4
HSA_HEAPTYPE_GPU_SCRATCH = 5
HSA_HEAPTYPE_DEVICE_SVM = 6
HSA_HEAPTYPE_MMIO_REMAP = 7
HSA_HEAPTYPE_NUMHEAPTYPES = 8
HSA_HEAPTYPE_SIZE = 4294967295
_HSA_HEAPTYPE = ctypes.c_uint32 # enum
class union__HsaMemoryProperties_0(Union):
    pass

class struct__HsaMemoryProperties_0_ui32(Structure):
    pass

struct__HsaMemoryProperties_0_ui32._pack_ = 1 # source:False
struct__HsaMemoryProperties_0_ui32._fields_ = [
    ('SizeInBytesLow', ctypes.c_uint32),
    ('SizeInBytesHigh', ctypes.c_uint32),
]

union__HsaMemoryProperties_0._pack_ = 1 # source:False
union__HsaMemoryProperties_0._fields_ = [
    ('SizeInBytes', ctypes.c_uint64),
    ('ui32', struct__HsaMemoryProperties_0_ui32),
]

class union_c__UA_HSA_MEMORYPROPERTY(Union):
    pass

class struct_c__UA_HSA_MEMORYPROPERTY_ui32(Structure):
    pass

struct_c__UA_HSA_MEMORYPROPERTY_ui32._pack_ = 1 # source:False
struct_c__UA_HSA_MEMORYPROPERTY_ui32._fields_ = [
    ('HotPluggable', ctypes.c_uint32, 1),
    ('NonVolatile', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 30),
]

union_c__UA_HSA_MEMORYPROPERTY._pack_ = 1 # source:False
union_c__UA_HSA_MEMORYPROPERTY._fields_ = [
    ('MemoryProperty', ctypes.c_uint32),
    ('ui32', struct_c__UA_HSA_MEMORYPROPERTY_ui32),
]

struct__HsaMemoryProperties._pack_ = 1 # source:False
struct__HsaMemoryProperties._anonymous_ = ('_0',)
struct__HsaMemoryProperties._fields_ = [
    ('HeapType', _HSA_HEAPTYPE),
    ('_0', union__HsaMemoryProperties_0),
    ('Flags', union_c__UA_HSA_MEMORYPROPERTY),
    ('Width', ctypes.c_uint32),
    ('MemoryClockMax', ctypes.c_uint32),
    ('VirtualBaseAddress', ctypes.c_uint64),
]

try:
    hsaKmtGetNodeMemoryProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeMemoryProperties
    hsaKmtGetNodeMemoryProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeMemoryProperties.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaMemoryProperties)]
except AttributeError:
    pass
class struct__HaCacheProperties(Structure):
    pass

class union_c__UA_HsaCacheType(Union):
    pass

class struct_c__UA_HsaCacheType_ui32(Structure):
    pass

struct_c__UA_HsaCacheType_ui32._pack_ = 1 # source:False
struct_c__UA_HsaCacheType_ui32._fields_ = [
    ('Data', ctypes.c_uint32, 1),
    ('Instruction', ctypes.c_uint32, 1),
    ('CPU', ctypes.c_uint32, 1),
    ('HSACU', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 28),
]

union_c__UA_HsaCacheType._pack_ = 1 # source:False
union_c__UA_HsaCacheType._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_c__UA_HsaCacheType_ui32),
]

struct__HaCacheProperties._pack_ = 1 # source:False
struct__HaCacheProperties._fields_ = [
    ('ProcessorIdLow', ctypes.c_uint32),
    ('CacheLevel', ctypes.c_uint32),
    ('CacheSize', ctypes.c_uint32),
    ('CacheLineSize', ctypes.c_uint32),
    ('CacheLinesPerTag', ctypes.c_uint32),
    ('CacheAssociativity', ctypes.c_uint32),
    ('CacheLatency', ctypes.c_uint32),
    ('CacheType', union_c__UA_HsaCacheType),
    ('SiblingMap', ctypes.c_uint32 * 256),
]

try:
    hsaKmtGetNodeCacheProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeCacheProperties
    hsaKmtGetNodeCacheProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeCacheProperties.argtypes = [HSAuint32, HSAuint32, HSAuint32, ctypes.POINTER(struct__HaCacheProperties)]
except AttributeError:
    pass
class struct__HsaIoLinkProperties(Structure):
    pass


# values for enumeration '_HSA_IOLINKTYPE'
_HSA_IOLINKTYPE__enumvalues = {
    0: 'HSA_IOLINKTYPE_UNDEFINED',
    1: 'HSA_IOLINKTYPE_HYPERTRANSPORT',
    2: 'HSA_IOLINKTYPE_PCIEXPRESS',
    3: 'HSA_IOLINKTYPE_AMBA',
    4: 'HSA_IOLINKTYPE_MIPI',
    5: 'HSA_IOLINK_TYPE_QPI_1_1',
    6: 'HSA_IOLINK_TYPE_RESERVED1',
    7: 'HSA_IOLINK_TYPE_RESERVED2',
    8: 'HSA_IOLINK_TYPE_RAPID_IO',
    9: 'HSA_IOLINK_TYPE_INFINIBAND',
    10: 'HSA_IOLINK_TYPE_RESERVED3',
    11: 'HSA_IOLINK_TYPE_XGMI',
    12: 'HSA_IOLINK_TYPE_XGOP',
    13: 'HSA_IOLINK_TYPE_GZ',
    14: 'HSA_IOLINK_TYPE_ETHERNET_RDMA',
    15: 'HSA_IOLINK_TYPE_RDMA_OTHER',
    16: 'HSA_IOLINK_TYPE_OTHER',
    17: 'HSA_IOLINKTYPE_NUMIOLINKTYPES',
    4294967295: 'HSA_IOLINKTYPE_SIZE',
}
HSA_IOLINKTYPE_UNDEFINED = 0
HSA_IOLINKTYPE_HYPERTRANSPORT = 1
HSA_IOLINKTYPE_PCIEXPRESS = 2
HSA_IOLINKTYPE_AMBA = 3
HSA_IOLINKTYPE_MIPI = 4
HSA_IOLINK_TYPE_QPI_1_1 = 5
HSA_IOLINK_TYPE_RESERVED1 = 6
HSA_IOLINK_TYPE_RESERVED2 = 7
HSA_IOLINK_TYPE_RAPID_IO = 8
HSA_IOLINK_TYPE_INFINIBAND = 9
HSA_IOLINK_TYPE_RESERVED3 = 10
HSA_IOLINK_TYPE_XGMI = 11
HSA_IOLINK_TYPE_XGOP = 12
HSA_IOLINK_TYPE_GZ = 13
HSA_IOLINK_TYPE_ETHERNET_RDMA = 14
HSA_IOLINK_TYPE_RDMA_OTHER = 15
HSA_IOLINK_TYPE_OTHER = 16
HSA_IOLINKTYPE_NUMIOLINKTYPES = 17
HSA_IOLINKTYPE_SIZE = 4294967295
_HSA_IOLINKTYPE = ctypes.c_uint32 # enum
class union_c__UA_HSA_LINKPROPERTY(Union):
    pass

class struct_c__UA_HSA_LINKPROPERTY_ui32(Structure):
    pass

struct_c__UA_HSA_LINKPROPERTY_ui32._pack_ = 1 # source:False
struct_c__UA_HSA_LINKPROPERTY_ui32._fields_ = [
    ('Override', ctypes.c_uint32, 1),
    ('NonCoherent', ctypes.c_uint32, 1),
    ('NoAtomics32bit', ctypes.c_uint32, 1),
    ('NoAtomics64bit', ctypes.c_uint32, 1),
    ('NoPeerToPeerDMA', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 27),
]

union_c__UA_HSA_LINKPROPERTY._pack_ = 1 # source:False
union_c__UA_HSA_LINKPROPERTY._fields_ = [
    ('LinkProperty', ctypes.c_uint32),
    ('ui32', struct_c__UA_HSA_LINKPROPERTY_ui32),
]

struct__HsaIoLinkProperties._pack_ = 1 # source:False
struct__HsaIoLinkProperties._fields_ = [
    ('IoLinkType', _HSA_IOLINKTYPE),
    ('VersionMajor', ctypes.c_uint32),
    ('VersionMinor', ctypes.c_uint32),
    ('NodeFrom', ctypes.c_uint32),
    ('NodeTo', ctypes.c_uint32),
    ('Weight', ctypes.c_uint32),
    ('MinimumLatency', ctypes.c_uint32),
    ('MaximumLatency', ctypes.c_uint32),
    ('MinimumBandwidth', ctypes.c_uint32),
    ('MaximumBandwidth', ctypes.c_uint32),
    ('RecTransferSize', ctypes.c_uint32),
    ('Flags', union_c__UA_HSA_LINKPROPERTY),
]

try:
    hsaKmtGetNodeIoLinkProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeIoLinkProperties
    hsaKmtGetNodeIoLinkProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeIoLinkProperties.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaIoLinkProperties)]
except AttributeError:
    pass
class struct__HsaEventDescriptor(Structure):
    pass


# values for enumeration '_HSA_EVENTTYPE'
_HSA_EVENTTYPE__enumvalues = {
    0: 'HSA_EVENTTYPE_SIGNAL',
    1: 'HSA_EVENTTYPE_NODECHANGE',
    2: 'HSA_EVENTTYPE_DEVICESTATECHANGE',
    3: 'HSA_EVENTTYPE_HW_EXCEPTION',
    4: 'HSA_EVENTTYPE_SYSTEM_EVENT',
    5: 'HSA_EVENTTYPE_DEBUG_EVENT',
    6: 'HSA_EVENTTYPE_PROFILE_EVENT',
    7: 'HSA_EVENTTYPE_QUEUE_EVENT',
    8: 'HSA_EVENTTYPE_MEMORY',
    9: 'HSA_EVENTTYPE_MAXID',
    4294967295: 'HSA_EVENTTYPE_TYPE_SIZE',
}
HSA_EVENTTYPE_SIGNAL = 0
HSA_EVENTTYPE_NODECHANGE = 1
HSA_EVENTTYPE_DEVICESTATECHANGE = 2
HSA_EVENTTYPE_HW_EXCEPTION = 3
HSA_EVENTTYPE_SYSTEM_EVENT = 4
HSA_EVENTTYPE_DEBUG_EVENT = 5
HSA_EVENTTYPE_PROFILE_EVENT = 6
HSA_EVENTTYPE_QUEUE_EVENT = 7
HSA_EVENTTYPE_MEMORY = 8
HSA_EVENTTYPE_MAXID = 9
HSA_EVENTTYPE_TYPE_SIZE = 4294967295
_HSA_EVENTTYPE = ctypes.c_uint32 # enum
class struct__HsaSyncVar(Structure):
    pass

class union__HsaSyncVar_SyncVar(Union):
    pass

union__HsaSyncVar_SyncVar._pack_ = 1 # source:False
union__HsaSyncVar_SyncVar._fields_ = [
    ('UserData', ctypes.POINTER(None)),
    ('UserDataPtrValue', ctypes.c_uint64),
]

struct__HsaSyncVar._pack_ = 1 # source:False
struct__HsaSyncVar._fields_ = [
    ('SyncVar', union__HsaSyncVar_SyncVar),
    ('SyncVarSize', ctypes.c_uint64),
]

struct__HsaEventDescriptor._pack_ = 1 # source:False
struct__HsaEventDescriptor._fields_ = [
    ('EventType', _HSA_EVENTTYPE),
    ('NodeId', ctypes.c_uint32),
    ('SyncVar', struct__HsaSyncVar),
]

class struct__HsaEvent(Structure):
    pass

class struct__HsaEventData(Structure):
    pass

class union__HsaEventData_EventData(Union):
    pass

class struct__HsaNodeChange(Structure):
    pass


# values for enumeration '_HSA_EVENTTYPE_NODECHANGE_FLAGS'
_HSA_EVENTTYPE_NODECHANGE_FLAGS__enumvalues = {
    0: 'HSA_EVENTTYPE_NODECHANGE_ADD',
    1: 'HSA_EVENTTYPE_NODECHANGE_REMOVE',
    4294967295: 'HSA_EVENTTYPE_NODECHANGE_SIZE',
}
HSA_EVENTTYPE_NODECHANGE_ADD = 0
HSA_EVENTTYPE_NODECHANGE_REMOVE = 1
HSA_EVENTTYPE_NODECHANGE_SIZE = 4294967295
_HSA_EVENTTYPE_NODECHANGE_FLAGS = ctypes.c_uint32 # enum
struct__HsaNodeChange._pack_ = 1 # source:False
struct__HsaNodeChange._fields_ = [
    ('Flags', _HSA_EVENTTYPE_NODECHANGE_FLAGS),
]

class struct__HsaDeviceStateChange(Structure):
    pass


# values for enumeration '_HSA_DEVICE'
_HSA_DEVICE__enumvalues = {
    0: 'HSA_DEVICE_CPU',
    1: 'HSA_DEVICE_GPU',
    2: 'MAX_HSA_DEVICE',
}
HSA_DEVICE_CPU = 0
HSA_DEVICE_GPU = 1
MAX_HSA_DEVICE = 2
_HSA_DEVICE = ctypes.c_uint32 # enum

# values for enumeration '_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS'
_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS__enumvalues = {
    0: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_START',
    1: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP',
    4294967295: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE',
}
HSA_EVENTTYPE_DEVICESTATUSCHANGE_START = 0
HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP = 1
HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE = 4294967295
_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS = ctypes.c_uint32 # enum
struct__HsaDeviceStateChange._pack_ = 1 # source:False
struct__HsaDeviceStateChange._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('Device', _HSA_DEVICE),
    ('Flags', _HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS),
]

class struct__HsaMemoryAccessFault(Structure):
    pass

class struct__HsaAccessAttributeFailure(Structure):
    pass

struct__HsaAccessAttributeFailure._pack_ = 1 # source:False
struct__HsaAccessAttributeFailure._fields_ = [
    ('NotPresent', ctypes.c_uint32, 1),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('NoExecute', ctypes.c_uint32, 1),
    ('GpuAccess', ctypes.c_uint32, 1),
    ('ECC', ctypes.c_uint32, 1),
    ('Imprecise', ctypes.c_uint32, 1),
    ('ErrorType', ctypes.c_uint32, 3),
    ('Reserved', ctypes.c_uint32, 23),
]


# values for enumeration '_HSA_EVENTID_MEMORYFLAGS'
_HSA_EVENTID_MEMORYFLAGS__enumvalues = {
    0: 'HSA_EVENTID_MEMORY_RECOVERABLE',
    1: 'HSA_EVENTID_MEMORY_FATAL_PROCESS',
    2: 'HSA_EVENTID_MEMORY_FATAL_VM',
}
HSA_EVENTID_MEMORY_RECOVERABLE = 0
HSA_EVENTID_MEMORY_FATAL_PROCESS = 1
HSA_EVENTID_MEMORY_FATAL_VM = 2
_HSA_EVENTID_MEMORYFLAGS = ctypes.c_uint32 # enum
struct__HsaMemoryAccessFault._pack_ = 1 # source:False
struct__HsaMemoryAccessFault._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('VirtualAddress', ctypes.c_uint64),
    ('Failure', struct__HsaAccessAttributeFailure),
    ('Flags', _HSA_EVENTID_MEMORYFLAGS),
]

class struct__HsaHwException(Structure):
    pass


# values for enumeration '_HSA_EVENTID_HW_EXCEPTION_CAUSE'
_HSA_EVENTID_HW_EXCEPTION_CAUSE__enumvalues = {
    0: 'HSA_EVENTID_HW_EXCEPTION_GPU_HANG',
    1: 'HSA_EVENTID_HW_EXCEPTION_ECC',
}
HSA_EVENTID_HW_EXCEPTION_GPU_HANG = 0
HSA_EVENTID_HW_EXCEPTION_ECC = 1
_HSA_EVENTID_HW_EXCEPTION_CAUSE = ctypes.c_uint32 # enum
struct__HsaHwException._pack_ = 1 # source:False
struct__HsaHwException._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('ResetType', ctypes.c_uint32),
    ('MemoryLost', ctypes.c_uint32),
    ('ResetCause', _HSA_EVENTID_HW_EXCEPTION_CAUSE),
]

union__HsaEventData_EventData._pack_ = 1 # source:False
union__HsaEventData_EventData._fields_ = [
    ('SyncVar', struct__HsaSyncVar),
    ('NodeChangeState', struct__HsaNodeChange),
    ('DeviceState', struct__HsaDeviceStateChange),
    ('MemoryAccessFault', struct__HsaMemoryAccessFault),
    ('HwException', struct__HsaHwException),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct__HsaEventData._pack_ = 1 # source:False
struct__HsaEventData._fields_ = [
    ('EventType', _HSA_EVENTTYPE),
    ('EventData', union__HsaEventData_EventData),
    ('HWData1', ctypes.c_uint64),
    ('HWData2', ctypes.c_uint64),
    ('HWData3', ctypes.c_uint32),
]

struct__HsaEvent._pack_ = 1 # source:False
struct__HsaEvent._fields_ = [
    ('EventId', ctypes.c_uint32),
    ('EventData', struct__HsaEventData),
]

try:
    hsaKmtCreateEvent = _libraries['libhsakmt.so'].hsaKmtCreateEvent
    hsaKmtCreateEvent.restype = HSAKMT_STATUS
    hsaKmtCreateEvent.argtypes = [ctypes.POINTER(struct__HsaEventDescriptor), ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.POINTER(struct__HsaEvent))]
except AttributeError:
    pass
try:
    hsaKmtDestroyEvent = _libraries['libhsakmt.so'].hsaKmtDestroyEvent
    hsaKmtDestroyEvent.restype = HSAKMT_STATUS
    hsaKmtDestroyEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtSetEvent = _libraries['libhsakmt.so'].hsaKmtSetEvent
    hsaKmtSetEvent.restype = HSAKMT_STATUS
    hsaKmtSetEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtResetEvent = _libraries['libhsakmt.so'].hsaKmtResetEvent
    hsaKmtResetEvent.restype = HSAKMT_STATUS
    hsaKmtResetEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtQueryEventState = _libraries['libhsakmt.so'].hsaKmtQueryEventState
    hsaKmtQueryEventState.restype = HSAKMT_STATUS
    hsaKmtQueryEventState.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtWaitOnEvent = _libraries['libhsakmt.so'].hsaKmtWaitOnEvent
    hsaKmtWaitOnEvent.restype = HSAKMT_STATUS
    hsaKmtWaitOnEvent.argtypes = [ctypes.POINTER(struct__HsaEvent), HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtWaitOnEvent_Ext = _libraries['libhsakmt.so'].hsaKmtWaitOnEvent_Ext
    hsaKmtWaitOnEvent_Ext.restype = HSAKMT_STATUS
    hsaKmtWaitOnEvent_Ext.argtypes = [ctypes.POINTER(struct__HsaEvent), HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtWaitOnMultipleEvents = _libraries['libhsakmt.so'].hsaKmtWaitOnMultipleEvents
    hsaKmtWaitOnMultipleEvents.restype = HSAKMT_STATUS
    hsaKmtWaitOnMultipleEvents.argtypes = [ctypes.POINTER(struct__HsaEvent) * 0, HSAuint32, ctypes.c_bool, HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtWaitOnMultipleEvents_Ext = _libraries['libhsakmt.so'].hsaKmtWaitOnMultipleEvents_Ext
    hsaKmtWaitOnMultipleEvents_Ext.restype = HSAKMT_STATUS
    hsaKmtWaitOnMultipleEvents_Ext.argtypes = [ctypes.POINTER(struct__HsaEvent) * 0, HSAuint32, ctypes.c_bool, HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
HSA_QUEUEID = ctypes.c_uint64
class struct__HsaQueueReport(Structure):
    pass

struct__HsaQueueReport._pack_ = 1 # source:False
struct__HsaQueueReport._fields_ = [
    ('VMID', ctypes.c_uint32),
    ('QueueAddress', ctypes.POINTER(None)),
    ('QueueSize', ctypes.c_uint64),
]

try:
    hsaKmtReportQueue = _libraries['FIXME_STUB'].hsaKmtReportQueue
    hsaKmtReportQueue.restype = HSAKMT_STATUS
    hsaKmtReportQueue.argtypes = [HSA_QUEUEID, ctypes.POINTER(struct__HsaQueueReport)]
except AttributeError:
    pass

# values for enumeration '_HSA_QUEUE_TYPE'
_HSA_QUEUE_TYPE__enumvalues = {
    1: 'HSA_QUEUE_COMPUTE',
    2: 'HSA_QUEUE_SDMA',
    3: 'HSA_QUEUE_MULTIMEDIA_DECODE',
    4: 'HSA_QUEUE_MULTIMEDIA_ENCODE',
    5: 'HSA_QUEUE_SDMA_XGMI',
    11: 'HSA_QUEUE_COMPUTE_OS',
    12: 'HSA_QUEUE_SDMA_OS',
    13: 'HSA_QUEUE_MULTIMEDIA_DECODE_OS',
    14: 'HSA_QUEUE_MULTIMEDIA_ENCODE_OS',
    21: 'HSA_QUEUE_COMPUTE_AQL',
    22: 'HSA_QUEUE_DMA_AQL',
    23: 'HSA_QUEUE_DMA_AQL_XGMI',
    4294967295: 'HSA_QUEUE_TYPE_SIZE',
}
HSA_QUEUE_COMPUTE = 1
HSA_QUEUE_SDMA = 2
HSA_QUEUE_MULTIMEDIA_DECODE = 3
HSA_QUEUE_MULTIMEDIA_ENCODE = 4
HSA_QUEUE_SDMA_XGMI = 5
HSA_QUEUE_COMPUTE_OS = 11
HSA_QUEUE_SDMA_OS = 12
HSA_QUEUE_MULTIMEDIA_DECODE_OS = 13
HSA_QUEUE_MULTIMEDIA_ENCODE_OS = 14
HSA_QUEUE_COMPUTE_AQL = 21
HSA_QUEUE_DMA_AQL = 22
HSA_QUEUE_DMA_AQL_XGMI = 23
HSA_QUEUE_TYPE_SIZE = 4294967295
_HSA_QUEUE_TYPE = ctypes.c_uint32 # enum
HSA_QUEUE_TYPE = _HSA_QUEUE_TYPE
HSA_QUEUE_TYPE__enumvalues = _HSA_QUEUE_TYPE__enumvalues

# values for enumeration '_HSA_QUEUE_PRIORITY'
_HSA_QUEUE_PRIORITY__enumvalues = {
    -3: 'HSA_QUEUE_PRIORITY_MINIMUM',
    -2: 'HSA_QUEUE_PRIORITY_LOW',
    -1: 'HSA_QUEUE_PRIORITY_BELOW_NORMAL',
    0: 'HSA_QUEUE_PRIORITY_NORMAL',
    1: 'HSA_QUEUE_PRIORITY_ABOVE_NORMAL',
    2: 'HSA_QUEUE_PRIORITY_HIGH',
    3: 'HSA_QUEUE_PRIORITY_MAXIMUM',
    4: 'HSA_QUEUE_PRIORITY_NUM_PRIORITY',
    4294967295: 'HSA_QUEUE_PRIORITY_SIZE',
}
HSA_QUEUE_PRIORITY_MINIMUM = -3
HSA_QUEUE_PRIORITY_LOW = -2
HSA_QUEUE_PRIORITY_BELOW_NORMAL = -1
HSA_QUEUE_PRIORITY_NORMAL = 0
HSA_QUEUE_PRIORITY_ABOVE_NORMAL = 1
HSA_QUEUE_PRIORITY_HIGH = 2
HSA_QUEUE_PRIORITY_MAXIMUM = 3
HSA_QUEUE_PRIORITY_NUM_PRIORITY = 4
HSA_QUEUE_PRIORITY_SIZE = 4294967295
_HSA_QUEUE_PRIORITY = ctypes.c_int64 # enum
HSA_QUEUE_PRIORITY = _HSA_QUEUE_PRIORITY
HSA_QUEUE_PRIORITY__enumvalues = _HSA_QUEUE_PRIORITY__enumvalues
HSAuint64 = ctypes.c_uint64
class struct__HsaQueueResource(Structure):
    pass

class union__HsaQueueResource_0(Union):
    pass

union__HsaQueueResource_0._pack_ = 1 # source:False
union__HsaQueueResource_0._fields_ = [
    ('Queue_DoorBell', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_DoorBell_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueDoorBell', ctypes.c_uint64),
]

class union__HsaQueueResource_1(Union):
    pass

union__HsaQueueResource_1._pack_ = 1 # source:False
union__HsaQueueResource_1._fields_ = [
    ('Queue_write_ptr', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_write_ptr_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueWptrValue', ctypes.c_uint64),
]

class union__HsaQueueResource_2(Union):
    pass

union__HsaQueueResource_2._pack_ = 1 # source:False
union__HsaQueueResource_2._fields_ = [
    ('Queue_read_ptr', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_read_ptr_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueRptrValue', ctypes.c_uint64),
]

struct__HsaQueueResource._pack_ = 1 # source:False
struct__HsaQueueResource._anonymous_ = ('_0', '_1', '_2',)
struct__HsaQueueResource._fields_ = [
    ('QueueId', ctypes.c_uint64),
    ('_0', union__HsaQueueResource_0),
    ('_1', union__HsaQueueResource_1),
    ('_2', union__HsaQueueResource_2),
    ('ErrorReason', ctypes.POINTER(ctypes.c_int64)),
]

try:
    hsaKmtCreateQueue = _libraries['libhsakmt.so'].hsaKmtCreateQueue
    hsaKmtCreateQueue.restype = HSAKMT_STATUS
    hsaKmtCreateQueue.argtypes = [HSAuint32, HSA_QUEUE_TYPE, HSAuint32, HSA_QUEUE_PRIORITY, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(struct__HsaEvent), ctypes.POINTER(struct__HsaQueueResource)]
except AttributeError:
    pass
try:
    hsaKmtUpdateQueue = _libraries['libhsakmt.so'].hsaKmtUpdateQueue
    hsaKmtUpdateQueue.restype = HSAKMT_STATUS
    hsaKmtUpdateQueue.argtypes = [HSA_QUEUEID, HSAuint32, HSA_QUEUE_PRIORITY, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtDestroyQueue = _libraries['libhsakmt.so'].hsaKmtDestroyQueue
    hsaKmtDestroyQueue.restype = HSAKMT_STATUS
    hsaKmtDestroyQueue.argtypes = [HSA_QUEUEID]
except AttributeError:
    pass
try:
    hsaKmtSetQueueCUMask = _libraries['libhsakmt.so'].hsaKmtSetQueueCUMask
    hsaKmtSetQueueCUMask.restype = HSAKMT_STATUS
    hsaKmtSetQueueCUMask.argtypes = [HSA_QUEUEID, HSAuint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_c__SA_HsaQueueInfo(Structure):
    pass

class struct_c__SA_HsaUserContextSaveAreaHeader(Structure):
    pass

struct_c__SA_HsaQueueInfo._pack_ = 1 # source:False
struct_c__SA_HsaQueueInfo._fields_ = [
    ('QueueDetailError', ctypes.c_uint32),
    ('QueueTypeExtended', ctypes.c_uint32),
    ('NumCUAssigned', ctypes.c_uint32),
    ('CUMaskInfo', ctypes.POINTER(ctypes.c_uint32)),
    ('UserContextSaveArea', ctypes.POINTER(ctypes.c_uint32)),
    ('SaveAreaSizeInBytes', ctypes.c_uint64),
    ('ControlStackTop', ctypes.POINTER(ctypes.c_uint32)),
    ('ControlStackUsedInBytes', ctypes.c_uint64),
    ('SaveAreaHeader', ctypes.POINTER(struct_c__SA_HsaUserContextSaveAreaHeader)),
    ('Reserved2', ctypes.c_uint64),
]

struct_c__SA_HsaUserContextSaveAreaHeader._pack_ = 1 # source:False
struct_c__SA_HsaUserContextSaveAreaHeader._fields_ = [
    ('ControlStackOffset', ctypes.c_uint32),
    ('ControlStackSize', ctypes.c_uint32),
    ('WaveStateOffset', ctypes.c_uint32),
    ('WaveStateSize', ctypes.c_uint32),
    ('DebugOffset', ctypes.c_uint32),
    ('DebugSize', ctypes.c_uint32),
    ('ErrorReason', ctypes.POINTER(ctypes.c_int64)),
    ('ErrorEventId', ctypes.c_uint32),
    ('Reserved1', ctypes.c_uint32),
]

try:
    hsaKmtGetQueueInfo = _libraries['libhsakmt.so'].hsaKmtGetQueueInfo
    hsaKmtGetQueueInfo.restype = HSAKMT_STATUS
    hsaKmtGetQueueInfo.argtypes = [HSA_QUEUEID, ctypes.POINTER(struct_c__SA_HsaQueueInfo)]
except AttributeError:
    pass
try:
    hsaKmtSetMemoryPolicy = _libraries['libhsakmt.so'].hsaKmtSetMemoryPolicy
    hsaKmtSetMemoryPolicy.restype = HSAKMT_STATUS
    hsaKmtSetMemoryPolicy.argtypes = [HSAuint32, HSAuint32, HSAuint32, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
class struct__HsaMemFlags(Structure):
    pass

class union__HsaMemFlags_0(Union):
    pass

class struct__HsaMemFlags_0_ui32(Structure):
    pass

struct__HsaMemFlags_0_ui32._pack_ = 1 # source:False
struct__HsaMemFlags_0_ui32._fields_ = [
    ('NonPaged', ctypes.c_uint32, 1),
    ('CachePolicy', ctypes.c_uint32, 2),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('PageSize', ctypes.c_uint32, 2),
    ('HostAccess', ctypes.c_uint32, 1),
    ('NoSubstitute', ctypes.c_uint32, 1),
    ('GDSMemory', ctypes.c_uint32, 1),
    ('Scratch', ctypes.c_uint32, 1),
    ('AtomicAccessFull', ctypes.c_uint32, 1),
    ('AtomicAccessPartial', ctypes.c_uint32, 1),
    ('ExecuteAccess', ctypes.c_uint32, 1),
    ('CoarseGrain', ctypes.c_uint32, 1),
    ('AQLQueueMemory', ctypes.c_uint32, 1),
    ('FixedAddress', ctypes.c_uint32, 1),
    ('NoNUMABind', ctypes.c_uint32, 1),
    ('Uncached', ctypes.c_uint32, 1),
    ('NoAddress', ctypes.c_uint32, 1),
    ('OnlyAddress', ctypes.c_uint32, 1),
    ('ExtendedCoherent', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 11),
]

union__HsaMemFlags_0._pack_ = 1 # source:False
union__HsaMemFlags_0._fields_ = [
    ('ui32', struct__HsaMemFlags_0_ui32),
    ('Value', ctypes.c_uint32),
]

struct__HsaMemFlags._pack_ = 1 # source:False
struct__HsaMemFlags._anonymous_ = ('_0',)
struct__HsaMemFlags._fields_ = [
    ('_0', union__HsaMemFlags_0),
]

HsaMemFlags = struct__HsaMemFlags
try:
    hsaKmtAllocMemory = _libraries['libhsakmt.so'].hsaKmtAllocMemory
    hsaKmtAllocMemory.restype = HSAKMT_STATUS
    hsaKmtAllocMemory.argtypes = [HSAuint32, HSAuint64, HsaMemFlags, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsaKmtFreeMemory = _libraries['libhsakmt.so'].hsaKmtFreeMemory
    hsaKmtFreeMemory.restype = HSAKMT_STATUS
    hsaKmtFreeMemory.argtypes = [ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtAvailableMemory = _libraries['libhsakmt.so'].hsaKmtAvailableMemory
    hsaKmtAvailableMemory.restype = HSAKMT_STATUS
    hsaKmtAvailableMemory.argtypes = [HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemory = _libraries['libhsakmt.so'].hsaKmtRegisterMemory
    hsaKmtRegisterMemory.restype = HSAKMT_STATUS
    hsaKmtRegisterMemory.argtypes = [ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemoryToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterMemoryToNodes
    hsaKmtRegisterMemoryToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterMemoryToNodes.argtypes = [ctypes.POINTER(None), HSAuint64, HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemoryWithFlags = _libraries['libhsakmt.so'].hsaKmtRegisterMemoryWithFlags
    hsaKmtRegisterMemoryWithFlags.restype = HSAKMT_STATUS
    hsaKmtRegisterMemoryWithFlags.argtypes = [ctypes.POINTER(None), HSAuint64, HsaMemFlags]
except AttributeError:
    pass
class struct__HsaGraphicsResourceInfo(Structure):
    pass

struct__HsaGraphicsResourceInfo._pack_ = 1 # source:False
struct__HsaGraphicsResourceInfo._fields_ = [
    ('MemoryAddress', ctypes.POINTER(None)),
    ('SizeInBytes', ctypes.c_uint64),
    ('Metadata', ctypes.POINTER(None)),
    ('MetadataSizeInBytes', ctypes.c_uint32),
    ('NodeId', ctypes.c_uint32),
]

try:
    hsaKmtRegisterGraphicsHandleToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterGraphicsHandleToNodes
    hsaKmtRegisterGraphicsHandleToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterGraphicsHandleToNodes.argtypes = [HSAuint64, ctypes.POINTER(struct__HsaGraphicsResourceInfo), HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtExportDMABufHandle = _libraries['libhsakmt.so'].hsaKmtExportDMABufHandle
    hsaKmtExportDMABufHandle.restype = HSAKMT_STATUS
    hsaKmtExportDMABufHandle.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtShareMemory = _libraries['libhsakmt.so'].hsaKmtShareMemory
    hsaKmtShareMemory.restype = HSAKMT_STATUS
    hsaKmtShareMemory.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint32 * 8)]
except AttributeError:
    pass
try:
    hsaKmtRegisterSharedHandle = _libraries['libhsakmt.so'].hsaKmtRegisterSharedHandle
    hsaKmtRegisterSharedHandle.restype = HSAKMT_STATUS
    hsaKmtRegisterSharedHandle.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 8), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtRegisterSharedHandleToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterSharedHandleToNodes
    hsaKmtRegisterSharedHandleToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterSharedHandleToNodes.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 8), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct__HsaMemoryRange(Structure):
    pass

struct__HsaMemoryRange._pack_ = 1 # source:False
struct__HsaMemoryRange._fields_ = [
    ('MemoryAddress', ctypes.POINTER(None)),
    ('SizeInBytes', ctypes.c_uint64),
]

try:
    hsaKmtProcessVMRead = _libraries['libhsakmt.so'].hsaKmtProcessVMRead
    hsaKmtProcessVMRead.restype = HSAKMT_STATUS
    hsaKmtProcessVMRead.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtProcessVMWrite = _libraries['libhsakmt.so'].hsaKmtProcessVMWrite
    hsaKmtProcessVMWrite.restype = HSAKMT_STATUS
    hsaKmtProcessVMWrite.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtDeregisterMemory = _libraries['libhsakmt.so'].hsaKmtDeregisterMemory
    hsaKmtDeregisterMemory.restype = HSAKMT_STATUS
    hsaKmtDeregisterMemory.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtMapMemoryToGPU = _libraries['libhsakmt.so'].hsaKmtMapMemoryToGPU
    hsaKmtMapMemoryToGPU.restype = HSAKMT_STATUS
    hsaKmtMapMemoryToGPU.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
class struct__HsaMemMapFlags(Structure):
    pass

class union__HsaMemMapFlags_0(Union):
    pass

class struct__HsaMemMapFlags_0_ui32(Structure):
    pass

struct__HsaMemMapFlags_0_ui32._pack_ = 1 # source:False
struct__HsaMemMapFlags_0_ui32._fields_ = [
    ('Reserved1', ctypes.c_uint32, 1),
    ('CachePolicy', ctypes.c_uint32, 2),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('PageSize', ctypes.c_uint32, 2),
    ('HostAccess', ctypes.c_uint32, 1),
    ('Migrate', ctypes.c_uint32, 1),
    ('Probe', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 23),
]

union__HsaMemMapFlags_0._pack_ = 1 # source:False
union__HsaMemMapFlags_0._fields_ = [
    ('ui32', struct__HsaMemMapFlags_0_ui32),
    ('Value', ctypes.c_uint32),
]

struct__HsaMemMapFlags._pack_ = 1 # source:False
struct__HsaMemMapFlags._anonymous_ = ('_0',)
struct__HsaMemMapFlags._fields_ = [
    ('_0', union__HsaMemMapFlags_0),
]

HsaMemMapFlags = struct__HsaMemMapFlags
try:
    hsaKmtMapMemoryToGPUNodes = _libraries['libhsakmt.so'].hsaKmtMapMemoryToGPUNodes
    hsaKmtMapMemoryToGPUNodes.restype = HSAKMT_STATUS
    hsaKmtMapMemoryToGPUNodes.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint64), HsaMemMapFlags, HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtUnmapMemoryToGPU = _libraries['libhsakmt.so'].hsaKmtUnmapMemoryToGPU
    hsaKmtUnmapMemoryToGPU.restype = HSAKMT_STATUS
    hsaKmtUnmapMemoryToGPU.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtMapGraphicHandle = _libraries['libhsakmt.so'].hsaKmtMapGraphicHandle
    hsaKmtMapGraphicHandle.restype = HSAKMT_STATUS
    hsaKmtMapGraphicHandle.argtypes = [HSAuint32, HSAuint64, HSAuint64, HSAuint64, HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtUnmapGraphicHandle = _libraries['libhsakmt.so'].hsaKmtUnmapGraphicHandle
    hsaKmtUnmapGraphicHandle.restype = HSAKMT_STATUS
    hsaKmtUnmapGraphicHandle.argtypes = [HSAuint32, HSAuint64, HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtGetAMDGPUDeviceHandle = _libraries['libhsakmt.so'].hsaKmtGetAMDGPUDeviceHandle
    hsaKmtGetAMDGPUDeviceHandle.restype = HSAKMT_STATUS
    hsaKmtGetAMDGPUDeviceHandle.argtypes = [HSAuint32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsaKmtAllocQueueGWS = _libraries['libhsakmt.so'].hsaKmtAllocQueueGWS
    hsaKmtAllocQueueGWS.restype = HSAKMT_STATUS
    hsaKmtAllocQueueGWS.argtypes = [HSA_QUEUEID, HSAuint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgRegister = _libraries['libhsakmt.so'].hsaKmtDbgRegister
    hsaKmtDbgRegister.restype = HSAKMT_STATUS
    hsaKmtDbgRegister.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtDbgUnregister = _libraries['libhsakmt.so'].hsaKmtDbgUnregister
    hsaKmtDbgUnregister.restype = HSAKMT_STATUS
    hsaKmtDbgUnregister.argtypes = [HSAuint32]
except AttributeError:
    pass

# values for enumeration '_HSA_DBG_WAVEOP'
_HSA_DBG_WAVEOP__enumvalues = {
    1: 'HSA_DBG_WAVEOP_HALT',
    2: 'HSA_DBG_WAVEOP_RESUME',
    3: 'HSA_DBG_WAVEOP_KILL',
    4: 'HSA_DBG_WAVEOP_DEBUG',
    5: 'HSA_DBG_WAVEOP_TRAP',
    5: 'HSA_DBG_NUM_WAVEOP',
    4294967295: 'HSA_DBG_MAX_WAVEOP',
}
HSA_DBG_WAVEOP_HALT = 1
HSA_DBG_WAVEOP_RESUME = 2
HSA_DBG_WAVEOP_KILL = 3
HSA_DBG_WAVEOP_DEBUG = 4
HSA_DBG_WAVEOP_TRAP = 5
HSA_DBG_NUM_WAVEOP = 5
HSA_DBG_MAX_WAVEOP = 4294967295
_HSA_DBG_WAVEOP = ctypes.c_uint32 # enum
HSA_DBG_WAVEOP = _HSA_DBG_WAVEOP
HSA_DBG_WAVEOP__enumvalues = _HSA_DBG_WAVEOP__enumvalues

# values for enumeration '_HSA_DBG_WAVEMODE'
_HSA_DBG_WAVEMODE__enumvalues = {
    0: 'HSA_DBG_WAVEMODE_SINGLE',
    2: 'HSA_DBG_WAVEMODE_BROADCAST_PROCESS',
    3: 'HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU',
    3: 'HSA_DBG_NUM_WAVEMODE',
    4294967295: 'HSA_DBG_MAX_WAVEMODE',
}
HSA_DBG_WAVEMODE_SINGLE = 0
HSA_DBG_WAVEMODE_BROADCAST_PROCESS = 2
HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU = 3
HSA_DBG_NUM_WAVEMODE = 3
HSA_DBG_MAX_WAVEMODE = 4294967295
_HSA_DBG_WAVEMODE = ctypes.c_uint32 # enum
HSA_DBG_WAVEMODE = _HSA_DBG_WAVEMODE
HSA_DBG_WAVEMODE__enumvalues = _HSA_DBG_WAVEMODE__enumvalues
class struct__HsaDbgWaveMessage(Structure):
    pass

class union__HsaDbgWaveMessageAMD(Union):
    pass

class struct__HsaDbgWaveMsgAMDGen2(Structure):
    pass

struct__HsaDbgWaveMsgAMDGen2._pack_ = 1 # source:False
struct__HsaDbgWaveMsgAMDGen2._fields_ = [
    ('Value', ctypes.c_uint32),
    ('Reserved2', ctypes.c_uint32),
]

union__HsaDbgWaveMessageAMD._pack_ = 1 # source:False
union__HsaDbgWaveMessageAMD._fields_ = [
    ('WaveMsgInfoGen2', struct__HsaDbgWaveMsgAMDGen2),
]

struct__HsaDbgWaveMessage._pack_ = 1 # source:False
struct__HsaDbgWaveMessage._fields_ = [
    ('MemoryVA', ctypes.POINTER(None)),
    ('DbgWaveMsg', union__HsaDbgWaveMessageAMD),
]

try:
    hsaKmtDbgWavefrontControl = _libraries['libhsakmt.so'].hsaKmtDbgWavefrontControl
    hsaKmtDbgWavefrontControl.restype = HSAKMT_STATUS
    hsaKmtDbgWavefrontControl.argtypes = [HSAuint32, HSA_DBG_WAVEOP, HSA_DBG_WAVEMODE, HSAuint32, ctypes.POINTER(struct__HsaDbgWaveMessage)]
except AttributeError:
    pass

# values for enumeration '_HSA_DBG_WATCH_MODE'
_HSA_DBG_WATCH_MODE__enumvalues = {
    0: 'HSA_DBG_WATCH_READ',
    1: 'HSA_DBG_WATCH_NONREAD',
    2: 'HSA_DBG_WATCH_ATOMIC',
    3: 'HSA_DBG_WATCH_ALL',
    4: 'HSA_DBG_WATCH_NUM',
}
HSA_DBG_WATCH_READ = 0
HSA_DBG_WATCH_NONREAD = 1
HSA_DBG_WATCH_ATOMIC = 2
HSA_DBG_WATCH_ALL = 3
HSA_DBG_WATCH_NUM = 4
_HSA_DBG_WATCH_MODE = ctypes.c_uint32 # enum
try:
    hsaKmtDbgAddressWatch = _libraries['libhsakmt.so'].hsaKmtDbgAddressWatch
    hsaKmtDbgAddressWatch.restype = HSAKMT_STATUS
    hsaKmtDbgAddressWatch.argtypes = [HSAuint32, HSAuint32, _HSA_DBG_WATCH_MODE * 0, ctypes.POINTER(None) * 0, ctypes.c_uint64 * 0, ctypes.POINTER(struct__HsaEvent) * 0]
except AttributeError:
    pass
try:
    hsaKmtRuntimeEnable = _libraries['libhsakmt.so'].hsaKmtRuntimeEnable
    hsaKmtRuntimeEnable.restype = HSAKMT_STATUS
    hsaKmtRuntimeEnable.argtypes = [ctypes.POINTER(None), ctypes.c_bool]
except AttributeError:
    pass
try:
    hsaKmtRuntimeDisable = _libraries['libhsakmt.so'].hsaKmtRuntimeDisable
    hsaKmtRuntimeDisable.restype = HSAKMT_STATUS
    hsaKmtRuntimeDisable.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtGetRuntimeCapabilities = _libraries['libhsakmt.so'].hsaKmtGetRuntimeCapabilities
    hsaKmtGetRuntimeCapabilities.restype = HSAKMT_STATUS
    hsaKmtGetRuntimeCapabilities.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgEnable = _libraries['libhsakmt.so'].hsaKmtDbgEnable
    hsaKmtDbgEnable.restype = HSAKMT_STATUS
    hsaKmtDbgEnable.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgDisable = _libraries['libhsakmt.so'].hsaKmtDbgDisable
    hsaKmtDbgDisable.restype = HSAKMT_STATUS
    hsaKmtDbgDisable.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtDbgGetDeviceData = _libraries['libhsakmt.so'].hsaKmtDbgGetDeviceData
    hsaKmtDbgGetDeviceData.restype = HSAKMT_STATUS
    hsaKmtDbgGetDeviceData.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgGetQueueData = _libraries['libhsakmt.so'].hsaKmtDbgGetQueueData
    hsaKmtDbgGetQueueData.restype = HSAKMT_STATUS
    hsaKmtDbgGetQueueData.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool]
except AttributeError:
    pass
try:
    hsaKmtCheckRuntimeDebugSupport = _libraries['FIXME_STUB'].hsaKmtCheckRuntimeDebugSupport
    hsaKmtCheckRuntimeDebugSupport.restype = HSAKMT_STATUS
    hsaKmtCheckRuntimeDebugSupport.argtypes = []
except AttributeError:
    pass
class struct_kfd_ioctl_dbg_trap_args(Structure):
    pass

try:
    hsaKmtDebugTrapIoctl = _libraries['libhsakmt.so'].hsaKmtDebugTrapIoctl
    hsaKmtDebugTrapIoctl.restype = HSAKMT_STATUS
    hsaKmtDebugTrapIoctl.argtypes = [ctypes.POINTER(struct_kfd_ioctl_dbg_trap_args), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
class struct__HsaClockCounters(Structure):
    pass

struct__HsaClockCounters._pack_ = 1 # source:False
struct__HsaClockCounters._fields_ = [
    ('GPUClockCounter', ctypes.c_uint64),
    ('CPUClockCounter', ctypes.c_uint64),
    ('SystemClockCounter', ctypes.c_uint64),
    ('SystemClockFrequencyHz', ctypes.c_uint64),
]

try:
    hsaKmtGetClockCounters = _libraries['libhsakmt.so'].hsaKmtGetClockCounters
    hsaKmtGetClockCounters.restype = HSAKMT_STATUS
    hsaKmtGetClockCounters.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaClockCounters)]
except AttributeError:
    pass
class struct__HsaCounterProperties(Structure):
    pass

class struct__HsaCounterBlockProperties(Structure):
    pass

class struct__HSA_UUID(Structure):
    pass

struct__HSA_UUID._pack_ = 1 # source:False
struct__HSA_UUID._fields_ = [
    ('Data1', ctypes.c_uint32),
    ('Data2', ctypes.c_uint16),
    ('Data3', ctypes.c_uint16),
    ('Data4', ctypes.c_ubyte * 8),
]

class struct__HsaCounter(Structure):
    pass


# values for enumeration '_HSA_PROFILE_TYPE'
_HSA_PROFILE_TYPE__enumvalues = {
    0: 'HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE',
    1: 'HSA_PROFILE_TYPE_PRIVILEGED_STREAMING',
    2: 'HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE',
    3: 'HSA_PROFILE_TYPE_NONPRIV_STREAMING',
    4: 'HSA_PROFILE_TYPE_NUM',
    4294967295: 'HSA_PROFILE_TYPE_SIZE',
}
HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE = 0
HSA_PROFILE_TYPE_PRIVILEGED_STREAMING = 1
HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE = 2
HSA_PROFILE_TYPE_NONPRIV_STREAMING = 3
HSA_PROFILE_TYPE_NUM = 4
HSA_PROFILE_TYPE_SIZE = 4294967295
_HSA_PROFILE_TYPE = ctypes.c_uint32 # enum
class struct__HsaCounterFlags(Structure):
    pass

class union__HsaCounterFlags_0(Union):
    pass

class struct__HsaCounterFlags_0_ui32(Structure):
    pass

struct__HsaCounterFlags_0_ui32._pack_ = 1 # source:False
struct__HsaCounterFlags_0_ui32._fields_ = [
    ('Global', ctypes.c_uint32, 1),
    ('Resettable', ctypes.c_uint32, 1),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('Stream', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 28),
]

union__HsaCounterFlags_0._pack_ = 1 # source:False
union__HsaCounterFlags_0._fields_ = [
    ('ui32', struct__HsaCounterFlags_0_ui32),
    ('Value', ctypes.c_uint32),
]

struct__HsaCounterFlags._pack_ = 1 # source:False
struct__HsaCounterFlags._anonymous_ = ('_0',)
struct__HsaCounterFlags._fields_ = [
    ('_0', union__HsaCounterFlags_0),
]

struct__HsaCounter._pack_ = 1 # source:False
struct__HsaCounter._fields_ = [
    ('Type', _HSA_PROFILE_TYPE),
    ('CounterId', ctypes.c_uint64),
    ('CounterSizeInBits', ctypes.c_uint32),
    ('CounterMask', ctypes.c_uint64),
    ('Flags', struct__HsaCounterFlags),
    ('BlockIndex', ctypes.c_uint32),
]

struct__HsaCounterBlockProperties._pack_ = 1 # source:False
struct__HsaCounterBlockProperties._fields_ = [
    ('BlockId', struct__HSA_UUID),
    ('NumCounters', ctypes.c_uint32),
    ('NumConcurrent', ctypes.c_uint32),
    ('Counters', struct__HsaCounter * 1),
]

struct__HsaCounterProperties._pack_ = 1 # source:False
struct__HsaCounterProperties._fields_ = [
    ('NumBlocks', ctypes.c_uint32),
    ('NumConcurrent', ctypes.c_uint32),
    ('Blocks', struct__HsaCounterBlockProperties * 1),
]

try:
    hsaKmtPmcGetCounterProperties = _libraries['libhsakmt.so'].hsaKmtPmcGetCounterProperties
    hsaKmtPmcGetCounterProperties.restype = HSAKMT_STATUS
    hsaKmtPmcGetCounterProperties.argtypes = [HSAuint32, ctypes.POINTER(ctypes.POINTER(struct__HsaCounterProperties))]
except AttributeError:
    pass
class struct__HsaPmcTraceRoot(Structure):
    pass

struct__HsaPmcTraceRoot._pack_ = 1 # source:False
struct__HsaPmcTraceRoot._fields_ = [
    ('TraceBufferMinSizeBytes', ctypes.c_uint64),
    ('NumberOfPasses', ctypes.c_uint32),
    ('TraceId', ctypes.c_uint64),
]

try:
    hsaKmtPmcRegisterTrace = _libraries['libhsakmt.so'].hsaKmtPmcRegisterTrace
    hsaKmtPmcRegisterTrace.restype = HSAKMT_STATUS
    hsaKmtPmcRegisterTrace.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaCounter), ctypes.POINTER(struct__HsaPmcTraceRoot)]
except AttributeError:
    pass
HSATraceId = ctypes.c_uint64
try:
    hsaKmtPmcUnregisterTrace = _libraries['libhsakmt.so'].hsaKmtPmcUnregisterTrace
    hsaKmtPmcUnregisterTrace.restype = HSAKMT_STATUS
    hsaKmtPmcUnregisterTrace.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcAcquireTraceAccess = _libraries['libhsakmt.so'].hsaKmtPmcAcquireTraceAccess
    hsaKmtPmcAcquireTraceAccess.restype = HSAKMT_STATUS
    hsaKmtPmcAcquireTraceAccess.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcReleaseTraceAccess = _libraries['libhsakmt.so'].hsaKmtPmcReleaseTraceAccess
    hsaKmtPmcReleaseTraceAccess.restype = HSAKMT_STATUS
    hsaKmtPmcReleaseTraceAccess.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcStartTrace = _libraries['libhsakmt.so'].hsaKmtPmcStartTrace
    hsaKmtPmcStartTrace.restype = HSAKMT_STATUS
    hsaKmtPmcStartTrace.argtypes = [HSATraceId, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtPmcQueryTrace = _libraries['libhsakmt.so'].hsaKmtPmcQueryTrace
    hsaKmtPmcQueryTrace.restype = HSAKMT_STATUS
    hsaKmtPmcQueryTrace.argtypes = [HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcStopTrace = _libraries['libhsakmt.so'].hsaKmtPmcStopTrace
    hsaKmtPmcStopTrace.restype = HSAKMT_STATUS
    hsaKmtPmcStopTrace.argtypes = [HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtSetTrapHandler = _libraries['libhsakmt.so'].hsaKmtSetTrapHandler
    hsaKmtSetTrapHandler.restype = HSAKMT_STATUS
    hsaKmtSetTrapHandler.argtypes = [HSAuint32, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
class struct__HsaGpuTileConfig(Structure):
    pass

struct__HsaGpuTileConfig._pack_ = 1 # source:False
struct__HsaGpuTileConfig._fields_ = [
    ('TileConfig', ctypes.POINTER(ctypes.c_uint32)),
    ('MacroTileConfig', ctypes.POINTER(ctypes.c_uint32)),
    ('NumTileConfigs', ctypes.c_uint32),
    ('NumMacroTileConfigs', ctypes.c_uint32),
    ('GbAddrConfig', ctypes.c_uint32),
    ('NumBanks', ctypes.c_uint32),
    ('NumRanks', ctypes.c_uint32),
    ('Reserved', ctypes.c_uint32 * 7),
]

try:
    hsaKmtGetTileConfig = _libraries['libhsakmt.so'].hsaKmtGetTileConfig
    hsaKmtGetTileConfig.restype = HSAKMT_STATUS
    hsaKmtGetTileConfig.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaGpuTileConfig)]
except AttributeError:
    pass
class struct__HsaPointerInfo(Structure):
    pass


# values for enumeration '_HSA_POINTER_TYPE'
_HSA_POINTER_TYPE__enumvalues = {
    0: 'HSA_POINTER_UNKNOWN',
    1: 'HSA_POINTER_ALLOCATED',
    2: 'HSA_POINTER_REGISTERED_USER',
    3: 'HSA_POINTER_REGISTERED_GRAPHICS',
    4: 'HSA_POINTER_REGISTERED_SHARED',
    5: 'HSA_POINTER_RESERVED_ADDR',
}
HSA_POINTER_UNKNOWN = 0
HSA_POINTER_ALLOCATED = 1
HSA_POINTER_REGISTERED_USER = 2
HSA_POINTER_REGISTERED_GRAPHICS = 3
HSA_POINTER_REGISTERED_SHARED = 4
HSA_POINTER_RESERVED_ADDR = 5
_HSA_POINTER_TYPE = ctypes.c_uint32 # enum
struct__HsaPointerInfo._pack_ = 1 # source:False
struct__HsaPointerInfo._fields_ = [
    ('Type', _HSA_POINTER_TYPE),
    ('Node', ctypes.c_uint32),
    ('MemFlags', HsaMemFlags),
    ('CPUAddress', ctypes.POINTER(None)),
    ('GPUAddress', ctypes.c_uint64),
    ('SizeInBytes', ctypes.c_uint64),
    ('NRegisteredNodes', ctypes.c_uint32),
    ('NMappedNodes', ctypes.c_uint32),
    ('RegisteredNodes', ctypes.POINTER(ctypes.c_uint32)),
    ('MappedNodes', ctypes.POINTER(ctypes.c_uint32)),
    ('UserData', ctypes.POINTER(None)),
]

try:
    hsaKmtQueryPointerInfo = _libraries['libhsakmt.so'].hsaKmtQueryPointerInfo
    hsaKmtQueryPointerInfo.restype = HSAKMT_STATUS
    hsaKmtQueryPointerInfo.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct__HsaPointerInfo)]
except AttributeError:
    pass
try:
    hsaKmtSetMemoryUserData = _libraries['libhsakmt.so'].hsaKmtSetMemoryUserData
    hsaKmtSetMemoryUserData.restype = HSAKMT_STATUS
    hsaKmtSetMemoryUserData.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtSPMAcquire = _libraries['libhsakmt.so'].hsaKmtSPMAcquire
    hsaKmtSPMAcquire.restype = HSAKMT_STATUS
    hsaKmtSPMAcquire.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtSPMRelease = _libraries['libhsakmt.so'].hsaKmtSPMRelease
    hsaKmtSPMRelease.restype = HSAKMT_STATUS
    hsaKmtSPMRelease.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtSPMSetDestBuffer = _libraries['libhsakmt.so'].hsaKmtSPMSetDestBuffer
    hsaKmtSPMSetDestBuffer.restype = HSAKMT_STATUS
    hsaKmtSPMSetDestBuffer.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
class struct__HSA_SVM_ATTRIBUTE(Structure):
    pass

struct__HSA_SVM_ATTRIBUTE._pack_ = 1 # source:False
struct__HSA_SVM_ATTRIBUTE._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

try:
    hsaKmtSVMSetAttr = _libraries['libhsakmt.so'].hsaKmtSVMSetAttr
    hsaKmtSVMSetAttr.restype = HSAKMT_STATUS
    hsaKmtSVMSetAttr.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.c_uint32, ctypes.POINTER(struct__HSA_SVM_ATTRIBUTE)]
except AttributeError:
    pass
try:
    hsaKmtSVMGetAttr = _libraries['libhsakmt.so'].hsaKmtSVMGetAttr
    hsaKmtSVMGetAttr.restype = HSAKMT_STATUS
    hsaKmtSVMGetAttr.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.c_uint32, ctypes.POINTER(struct__HSA_SVM_ATTRIBUTE)]
except AttributeError:
    pass
HSAint32 = ctypes.c_int32
try:
    hsaKmtSetXNACKMode = _libraries['libhsakmt.so'].hsaKmtSetXNACKMode
    hsaKmtSetXNACKMode.restype = HSAKMT_STATUS
    hsaKmtSetXNACKMode.argtypes = [HSAint32]
except AttributeError:
    pass
try:
    hsaKmtGetXNACKMode = _libraries['libhsakmt.so'].hsaKmtGetXNACKMode
    hsaKmtGetXNACKMode.restype = HSAKMT_STATUS
    hsaKmtGetXNACKMode.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hsaKmtOpenSMI = _libraries['libhsakmt.so'].hsaKmtOpenSMI
    hsaKmtOpenSMI.restype = HSAKMT_STATUS
    hsaKmtOpenSMI.argtypes = [HSAuint32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hsaKmtReplaceAsanHeaderPage = _libraries['libhsakmt.so'].hsaKmtReplaceAsanHeaderPage
    hsaKmtReplaceAsanHeaderPage.restype = HSAKMT_STATUS
    hsaKmtReplaceAsanHeaderPage.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtReturnAsanHeaderPage = _libraries['libhsakmt.so'].hsaKmtReturnAsanHeaderPage
    hsaKmtReturnAsanHeaderPage.restype = HSAKMT_STATUS
    hsaKmtReturnAsanHeaderPage.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
__all__ = \
    ['HSAKMT_STATUS', 'HSAKMT_STATUS_BUFFER_TOO_SMALL',
    'HSAKMT_STATUS_DRIVER_MISMATCH', 'HSAKMT_STATUS_ERROR',
    'HSAKMT_STATUS_HSAMMU_UNAVAILABLE',
    'HSAKMT_STATUS_INVALID_HANDLE', 'HSAKMT_STATUS_INVALID_NODE_UNIT',
    'HSAKMT_STATUS_INVALID_PARAMETER',
    'HSAKMT_STATUS_KERNEL_ALREADY_OPENED',
    'HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR',
    'HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED',
    'HSAKMT_STATUS_MEMORY_ALIGNMENT',
    'HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED',
    'HSAKMT_STATUS_MEMORY_NOT_REGISTERED',
    'HSAKMT_STATUS_NOT_IMPLEMENTED', 'HSAKMT_STATUS_NOT_SUPPORTED',
    'HSAKMT_STATUS_NO_MEMORY', 'HSAKMT_STATUS_OUT_OF_RESOURCES',
    'HSAKMT_STATUS_SUCCESS', 'HSAKMT_STATUS_UNAVAILABLE',
    'HSAKMT_STATUS_WAIT_FAILURE', 'HSAKMT_STATUS_WAIT_TIMEOUT',
    'HSAKMT_STATUS__enumvalues', 'HSATraceId', 'HSA_DBG_MAX_WAVEMODE',
    'HSA_DBG_MAX_WAVEOP', 'HSA_DBG_NUM_WAVEMODE',
    'HSA_DBG_NUM_WAVEOP', 'HSA_DBG_WATCH_ALL', 'HSA_DBG_WATCH_ATOMIC',
    'HSA_DBG_WATCH_NONREAD', 'HSA_DBG_WATCH_NUM',
    'HSA_DBG_WATCH_READ', 'HSA_DBG_WAVEMODE',
    'HSA_DBG_WAVEMODE_BROADCAST_PROCESS',
    'HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU',
    'HSA_DBG_WAVEMODE_SINGLE', 'HSA_DBG_WAVEMODE__enumvalues',
    'HSA_DBG_WAVEOP', 'HSA_DBG_WAVEOP_DEBUG', 'HSA_DBG_WAVEOP_HALT',
    'HSA_DBG_WAVEOP_KILL', 'HSA_DBG_WAVEOP_RESUME',
    'HSA_DBG_WAVEOP_TRAP', 'HSA_DBG_WAVEOP__enumvalues',
    'HSA_DEVICE_CPU', 'HSA_DEVICE_GPU',
    'HSA_EVENTID_HW_EXCEPTION_ECC',
    'HSA_EVENTID_HW_EXCEPTION_GPU_HANG',
    'HSA_EVENTID_MEMORY_FATAL_PROCESS', 'HSA_EVENTID_MEMORY_FATAL_VM',
    'HSA_EVENTID_MEMORY_RECOVERABLE', 'HSA_EVENTTYPE_DEBUG_EVENT',
    'HSA_EVENTTYPE_DEVICESTATECHANGE',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_START',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP',
    'HSA_EVENTTYPE_HW_EXCEPTION', 'HSA_EVENTTYPE_MAXID',
    'HSA_EVENTTYPE_MEMORY', 'HSA_EVENTTYPE_NODECHANGE',
    'HSA_EVENTTYPE_NODECHANGE_ADD', 'HSA_EVENTTYPE_NODECHANGE_REMOVE',
    'HSA_EVENTTYPE_NODECHANGE_SIZE', 'HSA_EVENTTYPE_PROFILE_EVENT',
    'HSA_EVENTTYPE_QUEUE_EVENT', 'HSA_EVENTTYPE_SIGNAL',
    'HSA_EVENTTYPE_SYSTEM_EVENT', 'HSA_EVENTTYPE_TYPE_SIZE',
    'HSA_HEAPTYPE_DEVICE_SVM', 'HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE',
    'HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC', 'HSA_HEAPTYPE_GPU_GDS',
    'HSA_HEAPTYPE_GPU_LDS', 'HSA_HEAPTYPE_GPU_SCRATCH',
    'HSA_HEAPTYPE_MMIO_REMAP', 'HSA_HEAPTYPE_NUMHEAPTYPES',
    'HSA_HEAPTYPE_SIZE', 'HSA_HEAPTYPE_SYSTEM', 'HSA_IOLINKTYPE_AMBA',
    'HSA_IOLINKTYPE_HYPERTRANSPORT', 'HSA_IOLINKTYPE_MIPI',
    'HSA_IOLINKTYPE_NUMIOLINKTYPES', 'HSA_IOLINKTYPE_PCIEXPRESS',
    'HSA_IOLINKTYPE_SIZE', 'HSA_IOLINKTYPE_UNDEFINED',
    'HSA_IOLINK_TYPE_ETHERNET_RDMA', 'HSA_IOLINK_TYPE_GZ',
    'HSA_IOLINK_TYPE_INFINIBAND', 'HSA_IOLINK_TYPE_OTHER',
    'HSA_IOLINK_TYPE_QPI_1_1', 'HSA_IOLINK_TYPE_RAPID_IO',
    'HSA_IOLINK_TYPE_RDMA_OTHER', 'HSA_IOLINK_TYPE_RESERVED1',
    'HSA_IOLINK_TYPE_RESERVED2', 'HSA_IOLINK_TYPE_RESERVED3',
    'HSA_IOLINK_TYPE_XGMI', 'HSA_IOLINK_TYPE_XGOP',
    'HSA_POINTER_ALLOCATED', 'HSA_POINTER_REGISTERED_GRAPHICS',
    'HSA_POINTER_REGISTERED_SHARED', 'HSA_POINTER_REGISTERED_USER',
    'HSA_POINTER_RESERVED_ADDR', 'HSA_POINTER_UNKNOWN',
    'HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE',
    'HSA_PROFILE_TYPE_NONPRIV_STREAMING', 'HSA_PROFILE_TYPE_NUM',
    'HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE',
    'HSA_PROFILE_TYPE_PRIVILEGED_STREAMING', 'HSA_PROFILE_TYPE_SIZE',
    'HSA_QUEUEID', 'HSA_QUEUE_COMPUTE', 'HSA_QUEUE_COMPUTE_AQL',
    'HSA_QUEUE_COMPUTE_OS', 'HSA_QUEUE_DMA_AQL',
    'HSA_QUEUE_DMA_AQL_XGMI', 'HSA_QUEUE_MULTIMEDIA_DECODE',
    'HSA_QUEUE_MULTIMEDIA_DECODE_OS', 'HSA_QUEUE_MULTIMEDIA_ENCODE',
    'HSA_QUEUE_MULTIMEDIA_ENCODE_OS', 'HSA_QUEUE_PRIORITY',
    'HSA_QUEUE_PRIORITY_ABOVE_NORMAL',
    'HSA_QUEUE_PRIORITY_BELOW_NORMAL', 'HSA_QUEUE_PRIORITY_HIGH',
    'HSA_QUEUE_PRIORITY_LOW', 'HSA_QUEUE_PRIORITY_MAXIMUM',
    'HSA_QUEUE_PRIORITY_MINIMUM', 'HSA_QUEUE_PRIORITY_NORMAL',
    'HSA_QUEUE_PRIORITY_NUM_PRIORITY', 'HSA_QUEUE_PRIORITY_SIZE',
    'HSA_QUEUE_PRIORITY__enumvalues', 'HSA_QUEUE_SDMA',
    'HSA_QUEUE_SDMA_OS', 'HSA_QUEUE_SDMA_XGMI', 'HSA_QUEUE_TYPE',
    'HSA_QUEUE_TYPE_SIZE', 'HSA_QUEUE_TYPE__enumvalues', 'HSAint32',
    'HSAuint32', 'HSAuint64', 'HsaMemFlags', 'HsaMemMapFlags',
    'MAX_HSA_DEVICE', '_HSAKMT_STATUS', '_HSA_DBG_WATCH_MODE',
    '_HSA_DBG_WAVEMODE', '_HSA_DBG_WAVEOP', '_HSA_DEVICE',
    '_HSA_EVENTID_HW_EXCEPTION_CAUSE', '_HSA_EVENTID_MEMORYFLAGS',
    '_HSA_EVENTTYPE', '_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS',
    '_HSA_EVENTTYPE_NODECHANGE_FLAGS', '_HSA_HEAPTYPE',
    '_HSA_IOLINKTYPE', '_HSA_POINTER_TYPE', '_HSA_PROFILE_TYPE',
    '_HSA_QUEUE_PRIORITY', '_HSA_QUEUE_TYPE',
    'hsaKmtAcquireSystemProperties', 'hsaKmtAllocMemory',
    'hsaKmtAllocQueueGWS', 'hsaKmtAvailableMemory',
    'hsaKmtCheckRuntimeDebugSupport', 'hsaKmtCloseKFD',
    'hsaKmtCreateEvent', 'hsaKmtCreateQueue', 'hsaKmtDbgAddressWatch',
    'hsaKmtDbgDisable', 'hsaKmtDbgEnable', 'hsaKmtDbgGetDeviceData',
    'hsaKmtDbgGetQueueData', 'hsaKmtDbgRegister',
    'hsaKmtDbgUnregister', 'hsaKmtDbgWavefrontControl',
    'hsaKmtDebugTrapIoctl', 'hsaKmtDeregisterMemory',
    'hsaKmtDestroyEvent', 'hsaKmtDestroyQueue',
    'hsaKmtExportDMABufHandle', 'hsaKmtFreeMemory',
    'hsaKmtGetAMDGPUDeviceHandle', 'hsaKmtGetClockCounters',
    'hsaKmtGetNodeCacheProperties', 'hsaKmtGetNodeIoLinkProperties',
    'hsaKmtGetNodeMemoryProperties', 'hsaKmtGetNodeProperties',
    'hsaKmtGetQueueInfo', 'hsaKmtGetRuntimeCapabilities',
    'hsaKmtGetTileConfig', 'hsaKmtGetVersion', 'hsaKmtGetXNACKMode',
    'hsaKmtMapGraphicHandle', 'hsaKmtMapMemoryToGPU',
    'hsaKmtMapMemoryToGPUNodes', 'hsaKmtOpenKFD', 'hsaKmtOpenSMI',
    'hsaKmtPmcAcquireTraceAccess', 'hsaKmtPmcGetCounterProperties',
    'hsaKmtPmcQueryTrace', 'hsaKmtPmcRegisterTrace',
    'hsaKmtPmcReleaseTraceAccess', 'hsaKmtPmcStartTrace',
    'hsaKmtPmcStopTrace', 'hsaKmtPmcUnregisterTrace',
    'hsaKmtProcessVMRead', 'hsaKmtProcessVMWrite',
    'hsaKmtQueryEventState', 'hsaKmtQueryPointerInfo',
    'hsaKmtRegisterGraphicsHandleToNodes', 'hsaKmtRegisterMemory',
    'hsaKmtRegisterMemoryToNodes', 'hsaKmtRegisterMemoryWithFlags',
    'hsaKmtRegisterSharedHandle', 'hsaKmtRegisterSharedHandleToNodes',
    'hsaKmtReleaseSystemProperties', 'hsaKmtReplaceAsanHeaderPage',
    'hsaKmtReportQueue', 'hsaKmtResetEvent',
    'hsaKmtReturnAsanHeaderPage', 'hsaKmtRuntimeDisable',
    'hsaKmtRuntimeEnable', 'hsaKmtSPMAcquire', 'hsaKmtSPMRelease',
    'hsaKmtSPMSetDestBuffer', 'hsaKmtSVMGetAttr', 'hsaKmtSVMSetAttr',
    'hsaKmtSetEvent', 'hsaKmtSetMemoryPolicy',
    'hsaKmtSetMemoryUserData', 'hsaKmtSetQueueCUMask',
    'hsaKmtSetTrapHandler', 'hsaKmtSetXNACKMode', 'hsaKmtShareMemory',
    'hsaKmtUnmapGraphicHandle', 'hsaKmtUnmapMemoryToGPU',
    'hsaKmtUpdateQueue', 'hsaKmtWaitOnEvent', 'hsaKmtWaitOnEvent_Ext',
    'hsaKmtWaitOnMultipleEvents', 'hsaKmtWaitOnMultipleEvents_Ext',
    'struct__HSA_SVM_ATTRIBUTE', 'struct__HSA_UUID',
    'struct__HaCacheProperties', 'struct__HsaAccessAttributeFailure',
    'struct__HsaClockCounters', 'struct__HsaCounter',
    'struct__HsaCounterBlockProperties', 'struct__HsaCounterFlags',
    'struct__HsaCounterFlags_0_ui32', 'struct__HsaCounterProperties',
    'struct__HsaDbgWaveMessage', 'struct__HsaDbgWaveMsgAMDGen2',
    'struct__HsaDeviceStateChange', 'struct__HsaEvent',
    'struct__HsaEventData', 'struct__HsaEventDescriptor',
    'struct__HsaGpuTileConfig', 'struct__HsaGraphicsResourceInfo',
    'struct__HsaHwException', 'struct__HsaIoLinkProperties',
    'struct__HsaMemFlags', 'struct__HsaMemFlags_0_ui32',
    'struct__HsaMemMapFlags', 'struct__HsaMemMapFlags_0_ui32',
    'struct__HsaMemoryAccessFault', 'struct__HsaMemoryProperties',
    'struct__HsaMemoryProperties_0_ui32', 'struct__HsaMemoryRange',
    'struct__HsaNodeChange', 'struct__HsaNodeProperties',
    'struct__HsaPmcTraceRoot', 'struct__HsaPointerInfo',
    'struct__HsaQueueReport', 'struct__HsaQueueResource',
    'struct__HsaSyncVar', 'struct__HsaSystemProperties',
    'struct__HsaVersionInfo', 'struct_c__SA_HsaQueueInfo',
    'struct_c__SA_HsaUserContextSaveAreaHeader',
    'struct_c__UA_HSA_CAPABILITY_ui32',
    'struct_c__UA_HSA_DEBUG_PROPERTIES_0',
    'struct_c__UA_HSA_ENGINE_ID_ui32',
    'struct_c__UA_HSA_ENGINE_VERSION_0',
    'struct_c__UA_HSA_LINKPROPERTY_ui32',
    'struct_c__UA_HSA_MEMORYPROPERTY_ui32',
    'struct_c__UA_HsaCacheType_ui32',
    'struct_kfd_ioctl_dbg_trap_args', 'union__HsaCounterFlags_0',
    'union__HsaDbgWaveMessageAMD', 'union__HsaEventData_EventData',
    'union__HsaMemFlags_0', 'union__HsaMemMapFlags_0',
    'union__HsaMemoryProperties_0', 'union__HsaQueueResource_0',
    'union__HsaQueueResource_1', 'union__HsaQueueResource_2',
    'union__HsaSyncVar_SyncVar', 'union_c__UA_HSA_CAPABILITY',
    'union_c__UA_HSA_DEBUG_PROPERTIES', 'union_c__UA_HSA_ENGINE_ID',
    'union_c__UA_HSA_ENGINE_VERSION', 'union_c__UA_HSA_LINKPROPERTY',
    'union_c__UA_HSA_MEMORYPROPERTY', 'union_c__UA_HsaCacheType']
