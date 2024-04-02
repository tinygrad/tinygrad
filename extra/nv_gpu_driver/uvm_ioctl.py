# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/src/common/inc', '-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/kernel-open/nvidia-uvm', '-I/home/nimlgen/cuda_ioctl_sniffer/open-gpu-kernel-modules/kernel-open/common/inc']
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



_UVM_IOCTL_H = True # macro
# def UVM_IOCTL_BASE(i):  # macro
#    return i  
UVM_RESERVE_VA = ['i', '(', '1', ')'] # macro
UVM_RELEASE_VA = ['i', '(', '2', ')'] # macro
UVM_REGION_COMMIT = ['i', '(', '3', ')'] # macro
UVM_REGION_DECOMMIT = ['i', '(', '4', ')'] # macro
UVM_REGION_SET_STREAM = ['i', '(', '5', ')'] # macro
UVM_SET_STREAM_RUNNING = ['i', '(', '6', ')'] # macro
UVM_MAX_STREAMS_PER_IOCTL_CALL = 32 # macro
UVM_SET_STREAM_STOPPED = ['i', '(', '7', ')'] # macro
UVM_RUN_TEST = ['i', '(', '9', ')'] # macro
UVM_EVENTS_OFFSET_BASE = (1<<63) # macro
UVM_COUNTERS_OFFSET_BASE = (1<<62) # macro
UVM_ADD_SESSION = ['i', '(', '10', ')'] # macro
UVM_REMOVE_SESSION = ['i', '(', '11', ')'] # macro
UVM_MAX_COUNTERS_PER_IOCTL_CALL = 32 # macro
UVM_ENABLE_COUNTERS = ['i', '(', '12', ')'] # macro
UVM_MAP_COUNTER = ['i', '(', '13', ')'] # macro
UVM_CREATE_EVENT_QUEUE = ['i', '(', '14', ')'] # macro
UVM_REMOVE_EVENT_QUEUE = ['i', '(', '15', ')'] # macro
UVM_MAP_EVENT_QUEUE = ['i', '(', '16', ')'] # macro
UVM_EVENT_CTRL = ['i', '(', '17', ')'] # macro
UVM_REGISTER_MPS_SERVER = ['i', '(', '18', ')'] # macro
UVM_REGISTER_MPS_CLIENT = ['i', '(', '19', ')'] # macro
UVM_GET_GPU_UUID_TABLE = ['i', '(', '20', ')'] # macro
UVM_CREATE_RANGE_GROUP = ['i', '(', '23', ')'] # macro
UVM_DESTROY_RANGE_GROUP = ['i', '(', '24', ')'] # macro
UVM_REGISTER_GPU_VASPACE = ['i', '(', '25', ')'] # macro
UVM_UNREGISTER_GPU_VASPACE = ['i', '(', '26', ')'] # macro
UVM_REGISTER_CHANNEL = ['i', '(', '27', ')'] # macro
UVM_UNREGISTER_CHANNEL = ['i', '(', '28', ')'] # macro
UVM_ENABLE_PEER_ACCESS = ['i', '(', '29', ')'] # macro
UVM_DISABLE_PEER_ACCESS = ['i', '(', '30', ')'] # macro
UVM_SET_RANGE_GROUP = ['i', '(', '31', ')'] # macro
UVM_MAP_EXTERNAL_ALLOCATION = ['i', '(', '33', ')'] # macro
UVM_FREE = ['i', '(', '34', ')'] # macro
UVM_MEM_MAP = ['i', '(', '35', ')'] # macro
UVM_DEBUG_ACCESS_MEMORY = ['i', '(', '36', ')'] # macro
UVM_REGISTER_GPU = ['i', '(', '37', ')'] # macro
UVM_UNREGISTER_GPU = ['i', '(', '38', ')'] # macro
UVM_PAGEABLE_MEM_ACCESS = ['i', '(', '39', ')'] # macro
UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL = 32 # macro
UVM_PREVENT_MIGRATION_RANGE_GROUPS = ['i', '(', '40', ')'] # macro
UVM_ALLOW_MIGRATION_RANGE_GROUPS = ['i', '(', '41', ')'] # macro
UVM_SET_PREFERRED_LOCATION = ['i', '(', '42', ')'] # macro
UVM_UNSET_PREFERRED_LOCATION = ['i', '(', '43', ')'] # macro
UVM_ENABLE_READ_DUPLICATION = ['i', '(', '44', ')'] # macro
UVM_DISABLE_READ_DUPLICATION = ['i', '(', '45', ')'] # macro
UVM_SET_ACCESSED_BY = ['i', '(', '46', ')'] # macro
UVM_UNSET_ACCESSED_BY = ['i', '(', '47', ')'] # macro
UVM_MIGRATE_FLAG_ASYNC = 0x00000001 # macro
UVM_MIGRATE_FLAG_SKIP_CPU_MAP = 0x00000002 # macro
UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE = 0x00000004 # macro
UVM_MIGRATE_FLAGS_TEST_ALL = (0x00000002|0x00000004) # macro
UVM_MIGRATE_FLAGS_ALL = (0x00000001|(0x00000002|0x00000004)) # macro
UVM_MIGRATE = ['i', '(', '51', ')'] # macro
UVM_MIGRATE_RANGE_GROUP = ['i', '(', '53', ')'] # macro
UVM_ENABLE_SYSTEM_WIDE_ATOMICS = ['i', '(', '54', ')'] # macro
UVM_DISABLE_SYSTEM_WIDE_ATOMICS = ['i', '(', '55', ')'] # macro
UVM_TOOLS_INIT_EVENT_TRACKER = ['i', '(', '56', ')'] # macro
UVM_TOOLS_SET_NOTIFICATION_THRESHOLD = ['i', '(', '57', ')'] # macro
UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS = ['i', '(', '58', ')'] # macro
UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS = ['i', '(', '59', ')'] # macro
UVM_TOOLS_ENABLE_COUNTERS = ['i', '(', '60', ')'] # macro
UVM_TOOLS_DISABLE_COUNTERS = ['i', '(', '61', ')'] # macro
UVM_TOOLS_READ_PROCESS_MEMORY = ['i', '(', '62', ')'] # macro
UVM_TOOLS_WRITE_PROCESS_MEMORY = ['i', '(', '63', ')'] # macro
UVM_TOOLS_GET_PROCESSOR_UUID_TABLE = ['i', '(', '64', ')'] # macro
UVM_MAP_DYNAMIC_PARALLELISM_REGION = ['i', '(', '65', ')'] # macro
UVM_UNMAP_EXTERNAL = ['i', '(', '66', ')'] # macro
UVM_TOOLS_FLUSH_EVENTS = ['i', '(', '67', ')'] # macro
UVM_ALLOC_SEMAPHORE_POOL = ['i', '(', '68', ')'] # macro
UVM_CLEAN_UP_ZOMBIE_RESOURCES = ['i', '(', '69', ')'] # macro
UVM_PAGEABLE_MEM_ACCESS_ON_GPU = ['i', '(', '70', ')'] # macro
UVM_POPULATE_PAGEABLE = ['i', '(', '71', ')'] # macro
UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED = 0x00000001 # macro
UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK = 0x00000002 # macro
UVM_POPULATE_PAGEABLE_FLAGS_TEST_ALL = (0x00000001|0x00000002) # macro
UVM_POPULATE_PAGEABLE_FLAGS_ALL = (0x00000001|0x00000002) # macro
UVM_VALIDATE_VA_RANGE = ['i', '(', '72', ')'] # macro
UVM_CREATE_EXTERNAL_RANGE = ['i', '(', '73', ')'] # macro
UVM_MAP_EXTERNAL_SPARSE = ['i', '(', '74', ')'] # macro
UVM_MM_INITIALIZE = ['i', '(', '75', ')'] # macro
UVM_IS_8_SUPPORTED = ['i', '(', '2047', ')'] # macro
class struct_c__SA_UVM_RESERVE_VA_PARAMS(Structure):
    pass

struct_c__SA_UVM_RESERVE_VA_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_RESERVE_VA_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_RESERVE_VA_PARAMS = struct_c__SA_UVM_RESERVE_VA_PARAMS
class struct_c__SA_UVM_RELEASE_VA_PARAMS(Structure):
    pass

struct_c__SA_UVM_RELEASE_VA_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_RELEASE_VA_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_RELEASE_VA_PARAMS = struct_c__SA_UVM_RELEASE_VA_PARAMS
class struct_c__SA_UVM_REGION_COMMIT_PARAMS(Structure):
    pass

class struct_nv_uuid(Structure):
    pass

struct_nv_uuid._pack_ = 1 # source:False
struct_nv_uuid._fields_ = [
    ('uuid', ctypes.c_ubyte * 16),
]

struct_c__SA_UVM_REGION_COMMIT_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGION_COMMIT_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('streamId', ctypes.c_uint64),
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_REGION_COMMIT_PARAMS = struct_c__SA_UVM_REGION_COMMIT_PARAMS
class struct_c__SA_UVM_REGION_DECOMMIT_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGION_DECOMMIT_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGION_DECOMMIT_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_REGION_DECOMMIT_PARAMS = struct_c__SA_UVM_REGION_DECOMMIT_PARAMS
class struct_c__SA_UVM_REGION_SET_STREAM_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGION_SET_STREAM_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGION_SET_STREAM_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('newStreamId', ctypes.c_uint64),
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_REGION_SET_STREAM_PARAMS = struct_c__SA_UVM_REGION_SET_STREAM_PARAMS
class struct_c__SA_UVM_SET_STREAM_RUNNING_PARAMS(Structure):
    pass

struct_c__SA_UVM_SET_STREAM_RUNNING_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_SET_STREAM_RUNNING_PARAMS._fields_ = [
    ('streamId', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_SET_STREAM_RUNNING_PARAMS = struct_c__SA_UVM_SET_STREAM_RUNNING_PARAMS
class struct_c__SA_UVM_SET_STREAM_STOPPED_PARAMS(Structure):
    pass

struct_c__SA_UVM_SET_STREAM_STOPPED_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_SET_STREAM_STOPPED_PARAMS._fields_ = [
    ('streamIdArray', ctypes.c_uint64 * 32),
    ('nStreams', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_SET_STREAM_STOPPED_PARAMS = struct_c__SA_UVM_SET_STREAM_STOPPED_PARAMS
class struct_c__SA_UVM_RUN_TEST_PARAMS(Structure):
    pass

class struct_c__SA_UVM_RUN_TEST_PARAMS_multiGpu(Structure):
    pass

struct_c__SA_UVM_RUN_TEST_PARAMS_multiGpu._pack_ = 1 # source:False
struct_c__SA_UVM_RUN_TEST_PARAMS_multiGpu._fields_ = [
    ('peerGpuUuid', struct_nv_uuid),
    ('peerId', ctypes.c_uint32),
]

struct_c__SA_UVM_RUN_TEST_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_RUN_TEST_PARAMS._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('test', ctypes.c_uint32),
    ('multiGpu', struct_c__SA_UVM_RUN_TEST_PARAMS_multiGpu),
    ('rmStatus', ctypes.c_uint32),
]

UVM_RUN_TEST_PARAMS = struct_c__SA_UVM_RUN_TEST_PARAMS
class struct_c__SA_UVM_ADD_SESSION_PARAMS(Structure):
    pass

struct_c__SA_UVM_ADD_SESSION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ADD_SESSION_PARAMS._fields_ = [
    ('pidTarget', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('countersBaseAddress', ctypes.POINTER(None)),
    ('sessionIndex', ctypes.c_int32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_ADD_SESSION_PARAMS = struct_c__SA_UVM_ADD_SESSION_PARAMS
class struct_c__SA_UVM_REMOVE_SESSION_PARAMS(Structure):
    pass

struct_c__SA_UVM_REMOVE_SESSION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REMOVE_SESSION_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_REMOVE_SESSION_PARAMS = struct_c__SA_UVM_REMOVE_SESSION_PARAMS
class struct_c__SA_UVM_ENABLE_COUNTERS_PARAMS(Structure):
    pass

class struct_c__SA_UvmCounterConfig(Structure):
    pass

struct_c__SA_UvmCounterConfig._pack_ = 1 # source:False
struct_c__SA_UvmCounterConfig._fields_ = [
    ('scope', ctypes.c_uint32),
    ('name', ctypes.c_uint32),
    ('gpuid', struct_nv_uuid),
    ('state', ctypes.c_uint32),
]

struct_c__SA_UVM_ENABLE_COUNTERS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ENABLE_COUNTERS_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('config', struct_c__SA_UvmCounterConfig * 32),
    ('count', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_ENABLE_COUNTERS_PARAMS = struct_c__SA_UVM_ENABLE_COUNTERS_PARAMS
class struct_c__SA_UVM_MAP_COUNTER_PARAMS(Structure):
    pass

struct_c__SA_UVM_MAP_COUNTER_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MAP_COUNTER_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('scope', ctypes.c_uint32),
    ('counterName', ctypes.c_uint32),
    ('gpuUuid', struct_nv_uuid),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('addr', ctypes.POINTER(None)),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

UVM_MAP_COUNTER_PARAMS = struct_c__SA_UVM_MAP_COUNTER_PARAMS
class struct_c__SA_UVM_CREATE_EVENT_QUEUE_PARAMS(Structure):
    pass

struct_c__SA_UVM_CREATE_EVENT_QUEUE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_CREATE_EVENT_QUEUE_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('eventQueueIndex', ctypes.c_uint32),
    ('queueSize', ctypes.c_uint64),
    ('notificationCount', ctypes.c_uint64),
    ('timeStampType', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_CREATE_EVENT_QUEUE_PARAMS = struct_c__SA_UVM_CREATE_EVENT_QUEUE_PARAMS
class struct_c__SA_UVM_REMOVE_EVENT_QUEUE_PARAMS(Structure):
    pass

struct_c__SA_UVM_REMOVE_EVENT_QUEUE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REMOVE_EVENT_QUEUE_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('eventQueueIndex', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_REMOVE_EVENT_QUEUE_PARAMS = struct_c__SA_UVM_REMOVE_EVENT_QUEUE_PARAMS
class struct_c__SA_UVM_MAP_EVENT_QUEUE_PARAMS(Structure):
    pass

struct_c__SA_UVM_MAP_EVENT_QUEUE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MAP_EVENT_QUEUE_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('eventQueueIndex', ctypes.c_uint32),
    ('userRODataAddr', ctypes.POINTER(None)),
    ('userRWDataAddr', ctypes.POINTER(None)),
    ('readIndexAddr', ctypes.POINTER(None)),
    ('writeIndexAddr', ctypes.POINTER(None)),
    ('queueBufferAddr', ctypes.POINTER(None)),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_MAP_EVENT_QUEUE_PARAMS = struct_c__SA_UVM_MAP_EVENT_QUEUE_PARAMS
class struct_c__SA_UVM_EVENT_CTRL_PARAMS(Structure):
    pass

struct_c__SA_UVM_EVENT_CTRL_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_EVENT_CTRL_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('eventQueueIndex', ctypes.c_uint32),
    ('eventType', ctypes.c_int32),
    ('enable', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_EVENT_CTRL_PARAMS = struct_c__SA_UVM_EVENT_CTRL_PARAMS
class struct_c__SA_UVM_REGISTER_MPS_SERVER_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGISTER_MPS_SERVER_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGISTER_MPS_SERVER_PARAMS._fields_ = [
    ('gpuUuidArray', struct_nv_uuid * 32),
    ('numGpus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('serverId', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

UVM_REGISTER_MPS_SERVER_PARAMS = struct_c__SA_UVM_REGISTER_MPS_SERVER_PARAMS
class struct_c__SA_UVM_REGISTER_MPS_CLIENT_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGISTER_MPS_CLIENT_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGISTER_MPS_CLIENT_PARAMS._fields_ = [
    ('serverId', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_REGISTER_MPS_CLIENT_PARAMS = struct_c__SA_UVM_REGISTER_MPS_CLIENT_PARAMS
class struct_c__SA_UVM_GET_GPU_UUID_TABLE_PARAMS(Structure):
    pass

struct_c__SA_UVM_GET_GPU_UUID_TABLE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_GET_GPU_UUID_TABLE_PARAMS._fields_ = [
    ('gpuUuidArray', struct_nv_uuid * 32),
    ('validCount', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_GET_GPU_UUID_TABLE_PARAMS = struct_c__SA_UVM_GET_GPU_UUID_TABLE_PARAMS
class struct_c__SA_UVM_CREATE_RANGE_GROUP_PARAMS(Structure):
    pass

struct_c__SA_UVM_CREATE_RANGE_GROUP_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_CREATE_RANGE_GROUP_PARAMS._fields_ = [
    ('rangeGroupId', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_CREATE_RANGE_GROUP_PARAMS = struct_c__SA_UVM_CREATE_RANGE_GROUP_PARAMS
class struct_c__SA_UVM_DESTROY_RANGE_GROUP_PARAMS(Structure):
    pass

struct_c__SA_UVM_DESTROY_RANGE_GROUP_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_DESTROY_RANGE_GROUP_PARAMS._fields_ = [
    ('rangeGroupId', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_DESTROY_RANGE_GROUP_PARAMS = struct_c__SA_UVM_DESTROY_RANGE_GROUP_PARAMS
class struct_c__SA_UVM_REGISTER_GPU_VASPACE_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGISTER_GPU_VASPACE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGISTER_GPU_VASPACE_PARAMS._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('rmCtrlFd', ctypes.c_int32),
    ('hClient', ctypes.c_uint32),
    ('hVaSpace', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_REGISTER_GPU_VASPACE_PARAMS = struct_c__SA_UVM_REGISTER_GPU_VASPACE_PARAMS
class struct_c__SA_UVM_UNREGISTER_GPU_VASPACE_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNREGISTER_GPU_VASPACE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNREGISTER_GPU_VASPACE_PARAMS._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_UNREGISTER_GPU_VASPACE_PARAMS = struct_c__SA_UVM_UNREGISTER_GPU_VASPACE_PARAMS
class struct_c__SA_UVM_REGISTER_CHANNEL_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGISTER_CHANNEL_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGISTER_CHANNEL_PARAMS._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('rmCtrlFd', ctypes.c_int32),
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

UVM_REGISTER_CHANNEL_PARAMS = struct_c__SA_UVM_REGISTER_CHANNEL_PARAMS
class struct_c__SA_UVM_UNREGISTER_CHANNEL_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNREGISTER_CHANNEL_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNREGISTER_CHANNEL_PARAMS._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_UNREGISTER_CHANNEL_PARAMS = struct_c__SA_UVM_UNREGISTER_CHANNEL_PARAMS
class struct_c__SA_UVM_ENABLE_PEER_ACCESS_PARAMS(Structure):
    pass

struct_c__SA_UVM_ENABLE_PEER_ACCESS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ENABLE_PEER_ACCESS_PARAMS._fields_ = [
    ('gpuUuidA', struct_nv_uuid),
    ('gpuUuidB', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_ENABLE_PEER_ACCESS_PARAMS = struct_c__SA_UVM_ENABLE_PEER_ACCESS_PARAMS
class struct_c__SA_UVM_DISABLE_PEER_ACCESS_PARAMS(Structure):
    pass

struct_c__SA_UVM_DISABLE_PEER_ACCESS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_DISABLE_PEER_ACCESS_PARAMS._fields_ = [
    ('gpuUuidA', struct_nv_uuid),
    ('gpuUuidB', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_DISABLE_PEER_ACCESS_PARAMS = struct_c__SA_UVM_DISABLE_PEER_ACCESS_PARAMS
class struct_c__SA_UVM_SET_RANGE_GROUP_PARAMS(Structure):
    pass

struct_c__SA_UVM_SET_RANGE_GROUP_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_SET_RANGE_GROUP_PARAMS._fields_ = [
    ('rangeGroupId', ctypes.c_uint64),
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_SET_RANGE_GROUP_PARAMS = struct_c__SA_UVM_SET_RANGE_GROUP_PARAMS
class struct_c__SA_UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(Structure):
    pass

class struct_c__SA_UvmGpuMappingAttributes(Structure):
    pass

struct_c__SA_UvmGpuMappingAttributes._pack_ = 1 # source:False
struct_c__SA_UvmGpuMappingAttributes._fields_ = [
    ('gpuUuid', struct_nv_uuid),
    ('gpuMappingType', ctypes.c_uint32),
    ('gpuCachingType', ctypes.c_uint32),
    ('gpuFormatType', ctypes.c_uint32),
    ('gpuElementBits', ctypes.c_uint32),
    ('gpuCompressionType', ctypes.c_uint32),
]

struct_c__SA_UVM_MAP_EXTERNAL_ALLOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MAP_EXTERNAL_ALLOCATION_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('perGpuAttributes', struct_c__SA_UvmGpuMappingAttributes * 256),
    ('gpuAttributesCount', ctypes.c_uint64),
    ('rmCtrlFd', ctypes.c_int32),
    ('hClient', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = struct_c__SA_UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
class struct_c__SA_UVM_FREE_PARAMS(Structure):
    pass

struct_c__SA_UVM_FREE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_FREE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_FREE_PARAMS = struct_c__SA_UVM_FREE_PARAMS
class struct_c__SA_UVM_MEM_MAP_PARAMS(Structure):
    pass

struct_c__SA_UVM_MEM_MAP_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MEM_MAP_PARAMS._fields_ = [
    ('regionBase', ctypes.POINTER(None)),
    ('regionLength', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_MEM_MAP_PARAMS = struct_c__SA_UVM_MEM_MAP_PARAMS
class struct_c__SA_UVM_DEBUG_ACCESS_MEMORY_PARAMS(Structure):
    pass

struct_c__SA_UVM_DEBUG_ACCESS_MEMORY_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_DEBUG_ACCESS_MEMORY_PARAMS._fields_ = [
    ('sessionIndex', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('baseAddress', ctypes.c_uint64),
    ('sizeInBytes', ctypes.c_uint64),
    ('accessType', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('buffer', ctypes.c_uint64),
    ('isBitmaskSet', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 7),
    ('bitmask', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_3', ctypes.c_ubyte * 4),
]

UVM_DEBUG_ACCESS_MEMORY_PARAMS = struct_c__SA_UVM_DEBUG_ACCESS_MEMORY_PARAMS
class struct_c__SA_UVM_REGISTER_GPU_PARAMS(Structure):
    pass

struct_c__SA_UVM_REGISTER_GPU_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_REGISTER_GPU_PARAMS._fields_ = [
    ('gpu_uuid', struct_nv_uuid),
    ('numaEnabled', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('numaNodeId', ctypes.c_int32),
    ('rmCtrlFd', ctypes.c_int32),
    ('hClient', ctypes.c_uint32),
    ('hSmcPartRef', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_REGISTER_GPU_PARAMS = struct_c__SA_UVM_REGISTER_GPU_PARAMS
class struct_c__SA_UVM_UNREGISTER_GPU_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNREGISTER_GPU_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNREGISTER_GPU_PARAMS._fields_ = [
    ('gpu_uuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_UNREGISTER_GPU_PARAMS = struct_c__SA_UVM_UNREGISTER_GPU_PARAMS
class struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_PARAMS(Structure):
    pass

struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_PARAMS._fields_ = [
    ('pageableMemAccess', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('rmStatus', ctypes.c_uint32),
]

UVM_PAGEABLE_MEM_ACCESS_PARAMS = struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_PARAMS
class struct_c__SA_UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS(Structure):
    pass

struct_c__SA_UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS._fields_ = [
    ('rangeGroupIds', ctypes.c_uint64 * 32),
    ('numGroupIds', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS = struct_c__SA_UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS
class struct_c__SA_UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS(Structure):
    pass

struct_c__SA_UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS._fields_ = [
    ('rangeGroupIds', ctypes.c_uint64 * 32),
    ('numGroupIds', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS = struct_c__SA_UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS
class struct_c__SA_UVM_SET_PREFERRED_LOCATION_PARAMS(Structure):
    pass

struct_c__SA_UVM_SET_PREFERRED_LOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_SET_PREFERRED_LOCATION_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('preferredLocation', struct_nv_uuid),
    ('preferredCpuNumaNode', ctypes.c_int32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_SET_PREFERRED_LOCATION_PARAMS = struct_c__SA_UVM_SET_PREFERRED_LOCATION_PARAMS
class struct_c__SA_UVM_UNSET_PREFERRED_LOCATION_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNSET_PREFERRED_LOCATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNSET_PREFERRED_LOCATION_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_UNSET_PREFERRED_LOCATION_PARAMS = struct_c__SA_UVM_UNSET_PREFERRED_LOCATION_PARAMS
class struct_c__SA_UVM_ENABLE_READ_DUPLICATION_PARAMS(Structure):
    pass

struct_c__SA_UVM_ENABLE_READ_DUPLICATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ENABLE_READ_DUPLICATION_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_ENABLE_READ_DUPLICATION_PARAMS = struct_c__SA_UVM_ENABLE_READ_DUPLICATION_PARAMS
class struct_c__SA_UVM_DISABLE_READ_DUPLICATION_PARAMS(Structure):
    pass

struct_c__SA_UVM_DISABLE_READ_DUPLICATION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_DISABLE_READ_DUPLICATION_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_DISABLE_READ_DUPLICATION_PARAMS = struct_c__SA_UVM_DISABLE_READ_DUPLICATION_PARAMS
class struct_c__SA_UVM_SET_ACCESSED_BY_PARAMS(Structure):
    pass

struct_c__SA_UVM_SET_ACCESSED_BY_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_SET_ACCESSED_BY_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('accessedByUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_SET_ACCESSED_BY_PARAMS = struct_c__SA_UVM_SET_ACCESSED_BY_PARAMS
class struct_c__SA_UVM_UNSET_ACCESSED_BY_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNSET_ACCESSED_BY_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNSET_ACCESSED_BY_PARAMS._fields_ = [
    ('requestedBase', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('accessedByUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_UNSET_ACCESSED_BY_PARAMS = struct_c__SA_UVM_UNSET_ACCESSED_BY_PARAMS
class struct_c__SA_UVM_MIGRATE_PARAMS(Structure):
    pass

struct_c__SA_UVM_MIGRATE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MIGRATE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('destinationUuid', struct_nv_uuid),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('semaphoreAddress', ctypes.c_uint64),
    ('semaphorePayload', ctypes.c_uint32),
    ('cpuNumaNode', ctypes.c_int32),
    ('userSpaceStart', ctypes.c_uint64),
    ('userSpaceLength', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

UVM_MIGRATE_PARAMS = struct_c__SA_UVM_MIGRATE_PARAMS
class struct_c__SA_UVM_MIGRATE_RANGE_GROUP_PARAMS(Structure):
    pass

struct_c__SA_UVM_MIGRATE_RANGE_GROUP_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MIGRATE_RANGE_GROUP_PARAMS._fields_ = [
    ('rangeGroupId', ctypes.c_uint64),
    ('destinationUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_MIGRATE_RANGE_GROUP_PARAMS = struct_c__SA_UVM_MIGRATE_RANGE_GROUP_PARAMS
class struct_c__SA_UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS(Structure):
    pass

struct_c__SA_UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS._fields_ = [
    ('gpu_uuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS = struct_c__SA_UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS
class struct_c__SA_UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS(Structure):
    pass

struct_c__SA_UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS._fields_ = [
    ('gpu_uuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
]

UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS = struct_c__SA_UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS
class struct_c__SA_UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS._fields_ = [
    ('queueBuffer', ctypes.c_uint64),
    ('queueBufferSize', ctypes.c_uint64),
    ('controlBuffer', ctypes.c_uint64),
    ('processor', struct_nv_uuid),
    ('allProcessors', ctypes.c_uint32),
    ('uvmFd', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
    ('requestedVersion', ctypes.c_uint32),
    ('grantedVersion', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS = struct_c__SA_UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS
class struct_c__SA_UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS._fields_ = [
    ('notificationThreshold', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS = struct_c__SA_UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS
class struct_c__SA_UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS._fields_ = [
    ('eventTypeFlags', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS = struct_c__SA_UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS
class struct_c__SA_UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS._fields_ = [
    ('eventTypeFlags', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS = struct_c__SA_UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS
class struct_c__SA_UVM_TOOLS_ENABLE_COUNTERS_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_ENABLE_COUNTERS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_ENABLE_COUNTERS_PARAMS._fields_ = [
    ('counterTypeFlags', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_ENABLE_COUNTERS_PARAMS = struct_c__SA_UVM_TOOLS_ENABLE_COUNTERS_PARAMS
class struct_c__SA_UVM_TOOLS_DISABLE_COUNTERS_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_DISABLE_COUNTERS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_DISABLE_COUNTERS_PARAMS._fields_ = [
    ('counterTypeFlags', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_DISABLE_COUNTERS_PARAMS = struct_c__SA_UVM_TOOLS_DISABLE_COUNTERS_PARAMS
class struct_c__SA_UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS._fields_ = [
    ('buffer', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('targetVa', ctypes.c_uint64),
    ('bytesRead', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS = struct_c__SA_UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS
class struct_c__SA_UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS._fields_ = [
    ('buffer', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('targetVa', ctypes.c_uint64),
    ('bytesWritten', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS = struct_c__SA_UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS
class struct_c__SA_UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS._fields_ = [
    ('tablePtr', ctypes.c_uint64),
    ('count', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
    ('version', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS = struct_c__SA_UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS
class struct_c__SA_UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS(Structure):
    pass

struct_c__SA_UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS = struct_c__SA_UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS
class struct_c__SA_UVM_UNMAP_EXTERNAL_PARAMS(Structure):
    pass

struct_c__SA_UVM_UNMAP_EXTERNAL_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_UNMAP_EXTERNAL_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_UNMAP_EXTERNAL_PARAMS = struct_c__SA_UVM_UNMAP_EXTERNAL_PARAMS
class struct_c__SA_UVM_TOOLS_FLUSH_EVENTS_PARAMS(Structure):
    pass

struct_c__SA_UVM_TOOLS_FLUSH_EVENTS_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_TOOLS_FLUSH_EVENTS_PARAMS._fields_ = [
    ('rmStatus', ctypes.c_uint32),
]

UVM_TOOLS_FLUSH_EVENTS_PARAMS = struct_c__SA_UVM_TOOLS_FLUSH_EVENTS_PARAMS
class struct_c__SA_UVM_ALLOC_SEMAPHORE_POOL_PARAMS(Structure):
    pass

struct_c__SA_UVM_ALLOC_SEMAPHORE_POOL_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_ALLOC_SEMAPHORE_POOL_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('perGpuAttributes', struct_c__SA_UvmGpuMappingAttributes * 256),
    ('gpuAttributesCount', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_ALLOC_SEMAPHORE_POOL_PARAMS = struct_c__SA_UVM_ALLOC_SEMAPHORE_POOL_PARAMS
class struct_c__SA_UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS(Structure):
    pass

struct_c__SA_UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS._fields_ = [
    ('rmStatus', ctypes.c_uint32),
]

UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS = struct_c__SA_UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS
class struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS(Structure):
    pass

struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS._fields_ = [
    ('gpu_uuid', struct_nv_uuid),
    ('pageableMemAccess', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('rmStatus', ctypes.c_uint32),
]

UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS = struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS
class struct_c__SA_UVM_POPULATE_PAGEABLE_PARAMS(Structure):
    pass

struct_c__SA_UVM_POPULATE_PAGEABLE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_POPULATE_PAGEABLE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_POPULATE_PAGEABLE_PARAMS = struct_c__SA_UVM_POPULATE_PAGEABLE_PARAMS
class struct_c__SA_UVM_VALIDATE_VA_RANGE_PARAMS(Structure):
    pass

struct_c__SA_UVM_VALIDATE_VA_RANGE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_VALIDATE_VA_RANGE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_VALIDATE_VA_RANGE_PARAMS = struct_c__SA_UVM_VALIDATE_VA_RANGE_PARAMS
class struct_c__SA_UVM_CREATE_EXTERNAL_RANGE_PARAMS(Structure):
    pass

struct_c__SA_UVM_CREATE_EXTERNAL_RANGE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_CREATE_EXTERNAL_RANGE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_CREATE_EXTERNAL_RANGE_PARAMS = struct_c__SA_UVM_CREATE_EXTERNAL_RANGE_PARAMS
class struct_c__SA_UVM_MAP_EXTERNAL_SPARSE_PARAMS(Structure):
    pass

struct_c__SA_UVM_MAP_EXTERNAL_SPARSE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MAP_EXTERNAL_SPARSE_PARAMS._fields_ = [
    ('base', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('gpuUuid', struct_nv_uuid),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_MAP_EXTERNAL_SPARSE_PARAMS = struct_c__SA_UVM_MAP_EXTERNAL_SPARSE_PARAMS
class struct_c__SA_UVM_MM_INITIALIZE_PARAMS(Structure):
    pass

struct_c__SA_UVM_MM_INITIALIZE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_MM_INITIALIZE_PARAMS._fields_ = [
    ('uvmFd', ctypes.c_int32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_MM_INITIALIZE_PARAMS = struct_c__SA_UVM_MM_INITIALIZE_PARAMS
class struct_c__SA_UVM_IS_8_SUPPORTED_PARAMS(Structure):
    pass

struct_c__SA_UVM_IS_8_SUPPORTED_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_IS_8_SUPPORTED_PARAMS._fields_ = [
    ('is8Supported', ctypes.c_uint32),
    ('rmStatus', ctypes.c_uint32),
]

UVM_IS_8_SUPPORTED_PARAMS = struct_c__SA_UVM_IS_8_SUPPORTED_PARAMS
_UVM_LINUX_IOCTL_H = True # macro
UVM_INITIALIZE = 0x30000001 # macro
UVM_DEINITIALIZE = 0x30000002 # macro
class struct_c__SA_UVM_INITIALIZE_PARAMS(Structure):
    pass

struct_c__SA_UVM_INITIALIZE_PARAMS._pack_ = 1 # source:False
struct_c__SA_UVM_INITIALIZE_PARAMS._fields_ = [
    ('flags', ctypes.c_uint64),
    ('rmStatus', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

UVM_INITIALIZE_PARAMS = struct_c__SA_UVM_INITIALIZE_PARAMS
__all__ = \
    ['UVM_ADD_SESSION', 'UVM_ADD_SESSION_PARAMS',
    'UVM_ALLOC_SEMAPHORE_POOL', 'UVM_ALLOC_SEMAPHORE_POOL_PARAMS',
    'UVM_ALLOW_MIGRATION_RANGE_GROUPS',
    'UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS',
    'UVM_CLEAN_UP_ZOMBIE_RESOURCES',
    'UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS',
    'UVM_COUNTERS_OFFSET_BASE', 'UVM_CREATE_EVENT_QUEUE',
    'UVM_CREATE_EVENT_QUEUE_PARAMS', 'UVM_CREATE_EXTERNAL_RANGE',
    'UVM_CREATE_EXTERNAL_RANGE_PARAMS', 'UVM_CREATE_RANGE_GROUP',
    'UVM_CREATE_RANGE_GROUP_PARAMS', 'UVM_DEBUG_ACCESS_MEMORY',
    'UVM_DEBUG_ACCESS_MEMORY_PARAMS', 'UVM_DEINITIALIZE',
    'UVM_DESTROY_RANGE_GROUP', 'UVM_DESTROY_RANGE_GROUP_PARAMS',
    'UVM_DISABLE_PEER_ACCESS', 'UVM_DISABLE_PEER_ACCESS_PARAMS',
    'UVM_DISABLE_READ_DUPLICATION',
    'UVM_DISABLE_READ_DUPLICATION_PARAMS',
    'UVM_DISABLE_SYSTEM_WIDE_ATOMICS',
    'UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS', 'UVM_ENABLE_COUNTERS',
    'UVM_ENABLE_COUNTERS_PARAMS', 'UVM_ENABLE_PEER_ACCESS',
    'UVM_ENABLE_PEER_ACCESS_PARAMS', 'UVM_ENABLE_READ_DUPLICATION',
    'UVM_ENABLE_READ_DUPLICATION_PARAMS',
    'UVM_ENABLE_SYSTEM_WIDE_ATOMICS',
    'UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS', 'UVM_EVENTS_OFFSET_BASE',
    'UVM_EVENT_CTRL', 'UVM_EVENT_CTRL_PARAMS', 'UVM_FREE',
    'UVM_FREE_PARAMS', 'UVM_GET_GPU_UUID_TABLE',
    'UVM_GET_GPU_UUID_TABLE_PARAMS', 'UVM_INITIALIZE',
    'UVM_INITIALIZE_PARAMS', 'UVM_IS_8_SUPPORTED',
    'UVM_IS_8_SUPPORTED_PARAMS', 'UVM_MAP_COUNTER',
    'UVM_MAP_COUNTER_PARAMS', 'UVM_MAP_DYNAMIC_PARALLELISM_REGION',
    'UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS',
    'UVM_MAP_EVENT_QUEUE', 'UVM_MAP_EVENT_QUEUE_PARAMS',
    'UVM_MAP_EXTERNAL_ALLOCATION',
    'UVM_MAP_EXTERNAL_ALLOCATION_PARAMS', 'UVM_MAP_EXTERNAL_SPARSE',
    'UVM_MAP_EXTERNAL_SPARSE_PARAMS',
    'UVM_MAX_COUNTERS_PER_IOCTL_CALL',
    'UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL',
    'UVM_MAX_STREAMS_PER_IOCTL_CALL', 'UVM_MEM_MAP',
    'UVM_MEM_MAP_PARAMS', 'UVM_MIGRATE', 'UVM_MIGRATE_FLAGS_ALL',
    'UVM_MIGRATE_FLAGS_TEST_ALL', 'UVM_MIGRATE_FLAG_ASYNC',
    'UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE',
    'UVM_MIGRATE_FLAG_SKIP_CPU_MAP', 'UVM_MIGRATE_PARAMS',
    'UVM_MIGRATE_RANGE_GROUP', 'UVM_MIGRATE_RANGE_GROUP_PARAMS',
    'UVM_MM_INITIALIZE', 'UVM_MM_INITIALIZE_PARAMS',
    'UVM_PAGEABLE_MEM_ACCESS', 'UVM_PAGEABLE_MEM_ACCESS_ON_GPU',
    'UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS',
    'UVM_PAGEABLE_MEM_ACCESS_PARAMS', 'UVM_POPULATE_PAGEABLE',
    'UVM_POPULATE_PAGEABLE_FLAGS_ALL',
    'UVM_POPULATE_PAGEABLE_FLAGS_TEST_ALL',
    'UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED',
    'UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK',
    'UVM_POPULATE_PAGEABLE_PARAMS',
    'UVM_PREVENT_MIGRATION_RANGE_GROUPS',
    'UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS', 'UVM_REGION_COMMIT',
    'UVM_REGION_COMMIT_PARAMS', 'UVM_REGION_DECOMMIT',
    'UVM_REGION_DECOMMIT_PARAMS', 'UVM_REGION_SET_STREAM',
    'UVM_REGION_SET_STREAM_PARAMS', 'UVM_REGISTER_CHANNEL',
    'UVM_REGISTER_CHANNEL_PARAMS', 'UVM_REGISTER_GPU',
    'UVM_REGISTER_GPU_PARAMS', 'UVM_REGISTER_GPU_VASPACE',
    'UVM_REGISTER_GPU_VASPACE_PARAMS', 'UVM_REGISTER_MPS_CLIENT',
    'UVM_REGISTER_MPS_CLIENT_PARAMS', 'UVM_REGISTER_MPS_SERVER',
    'UVM_REGISTER_MPS_SERVER_PARAMS', 'UVM_RELEASE_VA',
    'UVM_RELEASE_VA_PARAMS', 'UVM_REMOVE_EVENT_QUEUE',
    'UVM_REMOVE_EVENT_QUEUE_PARAMS', 'UVM_REMOVE_SESSION',
    'UVM_REMOVE_SESSION_PARAMS', 'UVM_RESERVE_VA',
    'UVM_RESERVE_VA_PARAMS', 'UVM_RUN_TEST', 'UVM_RUN_TEST_PARAMS',
    'UVM_SET_ACCESSED_BY', 'UVM_SET_ACCESSED_BY_PARAMS',
    'UVM_SET_PREFERRED_LOCATION', 'UVM_SET_PREFERRED_LOCATION_PARAMS',
    'UVM_SET_RANGE_GROUP', 'UVM_SET_RANGE_GROUP_PARAMS',
    'UVM_SET_STREAM_RUNNING', 'UVM_SET_STREAM_RUNNING_PARAMS',
    'UVM_SET_STREAM_STOPPED', 'UVM_SET_STREAM_STOPPED_PARAMS',
    'UVM_TOOLS_DISABLE_COUNTERS', 'UVM_TOOLS_DISABLE_COUNTERS_PARAMS',
    'UVM_TOOLS_ENABLE_COUNTERS', 'UVM_TOOLS_ENABLE_COUNTERS_PARAMS',
    'UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS',
    'UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS',
    'UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS',
    'UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS',
    'UVM_TOOLS_FLUSH_EVENTS', 'UVM_TOOLS_FLUSH_EVENTS_PARAMS',
    'UVM_TOOLS_GET_PROCESSOR_UUID_TABLE',
    'UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS',
    'UVM_TOOLS_INIT_EVENT_TRACKER',
    'UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS',
    'UVM_TOOLS_READ_PROCESS_MEMORY',
    'UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS',
    'UVM_TOOLS_SET_NOTIFICATION_THRESHOLD',
    'UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS',
    'UVM_TOOLS_WRITE_PROCESS_MEMORY',
    'UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS', 'UVM_UNMAP_EXTERNAL',
    'UVM_UNMAP_EXTERNAL_PARAMS', 'UVM_UNREGISTER_CHANNEL',
    'UVM_UNREGISTER_CHANNEL_PARAMS', 'UVM_UNREGISTER_GPU',
    'UVM_UNREGISTER_GPU_PARAMS', 'UVM_UNREGISTER_GPU_VASPACE',
    'UVM_UNREGISTER_GPU_VASPACE_PARAMS', 'UVM_UNSET_ACCESSED_BY',
    'UVM_UNSET_ACCESSED_BY_PARAMS', 'UVM_UNSET_PREFERRED_LOCATION',
    'UVM_UNSET_PREFERRED_LOCATION_PARAMS', 'UVM_VALIDATE_VA_RANGE',
    'UVM_VALIDATE_VA_RANGE_PARAMS', '_UVM_IOCTL_H',
    '_UVM_LINUX_IOCTL_H', 'struct_c__SA_UVM_ADD_SESSION_PARAMS',
    'struct_c__SA_UVM_ALLOC_SEMAPHORE_POOL_PARAMS',
    'struct_c__SA_UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS',
    'struct_c__SA_UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS',
    'struct_c__SA_UVM_CREATE_EVENT_QUEUE_PARAMS',
    'struct_c__SA_UVM_CREATE_EXTERNAL_RANGE_PARAMS',
    'struct_c__SA_UVM_CREATE_RANGE_GROUP_PARAMS',
    'struct_c__SA_UVM_DEBUG_ACCESS_MEMORY_PARAMS',
    'struct_c__SA_UVM_DESTROY_RANGE_GROUP_PARAMS',
    'struct_c__SA_UVM_DISABLE_PEER_ACCESS_PARAMS',
    'struct_c__SA_UVM_DISABLE_READ_DUPLICATION_PARAMS',
    'struct_c__SA_UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS',
    'struct_c__SA_UVM_ENABLE_COUNTERS_PARAMS',
    'struct_c__SA_UVM_ENABLE_PEER_ACCESS_PARAMS',
    'struct_c__SA_UVM_ENABLE_READ_DUPLICATION_PARAMS',
    'struct_c__SA_UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS',
    'struct_c__SA_UVM_EVENT_CTRL_PARAMS',
    'struct_c__SA_UVM_FREE_PARAMS',
    'struct_c__SA_UVM_GET_GPU_UUID_TABLE_PARAMS',
    'struct_c__SA_UVM_INITIALIZE_PARAMS',
    'struct_c__SA_UVM_IS_8_SUPPORTED_PARAMS',
    'struct_c__SA_UVM_MAP_COUNTER_PARAMS',
    'struct_c__SA_UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS',
    'struct_c__SA_UVM_MAP_EVENT_QUEUE_PARAMS',
    'struct_c__SA_UVM_MAP_EXTERNAL_ALLOCATION_PARAMS',
    'struct_c__SA_UVM_MAP_EXTERNAL_SPARSE_PARAMS',
    'struct_c__SA_UVM_MEM_MAP_PARAMS',
    'struct_c__SA_UVM_MIGRATE_PARAMS',
    'struct_c__SA_UVM_MIGRATE_RANGE_GROUP_PARAMS',
    'struct_c__SA_UVM_MM_INITIALIZE_PARAMS',
    'struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS',
    'struct_c__SA_UVM_PAGEABLE_MEM_ACCESS_PARAMS',
    'struct_c__SA_UVM_POPULATE_PAGEABLE_PARAMS',
    'struct_c__SA_UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS',
    'struct_c__SA_UVM_REGION_COMMIT_PARAMS',
    'struct_c__SA_UVM_REGION_DECOMMIT_PARAMS',
    'struct_c__SA_UVM_REGION_SET_STREAM_PARAMS',
    'struct_c__SA_UVM_REGISTER_CHANNEL_PARAMS',
    'struct_c__SA_UVM_REGISTER_GPU_PARAMS',
    'struct_c__SA_UVM_REGISTER_GPU_VASPACE_PARAMS',
    'struct_c__SA_UVM_REGISTER_MPS_CLIENT_PARAMS',
    'struct_c__SA_UVM_REGISTER_MPS_SERVER_PARAMS',
    'struct_c__SA_UVM_RELEASE_VA_PARAMS',
    'struct_c__SA_UVM_REMOVE_EVENT_QUEUE_PARAMS',
    'struct_c__SA_UVM_REMOVE_SESSION_PARAMS',
    'struct_c__SA_UVM_RESERVE_VA_PARAMS',
    'struct_c__SA_UVM_RUN_TEST_PARAMS',
    'struct_c__SA_UVM_RUN_TEST_PARAMS_multiGpu',
    'struct_c__SA_UVM_SET_ACCESSED_BY_PARAMS',
    'struct_c__SA_UVM_SET_PREFERRED_LOCATION_PARAMS',
    'struct_c__SA_UVM_SET_RANGE_GROUP_PARAMS',
    'struct_c__SA_UVM_SET_STREAM_RUNNING_PARAMS',
    'struct_c__SA_UVM_SET_STREAM_STOPPED_PARAMS',
    'struct_c__SA_UVM_TOOLS_DISABLE_COUNTERS_PARAMS',
    'struct_c__SA_UVM_TOOLS_ENABLE_COUNTERS_PARAMS',
    'struct_c__SA_UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS',
    'struct_c__SA_UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS',
    'struct_c__SA_UVM_TOOLS_FLUSH_EVENTS_PARAMS',
    'struct_c__SA_UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS',
    'struct_c__SA_UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS',
    'struct_c__SA_UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS',
    'struct_c__SA_UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS',
    'struct_c__SA_UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS',
    'struct_c__SA_UVM_UNMAP_EXTERNAL_PARAMS',
    'struct_c__SA_UVM_UNREGISTER_CHANNEL_PARAMS',
    'struct_c__SA_UVM_UNREGISTER_GPU_PARAMS',
    'struct_c__SA_UVM_UNREGISTER_GPU_VASPACE_PARAMS',
    'struct_c__SA_UVM_UNSET_ACCESSED_BY_PARAMS',
    'struct_c__SA_UVM_UNSET_PREFERRED_LOCATION_PARAMS',
    'struct_c__SA_UVM_VALIDATE_VA_RANGE_PARAMS',
    'struct_c__SA_UvmCounterConfig',
    'struct_c__SA_UvmGpuMappingAttributes', 'struct_nv_uuid']
