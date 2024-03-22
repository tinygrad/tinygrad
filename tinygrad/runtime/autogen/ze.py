# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/usr/include/']
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



_libraries = {}
_libraries['libze_loader.so'] = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libze_loader.so')


ze_bool_t = ctypes.c_ubyte
class struct__ze_driver_handle_t(Structure):
    pass

ze_driver_handle_t = ctypes.POINTER(struct__ze_driver_handle_t)
class struct__ze_device_handle_t(Structure):
    pass

ze_device_handle_t = ctypes.POINTER(struct__ze_device_handle_t)
class struct__ze_context_handle_t(Structure):
    pass

ze_context_handle_t = ctypes.POINTER(struct__ze_context_handle_t)
class struct__ze_command_queue_handle_t(Structure):
    pass

ze_command_queue_handle_t = ctypes.POINTER(struct__ze_command_queue_handle_t)
class struct__ze_command_list_handle_t(Structure):
    pass

ze_command_list_handle_t = ctypes.POINTER(struct__ze_command_list_handle_t)
class struct__ze_fence_handle_t(Structure):
    pass

ze_fence_handle_t = ctypes.POINTER(struct__ze_fence_handle_t)
class struct__ze_event_pool_handle_t(Structure):
    pass

ze_event_pool_handle_t = ctypes.POINTER(struct__ze_event_pool_handle_t)
class struct__ze_event_handle_t(Structure):
    pass

ze_event_handle_t = ctypes.POINTER(struct__ze_event_handle_t)
class struct__ze_image_handle_t(Structure):
    pass

ze_image_handle_t = ctypes.POINTER(struct__ze_image_handle_t)
class struct__ze_module_handle_t(Structure):
    pass

ze_module_handle_t = ctypes.POINTER(struct__ze_module_handle_t)
class struct__ze_module_build_log_handle_t(Structure):
    pass

ze_module_build_log_handle_t = ctypes.POINTER(struct__ze_module_build_log_handle_t)
class struct__ze_kernel_handle_t(Structure):
    pass

ze_kernel_handle_t = ctypes.POINTER(struct__ze_kernel_handle_t)
class struct__ze_sampler_handle_t(Structure):
    pass

ze_sampler_handle_t = ctypes.POINTER(struct__ze_sampler_handle_t)
class struct__ze_physical_mem_handle_t(Structure):
    pass

ze_physical_mem_handle_t = ctypes.POINTER(struct__ze_physical_mem_handle_t)
class struct__ze_fabric_vertex_handle_t(Structure):
    pass

ze_fabric_vertex_handle_t = ctypes.POINTER(struct__ze_fabric_vertex_handle_t)
class struct__ze_fabric_edge_handle_t(Structure):
    pass

ze_fabric_edge_handle_t = ctypes.POINTER(struct__ze_fabric_edge_handle_t)
class struct__ze_ipc_mem_handle_t(Structure):
    pass

struct__ze_ipc_mem_handle_t._pack_ = 1 # source:False
struct__ze_ipc_mem_handle_t._fields_ = [
    ('data', ctypes.c_char * 64),
]

ze_ipc_mem_handle_t = struct__ze_ipc_mem_handle_t
class struct__ze_ipc_event_pool_handle_t(Structure):
    pass

struct__ze_ipc_event_pool_handle_t._pack_ = 1 # source:False
struct__ze_ipc_event_pool_handle_t._fields_ = [
    ('data', ctypes.c_char * 64),
]

ze_ipc_event_pool_handle_t = struct__ze_ipc_event_pool_handle_t

# values for enumeration '_ze_result_t'
_ze_result_t__enumvalues = {
    0: 'ZE_RESULT_SUCCESS',
    1: 'ZE_RESULT_NOT_READY',
    1879048193: 'ZE_RESULT_ERROR_DEVICE_LOST',
    1879048194: 'ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY',
    1879048195: 'ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY',
    1879048196: 'ZE_RESULT_ERROR_MODULE_BUILD_FAILURE',
    1879048197: 'ZE_RESULT_ERROR_MODULE_LINK_FAILURE',
    1879048198: 'ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET',
    1879048199: 'ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE',
    2146435073: 'ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX',
    2146435074: 'ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE',
    2146435075: 'ZE_RESULT_EXP_ERROR_REMOTE_DEVICE',
    2146435076: 'ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE',
    2146435077: 'ZE_RESULT_EXP_RTAS_BUILD_RETRY',
    2146435078: 'ZE_RESULT_EXP_RTAS_BUILD_DEFERRED',
    1879113728: 'ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS',
    1879113729: 'ZE_RESULT_ERROR_NOT_AVAILABLE',
    1879179264: 'ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE',
    1879179265: 'ZE_RESULT_WARNING_DROPPED_DATA',
    2013265921: 'ZE_RESULT_ERROR_UNINITIALIZED',
    2013265922: 'ZE_RESULT_ERROR_UNSUPPORTED_VERSION',
    2013265923: 'ZE_RESULT_ERROR_UNSUPPORTED_FEATURE',
    2013265924: 'ZE_RESULT_ERROR_INVALID_ARGUMENT',
    2013265925: 'ZE_RESULT_ERROR_INVALID_NULL_HANDLE',
    2013265926: 'ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE',
    2013265927: 'ZE_RESULT_ERROR_INVALID_NULL_POINTER',
    2013265928: 'ZE_RESULT_ERROR_INVALID_SIZE',
    2013265929: 'ZE_RESULT_ERROR_UNSUPPORTED_SIZE',
    2013265930: 'ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT',
    2013265931: 'ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT',
    2013265932: 'ZE_RESULT_ERROR_INVALID_ENUMERATION',
    2013265933: 'ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION',
    2013265934: 'ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT',
    2013265935: 'ZE_RESULT_ERROR_INVALID_NATIVE_BINARY',
    2013265936: 'ZE_RESULT_ERROR_INVALID_GLOBAL_NAME',
    2013265937: 'ZE_RESULT_ERROR_INVALID_KERNEL_NAME',
    2013265938: 'ZE_RESULT_ERROR_INVALID_FUNCTION_NAME',
    2013265939: 'ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION',
    2013265940: 'ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION',
    2013265941: 'ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX',
    2013265942: 'ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE',
    2013265943: 'ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE',
    2013265944: 'ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED',
    2013265945: 'ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE',
    2013265946: 'ZE_RESULT_ERROR_OVERLAPPING_REGIONS',
    2013265947: 'ZE_RESULT_WARNING_ACTION_REQUIRED',
    2147483646: 'ZE_RESULT_ERROR_UNKNOWN',
    2147483647: 'ZE_RESULT_FORCE_UINT32',
}
ZE_RESULT_SUCCESS = 0
ZE_RESULT_NOT_READY = 1
ZE_RESULT_ERROR_DEVICE_LOST = 1879048193
ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 1879048194
ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 1879048195
ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 1879048196
ZE_RESULT_ERROR_MODULE_LINK_FAILURE = 1879048197
ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 1879048198
ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 1879048199
ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX = 2146435073
ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE = 2146435074
ZE_RESULT_EXP_ERROR_REMOTE_DEVICE = 2146435075
ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE = 2146435076
ZE_RESULT_EXP_RTAS_BUILD_RETRY = 2146435077
ZE_RESULT_EXP_RTAS_BUILD_DEFERRED = 2146435078
ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 1879113728
ZE_RESULT_ERROR_NOT_AVAILABLE = 1879113729
ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 1879179264
ZE_RESULT_WARNING_DROPPED_DATA = 1879179265
ZE_RESULT_ERROR_UNINITIALIZED = 2013265921
ZE_RESULT_ERROR_UNSUPPORTED_VERSION = 2013265922
ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 2013265923
ZE_RESULT_ERROR_INVALID_ARGUMENT = 2013265924
ZE_RESULT_ERROR_INVALID_NULL_HANDLE = 2013265925
ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 2013265926
ZE_RESULT_ERROR_INVALID_NULL_POINTER = 2013265927
ZE_RESULT_ERROR_INVALID_SIZE = 2013265928
ZE_RESULT_ERROR_UNSUPPORTED_SIZE = 2013265929
ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 2013265930
ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 2013265931
ZE_RESULT_ERROR_INVALID_ENUMERATION = 2013265932
ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 2013265933
ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 2013265934
ZE_RESULT_ERROR_INVALID_NATIVE_BINARY = 2013265935
ZE_RESULT_ERROR_INVALID_GLOBAL_NAME = 2013265936
ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 2013265937
ZE_RESULT_ERROR_INVALID_FUNCTION_NAME = 2013265938
ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 2013265939
ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 2013265940
ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 2013265941
ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 2013265942
ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 2013265943
ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 2013265944
ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 2013265945
ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 2013265946
ZE_RESULT_WARNING_ACTION_REQUIRED = 2013265947
ZE_RESULT_ERROR_UNKNOWN = 2147483646
ZE_RESULT_FORCE_UINT32 = 2147483647
_ze_result_t = ctypes.c_uint32 # enum
ze_result_t = _ze_result_t
ze_result_t__enumvalues = _ze_result_t__enumvalues

# values for enumeration '_ze_structure_type_t'
_ze_structure_type_t__enumvalues = {
    1: 'ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES',
    2: 'ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES',
    3: 'ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES',
    4: 'ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES',
    5: 'ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES',
    6: 'ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES',
    7: 'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES',
    8: 'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES',
    9: 'ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES',
    10: 'ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES',
    11: 'ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES',
    12: 'ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES',
    13: 'ZE_STRUCTURE_TYPE_CONTEXT_DESC',
    14: 'ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC',
    15: 'ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC',
    16: 'ZE_STRUCTURE_TYPE_EVENT_POOL_DESC',
    17: 'ZE_STRUCTURE_TYPE_EVENT_DESC',
    18: 'ZE_STRUCTURE_TYPE_FENCE_DESC',
    19: 'ZE_STRUCTURE_TYPE_IMAGE_DESC',
    20: 'ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES',
    21: 'ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC',
    22: 'ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC',
    23: 'ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES',
    24: 'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC',
    25: 'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD',
    26: 'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD',
    27: 'ZE_STRUCTURE_TYPE_MODULE_DESC',
    28: 'ZE_STRUCTURE_TYPE_MODULE_PROPERTIES',
    29: 'ZE_STRUCTURE_TYPE_KERNEL_DESC',
    30: 'ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES',
    31: 'ZE_STRUCTURE_TYPE_SAMPLER_DESC',
    32: 'ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC',
    33: 'ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES',
    34: 'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32',
    35: 'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32',
    65537: 'ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES',
    65538: 'ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC',
    65539: 'ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES',
    65540: 'ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC',
    65541: 'ZE_STRUCTURE_TYPE_EU_COUNT_EXT',
    65542: 'ZE_STRUCTURE_TYPE_SRGB_EXT_DESC',
    65543: 'ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC',
    65544: 'ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES',
    65545: 'ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES',
    65546: 'ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC',
    65547: 'ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC',
    65548: 'ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES',
    65549: 'ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES',
    65550: 'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES',
    65551: 'ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT',
    65552: 'ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC',
    65553: 'ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES',
    65554: 'ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES',
    65555: 'ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES',
    131073: 'ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC',
    131074: 'ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC',
    131075: 'ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES',
    131076: 'ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC',
    131077: 'ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC',
    131078: 'ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2',
    131079: 'ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES',
    131080: 'ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC',
    131081: 'ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES',
    131082: 'ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES',
    131083: 'ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES',
    131084: 'ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES',
    131085: 'ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES',
    131086: 'ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC',
    131087: 'ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC',
    131088: 'ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES',
    131089: 'ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES',
    131090: 'ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES',
    131091: 'ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS',
    2147483647: 'ZE_STRUCTURE_TYPE_FORCE_UINT32',
}
ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 1
ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES = 2
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 3
ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 4
ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 5
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 6
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 7
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES = 8
ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES = 9
ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES = 10
ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 11
ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 12
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 13
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 14
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 15
ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 16
ZE_STRUCTURE_TYPE_EVENT_DESC = 17
ZE_STRUCTURE_TYPE_FENCE_DESC = 18
ZE_STRUCTURE_TYPE_IMAGE_DESC = 19
ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES = 20
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 21
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 22
ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 23
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 24
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 25
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 26
ZE_STRUCTURE_TYPE_MODULE_DESC = 27
ZE_STRUCTURE_TYPE_MODULE_PROPERTIES = 28
ZE_STRUCTURE_TYPE_KERNEL_DESC = 29
ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES = 30
ZE_STRUCTURE_TYPE_SAMPLER_DESC = 31
ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 32
ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 33
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32 = 34
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32 = 35
ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES = 65537
ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC = 65538
ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES = 65539
ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC = 65540
ZE_STRUCTURE_TYPE_EU_COUNT_EXT = 65541
ZE_STRUCTURE_TYPE_SRGB_EXT_DESC = 65542
ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC = 65543
ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 65544
ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES = 65545
ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC = 65546
ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC = 65547
ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES = 65548
ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES = 65549
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES = 65550
ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT = 65551
ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC = 65552
ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES = 65553
ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES = 65554
ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES = 65555
ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 131073
ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC = 131074
ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES = 131075
ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC = 131076
ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC = 131077
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 = 131078
ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES = 131079
ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC = 131080
ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES = 131081
ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES = 131082
ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES = 131083
ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES = 131084
ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES = 131085
ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC = 131086
ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC = 131087
ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES = 131088
ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES = 131089
ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES = 131090
ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS = 131091
ZE_STRUCTURE_TYPE_FORCE_UINT32 = 2147483647
_ze_structure_type_t = ctypes.c_uint32 # enum
ze_structure_type_t = _ze_structure_type_t
ze_structure_type_t__enumvalues = _ze_structure_type_t__enumvalues
ze_external_memory_type_flags_t = ctypes.c_uint32

# values for enumeration '_ze_external_memory_type_flag_t'
_ze_external_memory_type_flag_t__enumvalues = {
    1: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD',
    2: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF',
    4: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32',
    8: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT',
    16: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE',
    32: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT',
    64: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP',
    128: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE',
    2147483647: 'ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32',
}
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = 1
ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 2
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 = 4
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT = 8
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE = 16
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT = 32
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP = 64
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE = 128
ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32 = 2147483647
_ze_external_memory_type_flag_t = ctypes.c_uint32 # enum
ze_external_memory_type_flag_t = _ze_external_memory_type_flag_t
ze_external_memory_type_flag_t__enumvalues = _ze_external_memory_type_flag_t__enumvalues

# values for enumeration '_ze_bandwidth_unit_t'
_ze_bandwidth_unit_t__enumvalues = {
    0: 'ZE_BANDWIDTH_UNIT_UNKNOWN',
    1: 'ZE_BANDWIDTH_UNIT_BYTES_PER_NANOSEC',
    2: 'ZE_BANDWIDTH_UNIT_BYTES_PER_CLOCK',
    2147483647: 'ZE_BANDWIDTH_UNIT_FORCE_UINT32',
}
ZE_BANDWIDTH_UNIT_UNKNOWN = 0
ZE_BANDWIDTH_UNIT_BYTES_PER_NANOSEC = 1
ZE_BANDWIDTH_UNIT_BYTES_PER_CLOCK = 2
ZE_BANDWIDTH_UNIT_FORCE_UINT32 = 2147483647
_ze_bandwidth_unit_t = ctypes.c_uint32 # enum
ze_bandwidth_unit_t = _ze_bandwidth_unit_t
ze_bandwidth_unit_t__enumvalues = _ze_bandwidth_unit_t__enumvalues

# values for enumeration '_ze_latency_unit_t'
_ze_latency_unit_t__enumvalues = {
    0: 'ZE_LATENCY_UNIT_UNKNOWN',
    1: 'ZE_LATENCY_UNIT_NANOSEC',
    2: 'ZE_LATENCY_UNIT_CLOCK',
    3: 'ZE_LATENCY_UNIT_HOP',
    2147483647: 'ZE_LATENCY_UNIT_FORCE_UINT32',
}
ZE_LATENCY_UNIT_UNKNOWN = 0
ZE_LATENCY_UNIT_NANOSEC = 1
ZE_LATENCY_UNIT_CLOCK = 2
ZE_LATENCY_UNIT_HOP = 3
ZE_LATENCY_UNIT_FORCE_UINT32 = 2147483647
_ze_latency_unit_t = ctypes.c_uint32 # enum
ze_latency_unit_t = _ze_latency_unit_t
ze_latency_unit_t__enumvalues = _ze_latency_unit_t__enumvalues
class struct__ze_uuid_t(Structure):
    pass

struct__ze_uuid_t._pack_ = 1 # source:False
struct__ze_uuid_t._fields_ = [
    ('id', ctypes.c_ubyte * 16),
]

ze_uuid_t = struct__ze_uuid_t
class struct__ze_base_cb_params_t(Structure):
    pass

struct__ze_base_cb_params_t._pack_ = 1 # source:False
struct__ze_base_cb_params_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
]

ze_base_cb_params_t = struct__ze_base_cb_params_t
class struct__ze_base_properties_t(Structure):
    pass

struct__ze_base_properties_t._pack_ = 1 # source:False
struct__ze_base_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
]

ze_base_properties_t = struct__ze_base_properties_t
class struct__ze_base_desc_t(Structure):
    pass

struct__ze_base_desc_t._pack_ = 1 # source:False
struct__ze_base_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
]

ze_base_desc_t = struct__ze_base_desc_t
class struct__ze_driver_uuid_t(Structure):
    pass

struct__ze_driver_uuid_t._pack_ = 1 # source:False
struct__ze_driver_uuid_t._fields_ = [
    ('id', ctypes.c_ubyte * 16),
]

ze_driver_uuid_t = struct__ze_driver_uuid_t
class struct__ze_driver_properties_t(Structure):
    pass

struct__ze_driver_properties_t._pack_ = 1 # source:False
struct__ze_driver_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('uuid', ze_driver_uuid_t),
    ('driverVersion', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_driver_properties_t = struct__ze_driver_properties_t
class struct__ze_driver_ipc_properties_t(Structure):
    pass

struct__ze_driver_ipc_properties_t._pack_ = 1 # source:False
struct__ze_driver_ipc_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_driver_ipc_properties_t = struct__ze_driver_ipc_properties_t
class struct__ze_driver_extension_properties_t(Structure):
    pass

struct__ze_driver_extension_properties_t._pack_ = 1 # source:False
struct__ze_driver_extension_properties_t._fields_ = [
    ('name', ctypes.c_char * 256),
    ('version', ctypes.c_uint32),
]

ze_driver_extension_properties_t = struct__ze_driver_extension_properties_t
class struct__ze_device_uuid_t(Structure):
    pass

struct__ze_device_uuid_t._pack_ = 1 # source:False
struct__ze_device_uuid_t._fields_ = [
    ('id', ctypes.c_ubyte * 16),
]

ze_device_uuid_t = struct__ze_device_uuid_t
class struct__ze_device_properties_t(Structure):
    pass


# values for enumeration '_ze_device_type_t'
_ze_device_type_t__enumvalues = {
    1: 'ZE_DEVICE_TYPE_GPU',
    2: 'ZE_DEVICE_TYPE_CPU',
    3: 'ZE_DEVICE_TYPE_FPGA',
    4: 'ZE_DEVICE_TYPE_MCA',
    5: 'ZE_DEVICE_TYPE_VPU',
    2147483647: 'ZE_DEVICE_TYPE_FORCE_UINT32',
}
ZE_DEVICE_TYPE_GPU = 1
ZE_DEVICE_TYPE_CPU = 2
ZE_DEVICE_TYPE_FPGA = 3
ZE_DEVICE_TYPE_MCA = 4
ZE_DEVICE_TYPE_VPU = 5
ZE_DEVICE_TYPE_FORCE_UINT32 = 2147483647
_ze_device_type_t = ctypes.c_uint32 # enum
ze_device_type_t = _ze_device_type_t
ze_device_type_t__enumvalues = _ze_device_type_t__enumvalues
struct__ze_device_properties_t._pack_ = 1 # source:False
struct__ze_device_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('type', ze_device_type_t),
    ('vendorId', ctypes.c_uint32),
    ('deviceId', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('subdeviceId', ctypes.c_uint32),
    ('coreClockRate', ctypes.c_uint32),
    ('maxMemAllocSize', ctypes.c_uint64),
    ('maxHardwareContexts', ctypes.c_uint32),
    ('maxCommandQueuePriority', ctypes.c_uint32),
    ('numThreadsPerEU', ctypes.c_uint32),
    ('physicalEUSimdWidth', ctypes.c_uint32),
    ('numEUsPerSubslice', ctypes.c_uint32),
    ('numSubslicesPerSlice', ctypes.c_uint32),
    ('numSlices', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('timerResolution', ctypes.c_uint64),
    ('timestampValidBits', ctypes.c_uint32),
    ('kernelTimestampValidBits', ctypes.c_uint32),
    ('uuid', ze_device_uuid_t),
    ('name', ctypes.c_char * 256),
]

ze_device_properties_t = struct__ze_device_properties_t
class struct__ze_device_thread_t(Structure):
    pass

struct__ze_device_thread_t._pack_ = 1 # source:False
struct__ze_device_thread_t._fields_ = [
    ('slice', ctypes.c_uint32),
    ('subslice', ctypes.c_uint32),
    ('eu', ctypes.c_uint32),
    ('thread', ctypes.c_uint32),
]

ze_device_thread_t = struct__ze_device_thread_t
class struct__ze_device_compute_properties_t(Structure):
    pass

struct__ze_device_compute_properties_t._pack_ = 1 # source:False
struct__ze_device_compute_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('maxTotalGroupSize', ctypes.c_uint32),
    ('maxGroupSizeX', ctypes.c_uint32),
    ('maxGroupSizeY', ctypes.c_uint32),
    ('maxGroupSizeZ', ctypes.c_uint32),
    ('maxGroupCountX', ctypes.c_uint32),
    ('maxGroupCountY', ctypes.c_uint32),
    ('maxGroupCountZ', ctypes.c_uint32),
    ('maxSharedLocalMemory', ctypes.c_uint32),
    ('numSubGroupSizes', ctypes.c_uint32),
    ('subGroupSizes', ctypes.c_uint32 * 8),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_compute_properties_t = struct__ze_device_compute_properties_t
class struct__ze_native_kernel_uuid_t(Structure):
    pass

struct__ze_native_kernel_uuid_t._pack_ = 1 # source:False
struct__ze_native_kernel_uuid_t._fields_ = [
    ('id', ctypes.c_ubyte * 16),
]

ze_native_kernel_uuid_t = struct__ze_native_kernel_uuid_t
class struct__ze_device_module_properties_t(Structure):
    pass

struct__ze_device_module_properties_t._pack_ = 1 # source:False
struct__ze_device_module_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('spirvVersionSupported', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('fp16flags', ctypes.c_uint32),
    ('fp32flags', ctypes.c_uint32),
    ('fp64flags', ctypes.c_uint32),
    ('maxArgumentsSize', ctypes.c_uint32),
    ('printfBufferSize', ctypes.c_uint32),
    ('nativeKernelSupported', ze_native_kernel_uuid_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_module_properties_t = struct__ze_device_module_properties_t
class struct__ze_command_queue_group_properties_t(Structure):
    pass

struct__ze_command_queue_group_properties_t._pack_ = 1 # source:False
struct__ze_command_queue_group_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('maxMemoryFillPatternSize', ctypes.c_uint64),
    ('numQueues', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

ze_command_queue_group_properties_t = struct__ze_command_queue_group_properties_t
class struct__ze_device_memory_properties_t(Structure):
    pass

struct__ze_device_memory_properties_t._pack_ = 1 # source:False
struct__ze_device_memory_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('maxClockRate', ctypes.c_uint32),
    ('maxBusWidth', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('totalSize', ctypes.c_uint64),
    ('name', ctypes.c_char * 256),
]

ze_device_memory_properties_t = struct__ze_device_memory_properties_t
class struct__ze_device_memory_access_properties_t(Structure):
    pass

struct__ze_device_memory_access_properties_t._pack_ = 1 # source:False
struct__ze_device_memory_access_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('hostAllocCapabilities', ctypes.c_uint32),
    ('deviceAllocCapabilities', ctypes.c_uint32),
    ('sharedSingleDeviceAllocCapabilities', ctypes.c_uint32),
    ('sharedCrossDeviceAllocCapabilities', ctypes.c_uint32),
    ('sharedSystemAllocCapabilities', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_memory_access_properties_t = struct__ze_device_memory_access_properties_t
class struct__ze_device_cache_properties_t(Structure):
    pass

struct__ze_device_cache_properties_t._pack_ = 1 # source:False
struct__ze_device_cache_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('cacheSize', ctypes.c_uint64),
]

ze_device_cache_properties_t = struct__ze_device_cache_properties_t
class struct__ze_device_image_properties_t(Structure):
    pass

struct__ze_device_image_properties_t._pack_ = 1 # source:False
struct__ze_device_image_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('maxImageDims1D', ctypes.c_uint32),
    ('maxImageDims2D', ctypes.c_uint32),
    ('maxImageDims3D', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('maxImageBufferSize', ctypes.c_uint64),
    ('maxImageArraySlices', ctypes.c_uint32),
    ('maxSamplers', ctypes.c_uint32),
    ('maxReadImageArgs', ctypes.c_uint32),
    ('maxWriteImageArgs', ctypes.c_uint32),
]

ze_device_image_properties_t = struct__ze_device_image_properties_t
class struct__ze_device_external_memory_properties_t(Structure):
    pass

struct__ze_device_external_memory_properties_t._pack_ = 1 # source:False
struct__ze_device_external_memory_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('memoryAllocationImportTypes', ctypes.c_uint32),
    ('memoryAllocationExportTypes', ctypes.c_uint32),
    ('imageImportTypes', ctypes.c_uint32),
    ('imageExportTypes', ctypes.c_uint32),
]

ze_device_external_memory_properties_t = struct__ze_device_external_memory_properties_t
class struct__ze_device_p2p_properties_t(Structure):
    pass

struct__ze_device_p2p_properties_t._pack_ = 1 # source:False
struct__ze_device_p2p_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_p2p_properties_t = struct__ze_device_p2p_properties_t
class struct__ze_context_desc_t(Structure):
    pass

struct__ze_context_desc_t._pack_ = 1 # source:False
struct__ze_context_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_context_desc_t = struct__ze_context_desc_t
class struct__ze_command_queue_desc_t(Structure):
    pass


# values for enumeration '_ze_command_queue_mode_t'
_ze_command_queue_mode_t__enumvalues = {
    0: 'ZE_COMMAND_QUEUE_MODE_DEFAULT',
    1: 'ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS',
    2: 'ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS',
    2147483647: 'ZE_COMMAND_QUEUE_MODE_FORCE_UINT32',
}
ZE_COMMAND_QUEUE_MODE_DEFAULT = 0
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2
ZE_COMMAND_QUEUE_MODE_FORCE_UINT32 = 2147483647
_ze_command_queue_mode_t = ctypes.c_uint32 # enum
ze_command_queue_mode_t = _ze_command_queue_mode_t
ze_command_queue_mode_t__enumvalues = _ze_command_queue_mode_t__enumvalues

# values for enumeration '_ze_command_queue_priority_t'
_ze_command_queue_priority_t__enumvalues = {
    0: 'ZE_COMMAND_QUEUE_PRIORITY_NORMAL',
    1: 'ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW',
    2: 'ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH',
    2147483647: 'ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32',
}
ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1
ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2
ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32 = 2147483647
_ze_command_queue_priority_t = ctypes.c_uint32 # enum
ze_command_queue_priority_t = _ze_command_queue_priority_t
ze_command_queue_priority_t__enumvalues = _ze_command_queue_priority_t__enumvalues
struct__ze_command_queue_desc_t._pack_ = 1 # source:False
struct__ze_command_queue_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('ordinal', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('mode', ze_command_queue_mode_t),
    ('priority', ze_command_queue_priority_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_command_queue_desc_t = struct__ze_command_queue_desc_t
class struct__ze_command_list_desc_t(Structure):
    pass

struct__ze_command_list_desc_t._pack_ = 1 # source:False
struct__ze_command_list_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('commandQueueGroupOrdinal', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

ze_command_list_desc_t = struct__ze_command_list_desc_t
class struct__ze_copy_region_t(Structure):
    pass

struct__ze_copy_region_t._pack_ = 1 # source:False
struct__ze_copy_region_t._fields_ = [
    ('originX', ctypes.c_uint32),
    ('originY', ctypes.c_uint32),
    ('originZ', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
]

ze_copy_region_t = struct__ze_copy_region_t
class struct__ze_image_region_t(Structure):
    pass

struct__ze_image_region_t._pack_ = 1 # source:False
struct__ze_image_region_t._fields_ = [
    ('originX', ctypes.c_uint32),
    ('originY', ctypes.c_uint32),
    ('originZ', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
]

ze_image_region_t = struct__ze_image_region_t
class struct__ze_event_pool_desc_t(Structure):
    pass

struct__ze_event_pool_desc_t._pack_ = 1 # source:False
struct__ze_event_pool_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

ze_event_pool_desc_t = struct__ze_event_pool_desc_t
class struct__ze_event_desc_t(Structure):
    pass

struct__ze_event_desc_t._pack_ = 1 # source:False
struct__ze_event_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('index', ctypes.c_uint32),
    ('signal', ctypes.c_uint32),
    ('wait', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_event_desc_t = struct__ze_event_desc_t
class struct__ze_kernel_timestamp_data_t(Structure):
    pass

struct__ze_kernel_timestamp_data_t._pack_ = 1 # source:False
struct__ze_kernel_timestamp_data_t._fields_ = [
    ('kernelStart', ctypes.c_uint64),
    ('kernelEnd', ctypes.c_uint64),
]

ze_kernel_timestamp_data_t = struct__ze_kernel_timestamp_data_t
class struct__ze_kernel_timestamp_result_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('global', ze_kernel_timestamp_data_t),
    ('context', ze_kernel_timestamp_data_t),
     ]

ze_kernel_timestamp_result_t = struct__ze_kernel_timestamp_result_t
class struct__ze_fence_desc_t(Structure):
    pass

struct__ze_fence_desc_t._pack_ = 1 # source:False
struct__ze_fence_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_fence_desc_t = struct__ze_fence_desc_t
class struct__ze_image_format_t(Structure):
    pass


# values for enumeration '_ze_image_format_layout_t'
_ze_image_format_layout_t__enumvalues = {
    0: 'ZE_IMAGE_FORMAT_LAYOUT_8',
    1: 'ZE_IMAGE_FORMAT_LAYOUT_16',
    2: 'ZE_IMAGE_FORMAT_LAYOUT_32',
    3: 'ZE_IMAGE_FORMAT_LAYOUT_8_8',
    4: 'ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8',
    5: 'ZE_IMAGE_FORMAT_LAYOUT_16_16',
    6: 'ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16',
    7: 'ZE_IMAGE_FORMAT_LAYOUT_32_32',
    8: 'ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32',
    9: 'ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2',
    10: 'ZE_IMAGE_FORMAT_LAYOUT_11_11_10',
    11: 'ZE_IMAGE_FORMAT_LAYOUT_5_6_5',
    12: 'ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1',
    13: 'ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4',
    14: 'ZE_IMAGE_FORMAT_LAYOUT_Y8',
    15: 'ZE_IMAGE_FORMAT_LAYOUT_NV12',
    16: 'ZE_IMAGE_FORMAT_LAYOUT_YUYV',
    17: 'ZE_IMAGE_FORMAT_LAYOUT_VYUY',
    18: 'ZE_IMAGE_FORMAT_LAYOUT_YVYU',
    19: 'ZE_IMAGE_FORMAT_LAYOUT_UYVY',
    20: 'ZE_IMAGE_FORMAT_LAYOUT_AYUV',
    21: 'ZE_IMAGE_FORMAT_LAYOUT_P010',
    22: 'ZE_IMAGE_FORMAT_LAYOUT_Y410',
    23: 'ZE_IMAGE_FORMAT_LAYOUT_P012',
    24: 'ZE_IMAGE_FORMAT_LAYOUT_Y16',
    25: 'ZE_IMAGE_FORMAT_LAYOUT_P016',
    26: 'ZE_IMAGE_FORMAT_LAYOUT_Y216',
    27: 'ZE_IMAGE_FORMAT_LAYOUT_P216',
    28: 'ZE_IMAGE_FORMAT_LAYOUT_P8',
    29: 'ZE_IMAGE_FORMAT_LAYOUT_YUY2',
    30: 'ZE_IMAGE_FORMAT_LAYOUT_A8P8',
    31: 'ZE_IMAGE_FORMAT_LAYOUT_IA44',
    32: 'ZE_IMAGE_FORMAT_LAYOUT_AI44',
    33: 'ZE_IMAGE_FORMAT_LAYOUT_Y416',
    34: 'ZE_IMAGE_FORMAT_LAYOUT_Y210',
    35: 'ZE_IMAGE_FORMAT_LAYOUT_I420',
    36: 'ZE_IMAGE_FORMAT_LAYOUT_YV12',
    37: 'ZE_IMAGE_FORMAT_LAYOUT_400P',
    38: 'ZE_IMAGE_FORMAT_LAYOUT_422H',
    39: 'ZE_IMAGE_FORMAT_LAYOUT_422V',
    40: 'ZE_IMAGE_FORMAT_LAYOUT_444P',
    41: 'ZE_IMAGE_FORMAT_LAYOUT_RGBP',
    42: 'ZE_IMAGE_FORMAT_LAYOUT_BRGP',
    2147483647: 'ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32',
}
ZE_IMAGE_FORMAT_LAYOUT_8 = 0
ZE_IMAGE_FORMAT_LAYOUT_16 = 1
ZE_IMAGE_FORMAT_LAYOUT_32 = 2
ZE_IMAGE_FORMAT_LAYOUT_8_8 = 3
ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 = 4
ZE_IMAGE_FORMAT_LAYOUT_16_16 = 5
ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 = 6
ZE_IMAGE_FORMAT_LAYOUT_32_32 = 7
ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 = 8
ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2 = 9
ZE_IMAGE_FORMAT_LAYOUT_11_11_10 = 10
ZE_IMAGE_FORMAT_LAYOUT_5_6_5 = 11
ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1 = 12
ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4 = 13
ZE_IMAGE_FORMAT_LAYOUT_Y8 = 14
ZE_IMAGE_FORMAT_LAYOUT_NV12 = 15
ZE_IMAGE_FORMAT_LAYOUT_YUYV = 16
ZE_IMAGE_FORMAT_LAYOUT_VYUY = 17
ZE_IMAGE_FORMAT_LAYOUT_YVYU = 18
ZE_IMAGE_FORMAT_LAYOUT_UYVY = 19
ZE_IMAGE_FORMAT_LAYOUT_AYUV = 20
ZE_IMAGE_FORMAT_LAYOUT_P010 = 21
ZE_IMAGE_FORMAT_LAYOUT_Y410 = 22
ZE_IMAGE_FORMAT_LAYOUT_P012 = 23
ZE_IMAGE_FORMAT_LAYOUT_Y16 = 24
ZE_IMAGE_FORMAT_LAYOUT_P016 = 25
ZE_IMAGE_FORMAT_LAYOUT_Y216 = 26
ZE_IMAGE_FORMAT_LAYOUT_P216 = 27
ZE_IMAGE_FORMAT_LAYOUT_P8 = 28
ZE_IMAGE_FORMAT_LAYOUT_YUY2 = 29
ZE_IMAGE_FORMAT_LAYOUT_A8P8 = 30
ZE_IMAGE_FORMAT_LAYOUT_IA44 = 31
ZE_IMAGE_FORMAT_LAYOUT_AI44 = 32
ZE_IMAGE_FORMAT_LAYOUT_Y416 = 33
ZE_IMAGE_FORMAT_LAYOUT_Y210 = 34
ZE_IMAGE_FORMAT_LAYOUT_I420 = 35
ZE_IMAGE_FORMAT_LAYOUT_YV12 = 36
ZE_IMAGE_FORMAT_LAYOUT_400P = 37
ZE_IMAGE_FORMAT_LAYOUT_422H = 38
ZE_IMAGE_FORMAT_LAYOUT_422V = 39
ZE_IMAGE_FORMAT_LAYOUT_444P = 40
ZE_IMAGE_FORMAT_LAYOUT_RGBP = 41
ZE_IMAGE_FORMAT_LAYOUT_BRGP = 42
ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32 = 2147483647
_ze_image_format_layout_t = ctypes.c_uint32 # enum
ze_image_format_layout_t = _ze_image_format_layout_t
ze_image_format_layout_t__enumvalues = _ze_image_format_layout_t__enumvalues

# values for enumeration '_ze_image_format_type_t'
_ze_image_format_type_t__enumvalues = {
    0: 'ZE_IMAGE_FORMAT_TYPE_UINT',
    1: 'ZE_IMAGE_FORMAT_TYPE_SINT',
    2: 'ZE_IMAGE_FORMAT_TYPE_UNORM',
    3: 'ZE_IMAGE_FORMAT_TYPE_SNORM',
    4: 'ZE_IMAGE_FORMAT_TYPE_FLOAT',
    2147483647: 'ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32',
}
ZE_IMAGE_FORMAT_TYPE_UINT = 0
ZE_IMAGE_FORMAT_TYPE_SINT = 1
ZE_IMAGE_FORMAT_TYPE_UNORM = 2
ZE_IMAGE_FORMAT_TYPE_SNORM = 3
ZE_IMAGE_FORMAT_TYPE_FLOAT = 4
ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32 = 2147483647
_ze_image_format_type_t = ctypes.c_uint32 # enum
ze_image_format_type_t = _ze_image_format_type_t
ze_image_format_type_t__enumvalues = _ze_image_format_type_t__enumvalues

# values for enumeration '_ze_image_format_swizzle_t'
_ze_image_format_swizzle_t__enumvalues = {
    0: 'ZE_IMAGE_FORMAT_SWIZZLE_R',
    1: 'ZE_IMAGE_FORMAT_SWIZZLE_G',
    2: 'ZE_IMAGE_FORMAT_SWIZZLE_B',
    3: 'ZE_IMAGE_FORMAT_SWIZZLE_A',
    4: 'ZE_IMAGE_FORMAT_SWIZZLE_0',
    5: 'ZE_IMAGE_FORMAT_SWIZZLE_1',
    6: 'ZE_IMAGE_FORMAT_SWIZZLE_X',
    2147483647: 'ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32',
}
ZE_IMAGE_FORMAT_SWIZZLE_R = 0
ZE_IMAGE_FORMAT_SWIZZLE_G = 1
ZE_IMAGE_FORMAT_SWIZZLE_B = 2
ZE_IMAGE_FORMAT_SWIZZLE_A = 3
ZE_IMAGE_FORMAT_SWIZZLE_0 = 4
ZE_IMAGE_FORMAT_SWIZZLE_1 = 5
ZE_IMAGE_FORMAT_SWIZZLE_X = 6
ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32 = 2147483647
_ze_image_format_swizzle_t = ctypes.c_uint32 # enum
ze_image_format_swizzle_t = _ze_image_format_swizzle_t
ze_image_format_swizzle_t__enumvalues = _ze_image_format_swizzle_t__enumvalues
struct__ze_image_format_t._pack_ = 1 # source:False
struct__ze_image_format_t._fields_ = [
    ('layout', ze_image_format_layout_t),
    ('type', ze_image_format_type_t),
    ('x', ze_image_format_swizzle_t),
    ('y', ze_image_format_swizzle_t),
    ('z', ze_image_format_swizzle_t),
    ('w', ze_image_format_swizzle_t),
]

ze_image_format_t = struct__ze_image_format_t
class struct__ze_image_desc_t(Structure):
    pass


# values for enumeration '_ze_image_type_t'
_ze_image_type_t__enumvalues = {
    0: 'ZE_IMAGE_TYPE_1D',
    1: 'ZE_IMAGE_TYPE_1DARRAY',
    2: 'ZE_IMAGE_TYPE_2D',
    3: 'ZE_IMAGE_TYPE_2DARRAY',
    4: 'ZE_IMAGE_TYPE_3D',
    5: 'ZE_IMAGE_TYPE_BUFFER',
    2147483647: 'ZE_IMAGE_TYPE_FORCE_UINT32',
}
ZE_IMAGE_TYPE_1D = 0
ZE_IMAGE_TYPE_1DARRAY = 1
ZE_IMAGE_TYPE_2D = 2
ZE_IMAGE_TYPE_2DARRAY = 3
ZE_IMAGE_TYPE_3D = 4
ZE_IMAGE_TYPE_BUFFER = 5
ZE_IMAGE_TYPE_FORCE_UINT32 = 2147483647
_ze_image_type_t = ctypes.c_uint32 # enum
ze_image_type_t = _ze_image_type_t
ze_image_type_t__enumvalues = _ze_image_type_t__enumvalues
struct__ze_image_desc_t._pack_ = 1 # source:False
struct__ze_image_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('type', ze_image_type_t),
    ('format', ze_image_format_t),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
    ('arraylevels', ctypes.c_uint32),
    ('miplevels', ctypes.c_uint32),
]

ze_image_desc_t = struct__ze_image_desc_t
class struct__ze_image_properties_t(Structure):
    pass

struct__ze_image_properties_t._pack_ = 1 # source:False
struct__ze_image_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('samplerFilterFlags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_image_properties_t = struct__ze_image_properties_t
class struct__ze_device_mem_alloc_desc_t(Structure):
    pass

struct__ze_device_mem_alloc_desc_t._pack_ = 1 # source:False
struct__ze_device_mem_alloc_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('ordinal', ctypes.c_uint32),
]

ze_device_mem_alloc_desc_t = struct__ze_device_mem_alloc_desc_t
class struct__ze_host_mem_alloc_desc_t(Structure):
    pass

struct__ze_host_mem_alloc_desc_t._pack_ = 1 # source:False
struct__ze_host_mem_alloc_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_host_mem_alloc_desc_t = struct__ze_host_mem_alloc_desc_t
class struct__ze_memory_allocation_properties_t(Structure):
    pass


# values for enumeration '_ze_memory_type_t'
_ze_memory_type_t__enumvalues = {
    0: 'ZE_MEMORY_TYPE_UNKNOWN',
    1: 'ZE_MEMORY_TYPE_HOST',
    2: 'ZE_MEMORY_TYPE_DEVICE',
    3: 'ZE_MEMORY_TYPE_SHARED',
    2147483647: 'ZE_MEMORY_TYPE_FORCE_UINT32',
}
ZE_MEMORY_TYPE_UNKNOWN = 0
ZE_MEMORY_TYPE_HOST = 1
ZE_MEMORY_TYPE_DEVICE = 2
ZE_MEMORY_TYPE_SHARED = 3
ZE_MEMORY_TYPE_FORCE_UINT32 = 2147483647
_ze_memory_type_t = ctypes.c_uint32 # enum
ze_memory_type_t = _ze_memory_type_t
ze_memory_type_t__enumvalues = _ze_memory_type_t__enumvalues
struct__ze_memory_allocation_properties_t._pack_ = 1 # source:False
struct__ze_memory_allocation_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('type', ze_memory_type_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('id', ctypes.c_uint64),
    ('pageSize', ctypes.c_uint64),
]

ze_memory_allocation_properties_t = struct__ze_memory_allocation_properties_t
class struct__ze_external_memory_export_desc_t(Structure):
    pass

struct__ze_external_memory_export_desc_t._pack_ = 1 # source:False
struct__ze_external_memory_export_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_external_memory_export_desc_t = struct__ze_external_memory_export_desc_t
class struct__ze_external_memory_import_fd_t(Structure):
    pass

struct__ze_external_memory_import_fd_t._pack_ = 1 # source:False
struct__ze_external_memory_import_fd_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('fd', ctypes.c_int32),
]

ze_external_memory_import_fd_t = struct__ze_external_memory_import_fd_t
class struct__ze_external_memory_export_fd_t(Structure):
    pass

struct__ze_external_memory_export_fd_t._pack_ = 1 # source:False
struct__ze_external_memory_export_fd_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('fd', ctypes.c_int32),
]

ze_external_memory_export_fd_t = struct__ze_external_memory_export_fd_t
class struct__ze_external_memory_import_win32_handle_t(Structure):
    pass

struct__ze_external_memory_import_win32_handle_t._pack_ = 1 # source:False
struct__ze_external_memory_import_win32_handle_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

ze_external_memory_import_win32_handle_t = struct__ze_external_memory_import_win32_handle_t
class struct__ze_external_memory_export_win32_handle_t(Structure):
    pass

struct__ze_external_memory_export_win32_handle_t._pack_ = 1 # source:False
struct__ze_external_memory_export_win32_handle_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('handle', ctypes.POINTER(None)),
]

ze_external_memory_export_win32_handle_t = struct__ze_external_memory_export_win32_handle_t
class struct__ze_module_constants_t(Structure):
    pass

struct__ze_module_constants_t._pack_ = 1 # source:False
struct__ze_module_constants_t._fields_ = [
    ('numConstants', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pConstantIds', ctypes.POINTER(ctypes.c_uint32)),
    ('pConstantValues', ctypes.POINTER(ctypes.POINTER(None))),
]

ze_module_constants_t = struct__ze_module_constants_t
class struct__ze_module_desc_t(Structure):
    pass


# values for enumeration '_ze_module_format_t'
_ze_module_format_t__enumvalues = {
    0: 'ZE_MODULE_FORMAT_IL_SPIRV',
    1: 'ZE_MODULE_FORMAT_NATIVE',
    2147483647: 'ZE_MODULE_FORMAT_FORCE_UINT32',
}
ZE_MODULE_FORMAT_IL_SPIRV = 0
ZE_MODULE_FORMAT_NATIVE = 1
ZE_MODULE_FORMAT_FORCE_UINT32 = 2147483647
_ze_module_format_t = ctypes.c_uint32 # enum
ze_module_format_t = _ze_module_format_t
ze_module_format_t__enumvalues = _ze_module_format_t__enumvalues
struct__ze_module_desc_t._pack_ = 1 # source:False
struct__ze_module_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('format', ze_module_format_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('inputSize', ctypes.c_uint64),
    ('pInputModule', ctypes.POINTER(ctypes.c_ubyte)),
    ('pBuildFlags', ctypes.POINTER(ctypes.c_char)),
    ('pConstants', ctypes.POINTER(struct__ze_module_constants_t)),
]

ze_module_desc_t = struct__ze_module_desc_t
class struct__ze_module_properties_t(Structure):
    pass

struct__ze_module_properties_t._pack_ = 1 # source:False
struct__ze_module_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_module_properties_t = struct__ze_module_properties_t
class struct__ze_kernel_desc_t(Structure):
    pass

struct__ze_kernel_desc_t._pack_ = 1 # source:False
struct__ze_kernel_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pKernelName', ctypes.POINTER(ctypes.c_char)),
]

ze_kernel_desc_t = struct__ze_kernel_desc_t
class struct__ze_kernel_uuid_t(Structure):
    pass

struct__ze_kernel_uuid_t._pack_ = 1 # source:False
struct__ze_kernel_uuid_t._fields_ = [
    ('kid', ctypes.c_ubyte * 16),
    ('mid', ctypes.c_ubyte * 16),
]

ze_kernel_uuid_t = struct__ze_kernel_uuid_t
class struct__ze_kernel_properties_t(Structure):
    pass

struct__ze_kernel_properties_t._pack_ = 1 # source:False
struct__ze_kernel_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('numKernelArgs', ctypes.c_uint32),
    ('requiredGroupSizeX', ctypes.c_uint32),
    ('requiredGroupSizeY', ctypes.c_uint32),
    ('requiredGroupSizeZ', ctypes.c_uint32),
    ('requiredNumSubGroups', ctypes.c_uint32),
    ('requiredSubgroupSize', ctypes.c_uint32),
    ('maxSubgroupSize', ctypes.c_uint32),
    ('maxNumSubgroups', ctypes.c_uint32),
    ('localMemSize', ctypes.c_uint32),
    ('privateMemSize', ctypes.c_uint32),
    ('spillMemSize', ctypes.c_uint32),
    ('uuid', ze_kernel_uuid_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_kernel_properties_t = struct__ze_kernel_properties_t
class struct__ze_kernel_preferred_group_size_properties_t(Structure):
    pass

struct__ze_kernel_preferred_group_size_properties_t._pack_ = 1 # source:False
struct__ze_kernel_preferred_group_size_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('preferredMultiple', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_kernel_preferred_group_size_properties_t = struct__ze_kernel_preferred_group_size_properties_t
class struct__ze_group_count_t(Structure):
    pass

struct__ze_group_count_t._pack_ = 1 # source:False
struct__ze_group_count_t._fields_ = [
    ('groupCountX', ctypes.c_uint32),
    ('groupCountY', ctypes.c_uint32),
    ('groupCountZ', ctypes.c_uint32),
]

ze_group_count_t = struct__ze_group_count_t
class struct__ze_module_program_exp_desc_t(Structure):
    pass

struct__ze_module_program_exp_desc_t._pack_ = 1 # source:False
struct__ze_module_program_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('count', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('inputSizes', ctypes.POINTER(ctypes.c_uint64)),
    ('pInputModules', ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))),
    ('pBuildFlags', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ('pConstants', ctypes.POINTER(ctypes.POINTER(struct__ze_module_constants_t))),
]

ze_module_program_exp_desc_t = struct__ze_module_program_exp_desc_t
class struct__ze_device_raytracing_ext_properties_t(Structure):
    pass

struct__ze_device_raytracing_ext_properties_t._pack_ = 1 # source:False
struct__ze_device_raytracing_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('maxBVHLevels', ctypes.c_uint32),
]

ze_device_raytracing_ext_properties_t = struct__ze_device_raytracing_ext_properties_t
class struct__ze_raytracing_mem_alloc_ext_desc_t(Structure):
    pass

struct__ze_raytracing_mem_alloc_ext_desc_t._pack_ = 1 # source:False
struct__ze_raytracing_mem_alloc_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_raytracing_mem_alloc_ext_desc_t = struct__ze_raytracing_mem_alloc_ext_desc_t
class struct__ze_sampler_desc_t(Structure):
    pass


# values for enumeration '_ze_sampler_address_mode_t'
_ze_sampler_address_mode_t__enumvalues = {
    0: 'ZE_SAMPLER_ADDRESS_MODE_NONE',
    1: 'ZE_SAMPLER_ADDRESS_MODE_REPEAT',
    2: 'ZE_SAMPLER_ADDRESS_MODE_CLAMP',
    3: 'ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER',
    4: 'ZE_SAMPLER_ADDRESS_MODE_MIRROR',
    2147483647: 'ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32',
}
ZE_SAMPLER_ADDRESS_MODE_NONE = 0
ZE_SAMPLER_ADDRESS_MODE_REPEAT = 1
ZE_SAMPLER_ADDRESS_MODE_CLAMP = 2
ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3
ZE_SAMPLER_ADDRESS_MODE_MIRROR = 4
ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32 = 2147483647
_ze_sampler_address_mode_t = ctypes.c_uint32 # enum
ze_sampler_address_mode_t = _ze_sampler_address_mode_t
ze_sampler_address_mode_t__enumvalues = _ze_sampler_address_mode_t__enumvalues

# values for enumeration '_ze_sampler_filter_mode_t'
_ze_sampler_filter_mode_t__enumvalues = {
    0: 'ZE_SAMPLER_FILTER_MODE_NEAREST',
    1: 'ZE_SAMPLER_FILTER_MODE_LINEAR',
    2147483647: 'ZE_SAMPLER_FILTER_MODE_FORCE_UINT32',
}
ZE_SAMPLER_FILTER_MODE_NEAREST = 0
ZE_SAMPLER_FILTER_MODE_LINEAR = 1
ZE_SAMPLER_FILTER_MODE_FORCE_UINT32 = 2147483647
_ze_sampler_filter_mode_t = ctypes.c_uint32 # enum
ze_sampler_filter_mode_t = _ze_sampler_filter_mode_t
ze_sampler_filter_mode_t__enumvalues = _ze_sampler_filter_mode_t__enumvalues
struct__ze_sampler_desc_t._pack_ = 1 # source:False
struct__ze_sampler_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('addressMode', ze_sampler_address_mode_t),
    ('filterMode', ze_sampler_filter_mode_t),
    ('isNormalized', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

ze_sampler_desc_t = struct__ze_sampler_desc_t
class struct__ze_physical_mem_desc_t(Structure):
    pass

struct__ze_physical_mem_desc_t._pack_ = 1 # source:False
struct__ze_physical_mem_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('size', ctypes.c_uint64),
]

ze_physical_mem_desc_t = struct__ze_physical_mem_desc_t
class struct__ze_float_atomic_ext_properties_t(Structure):
    pass

struct__ze_float_atomic_ext_properties_t._pack_ = 1 # source:False
struct__ze_float_atomic_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('fp16Flags', ctypes.c_uint32),
    ('fp32Flags', ctypes.c_uint32),
    ('fp64Flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_float_atomic_ext_properties_t = struct__ze_float_atomic_ext_properties_t
class struct__ze_relaxed_allocation_limits_exp_desc_t(Structure):
    pass

struct__ze_relaxed_allocation_limits_exp_desc_t._pack_ = 1 # source:False
struct__ze_relaxed_allocation_limits_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_relaxed_allocation_limits_exp_desc_t = struct__ze_relaxed_allocation_limits_exp_desc_t
class struct__ze_cache_reservation_ext_desc_t(Structure):
    pass

struct__ze_cache_reservation_ext_desc_t._pack_ = 1 # source:False
struct__ze_cache_reservation_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('maxCacheReservationSize', ctypes.c_uint64),
]

ze_cache_reservation_ext_desc_t = struct__ze_cache_reservation_ext_desc_t
class struct__ze_image_memory_properties_exp_t(Structure):
    pass

struct__ze_image_memory_properties_exp_t._pack_ = 1 # source:False
struct__ze_image_memory_properties_exp_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
    ('rowPitch', ctypes.c_uint64),
    ('slicePitch', ctypes.c_uint64),
]

ze_image_memory_properties_exp_t = struct__ze_image_memory_properties_exp_t
class struct__ze_image_view_planar_ext_desc_t(Structure):
    pass

struct__ze_image_view_planar_ext_desc_t._pack_ = 1 # source:False
struct__ze_image_view_planar_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('planeIndex', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_image_view_planar_ext_desc_t = struct__ze_image_view_planar_ext_desc_t
class struct__ze_image_view_planar_exp_desc_t(Structure):
    pass

struct__ze_image_view_planar_exp_desc_t._pack_ = 1 # source:False
struct__ze_image_view_planar_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('planeIndex', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_image_view_planar_exp_desc_t = struct__ze_image_view_planar_exp_desc_t
class struct__ze_scheduling_hint_exp_properties_t(Structure):
    pass

struct__ze_scheduling_hint_exp_properties_t._pack_ = 1 # source:False
struct__ze_scheduling_hint_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('schedulingHintFlags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_scheduling_hint_exp_properties_t = struct__ze_scheduling_hint_exp_properties_t
class struct__ze_scheduling_hint_exp_desc_t(Structure):
    pass

struct__ze_scheduling_hint_exp_desc_t._pack_ = 1 # source:False
struct__ze_scheduling_hint_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_scheduling_hint_exp_desc_t = struct__ze_scheduling_hint_exp_desc_t
class struct__ze_context_power_saving_hint_exp_desc_t(Structure):
    pass

struct__ze_context_power_saving_hint_exp_desc_t._pack_ = 1 # source:False
struct__ze_context_power_saving_hint_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('hint', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_context_power_saving_hint_exp_desc_t = struct__ze_context_power_saving_hint_exp_desc_t
class struct__ze_eu_count_ext_t(Structure):
    pass

struct__ze_eu_count_ext_t._pack_ = 1 # source:False
struct__ze_eu_count_ext_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('numTotalEUs', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_eu_count_ext_t = struct__ze_eu_count_ext_t
class struct__ze_pci_address_ext_t(Structure):
    pass

struct__ze_pci_address_ext_t._pack_ = 1 # source:False
struct__ze_pci_address_ext_t._fields_ = [
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint32),
    ('device', ctypes.c_uint32),
    ('function', ctypes.c_uint32),
]

ze_pci_address_ext_t = struct__ze_pci_address_ext_t
class struct__ze_pci_speed_ext_t(Structure):
    pass

struct__ze_pci_speed_ext_t._pack_ = 1 # source:False
struct__ze_pci_speed_ext_t._fields_ = [
    ('genVersion', ctypes.c_int32),
    ('width', ctypes.c_int32),
    ('maxBandwidth', ctypes.c_int64),
]

ze_pci_speed_ext_t = struct__ze_pci_speed_ext_t
class struct__ze_pci_ext_properties_t(Structure):
    pass

struct__ze_pci_ext_properties_t._pack_ = 1 # source:False
struct__ze_pci_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('address', ze_pci_address_ext_t),
    ('maxSpeed', ze_pci_speed_ext_t),
]

ze_pci_ext_properties_t = struct__ze_pci_ext_properties_t
class struct__ze_srgb_ext_desc_t(Structure):
    pass

struct__ze_srgb_ext_desc_t._pack_ = 1 # source:False
struct__ze_srgb_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('sRGB', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

ze_srgb_ext_desc_t = struct__ze_srgb_ext_desc_t
class struct__ze_image_allocation_ext_properties_t(Structure):
    pass

struct__ze_image_allocation_ext_properties_t._pack_ = 1 # source:False
struct__ze_image_allocation_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('id', ctypes.c_uint64),
]

ze_image_allocation_ext_properties_t = struct__ze_image_allocation_ext_properties_t
class struct__ze_linkage_inspection_ext_desc_t(Structure):
    pass

struct__ze_linkage_inspection_ext_desc_t._pack_ = 1 # source:False
struct__ze_linkage_inspection_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_linkage_inspection_ext_desc_t = struct__ze_linkage_inspection_ext_desc_t
class struct__ze_memory_compression_hints_ext_desc_t(Structure):
    pass

struct__ze_memory_compression_hints_ext_desc_t._pack_ = 1 # source:False
struct__ze_memory_compression_hints_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_memory_compression_hints_ext_desc_t = struct__ze_memory_compression_hints_ext_desc_t
class struct__ze_driver_memory_free_ext_properties_t(Structure):
    pass

struct__ze_driver_memory_free_ext_properties_t._pack_ = 1 # source:False
struct__ze_driver_memory_free_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('freePolicies', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_driver_memory_free_ext_properties_t = struct__ze_driver_memory_free_ext_properties_t
class struct__ze_memory_free_ext_desc_t(Structure):
    pass

struct__ze_memory_free_ext_desc_t._pack_ = 1 # source:False
struct__ze_memory_free_ext_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('freePolicy', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_memory_free_ext_desc_t = struct__ze_memory_free_ext_desc_t
class struct__ze_device_p2p_bandwidth_exp_properties_t(Structure):
    pass

struct__ze_device_p2p_bandwidth_exp_properties_t._pack_ = 1 # source:False
struct__ze_device_p2p_bandwidth_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('logicalBandwidth', ctypes.c_uint32),
    ('physicalBandwidth', ctypes.c_uint32),
    ('bandwidthUnit', ze_bandwidth_unit_t),
    ('logicalLatency', ctypes.c_uint32),
    ('physicalLatency', ctypes.c_uint32),
    ('latencyUnit', ze_latency_unit_t),
]

ze_device_p2p_bandwidth_exp_properties_t = struct__ze_device_p2p_bandwidth_exp_properties_t
class struct__ze_copy_bandwidth_exp_properties_t(Structure):
    pass

struct__ze_copy_bandwidth_exp_properties_t._pack_ = 1 # source:False
struct__ze_copy_bandwidth_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('copyBandwidth', ctypes.c_uint32),
    ('copyBandwidthUnit', ze_bandwidth_unit_t),
]

ze_copy_bandwidth_exp_properties_t = struct__ze_copy_bandwidth_exp_properties_t
class struct__ze_device_luid_ext_t(Structure):
    pass

struct__ze_device_luid_ext_t._pack_ = 1 # source:False
struct__ze_device_luid_ext_t._fields_ = [
    ('id', ctypes.c_ubyte * 8),
]

ze_device_luid_ext_t = struct__ze_device_luid_ext_t
class struct__ze_device_luid_ext_properties_t(Structure):
    pass

struct__ze_device_luid_ext_properties_t._pack_ = 1 # source:False
struct__ze_device_luid_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('luid', ze_device_luid_ext_t),
    ('nodeMask', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_luid_ext_properties_t = struct__ze_device_luid_ext_properties_t
class struct__ze_fabric_vertex_pci_exp_address_t(Structure):
    pass

struct__ze_fabric_vertex_pci_exp_address_t._pack_ = 1 # source:False
struct__ze_fabric_vertex_pci_exp_address_t._fields_ = [
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint32),
    ('device', ctypes.c_uint32),
    ('function', ctypes.c_uint32),
]

ze_fabric_vertex_pci_exp_address_t = struct__ze_fabric_vertex_pci_exp_address_t
class struct__ze_fabric_vertex_exp_properties_t(Structure):
    pass


# values for enumeration '_ze_fabric_vertex_exp_type_t'
_ze_fabric_vertex_exp_type_t__enumvalues = {
    0: 'ZE_FABRIC_VERTEX_EXP_TYPE_UNKNOWN',
    1: 'ZE_FABRIC_VERTEX_EXP_TYPE_DEVICE',
    2: 'ZE_FABRIC_VERTEX_EXP_TYPE_SUBDEVICE',
    3: 'ZE_FABRIC_VERTEX_EXP_TYPE_SWITCH',
    2147483647: 'ZE_FABRIC_VERTEX_EXP_TYPE_FORCE_UINT32',
}
ZE_FABRIC_VERTEX_EXP_TYPE_UNKNOWN = 0
ZE_FABRIC_VERTEX_EXP_TYPE_DEVICE = 1
ZE_FABRIC_VERTEX_EXP_TYPE_SUBDEVICE = 2
ZE_FABRIC_VERTEX_EXP_TYPE_SWITCH = 3
ZE_FABRIC_VERTEX_EXP_TYPE_FORCE_UINT32 = 2147483647
_ze_fabric_vertex_exp_type_t = ctypes.c_uint32 # enum
ze_fabric_vertex_exp_type_t = _ze_fabric_vertex_exp_type_t
ze_fabric_vertex_exp_type_t__enumvalues = _ze_fabric_vertex_exp_type_t__enumvalues
struct__ze_fabric_vertex_exp_properties_t._pack_ = 1 # source:False
struct__ze_fabric_vertex_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('uuid', ze_uuid_t),
    ('type', ze_fabric_vertex_exp_type_t),
    ('remote', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('address', ze_fabric_vertex_pci_exp_address_t),
]

ze_fabric_vertex_exp_properties_t = struct__ze_fabric_vertex_exp_properties_t
class struct__ze_fabric_edge_exp_properties_t(Structure):
    pass


# values for enumeration '_ze_fabric_edge_exp_duplexity_t'
_ze_fabric_edge_exp_duplexity_t__enumvalues = {
    0: 'ZE_FABRIC_EDGE_EXP_DUPLEXITY_UNKNOWN',
    1: 'ZE_FABRIC_EDGE_EXP_DUPLEXITY_HALF_DUPLEX',
    2: 'ZE_FABRIC_EDGE_EXP_DUPLEXITY_FULL_DUPLEX',
    2147483647: 'ZE_FABRIC_EDGE_EXP_DUPLEXITY_FORCE_UINT32',
}
ZE_FABRIC_EDGE_EXP_DUPLEXITY_UNKNOWN = 0
ZE_FABRIC_EDGE_EXP_DUPLEXITY_HALF_DUPLEX = 1
ZE_FABRIC_EDGE_EXP_DUPLEXITY_FULL_DUPLEX = 2
ZE_FABRIC_EDGE_EXP_DUPLEXITY_FORCE_UINT32 = 2147483647
_ze_fabric_edge_exp_duplexity_t = ctypes.c_uint32 # enum
ze_fabric_edge_exp_duplexity_t = _ze_fabric_edge_exp_duplexity_t
ze_fabric_edge_exp_duplexity_t__enumvalues = _ze_fabric_edge_exp_duplexity_t__enumvalues
struct__ze_fabric_edge_exp_properties_t._pack_ = 1 # source:False
struct__ze_fabric_edge_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('uuid', ze_uuid_t),
    ('model', ctypes.c_char * 256),
    ('bandwidth', ctypes.c_uint32),
    ('bandwidthUnit', ze_bandwidth_unit_t),
    ('latency', ctypes.c_uint32),
    ('latencyUnit', ze_latency_unit_t),
    ('duplexity', ze_fabric_edge_exp_duplexity_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_fabric_edge_exp_properties_t = struct__ze_fabric_edge_exp_properties_t
class struct__ze_device_memory_ext_properties_t(Structure):
    pass


# values for enumeration '_ze_device_memory_ext_type_t'
_ze_device_memory_ext_type_t__enumvalues = {
    0: 'ZE_DEVICE_MEMORY_EXT_TYPE_HBM',
    1: 'ZE_DEVICE_MEMORY_EXT_TYPE_HBM2',
    2: 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR',
    3: 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR2',
    4: 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR3',
    5: 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR4',
    6: 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR5',
    7: 'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR',
    8: 'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR3',
    9: 'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR4',
    10: 'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR5',
    11: 'ZE_DEVICE_MEMORY_EXT_TYPE_SRAM',
    12: 'ZE_DEVICE_MEMORY_EXT_TYPE_L1',
    13: 'ZE_DEVICE_MEMORY_EXT_TYPE_L3',
    14: 'ZE_DEVICE_MEMORY_EXT_TYPE_GRF',
    15: 'ZE_DEVICE_MEMORY_EXT_TYPE_SLM',
    16: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR4',
    17: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5',
    18: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5X',
    19: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6',
    20: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6X',
    21: 'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR7',
    2147483647: 'ZE_DEVICE_MEMORY_EXT_TYPE_FORCE_UINT32',
}
ZE_DEVICE_MEMORY_EXT_TYPE_HBM = 0
ZE_DEVICE_MEMORY_EXT_TYPE_HBM2 = 1
ZE_DEVICE_MEMORY_EXT_TYPE_DDR = 2
ZE_DEVICE_MEMORY_EXT_TYPE_DDR2 = 3
ZE_DEVICE_MEMORY_EXT_TYPE_DDR3 = 4
ZE_DEVICE_MEMORY_EXT_TYPE_DDR4 = 5
ZE_DEVICE_MEMORY_EXT_TYPE_DDR5 = 6
ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR = 7
ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR3 = 8
ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR4 = 9
ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR5 = 10
ZE_DEVICE_MEMORY_EXT_TYPE_SRAM = 11
ZE_DEVICE_MEMORY_EXT_TYPE_L1 = 12
ZE_DEVICE_MEMORY_EXT_TYPE_L3 = 13
ZE_DEVICE_MEMORY_EXT_TYPE_GRF = 14
ZE_DEVICE_MEMORY_EXT_TYPE_SLM = 15
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR4 = 16
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5 = 17
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5X = 18
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6 = 19
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6X = 20
ZE_DEVICE_MEMORY_EXT_TYPE_GDDR7 = 21
ZE_DEVICE_MEMORY_EXT_TYPE_FORCE_UINT32 = 2147483647
_ze_device_memory_ext_type_t = ctypes.c_uint32 # enum
ze_device_memory_ext_type_t = _ze_device_memory_ext_type_t
ze_device_memory_ext_type_t__enumvalues = _ze_device_memory_ext_type_t__enumvalues
struct__ze_device_memory_ext_properties_t._pack_ = 1 # source:False
struct__ze_device_memory_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('type', ze_device_memory_ext_type_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('physicalSize', ctypes.c_uint64),
    ('readBandwidth', ctypes.c_uint32),
    ('writeBandwidth', ctypes.c_uint32),
    ('bandwidthUnit', ze_bandwidth_unit_t),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

ze_device_memory_ext_properties_t = struct__ze_device_memory_ext_properties_t
class struct__ze_device_ip_version_ext_t(Structure):
    pass

struct__ze_device_ip_version_ext_t._pack_ = 1 # source:False
struct__ze_device_ip_version_ext_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('ipVersion', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_device_ip_version_ext_t = struct__ze_device_ip_version_ext_t
class struct__ze_kernel_max_group_size_properties_ext_t(Structure):
    pass

struct__ze_kernel_max_group_size_properties_ext_t._pack_ = 1 # source:False
struct__ze_kernel_max_group_size_properties_ext_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('maxGroupSize', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_kernel_max_group_size_properties_ext_t = struct__ze_kernel_max_group_size_properties_ext_t
class struct__ze_sub_allocation_t(Structure):
    pass

struct__ze_sub_allocation_t._pack_ = 1 # source:False
struct__ze_sub_allocation_t._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
]

ze_sub_allocation_t = struct__ze_sub_allocation_t
class struct__ze_memory_sub_allocations_exp_properties_t(Structure):
    pass

struct__ze_memory_sub_allocations_exp_properties_t._pack_ = 1 # source:False
struct__ze_memory_sub_allocations_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('pCount', ctypes.POINTER(ctypes.c_uint32)),
    ('pSubAllocations', ctypes.POINTER(struct__ze_sub_allocation_t)),
]

ze_memory_sub_allocations_exp_properties_t = struct__ze_memory_sub_allocations_exp_properties_t
class struct__ze_event_query_kernel_timestamps_ext_properties_t(Structure):
    pass

struct__ze_event_query_kernel_timestamps_ext_properties_t._pack_ = 1 # source:False
struct__ze_event_query_kernel_timestamps_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_event_query_kernel_timestamps_ext_properties_t = struct__ze_event_query_kernel_timestamps_ext_properties_t
class struct__ze_synchronized_timestamp_data_ext_t(Structure):
    pass

struct__ze_synchronized_timestamp_data_ext_t._pack_ = 1 # source:False
struct__ze_synchronized_timestamp_data_ext_t._fields_ = [
    ('kernelStart', ctypes.c_uint64),
    ('kernelEnd', ctypes.c_uint64),
]

ze_synchronized_timestamp_data_ext_t = struct__ze_synchronized_timestamp_data_ext_t
class struct__ze_synchronized_timestamp_result_ext_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('global', ze_synchronized_timestamp_data_ext_t),
    ('context', ze_synchronized_timestamp_data_ext_t),
     ]

ze_synchronized_timestamp_result_ext_t = struct__ze_synchronized_timestamp_result_ext_t
class struct__ze_event_query_kernel_timestamps_results_ext_properties_t(Structure):
    pass

struct__ze_event_query_kernel_timestamps_results_ext_properties_t._pack_ = 1 # source:False
struct__ze_event_query_kernel_timestamps_results_ext_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('pKernelTimestampsBuffer', ctypes.POINTER(struct__ze_kernel_timestamp_result_t)),
    ('pSynchronizedTimestampsBuffer', ctypes.POINTER(struct__ze_synchronized_timestamp_result_ext_t)),
]

ze_event_query_kernel_timestamps_results_ext_properties_t = struct__ze_event_query_kernel_timestamps_results_ext_properties_t
class struct__ze_rtas_builder_exp_desc_t(Structure):
    pass


# values for enumeration '_ze_rtas_builder_exp_version_t'
_ze_rtas_builder_exp_version_t__enumvalues = {
    65536: 'ZE_RTAS_BUILDER_EXP_VERSION_1_0',
    65536: 'ZE_RTAS_BUILDER_EXP_VERSION_CURRENT',
    2147483647: 'ZE_RTAS_BUILDER_EXP_VERSION_FORCE_UINT32',
}
ZE_RTAS_BUILDER_EXP_VERSION_1_0 = 65536
ZE_RTAS_BUILDER_EXP_VERSION_CURRENT = 65536
ZE_RTAS_BUILDER_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_rtas_builder_exp_version_t = ctypes.c_uint32 # enum
ze_rtas_builder_exp_version_t = _ze_rtas_builder_exp_version_t
ze_rtas_builder_exp_version_t__enumvalues = _ze_rtas_builder_exp_version_t__enumvalues
struct__ze_rtas_builder_exp_desc_t._pack_ = 1 # source:False
struct__ze_rtas_builder_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('builderVersion', ze_rtas_builder_exp_version_t),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_rtas_builder_exp_desc_t = struct__ze_rtas_builder_exp_desc_t
class struct__ze_rtas_builder_exp_properties_t(Structure):
    pass

struct__ze_rtas_builder_exp_properties_t._pack_ = 1 # source:False
struct__ze_rtas_builder_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('rtasBufferSizeBytesExpected', ctypes.c_uint64),
    ('rtasBufferSizeBytesMaxRequired', ctypes.c_uint64),
    ('scratchBufferSizeBytes', ctypes.c_uint64),
]

ze_rtas_builder_exp_properties_t = struct__ze_rtas_builder_exp_properties_t
class struct__ze_rtas_parallel_operation_exp_properties_t(Structure):
    pass

struct__ze_rtas_parallel_operation_exp_properties_t._pack_ = 1 # source:False
struct__ze_rtas_parallel_operation_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('maxConcurrency', ctypes.c_uint32),
]

ze_rtas_parallel_operation_exp_properties_t = struct__ze_rtas_parallel_operation_exp_properties_t
class struct__ze_rtas_device_exp_properties_t(Structure):
    pass


# values for enumeration '_ze_rtas_format_exp_t'
_ze_rtas_format_exp_t__enumvalues = {
    0: 'ZE_RTAS_FORMAT_EXP_INVALID',
    2147483647: 'ZE_RTAS_FORMAT_EXP_FORCE_UINT32',
}
ZE_RTAS_FORMAT_EXP_INVALID = 0
ZE_RTAS_FORMAT_EXP_FORCE_UINT32 = 2147483647
_ze_rtas_format_exp_t = ctypes.c_uint32 # enum
ze_rtas_format_exp_t = _ze_rtas_format_exp_t
ze_rtas_format_exp_t__enumvalues = _ze_rtas_format_exp_t__enumvalues
struct__ze_rtas_device_exp_properties_t._pack_ = 1 # source:False
struct__ze_rtas_device_exp_properties_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('flags', ctypes.c_uint32),
    ('rtasFormat', ze_rtas_format_exp_t),
    ('rtasBufferAlignment', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ze_rtas_device_exp_properties_t = struct__ze_rtas_device_exp_properties_t
class struct__ze_rtas_float3_exp_t(Structure):
    pass

struct__ze_rtas_float3_exp_t._pack_ = 1 # source:False
struct__ze_rtas_float3_exp_t._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('z', ctypes.c_float),
]

ze_rtas_float3_exp_t = struct__ze_rtas_float3_exp_t
class struct__ze_rtas_transform_float3x4_column_major_exp_t(Structure):
    pass

struct__ze_rtas_transform_float3x4_column_major_exp_t._pack_ = 1 # source:False
struct__ze_rtas_transform_float3x4_column_major_exp_t._fields_ = [
    ('vx_x', ctypes.c_float),
    ('vx_y', ctypes.c_float),
    ('vx_z', ctypes.c_float),
    ('vy_x', ctypes.c_float),
    ('vy_y', ctypes.c_float),
    ('vy_z', ctypes.c_float),
    ('vz_x', ctypes.c_float),
    ('vz_y', ctypes.c_float),
    ('vz_z', ctypes.c_float),
    ('p_x', ctypes.c_float),
    ('p_y', ctypes.c_float),
    ('p_z', ctypes.c_float),
]

ze_rtas_transform_float3x4_column_major_exp_t = struct__ze_rtas_transform_float3x4_column_major_exp_t
class struct__ze_rtas_transform_float3x4_aligned_column_major_exp_t(Structure):
    pass

struct__ze_rtas_transform_float3x4_aligned_column_major_exp_t._pack_ = 1 # source:False
struct__ze_rtas_transform_float3x4_aligned_column_major_exp_t._fields_ = [
    ('vx_x', ctypes.c_float),
    ('vx_y', ctypes.c_float),
    ('vx_z', ctypes.c_float),
    ('pad0', ctypes.c_float),
    ('vy_x', ctypes.c_float),
    ('vy_y', ctypes.c_float),
    ('vy_z', ctypes.c_float),
    ('pad1', ctypes.c_float),
    ('vz_x', ctypes.c_float),
    ('vz_y', ctypes.c_float),
    ('vz_z', ctypes.c_float),
    ('pad2', ctypes.c_float),
    ('p_x', ctypes.c_float),
    ('p_y', ctypes.c_float),
    ('p_z', ctypes.c_float),
    ('pad3', ctypes.c_float),
]

ze_rtas_transform_float3x4_aligned_column_major_exp_t = struct__ze_rtas_transform_float3x4_aligned_column_major_exp_t
class struct__ze_rtas_transform_float3x4_row_major_exp_t(Structure):
    pass

struct__ze_rtas_transform_float3x4_row_major_exp_t._pack_ = 1 # source:False
struct__ze_rtas_transform_float3x4_row_major_exp_t._fields_ = [
    ('vx_x', ctypes.c_float),
    ('vy_x', ctypes.c_float),
    ('vz_x', ctypes.c_float),
    ('p_x', ctypes.c_float),
    ('vx_y', ctypes.c_float),
    ('vy_y', ctypes.c_float),
    ('vz_y', ctypes.c_float),
    ('p_y', ctypes.c_float),
    ('vx_z', ctypes.c_float),
    ('vy_z', ctypes.c_float),
    ('vz_z', ctypes.c_float),
    ('p_z', ctypes.c_float),
]

ze_rtas_transform_float3x4_row_major_exp_t = struct__ze_rtas_transform_float3x4_row_major_exp_t
class struct__ze_rtas_aabb_exp_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lower', ze_rtas_float3_exp_t),
    ('upper', ze_rtas_float3_exp_t),
     ]

ze_rtas_aabb_exp_t = struct__ze_rtas_aabb_exp_t
class struct__ze_rtas_triangle_indices_uint32_exp_t(Structure):
    pass

struct__ze_rtas_triangle_indices_uint32_exp_t._pack_ = 1 # source:False
struct__ze_rtas_triangle_indices_uint32_exp_t._fields_ = [
    ('v0', ctypes.c_uint32),
    ('v1', ctypes.c_uint32),
    ('v2', ctypes.c_uint32),
]

ze_rtas_triangle_indices_uint32_exp_t = struct__ze_rtas_triangle_indices_uint32_exp_t
class struct__ze_rtas_quad_indices_uint32_exp_t(Structure):
    pass

struct__ze_rtas_quad_indices_uint32_exp_t._pack_ = 1 # source:False
struct__ze_rtas_quad_indices_uint32_exp_t._fields_ = [
    ('v0', ctypes.c_uint32),
    ('v1', ctypes.c_uint32),
    ('v2', ctypes.c_uint32),
    ('v3', ctypes.c_uint32),
]

ze_rtas_quad_indices_uint32_exp_t = struct__ze_rtas_quad_indices_uint32_exp_t
class struct__ze_rtas_builder_geometry_info_exp_t(Structure):
    pass

struct__ze_rtas_builder_geometry_info_exp_t._pack_ = 1 # source:False
struct__ze_rtas_builder_geometry_info_exp_t._fields_ = [
    ('geometryType', ctypes.c_ubyte),
]

ze_rtas_builder_geometry_info_exp_t = struct__ze_rtas_builder_geometry_info_exp_t
class struct__ze_rtas_builder_triangles_geometry_info_exp_t(Structure):
    pass

struct__ze_rtas_builder_triangles_geometry_info_exp_t._pack_ = 1 # source:False
struct__ze_rtas_builder_triangles_geometry_info_exp_t._fields_ = [
    ('geometryType', ctypes.c_ubyte),
    ('geometryFlags', ctypes.c_ubyte),
    ('geometryMask', ctypes.c_ubyte),
    ('triangleFormat', ctypes.c_ubyte),
    ('vertexFormat', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('triangleCount', ctypes.c_uint32),
    ('vertexCount', ctypes.c_uint32),
    ('triangleStride', ctypes.c_uint32),
    ('vertexStride', ctypes.c_uint32),
    ('pTriangleBuffer', ctypes.POINTER(None)),
    ('pVertexBuffer', ctypes.POINTER(None)),
]

ze_rtas_builder_triangles_geometry_info_exp_t = struct__ze_rtas_builder_triangles_geometry_info_exp_t
class struct__ze_rtas_builder_quads_geometry_info_exp_t(Structure):
    pass

struct__ze_rtas_builder_quads_geometry_info_exp_t._pack_ = 1 # source:False
struct__ze_rtas_builder_quads_geometry_info_exp_t._fields_ = [
    ('geometryType', ctypes.c_ubyte),
    ('geometryFlags', ctypes.c_ubyte),
    ('geometryMask', ctypes.c_ubyte),
    ('quadFormat', ctypes.c_ubyte),
    ('vertexFormat', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('quadCount', ctypes.c_uint32),
    ('vertexCount', ctypes.c_uint32),
    ('quadStride', ctypes.c_uint32),
    ('vertexStride', ctypes.c_uint32),
    ('pQuadBuffer', ctypes.POINTER(None)),
    ('pVertexBuffer', ctypes.POINTER(None)),
]

ze_rtas_builder_quads_geometry_info_exp_t = struct__ze_rtas_builder_quads_geometry_info_exp_t
class struct__ze_rtas_geometry_aabbs_exp_cb_params_t(Structure):
    pass

struct__ze_rtas_geometry_aabbs_exp_cb_params_t._pack_ = 1 # source:False
struct__ze_rtas_geometry_aabbs_exp_cb_params_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('primID', ctypes.c_uint32),
    ('primIDCount', ctypes.c_uint32),
    ('pGeomUserPtr', ctypes.POINTER(None)),
    ('pBuildUserPtr', ctypes.POINTER(None)),
    ('pBoundsOut', ctypes.POINTER(struct__ze_rtas_aabb_exp_t)),
]

ze_rtas_geometry_aabbs_exp_cb_params_t = struct__ze_rtas_geometry_aabbs_exp_cb_params_t
class struct__ze_rtas_builder_procedural_geometry_info_exp_t(Structure):
    pass

struct__ze_rtas_builder_procedural_geometry_info_exp_t._pack_ = 1 # source:False
struct__ze_rtas_builder_procedural_geometry_info_exp_t._fields_ = [
    ('geometryType', ctypes.c_ubyte),
    ('geometryFlags', ctypes.c_ubyte),
    ('geometryMask', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
    ('primCount', ctypes.c_uint32),
    ('pfnGetBoundsCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_rtas_geometry_aabbs_exp_cb_params_t))),
    ('pGeomUserPtr', ctypes.POINTER(None)),
]

ze_rtas_builder_procedural_geometry_info_exp_t = struct__ze_rtas_builder_procedural_geometry_info_exp_t
class struct__ze_rtas_builder_instance_geometry_info_exp_t(Structure):
    pass

struct__ze_rtas_builder_instance_geometry_info_exp_t._pack_ = 1 # source:False
struct__ze_rtas_builder_instance_geometry_info_exp_t._fields_ = [
    ('geometryType', ctypes.c_ubyte),
    ('instanceFlags', ctypes.c_ubyte),
    ('geometryMask', ctypes.c_ubyte),
    ('transformFormat', ctypes.c_ubyte),
    ('instanceUserID', ctypes.c_uint32),
    ('pTransform', ctypes.POINTER(None)),
    ('pBounds', ctypes.POINTER(struct__ze_rtas_aabb_exp_t)),
    ('pAccelerationStructure', ctypes.POINTER(None)),
]

ze_rtas_builder_instance_geometry_info_exp_t = struct__ze_rtas_builder_instance_geometry_info_exp_t
class struct__ze_rtas_builder_build_op_exp_desc_t(Structure):
    pass


# values for enumeration '_ze_rtas_builder_build_quality_hint_exp_t'
_ze_rtas_builder_build_quality_hint_exp_t__enumvalues = {
    0: 'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_LOW',
    1: 'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM',
    2: 'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH',
    2147483647: 'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_FORCE_UINT32',
}
ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_LOW = 0
ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM = 1
ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH = 2
ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_FORCE_UINT32 = 2147483647
_ze_rtas_builder_build_quality_hint_exp_t = ctypes.c_uint32 # enum
ze_rtas_builder_build_quality_hint_exp_t = _ze_rtas_builder_build_quality_hint_exp_t
ze_rtas_builder_build_quality_hint_exp_t__enumvalues = _ze_rtas_builder_build_quality_hint_exp_t__enumvalues
struct__ze_rtas_builder_build_op_exp_desc_t._pack_ = 1 # source:False
struct__ze_rtas_builder_build_op_exp_desc_t._fields_ = [
    ('stype', ze_structure_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pNext', ctypes.POINTER(None)),
    ('rtasFormat', ze_rtas_format_exp_t),
    ('buildQuality', ze_rtas_builder_build_quality_hint_exp_t),
    ('buildFlags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('ppGeometries', ctypes.POINTER(ctypes.POINTER(struct__ze_rtas_builder_geometry_info_exp_t))),
    ('numGeometries', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

ze_rtas_builder_build_op_exp_desc_t = struct__ze_rtas_builder_build_op_exp_desc_t
ze_init_flags_t = ctypes.c_uint32

# values for enumeration '_ze_init_flag_t'
_ze_init_flag_t__enumvalues = {
    1: 'ZE_INIT_FLAG_GPU_ONLY',
    2: 'ZE_INIT_FLAG_VPU_ONLY',
    2147483647: 'ZE_INIT_FLAG_FORCE_UINT32',
}
ZE_INIT_FLAG_GPU_ONLY = 1
ZE_INIT_FLAG_VPU_ONLY = 2
ZE_INIT_FLAG_FORCE_UINT32 = 2147483647
_ze_init_flag_t = ctypes.c_uint32 # enum
ze_init_flag_t = _ze_init_flag_t
ze_init_flag_t__enumvalues = _ze_init_flag_t__enumvalues
try:
    zeInit = _libraries['libze_loader.so'].zeInit
    zeInit.restype = ze_result_t
    zeInit.argtypes = [ze_init_flags_t]
except AttributeError:
    pass
try:
    zeDriverGet = _libraries['libze_loader.so'].zeDriverGet
    zeDriverGet.restype = ze_result_t
    zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_api_version_t'
_ze_api_version_t__enumvalues = {
    65536: 'ZE_API_VERSION_1_0',
    65537: 'ZE_API_VERSION_1_1',
    65538: 'ZE_API_VERSION_1_2',
    65539: 'ZE_API_VERSION_1_3',
    65540: 'ZE_API_VERSION_1_4',
    65541: 'ZE_API_VERSION_1_5',
    65542: 'ZE_API_VERSION_1_6',
    65543: 'ZE_API_VERSION_1_7',
    65543: 'ZE_API_VERSION_CURRENT',
    2147483647: 'ZE_API_VERSION_FORCE_UINT32',
}
ZE_API_VERSION_1_0 = 65536
ZE_API_VERSION_1_1 = 65537
ZE_API_VERSION_1_2 = 65538
ZE_API_VERSION_1_3 = 65539
ZE_API_VERSION_1_4 = 65540
ZE_API_VERSION_1_5 = 65541
ZE_API_VERSION_1_6 = 65542
ZE_API_VERSION_1_7 = 65543
ZE_API_VERSION_CURRENT = 65543
ZE_API_VERSION_FORCE_UINT32 = 2147483647
_ze_api_version_t = ctypes.c_uint32 # enum
ze_api_version_t = _ze_api_version_t
ze_api_version_t__enumvalues = _ze_api_version_t__enumvalues
try:
    zeDriverGetApiVersion = _libraries['libze_loader.so'].zeDriverGetApiVersion
    zeDriverGetApiVersion.restype = ze_result_t
    zeDriverGetApiVersion.argtypes = [ze_driver_handle_t, ctypes.POINTER(_ze_api_version_t)]
except AttributeError:
    pass
try:
    zeDriverGetProperties = _libraries['libze_loader.so'].zeDriverGetProperties
    zeDriverGetProperties.restype = ze_result_t
    zeDriverGetProperties.argtypes = [ze_driver_handle_t, ctypes.POINTER(struct__ze_driver_properties_t)]
except AttributeError:
    pass
ze_ipc_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_ipc_property_flag_t'
_ze_ipc_property_flag_t__enumvalues = {
    1: 'ZE_IPC_PROPERTY_FLAG_MEMORY',
    2: 'ZE_IPC_PROPERTY_FLAG_EVENT_POOL',
    2147483647: 'ZE_IPC_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_IPC_PROPERTY_FLAG_MEMORY = 1
ZE_IPC_PROPERTY_FLAG_EVENT_POOL = 2
ZE_IPC_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_ipc_property_flag_t = ctypes.c_uint32 # enum
ze_ipc_property_flag_t = _ze_ipc_property_flag_t
ze_ipc_property_flag_t__enumvalues = _ze_ipc_property_flag_t__enumvalues
try:
    zeDriverGetIpcProperties = _libraries['libze_loader.so'].zeDriverGetIpcProperties
    zeDriverGetIpcProperties.restype = ze_result_t
    zeDriverGetIpcProperties.argtypes = [ze_driver_handle_t, ctypes.POINTER(struct__ze_driver_ipc_properties_t)]
except AttributeError:
    pass
try:
    zeDriverGetExtensionProperties = _libraries['libze_loader.so'].zeDriverGetExtensionProperties
    zeDriverGetExtensionProperties.restype = ze_result_t
    zeDriverGetExtensionProperties.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_driver_extension_properties_t)]
except AttributeError:
    pass
try:
    zeDriverGetExtensionFunctionAddress = _libraries['libze_loader.so'].zeDriverGetExtensionFunctionAddress
    zeDriverGetExtensionFunctionAddress.restype = ze_result_t
    zeDriverGetExtensionFunctionAddress.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeDriverGetLastErrorDescription = _libraries['libze_loader.so'].zeDriverGetLastErrorDescription
    zeDriverGetLastErrorDescription.restype = ze_result_t
    zeDriverGetLastErrorDescription.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    zeDeviceGet = _libraries['libze_loader.so'].zeDeviceGet
    zeDeviceGet.restype = ze_result_t
    zeDeviceGet.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))]
except AttributeError:
    pass
try:
    zeDeviceGetRootDevice = _libraries['libze_loader.so'].zeDeviceGetRootDevice
    zeDeviceGetRootDevice.restype = ze_result_t
    zeDeviceGetRootDevice.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))]
except AttributeError:
    pass
try:
    zeDeviceGetSubDevices = _libraries['libze_loader.so'].zeDeviceGetSubDevices
    zeDeviceGetSubDevices.restype = ze_result_t
    zeDeviceGetSubDevices.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))]
except AttributeError:
    pass
ze_device_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_property_flag_t'
_ze_device_property_flag_t__enumvalues = {
    1: 'ZE_DEVICE_PROPERTY_FLAG_INTEGRATED',
    2: 'ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE',
    4: 'ZE_DEVICE_PROPERTY_FLAG_ECC',
    8: 'ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING',
    2147483647: 'ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1
ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = 2
ZE_DEVICE_PROPERTY_FLAG_ECC = 4
ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = 8
ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_device_property_flag_t = ctypes.c_uint32 # enum
ze_device_property_flag_t = _ze_device_property_flag_t
ze_device_property_flag_t__enumvalues = _ze_device_property_flag_t__enumvalues
try:
    zeDeviceGetProperties = _libraries['libze_loader.so'].zeDeviceGetProperties
    zeDeviceGetProperties.restype = ze_result_t
    zeDeviceGetProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_properties_t)]
except AttributeError:
    pass
try:
    zeDeviceGetComputeProperties = _libraries['libze_loader.so'].zeDeviceGetComputeProperties
    zeDeviceGetComputeProperties.restype = ze_result_t
    zeDeviceGetComputeProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_compute_properties_t)]
except AttributeError:
    pass
ze_device_module_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_module_flag_t'
_ze_device_module_flag_t__enumvalues = {
    1: 'ZE_DEVICE_MODULE_FLAG_FP16',
    2: 'ZE_DEVICE_MODULE_FLAG_FP64',
    4: 'ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS',
    8: 'ZE_DEVICE_MODULE_FLAG_DP4A',
    2147483647: 'ZE_DEVICE_MODULE_FLAG_FORCE_UINT32',
}
ZE_DEVICE_MODULE_FLAG_FP16 = 1
ZE_DEVICE_MODULE_FLAG_FP64 = 2
ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = 4
ZE_DEVICE_MODULE_FLAG_DP4A = 8
ZE_DEVICE_MODULE_FLAG_FORCE_UINT32 = 2147483647
_ze_device_module_flag_t = ctypes.c_uint32 # enum
ze_device_module_flag_t = _ze_device_module_flag_t
ze_device_module_flag_t__enumvalues = _ze_device_module_flag_t__enumvalues
ze_device_fp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_fp_flag_t'
_ze_device_fp_flag_t__enumvalues = {
    1: 'ZE_DEVICE_FP_FLAG_DENORM',
    2: 'ZE_DEVICE_FP_FLAG_INF_NAN',
    4: 'ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST',
    8: 'ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO',
    16: 'ZE_DEVICE_FP_FLAG_ROUND_TO_INF',
    32: 'ZE_DEVICE_FP_FLAG_FMA',
    64: 'ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT',
    128: 'ZE_DEVICE_FP_FLAG_SOFT_FLOAT',
    2147483647: 'ZE_DEVICE_FP_FLAG_FORCE_UINT32',
}
ZE_DEVICE_FP_FLAG_DENORM = 1
ZE_DEVICE_FP_FLAG_INF_NAN = 2
ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST = 4
ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO = 8
ZE_DEVICE_FP_FLAG_ROUND_TO_INF = 16
ZE_DEVICE_FP_FLAG_FMA = 32
ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT = 64
ZE_DEVICE_FP_FLAG_SOFT_FLOAT = 128
ZE_DEVICE_FP_FLAG_FORCE_UINT32 = 2147483647
_ze_device_fp_flag_t = ctypes.c_uint32 # enum
ze_device_fp_flag_t = _ze_device_fp_flag_t
ze_device_fp_flag_t__enumvalues = _ze_device_fp_flag_t__enumvalues
try:
    zeDeviceGetModuleProperties = _libraries['libze_loader.so'].zeDeviceGetModuleProperties
    zeDeviceGetModuleProperties.restype = ze_result_t
    zeDeviceGetModuleProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_module_properties_t)]
except AttributeError:
    pass
ze_command_queue_group_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_command_queue_group_property_flag_t'
_ze_command_queue_group_property_flag_t__enumvalues = {
    1: 'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE',
    2: 'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY',
    4: 'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS',
    8: 'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS',
    2147483647: 'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 1
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = 2
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = 4
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = 8
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_command_queue_group_property_flag_t = ctypes.c_uint32 # enum
ze_command_queue_group_property_flag_t = _ze_command_queue_group_property_flag_t
ze_command_queue_group_property_flag_t__enumvalues = _ze_command_queue_group_property_flag_t__enumvalues
try:
    zeDeviceGetCommandQueueGroupProperties = _libraries['libze_loader.so'].zeDeviceGetCommandQueueGroupProperties
    zeDeviceGetCommandQueueGroupProperties.restype = ze_result_t
    zeDeviceGetCommandQueueGroupProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_command_queue_group_properties_t)]
except AttributeError:
    pass
ze_device_memory_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_memory_property_flag_t'
_ze_device_memory_property_flag_t__enumvalues = {
    1: 'ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD',
    2147483647: 'ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD = 1
ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_device_memory_property_flag_t = ctypes.c_uint32 # enum
ze_device_memory_property_flag_t = _ze_device_memory_property_flag_t
ze_device_memory_property_flag_t__enumvalues = _ze_device_memory_property_flag_t__enumvalues
try:
    zeDeviceGetMemoryProperties = _libraries['libze_loader.so'].zeDeviceGetMemoryProperties
    zeDeviceGetMemoryProperties.restype = ze_result_t
    zeDeviceGetMemoryProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_device_memory_properties_t)]
except AttributeError:
    pass
ze_memory_access_cap_flags_t = ctypes.c_uint32

# values for enumeration '_ze_memory_access_cap_flag_t'
_ze_memory_access_cap_flag_t__enumvalues = {
    1: 'ZE_MEMORY_ACCESS_CAP_FLAG_RW',
    2: 'ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC',
    4: 'ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT',
    8: 'ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC',
    2147483647: 'ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32',
}
ZE_MEMORY_ACCESS_CAP_FLAG_RW = 1
ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC = 2
ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT = 4
ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC = 8
ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32 = 2147483647
_ze_memory_access_cap_flag_t = ctypes.c_uint32 # enum
ze_memory_access_cap_flag_t = _ze_memory_access_cap_flag_t
ze_memory_access_cap_flag_t__enumvalues = _ze_memory_access_cap_flag_t__enumvalues
try:
    zeDeviceGetMemoryAccessProperties = _libraries['libze_loader.so'].zeDeviceGetMemoryAccessProperties
    zeDeviceGetMemoryAccessProperties.restype = ze_result_t
    zeDeviceGetMemoryAccessProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_memory_access_properties_t)]
except AttributeError:
    pass
ze_device_cache_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_cache_property_flag_t'
_ze_device_cache_property_flag_t__enumvalues = {
    1: 'ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL',
    2147483647: 'ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL = 1
ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_device_cache_property_flag_t = ctypes.c_uint32 # enum
ze_device_cache_property_flag_t = _ze_device_cache_property_flag_t
ze_device_cache_property_flag_t__enumvalues = _ze_device_cache_property_flag_t__enumvalues
try:
    zeDeviceGetCacheProperties = _libraries['libze_loader.so'].zeDeviceGetCacheProperties
    zeDeviceGetCacheProperties.restype = ze_result_t
    zeDeviceGetCacheProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_device_cache_properties_t)]
except AttributeError:
    pass
try:
    zeDeviceGetImageProperties = _libraries['libze_loader.so'].zeDeviceGetImageProperties
    zeDeviceGetImageProperties.restype = ze_result_t
    zeDeviceGetImageProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_image_properties_t)]
except AttributeError:
    pass
try:
    zeDeviceGetExternalMemoryProperties = _libraries['libze_loader.so'].zeDeviceGetExternalMemoryProperties
    zeDeviceGetExternalMemoryProperties.restype = ze_result_t
    zeDeviceGetExternalMemoryProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_device_external_memory_properties_t)]
except AttributeError:
    pass
ze_device_p2p_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_p2p_property_flag_t'
_ze_device_p2p_property_flag_t__enumvalues = {
    1: 'ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS',
    2: 'ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS',
    2147483647: 'ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = 1
ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = 2
ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_device_p2p_property_flag_t = ctypes.c_uint32 # enum
ze_device_p2p_property_flag_t = _ze_device_p2p_property_flag_t
ze_device_p2p_property_flag_t__enumvalues = _ze_device_p2p_property_flag_t__enumvalues
try:
    zeDeviceGetP2PProperties = _libraries['libze_loader.so'].zeDeviceGetP2PProperties
    zeDeviceGetP2PProperties.restype = ze_result_t
    zeDeviceGetP2PProperties.argtypes = [ze_device_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_device_p2p_properties_t)]
except AttributeError:
    pass
try:
    zeDeviceCanAccessPeer = _libraries['libze_loader.so'].zeDeviceCanAccessPeer
    zeDeviceCanAccessPeer.restype = ze_result_t
    zeDeviceCanAccessPeer.argtypes = [ze_device_handle_t, ze_device_handle_t, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    zeDeviceGetStatus = _libraries['libze_loader.so'].zeDeviceGetStatus
    zeDeviceGetStatus.restype = ze_result_t
    zeDeviceGetStatus.argtypes = [ze_device_handle_t]
except AttributeError:
    pass
try:
    zeDeviceGetGlobalTimestamps = _libraries['libze_loader.so'].zeDeviceGetGlobalTimestamps
    zeDeviceGetGlobalTimestamps.restype = ze_result_t
    zeDeviceGetGlobalTimestamps.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
ze_context_flags_t = ctypes.c_uint32

# values for enumeration '_ze_context_flag_t'
_ze_context_flag_t__enumvalues = {
    1: 'ZE_CONTEXT_FLAG_TBD',
    2147483647: 'ZE_CONTEXT_FLAG_FORCE_UINT32',
}
ZE_CONTEXT_FLAG_TBD = 1
ZE_CONTEXT_FLAG_FORCE_UINT32 = 2147483647
_ze_context_flag_t = ctypes.c_uint32 # enum
ze_context_flag_t = _ze_context_flag_t
ze_context_flag_t__enumvalues = _ze_context_flag_t__enumvalues
try:
    zeContextCreate = _libraries['libze_loader.so'].zeContextCreate
    zeContextCreate.restype = ze_result_t
    zeContextCreate.argtypes = [ze_driver_handle_t, ctypes.POINTER(struct__ze_context_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    zeContextCreateEx = _libraries['libze_loader.so'].zeContextCreateEx
    zeContextCreateEx.restype = ze_result_t
    zeContextCreateEx.argtypes = [ze_driver_handle_t, ctypes.POINTER(struct__ze_context_desc_t), uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))]
except AttributeError:
    pass
try:
    zeContextDestroy = _libraries['libze_loader.so'].zeContextDestroy
    zeContextDestroy.restype = ze_result_t
    zeContextDestroy.argtypes = [ze_context_handle_t]
except AttributeError:
    pass
try:
    zeContextGetStatus = _libraries['libze_loader.so'].zeContextGetStatus
    zeContextGetStatus.restype = ze_result_t
    zeContextGetStatus.argtypes = [ze_context_handle_t]
except AttributeError:
    pass
ze_command_queue_flags_t = ctypes.c_uint32

# values for enumeration '_ze_command_queue_flag_t'
_ze_command_queue_flag_t__enumvalues = {
    1: 'ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY',
    2: 'ZE_COMMAND_QUEUE_FLAG_IN_ORDER',
    2147483647: 'ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32',
}
ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = 1
ZE_COMMAND_QUEUE_FLAG_IN_ORDER = 2
ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 2147483647
_ze_command_queue_flag_t = ctypes.c_uint32 # enum
ze_command_queue_flag_t = _ze_command_queue_flag_t
ze_command_queue_flag_t__enumvalues = _ze_command_queue_flag_t__enumvalues
try:
    zeCommandQueueCreate = _libraries['libze_loader.so'].zeCommandQueueCreate
    zeCommandQueueCreate.restype = ze_result_t
    zeCommandQueueCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_command_queue_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t))]
except AttributeError:
    pass
try:
    zeCommandQueueDestroy = _libraries['libze_loader.so'].zeCommandQueueDestroy
    zeCommandQueueDestroy.restype = ze_result_t
    zeCommandQueueDestroy.argtypes = [ze_command_queue_handle_t]
except AttributeError:
    pass
try:
    zeCommandQueueExecuteCommandLists = _libraries['libze_loader.so'].zeCommandQueueExecuteCommandLists
    zeCommandQueueExecuteCommandLists.restype = ze_result_t
    zeCommandQueueExecuteCommandLists.argtypes = [ze_command_queue_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t)), ze_fence_handle_t]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    zeCommandQueueSynchronize = _libraries['libze_loader.so'].zeCommandQueueSynchronize
    zeCommandQueueSynchronize.restype = ze_result_t
    zeCommandQueueSynchronize.argtypes = [ze_command_queue_handle_t, uint64_t]
except AttributeError:
    pass
ze_command_list_flags_t = ctypes.c_uint32

# values for enumeration '_ze_command_list_flag_t'
_ze_command_list_flag_t__enumvalues = {
    1: 'ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING',
    2: 'ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT',
    4: 'ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY',
    8: 'ZE_COMMAND_LIST_FLAG_IN_ORDER',
    2147483647: 'ZE_COMMAND_LIST_FLAG_FORCE_UINT32',
}
ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = 1
ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = 2
ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = 4
ZE_COMMAND_LIST_FLAG_IN_ORDER = 8
ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 2147483647
_ze_command_list_flag_t = ctypes.c_uint32 # enum
ze_command_list_flag_t = _ze_command_list_flag_t
ze_command_list_flag_t__enumvalues = _ze_command_list_flag_t__enumvalues
try:
    zeCommandListCreate = _libraries['libze_loader.so'].zeCommandListCreate
    zeCommandListCreate.restype = ze_result_t
    zeCommandListCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_command_list_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListCreateImmediate = _libraries['libze_loader.so'].zeCommandListCreateImmediate
    zeCommandListCreateImmediate.restype = ze_result_t
    zeCommandListCreateImmediate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_command_queue_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListDestroy = _libraries['libze_loader.so'].zeCommandListDestroy
    zeCommandListDestroy.restype = ze_result_t
    zeCommandListDestroy.argtypes = [ze_command_list_handle_t]
except AttributeError:
    pass
try:
    zeCommandListClose = _libraries['libze_loader.so'].zeCommandListClose
    zeCommandListClose.restype = ze_result_t
    zeCommandListClose.argtypes = [ze_command_list_handle_t]
except AttributeError:
    pass
try:
    zeCommandListReset = _libraries['libze_loader.so'].zeCommandListReset
    zeCommandListReset.restype = ze_result_t
    zeCommandListReset.argtypes = [ze_command_list_handle_t]
except AttributeError:
    pass
try:
    zeCommandListAppendWriteGlobalTimestamp = _libraries['libze_loader.so'].zeCommandListAppendWriteGlobalTimestamp
    zeCommandListAppendWriteGlobalTimestamp.restype = ze_result_t
    zeCommandListAppendWriteGlobalTimestamp.argtypes = [ze_command_list_handle_t, ctypes.POINTER(ctypes.c_uint64), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListHostSynchronize = _libraries['libze_loader.so'].zeCommandListHostSynchronize
    zeCommandListHostSynchronize.restype = ze_result_t
    zeCommandListHostSynchronize.argtypes = [ze_command_list_handle_t, uint64_t]
except AttributeError:
    pass
try:
    zeCommandListAppendBarrier = _libraries['libze_loader.so'].zeCommandListAppendBarrier
    zeCommandListAppendBarrier.restype = ze_result_t
    zeCommandListAppendBarrier.argtypes = [ze_command_list_handle_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendMemoryRangesBarrier = _libraries['libze_loader.so'].zeCommandListAppendMemoryRangesBarrier
    zeCommandListAppendMemoryRangesBarrier.restype = ze_result_t
    zeCommandListAppendMemoryRangesBarrier.argtypes = [ze_command_list_handle_t, uint32_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None)), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeContextSystemBarrier = _libraries['libze_loader.so'].zeContextSystemBarrier
    zeContextSystemBarrier.restype = ze_result_t
    zeContextSystemBarrier.argtypes = [ze_context_handle_t, ze_device_handle_t]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    zeCommandListAppendMemoryCopy = _libraries['libze_loader.so'].zeCommandListAppendMemoryCopy
    zeCommandListAppendMemoryCopy.restype = ze_result_t
    zeCommandListAppendMemoryCopy.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendMemoryFill = _libraries['libze_loader.so'].zeCommandListAppendMemoryFill
    zeCommandListAppendMemoryFill.restype = ze_result_t
    zeCommandListAppendMemoryFill.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendMemoryCopyRegion = _libraries['libze_loader.so'].zeCommandListAppendMemoryCopyRegion
    zeCommandListAppendMemoryCopyRegion.restype = ze_result_t
    zeCommandListAppendMemoryCopyRegion.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_copy_region_t), uint32_t, uint32_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_copy_region_t), uint32_t, uint32_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendMemoryCopyFromContext = _libraries['libze_loader.so'].zeCommandListAppendMemoryCopyFromContext
    zeCommandListAppendMemoryCopyFromContext.restype = ze_result_t
    zeCommandListAppendMemoryCopyFromContext.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ze_context_handle_t, ctypes.POINTER(None), size_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendImageCopy = _libraries['libze_loader.so'].zeCommandListAppendImageCopy
    zeCommandListAppendImageCopy.restype = ze_result_t
    zeCommandListAppendImageCopy.argtypes = [ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendImageCopyRegion = _libraries['libze_loader.so'].zeCommandListAppendImageCopyRegion
    zeCommandListAppendImageCopyRegion.restype = ze_result_t
    zeCommandListAppendImageCopyRegion.argtypes = [ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, ctypes.POINTER(struct__ze_image_region_t), ctypes.POINTER(struct__ze_image_region_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendImageCopyToMemory = _libraries['libze_loader.so'].zeCommandListAppendImageCopyToMemory
    zeCommandListAppendImageCopyToMemory.restype = ze_result_t
    zeCommandListAppendImageCopyToMemory.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ze_image_handle_t, ctypes.POINTER(struct__ze_image_region_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendImageCopyFromMemory = _libraries['libze_loader.so'].zeCommandListAppendImageCopyFromMemory
    zeCommandListAppendImageCopyFromMemory.restype = ze_result_t
    zeCommandListAppendImageCopyFromMemory.argtypes = [ze_command_list_handle_t, ze_image_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_image_region_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendMemoryPrefetch = _libraries['libze_loader.so'].zeCommandListAppendMemoryPrefetch
    zeCommandListAppendMemoryPrefetch.restype = ze_result_t
    zeCommandListAppendMemoryPrefetch.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass

# values for enumeration '_ze_memory_advice_t'
_ze_memory_advice_t__enumvalues = {
    0: 'ZE_MEMORY_ADVICE_SET_READ_MOSTLY',
    1: 'ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY',
    2: 'ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION',
    3: 'ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION',
    4: 'ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY',
    5: 'ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY',
    6: 'ZE_MEMORY_ADVICE_BIAS_CACHED',
    7: 'ZE_MEMORY_ADVICE_BIAS_UNCACHED',
    8: 'ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION',
    9: 'ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION',
    2147483647: 'ZE_MEMORY_ADVICE_FORCE_UINT32',
}
ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0
ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1
ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2
ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3
ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 4
ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 5
ZE_MEMORY_ADVICE_BIAS_CACHED = 6
ZE_MEMORY_ADVICE_BIAS_UNCACHED = 7
ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION = 8
ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION = 9
ZE_MEMORY_ADVICE_FORCE_UINT32 = 2147483647
_ze_memory_advice_t = ctypes.c_uint32 # enum
ze_memory_advice_t = _ze_memory_advice_t
ze_memory_advice_t__enumvalues = _ze_memory_advice_t__enumvalues
try:
    zeCommandListAppendMemAdvise = _libraries['libze_loader.so'].zeCommandListAppendMemAdvise
    zeCommandListAppendMemAdvise.restype = ze_result_t
    zeCommandListAppendMemAdvise.argtypes = [ze_command_list_handle_t, ze_device_handle_t, ctypes.POINTER(None), size_t, ze_memory_advice_t]
except AttributeError:
    pass
ze_event_pool_flags_t = ctypes.c_uint32

# values for enumeration '_ze_event_pool_flag_t'
_ze_event_pool_flag_t__enumvalues = {
    1: 'ZE_EVENT_POOL_FLAG_HOST_VISIBLE',
    2: 'ZE_EVENT_POOL_FLAG_IPC',
    4: 'ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP',
    8: 'ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP',
    2147483647: 'ZE_EVENT_POOL_FLAG_FORCE_UINT32',
}
ZE_EVENT_POOL_FLAG_HOST_VISIBLE = 1
ZE_EVENT_POOL_FLAG_IPC = 2
ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = 4
ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP = 8
ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 2147483647
_ze_event_pool_flag_t = ctypes.c_uint32 # enum
ze_event_pool_flag_t = _ze_event_pool_flag_t
ze_event_pool_flag_t__enumvalues = _ze_event_pool_flag_t__enumvalues
try:
    zeEventPoolCreate = _libraries['libze_loader.so'].zeEventPoolCreate
    zeEventPoolCreate.restype = ze_result_t
    zeEventPoolCreate.argtypes = [ze_context_handle_t, ctypes.POINTER(struct__ze_event_pool_desc_t), uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))]
except AttributeError:
    pass
try:
    zeEventPoolDestroy = _libraries['libze_loader.so'].zeEventPoolDestroy
    zeEventPoolDestroy.restype = ze_result_t
    zeEventPoolDestroy.argtypes = [ze_event_pool_handle_t]
except AttributeError:
    pass
ze_event_scope_flags_t = ctypes.c_uint32

# values for enumeration '_ze_event_scope_flag_t'
_ze_event_scope_flag_t__enumvalues = {
    1: 'ZE_EVENT_SCOPE_FLAG_SUBDEVICE',
    2: 'ZE_EVENT_SCOPE_FLAG_DEVICE',
    4: 'ZE_EVENT_SCOPE_FLAG_HOST',
    2147483647: 'ZE_EVENT_SCOPE_FLAG_FORCE_UINT32',
}
ZE_EVENT_SCOPE_FLAG_SUBDEVICE = 1
ZE_EVENT_SCOPE_FLAG_DEVICE = 2
ZE_EVENT_SCOPE_FLAG_HOST = 4
ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 2147483647
_ze_event_scope_flag_t = ctypes.c_uint32 # enum
ze_event_scope_flag_t = _ze_event_scope_flag_t
ze_event_scope_flag_t__enumvalues = _ze_event_scope_flag_t__enumvalues
try:
    zeEventCreate = _libraries['libze_loader.so'].zeEventCreate
    zeEventCreate.restype = ze_result_t
    zeEventCreate.argtypes = [ze_event_pool_handle_t, ctypes.POINTER(struct__ze_event_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeEventDestroy = _libraries['libze_loader.so'].zeEventDestroy
    zeEventDestroy.restype = ze_result_t
    zeEventDestroy.argtypes = [ze_event_handle_t]
except AttributeError:
    pass
try:
    zeEventPoolGetIpcHandle = _libraries['libze_loader.so'].zeEventPoolGetIpcHandle
    zeEventPoolGetIpcHandle.restype = ze_result_t
    zeEventPoolGetIpcHandle.argtypes = [ze_event_pool_handle_t, ctypes.POINTER(struct__ze_ipc_event_pool_handle_t)]
except AttributeError:
    pass
try:
    zeEventPoolPutIpcHandle = _libraries['libze_loader.so'].zeEventPoolPutIpcHandle
    zeEventPoolPutIpcHandle.restype = ze_result_t
    zeEventPoolPutIpcHandle.argtypes = [ze_context_handle_t, ze_ipc_event_pool_handle_t]
except AttributeError:
    pass
try:
    zeEventPoolOpenIpcHandle = _libraries['libze_loader.so'].zeEventPoolOpenIpcHandle
    zeEventPoolOpenIpcHandle.restype = ze_result_t
    zeEventPoolOpenIpcHandle.argtypes = [ze_context_handle_t, ze_ipc_event_pool_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))]
except AttributeError:
    pass
try:
    zeEventPoolCloseIpcHandle = _libraries['libze_loader.so'].zeEventPoolCloseIpcHandle
    zeEventPoolCloseIpcHandle.restype = ze_result_t
    zeEventPoolCloseIpcHandle.argtypes = [ze_event_pool_handle_t]
except AttributeError:
    pass
try:
    zeCommandListAppendSignalEvent = _libraries['libze_loader.so'].zeCommandListAppendSignalEvent
    zeCommandListAppendSignalEvent.restype = ze_result_t
    zeCommandListAppendSignalEvent.argtypes = [ze_command_list_handle_t, ze_event_handle_t]
except AttributeError:
    pass
try:
    zeCommandListAppendWaitOnEvents = _libraries['libze_loader.so'].zeCommandListAppendWaitOnEvents
    zeCommandListAppendWaitOnEvents.restype = ze_result_t
    zeCommandListAppendWaitOnEvents.argtypes = [ze_command_list_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeEventHostSignal = _libraries['libze_loader.so'].zeEventHostSignal
    zeEventHostSignal.restype = ze_result_t
    zeEventHostSignal.argtypes = [ze_event_handle_t]
except AttributeError:
    pass
try:
    zeEventHostSynchronize = _libraries['libze_loader.so'].zeEventHostSynchronize
    zeEventHostSynchronize.restype = ze_result_t
    zeEventHostSynchronize.argtypes = [ze_event_handle_t, uint64_t]
except AttributeError:
    pass
try:
    zeEventQueryStatus = _libraries['libze_loader.so'].zeEventQueryStatus
    zeEventQueryStatus.restype = ze_result_t
    zeEventQueryStatus.argtypes = [ze_event_handle_t]
except AttributeError:
    pass
try:
    zeCommandListAppendEventReset = _libraries['libze_loader.so'].zeCommandListAppendEventReset
    zeCommandListAppendEventReset.restype = ze_result_t
    zeCommandListAppendEventReset.argtypes = [ze_command_list_handle_t, ze_event_handle_t]
except AttributeError:
    pass
try:
    zeEventHostReset = _libraries['libze_loader.so'].zeEventHostReset
    zeEventHostReset.restype = ze_result_t
    zeEventHostReset.argtypes = [ze_event_handle_t]
except AttributeError:
    pass
try:
    zeEventQueryKernelTimestamp = _libraries['libze_loader.so'].zeEventQueryKernelTimestamp
    zeEventQueryKernelTimestamp.restype = ze_result_t
    zeEventQueryKernelTimestamp.argtypes = [ze_event_handle_t, ctypes.POINTER(struct__ze_kernel_timestamp_result_t)]
except AttributeError:
    pass
try:
    zeCommandListAppendQueryKernelTimestamps = _libraries['libze_loader.so'].zeCommandListAppendQueryKernelTimestamps
    zeCommandListAppendQueryKernelTimestamps.restype = ze_result_t
    zeCommandListAppendQueryKernelTimestamps.argtypes = [ze_command_list_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
ze_fence_flags_t = ctypes.c_uint32

# values for enumeration '_ze_fence_flag_t'
_ze_fence_flag_t__enumvalues = {
    1: 'ZE_FENCE_FLAG_SIGNALED',
    2147483647: 'ZE_FENCE_FLAG_FORCE_UINT32',
}
ZE_FENCE_FLAG_SIGNALED = 1
ZE_FENCE_FLAG_FORCE_UINT32 = 2147483647
_ze_fence_flag_t = ctypes.c_uint32 # enum
ze_fence_flag_t = _ze_fence_flag_t
ze_fence_flag_t__enumvalues = _ze_fence_flag_t__enumvalues
try:
    zeFenceCreate = _libraries['libze_loader.so'].zeFenceCreate
    zeFenceCreate.restype = ze_result_t
    zeFenceCreate.argtypes = [ze_command_queue_handle_t, ctypes.POINTER(struct__ze_fence_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))]
except AttributeError:
    pass
try:
    zeFenceDestroy = _libraries['libze_loader.so'].zeFenceDestroy
    zeFenceDestroy.restype = ze_result_t
    zeFenceDestroy.argtypes = [ze_fence_handle_t]
except AttributeError:
    pass
try:
    zeFenceHostSynchronize = _libraries['libze_loader.so'].zeFenceHostSynchronize
    zeFenceHostSynchronize.restype = ze_result_t
    zeFenceHostSynchronize.argtypes = [ze_fence_handle_t, uint64_t]
except AttributeError:
    pass
try:
    zeFenceQueryStatus = _libraries['libze_loader.so'].zeFenceQueryStatus
    zeFenceQueryStatus.restype = ze_result_t
    zeFenceQueryStatus.argtypes = [ze_fence_handle_t]
except AttributeError:
    pass
try:
    zeFenceReset = _libraries['libze_loader.so'].zeFenceReset
    zeFenceReset.restype = ze_result_t
    zeFenceReset.argtypes = [ze_fence_handle_t]
except AttributeError:
    pass
ze_image_flags_t = ctypes.c_uint32

# values for enumeration '_ze_image_flag_t'
_ze_image_flag_t__enumvalues = {
    1: 'ZE_IMAGE_FLAG_KERNEL_WRITE',
    2: 'ZE_IMAGE_FLAG_BIAS_UNCACHED',
    2147483647: 'ZE_IMAGE_FLAG_FORCE_UINT32',
}
ZE_IMAGE_FLAG_KERNEL_WRITE = 1
ZE_IMAGE_FLAG_BIAS_UNCACHED = 2
ZE_IMAGE_FLAG_FORCE_UINT32 = 2147483647
_ze_image_flag_t = ctypes.c_uint32 # enum
ze_image_flag_t = _ze_image_flag_t
ze_image_flag_t__enumvalues = _ze_image_flag_t__enumvalues
ze_image_sampler_filter_flags_t = ctypes.c_uint32

# values for enumeration '_ze_image_sampler_filter_flag_t'
_ze_image_sampler_filter_flag_t__enumvalues = {
    1: 'ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT',
    2: 'ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR',
    2147483647: 'ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32',
}
ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT = 1
ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR = 2
ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32 = 2147483647
_ze_image_sampler_filter_flag_t = ctypes.c_uint32 # enum
ze_image_sampler_filter_flag_t = _ze_image_sampler_filter_flag_t
ze_image_sampler_filter_flag_t__enumvalues = _ze_image_sampler_filter_flag_t__enumvalues
try:
    zeImageGetProperties = _libraries['libze_loader.so'].zeImageGetProperties
    zeImageGetProperties.restype = ze_result_t
    zeImageGetProperties.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_image_desc_t), ctypes.POINTER(struct__ze_image_properties_t)]
except AttributeError:
    pass
try:
    zeImageCreate = _libraries['libze_loader.so'].zeImageCreate
    zeImageCreate.restype = ze_result_t
    zeImageCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_image_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))]
except AttributeError:
    pass
try:
    zeImageDestroy = _libraries['libze_loader.so'].zeImageDestroy
    zeImageDestroy.restype = ze_result_t
    zeImageDestroy.argtypes = [ze_image_handle_t]
except AttributeError:
    pass
ze_device_mem_alloc_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_mem_alloc_flag_t'
_ze_device_mem_alloc_flag_t__enumvalues = {
    1: 'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED',
    2: 'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED',
    4: 'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT',
    2147483647: 'ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32',
}
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = 1
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = 4
ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
_ze_device_mem_alloc_flag_t = ctypes.c_uint32 # enum
ze_device_mem_alloc_flag_t = _ze_device_mem_alloc_flag_t
ze_device_mem_alloc_flag_t__enumvalues = _ze_device_mem_alloc_flag_t__enumvalues
ze_host_mem_alloc_flags_t = ctypes.c_uint32

# values for enumeration '_ze_host_mem_alloc_flag_t'
_ze_host_mem_alloc_flag_t__enumvalues = {
    1: 'ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED',
    2: 'ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED',
    4: 'ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED',
    8: 'ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT',
    2147483647: 'ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32',
}
ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = 1
ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = 4
ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = 8
ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
_ze_host_mem_alloc_flag_t = ctypes.c_uint32 # enum
ze_host_mem_alloc_flag_t = _ze_host_mem_alloc_flag_t
ze_host_mem_alloc_flag_t__enumvalues = _ze_host_mem_alloc_flag_t__enumvalues
try:
    zeMemAllocShared = _libraries['libze_loader.so'].zeMemAllocShared
    zeMemAllocShared.restype = ze_result_t
    zeMemAllocShared.argtypes = [ze_context_handle_t, ctypes.POINTER(struct__ze_device_mem_alloc_desc_t), ctypes.POINTER(struct__ze_host_mem_alloc_desc_t), size_t, size_t, ze_device_handle_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeMemAllocDevice = _libraries['libze_loader.so'].zeMemAllocDevice
    zeMemAllocDevice.restype = ze_result_t
    zeMemAllocDevice.argtypes = [ze_context_handle_t, ctypes.POINTER(struct__ze_device_mem_alloc_desc_t), size_t, size_t, ze_device_handle_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeMemAllocHost = _libraries['libze_loader.so'].zeMemAllocHost
    zeMemAllocHost.restype = ze_result_t
    zeMemAllocHost.argtypes = [ze_context_handle_t, ctypes.POINTER(struct__ze_host_mem_alloc_desc_t), size_t, size_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeMemFree = _libraries['libze_loader.so'].zeMemFree
    zeMemFree.restype = ze_result_t
    zeMemFree.argtypes = [ze_context_handle_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    zeMemGetAllocProperties = _libraries['libze_loader.so'].zeMemGetAllocProperties
    zeMemGetAllocProperties.restype = ze_result_t
    zeMemGetAllocProperties.argtypes = [ze_context_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_memory_allocation_properties_t), ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))]
except AttributeError:
    pass
try:
    zeMemGetAddressRange = _libraries['libze_loader.so'].zeMemGetAddressRange
    zeMemGetAddressRange.restype = ze_result_t
    zeMemGetAddressRange.argtypes = [ze_context_handle_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    zeMemGetIpcHandle = _libraries['libze_loader.so'].zeMemGetIpcHandle
    zeMemGetIpcHandle.restype = ze_result_t
    zeMemGetIpcHandle.argtypes = [ze_context_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_ipc_mem_handle_t)]
except AttributeError:
    pass
try:
    zeMemGetIpcHandleFromFileDescriptorExp = _libraries['libze_loader.so'].zeMemGetIpcHandleFromFileDescriptorExp
    zeMemGetIpcHandleFromFileDescriptorExp.restype = ze_result_t
    zeMemGetIpcHandleFromFileDescriptorExp.argtypes = [ze_context_handle_t, uint64_t, ctypes.POINTER(struct__ze_ipc_mem_handle_t)]
except AttributeError:
    pass
try:
    zeMemGetFileDescriptorFromIpcHandleExp = _libraries['libze_loader.so'].zeMemGetFileDescriptorFromIpcHandleExp
    zeMemGetFileDescriptorFromIpcHandleExp.restype = ze_result_t
    zeMemGetFileDescriptorFromIpcHandleExp.argtypes = [ze_context_handle_t, ze_ipc_mem_handle_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    zeMemPutIpcHandle = _libraries['libze_loader.so'].zeMemPutIpcHandle
    zeMemPutIpcHandle.restype = ze_result_t
    zeMemPutIpcHandle.argtypes = [ze_context_handle_t, ze_ipc_mem_handle_t]
except AttributeError:
    pass
ze_ipc_memory_flags_t = ctypes.c_uint32

# values for enumeration '_ze_ipc_memory_flag_t'
_ze_ipc_memory_flag_t__enumvalues = {
    1: 'ZE_IPC_MEMORY_FLAG_BIAS_CACHED',
    2: 'ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED',
    2147483647: 'ZE_IPC_MEMORY_FLAG_FORCE_UINT32',
}
ZE_IPC_MEMORY_FLAG_BIAS_CACHED = 1
ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED = 2
ZE_IPC_MEMORY_FLAG_FORCE_UINT32 = 2147483647
_ze_ipc_memory_flag_t = ctypes.c_uint32 # enum
ze_ipc_memory_flag_t = _ze_ipc_memory_flag_t
ze_ipc_memory_flag_t__enumvalues = _ze_ipc_memory_flag_t__enumvalues
try:
    zeMemOpenIpcHandle = _libraries['libze_loader.so'].zeMemOpenIpcHandle
    zeMemOpenIpcHandle.restype = ze_result_t
    zeMemOpenIpcHandle.argtypes = [ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t, ze_ipc_memory_flags_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeMemCloseIpcHandle = _libraries['libze_loader.so'].zeMemCloseIpcHandle
    zeMemCloseIpcHandle.restype = ze_result_t
    zeMemCloseIpcHandle.argtypes = [ze_context_handle_t, ctypes.POINTER(None)]
except AttributeError:
    pass
ze_memory_atomic_attr_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_memory_atomic_attr_exp_flag_t'
_ze_memory_atomic_attr_exp_flag_t__enumvalues = {
    1: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_ATOMICS',
    2: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_HOST_ATOMICS',
    4: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_HOST_ATOMICS',
    8: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_DEVICE_ATOMICS',
    16: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_DEVICE_ATOMICS',
    32: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_SYSTEM_ATOMICS',
    64: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_SYSTEM_ATOMICS',
    2147483647: 'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_FORCE_UINT32',
}
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_ATOMICS = 1
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_HOST_ATOMICS = 2
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_HOST_ATOMICS = 4
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_DEVICE_ATOMICS = 8
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_DEVICE_ATOMICS = 16
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_SYSTEM_ATOMICS = 32
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_SYSTEM_ATOMICS = 64
ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_memory_atomic_attr_exp_flag_t = ctypes.c_uint32 # enum
ze_memory_atomic_attr_exp_flag_t = _ze_memory_atomic_attr_exp_flag_t
ze_memory_atomic_attr_exp_flag_t__enumvalues = _ze_memory_atomic_attr_exp_flag_t__enumvalues
try:
    zeMemSetAtomicAccessAttributeExp = _libraries['libze_loader.so'].zeMemSetAtomicAccessAttributeExp
    zeMemSetAtomicAccessAttributeExp.restype = ze_result_t
    zeMemSetAtomicAccessAttributeExp.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(None), size_t, ze_memory_atomic_attr_exp_flags_t]
except AttributeError:
    pass
try:
    zeMemGetAtomicAccessAttributeExp = _libraries['libze_loader.so'].zeMemGetAtomicAccessAttributeExp
    zeMemGetAtomicAccessAttributeExp.restype = ze_result_t
    zeMemGetAtomicAccessAttributeExp.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    zeModuleCreate = _libraries['libze_loader.so'].zeModuleCreate
    zeModuleCreate.restype = ze_result_t
    zeModuleCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_module_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t))]
except AttributeError:
    pass
try:
    zeModuleDestroy = _libraries['libze_loader.so'].zeModuleDestroy
    zeModuleDestroy.restype = ze_result_t
    zeModuleDestroy.argtypes = [ze_module_handle_t]
except AttributeError:
    pass
try:
    zeModuleDynamicLink = _libraries['libze_loader.so'].zeModuleDynamicLink
    zeModuleDynamicLink.restype = ze_result_t
    zeModuleDynamicLink.argtypes = [uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t))]
except AttributeError:
    pass
try:
    zeModuleBuildLogDestroy = _libraries['libze_loader.so'].zeModuleBuildLogDestroy
    zeModuleBuildLogDestroy.restype = ze_result_t
    zeModuleBuildLogDestroy.argtypes = [ze_module_build_log_handle_t]
except AttributeError:
    pass
try:
    zeModuleBuildLogGetString = _libraries['libze_loader.so'].zeModuleBuildLogGetString
    zeModuleBuildLogGetString.restype = ze_result_t
    zeModuleBuildLogGetString.argtypes = [ze_module_build_log_handle_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    zeModuleGetNativeBinary = _libraries['libze_loader.so'].zeModuleGetNativeBinary
    zeModuleGetNativeBinary.restype = ze_result_t
    zeModuleGetNativeBinary.argtypes = [ze_module_handle_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    zeModuleGetGlobalPointer = _libraries['libze_loader.so'].zeModuleGetGlobalPointer
    zeModuleGetGlobalPointer.restype = ze_result_t
    zeModuleGetGlobalPointer.argtypes = [ze_module_handle_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeModuleGetKernelNames = _libraries['libze_loader.so'].zeModuleGetKernelNames
    zeModuleGetKernelNames.restype = ze_result_t
    zeModuleGetKernelNames.argtypes = [ze_module_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
ze_module_property_flags_t = ctypes.c_uint32

# values for enumeration '_ze_module_property_flag_t'
_ze_module_property_flag_t__enumvalues = {
    1: 'ZE_MODULE_PROPERTY_FLAG_IMPORTS',
    2147483647: 'ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32',
}
ZE_MODULE_PROPERTY_FLAG_IMPORTS = 1
ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
_ze_module_property_flag_t = ctypes.c_uint32 # enum
ze_module_property_flag_t = _ze_module_property_flag_t
ze_module_property_flag_t__enumvalues = _ze_module_property_flag_t__enumvalues
try:
    zeModuleGetProperties = _libraries['libze_loader.so'].zeModuleGetProperties
    zeModuleGetProperties.restype = ze_result_t
    zeModuleGetProperties.argtypes = [ze_module_handle_t, ctypes.POINTER(struct__ze_module_properties_t)]
except AttributeError:
    pass
ze_kernel_flags_t = ctypes.c_uint32

# values for enumeration '_ze_kernel_flag_t'
_ze_kernel_flag_t__enumvalues = {
    1: 'ZE_KERNEL_FLAG_FORCE_RESIDENCY',
    2: 'ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY',
    2147483647: 'ZE_KERNEL_FLAG_FORCE_UINT32',
}
ZE_KERNEL_FLAG_FORCE_RESIDENCY = 1
ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY = 2
ZE_KERNEL_FLAG_FORCE_UINT32 = 2147483647
_ze_kernel_flag_t = ctypes.c_uint32 # enum
ze_kernel_flag_t = _ze_kernel_flag_t
ze_kernel_flag_t__enumvalues = _ze_kernel_flag_t__enumvalues
try:
    zeKernelCreate = _libraries['libze_loader.so'].zeKernelCreate
    zeKernelCreate.restype = ze_result_t
    zeKernelCreate.argtypes = [ze_module_handle_t, ctypes.POINTER(struct__ze_kernel_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))]
except AttributeError:
    pass
try:
    zeKernelDestroy = _libraries['libze_loader.so'].zeKernelDestroy
    zeKernelDestroy.restype = ze_result_t
    zeKernelDestroy.argtypes = [ze_kernel_handle_t]
except AttributeError:
    pass
try:
    zeModuleGetFunctionPointer = _libraries['libze_loader.so'].zeModuleGetFunctionPointer
    zeModuleGetFunctionPointer.restype = ze_result_t
    zeModuleGetFunctionPointer.argtypes = [ze_module_handle_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeKernelSetGroupSize = _libraries['libze_loader.so'].zeKernelSetGroupSize
    zeKernelSetGroupSize.restype = ze_result_t
    zeKernelSetGroupSize.argtypes = [ze_kernel_handle_t, uint32_t, uint32_t, uint32_t]
except AttributeError:
    pass
try:
    zeKernelSuggestGroupSize = _libraries['libze_loader.so'].zeKernelSuggestGroupSize
    zeKernelSuggestGroupSize.restype = ze_result_t
    zeKernelSuggestGroupSize.argtypes = [ze_kernel_handle_t, uint32_t, uint32_t, uint32_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    zeKernelSuggestMaxCooperativeGroupCount = _libraries['libze_loader.so'].zeKernelSuggestMaxCooperativeGroupCount
    zeKernelSuggestMaxCooperativeGroupCount.restype = ze_result_t
    zeKernelSuggestMaxCooperativeGroupCount.argtypes = [ze_kernel_handle_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    zeKernelSetArgumentValue = _libraries['libze_loader.so'].zeKernelSetArgumentValue
    zeKernelSetArgumentValue.restype = ze_result_t
    zeKernelSetArgumentValue.argtypes = [ze_kernel_handle_t, uint32_t, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
ze_kernel_indirect_access_flags_t = ctypes.c_uint32

# values for enumeration '_ze_kernel_indirect_access_flag_t'
_ze_kernel_indirect_access_flag_t__enumvalues = {
    1: 'ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST',
    2: 'ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE',
    4: 'ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED',
    2147483647: 'ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32',
}
ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = 1
ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = 2
ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = 4
ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 2147483647
_ze_kernel_indirect_access_flag_t = ctypes.c_uint32 # enum
ze_kernel_indirect_access_flag_t = _ze_kernel_indirect_access_flag_t
ze_kernel_indirect_access_flag_t__enumvalues = _ze_kernel_indirect_access_flag_t__enumvalues
try:
    zeKernelSetIndirectAccess = _libraries['libze_loader.so'].zeKernelSetIndirectAccess
    zeKernelSetIndirectAccess.restype = ze_result_t
    zeKernelSetIndirectAccess.argtypes = [ze_kernel_handle_t, ze_kernel_indirect_access_flags_t]
except AttributeError:
    pass
try:
    zeKernelGetIndirectAccess = _libraries['libze_loader.so'].zeKernelGetIndirectAccess
    zeKernelGetIndirectAccess.restype = ze_result_t
    zeKernelGetIndirectAccess.argtypes = [ze_kernel_handle_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    zeKernelGetSourceAttributes = _libraries['libze_loader.so'].zeKernelGetSourceAttributes
    zeKernelGetSourceAttributes.restype = ze_result_t
    zeKernelGetSourceAttributes.argtypes = [ze_kernel_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
ze_cache_config_flags_t = ctypes.c_uint32

# values for enumeration '_ze_cache_config_flag_t'
_ze_cache_config_flag_t__enumvalues = {
    1: 'ZE_CACHE_CONFIG_FLAG_LARGE_SLM',
    2: 'ZE_CACHE_CONFIG_FLAG_LARGE_DATA',
    2147483647: 'ZE_CACHE_CONFIG_FLAG_FORCE_UINT32',
}
ZE_CACHE_CONFIG_FLAG_LARGE_SLM = 1
ZE_CACHE_CONFIG_FLAG_LARGE_DATA = 2
ZE_CACHE_CONFIG_FLAG_FORCE_UINT32 = 2147483647
_ze_cache_config_flag_t = ctypes.c_uint32 # enum
ze_cache_config_flag_t = _ze_cache_config_flag_t
ze_cache_config_flag_t__enumvalues = _ze_cache_config_flag_t__enumvalues
try:
    zeKernelSetCacheConfig = _libraries['libze_loader.so'].zeKernelSetCacheConfig
    zeKernelSetCacheConfig.restype = ze_result_t
    zeKernelSetCacheConfig.argtypes = [ze_kernel_handle_t, ze_cache_config_flags_t]
except AttributeError:
    pass
try:
    zeKernelGetProperties = _libraries['libze_loader.so'].zeKernelGetProperties
    zeKernelGetProperties.restype = ze_result_t
    zeKernelGetProperties.argtypes = [ze_kernel_handle_t, ctypes.POINTER(struct__ze_kernel_properties_t)]
except AttributeError:
    pass
try:
    zeKernelGetName = _libraries['libze_loader.so'].zeKernelGetName
    zeKernelGetName.restype = ze_result_t
    zeKernelGetName.argtypes = [ze_kernel_handle_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    zeCommandListAppendLaunchKernel = _libraries['libze_loader.so'].zeCommandListAppendLaunchKernel
    zeCommandListAppendLaunchKernel.restype = ze_result_t
    zeCommandListAppendLaunchKernel.argtypes = [ze_command_list_handle_t, ze_kernel_handle_t, ctypes.POINTER(struct__ze_group_count_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendLaunchCooperativeKernel = _libraries['libze_loader.so'].zeCommandListAppendLaunchCooperativeKernel
    zeCommandListAppendLaunchCooperativeKernel.restype = ze_result_t
    zeCommandListAppendLaunchCooperativeKernel.argtypes = [ze_command_list_handle_t, ze_kernel_handle_t, ctypes.POINTER(struct__ze_group_count_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendLaunchKernelIndirect = _libraries['libze_loader.so'].zeCommandListAppendLaunchKernelIndirect
    zeCommandListAppendLaunchKernelIndirect.restype = ze_result_t
    zeCommandListAppendLaunchKernelIndirect.argtypes = [ze_command_list_handle_t, ze_kernel_handle_t, ctypes.POINTER(struct__ze_group_count_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendLaunchMultipleKernelsIndirect = _libraries['libze_loader.so'].zeCommandListAppendLaunchMultipleKernelsIndirect
    zeCommandListAppendLaunchMultipleKernelsIndirect.restype = ze_result_t
    zeCommandListAppendLaunchMultipleKernelsIndirect.argtypes = [ze_command_list_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_group_count_t), ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_module_program_exp_version_t'
_ze_module_program_exp_version_t__enumvalues = {
    65536: 'ZE_MODULE_PROGRAM_EXP_VERSION_1_0',
    65536: 'ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT',
    2147483647: 'ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32',
}
ZE_MODULE_PROGRAM_EXP_VERSION_1_0 = 65536
ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT = 65536
ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_module_program_exp_version_t = ctypes.c_uint32 # enum
ze_module_program_exp_version_t = _ze_module_program_exp_version_t
ze_module_program_exp_version_t__enumvalues = _ze_module_program_exp_version_t__enumvalues

# values for enumeration '_ze_raytracing_ext_version_t'
_ze_raytracing_ext_version_t__enumvalues = {
    65536: 'ZE_RAYTRACING_EXT_VERSION_1_0',
    65536: 'ZE_RAYTRACING_EXT_VERSION_CURRENT',
    2147483647: 'ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32',
}
ZE_RAYTRACING_EXT_VERSION_1_0 = 65536
ZE_RAYTRACING_EXT_VERSION_CURRENT = 65536
ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_raytracing_ext_version_t = ctypes.c_uint32 # enum
ze_raytracing_ext_version_t = _ze_raytracing_ext_version_t
ze_raytracing_ext_version_t__enumvalues = _ze_raytracing_ext_version_t__enumvalues
ze_device_raytracing_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_raytracing_ext_flag_t'
_ze_device_raytracing_ext_flag_t__enumvalues = {
    1: 'ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY',
    2147483647: 'ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32',
}
ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY = 1
ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_device_raytracing_ext_flag_t = ctypes.c_uint32 # enum
ze_device_raytracing_ext_flag_t = _ze_device_raytracing_ext_flag_t
ze_device_raytracing_ext_flag_t__enumvalues = _ze_device_raytracing_ext_flag_t__enumvalues
ze_raytracing_mem_alloc_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_raytracing_mem_alloc_ext_flag_t'
_ze_raytracing_mem_alloc_ext_flag_t__enumvalues = {
    1: 'ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD',
    2147483647: 'ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32',
}
ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD = 1
ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_raytracing_mem_alloc_ext_flag_t = ctypes.c_uint32 # enum
ze_raytracing_mem_alloc_ext_flag_t = _ze_raytracing_mem_alloc_ext_flag_t
ze_raytracing_mem_alloc_ext_flag_t__enumvalues = _ze_raytracing_mem_alloc_ext_flag_t__enumvalues
try:
    zeContextMakeMemoryResident = _libraries['libze_loader.so'].zeContextMakeMemoryResident
    zeContextMakeMemoryResident.restype = ze_result_t
    zeContextMakeMemoryResident.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    zeContextEvictMemory = _libraries['libze_loader.so'].zeContextEvictMemory
    zeContextEvictMemory.restype = ze_result_t
    zeContextEvictMemory.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    zeContextMakeImageResident = _libraries['libze_loader.so'].zeContextMakeImageResident
    zeContextMakeImageResident.restype = ze_result_t
    zeContextMakeImageResident.argtypes = [ze_context_handle_t, ze_device_handle_t, ze_image_handle_t]
except AttributeError:
    pass
try:
    zeContextEvictImage = _libraries['libze_loader.so'].zeContextEvictImage
    zeContextEvictImage.restype = ze_result_t
    zeContextEvictImage.argtypes = [ze_context_handle_t, ze_device_handle_t, ze_image_handle_t]
except AttributeError:
    pass
try:
    zeSamplerCreate = _libraries['libze_loader.so'].zeSamplerCreate
    zeSamplerCreate.restype = ze_result_t
    zeSamplerCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_sampler_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_sampler_handle_t))]
except AttributeError:
    pass
try:
    zeSamplerDestroy = _libraries['libze_loader.so'].zeSamplerDestroy
    zeSamplerDestroy.restype = ze_result_t
    zeSamplerDestroy.argtypes = [ze_sampler_handle_t]
except AttributeError:
    pass

# values for enumeration '_ze_memory_access_attribute_t'
_ze_memory_access_attribute_t__enumvalues = {
    0: 'ZE_MEMORY_ACCESS_ATTRIBUTE_NONE',
    1: 'ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE',
    2: 'ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY',
    2147483647: 'ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32',
}
ZE_MEMORY_ACCESS_ATTRIBUTE_NONE = 0
ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1
ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY = 2
ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32 = 2147483647
_ze_memory_access_attribute_t = ctypes.c_uint32 # enum
ze_memory_access_attribute_t = _ze_memory_access_attribute_t
ze_memory_access_attribute_t__enumvalues = _ze_memory_access_attribute_t__enumvalues
try:
    zeVirtualMemReserve = _libraries['libze_loader.so'].zeVirtualMemReserve
    zeVirtualMemReserve.restype = ze_result_t
    zeVirtualMemReserve.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zeVirtualMemFree = _libraries['libze_loader.so'].zeVirtualMemFree
    zeVirtualMemFree.restype = ze_result_t
    zeVirtualMemFree.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    zeVirtualMemQueryPageSize = _libraries['libze_loader.so'].zeVirtualMemQueryPageSize
    zeVirtualMemQueryPageSize.restype = ze_result_t
    zeVirtualMemQueryPageSize.argtypes = [ze_context_handle_t, ze_device_handle_t, size_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
ze_physical_mem_flags_t = ctypes.c_uint32

# values for enumeration '_ze_physical_mem_flag_t'
_ze_physical_mem_flag_t__enumvalues = {
    1: 'ZE_PHYSICAL_MEM_FLAG_TBD',
    2147483647: 'ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32',
}
ZE_PHYSICAL_MEM_FLAG_TBD = 1
ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 2147483647
_ze_physical_mem_flag_t = ctypes.c_uint32 # enum
ze_physical_mem_flag_t = _ze_physical_mem_flag_t
ze_physical_mem_flag_t__enumvalues = _ze_physical_mem_flag_t__enumvalues
try:
    zePhysicalMemCreate = _libraries['libze_loader.so'].zePhysicalMemCreate
    zePhysicalMemCreate.restype = ze_result_t
    zePhysicalMemCreate.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_physical_mem_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_physical_mem_handle_t))]
except AttributeError:
    pass
try:
    zePhysicalMemDestroy = _libraries['libze_loader.so'].zePhysicalMemDestroy
    zePhysicalMemDestroy.restype = ze_result_t
    zePhysicalMemDestroy.argtypes = [ze_context_handle_t, ze_physical_mem_handle_t]
except AttributeError:
    pass
try:
    zeVirtualMemMap = _libraries['libze_loader.so'].zeVirtualMemMap
    zeVirtualMemMap.restype = ze_result_t
    zeVirtualMemMap.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t, ze_physical_mem_handle_t, size_t, ze_memory_access_attribute_t]
except AttributeError:
    pass
try:
    zeVirtualMemUnmap = _libraries['libze_loader.so'].zeVirtualMemUnmap
    zeVirtualMemUnmap.restype = ze_result_t
    zeVirtualMemUnmap.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    zeVirtualMemSetAccessAttribute = _libraries['libze_loader.so'].zeVirtualMemSetAccessAttribute
    zeVirtualMemSetAccessAttribute.restype = ze_result_t
    zeVirtualMemSetAccessAttribute.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t, ze_memory_access_attribute_t]
except AttributeError:
    pass
try:
    zeVirtualMemGetAccessAttribute = _libraries['libze_loader.so'].zeVirtualMemGetAccessAttribute
    zeVirtualMemGetAccessAttribute.restype = ze_result_t
    zeVirtualMemGetAccessAttribute.argtypes = [ze_context_handle_t, ctypes.POINTER(None), size_t, ctypes.POINTER(_ze_memory_access_attribute_t), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass

# values for enumeration '_ze_float_atomics_ext_version_t'
_ze_float_atomics_ext_version_t__enumvalues = {
    65536: 'ZE_FLOAT_ATOMICS_EXT_VERSION_1_0',
    65536: 'ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT',
    2147483647: 'ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32',
}
ZE_FLOAT_ATOMICS_EXT_VERSION_1_0 = 65536
ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT = 65536
ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_float_atomics_ext_version_t = ctypes.c_uint32 # enum
ze_float_atomics_ext_version_t = _ze_float_atomics_ext_version_t
ze_float_atomics_ext_version_t__enumvalues = _ze_float_atomics_ext_version_t__enumvalues
ze_device_fp_atomic_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_device_fp_atomic_ext_flag_t'
_ze_device_fp_atomic_ext_flag_t__enumvalues = {
    1: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE',
    2: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD',
    4: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX',
    65536: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE',
    131072: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD',
    262144: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX',
    2147483647: 'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32',
}
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE = 1
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD = 2
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX = 4
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE = 65536
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD = 131072
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX = 262144
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_device_fp_atomic_ext_flag_t = ctypes.c_uint32 # enum
ze_device_fp_atomic_ext_flag_t = _ze_device_fp_atomic_ext_flag_t
ze_device_fp_atomic_ext_flag_t__enumvalues = _ze_device_fp_atomic_ext_flag_t__enumvalues

# values for enumeration '_ze_global_offset_exp_version_t'
_ze_global_offset_exp_version_t__enumvalues = {
    65536: 'ZE_GLOBAL_OFFSET_EXP_VERSION_1_0',
    65536: 'ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT',
    2147483647: 'ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32',
}
ZE_GLOBAL_OFFSET_EXP_VERSION_1_0 = 65536
ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT = 65536
ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_global_offset_exp_version_t = ctypes.c_uint32 # enum
ze_global_offset_exp_version_t = _ze_global_offset_exp_version_t
ze_global_offset_exp_version_t__enumvalues = _ze_global_offset_exp_version_t__enumvalues
try:
    zeKernelSetGlobalOffsetExp = _libraries['libze_loader.so'].zeKernelSetGlobalOffsetExp
    zeKernelSetGlobalOffsetExp.restype = ze_result_t
    zeKernelSetGlobalOffsetExp.argtypes = [ze_kernel_handle_t, uint32_t, uint32_t, uint32_t]
except AttributeError:
    pass

# values for enumeration '_ze_relaxed_allocation_limits_exp_version_t'
_ze_relaxed_allocation_limits_exp_version_t__enumvalues = {
    65536: 'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0',
    65536: 'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT',
    2147483647: 'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32',
}
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0 = 65536
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT = 65536
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_relaxed_allocation_limits_exp_version_t = ctypes.c_uint32 # enum
ze_relaxed_allocation_limits_exp_version_t = _ze_relaxed_allocation_limits_exp_version_t
ze_relaxed_allocation_limits_exp_version_t__enumvalues = _ze_relaxed_allocation_limits_exp_version_t__enumvalues
ze_relaxed_allocation_limits_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_relaxed_allocation_limits_exp_flag_t'
_ze_relaxed_allocation_limits_exp_flag_t__enumvalues = {
    1: 'ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE',
    2147483647: 'ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32',
}
ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = 1
ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_relaxed_allocation_limits_exp_flag_t = ctypes.c_uint32 # enum
ze_relaxed_allocation_limits_exp_flag_t = _ze_relaxed_allocation_limits_exp_flag_t
ze_relaxed_allocation_limits_exp_flag_t__enumvalues = _ze_relaxed_allocation_limits_exp_flag_t__enumvalues

# values for enumeration '_ze_cache_reservation_ext_version_t'
_ze_cache_reservation_ext_version_t__enumvalues = {
    65536: 'ZE_CACHE_RESERVATION_EXT_VERSION_1_0',
    65536: 'ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT',
    2147483647: 'ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32',
}
ZE_CACHE_RESERVATION_EXT_VERSION_1_0 = 65536
ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT = 65536
ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_cache_reservation_ext_version_t = ctypes.c_uint32 # enum
ze_cache_reservation_ext_version_t = _ze_cache_reservation_ext_version_t
ze_cache_reservation_ext_version_t__enumvalues = _ze_cache_reservation_ext_version_t__enumvalues

# values for enumeration '_ze_cache_ext_region_t'
_ze_cache_ext_region_t__enumvalues = {
    0: 'ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT',
    1: 'ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION',
    2: 'ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION',
    0: 'ZE_CACHE_EXT_REGION_DEFAULT',
    1: 'ZE_CACHE_EXT_REGION_RESERVED',
    2: 'ZE_CACHE_EXT_REGION_NON_RESERVED',
    2147483647: 'ZE_CACHE_EXT_REGION_FORCE_UINT32',
}
ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT = 0
ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION = 1
ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION = 2
ZE_CACHE_EXT_REGION_DEFAULT = 0
ZE_CACHE_EXT_REGION_RESERVED = 1
ZE_CACHE_EXT_REGION_NON_RESERVED = 2
ZE_CACHE_EXT_REGION_FORCE_UINT32 = 2147483647
_ze_cache_ext_region_t = ctypes.c_uint32 # enum
ze_cache_ext_region_t = _ze_cache_ext_region_t
ze_cache_ext_region_t__enumvalues = _ze_cache_ext_region_t__enumvalues
try:
    zeDeviceReserveCacheExt = _libraries['libze_loader.so'].zeDeviceReserveCacheExt
    zeDeviceReserveCacheExt.restype = ze_result_t
    zeDeviceReserveCacheExt.argtypes = [ze_device_handle_t, size_t, size_t]
except AttributeError:
    pass
try:
    zeDeviceSetCacheAdviceExt = _libraries['libze_loader.so'].zeDeviceSetCacheAdviceExt
    zeDeviceSetCacheAdviceExt.restype = ze_result_t
    zeDeviceSetCacheAdviceExt.argtypes = [ze_device_handle_t, ctypes.POINTER(None), size_t, ze_cache_ext_region_t]
except AttributeError:
    pass

# values for enumeration '_ze_event_query_timestamps_exp_version_t'
_ze_event_query_timestamps_exp_version_t__enumvalues = {
    65536: 'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0',
    65536: 'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT',
    2147483647: 'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32',
}
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0 = 65536
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT = 65536
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_event_query_timestamps_exp_version_t = ctypes.c_uint32 # enum
ze_event_query_timestamps_exp_version_t = _ze_event_query_timestamps_exp_version_t
ze_event_query_timestamps_exp_version_t__enumvalues = _ze_event_query_timestamps_exp_version_t__enumvalues
try:
    zeEventQueryTimestampsExp = _libraries['libze_loader.so'].zeEventQueryTimestampsExp
    zeEventQueryTimestampsExp.restype = ze_result_t
    zeEventQueryTimestampsExp.argtypes = [ze_event_handle_t, ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_kernel_timestamp_result_t)]
except AttributeError:
    pass

# values for enumeration '_ze_image_memory_properties_exp_version_t'
_ze_image_memory_properties_exp_version_t__enumvalues = {
    65536: 'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0',
    65536: 'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32',
}
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0 = 65536
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT = 65536
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_image_memory_properties_exp_version_t = ctypes.c_uint32 # enum
ze_image_memory_properties_exp_version_t = _ze_image_memory_properties_exp_version_t
ze_image_memory_properties_exp_version_t__enumvalues = _ze_image_memory_properties_exp_version_t__enumvalues
try:
    zeImageGetMemoryPropertiesExp = _libraries['libze_loader.so'].zeImageGetMemoryPropertiesExp
    zeImageGetMemoryPropertiesExp.restype = ze_result_t
    zeImageGetMemoryPropertiesExp.argtypes = [ze_image_handle_t, ctypes.POINTER(struct__ze_image_memory_properties_exp_t)]
except AttributeError:
    pass

# values for enumeration '_ze_image_view_ext_version_t'
_ze_image_view_ext_version_t__enumvalues = {
    65536: 'ZE_IMAGE_VIEW_EXT_VERSION_1_0',
    65536: 'ZE_IMAGE_VIEW_EXT_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_VIEW_EXT_VERSION_FORCE_UINT32',
}
ZE_IMAGE_VIEW_EXT_VERSION_1_0 = 65536
ZE_IMAGE_VIEW_EXT_VERSION_CURRENT = 65536
ZE_IMAGE_VIEW_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_image_view_ext_version_t = ctypes.c_uint32 # enum
ze_image_view_ext_version_t = _ze_image_view_ext_version_t
ze_image_view_ext_version_t__enumvalues = _ze_image_view_ext_version_t__enumvalues
try:
    zeImageViewCreateExt = _libraries['libze_loader.so'].zeImageViewCreateExt
    zeImageViewCreateExt.restype = ze_result_t
    zeImageViewCreateExt.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_image_desc_t), ze_image_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_image_view_exp_version_t'
_ze_image_view_exp_version_t__enumvalues = {
    65536: 'ZE_IMAGE_VIEW_EXP_VERSION_1_0',
    65536: 'ZE_IMAGE_VIEW_EXP_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32',
}
ZE_IMAGE_VIEW_EXP_VERSION_1_0 = 65536
ZE_IMAGE_VIEW_EXP_VERSION_CURRENT = 65536
ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_image_view_exp_version_t = ctypes.c_uint32 # enum
ze_image_view_exp_version_t = _ze_image_view_exp_version_t
ze_image_view_exp_version_t__enumvalues = _ze_image_view_exp_version_t__enumvalues
try:
    zeImageViewCreateExp = _libraries['libze_loader.so'].zeImageViewCreateExp
    zeImageViewCreateExp.restype = ze_result_t
    zeImageViewCreateExp.argtypes = [ze_context_handle_t, ze_device_handle_t, ctypes.POINTER(struct__ze_image_desc_t), ze_image_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_image_view_planar_ext_version_t'
_ze_image_view_planar_ext_version_t__enumvalues = {
    65536: 'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_1_0',
    65536: 'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_FORCE_UINT32',
}
ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_1_0 = 65536
ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_CURRENT = 65536
ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_image_view_planar_ext_version_t = ctypes.c_uint32 # enum
ze_image_view_planar_ext_version_t = _ze_image_view_planar_ext_version_t
ze_image_view_planar_ext_version_t__enumvalues = _ze_image_view_planar_ext_version_t__enumvalues

# values for enumeration '_ze_image_view_planar_exp_version_t'
_ze_image_view_planar_exp_version_t__enumvalues = {
    65536: 'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0',
    65536: 'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32',
}
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0 = 65536
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT = 65536
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_image_view_planar_exp_version_t = ctypes.c_uint32 # enum
ze_image_view_planar_exp_version_t = _ze_image_view_planar_exp_version_t
ze_image_view_planar_exp_version_t__enumvalues = _ze_image_view_planar_exp_version_t__enumvalues

# values for enumeration '_ze_scheduling_hints_exp_version_t'
_ze_scheduling_hints_exp_version_t__enumvalues = {
    65536: 'ZE_SCHEDULING_HINTS_EXP_VERSION_1_0',
    65536: 'ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT',
    2147483647: 'ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32',
}
ZE_SCHEDULING_HINTS_EXP_VERSION_1_0 = 65536
ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT = 65536
ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_scheduling_hints_exp_version_t = ctypes.c_uint32 # enum
ze_scheduling_hints_exp_version_t = _ze_scheduling_hints_exp_version_t
ze_scheduling_hints_exp_version_t__enumvalues = _ze_scheduling_hints_exp_version_t__enumvalues
ze_scheduling_hint_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_scheduling_hint_exp_flag_t'
_ze_scheduling_hint_exp_flag_t__enumvalues = {
    1: 'ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST',
    2: 'ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN',
    4: 'ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN',
    2147483647: 'ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32',
}
ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST = 1
ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN = 2
ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN = 4
ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_scheduling_hint_exp_flag_t = ctypes.c_uint32 # enum
ze_scheduling_hint_exp_flag_t = _ze_scheduling_hint_exp_flag_t
ze_scheduling_hint_exp_flag_t__enumvalues = _ze_scheduling_hint_exp_flag_t__enumvalues
try:
    zeKernelSchedulingHintExp = _libraries['libze_loader.so'].zeKernelSchedulingHintExp
    zeKernelSchedulingHintExp.restype = ze_result_t
    zeKernelSchedulingHintExp.argtypes = [ze_kernel_handle_t, ctypes.POINTER(struct__ze_scheduling_hint_exp_desc_t)]
except AttributeError:
    pass

# values for enumeration '_ze_linkonce_odr_ext_version_t'
_ze_linkonce_odr_ext_version_t__enumvalues = {
    65536: 'ZE_LINKONCE_ODR_EXT_VERSION_1_0',
    65536: 'ZE_LINKONCE_ODR_EXT_VERSION_CURRENT',
    2147483647: 'ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32',
}
ZE_LINKONCE_ODR_EXT_VERSION_1_0 = 65536
ZE_LINKONCE_ODR_EXT_VERSION_CURRENT = 65536
ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_linkonce_odr_ext_version_t = ctypes.c_uint32 # enum
ze_linkonce_odr_ext_version_t = _ze_linkonce_odr_ext_version_t
ze_linkonce_odr_ext_version_t__enumvalues = _ze_linkonce_odr_ext_version_t__enumvalues

# values for enumeration '_ze_power_saving_hint_exp_version_t'
_ze_power_saving_hint_exp_version_t__enumvalues = {
    65536: 'ZE_POWER_SAVING_HINT_EXP_VERSION_1_0',
    65536: 'ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT',
    2147483647: 'ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32',
}
ZE_POWER_SAVING_HINT_EXP_VERSION_1_0 = 65536
ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT = 65536
ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_power_saving_hint_exp_version_t = ctypes.c_uint32 # enum
ze_power_saving_hint_exp_version_t = _ze_power_saving_hint_exp_version_t
ze_power_saving_hint_exp_version_t__enumvalues = _ze_power_saving_hint_exp_version_t__enumvalues

# values for enumeration '_ze_power_saving_hint_type_t'
_ze_power_saving_hint_type_t__enumvalues = {
    0: 'ZE_POWER_SAVING_HINT_TYPE_MIN',
    100: 'ZE_POWER_SAVING_HINT_TYPE_MAX',
    2147483647: 'ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32',
}
ZE_POWER_SAVING_HINT_TYPE_MIN = 0
ZE_POWER_SAVING_HINT_TYPE_MAX = 100
ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32 = 2147483647
_ze_power_saving_hint_type_t = ctypes.c_uint32 # enum
ze_power_saving_hint_type_t = _ze_power_saving_hint_type_t
ze_power_saving_hint_type_t__enumvalues = _ze_power_saving_hint_type_t__enumvalues

# values for enumeration '_ze_subgroup_ext_version_t'
_ze_subgroup_ext_version_t__enumvalues = {
    65536: 'ZE_SUBGROUP_EXT_VERSION_1_0',
    65536: 'ZE_SUBGROUP_EXT_VERSION_CURRENT',
    2147483647: 'ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32',
}
ZE_SUBGROUP_EXT_VERSION_1_0 = 65536
ZE_SUBGROUP_EXT_VERSION_CURRENT = 65536
ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_subgroup_ext_version_t = ctypes.c_uint32 # enum
ze_subgroup_ext_version_t = _ze_subgroup_ext_version_t
ze_subgroup_ext_version_t__enumvalues = _ze_subgroup_ext_version_t__enumvalues

# values for enumeration '_ze_eu_count_ext_version_t'
_ze_eu_count_ext_version_t__enumvalues = {
    65536: 'ZE_EU_COUNT_EXT_VERSION_1_0',
    65536: 'ZE_EU_COUNT_EXT_VERSION_CURRENT',
    2147483647: 'ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32',
}
ZE_EU_COUNT_EXT_VERSION_1_0 = 65536
ZE_EU_COUNT_EXT_VERSION_CURRENT = 65536
ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_eu_count_ext_version_t = ctypes.c_uint32 # enum
ze_eu_count_ext_version_t = _ze_eu_count_ext_version_t
ze_eu_count_ext_version_t__enumvalues = _ze_eu_count_ext_version_t__enumvalues

# values for enumeration '_ze_pci_properties_ext_version_t'
_ze_pci_properties_ext_version_t__enumvalues = {
    65536: 'ZE_PCI_PROPERTIES_EXT_VERSION_1_0',
    65536: 'ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT',
    2147483647: 'ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32',
}
ZE_PCI_PROPERTIES_EXT_VERSION_1_0 = 65536
ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT = 65536
ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_pci_properties_ext_version_t = ctypes.c_uint32 # enum
ze_pci_properties_ext_version_t = _ze_pci_properties_ext_version_t
ze_pci_properties_ext_version_t__enumvalues = _ze_pci_properties_ext_version_t__enumvalues
try:
    zeDevicePciGetPropertiesExt = _libraries['libze_loader.so'].zeDevicePciGetPropertiesExt
    zeDevicePciGetPropertiesExt.restype = ze_result_t
    zeDevicePciGetPropertiesExt.argtypes = [ze_device_handle_t, ctypes.POINTER(struct__ze_pci_ext_properties_t)]
except AttributeError:
    pass

# values for enumeration '_ze_srgb_ext_version_t'
_ze_srgb_ext_version_t__enumvalues = {
    65536: 'ZE_SRGB_EXT_VERSION_1_0',
    65536: 'ZE_SRGB_EXT_VERSION_CURRENT',
    2147483647: 'ZE_SRGB_EXT_VERSION_FORCE_UINT32',
}
ZE_SRGB_EXT_VERSION_1_0 = 65536
ZE_SRGB_EXT_VERSION_CURRENT = 65536
ZE_SRGB_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_srgb_ext_version_t = ctypes.c_uint32 # enum
ze_srgb_ext_version_t = _ze_srgb_ext_version_t
ze_srgb_ext_version_t__enumvalues = _ze_srgb_ext_version_t__enumvalues

# values for enumeration '_ze_image_copy_ext_version_t'
_ze_image_copy_ext_version_t__enumvalues = {
    65536: 'ZE_IMAGE_COPY_EXT_VERSION_1_0',
    65536: 'ZE_IMAGE_COPY_EXT_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32',
}
ZE_IMAGE_COPY_EXT_VERSION_1_0 = 65536
ZE_IMAGE_COPY_EXT_VERSION_CURRENT = 65536
ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_image_copy_ext_version_t = ctypes.c_uint32 # enum
ze_image_copy_ext_version_t = _ze_image_copy_ext_version_t
ze_image_copy_ext_version_t__enumvalues = _ze_image_copy_ext_version_t__enumvalues
try:
    zeCommandListAppendImageCopyToMemoryExt = _libraries['libze_loader.so'].zeCommandListAppendImageCopyToMemoryExt
    zeCommandListAppendImageCopyToMemoryExt.restype = ze_result_t
    zeCommandListAppendImageCopyToMemoryExt.argtypes = [ze_command_list_handle_t, ctypes.POINTER(None), ze_image_handle_t, ctypes.POINTER(struct__ze_image_region_t), uint32_t, uint32_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass
try:
    zeCommandListAppendImageCopyFromMemoryExt = _libraries['libze_loader.so'].zeCommandListAppendImageCopyFromMemoryExt
    zeCommandListAppendImageCopyFromMemoryExt.restype = ze_result_t
    zeCommandListAppendImageCopyFromMemoryExt.argtypes = [ze_command_list_handle_t, ze_image_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_image_region_t), uint32_t, uint32_t, ze_event_handle_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_image_query_alloc_properties_ext_version_t'
_ze_image_query_alloc_properties_ext_version_t__enumvalues = {
    65536: 'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0',
    65536: 'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT',
    2147483647: 'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32',
}
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0 = 65536
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT = 65536
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_image_query_alloc_properties_ext_version_t = ctypes.c_uint32 # enum
ze_image_query_alloc_properties_ext_version_t = _ze_image_query_alloc_properties_ext_version_t
ze_image_query_alloc_properties_ext_version_t__enumvalues = _ze_image_query_alloc_properties_ext_version_t__enumvalues
try:
    zeImageGetAllocPropertiesExt = _libraries['libze_loader.so'].zeImageGetAllocPropertiesExt
    zeImageGetAllocPropertiesExt.restype = ze_result_t
    zeImageGetAllocPropertiesExt.argtypes = [ze_context_handle_t, ze_image_handle_t, ctypes.POINTER(struct__ze_image_allocation_ext_properties_t)]
except AttributeError:
    pass

# values for enumeration '_ze_linkage_inspection_ext_version_t'
_ze_linkage_inspection_ext_version_t__enumvalues = {
    65536: 'ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0',
    65536: 'ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT',
    2147483647: 'ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32',
}
ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0 = 65536
ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT = 65536
ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_linkage_inspection_ext_version_t = ctypes.c_uint32 # enum
ze_linkage_inspection_ext_version_t = _ze_linkage_inspection_ext_version_t
ze_linkage_inspection_ext_version_t__enumvalues = _ze_linkage_inspection_ext_version_t__enumvalues
ze_linkage_inspection_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_linkage_inspection_ext_flag_t'
_ze_linkage_inspection_ext_flag_t__enumvalues = {
    1: 'ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS',
    2: 'ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS',
    4: 'ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS',
    2147483647: 'ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32',
}
ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS = 1
ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS = 2
ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS = 4
ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_linkage_inspection_ext_flag_t = ctypes.c_uint32 # enum
ze_linkage_inspection_ext_flag_t = _ze_linkage_inspection_ext_flag_t
ze_linkage_inspection_ext_flag_t__enumvalues = _ze_linkage_inspection_ext_flag_t__enumvalues
try:
    zeModuleInspectLinkageExt = _libraries['libze_loader.so'].zeModuleInspectLinkageExt
    zeModuleInspectLinkageExt.restype = ze_result_t
    zeModuleInspectLinkageExt.argtypes = [ctypes.POINTER(struct__ze_linkage_inspection_ext_desc_t), uint32_t, ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t))]
except AttributeError:
    pass

# values for enumeration '_ze_memory_compression_hints_ext_version_t'
_ze_memory_compression_hints_ext_version_t__enumvalues = {
    65536: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0',
    65536: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT',
    2147483647: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32',
}
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0 = 65536
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT = 65536
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_memory_compression_hints_ext_version_t = ctypes.c_uint32 # enum
ze_memory_compression_hints_ext_version_t = _ze_memory_compression_hints_ext_version_t
ze_memory_compression_hints_ext_version_t__enumvalues = _ze_memory_compression_hints_ext_version_t__enumvalues
ze_memory_compression_hints_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_memory_compression_hints_ext_flag_t'
_ze_memory_compression_hints_ext_flag_t__enumvalues = {
    1: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED',
    2: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED',
    2147483647: 'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32',
}
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED = 1
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED = 2
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_memory_compression_hints_ext_flag_t = ctypes.c_uint32 # enum
ze_memory_compression_hints_ext_flag_t = _ze_memory_compression_hints_ext_flag_t
ze_memory_compression_hints_ext_flag_t__enumvalues = _ze_memory_compression_hints_ext_flag_t__enumvalues

# values for enumeration '_ze_memory_free_policies_ext_version_t'
_ze_memory_free_policies_ext_version_t__enumvalues = {
    65536: 'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0',
    65536: 'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT',
    2147483647: 'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32',
}
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0 = 65536
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT = 65536
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_memory_free_policies_ext_version_t = ctypes.c_uint32 # enum
ze_memory_free_policies_ext_version_t = _ze_memory_free_policies_ext_version_t
ze_memory_free_policies_ext_version_t__enumvalues = _ze_memory_free_policies_ext_version_t__enumvalues
ze_driver_memory_free_policy_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_driver_memory_free_policy_ext_flag_t'
_ze_driver_memory_free_policy_ext_flag_t__enumvalues = {
    1: 'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE',
    2: 'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE',
    2147483647: 'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32',
}
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE = 1
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE = 2
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_driver_memory_free_policy_ext_flag_t = ctypes.c_uint32 # enum
ze_driver_memory_free_policy_ext_flag_t = _ze_driver_memory_free_policy_ext_flag_t
ze_driver_memory_free_policy_ext_flag_t__enumvalues = _ze_driver_memory_free_policy_ext_flag_t__enumvalues
try:
    zeMemFreeExt = _libraries['libze_loader.so'].zeMemFreeExt
    zeMemFreeExt.restype = ze_result_t
    zeMemFreeExt.argtypes = [ze_context_handle_t, ctypes.POINTER(struct__ze_memory_free_ext_desc_t), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration '_ze_device_luid_ext_version_t'
_ze_device_luid_ext_version_t__enumvalues = {
    65536: 'ZE_DEVICE_LUID_EXT_VERSION_1_0',
    65536: 'ZE_DEVICE_LUID_EXT_VERSION_CURRENT',
    2147483647: 'ZE_DEVICE_LUID_EXT_VERSION_FORCE_UINT32',
}
ZE_DEVICE_LUID_EXT_VERSION_1_0 = 65536
ZE_DEVICE_LUID_EXT_VERSION_CURRENT = 65536
ZE_DEVICE_LUID_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_device_luid_ext_version_t = ctypes.c_uint32 # enum
ze_device_luid_ext_version_t = _ze_device_luid_ext_version_t
ze_device_luid_ext_version_t__enumvalues = _ze_device_luid_ext_version_t__enumvalues
try:
    zeFabricVertexGetExp = _libraries['libze_loader.so'].zeFabricVertexGetExp
    zeFabricVertexGetExp.restype = ze_result_t
    zeFabricVertexGetExp.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_vertex_handle_t))]
except AttributeError:
    pass
try:
    zeFabricVertexGetSubVerticesExp = _libraries['libze_loader.so'].zeFabricVertexGetSubVerticesExp
    zeFabricVertexGetSubVerticesExp.restype = ze_result_t
    zeFabricVertexGetSubVerticesExp.argtypes = [ze_fabric_vertex_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_vertex_handle_t))]
except AttributeError:
    pass
try:
    zeFabricVertexGetPropertiesExp = _libraries['libze_loader.so'].zeFabricVertexGetPropertiesExp
    zeFabricVertexGetPropertiesExp.restype = ze_result_t
    zeFabricVertexGetPropertiesExp.argtypes = [ze_fabric_vertex_handle_t, ctypes.POINTER(struct__ze_fabric_vertex_exp_properties_t)]
except AttributeError:
    pass
try:
    zeFabricVertexGetDeviceExp = _libraries['libze_loader.so'].zeFabricVertexGetDeviceExp
    zeFabricVertexGetDeviceExp.restype = ze_result_t
    zeFabricVertexGetDeviceExp.argtypes = [ze_fabric_vertex_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))]
except AttributeError:
    pass
try:
    zeDeviceGetFabricVertexExp = _libraries['libze_loader.so'].zeDeviceGetFabricVertexExp
    zeDeviceGetFabricVertexExp.restype = ze_result_t
    zeDeviceGetFabricVertexExp.argtypes = [ze_device_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_vertex_handle_t))]
except AttributeError:
    pass
try:
    zeFabricEdgeGetExp = _libraries['libze_loader.so'].zeFabricEdgeGetExp
    zeFabricEdgeGetExp.restype = ze_result_t
    zeFabricEdgeGetExp.argtypes = [ze_fabric_vertex_handle_t, ze_fabric_vertex_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_edge_handle_t))]
except AttributeError:
    pass
try:
    zeFabricEdgeGetVerticesExp = _libraries['libze_loader.so'].zeFabricEdgeGetVerticesExp
    zeFabricEdgeGetVerticesExp.restype = ze_result_t
    zeFabricEdgeGetVerticesExp.argtypes = [ze_fabric_edge_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_vertex_handle_t)), ctypes.POINTER(ctypes.POINTER(struct__ze_fabric_vertex_handle_t))]
except AttributeError:
    pass
try:
    zeFabricEdgeGetPropertiesExp = _libraries['libze_loader.so'].zeFabricEdgeGetPropertiesExp
    zeFabricEdgeGetPropertiesExp.restype = ze_result_t
    zeFabricEdgeGetPropertiesExp.argtypes = [ze_fabric_edge_handle_t, ctypes.POINTER(struct__ze_fabric_edge_exp_properties_t)]
except AttributeError:
    pass

# values for enumeration '_ze_device_memory_properties_ext_version_t'
_ze_device_memory_properties_ext_version_t__enumvalues = {
    65536: 'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_1_0',
    65536: 'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_CURRENT',
    2147483647: 'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_FORCE_UINT32',
}
ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_1_0 = 65536
ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_CURRENT = 65536
ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_device_memory_properties_ext_version_t = ctypes.c_uint32 # enum
ze_device_memory_properties_ext_version_t = _ze_device_memory_properties_ext_version_t
ze_device_memory_properties_ext_version_t__enumvalues = _ze_device_memory_properties_ext_version_t__enumvalues

# values for enumeration '_ze_bfloat16_conversions_ext_version_t'
_ze_bfloat16_conversions_ext_version_t__enumvalues = {
    65536: 'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_1_0',
    65536: 'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_CURRENT',
    2147483647: 'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_FORCE_UINT32',
}
ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_1_0 = 65536
ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_CURRENT = 65536
ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_bfloat16_conversions_ext_version_t = ctypes.c_uint32 # enum
ze_bfloat16_conversions_ext_version_t = _ze_bfloat16_conversions_ext_version_t
ze_bfloat16_conversions_ext_version_t__enumvalues = _ze_bfloat16_conversions_ext_version_t__enumvalues

# values for enumeration '_ze_device_ip_version_version_t'
_ze_device_ip_version_version_t__enumvalues = {
    65536: 'ZE_DEVICE_IP_VERSION_VERSION_1_0',
    65536: 'ZE_DEVICE_IP_VERSION_VERSION_CURRENT',
    2147483647: 'ZE_DEVICE_IP_VERSION_VERSION_FORCE_UINT32',
}
ZE_DEVICE_IP_VERSION_VERSION_1_0 = 65536
ZE_DEVICE_IP_VERSION_VERSION_CURRENT = 65536
ZE_DEVICE_IP_VERSION_VERSION_FORCE_UINT32 = 2147483647
_ze_device_ip_version_version_t = ctypes.c_uint32 # enum
ze_device_ip_version_version_t = _ze_device_ip_version_version_t
ze_device_ip_version_version_t__enumvalues = _ze_device_ip_version_version_t__enumvalues

# values for enumeration '_ze_kernel_max_group_size_properties_ext_version_t'
_ze_kernel_max_group_size_properties_ext_version_t__enumvalues = {
    65536: 'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_1_0',
    65536: 'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_CURRENT',
    2147483647: 'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_FORCE_UINT32',
}
ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_1_0 = 65536
ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_CURRENT = 65536
ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_kernel_max_group_size_properties_ext_version_t = ctypes.c_uint32 # enum
ze_kernel_max_group_size_properties_ext_version_t = _ze_kernel_max_group_size_properties_ext_version_t
ze_kernel_max_group_size_properties_ext_version_t__enumvalues = _ze_kernel_max_group_size_properties_ext_version_t__enumvalues
ze_kernel_max_group_size_ext_properties_t = struct__ze_kernel_max_group_size_properties_ext_t

# values for enumeration '_ze_sub_allocations_exp_version_t'
_ze_sub_allocations_exp_version_t__enumvalues = {
    65536: 'ZE_SUB_ALLOCATIONS_EXP_VERSION_1_0',
    65536: 'ZE_SUB_ALLOCATIONS_EXP_VERSION_CURRENT',
    2147483647: 'ZE_SUB_ALLOCATIONS_EXP_VERSION_FORCE_UINT32',
}
ZE_SUB_ALLOCATIONS_EXP_VERSION_1_0 = 65536
ZE_SUB_ALLOCATIONS_EXP_VERSION_CURRENT = 65536
ZE_SUB_ALLOCATIONS_EXP_VERSION_FORCE_UINT32 = 2147483647
_ze_sub_allocations_exp_version_t = ctypes.c_uint32 # enum
ze_sub_allocations_exp_version_t = _ze_sub_allocations_exp_version_t
ze_sub_allocations_exp_version_t__enumvalues = _ze_sub_allocations_exp_version_t__enumvalues

# values for enumeration '_ze_event_query_kernel_timestamps_ext_version_t'
_ze_event_query_kernel_timestamps_ext_version_t__enumvalues = {
    65536: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_1_0',
    65536: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_CURRENT',
    2147483647: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_FORCE_UINT32',
}
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_1_0 = 65536
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_CURRENT = 65536
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_FORCE_UINT32 = 2147483647
_ze_event_query_kernel_timestamps_ext_version_t = ctypes.c_uint32 # enum
ze_event_query_kernel_timestamps_ext_version_t = _ze_event_query_kernel_timestamps_ext_version_t
ze_event_query_kernel_timestamps_ext_version_t__enumvalues = _ze_event_query_kernel_timestamps_ext_version_t__enumvalues
ze_event_query_kernel_timestamps_ext_flags_t = ctypes.c_uint32

# values for enumeration '_ze_event_query_kernel_timestamps_ext_flag_t'
_ze_event_query_kernel_timestamps_ext_flag_t__enumvalues = {
    1: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_KERNEL',
    2: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_SYNCHRONIZED',
    2147483647: 'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_FORCE_UINT32',
}
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_KERNEL = 1
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_SYNCHRONIZED = 2
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_FORCE_UINT32 = 2147483647
_ze_event_query_kernel_timestamps_ext_flag_t = ctypes.c_uint32 # enum
ze_event_query_kernel_timestamps_ext_flag_t = _ze_event_query_kernel_timestamps_ext_flag_t
ze_event_query_kernel_timestamps_ext_flag_t__enumvalues = _ze_event_query_kernel_timestamps_ext_flag_t__enumvalues
try:
    zeEventQueryKernelTimestampsExt = _libraries['libze_loader.so'].zeEventQueryKernelTimestampsExt
    zeEventQueryKernelTimestampsExt.restype = ze_result_t
    zeEventQueryKernelTimestampsExt.argtypes = [ze_event_handle_t, ze_device_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__ze_event_query_kernel_timestamps_results_ext_properties_t)]
except AttributeError:
    pass
ze_rtas_device_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_device_exp_flag_t'
_ze_rtas_device_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_DEVICE_EXP_FLAG_RESERVED',
    2147483647: 'ZE_RTAS_DEVICE_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_DEVICE_EXP_FLAG_RESERVED = 1
ZE_RTAS_DEVICE_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_device_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_device_exp_flag_t = _ze_rtas_device_exp_flag_t
ze_rtas_device_exp_flag_t__enumvalues = _ze_rtas_device_exp_flag_t__enumvalues
ze_rtas_builder_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_builder_exp_flag_t'
_ze_rtas_builder_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_BUILDER_EXP_FLAG_RESERVED',
    2147483647: 'ZE_RTAS_BUILDER_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_BUILDER_EXP_FLAG_RESERVED = 1
ZE_RTAS_BUILDER_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_builder_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_builder_exp_flag_t = _ze_rtas_builder_exp_flag_t
ze_rtas_builder_exp_flag_t__enumvalues = _ze_rtas_builder_exp_flag_t__enumvalues
ze_rtas_parallel_operation_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_parallel_operation_exp_flag_t'
_ze_rtas_parallel_operation_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_RESERVED',
    2147483647: 'ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_RESERVED = 1
ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_parallel_operation_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_parallel_operation_exp_flag_t = _ze_rtas_parallel_operation_exp_flag_t
ze_rtas_parallel_operation_exp_flag_t__enumvalues = _ze_rtas_parallel_operation_exp_flag_t__enumvalues
ze_rtas_builder_geometry_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_builder_geometry_exp_flag_t'
_ze_rtas_builder_geometry_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE',
    2147483647: 'ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE = 1
ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_builder_geometry_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_builder_geometry_exp_flag_t = _ze_rtas_builder_geometry_exp_flag_t
ze_rtas_builder_geometry_exp_flag_t__enumvalues = _ze_rtas_builder_geometry_exp_flag_t__enumvalues
ze_rtas_builder_packed_geometry_exp_flags_t = ctypes.c_ubyte
ze_rtas_builder_instance_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_builder_instance_exp_flag_t'
_ze_rtas_builder_instance_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_CULL_DISABLE',
    2: 'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE',
    4: 'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_OPAQUE',
    8: 'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_NON_OPAQUE',
    2147483647: 'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_CULL_DISABLE = 1
ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE = 2
ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_OPAQUE = 4
ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_NON_OPAQUE = 8
ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_builder_instance_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_builder_instance_exp_flag_t = _ze_rtas_builder_instance_exp_flag_t
ze_rtas_builder_instance_exp_flag_t__enumvalues = _ze_rtas_builder_instance_exp_flag_t__enumvalues
ze_rtas_builder_packed_instance_exp_flags_t = ctypes.c_ubyte
ze_rtas_builder_build_op_exp_flags_t = ctypes.c_uint32

# values for enumeration '_ze_rtas_builder_build_op_exp_flag_t'
_ze_rtas_builder_build_op_exp_flag_t__enumvalues = {
    1: 'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_COMPACT',
    2: 'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION',
    2147483647: 'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_FORCE_UINT32',
}
ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_COMPACT = 1
ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION = 2
ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_FORCE_UINT32 = 2147483647
_ze_rtas_builder_build_op_exp_flag_t = ctypes.c_uint32 # enum
ze_rtas_builder_build_op_exp_flag_t = _ze_rtas_builder_build_op_exp_flag_t
ze_rtas_builder_build_op_exp_flag_t__enumvalues = _ze_rtas_builder_build_op_exp_flag_t__enumvalues

# values for enumeration '_ze_rtas_builder_geometry_type_exp_t'
_ze_rtas_builder_geometry_type_exp_t__enumvalues = {
    0: 'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES',
    1: 'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS',
    2: 'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL',
    3: 'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE',
    2147483647: 'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_FORCE_UINT32',
}
ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES = 0
ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS = 1
ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL = 2
ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE = 3
ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_FORCE_UINT32 = 2147483647
_ze_rtas_builder_geometry_type_exp_t = ctypes.c_uint32 # enum
ze_rtas_builder_geometry_type_exp_t = _ze_rtas_builder_geometry_type_exp_t
ze_rtas_builder_geometry_type_exp_t__enumvalues = _ze_rtas_builder_geometry_type_exp_t__enumvalues
ze_rtas_builder_packed_geometry_type_exp_t = ctypes.c_ubyte

# values for enumeration '_ze_rtas_builder_input_data_format_exp_t'
_ze_rtas_builder_input_data_format_exp_t__enumvalues = {
    0: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3',
    1: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR',
    2: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR',
    3: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR',
    4: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_AABB',
    5: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32',
    6: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32',
    2147483647: 'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FORCE_UINT32',
}
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3 = 0
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR = 1
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR = 2
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR = 3
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_AABB = 4
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32 = 5
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32 = 6
ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FORCE_UINT32 = 2147483647
_ze_rtas_builder_input_data_format_exp_t = ctypes.c_uint32 # enum
ze_rtas_builder_input_data_format_exp_t = _ze_rtas_builder_input_data_format_exp_t
ze_rtas_builder_input_data_format_exp_t__enumvalues = _ze_rtas_builder_input_data_format_exp_t__enumvalues
ze_rtas_builder_packed_input_data_format_exp_t = ctypes.c_ubyte
class struct__ze_rtas_builder_exp_handle_t(Structure):
    pass

ze_rtas_builder_exp_handle_t = ctypes.POINTER(struct__ze_rtas_builder_exp_handle_t)
class struct__ze_rtas_parallel_operation_exp_handle_t(Structure):
    pass

ze_rtas_parallel_operation_exp_handle_t = ctypes.POINTER(struct__ze_rtas_parallel_operation_exp_handle_t)
ze_rtas_geometry_aabbs_cb_exp_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_rtas_geometry_aabbs_exp_cb_params_t))
try:
    zeRTASBuilderCreateExp = _libraries['libze_loader.so'].zeRTASBuilderCreateExp
    zeRTASBuilderCreateExp.restype = ze_result_t
    zeRTASBuilderCreateExp.argtypes = [ze_driver_handle_t, ctypes.POINTER(struct__ze_rtas_builder_exp_desc_t), ctypes.POINTER(ctypes.POINTER(struct__ze_rtas_builder_exp_handle_t))]
except AttributeError:
    pass
try:
    zeRTASBuilderGetBuildPropertiesExp = _libraries['libze_loader.so'].zeRTASBuilderGetBuildPropertiesExp
    zeRTASBuilderGetBuildPropertiesExp.restype = ze_result_t
    zeRTASBuilderGetBuildPropertiesExp.argtypes = [ze_rtas_builder_exp_handle_t, ctypes.POINTER(struct__ze_rtas_builder_build_op_exp_desc_t), ctypes.POINTER(struct__ze_rtas_builder_exp_properties_t)]
except AttributeError:
    pass
try:
    zeDriverRTASFormatCompatibilityCheckExp = _libraries['libze_loader.so'].zeDriverRTASFormatCompatibilityCheckExp
    zeDriverRTASFormatCompatibilityCheckExp.restype = ze_result_t
    zeDriverRTASFormatCompatibilityCheckExp.argtypes = [ze_driver_handle_t, ze_rtas_format_exp_t, ze_rtas_format_exp_t]
except AttributeError:
    pass
try:
    zeRTASBuilderBuildExp = _libraries['libze_loader.so'].zeRTASBuilderBuildExp
    zeRTASBuilderBuildExp.restype = ze_result_t
    zeRTASBuilderBuildExp.argtypes = [ze_rtas_builder_exp_handle_t, ctypes.POINTER(struct__ze_rtas_builder_build_op_exp_desc_t), ctypes.POINTER(None), size_t, ctypes.POINTER(None), size_t, ze_rtas_parallel_operation_exp_handle_t, ctypes.POINTER(None), ctypes.POINTER(struct__ze_rtas_aabb_exp_t), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    zeRTASBuilderDestroyExp = _libraries['libze_loader.so'].zeRTASBuilderDestroyExp
    zeRTASBuilderDestroyExp.restype = ze_result_t
    zeRTASBuilderDestroyExp.argtypes = [ze_rtas_builder_exp_handle_t]
except AttributeError:
    pass
try:
    zeRTASParallelOperationCreateExp = _libraries['libze_loader.so'].zeRTASParallelOperationCreateExp
    zeRTASParallelOperationCreateExp.restype = ze_result_t
    zeRTASParallelOperationCreateExp.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.POINTER(struct__ze_rtas_parallel_operation_exp_handle_t))]
except AttributeError:
    pass
try:
    zeRTASParallelOperationGetPropertiesExp = _libraries['libze_loader.so'].zeRTASParallelOperationGetPropertiesExp
    zeRTASParallelOperationGetPropertiesExp.restype = ze_result_t
    zeRTASParallelOperationGetPropertiesExp.argtypes = [ze_rtas_parallel_operation_exp_handle_t, ctypes.POINTER(struct__ze_rtas_parallel_operation_exp_properties_t)]
except AttributeError:
    pass
try:
    zeRTASParallelOperationJoinExp = _libraries['libze_loader.so'].zeRTASParallelOperationJoinExp
    zeRTASParallelOperationJoinExp.restype = ze_result_t
    zeRTASParallelOperationJoinExp.argtypes = [ze_rtas_parallel_operation_exp_handle_t]
except AttributeError:
    pass
try:
    zeRTASParallelOperationDestroyExp = _libraries['libze_loader.so'].zeRTASParallelOperationDestroyExp
    zeRTASParallelOperationDestroyExp.restype = ze_result_t
    zeRTASParallelOperationDestroyExp.argtypes = [ze_rtas_parallel_operation_exp_handle_t]
except AttributeError:
    pass
class struct__ze_init_params_t(Structure):
    pass

struct__ze_init_params_t._pack_ = 1 # source:False
struct__ze_init_params_t._fields_ = [
    ('pflags', ctypes.POINTER(ctypes.c_uint32)),
]

ze_init_params_t = struct__ze_init_params_t
ze_pfnInitCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_init_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_global_callbacks_t(Structure):
    pass

struct__ze_global_callbacks_t._pack_ = 1 # source:False
struct__ze_global_callbacks_t._fields_ = [
    ('pfnInitCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_init_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_global_callbacks_t = struct__ze_global_callbacks_t
class struct__ze_driver_get_params_t(Structure):
    pass

struct__ze_driver_get_params_t._pack_ = 1 # source:False
struct__ze_driver_get_params_t._fields_ = [
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('pphDrivers', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t)))),
]

ze_driver_get_params_t = struct__ze_driver_get_params_t
ze_pfnDriverGetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_driver_get_api_version_params_t(Structure):
    pass

struct__ze_driver_get_api_version_params_t._pack_ = 1 # source:False
struct__ze_driver_get_api_version_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('pversion', ctypes.POINTER(ctypes.POINTER(_ze_api_version_t))),
]

ze_driver_get_api_version_params_t = struct__ze_driver_get_api_version_params_t
ze_pfnDriverGetApiVersionCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_api_version_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_driver_get_properties_params_t(Structure):
    pass

struct__ze_driver_get_properties_params_t._pack_ = 1 # source:False
struct__ze_driver_get_properties_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('ppDriverProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_properties_t))),
]

ze_driver_get_properties_params_t = struct__ze_driver_get_properties_params_t
ze_pfnDriverGetPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_driver_get_ipc_properties_params_t(Structure):
    pass

struct__ze_driver_get_ipc_properties_params_t._pack_ = 1 # source:False
struct__ze_driver_get_ipc_properties_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('ppIpcProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_ipc_properties_t))),
]

ze_driver_get_ipc_properties_params_t = struct__ze_driver_get_ipc_properties_params_t
ze_pfnDriverGetIpcPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_ipc_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_driver_get_extension_properties_params_t(Structure):
    pass

struct__ze_driver_get_extension_properties_params_t._pack_ = 1 # source:False
struct__ze_driver_get_extension_properties_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppExtensionProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_extension_properties_t))),
]

ze_driver_get_extension_properties_params_t = struct__ze_driver_get_extension_properties_params_t
ze_pfnDriverGetExtensionPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_extension_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_driver_callbacks_t(Structure):
    pass

struct__ze_driver_callbacks_t._pack_ = 1 # source:False
struct__ze_driver_callbacks_t._fields_ = [
    ('pfnGetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetApiVersionCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_api_version_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetIpcPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_ipc_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetExtensionPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_driver_get_extension_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_driver_callbacks_t = struct__ze_driver_callbacks_t
class struct__ze_device_get_params_t(Structure):
    pass

struct__ze_device_get_params_t._pack_ = 1 # source:False
struct__ze_device_get_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('pphDevices', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)))),
]

ze_device_get_params_t = struct__ze_device_get_params_t
ze_pfnDeviceGetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_sub_devices_params_t(Structure):
    pass

struct__ze_device_get_sub_devices_params_t._pack_ = 1 # source:False
struct__ze_device_get_sub_devices_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('pphSubdevices', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)))),
]

ze_device_get_sub_devices_params_t = struct__ze_device_get_sub_devices_params_t
ze_pfnDeviceGetSubDevicesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_sub_devices_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_properties_params_t(Structure):
    pass

struct__ze_device_get_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppDeviceProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_properties_t))),
]

ze_device_get_properties_params_t = struct__ze_device_get_properties_params_t
ze_pfnDeviceGetPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_compute_properties_params_t(Structure):
    pass

struct__ze_device_get_compute_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_compute_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppComputeProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_compute_properties_t))),
]

ze_device_get_compute_properties_params_t = struct__ze_device_get_compute_properties_params_t
ze_pfnDeviceGetComputePropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_compute_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_module_properties_params_t(Structure):
    pass

struct__ze_device_get_module_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_module_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppModuleProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_module_properties_t))),
]

ze_device_get_module_properties_params_t = struct__ze_device_get_module_properties_params_t
ze_pfnDeviceGetModulePropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_module_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_command_queue_group_properties_params_t(Structure):
    pass

struct__ze_device_get_command_queue_group_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_command_queue_group_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppCommandQueueGroupProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_group_properties_t))),
]

ze_device_get_command_queue_group_properties_params_t = struct__ze_device_get_command_queue_group_properties_params_t
ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_command_queue_group_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_memory_properties_params_t(Structure):
    pass

struct__ze_device_get_memory_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_memory_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppMemProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_memory_properties_t))),
]

ze_device_get_memory_properties_params_t = struct__ze_device_get_memory_properties_params_t
ze_pfnDeviceGetMemoryPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_memory_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_memory_access_properties_params_t(Structure):
    pass

struct__ze_device_get_memory_access_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_memory_access_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppMemAccessProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_memory_access_properties_t))),
]

ze_device_get_memory_access_properties_params_t = struct__ze_device_get_memory_access_properties_params_t
ze_pfnDeviceGetMemoryAccessPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_memory_access_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_cache_properties_params_t(Structure):
    pass

struct__ze_device_get_cache_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_cache_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppCacheProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_cache_properties_t))),
]

ze_device_get_cache_properties_params_t = struct__ze_device_get_cache_properties_params_t
ze_pfnDeviceGetCachePropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_cache_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_image_properties_params_t(Structure):
    pass

struct__ze_device_get_image_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_image_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppImageProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_image_properties_t))),
]

ze_device_get_image_properties_params_t = struct__ze_device_get_image_properties_params_t
ze_pfnDeviceGetImagePropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_image_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_external_memory_properties_params_t(Structure):
    pass

struct__ze_device_get_external_memory_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_external_memory_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppExternalMemoryProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_external_memory_properties_t))),
]

ze_device_get_external_memory_properties_params_t = struct__ze_device_get_external_memory_properties_params_t
ze_pfnDeviceGetExternalMemoryPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_external_memory_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_p2_p_properties_params_t(Structure):
    pass

struct__ze_device_get_p2_p_properties_params_t._pack_ = 1 # source:False
struct__ze_device_get_p2_p_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('phPeerDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppP2PProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_device_p2p_properties_t))),
]

ze_device_get_p2_p_properties_params_t = struct__ze_device_get_p2_p_properties_params_t
ze_pfnDeviceGetP2PPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_p2_p_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_can_access_peer_params_t(Structure):
    pass

struct__ze_device_can_access_peer_params_t._pack_ = 1 # source:False
struct__ze_device_can_access_peer_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('phPeerDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pvalue', ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))),
]

ze_device_can_access_peer_params_t = struct__ze_device_can_access_peer_params_t
ze_pfnDeviceCanAccessPeerCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_can_access_peer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_get_status_params_t(Structure):
    pass

struct__ze_device_get_status_params_t._pack_ = 1 # source:False
struct__ze_device_get_status_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
]

ze_device_get_status_params_t = struct__ze_device_get_status_params_t
ze_pfnDeviceGetStatusCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_device_callbacks_t(Structure):
    pass

struct__ze_device_callbacks_t._pack_ = 1 # source:False
struct__ze_device_callbacks_t._fields_ = [
    ('pfnGetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetSubDevicesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_sub_devices_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetComputePropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_compute_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetModulePropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_module_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetCommandQueueGroupPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_command_queue_group_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetMemoryPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_memory_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetMemoryAccessPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_memory_access_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetCachePropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_cache_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetImagePropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_image_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetExternalMemoryPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_external_memory_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetP2PPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_p2_p_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCanAccessPeerCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_can_access_peer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetStatusCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_device_get_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_device_callbacks_t = struct__ze_device_callbacks_t
class struct__ze_context_create_params_t(Structure):
    pass

struct__ze_context_create_params_t._pack_ = 1 # source:False
struct__ze_context_create_params_t._fields_ = [
    ('phDriver', ctypes.POINTER(ctypes.POINTER(struct__ze_driver_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_context_desc_t))),
    ('pphContext', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t)))),
]

ze_context_create_params_t = struct__ze_context_create_params_t
ze_pfnContextCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_destroy_params_t(Structure):
    pass

struct__ze_context_destroy_params_t._pack_ = 1 # source:False
struct__ze_context_destroy_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
]

ze_context_destroy_params_t = struct__ze_context_destroy_params_t
ze_pfnContextDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_get_status_params_t(Structure):
    pass

struct__ze_context_get_status_params_t._pack_ = 1 # source:False
struct__ze_context_get_status_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
]

ze_context_get_status_params_t = struct__ze_context_get_status_params_t
ze_pfnContextGetStatusCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_get_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_system_barrier_params_t(Structure):
    pass

struct__ze_context_system_barrier_params_t._pack_ = 1 # source:False
struct__ze_context_system_barrier_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
]

ze_context_system_barrier_params_t = struct__ze_context_system_barrier_params_t
ze_pfnContextSystemBarrierCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_system_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_make_memory_resident_params_t(Structure):
    pass

struct__ze_context_make_memory_resident_params_t._pack_ = 1 # source:False
struct__ze_context_make_memory_resident_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
]

ze_context_make_memory_resident_params_t = struct__ze_context_make_memory_resident_params_t
ze_pfnContextMakeMemoryResidentCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_make_memory_resident_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_evict_memory_params_t(Structure):
    pass

struct__ze_context_evict_memory_params_t._pack_ = 1 # source:False
struct__ze_context_evict_memory_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
]

ze_context_evict_memory_params_t = struct__ze_context_evict_memory_params_t
ze_pfnContextEvictMemoryCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_evict_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_make_image_resident_params_t(Structure):
    pass

struct__ze_context_make_image_resident_params_t._pack_ = 1 # source:False
struct__ze_context_make_image_resident_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('phImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
]

ze_context_make_image_resident_params_t = struct__ze_context_make_image_resident_params_t
ze_pfnContextMakeImageResidentCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_make_image_resident_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_evict_image_params_t(Structure):
    pass

struct__ze_context_evict_image_params_t._pack_ = 1 # source:False
struct__ze_context_evict_image_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('phImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
]

ze_context_evict_image_params_t = struct__ze_context_evict_image_params_t
ze_pfnContextEvictImageCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_evict_image_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_context_callbacks_t(Structure):
    pass

struct__ze_context_callbacks_t._pack_ = 1 # source:False
struct__ze_context_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetStatusCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_get_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSystemBarrierCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_system_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnMakeMemoryResidentCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_make_memory_resident_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnEvictMemoryCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_evict_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnMakeImageResidentCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_make_image_resident_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnEvictImageCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_context_evict_image_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_context_callbacks_t = struct__ze_context_callbacks_t
class struct__ze_command_queue_create_params_t(Structure):
    pass

struct__ze_command_queue_create_params_t._pack_ = 1 # source:False
struct__ze_command_queue_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_desc_t))),
    ('pphCommandQueue', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t)))),
]

ze_command_queue_create_params_t = struct__ze_command_queue_create_params_t
ze_pfnCommandQueueCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_queue_destroy_params_t(Structure):
    pass

struct__ze_command_queue_destroy_params_t._pack_ = 1 # source:False
struct__ze_command_queue_destroy_params_t._fields_ = [
    ('phCommandQueue', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t))),
]

ze_command_queue_destroy_params_t = struct__ze_command_queue_destroy_params_t
ze_pfnCommandQueueDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_queue_execute_command_lists_params_t(Structure):
    pass

struct__ze_command_queue_execute_command_lists_params_t._pack_ = 1 # source:False
struct__ze_command_queue_execute_command_lists_params_t._fields_ = [
    ('phCommandQueue', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t))),
    ('pnumCommandLists', ctypes.POINTER(ctypes.c_uint32)),
    ('pphCommandLists', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t)))),
    ('phFence', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))),
]

ze_command_queue_execute_command_lists_params_t = struct__ze_command_queue_execute_command_lists_params_t
ze_pfnCommandQueueExecuteCommandListsCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_execute_command_lists_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_queue_synchronize_params_t(Structure):
    pass

struct__ze_command_queue_synchronize_params_t._pack_ = 1 # source:False
struct__ze_command_queue_synchronize_params_t._fields_ = [
    ('phCommandQueue', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t))),
    ('ptimeout', ctypes.POINTER(ctypes.c_uint64)),
]

ze_command_queue_synchronize_params_t = struct__ze_command_queue_synchronize_params_t
ze_pfnCommandQueueSynchronizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_queue_callbacks_t(Structure):
    pass

struct__ze_command_queue_callbacks_t._pack_ = 1 # source:False
struct__ze_command_queue_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnExecuteCommandListsCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_execute_command_lists_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSynchronizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_queue_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_command_queue_callbacks_t = struct__ze_command_queue_callbacks_t
class struct__ze_command_list_create_params_t(Structure):
    pass

struct__ze_command_list_create_params_t._pack_ = 1 # source:False
struct__ze_command_list_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_desc_t))),
    ('pphCommandList', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t)))),
]

ze_command_list_create_params_t = struct__ze_command_list_create_params_t
ze_pfnCommandListCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_create_immediate_params_t(Structure):
    pass

struct__ze_command_list_create_immediate_params_t._pack_ = 1 # source:False
struct__ze_command_list_create_immediate_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('paltdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_desc_t))),
    ('pphCommandList', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t)))),
]

ze_command_list_create_immediate_params_t = struct__ze_command_list_create_immediate_params_t
ze_pfnCommandListCreateImmediateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_create_immediate_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_destroy_params_t(Structure):
    pass

struct__ze_command_list_destroy_params_t._pack_ = 1 # source:False
struct__ze_command_list_destroy_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
]

ze_command_list_destroy_params_t = struct__ze_command_list_destroy_params_t
ze_pfnCommandListDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_close_params_t(Structure):
    pass

struct__ze_command_list_close_params_t._pack_ = 1 # source:False
struct__ze_command_list_close_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
]

ze_command_list_close_params_t = struct__ze_command_list_close_params_t
ze_pfnCommandListCloseCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_close_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_reset_params_t(Structure):
    pass

struct__ze_command_list_reset_params_t._pack_ = 1 # source:False
struct__ze_command_list_reset_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
]

ze_command_list_reset_params_t = struct__ze_command_list_reset_params_t
ze_pfnCommandListResetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_write_global_timestamp_params_t(Structure):
    pass

struct__ze_command_list_append_write_global_timestamp_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_write_global_timestamp_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_write_global_timestamp_params_t = struct__ze_command_list_append_write_global_timestamp_params_t
ze_pfnCommandListAppendWriteGlobalTimestampCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_write_global_timestamp_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_barrier_params_t(Structure):
    pass

struct__ze_command_list_append_barrier_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_barrier_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_barrier_params_t = struct__ze_command_list_append_barrier_params_t
ze_pfnCommandListAppendBarrierCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_ranges_barrier_params_t(Structure):
    pass

struct__ze_command_list_append_memory_ranges_barrier_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_ranges_barrier_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pnumRanges', ctypes.POINTER(ctypes.c_uint32)),
    ('ppRangeSizes', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('ppRanges', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_memory_ranges_barrier_params_t = struct__ze_command_list_append_memory_ranges_barrier_params_t
ze_pfnCommandListAppendMemoryRangesBarrierCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_ranges_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_copy_params_t(Structure):
    pass

struct__ze_command_list_append_memory_copy_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_copy_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psrcptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_memory_copy_params_t = struct__ze_command_list_append_memory_copy_params_t
ze_pfnCommandListAppendMemoryCopyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_fill_params_t(Structure):
    pass

struct__ze_command_list_append_memory_fill_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_fill_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppattern', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppattern_size', ctypes.POINTER(ctypes.c_uint64)),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_memory_fill_params_t = struct__ze_command_list_append_memory_fill_params_t
ze_pfnCommandListAppendMemoryFillCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_fill_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_copy_region_params_t(Structure):
    pass

struct__ze_command_list_append_memory_copy_region_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_copy_region_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('pdstRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_copy_region_t))),
    ('pdstPitch', ctypes.POINTER(ctypes.c_uint32)),
    ('pdstSlicePitch', ctypes.POINTER(ctypes.c_uint32)),
    ('psrcptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psrcRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_copy_region_t))),
    ('psrcPitch', ctypes.POINTER(ctypes.c_uint32)),
    ('psrcSlicePitch', ctypes.POINTER(ctypes.c_uint32)),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_memory_copy_region_params_t = struct__ze_command_list_append_memory_copy_region_params_t
ze_pfnCommandListAppendMemoryCopyRegionCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_region_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_copy_from_context_params_t(Structure):
    pass

struct__ze_command_list_append_memory_copy_from_context_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_copy_from_context_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('phContextSrc', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('psrcptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_memory_copy_from_context_params_t = struct__ze_command_list_append_memory_copy_from_context_params_t
ze_pfnCommandListAppendMemoryCopyFromContextCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_from_context_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_image_copy_params_t(Structure):
    pass

struct__ze_command_list_append_image_copy_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_image_copy_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phDstImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('phSrcImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_image_copy_params_t = struct__ze_command_list_append_image_copy_params_t
ze_pfnCommandListAppendImageCopyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_image_copy_region_params_t(Structure):
    pass

struct__ze_command_list_append_image_copy_region_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_image_copy_region_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phDstImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('phSrcImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('ppDstRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_image_region_t))),
    ('ppSrcRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_image_region_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_image_copy_region_params_t = struct__ze_command_list_append_image_copy_region_params_t
ze_pfnCommandListAppendImageCopyRegionCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_region_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_image_copy_to_memory_params_t(Structure):
    pass

struct__ze_command_list_append_image_copy_to_memory_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_image_copy_to_memory_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('phSrcImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('ppSrcRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_image_region_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_image_copy_to_memory_params_t = struct__ze_command_list_append_image_copy_to_memory_params_t
ze_pfnCommandListAppendImageCopyToMemoryCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_to_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_image_copy_from_memory_params_t(Structure):
    pass

struct__ze_command_list_append_image_copy_from_memory_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_image_copy_from_memory_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phDstImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
    ('psrcptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppDstRegion', ctypes.POINTER(ctypes.POINTER(struct__ze_image_region_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_image_copy_from_memory_params_t = struct__ze_command_list_append_image_copy_from_memory_params_t
ze_pfnCommandListAppendImageCopyFromMemoryCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_from_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_memory_prefetch_params_t(Structure):
    pass

struct__ze_command_list_append_memory_prefetch_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_memory_prefetch_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
]

ze_command_list_append_memory_prefetch_params_t = struct__ze_command_list_append_memory_prefetch_params_t
ze_pfnCommandListAppendMemoryPrefetchCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_prefetch_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_mem_advise_params_t(Structure):
    pass

struct__ze_command_list_append_mem_advise_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_mem_advise_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('padvice', ctypes.POINTER(_ze_memory_advice_t)),
]

ze_command_list_append_mem_advise_params_t = struct__ze_command_list_append_mem_advise_params_t
ze_pfnCommandListAppendMemAdviseCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_mem_advise_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_signal_event_params_t(Structure):
    pass

struct__ze_command_list_append_signal_event_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_signal_event_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_command_list_append_signal_event_params_t = struct__ze_command_list_append_signal_event_params_t
ze_pfnCommandListAppendSignalEventCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_signal_event_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_wait_on_events_params_t(Structure):
    pass

struct__ze_command_list_append_wait_on_events_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_wait_on_events_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pnumEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_wait_on_events_params_t = struct__ze_command_list_append_wait_on_events_params_t
ze_pfnCommandListAppendWaitOnEventsCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_wait_on_events_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_event_reset_params_t(Structure):
    pass

struct__ze_command_list_append_event_reset_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_event_reset_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_command_list_append_event_reset_params_t = struct__ze_command_list_append_event_reset_params_t
ze_pfnCommandListAppendEventResetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_event_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_query_kernel_timestamps_params_t(Structure):
    pass

struct__ze_command_list_append_query_kernel_timestamps_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_query_kernel_timestamps_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pnumEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppOffsets', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_query_kernel_timestamps_params_t = struct__ze_command_list_append_query_kernel_timestamps_params_t
ze_pfnCommandListAppendQueryKernelTimestampsCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_query_kernel_timestamps_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_launch_kernel_params_t(Structure):
    pass

struct__ze_command_list_append_launch_kernel_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_launch_kernel_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppLaunchFuncArgs', ctypes.POINTER(ctypes.POINTER(struct__ze_group_count_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_launch_kernel_params_t = struct__ze_command_list_append_launch_kernel_params_t
ze_pfnCommandListAppendLaunchKernelCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_kernel_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_launch_cooperative_kernel_params_t(Structure):
    pass

struct__ze_command_list_append_launch_cooperative_kernel_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_launch_cooperative_kernel_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppLaunchFuncArgs', ctypes.POINTER(ctypes.POINTER(struct__ze_group_count_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_launch_cooperative_kernel_params_t = struct__ze_command_list_append_launch_cooperative_kernel_params_t
ze_pfnCommandListAppendLaunchCooperativeKernelCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_cooperative_kernel_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_launch_kernel_indirect_params_t(Structure):
    pass

struct__ze_command_list_append_launch_kernel_indirect_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_launch_kernel_indirect_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppLaunchArgumentsBuffer', ctypes.POINTER(ctypes.POINTER(struct__ze_group_count_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_launch_kernel_indirect_params_t = struct__ze_command_list_append_launch_kernel_indirect_params_t
ze_pfnCommandListAppendLaunchKernelIndirectCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_kernel_indirect_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t(Structure):
    pass

struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t._pack_ = 1 # source:False
struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t._fields_ = [
    ('phCommandList', ctypes.POINTER(ctypes.POINTER(struct__ze_command_list_handle_t))),
    ('pnumKernels', ctypes.POINTER(ctypes.c_uint32)),
    ('pphKernels', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t)))),
    ('ppCountBuffer', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppLaunchArgumentsBuffer', ctypes.POINTER(ctypes.POINTER(struct__ze_group_count_t))),
    ('phSignalEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pnumWaitEvents', ctypes.POINTER(ctypes.c_uint32)),
    ('pphWaitEvents', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_command_list_append_launch_multiple_kernels_indirect_params_t = struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t
ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_command_list_callbacks_t(Structure):
    pass

struct__ze_command_list_callbacks_t._pack_ = 1 # source:False
struct__ze_command_list_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCreateImmediateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_create_immediate_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCloseCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_close_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnResetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendWriteGlobalTimestampCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_write_global_timestamp_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendBarrierCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryRangesBarrierCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_ranges_barrier_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryCopyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryFillCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_fill_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryCopyRegionCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_region_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryCopyFromContextCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_copy_from_context_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendImageCopyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendImageCopyRegionCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_region_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendImageCopyToMemoryCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_to_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendImageCopyFromMemoryCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_image_copy_from_memory_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemoryPrefetchCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_memory_prefetch_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendMemAdviseCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_mem_advise_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendSignalEventCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_signal_event_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendWaitOnEventsCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_wait_on_events_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendEventResetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_event_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendQueryKernelTimestampsCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_query_kernel_timestamps_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendLaunchKernelCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_kernel_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendLaunchCooperativeKernelCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_cooperative_kernel_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendLaunchKernelIndirectCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_kernel_indirect_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAppendLaunchMultipleKernelsIndirectCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_command_list_callbacks_t = struct__ze_command_list_callbacks_t
class struct__ze_image_get_properties_params_t(Structure):
    pass

struct__ze_image_get_properties_params_t._pack_ = 1 # source:False
struct__ze_image_get_properties_params_t._fields_ = [
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_image_desc_t))),
    ('ppImageProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_image_properties_t))),
]

ze_image_get_properties_params_t = struct__ze_image_get_properties_params_t
ze_pfnImageGetPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_image_create_params_t(Structure):
    pass

struct__ze_image_create_params_t._pack_ = 1 # source:False
struct__ze_image_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_image_desc_t))),
    ('pphImage', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t)))),
]

ze_image_create_params_t = struct__ze_image_create_params_t
ze_pfnImageCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_image_destroy_params_t(Structure):
    pass

struct__ze_image_destroy_params_t._pack_ = 1 # source:False
struct__ze_image_destroy_params_t._fields_ = [
    ('phImage', ctypes.POINTER(ctypes.POINTER(struct__ze_image_handle_t))),
]

ze_image_destroy_params_t = struct__ze_image_destroy_params_t
ze_pfnImageDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_image_callbacks_t(Structure):
    pass

struct__ze_image_callbacks_t._pack_ = 1 # source:False
struct__ze_image_callbacks_t._fields_ = [
    ('pfnGetPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_image_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_image_callbacks_t = struct__ze_image_callbacks_t
class struct__ze_fence_create_params_t(Structure):
    pass

struct__ze_fence_create_params_t._pack_ = 1 # source:False
struct__ze_fence_create_params_t._fields_ = [
    ('phCommandQueue', ctypes.POINTER(ctypes.POINTER(struct__ze_command_queue_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_desc_t))),
    ('pphFence', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t)))),
]

ze_fence_create_params_t = struct__ze_fence_create_params_t
ze_pfnFenceCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_fence_destroy_params_t(Structure):
    pass

struct__ze_fence_destroy_params_t._pack_ = 1 # source:False
struct__ze_fence_destroy_params_t._fields_ = [
    ('phFence', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))),
]

ze_fence_destroy_params_t = struct__ze_fence_destroy_params_t
ze_pfnFenceDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_fence_host_synchronize_params_t(Structure):
    pass

struct__ze_fence_host_synchronize_params_t._pack_ = 1 # source:False
struct__ze_fence_host_synchronize_params_t._fields_ = [
    ('phFence', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))),
    ('ptimeout', ctypes.POINTER(ctypes.c_uint64)),
]

ze_fence_host_synchronize_params_t = struct__ze_fence_host_synchronize_params_t
ze_pfnFenceHostSynchronizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_host_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_fence_query_status_params_t(Structure):
    pass

struct__ze_fence_query_status_params_t._pack_ = 1 # source:False
struct__ze_fence_query_status_params_t._fields_ = [
    ('phFence', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))),
]

ze_fence_query_status_params_t = struct__ze_fence_query_status_params_t
ze_pfnFenceQueryStatusCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_query_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_fence_reset_params_t(Structure):
    pass

struct__ze_fence_reset_params_t._pack_ = 1 # source:False
struct__ze_fence_reset_params_t._fields_ = [
    ('phFence', ctypes.POINTER(ctypes.POINTER(struct__ze_fence_handle_t))),
]

ze_fence_reset_params_t = struct__ze_fence_reset_params_t
ze_pfnFenceResetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_fence_callbacks_t(Structure):
    pass

struct__ze_fence_callbacks_t._pack_ = 1 # source:False
struct__ze_fence_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnHostSynchronizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_host_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnQueryStatusCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_query_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnResetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_fence_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_fence_callbacks_t = struct__ze_fence_callbacks_t
class struct__ze_event_pool_create_params_t(Structure):
    pass

struct__ze_event_pool_create_params_t._pack_ = 1 # source:False
struct__ze_event_pool_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_desc_t))),
    ('pnumDevices', ctypes.POINTER(ctypes.c_uint32)),
    ('pphDevices', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)))),
    ('pphEventPool', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t)))),
]

ze_event_pool_create_params_t = struct__ze_event_pool_create_params_t
ze_pfnEventPoolCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_pool_destroy_params_t(Structure):
    pass

struct__ze_event_pool_destroy_params_t._pack_ = 1 # source:False
struct__ze_event_pool_destroy_params_t._fields_ = [
    ('phEventPool', ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))),
]

ze_event_pool_destroy_params_t = struct__ze_event_pool_destroy_params_t
ze_pfnEventPoolDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_pool_get_ipc_handle_params_t(Structure):
    pass

struct__ze_event_pool_get_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_event_pool_get_ipc_handle_params_t._fields_ = [
    ('phEventPool', ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))),
    ('pphIpc', ctypes.POINTER(ctypes.POINTER(struct__ze_ipc_event_pool_handle_t))),
]

ze_event_pool_get_ipc_handle_params_t = struct__ze_event_pool_get_ipc_handle_params_t
ze_pfnEventPoolGetIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_get_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_pool_open_ipc_handle_params_t(Structure):
    pass

struct__ze_event_pool_open_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_event_pool_open_ipc_handle_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phIpc', ctypes.POINTER(struct__ze_ipc_event_pool_handle_t)),
    ('pphEventPool', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t)))),
]

ze_event_pool_open_ipc_handle_params_t = struct__ze_event_pool_open_ipc_handle_params_t
ze_pfnEventPoolOpenIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_open_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_pool_close_ipc_handle_params_t(Structure):
    pass

struct__ze_event_pool_close_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_event_pool_close_ipc_handle_params_t._fields_ = [
    ('phEventPool', ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))),
]

ze_event_pool_close_ipc_handle_params_t = struct__ze_event_pool_close_ipc_handle_params_t
ze_pfnEventPoolCloseIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_close_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_pool_callbacks_t(Structure):
    pass

struct__ze_event_pool_callbacks_t._pack_ = 1 # source:False
struct__ze_event_pool_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_get_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnOpenIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_open_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCloseIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_pool_close_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_event_pool_callbacks_t = struct__ze_event_pool_callbacks_t
class struct__ze_event_create_params_t(Structure):
    pass

struct__ze_event_create_params_t._pack_ = 1 # source:False
struct__ze_event_create_params_t._fields_ = [
    ('phEventPool', ctypes.POINTER(ctypes.POINTER(struct__ze_event_pool_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_event_desc_t))),
    ('pphEvent', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t)))),
]

ze_event_create_params_t = struct__ze_event_create_params_t
ze_pfnEventCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_destroy_params_t(Structure):
    pass

struct__ze_event_destroy_params_t._pack_ = 1 # source:False
struct__ze_event_destroy_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_event_destroy_params_t = struct__ze_event_destroy_params_t
ze_pfnEventDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_host_signal_params_t(Structure):
    pass

struct__ze_event_host_signal_params_t._pack_ = 1 # source:False
struct__ze_event_host_signal_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_event_host_signal_params_t = struct__ze_event_host_signal_params_t
ze_pfnEventHostSignalCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_signal_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_host_synchronize_params_t(Structure):
    pass

struct__ze_event_host_synchronize_params_t._pack_ = 1 # source:False
struct__ze_event_host_synchronize_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('ptimeout', ctypes.POINTER(ctypes.c_uint64)),
]

ze_event_host_synchronize_params_t = struct__ze_event_host_synchronize_params_t
ze_pfnEventHostSynchronizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_query_status_params_t(Structure):
    pass

struct__ze_event_query_status_params_t._pack_ = 1 # source:False
struct__ze_event_query_status_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_event_query_status_params_t = struct__ze_event_query_status_params_t
ze_pfnEventQueryStatusCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_query_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_host_reset_params_t(Structure):
    pass

struct__ze_event_host_reset_params_t._pack_ = 1 # source:False
struct__ze_event_host_reset_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
]

ze_event_host_reset_params_t = struct__ze_event_host_reset_params_t
ze_pfnEventHostResetCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_query_kernel_timestamp_params_t(Structure):
    pass

struct__ze_event_query_kernel_timestamp_params_t._pack_ = 1 # source:False
struct__ze_event_query_kernel_timestamp_params_t._fields_ = [
    ('phEvent', ctypes.POINTER(ctypes.POINTER(struct__ze_event_handle_t))),
    ('pdstptr', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_timestamp_result_t))),
]

ze_event_query_kernel_timestamp_params_t = struct__ze_event_query_kernel_timestamp_params_t
ze_pfnEventQueryKernelTimestampCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_query_kernel_timestamp_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_event_callbacks_t(Structure):
    pass

struct__ze_event_callbacks_t._pack_ = 1 # source:False
struct__ze_event_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnHostSignalCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_signal_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnHostSynchronizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_synchronize_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnQueryStatusCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_query_status_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnHostResetCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_host_reset_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnQueryKernelTimestampCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_event_query_kernel_timestamp_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_event_callbacks_t = struct__ze_event_callbacks_t
class struct__ze_module_create_params_t(Structure):
    pass

struct__ze_module_create_params_t._pack_ = 1 # source:False
struct__ze_module_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_module_desc_t))),
    ('pphModule', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t)))),
    ('pphBuildLog', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t)))),
]

ze_module_create_params_t = struct__ze_module_create_params_t
ze_pfnModuleCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_destroy_params_t(Structure):
    pass

struct__ze_module_destroy_params_t._pack_ = 1 # source:False
struct__ze_module_destroy_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
]

ze_module_destroy_params_t = struct__ze_module_destroy_params_t
ze_pfnModuleDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_dynamic_link_params_t(Structure):
    pass

struct__ze_module_dynamic_link_params_t._pack_ = 1 # source:False
struct__ze_module_dynamic_link_params_t._fields_ = [
    ('pnumModules', ctypes.POINTER(ctypes.c_uint32)),
    ('pphModules', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t)))),
    ('pphLinkLog', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t)))),
]

ze_module_dynamic_link_params_t = struct__ze_module_dynamic_link_params_t
ze_pfnModuleDynamicLinkCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_dynamic_link_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_get_native_binary_params_t(Structure):
    pass

struct__ze_module_get_native_binary_params_t._pack_ = 1 # source:False
struct__ze_module_get_native_binary_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('ppModuleNativeBinary', ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))),
]

ze_module_get_native_binary_params_t = struct__ze_module_get_native_binary_params_t
ze_pfnModuleGetNativeBinaryCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_native_binary_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_get_global_pointer_params_t(Structure):
    pass

struct__ze_module_get_global_pointer_params_t._pack_ = 1 # source:False
struct__ze_module_get_global_pointer_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('ppGlobalName', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_module_get_global_pointer_params_t = struct__ze_module_get_global_pointer_params_t
ze_pfnModuleGetGlobalPointerCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_global_pointer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_get_kernel_names_params_t(Structure):
    pass

struct__ze_module_get_kernel_names_params_t._pack_ = 1 # source:False
struct__ze_module_get_kernel_names_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('ppCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppNames', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
]

ze_module_get_kernel_names_params_t = struct__ze_module_get_kernel_names_params_t
ze_pfnModuleGetKernelNamesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_kernel_names_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_get_properties_params_t(Structure):
    pass

struct__ze_module_get_properties_params_t._pack_ = 1 # source:False
struct__ze_module_get_properties_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('ppModuleProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_module_properties_t))),
]

ze_module_get_properties_params_t = struct__ze_module_get_properties_params_t
ze_pfnModuleGetPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_get_function_pointer_params_t(Structure):
    pass

struct__ze_module_get_function_pointer_params_t._pack_ = 1 # source:False
struct__ze_module_get_function_pointer_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('ppFunctionName', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ('ppfnFunction', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_module_get_function_pointer_params_t = struct__ze_module_get_function_pointer_params_t
ze_pfnModuleGetFunctionPointerCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_function_pointer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_callbacks_t(Structure):
    pass

struct__ze_module_callbacks_t._pack_ = 1 # source:False
struct__ze_module_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDynamicLinkCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_dynamic_link_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetNativeBinaryCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_native_binary_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetGlobalPointerCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_global_pointer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetKernelNamesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_kernel_names_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetFunctionPointerCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_get_function_pointer_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_module_callbacks_t = struct__ze_module_callbacks_t
class struct__ze_module_build_log_destroy_params_t(Structure):
    pass

struct__ze_module_build_log_destroy_params_t._pack_ = 1 # source:False
struct__ze_module_build_log_destroy_params_t._fields_ = [
    ('phModuleBuildLog', ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t))),
]

ze_module_build_log_destroy_params_t = struct__ze_module_build_log_destroy_params_t
ze_pfnModuleBuildLogDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_build_log_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_build_log_get_string_params_t(Structure):
    pass

struct__ze_module_build_log_get_string_params_t._pack_ = 1 # source:False
struct__ze_module_build_log_get_string_params_t._fields_ = [
    ('phModuleBuildLog', ctypes.POINTER(ctypes.POINTER(struct__ze_module_build_log_handle_t))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('ppBuildLog', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
]

ze_module_build_log_get_string_params_t = struct__ze_module_build_log_get_string_params_t
ze_pfnModuleBuildLogGetStringCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_build_log_get_string_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_module_build_log_callbacks_t(Structure):
    pass

struct__ze_module_build_log_callbacks_t._pack_ = 1 # source:False
struct__ze_module_build_log_callbacks_t._fields_ = [
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_build_log_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetStringCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_module_build_log_get_string_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_module_build_log_callbacks_t = struct__ze_module_build_log_callbacks_t
class struct__ze_kernel_create_params_t(Structure):
    pass

struct__ze_kernel_create_params_t._pack_ = 1 # source:False
struct__ze_kernel_create_params_t._fields_ = [
    ('phModule', ctypes.POINTER(ctypes.POINTER(struct__ze_module_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_desc_t))),
    ('pphKernel', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t)))),
]

ze_kernel_create_params_t = struct__ze_kernel_create_params_t
ze_pfnKernelCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_destroy_params_t(Structure):
    pass

struct__ze_kernel_destroy_params_t._pack_ = 1 # source:False
struct__ze_kernel_destroy_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
]

ze_kernel_destroy_params_t = struct__ze_kernel_destroy_params_t
ze_pfnKernelDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_set_cache_config_params_t(Structure):
    pass

struct__ze_kernel_set_cache_config_params_t._pack_ = 1 # source:False
struct__ze_kernel_set_cache_config_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('pflags', ctypes.POINTER(ctypes.c_uint32)),
]

ze_kernel_set_cache_config_params_t = struct__ze_kernel_set_cache_config_params_t
ze_pfnKernelSetCacheConfigCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_cache_config_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_set_group_size_params_t(Structure):
    pass

struct__ze_kernel_set_group_size_params_t._pack_ = 1 # source:False
struct__ze_kernel_set_group_size_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('pgroupSizeX', ctypes.POINTER(ctypes.c_uint32)),
    ('pgroupSizeY', ctypes.POINTER(ctypes.c_uint32)),
    ('pgroupSizeZ', ctypes.POINTER(ctypes.c_uint32)),
]

ze_kernel_set_group_size_params_t = struct__ze_kernel_set_group_size_params_t
ze_pfnKernelSetGroupSizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_group_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_suggest_group_size_params_t(Structure):
    pass

struct__ze_kernel_suggest_group_size_params_t._pack_ = 1 # source:False
struct__ze_kernel_suggest_group_size_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('pglobalSizeX', ctypes.POINTER(ctypes.c_uint32)),
    ('pglobalSizeY', ctypes.POINTER(ctypes.c_uint32)),
    ('pglobalSizeZ', ctypes.POINTER(ctypes.c_uint32)),
    ('pgroupSizeX', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('pgroupSizeY', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('pgroupSizeZ', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
]

ze_kernel_suggest_group_size_params_t = struct__ze_kernel_suggest_group_size_params_t
ze_pfnKernelSuggestGroupSizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_suggest_group_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_suggest_max_cooperative_group_count_params_t(Structure):
    pass

struct__ze_kernel_suggest_max_cooperative_group_count_params_t._pack_ = 1 # source:False
struct__ze_kernel_suggest_max_cooperative_group_count_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ptotalGroupCount', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
]

ze_kernel_suggest_max_cooperative_group_count_params_t = struct__ze_kernel_suggest_max_cooperative_group_count_params_t
ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_suggest_max_cooperative_group_count_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_set_argument_value_params_t(Structure):
    pass

struct__ze_kernel_set_argument_value_params_t._pack_ = 1 # source:False
struct__ze_kernel_set_argument_value_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('pargIndex', ctypes.POINTER(ctypes.c_uint32)),
    ('pargSize', ctypes.POINTER(ctypes.c_uint64)),
    ('ppArgValue', ctypes.POINTER(ctypes.POINTER(None))),
]

ze_kernel_set_argument_value_params_t = struct__ze_kernel_set_argument_value_params_t
ze_pfnKernelSetArgumentValueCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_argument_value_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_set_indirect_access_params_t(Structure):
    pass

struct__ze_kernel_set_indirect_access_params_t._pack_ = 1 # source:False
struct__ze_kernel_set_indirect_access_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('pflags', ctypes.POINTER(ctypes.c_uint32)),
]

ze_kernel_set_indirect_access_params_t = struct__ze_kernel_set_indirect_access_params_t
ze_pfnKernelSetIndirectAccessCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_indirect_access_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_get_indirect_access_params_t(Structure):
    pass

struct__ze_kernel_get_indirect_access_params_t._pack_ = 1 # source:False
struct__ze_kernel_get_indirect_access_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppFlags', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
]

ze_kernel_get_indirect_access_params_t = struct__ze_kernel_get_indirect_access_params_t
ze_pfnKernelGetIndirectAccessCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_indirect_access_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_get_source_attributes_params_t(Structure):
    pass

struct__ze_kernel_get_source_attributes_params_t._pack_ = 1 # source:False
struct__ze_kernel_get_source_attributes_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
    ('ppString', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
]

ze_kernel_get_source_attributes_params_t = struct__ze_kernel_get_source_attributes_params_t
ze_pfnKernelGetSourceAttributesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_source_attributes_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_get_properties_params_t(Structure):
    pass

struct__ze_kernel_get_properties_params_t._pack_ = 1 # source:False
struct__ze_kernel_get_properties_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppKernelProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_properties_t))),
]

ze_kernel_get_properties_params_t = struct__ze_kernel_get_properties_params_t
ze_pfnKernelGetPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_get_name_params_t(Structure):
    pass

struct__ze_kernel_get_name_params_t._pack_ = 1 # source:False
struct__ze_kernel_get_name_params_t._fields_ = [
    ('phKernel', ctypes.POINTER(ctypes.POINTER(struct__ze_kernel_handle_t))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
    ('ppName', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
]

ze_kernel_get_name_params_t = struct__ze_kernel_get_name_params_t
ze_pfnKernelGetNameCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_name_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_kernel_callbacks_t(Structure):
    pass

struct__ze_kernel_callbacks_t._pack_ = 1 # source:False
struct__ze_kernel_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSetCacheConfigCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_cache_config_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSetGroupSizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_group_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSuggestGroupSizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_suggest_group_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSuggestMaxCooperativeGroupCountCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_suggest_max_cooperative_group_count_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSetArgumentValueCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_argument_value_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSetIndirectAccessCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_set_indirect_access_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetIndirectAccessCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_indirect_access_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetSourceAttributesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_source_attributes_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetNameCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_kernel_get_name_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_kernel_callbacks_t = struct__ze_kernel_callbacks_t
class struct__ze_sampler_create_params_t(Structure):
    pass

struct__ze_sampler_create_params_t._pack_ = 1 # source:False
struct__ze_sampler_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_sampler_desc_t))),
    ('pphSampler', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_sampler_handle_t)))),
]

ze_sampler_create_params_t = struct__ze_sampler_create_params_t
ze_pfnSamplerCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_sampler_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_sampler_destroy_params_t(Structure):
    pass

struct__ze_sampler_destroy_params_t._pack_ = 1 # source:False
struct__ze_sampler_destroy_params_t._fields_ = [
    ('phSampler', ctypes.POINTER(ctypes.POINTER(struct__ze_sampler_handle_t))),
]

ze_sampler_destroy_params_t = struct__ze_sampler_destroy_params_t
ze_pfnSamplerDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_sampler_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_sampler_callbacks_t(Structure):
    pass

struct__ze_sampler_callbacks_t._pack_ = 1 # source:False
struct__ze_sampler_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_sampler_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_sampler_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_sampler_callbacks_t = struct__ze_sampler_callbacks_t
class struct__ze_physical_mem_create_params_t(Structure):
    pass

struct__ze_physical_mem_create_params_t._pack_ = 1 # source:False
struct__ze_physical_mem_create_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('pdesc', ctypes.POINTER(ctypes.POINTER(struct__ze_physical_mem_desc_t))),
    ('pphPhysicalMemory', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_physical_mem_handle_t)))),
]

ze_physical_mem_create_params_t = struct__ze_physical_mem_create_params_t
ze_pfnPhysicalMemCreateCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_physical_mem_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_physical_mem_destroy_params_t(Structure):
    pass

struct__ze_physical_mem_destroy_params_t._pack_ = 1 # source:False
struct__ze_physical_mem_destroy_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phPhysicalMemory', ctypes.POINTER(ctypes.POINTER(struct__ze_physical_mem_handle_t))),
]

ze_physical_mem_destroy_params_t = struct__ze_physical_mem_destroy_params_t
ze_pfnPhysicalMemDestroyCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_physical_mem_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_physical_mem_callbacks_t(Structure):
    pass

struct__ze_physical_mem_callbacks_t._pack_ = 1 # source:False
struct__ze_physical_mem_callbacks_t._fields_ = [
    ('pfnCreateCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_physical_mem_create_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnDestroyCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_physical_mem_destroy_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_physical_mem_callbacks_t = struct__ze_physical_mem_callbacks_t
class struct__ze_mem_alloc_shared_params_t(Structure):
    pass

struct__ze_mem_alloc_shared_params_t._pack_ = 1 # source:False
struct__ze_mem_alloc_shared_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pdevice_desc', ctypes.POINTER(ctypes.POINTER(struct__ze_device_mem_alloc_desc_t))),
    ('phost_desc', ctypes.POINTER(ctypes.POINTER(struct__ze_host_mem_alloc_desc_t))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('palignment', ctypes.POINTER(ctypes.c_uint64)),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_mem_alloc_shared_params_t = struct__ze_mem_alloc_shared_params_t
ze_pfnMemAllocSharedCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_shared_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_alloc_device_params_t(Structure):
    pass

struct__ze_mem_alloc_device_params_t._pack_ = 1 # source:False
struct__ze_mem_alloc_device_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pdevice_desc', ctypes.POINTER(ctypes.POINTER(struct__ze_device_mem_alloc_desc_t))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('palignment', ctypes.POINTER(ctypes.c_uint64)),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_mem_alloc_device_params_t = struct__ze_mem_alloc_device_params_t
ze_pfnMemAllocDeviceCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_device_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_alloc_host_params_t(Structure):
    pass

struct__ze_mem_alloc_host_params_t._pack_ = 1 # source:False
struct__ze_mem_alloc_host_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phost_desc', ctypes.POINTER(ctypes.POINTER(struct__ze_host_mem_alloc_desc_t))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('palignment', ctypes.POINTER(ctypes.c_uint64)),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_mem_alloc_host_params_t = struct__ze_mem_alloc_host_params_t
ze_pfnMemAllocHostCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_host_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_free_params_t(Structure):
    pass

struct__ze_mem_free_params_t._pack_ = 1 # source:False
struct__ze_mem_free_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
]

ze_mem_free_params_t = struct__ze_mem_free_params_t
ze_pfnMemFreeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_free_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_get_alloc_properties_params_t(Structure):
    pass

struct__ze_mem_get_alloc_properties_params_t._pack_ = 1 # source:False
struct__ze_mem_get_alloc_properties_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppMemAllocProperties', ctypes.POINTER(ctypes.POINTER(struct__ze_memory_allocation_properties_t))),
    ('pphDevice', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t)))),
]

ze_mem_get_alloc_properties_params_t = struct__ze_mem_get_alloc_properties_params_t
ze_pfnMemGetAllocPropertiesCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_alloc_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_get_address_range_params_t(Structure):
    pass

struct__ze_mem_get_address_range_params_t._pack_ = 1 # source:False
struct__ze_mem_get_address_range_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppBase', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
    ('ppSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
]

ze_mem_get_address_range_params_t = struct__ze_mem_get_address_range_params_t
ze_pfnMemGetAddressRangeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_address_range_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_get_ipc_handle_params_t(Structure):
    pass

struct__ze_mem_get_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_mem_get_ipc_handle_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('ppIpcHandle', ctypes.POINTER(ctypes.POINTER(struct__ze_ipc_mem_handle_t))),
]

ze_mem_get_ipc_handle_params_t = struct__ze_mem_get_ipc_handle_params_t
ze_pfnMemGetIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_open_ipc_handle_params_t(Structure):
    pass

struct__ze_mem_open_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_mem_open_ipc_handle_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('phandle', ctypes.POINTER(struct__ze_ipc_mem_handle_t)),
    ('pflags', ctypes.POINTER(ctypes.c_uint32)),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_mem_open_ipc_handle_params_t = struct__ze_mem_open_ipc_handle_params_t
ze_pfnMemOpenIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_open_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_close_ipc_handle_params_t(Structure):
    pass

struct__ze_mem_close_ipc_handle_params_t._pack_ = 1 # source:False
struct__ze_mem_close_ipc_handle_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
]

ze_mem_close_ipc_handle_params_t = struct__ze_mem_close_ipc_handle_params_t
ze_pfnMemCloseIpcHandleCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_close_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_mem_callbacks_t(Structure):
    pass

struct__ze_mem_callbacks_t._pack_ = 1 # source:False
struct__ze_mem_callbacks_t._fields_ = [
    ('pfnAllocSharedCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_shared_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAllocDeviceCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_device_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnAllocHostCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_alloc_host_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnFreeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_free_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetAllocPropertiesCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_alloc_properties_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetAddressRangeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_address_range_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_get_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnOpenIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_open_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnCloseIpcHandleCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_mem_close_ipc_handle_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_mem_callbacks_t = struct__ze_mem_callbacks_t
class struct__ze_virtual_mem_reserve_params_t(Structure):
    pass

struct__ze_virtual_mem_reserve_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_reserve_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('ppStart', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('ppptr', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_virtual_mem_reserve_params_t = struct__ze_virtual_mem_reserve_params_t
ze_pfnVirtualMemReserveCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_reserve_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_free_params_t(Structure):
    pass

struct__ze_virtual_mem_free_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_free_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
]

ze_virtual_mem_free_params_t = struct__ze_virtual_mem_free_params_t
ze_pfnVirtualMemFreeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_free_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_query_page_size_params_t(Structure):
    pass

struct__ze_virtual_mem_query_page_size_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_query_page_size_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('phDevice', ctypes.POINTER(ctypes.POINTER(struct__ze_device_handle_t))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('ppagesize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
]

ze_virtual_mem_query_page_size_params_t = struct__ze_virtual_mem_query_page_size_params_t
ze_pfnVirtualMemQueryPageSizeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_query_page_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_map_params_t(Structure):
    pass

struct__ze_virtual_mem_map_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_map_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('phPhysicalMemory', ctypes.POINTER(ctypes.POINTER(struct__ze_physical_mem_handle_t))),
    ('poffset', ctypes.POINTER(ctypes.c_uint64)),
    ('paccess', ctypes.POINTER(_ze_memory_access_attribute_t)),
]

ze_virtual_mem_map_params_t = struct__ze_virtual_mem_map_params_t
ze_pfnVirtualMemMapCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_map_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_unmap_params_t(Structure):
    pass

struct__ze_virtual_mem_unmap_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_unmap_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
]

ze_virtual_mem_unmap_params_t = struct__ze_virtual_mem_unmap_params_t
ze_pfnVirtualMemUnmapCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_unmap_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_set_access_attribute_params_t(Structure):
    pass

struct__ze_virtual_mem_set_access_attribute_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_set_access_attribute_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('paccess', ctypes.POINTER(_ze_memory_access_attribute_t)),
]

ze_virtual_mem_set_access_attribute_params_t = struct__ze_virtual_mem_set_access_attribute_params_t
ze_pfnVirtualMemSetAccessAttributeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_set_access_attribute_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_get_access_attribute_params_t(Structure):
    pass

struct__ze_virtual_mem_get_access_attribute_params_t._pack_ = 1 # source:False
struct__ze_virtual_mem_get_access_attribute_params_t._fields_ = [
    ('phContext', ctypes.POINTER(ctypes.POINTER(struct__ze_context_handle_t))),
    ('pptr', ctypes.POINTER(ctypes.POINTER(None))),
    ('psize', ctypes.POINTER(ctypes.c_uint64)),
    ('paccess', ctypes.POINTER(ctypes.POINTER(_ze_memory_access_attribute_t))),
    ('poutSize', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))),
]

ze_virtual_mem_get_access_attribute_params_t = struct__ze_virtual_mem_get_access_attribute_params_t
ze_pfnVirtualMemGetAccessAttributeCb_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_get_access_attribute_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))
class struct__ze_virtual_mem_callbacks_t(Structure):
    pass

struct__ze_virtual_mem_callbacks_t._pack_ = 1 # source:False
struct__ze_virtual_mem_callbacks_t._fields_ = [
    ('pfnReserveCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_reserve_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnFreeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_free_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnQueryPageSizeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_query_page_size_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnMapCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_map_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnUnmapCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_unmap_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnSetAccessAttributeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_set_access_attribute_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
    ('pfnGetAccessAttributeCb', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__ze_virtual_mem_get_access_attribute_params_t), _ze_result_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None)))),
]

ze_virtual_mem_callbacks_t = struct__ze_virtual_mem_callbacks_t
class struct__ze_callbacks_t(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('Global', ze_global_callbacks_t),
    ('Driver', ze_driver_callbacks_t),
    ('Device', ze_device_callbacks_t),
    ('Context', ze_context_callbacks_t),
    ('CommandQueue', ze_command_queue_callbacks_t),
    ('CommandList', ze_command_list_callbacks_t),
    ('Fence', ze_fence_callbacks_t),
    ('EventPool', ze_event_pool_callbacks_t),
    ('Event', ze_event_callbacks_t),
    ('Image', ze_image_callbacks_t),
    ('Module', ze_module_callbacks_t),
    ('ModuleBuildLog', ze_module_build_log_callbacks_t),
    ('Kernel', ze_kernel_callbacks_t),
    ('Sampler', ze_sampler_callbacks_t),
    ('PhysicalMem', ze_physical_mem_callbacks_t),
    ('Mem', ze_mem_callbacks_t),
    ('VirtualMem', ze_virtual_mem_callbacks_t),
     ]

ze_callbacks_t = struct__ze_callbacks_t
class struct__zel_version(Structure):
    pass

struct__zel_version._pack_ = 1 # source:False
struct__zel_version._fields_ = [
    ('major', ctypes.c_int32),
    ('minor', ctypes.c_int32),
    ('patch', ctypes.c_int32),
]

zel_version_t = struct__zel_version
class struct_zel_component_version(Structure):
    pass

struct_zel_component_version._pack_ = 1 # source:False
struct_zel_component_version._fields_ = [
    ('component_name', ctypes.c_char * 64),
    ('spec_version', ze_api_version_t),
    ('component_lib_version', zel_version_t),
]

zel_component_version_t = struct_zel_component_version
try:
    zelLoaderGetVersions = _libraries['libze_loader.so'].zelLoaderGetVersions
    zelLoaderGetVersions.restype = ze_result_t
    zelLoaderGetVersions.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_zel_component_version)]
except AttributeError:
    pass

# values for enumeration '_zel_handle_type_t'
_zel_handle_type_t__enumvalues = {
    0: 'ZEL_HANDLE_DRIVER',
    1: 'ZEL_HANDLE_DEVICE',
    2: 'ZEL_HANDLE_CONTEXT',
    3: 'ZEL_HANDLE_COMMAND_QUEUE',
    4: 'ZEL_HANDLE_COMMAND_LIST',
    5: 'ZEL_HANDLE_FENCE',
    6: 'ZEL_HANDLE_EVENT_POOL',
    7: 'ZEL_HANDLE_EVENT',
    8: 'ZEL_HANDLE_IMAGE',
    9: 'ZEL_HANDLE_MODULE',
    10: 'ZEL_HANDLE_MODULE_BUILD_LOG',
    11: 'ZEL_HANDLE_KERNEL',
    12: 'ZEL_HANDLE_SAMPLER',
    13: 'ZEL_HANDLE_PHYSICAL_MEM',
}
ZEL_HANDLE_DRIVER = 0
ZEL_HANDLE_DEVICE = 1
ZEL_HANDLE_CONTEXT = 2
ZEL_HANDLE_COMMAND_QUEUE = 3
ZEL_HANDLE_COMMAND_LIST = 4
ZEL_HANDLE_FENCE = 5
ZEL_HANDLE_EVENT_POOL = 6
ZEL_HANDLE_EVENT = 7
ZEL_HANDLE_IMAGE = 8
ZEL_HANDLE_MODULE = 9
ZEL_HANDLE_MODULE_BUILD_LOG = 10
ZEL_HANDLE_KERNEL = 11
ZEL_HANDLE_SAMPLER = 12
ZEL_HANDLE_PHYSICAL_MEM = 13
_zel_handle_type_t = ctypes.c_uint32 # enum
zel_handle_type_t = _zel_handle_type_t
zel_handle_type_t__enumvalues = _zel_handle_type_t__enumvalues
try:
    zelLoaderTranslateHandle = _libraries['libze_loader.so'].zelLoaderTranslateHandle
    zelLoaderTranslateHandle.restype = ze_result_t
    zelLoaderTranslateHandle.argtypes = [zel_handle_type_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    zelSetDriverTeardown = _libraries['libze_loader.so'].zelSetDriverTeardown
    zelSetDriverTeardown.restype = ze_result_t
    zelSetDriverTeardown.argtypes = []
except AttributeError:
    pass
__all__ = \
    ['ZEL_HANDLE_COMMAND_LIST', 'ZEL_HANDLE_COMMAND_QUEUE',
    'ZEL_HANDLE_CONTEXT', 'ZEL_HANDLE_DEVICE', 'ZEL_HANDLE_DRIVER',
    'ZEL_HANDLE_EVENT', 'ZEL_HANDLE_EVENT_POOL', 'ZEL_HANDLE_FENCE',
    'ZEL_HANDLE_IMAGE', 'ZEL_HANDLE_KERNEL', 'ZEL_HANDLE_MODULE',
    'ZEL_HANDLE_MODULE_BUILD_LOG', 'ZEL_HANDLE_PHYSICAL_MEM',
    'ZEL_HANDLE_SAMPLER', 'ZE_API_VERSION_1_0', 'ZE_API_VERSION_1_1',
    'ZE_API_VERSION_1_2', 'ZE_API_VERSION_1_3', 'ZE_API_VERSION_1_4',
    'ZE_API_VERSION_1_5', 'ZE_API_VERSION_1_6', 'ZE_API_VERSION_1_7',
    'ZE_API_VERSION_CURRENT', 'ZE_API_VERSION_FORCE_UINT32',
    'ZE_BANDWIDTH_UNIT_BYTES_PER_CLOCK',
    'ZE_BANDWIDTH_UNIT_BYTES_PER_NANOSEC',
    'ZE_BANDWIDTH_UNIT_FORCE_UINT32', 'ZE_BANDWIDTH_UNIT_UNKNOWN',
    'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_1_0',
    'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_CURRENT',
    'ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_FORCE_UINT32',
    'ZE_CACHE_CONFIG_FLAG_FORCE_UINT32',
    'ZE_CACHE_CONFIG_FLAG_LARGE_DATA',
    'ZE_CACHE_CONFIG_FLAG_LARGE_SLM', 'ZE_CACHE_EXT_REGION_DEFAULT',
    'ZE_CACHE_EXT_REGION_FORCE_UINT32',
    'ZE_CACHE_EXT_REGION_NON_RESERVED',
    'ZE_CACHE_EXT_REGION_RESERVED',
    'ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION',
    'ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT',
    'ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION',
    'ZE_CACHE_RESERVATION_EXT_VERSION_1_0',
    'ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT',
    'ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32',
    'ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY',
    'ZE_COMMAND_LIST_FLAG_FORCE_UINT32',
    'ZE_COMMAND_LIST_FLAG_IN_ORDER',
    'ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT',
    'ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING',
    'ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY',
    'ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32',
    'ZE_COMMAND_QUEUE_FLAG_IN_ORDER',
    'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE',
    'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS',
    'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY',
    'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS',
    'ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS',
    'ZE_COMMAND_QUEUE_MODE_DEFAULT',
    'ZE_COMMAND_QUEUE_MODE_FORCE_UINT32',
    'ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS',
    'ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32',
    'ZE_COMMAND_QUEUE_PRIORITY_NORMAL',
    'ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH',
    'ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW',
    'ZE_CONTEXT_FLAG_FORCE_UINT32', 'ZE_CONTEXT_FLAG_TBD',
    'ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE',
    'ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX',
    'ZE_DEVICE_FP_FLAG_DENORM', 'ZE_DEVICE_FP_FLAG_FMA',
    'ZE_DEVICE_FP_FLAG_FORCE_UINT32', 'ZE_DEVICE_FP_FLAG_INF_NAN',
    'ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT',
    'ZE_DEVICE_FP_FLAG_ROUND_TO_INF',
    'ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST',
    'ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO', 'ZE_DEVICE_FP_FLAG_SOFT_FLOAT',
    'ZE_DEVICE_IP_VERSION_VERSION_1_0',
    'ZE_DEVICE_IP_VERSION_VERSION_CURRENT',
    'ZE_DEVICE_IP_VERSION_VERSION_FORCE_UINT32',
    'ZE_DEVICE_LUID_EXT_VERSION_1_0',
    'ZE_DEVICE_LUID_EXT_VERSION_CURRENT',
    'ZE_DEVICE_LUID_EXT_VERSION_FORCE_UINT32',
    'ZE_DEVICE_MEMORY_EXT_TYPE_DDR', 'ZE_DEVICE_MEMORY_EXT_TYPE_DDR2',
    'ZE_DEVICE_MEMORY_EXT_TYPE_DDR3',
    'ZE_DEVICE_MEMORY_EXT_TYPE_DDR4',
    'ZE_DEVICE_MEMORY_EXT_TYPE_DDR5',
    'ZE_DEVICE_MEMORY_EXT_TYPE_FORCE_UINT32',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR4',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5X',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6X',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GDDR7',
    'ZE_DEVICE_MEMORY_EXT_TYPE_GRF', 'ZE_DEVICE_MEMORY_EXT_TYPE_HBM',
    'ZE_DEVICE_MEMORY_EXT_TYPE_HBM2', 'ZE_DEVICE_MEMORY_EXT_TYPE_L1',
    'ZE_DEVICE_MEMORY_EXT_TYPE_L3', 'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR',
    'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR3',
    'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR4',
    'ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR5',
    'ZE_DEVICE_MEMORY_EXT_TYPE_SLM', 'ZE_DEVICE_MEMORY_EXT_TYPE_SRAM',
    'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_1_0',
    'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_CURRENT',
    'ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_FORCE_UINT32',
    'ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD',
    'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED',
    'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT',
    'ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED',
    'ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32',
    'ZE_DEVICE_MODULE_FLAG_DP4A',
    'ZE_DEVICE_MODULE_FLAG_FORCE_UINT32',
    'ZE_DEVICE_MODULE_FLAG_FP16', 'ZE_DEVICE_MODULE_FLAG_FP64',
    'ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS',
    'ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS',
    'ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS',
    'ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_DEVICE_PROPERTY_FLAG_ECC',
    'ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_DEVICE_PROPERTY_FLAG_INTEGRATED',
    'ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING',
    'ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE',
    'ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32',
    'ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY', 'ZE_DEVICE_TYPE_CPU',
    'ZE_DEVICE_TYPE_FORCE_UINT32', 'ZE_DEVICE_TYPE_FPGA',
    'ZE_DEVICE_TYPE_GPU', 'ZE_DEVICE_TYPE_MCA', 'ZE_DEVICE_TYPE_VPU',
    'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE',
    'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE',
    'ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32',
    'ZE_EU_COUNT_EXT_VERSION_1_0', 'ZE_EU_COUNT_EXT_VERSION_CURRENT',
    'ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32',
    'ZE_EVENT_POOL_FLAG_FORCE_UINT32',
    'ZE_EVENT_POOL_FLAG_HOST_VISIBLE', 'ZE_EVENT_POOL_FLAG_IPC',
    'ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP',
    'ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_FORCE_UINT32',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_KERNEL',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_SYNCHRONIZED',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_1_0',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_CURRENT',
    'ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_FORCE_UINT32',
    'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0',
    'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT',
    'ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32',
    'ZE_EVENT_SCOPE_FLAG_DEVICE', 'ZE_EVENT_SCOPE_FLAG_FORCE_UINT32',
    'ZE_EVENT_SCOPE_FLAG_HOST', 'ZE_EVENT_SCOPE_FLAG_SUBDEVICE',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32',
    'ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT',
    'ZE_FABRIC_EDGE_EXP_DUPLEXITY_FORCE_UINT32',
    'ZE_FABRIC_EDGE_EXP_DUPLEXITY_FULL_DUPLEX',
    'ZE_FABRIC_EDGE_EXP_DUPLEXITY_HALF_DUPLEX',
    'ZE_FABRIC_EDGE_EXP_DUPLEXITY_UNKNOWN',
    'ZE_FABRIC_VERTEX_EXP_TYPE_DEVICE',
    'ZE_FABRIC_VERTEX_EXP_TYPE_FORCE_UINT32',
    'ZE_FABRIC_VERTEX_EXP_TYPE_SUBDEVICE',
    'ZE_FABRIC_VERTEX_EXP_TYPE_SWITCH',
    'ZE_FABRIC_VERTEX_EXP_TYPE_UNKNOWN', 'ZE_FENCE_FLAG_FORCE_UINT32',
    'ZE_FENCE_FLAG_SIGNALED', 'ZE_FLOAT_ATOMICS_EXT_VERSION_1_0',
    'ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT',
    'ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32',
    'ZE_GLOBAL_OFFSET_EXP_VERSION_1_0',
    'ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT',
    'ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32',
    'ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED',
    'ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT',
    'ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED',
    'ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED',
    'ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32',
    'ZE_IMAGE_COPY_EXT_VERSION_1_0',
    'ZE_IMAGE_COPY_EXT_VERSION_CURRENT',
    'ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32',
    'ZE_IMAGE_FLAG_BIAS_UNCACHED', 'ZE_IMAGE_FLAG_FORCE_UINT32',
    'ZE_IMAGE_FLAG_KERNEL_WRITE', 'ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2',
    'ZE_IMAGE_FORMAT_LAYOUT_11_11_10', 'ZE_IMAGE_FORMAT_LAYOUT_16',
    'ZE_IMAGE_FORMAT_LAYOUT_16_16',
    'ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16', 'ZE_IMAGE_FORMAT_LAYOUT_32',
    'ZE_IMAGE_FORMAT_LAYOUT_32_32',
    'ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32',
    'ZE_IMAGE_FORMAT_LAYOUT_400P', 'ZE_IMAGE_FORMAT_LAYOUT_422H',
    'ZE_IMAGE_FORMAT_LAYOUT_422V', 'ZE_IMAGE_FORMAT_LAYOUT_444P',
    'ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4',
    'ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1', 'ZE_IMAGE_FORMAT_LAYOUT_5_6_5',
    'ZE_IMAGE_FORMAT_LAYOUT_8', 'ZE_IMAGE_FORMAT_LAYOUT_8_8',
    'ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8', 'ZE_IMAGE_FORMAT_LAYOUT_A8P8',
    'ZE_IMAGE_FORMAT_LAYOUT_AI44', 'ZE_IMAGE_FORMAT_LAYOUT_AYUV',
    'ZE_IMAGE_FORMAT_LAYOUT_BRGP',
    'ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32',
    'ZE_IMAGE_FORMAT_LAYOUT_I420', 'ZE_IMAGE_FORMAT_LAYOUT_IA44',
    'ZE_IMAGE_FORMAT_LAYOUT_NV12', 'ZE_IMAGE_FORMAT_LAYOUT_P010',
    'ZE_IMAGE_FORMAT_LAYOUT_P012', 'ZE_IMAGE_FORMAT_LAYOUT_P016',
    'ZE_IMAGE_FORMAT_LAYOUT_P216', 'ZE_IMAGE_FORMAT_LAYOUT_P8',
    'ZE_IMAGE_FORMAT_LAYOUT_RGBP', 'ZE_IMAGE_FORMAT_LAYOUT_UYVY',
    'ZE_IMAGE_FORMAT_LAYOUT_VYUY', 'ZE_IMAGE_FORMAT_LAYOUT_Y16',
    'ZE_IMAGE_FORMAT_LAYOUT_Y210', 'ZE_IMAGE_FORMAT_LAYOUT_Y216',
    'ZE_IMAGE_FORMAT_LAYOUT_Y410', 'ZE_IMAGE_FORMAT_LAYOUT_Y416',
    'ZE_IMAGE_FORMAT_LAYOUT_Y8', 'ZE_IMAGE_FORMAT_LAYOUT_YUY2',
    'ZE_IMAGE_FORMAT_LAYOUT_YUYV', 'ZE_IMAGE_FORMAT_LAYOUT_YV12',
    'ZE_IMAGE_FORMAT_LAYOUT_YVYU', 'ZE_IMAGE_FORMAT_SWIZZLE_0',
    'ZE_IMAGE_FORMAT_SWIZZLE_1', 'ZE_IMAGE_FORMAT_SWIZZLE_A',
    'ZE_IMAGE_FORMAT_SWIZZLE_B',
    'ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32',
    'ZE_IMAGE_FORMAT_SWIZZLE_G', 'ZE_IMAGE_FORMAT_SWIZZLE_R',
    'ZE_IMAGE_FORMAT_SWIZZLE_X', 'ZE_IMAGE_FORMAT_TYPE_FLOAT',
    'ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32', 'ZE_IMAGE_FORMAT_TYPE_SINT',
    'ZE_IMAGE_FORMAT_TYPE_SNORM', 'ZE_IMAGE_FORMAT_TYPE_UINT',
    'ZE_IMAGE_FORMAT_TYPE_UNORM',
    'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0',
    'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT',
    'ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32',
    'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0',
    'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT',
    'ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32',
    'ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32',
    'ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR',
    'ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT', 'ZE_IMAGE_TYPE_1D',
    'ZE_IMAGE_TYPE_1DARRAY', 'ZE_IMAGE_TYPE_2D',
    'ZE_IMAGE_TYPE_2DARRAY', 'ZE_IMAGE_TYPE_3D',
    'ZE_IMAGE_TYPE_BUFFER', 'ZE_IMAGE_TYPE_FORCE_UINT32',
    'ZE_IMAGE_VIEW_EXP_VERSION_1_0',
    'ZE_IMAGE_VIEW_EXP_VERSION_CURRENT',
    'ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32',
    'ZE_IMAGE_VIEW_EXT_VERSION_1_0',
    'ZE_IMAGE_VIEW_EXT_VERSION_CURRENT',
    'ZE_IMAGE_VIEW_EXT_VERSION_FORCE_UINT32',
    'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0',
    'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT',
    'ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32',
    'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_1_0',
    'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_CURRENT',
    'ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_FORCE_UINT32',
    'ZE_INIT_FLAG_FORCE_UINT32', 'ZE_INIT_FLAG_GPU_ONLY',
    'ZE_INIT_FLAG_VPU_ONLY', 'ZE_IPC_MEMORY_FLAG_BIAS_CACHED',
    'ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED',
    'ZE_IPC_MEMORY_FLAG_FORCE_UINT32',
    'ZE_IPC_PROPERTY_FLAG_EVENT_POOL',
    'ZE_IPC_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_IPC_PROPERTY_FLAG_MEMORY',
    'ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY',
    'ZE_KERNEL_FLAG_FORCE_RESIDENCY', 'ZE_KERNEL_FLAG_FORCE_UINT32',
    'ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE',
    'ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32',
    'ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST',
    'ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED',
    'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_1_0',
    'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_CURRENT',
    'ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_FORCE_UINT32',
    'ZE_LATENCY_UNIT_CLOCK', 'ZE_LATENCY_UNIT_FORCE_UINT32',
    'ZE_LATENCY_UNIT_HOP', 'ZE_LATENCY_UNIT_NANOSEC',
    'ZE_LATENCY_UNIT_UNKNOWN',
    'ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS',
    'ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32',
    'ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS',
    'ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS',
    'ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0',
    'ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT',
    'ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32',
    'ZE_LINKONCE_ODR_EXT_VERSION_1_0',
    'ZE_LINKONCE_ODR_EXT_VERSION_CURRENT',
    'ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32',
    'ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32',
    'ZE_MEMORY_ACCESS_ATTRIBUTE_NONE',
    'ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY',
    'ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE',
    'ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC',
    'ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT',
    'ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC',
    'ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32',
    'ZE_MEMORY_ACCESS_CAP_FLAG_RW', 'ZE_MEMORY_ADVICE_BIAS_CACHED',
    'ZE_MEMORY_ADVICE_BIAS_UNCACHED',
    'ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY',
    'ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION',
    'ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY',
    'ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION',
    'ZE_MEMORY_ADVICE_FORCE_UINT32',
    'ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY',
    'ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION',
    'ZE_MEMORY_ADVICE_SET_READ_MOSTLY',
    'ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_DEVICE_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_FORCE_UINT32',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_HOST_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_DEVICE_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_HOST_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_SYSTEM_ATOMICS',
    'ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_SYSTEM_ATOMICS',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT',
    'ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32',
    'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0',
    'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT',
    'ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32',
    'ZE_MEMORY_TYPE_DEVICE', 'ZE_MEMORY_TYPE_FORCE_UINT32',
    'ZE_MEMORY_TYPE_HOST', 'ZE_MEMORY_TYPE_SHARED',
    'ZE_MEMORY_TYPE_UNKNOWN', 'ZE_MODULE_FORMAT_FORCE_UINT32',
    'ZE_MODULE_FORMAT_IL_SPIRV', 'ZE_MODULE_FORMAT_NATIVE',
    'ZE_MODULE_PROGRAM_EXP_VERSION_1_0',
    'ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT',
    'ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32',
    'ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32',
    'ZE_MODULE_PROPERTY_FLAG_IMPORTS',
    'ZE_PCI_PROPERTIES_EXT_VERSION_1_0',
    'ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT',
    'ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32',
    'ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32', 'ZE_PHYSICAL_MEM_FLAG_TBD',
    'ZE_POWER_SAVING_HINT_EXP_VERSION_1_0',
    'ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT',
    'ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32',
    'ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32',
    'ZE_POWER_SAVING_HINT_TYPE_MAX', 'ZE_POWER_SAVING_HINT_TYPE_MIN',
    'ZE_RAYTRACING_EXT_VERSION_1_0',
    'ZE_RAYTRACING_EXT_VERSION_CURRENT',
    'ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32',
    'ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32',
    'ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD',
    'ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32',
    'ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE',
    'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0',
    'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT',
    'ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32',
    'ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE',
    'ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE',
    'ZE_RESULT_ERROR_DEVICE_LOST',
    'ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET',
    'ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE',
    'ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS',
    'ZE_RESULT_ERROR_INVALID_ARGUMENT',
    'ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE',
    'ZE_RESULT_ERROR_INVALID_ENUMERATION',
    'ZE_RESULT_ERROR_INVALID_FUNCTION_NAME',
    'ZE_RESULT_ERROR_INVALID_GLOBAL_NAME',
    'ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION',
    'ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION',
    'ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX',
    'ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE',
    'ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE',
    'ZE_RESULT_ERROR_INVALID_KERNEL_NAME',
    'ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED',
    'ZE_RESULT_ERROR_INVALID_NATIVE_BINARY',
    'ZE_RESULT_ERROR_INVALID_NULL_HANDLE',
    'ZE_RESULT_ERROR_INVALID_NULL_POINTER',
    'ZE_RESULT_ERROR_INVALID_SIZE',
    'ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT',
    'ZE_RESULT_ERROR_MODULE_BUILD_FAILURE',
    'ZE_RESULT_ERROR_MODULE_LINK_FAILURE',
    'ZE_RESULT_ERROR_NOT_AVAILABLE',
    'ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY',
    'ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY',
    'ZE_RESULT_ERROR_OVERLAPPING_REGIONS',
    'ZE_RESULT_ERROR_UNINITIALIZED', 'ZE_RESULT_ERROR_UNKNOWN',
    'ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT',
    'ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION',
    'ZE_RESULT_ERROR_UNSUPPORTED_FEATURE',
    'ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT',
    'ZE_RESULT_ERROR_UNSUPPORTED_SIZE',
    'ZE_RESULT_ERROR_UNSUPPORTED_VERSION',
    'ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX',
    'ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE',
    'ZE_RESULT_EXP_ERROR_REMOTE_DEVICE',
    'ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE',
    'ZE_RESULT_EXP_RTAS_BUILD_DEFERRED',
    'ZE_RESULT_EXP_RTAS_BUILD_RETRY', 'ZE_RESULT_FORCE_UINT32',
    'ZE_RESULT_NOT_READY', 'ZE_RESULT_SUCCESS',
    'ZE_RESULT_WARNING_ACTION_REQUIRED',
    'ZE_RESULT_WARNING_DROPPED_DATA',
    'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_COMPACT',
    'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION',
    'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_FORCE_UINT32',
    'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH',
    'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_LOW',
    'ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM',
    'ZE_RTAS_BUILDER_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_BUILDER_EXP_FLAG_RESERVED',
    'ZE_RTAS_BUILDER_EXP_VERSION_1_0',
    'ZE_RTAS_BUILDER_EXP_VERSION_CURRENT',
    'ZE_RTAS_BUILDER_EXP_VERSION_FORCE_UINT32',
    'ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE',
    'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_FORCE_UINT32',
    'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE',
    'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL',
    'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS',
    'ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_AABB',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FORCE_UINT32',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32',
    'ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32',
    'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_CULL_DISABLE',
    'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_NON_OPAQUE',
    'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_OPAQUE',
    'ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE',
    'ZE_RTAS_DEVICE_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_DEVICE_EXP_FLAG_RESERVED',
    'ZE_RTAS_FORMAT_EXP_FORCE_UINT32', 'ZE_RTAS_FORMAT_EXP_INVALID',
    'ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_FORCE_UINT32',
    'ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_RESERVED',
    'ZE_SAMPLER_ADDRESS_MODE_CLAMP',
    'ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER',
    'ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32',
    'ZE_SAMPLER_ADDRESS_MODE_MIRROR', 'ZE_SAMPLER_ADDRESS_MODE_NONE',
    'ZE_SAMPLER_ADDRESS_MODE_REPEAT',
    'ZE_SAMPLER_FILTER_MODE_FORCE_UINT32',
    'ZE_SAMPLER_FILTER_MODE_LINEAR', 'ZE_SAMPLER_FILTER_MODE_NEAREST',
    'ZE_SCHEDULING_HINTS_EXP_VERSION_1_0',
    'ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT',
    'ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32',
    'ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32',
    'ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST',
    'ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN',
    'ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN',
    'ZE_SRGB_EXT_VERSION_1_0', 'ZE_SRGB_EXT_VERSION_CURRENT',
    'ZE_SRGB_EXT_VERSION_FORCE_UINT32',
    'ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC',
    'ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC',
    'ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC',
    'ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_CONTEXT_DESC',
    'ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT',
    'ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC',
    'ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2',
    'ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES',
    'ZE_STRUCTURE_TYPE_EU_COUNT_EXT', 'ZE_STRUCTURE_TYPE_EVENT_DESC',
    'ZE_STRUCTURE_TYPE_EVENT_POOL_DESC',
    'ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC',
    'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD',
    'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32',
    'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD',
    'ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32',
    'ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_FENCE_DESC',
    'ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_FORCE_UINT32',
    'ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC',
    'ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_IMAGE_DESC',
    'ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC',
    'ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC',
    'ZE_STRUCTURE_TYPE_KERNEL_DESC',
    'ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES',
    'ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC',
    'ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES',
    'ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC',
    'ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC',
    'ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_MODULE_DESC',
    'ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC',
    'ZE_STRUCTURE_TYPE_MODULE_PROPERTIES',
    'ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES',
    'ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC',
    'ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC',
    'ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC',
    'ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC',
    'ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC',
    'ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC',
    'ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS',
    'ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_SAMPLER_DESC',
    'ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC',
    'ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES',
    'ZE_STRUCTURE_TYPE_SRGB_EXT_DESC', 'ZE_SUBGROUP_EXT_VERSION_1_0',
    'ZE_SUBGROUP_EXT_VERSION_CURRENT',
    'ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32',
    'ZE_SUB_ALLOCATIONS_EXP_VERSION_1_0',
    'ZE_SUB_ALLOCATIONS_EXP_VERSION_CURRENT',
    'ZE_SUB_ALLOCATIONS_EXP_VERSION_FORCE_UINT32',
    '_ze_api_version_t', '_ze_bandwidth_unit_t',
    '_ze_bfloat16_conversions_ext_version_t',
    '_ze_cache_config_flag_t', '_ze_cache_ext_region_t',
    '_ze_cache_reservation_ext_version_t', '_ze_command_list_flag_t',
    '_ze_command_queue_flag_t',
    '_ze_command_queue_group_property_flag_t',
    '_ze_command_queue_mode_t', '_ze_command_queue_priority_t',
    '_ze_context_flag_t', '_ze_device_cache_property_flag_t',
    '_ze_device_fp_atomic_ext_flag_t', '_ze_device_fp_flag_t',
    '_ze_device_ip_version_version_t',
    '_ze_device_luid_ext_version_t', '_ze_device_mem_alloc_flag_t',
    '_ze_device_memory_ext_type_t',
    '_ze_device_memory_properties_ext_version_t',
    '_ze_device_memory_property_flag_t', '_ze_device_module_flag_t',
    '_ze_device_p2p_property_flag_t', '_ze_device_property_flag_t',
    '_ze_device_raytracing_ext_flag_t', '_ze_device_type_t',
    '_ze_driver_memory_free_policy_ext_flag_t',
    '_ze_eu_count_ext_version_t', '_ze_event_pool_flag_t',
    '_ze_event_query_kernel_timestamps_ext_flag_t',
    '_ze_event_query_kernel_timestamps_ext_version_t',
    '_ze_event_query_timestamps_exp_version_t',
    '_ze_event_scope_flag_t', '_ze_external_memory_type_flag_t',
    '_ze_fabric_edge_exp_duplexity_t', '_ze_fabric_vertex_exp_type_t',
    '_ze_fence_flag_t', '_ze_float_atomics_ext_version_t',
    '_ze_global_offset_exp_version_t', '_ze_host_mem_alloc_flag_t',
    '_ze_image_copy_ext_version_t', '_ze_image_flag_t',
    '_ze_image_format_layout_t', '_ze_image_format_swizzle_t',
    '_ze_image_format_type_t',
    '_ze_image_memory_properties_exp_version_t',
    '_ze_image_query_alloc_properties_ext_version_t',
    '_ze_image_sampler_filter_flag_t', '_ze_image_type_t',
    '_ze_image_view_exp_version_t', '_ze_image_view_ext_version_t',
    '_ze_image_view_planar_exp_version_t',
    '_ze_image_view_planar_ext_version_t', '_ze_init_flag_t',
    '_ze_ipc_memory_flag_t', '_ze_ipc_property_flag_t',
    '_ze_kernel_flag_t', '_ze_kernel_indirect_access_flag_t',
    '_ze_kernel_max_group_size_properties_ext_version_t',
    '_ze_latency_unit_t', '_ze_linkage_inspection_ext_flag_t',
    '_ze_linkage_inspection_ext_version_t',
    '_ze_linkonce_odr_ext_version_t', '_ze_memory_access_attribute_t',
    '_ze_memory_access_cap_flag_t', '_ze_memory_advice_t',
    '_ze_memory_atomic_attr_exp_flag_t',
    '_ze_memory_compression_hints_ext_flag_t',
    '_ze_memory_compression_hints_ext_version_t',
    '_ze_memory_free_policies_ext_version_t', '_ze_memory_type_t',
    '_ze_module_format_t', '_ze_module_program_exp_version_t',
    '_ze_module_property_flag_t', '_ze_pci_properties_ext_version_t',
    '_ze_physical_mem_flag_t', '_ze_power_saving_hint_exp_version_t',
    '_ze_power_saving_hint_type_t', '_ze_raytracing_ext_version_t',
    '_ze_raytracing_mem_alloc_ext_flag_t',
    '_ze_relaxed_allocation_limits_exp_flag_t',
    '_ze_relaxed_allocation_limits_exp_version_t', '_ze_result_t',
    '_ze_rtas_builder_build_op_exp_flag_t',
    '_ze_rtas_builder_build_quality_hint_exp_t',
    '_ze_rtas_builder_exp_flag_t', '_ze_rtas_builder_exp_version_t',
    '_ze_rtas_builder_geometry_exp_flag_t',
    '_ze_rtas_builder_geometry_type_exp_t',
    '_ze_rtas_builder_input_data_format_exp_t',
    '_ze_rtas_builder_instance_exp_flag_t',
    '_ze_rtas_device_exp_flag_t', '_ze_rtas_format_exp_t',
    '_ze_rtas_parallel_operation_exp_flag_t',
    '_ze_sampler_address_mode_t', '_ze_sampler_filter_mode_t',
    '_ze_scheduling_hint_exp_flag_t',
    '_ze_scheduling_hints_exp_version_t', '_ze_srgb_ext_version_t',
    '_ze_structure_type_t', '_ze_sub_allocations_exp_version_t',
    '_ze_subgroup_ext_version_t', '_zel_handle_type_t', 'size_t',
    'struct__ze_base_cb_params_t', 'struct__ze_base_desc_t',
    'struct__ze_base_properties_t',
    'struct__ze_cache_reservation_ext_desc_t',
    'struct__ze_callbacks_t',
    'struct__ze_command_list_append_barrier_params_t',
    'struct__ze_command_list_append_event_reset_params_t',
    'struct__ze_command_list_append_image_copy_from_memory_params_t',
    'struct__ze_command_list_append_image_copy_params_t',
    'struct__ze_command_list_append_image_copy_region_params_t',
    'struct__ze_command_list_append_image_copy_to_memory_params_t',
    'struct__ze_command_list_append_launch_cooperative_kernel_params_t',
    'struct__ze_command_list_append_launch_kernel_indirect_params_t',
    'struct__ze_command_list_append_launch_kernel_params_t',
    'struct__ze_command_list_append_launch_multiple_kernels_indirect_params_t',
    'struct__ze_command_list_append_mem_advise_params_t',
    'struct__ze_command_list_append_memory_copy_from_context_params_t',
    'struct__ze_command_list_append_memory_copy_params_t',
    'struct__ze_command_list_append_memory_copy_region_params_t',
    'struct__ze_command_list_append_memory_fill_params_t',
    'struct__ze_command_list_append_memory_prefetch_params_t',
    'struct__ze_command_list_append_memory_ranges_barrier_params_t',
    'struct__ze_command_list_append_query_kernel_timestamps_params_t',
    'struct__ze_command_list_append_signal_event_params_t',
    'struct__ze_command_list_append_wait_on_events_params_t',
    'struct__ze_command_list_append_write_global_timestamp_params_t',
    'struct__ze_command_list_callbacks_t',
    'struct__ze_command_list_close_params_t',
    'struct__ze_command_list_create_immediate_params_t',
    'struct__ze_command_list_create_params_t',
    'struct__ze_command_list_desc_t',
    'struct__ze_command_list_destroy_params_t',
    'struct__ze_command_list_handle_t',
    'struct__ze_command_list_reset_params_t',
    'struct__ze_command_queue_callbacks_t',
    'struct__ze_command_queue_create_params_t',
    'struct__ze_command_queue_desc_t',
    'struct__ze_command_queue_destroy_params_t',
    'struct__ze_command_queue_execute_command_lists_params_t',
    'struct__ze_command_queue_group_properties_t',
    'struct__ze_command_queue_handle_t',
    'struct__ze_command_queue_synchronize_params_t',
    'struct__ze_context_callbacks_t',
    'struct__ze_context_create_params_t', 'struct__ze_context_desc_t',
    'struct__ze_context_destroy_params_t',
    'struct__ze_context_evict_image_params_t',
    'struct__ze_context_evict_memory_params_t',
    'struct__ze_context_get_status_params_t',
    'struct__ze_context_handle_t',
    'struct__ze_context_make_image_resident_params_t',
    'struct__ze_context_make_memory_resident_params_t',
    'struct__ze_context_power_saving_hint_exp_desc_t',
    'struct__ze_context_system_barrier_params_t',
    'struct__ze_copy_bandwidth_exp_properties_t',
    'struct__ze_copy_region_t',
    'struct__ze_device_cache_properties_t',
    'struct__ze_device_callbacks_t',
    'struct__ze_device_can_access_peer_params_t',
    'struct__ze_device_compute_properties_t',
    'struct__ze_device_external_memory_properties_t',
    'struct__ze_device_get_cache_properties_params_t',
    'struct__ze_device_get_command_queue_group_properties_params_t',
    'struct__ze_device_get_compute_properties_params_t',
    'struct__ze_device_get_external_memory_properties_params_t',
    'struct__ze_device_get_image_properties_params_t',
    'struct__ze_device_get_memory_access_properties_params_t',
    'struct__ze_device_get_memory_properties_params_t',
    'struct__ze_device_get_module_properties_params_t',
    'struct__ze_device_get_p2_p_properties_params_t',
    'struct__ze_device_get_params_t',
    'struct__ze_device_get_properties_params_t',
    'struct__ze_device_get_status_params_t',
    'struct__ze_device_get_sub_devices_params_t',
    'struct__ze_device_handle_t',
    'struct__ze_device_image_properties_t',
    'struct__ze_device_ip_version_ext_t',
    'struct__ze_device_luid_ext_properties_t',
    'struct__ze_device_luid_ext_t',
    'struct__ze_device_mem_alloc_desc_t',
    'struct__ze_device_memory_access_properties_t',
    'struct__ze_device_memory_ext_properties_t',
    'struct__ze_device_memory_properties_t',
    'struct__ze_device_module_properties_t',
    'struct__ze_device_p2p_bandwidth_exp_properties_t',
    'struct__ze_device_p2p_properties_t',
    'struct__ze_device_properties_t',
    'struct__ze_device_raytracing_ext_properties_t',
    'struct__ze_device_thread_t', 'struct__ze_device_uuid_t',
    'struct__ze_driver_callbacks_t',
    'struct__ze_driver_extension_properties_t',
    'struct__ze_driver_get_api_version_params_t',
    'struct__ze_driver_get_extension_properties_params_t',
    'struct__ze_driver_get_ipc_properties_params_t',
    'struct__ze_driver_get_params_t',
    'struct__ze_driver_get_properties_params_t',
    'struct__ze_driver_handle_t',
    'struct__ze_driver_ipc_properties_t',
    'struct__ze_driver_memory_free_ext_properties_t',
    'struct__ze_driver_properties_t', 'struct__ze_driver_uuid_t',
    'struct__ze_eu_count_ext_t', 'struct__ze_event_callbacks_t',
    'struct__ze_event_create_params_t', 'struct__ze_event_desc_t',
    'struct__ze_event_destroy_params_t', 'struct__ze_event_handle_t',
    'struct__ze_event_host_reset_params_t',
    'struct__ze_event_host_signal_params_t',
    'struct__ze_event_host_synchronize_params_t',
    'struct__ze_event_pool_callbacks_t',
    'struct__ze_event_pool_close_ipc_handle_params_t',
    'struct__ze_event_pool_create_params_t',
    'struct__ze_event_pool_desc_t',
    'struct__ze_event_pool_destroy_params_t',
    'struct__ze_event_pool_get_ipc_handle_params_t',
    'struct__ze_event_pool_handle_t',
    'struct__ze_event_pool_open_ipc_handle_params_t',
    'struct__ze_event_query_kernel_timestamp_params_t',
    'struct__ze_event_query_kernel_timestamps_ext_properties_t',
    'struct__ze_event_query_kernel_timestamps_results_ext_properties_t',
    'struct__ze_event_query_status_params_t',
    'struct__ze_external_memory_export_desc_t',
    'struct__ze_external_memory_export_fd_t',
    'struct__ze_external_memory_export_win32_handle_t',
    'struct__ze_external_memory_import_fd_t',
    'struct__ze_external_memory_import_win32_handle_t',
    'struct__ze_fabric_edge_exp_properties_t',
    'struct__ze_fabric_edge_handle_t',
    'struct__ze_fabric_vertex_exp_properties_t',
    'struct__ze_fabric_vertex_handle_t',
    'struct__ze_fabric_vertex_pci_exp_address_t',
    'struct__ze_fence_callbacks_t',
    'struct__ze_fence_create_params_t', 'struct__ze_fence_desc_t',
    'struct__ze_fence_destroy_params_t', 'struct__ze_fence_handle_t',
    'struct__ze_fence_host_synchronize_params_t',
    'struct__ze_fence_query_status_params_t',
    'struct__ze_fence_reset_params_t',
    'struct__ze_float_atomic_ext_properties_t',
    'struct__ze_global_callbacks_t', 'struct__ze_group_count_t',
    'struct__ze_host_mem_alloc_desc_t',
    'struct__ze_image_allocation_ext_properties_t',
    'struct__ze_image_callbacks_t',
    'struct__ze_image_create_params_t', 'struct__ze_image_desc_t',
    'struct__ze_image_destroy_params_t', 'struct__ze_image_format_t',
    'struct__ze_image_get_properties_params_t',
    'struct__ze_image_handle_t',
    'struct__ze_image_memory_properties_exp_t',
    'struct__ze_image_properties_t', 'struct__ze_image_region_t',
    'struct__ze_image_view_planar_exp_desc_t',
    'struct__ze_image_view_planar_ext_desc_t',
    'struct__ze_init_params_t', 'struct__ze_ipc_event_pool_handle_t',
    'struct__ze_ipc_mem_handle_t', 'struct__ze_kernel_callbacks_t',
    'struct__ze_kernel_create_params_t', 'struct__ze_kernel_desc_t',
    'struct__ze_kernel_destroy_params_t',
    'struct__ze_kernel_get_indirect_access_params_t',
    'struct__ze_kernel_get_name_params_t',
    'struct__ze_kernel_get_properties_params_t',
    'struct__ze_kernel_get_source_attributes_params_t',
    'struct__ze_kernel_handle_t',
    'struct__ze_kernel_max_group_size_properties_ext_t',
    'struct__ze_kernel_preferred_group_size_properties_t',
    'struct__ze_kernel_properties_t',
    'struct__ze_kernel_set_argument_value_params_t',
    'struct__ze_kernel_set_cache_config_params_t',
    'struct__ze_kernel_set_group_size_params_t',
    'struct__ze_kernel_set_indirect_access_params_t',
    'struct__ze_kernel_suggest_group_size_params_t',
    'struct__ze_kernel_suggest_max_cooperative_group_count_params_t',
    'struct__ze_kernel_timestamp_data_t',
    'struct__ze_kernel_timestamp_result_t',
    'struct__ze_kernel_uuid_t',
    'struct__ze_linkage_inspection_ext_desc_t',
    'struct__ze_mem_alloc_device_params_t',
    'struct__ze_mem_alloc_host_params_t',
    'struct__ze_mem_alloc_shared_params_t',
    'struct__ze_mem_callbacks_t',
    'struct__ze_mem_close_ipc_handle_params_t',
    'struct__ze_mem_free_params_t',
    'struct__ze_mem_get_address_range_params_t',
    'struct__ze_mem_get_alloc_properties_params_t',
    'struct__ze_mem_get_ipc_handle_params_t',
    'struct__ze_mem_open_ipc_handle_params_t',
    'struct__ze_memory_allocation_properties_t',
    'struct__ze_memory_compression_hints_ext_desc_t',
    'struct__ze_memory_free_ext_desc_t',
    'struct__ze_memory_sub_allocations_exp_properties_t',
    'struct__ze_module_build_log_callbacks_t',
    'struct__ze_module_build_log_destroy_params_t',
    'struct__ze_module_build_log_get_string_params_t',
    'struct__ze_module_build_log_handle_t',
    'struct__ze_module_callbacks_t', 'struct__ze_module_constants_t',
    'struct__ze_module_create_params_t', 'struct__ze_module_desc_t',
    'struct__ze_module_destroy_params_t',
    'struct__ze_module_dynamic_link_params_t',
    'struct__ze_module_get_function_pointer_params_t',
    'struct__ze_module_get_global_pointer_params_t',
    'struct__ze_module_get_kernel_names_params_t',
    'struct__ze_module_get_native_binary_params_t',
    'struct__ze_module_get_properties_params_t',
    'struct__ze_module_handle_t',
    'struct__ze_module_program_exp_desc_t',
    'struct__ze_module_properties_t',
    'struct__ze_native_kernel_uuid_t', 'struct__ze_pci_address_ext_t',
    'struct__ze_pci_ext_properties_t', 'struct__ze_pci_speed_ext_t',
    'struct__ze_physical_mem_callbacks_t',
    'struct__ze_physical_mem_create_params_t',
    'struct__ze_physical_mem_desc_t',
    'struct__ze_physical_mem_destroy_params_t',
    'struct__ze_physical_mem_handle_t',
    'struct__ze_raytracing_mem_alloc_ext_desc_t',
    'struct__ze_relaxed_allocation_limits_exp_desc_t',
    'struct__ze_rtas_aabb_exp_t',
    'struct__ze_rtas_builder_build_op_exp_desc_t',
    'struct__ze_rtas_builder_exp_desc_t',
    'struct__ze_rtas_builder_exp_handle_t',
    'struct__ze_rtas_builder_exp_properties_t',
    'struct__ze_rtas_builder_geometry_info_exp_t',
    'struct__ze_rtas_builder_instance_geometry_info_exp_t',
    'struct__ze_rtas_builder_procedural_geometry_info_exp_t',
    'struct__ze_rtas_builder_quads_geometry_info_exp_t',
    'struct__ze_rtas_builder_triangles_geometry_info_exp_t',
    'struct__ze_rtas_device_exp_properties_t',
    'struct__ze_rtas_float3_exp_t',
    'struct__ze_rtas_geometry_aabbs_exp_cb_params_t',
    'struct__ze_rtas_parallel_operation_exp_handle_t',
    'struct__ze_rtas_parallel_operation_exp_properties_t',
    'struct__ze_rtas_quad_indices_uint32_exp_t',
    'struct__ze_rtas_transform_float3x4_aligned_column_major_exp_t',
    'struct__ze_rtas_transform_float3x4_column_major_exp_t',
    'struct__ze_rtas_transform_float3x4_row_major_exp_t',
    'struct__ze_rtas_triangle_indices_uint32_exp_t',
    'struct__ze_sampler_callbacks_t',
    'struct__ze_sampler_create_params_t', 'struct__ze_sampler_desc_t',
    'struct__ze_sampler_destroy_params_t',
    'struct__ze_sampler_handle_t',
    'struct__ze_scheduling_hint_exp_desc_t',
    'struct__ze_scheduling_hint_exp_properties_t',
    'struct__ze_srgb_ext_desc_t', 'struct__ze_sub_allocation_t',
    'struct__ze_synchronized_timestamp_data_ext_t',
    'struct__ze_synchronized_timestamp_result_ext_t',
    'struct__ze_uuid_t', 'struct__ze_virtual_mem_callbacks_t',
    'struct__ze_virtual_mem_free_params_t',
    'struct__ze_virtual_mem_get_access_attribute_params_t',
    'struct__ze_virtual_mem_map_params_t',
    'struct__ze_virtual_mem_query_page_size_params_t',
    'struct__ze_virtual_mem_reserve_params_t',
    'struct__ze_virtual_mem_set_access_attribute_params_t',
    'struct__ze_virtual_mem_unmap_params_t', 'struct__zel_version',
    'struct_zel_component_version', 'uint32_t', 'uint64_t',
    'zeCommandListAppendBarrier', 'zeCommandListAppendEventReset',
    'zeCommandListAppendImageCopy',
    'zeCommandListAppendImageCopyFromMemory',
    'zeCommandListAppendImageCopyFromMemoryExt',
    'zeCommandListAppendImageCopyRegion',
    'zeCommandListAppendImageCopyToMemory',
    'zeCommandListAppendImageCopyToMemoryExt',
    'zeCommandListAppendLaunchCooperativeKernel',
    'zeCommandListAppendLaunchKernel',
    'zeCommandListAppendLaunchKernelIndirect',
    'zeCommandListAppendLaunchMultipleKernelsIndirect',
    'zeCommandListAppendMemAdvise', 'zeCommandListAppendMemoryCopy',
    'zeCommandListAppendMemoryCopyFromContext',
    'zeCommandListAppendMemoryCopyRegion',
    'zeCommandListAppendMemoryFill',
    'zeCommandListAppendMemoryPrefetch',
    'zeCommandListAppendMemoryRangesBarrier',
    'zeCommandListAppendQueryKernelTimestamps',
    'zeCommandListAppendSignalEvent',
    'zeCommandListAppendWaitOnEvents',
    'zeCommandListAppendWriteGlobalTimestamp', 'zeCommandListClose',
    'zeCommandListCreate', 'zeCommandListCreateImmediate',
    'zeCommandListDestroy', 'zeCommandListHostSynchronize',
    'zeCommandListReset', 'zeCommandQueueCreate',
    'zeCommandQueueDestroy', 'zeCommandQueueExecuteCommandLists',
    'zeCommandQueueSynchronize', 'zeContextCreate',
    'zeContextCreateEx', 'zeContextDestroy', 'zeContextEvictImage',
    'zeContextEvictMemory', 'zeContextGetStatus',
    'zeContextMakeImageResident', 'zeContextMakeMemoryResident',
    'zeContextSystemBarrier', 'zeDeviceCanAccessPeer', 'zeDeviceGet',
    'zeDeviceGetCacheProperties',
    'zeDeviceGetCommandQueueGroupProperties',
    'zeDeviceGetComputeProperties',
    'zeDeviceGetExternalMemoryProperties',
    'zeDeviceGetFabricVertexExp', 'zeDeviceGetGlobalTimestamps',
    'zeDeviceGetImageProperties', 'zeDeviceGetMemoryAccessProperties',
    'zeDeviceGetMemoryProperties', 'zeDeviceGetModuleProperties',
    'zeDeviceGetP2PProperties', 'zeDeviceGetProperties',
    'zeDeviceGetRootDevice', 'zeDeviceGetStatus',
    'zeDeviceGetSubDevices', 'zeDevicePciGetPropertiesExt',
    'zeDeviceReserveCacheExt', 'zeDeviceSetCacheAdviceExt',
    'zeDriverGet', 'zeDriverGetApiVersion',
    'zeDriverGetExtensionFunctionAddress',
    'zeDriverGetExtensionProperties', 'zeDriverGetIpcProperties',
    'zeDriverGetLastErrorDescription', 'zeDriverGetProperties',
    'zeDriverRTASFormatCompatibilityCheckExp', 'zeEventCreate',
    'zeEventDestroy', 'zeEventHostReset', 'zeEventHostSignal',
    'zeEventHostSynchronize', 'zeEventPoolCloseIpcHandle',
    'zeEventPoolCreate', 'zeEventPoolDestroy',
    'zeEventPoolGetIpcHandle', 'zeEventPoolOpenIpcHandle',
    'zeEventPoolPutIpcHandle', 'zeEventQueryKernelTimestamp',
    'zeEventQueryKernelTimestampsExt', 'zeEventQueryStatus',
    'zeEventQueryTimestampsExp', 'zeFabricEdgeGetExp',
    'zeFabricEdgeGetPropertiesExp', 'zeFabricEdgeGetVerticesExp',
    'zeFabricVertexGetDeviceExp', 'zeFabricVertexGetExp',
    'zeFabricVertexGetPropertiesExp',
    'zeFabricVertexGetSubVerticesExp', 'zeFenceCreate',
    'zeFenceDestroy', 'zeFenceHostSynchronize', 'zeFenceQueryStatus',
    'zeFenceReset', 'zeImageCreate', 'zeImageDestroy',
    'zeImageGetAllocPropertiesExt', 'zeImageGetMemoryPropertiesExp',
    'zeImageGetProperties', 'zeImageViewCreateExp',
    'zeImageViewCreateExt', 'zeInit', 'zeKernelCreate',
    'zeKernelDestroy', 'zeKernelGetIndirectAccess', 'zeKernelGetName',
    'zeKernelGetProperties', 'zeKernelGetSourceAttributes',
    'zeKernelSchedulingHintExp', 'zeKernelSetArgumentValue',
    'zeKernelSetCacheConfig', 'zeKernelSetGlobalOffsetExp',
    'zeKernelSetGroupSize', 'zeKernelSetIndirectAccess',
    'zeKernelSuggestGroupSize',
    'zeKernelSuggestMaxCooperativeGroupCount', 'zeMemAllocDevice',
    'zeMemAllocHost', 'zeMemAllocShared', 'zeMemCloseIpcHandle',
    'zeMemFree', 'zeMemFreeExt', 'zeMemGetAddressRange',
    'zeMemGetAllocProperties', 'zeMemGetAtomicAccessAttributeExp',
    'zeMemGetFileDescriptorFromIpcHandleExp', 'zeMemGetIpcHandle',
    'zeMemGetIpcHandleFromFileDescriptorExp', 'zeMemOpenIpcHandle',
    'zeMemPutIpcHandle', 'zeMemSetAtomicAccessAttributeExp',
    'zeModuleBuildLogDestroy', 'zeModuleBuildLogGetString',
    'zeModuleCreate', 'zeModuleDestroy', 'zeModuleDynamicLink',
    'zeModuleGetFunctionPointer', 'zeModuleGetGlobalPointer',
    'zeModuleGetKernelNames', 'zeModuleGetNativeBinary',
    'zeModuleGetProperties', 'zeModuleInspectLinkageExt',
    'zePhysicalMemCreate', 'zePhysicalMemDestroy',
    'zeRTASBuilderBuildExp', 'zeRTASBuilderCreateExp',
    'zeRTASBuilderDestroyExp', 'zeRTASBuilderGetBuildPropertiesExp',
    'zeRTASParallelOperationCreateExp',
    'zeRTASParallelOperationDestroyExp',
    'zeRTASParallelOperationGetPropertiesExp',
    'zeRTASParallelOperationJoinExp', 'zeSamplerCreate',
    'zeSamplerDestroy', 'zeVirtualMemFree',
    'zeVirtualMemGetAccessAttribute', 'zeVirtualMemMap',
    'zeVirtualMemQueryPageSize', 'zeVirtualMemReserve',
    'zeVirtualMemSetAccessAttribute', 'zeVirtualMemUnmap',
    'ze_api_version_t', 'ze_api_version_t__enumvalues',
    'ze_bandwidth_unit_t', 'ze_bandwidth_unit_t__enumvalues',
    'ze_base_cb_params_t', 'ze_base_desc_t', 'ze_base_properties_t',
    'ze_bfloat16_conversions_ext_version_t',
    'ze_bfloat16_conversions_ext_version_t__enumvalues', 'ze_bool_t',
    'ze_cache_config_flag_t', 'ze_cache_config_flag_t__enumvalues',
    'ze_cache_config_flags_t', 'ze_cache_ext_region_t',
    'ze_cache_ext_region_t__enumvalues',
    'ze_cache_reservation_ext_desc_t',
    'ze_cache_reservation_ext_version_t',
    'ze_cache_reservation_ext_version_t__enumvalues',
    'ze_callbacks_t', 'ze_command_list_append_barrier_params_t',
    'ze_command_list_append_event_reset_params_t',
    'ze_command_list_append_image_copy_from_memory_params_t',
    'ze_command_list_append_image_copy_params_t',
    'ze_command_list_append_image_copy_region_params_t',
    'ze_command_list_append_image_copy_to_memory_params_t',
    'ze_command_list_append_launch_cooperative_kernel_params_t',
    'ze_command_list_append_launch_kernel_indirect_params_t',
    'ze_command_list_append_launch_kernel_params_t',
    'ze_command_list_append_launch_multiple_kernels_indirect_params_t',
    'ze_command_list_append_mem_advise_params_t',
    'ze_command_list_append_memory_copy_from_context_params_t',
    'ze_command_list_append_memory_copy_params_t',
    'ze_command_list_append_memory_copy_region_params_t',
    'ze_command_list_append_memory_fill_params_t',
    'ze_command_list_append_memory_prefetch_params_t',
    'ze_command_list_append_memory_ranges_barrier_params_t',
    'ze_command_list_append_query_kernel_timestamps_params_t',
    'ze_command_list_append_signal_event_params_t',
    'ze_command_list_append_wait_on_events_params_t',
    'ze_command_list_append_write_global_timestamp_params_t',
    'ze_command_list_callbacks_t', 'ze_command_list_close_params_t',
    'ze_command_list_create_immediate_params_t',
    'ze_command_list_create_params_t', 'ze_command_list_desc_t',
    'ze_command_list_destroy_params_t', 'ze_command_list_flag_t',
    'ze_command_list_flag_t__enumvalues', 'ze_command_list_flags_t',
    'ze_command_list_handle_t', 'ze_command_list_reset_params_t',
    'ze_command_queue_callbacks_t',
    'ze_command_queue_create_params_t', 'ze_command_queue_desc_t',
    'ze_command_queue_destroy_params_t',
    'ze_command_queue_execute_command_lists_params_t',
    'ze_command_queue_flag_t', 'ze_command_queue_flag_t__enumvalues',
    'ze_command_queue_flags_t', 'ze_command_queue_group_properties_t',
    'ze_command_queue_group_property_flag_t',
    'ze_command_queue_group_property_flag_t__enumvalues',
    'ze_command_queue_group_property_flags_t',
    'ze_command_queue_handle_t', 'ze_command_queue_mode_t',
    'ze_command_queue_mode_t__enumvalues',
    'ze_command_queue_priority_t',
    'ze_command_queue_priority_t__enumvalues',
    'ze_command_queue_synchronize_params_t', 'ze_context_callbacks_t',
    'ze_context_create_params_t', 'ze_context_desc_t',
    'ze_context_destroy_params_t', 'ze_context_evict_image_params_t',
    'ze_context_evict_memory_params_t', 'ze_context_flag_t',
    'ze_context_flag_t__enumvalues', 'ze_context_flags_t',
    'ze_context_get_status_params_t', 'ze_context_handle_t',
    'ze_context_make_image_resident_params_t',
    'ze_context_make_memory_resident_params_t',
    'ze_context_power_saving_hint_exp_desc_t',
    'ze_context_system_barrier_params_t',
    'ze_copy_bandwidth_exp_properties_t', 'ze_copy_region_t',
    'ze_device_cache_properties_t', 'ze_device_cache_property_flag_t',
    'ze_device_cache_property_flag_t__enumvalues',
    'ze_device_cache_property_flags_t', 'ze_device_callbacks_t',
    'ze_device_can_access_peer_params_t',
    'ze_device_compute_properties_t',
    'ze_device_external_memory_properties_t',
    'ze_device_fp_atomic_ext_flag_t',
    'ze_device_fp_atomic_ext_flag_t__enumvalues',
    'ze_device_fp_atomic_ext_flags_t', 'ze_device_fp_flag_t',
    'ze_device_fp_flag_t__enumvalues', 'ze_device_fp_flags_t',
    'ze_device_get_cache_properties_params_t',
    'ze_device_get_command_queue_group_properties_params_t',
    'ze_device_get_compute_properties_params_t',
    'ze_device_get_external_memory_properties_params_t',
    'ze_device_get_image_properties_params_t',
    'ze_device_get_memory_access_properties_params_t',
    'ze_device_get_memory_properties_params_t',
    'ze_device_get_module_properties_params_t',
    'ze_device_get_p2_p_properties_params_t',
    'ze_device_get_params_t', 'ze_device_get_properties_params_t',
    'ze_device_get_status_params_t',
    'ze_device_get_sub_devices_params_t', 'ze_device_handle_t',
    'ze_device_image_properties_t', 'ze_device_ip_version_ext_t',
    'ze_device_ip_version_version_t',
    'ze_device_ip_version_version_t__enumvalues',
    'ze_device_luid_ext_properties_t', 'ze_device_luid_ext_t',
    'ze_device_luid_ext_version_t',
    'ze_device_luid_ext_version_t__enumvalues',
    'ze_device_mem_alloc_desc_t', 'ze_device_mem_alloc_flag_t',
    'ze_device_mem_alloc_flag_t__enumvalues',
    'ze_device_mem_alloc_flags_t',
    'ze_device_memory_access_properties_t',
    'ze_device_memory_ext_properties_t',
    'ze_device_memory_ext_type_t',
    'ze_device_memory_ext_type_t__enumvalues',
    'ze_device_memory_properties_ext_version_t',
    'ze_device_memory_properties_ext_version_t__enumvalues',
    'ze_device_memory_properties_t',
    'ze_device_memory_property_flag_t',
    'ze_device_memory_property_flag_t__enumvalues',
    'ze_device_memory_property_flags_t', 'ze_device_module_flag_t',
    'ze_device_module_flag_t__enumvalues', 'ze_device_module_flags_t',
    'ze_device_module_properties_t',
    'ze_device_p2p_bandwidth_exp_properties_t',
    'ze_device_p2p_properties_t', 'ze_device_p2p_property_flag_t',
    'ze_device_p2p_property_flag_t__enumvalues',
    'ze_device_p2p_property_flags_t', 'ze_device_properties_t',
    'ze_device_property_flag_t',
    'ze_device_property_flag_t__enumvalues',
    'ze_device_property_flags_t', 'ze_device_raytracing_ext_flag_t',
    'ze_device_raytracing_ext_flag_t__enumvalues',
    'ze_device_raytracing_ext_flags_t',
    'ze_device_raytracing_ext_properties_t', 'ze_device_thread_t',
    'ze_device_type_t', 'ze_device_type_t__enumvalues',
    'ze_device_uuid_t', 'ze_driver_callbacks_t',
    'ze_driver_extension_properties_t',
    'ze_driver_get_api_version_params_t',
    'ze_driver_get_extension_properties_params_t',
    'ze_driver_get_ipc_properties_params_t', 'ze_driver_get_params_t',
    'ze_driver_get_properties_params_t', 'ze_driver_handle_t',
    'ze_driver_ipc_properties_t',
    'ze_driver_memory_free_ext_properties_t',
    'ze_driver_memory_free_policy_ext_flag_t',
    'ze_driver_memory_free_policy_ext_flag_t__enumvalues',
    'ze_driver_memory_free_policy_ext_flags_t',
    'ze_driver_properties_t', 'ze_driver_uuid_t', 'ze_eu_count_ext_t',
    'ze_eu_count_ext_version_t',
    'ze_eu_count_ext_version_t__enumvalues', 'ze_event_callbacks_t',
    'ze_event_create_params_t', 'ze_event_desc_t',
    'ze_event_destroy_params_t', 'ze_event_handle_t',
    'ze_event_host_reset_params_t', 'ze_event_host_signal_params_t',
    'ze_event_host_synchronize_params_t', 'ze_event_pool_callbacks_t',
    'ze_event_pool_close_ipc_handle_params_t',
    'ze_event_pool_create_params_t', 'ze_event_pool_desc_t',
    'ze_event_pool_destroy_params_t', 'ze_event_pool_flag_t',
    'ze_event_pool_flag_t__enumvalues', 'ze_event_pool_flags_t',
    'ze_event_pool_get_ipc_handle_params_t', 'ze_event_pool_handle_t',
    'ze_event_pool_open_ipc_handle_params_t',
    'ze_event_query_kernel_timestamp_params_t',
    'ze_event_query_kernel_timestamps_ext_flag_t',
    'ze_event_query_kernel_timestamps_ext_flag_t__enumvalues',
    'ze_event_query_kernel_timestamps_ext_flags_t',
    'ze_event_query_kernel_timestamps_ext_properties_t',
    'ze_event_query_kernel_timestamps_ext_version_t',
    'ze_event_query_kernel_timestamps_ext_version_t__enumvalues',
    'ze_event_query_kernel_timestamps_results_ext_properties_t',
    'ze_event_query_status_params_t',
    'ze_event_query_timestamps_exp_version_t',
    'ze_event_query_timestamps_exp_version_t__enumvalues',
    'ze_event_scope_flag_t', 'ze_event_scope_flag_t__enumvalues',
    'ze_event_scope_flags_t', 'ze_external_memory_export_desc_t',
    'ze_external_memory_export_fd_t',
    'ze_external_memory_export_win32_handle_t',
    'ze_external_memory_import_fd_t',
    'ze_external_memory_import_win32_handle_t',
    'ze_external_memory_type_flag_t',
    'ze_external_memory_type_flag_t__enumvalues',
    'ze_external_memory_type_flags_t',
    'ze_fabric_edge_exp_duplexity_t',
    'ze_fabric_edge_exp_duplexity_t__enumvalues',
    'ze_fabric_edge_exp_properties_t', 'ze_fabric_edge_handle_t',
    'ze_fabric_vertex_exp_properties_t',
    'ze_fabric_vertex_exp_type_t',
    'ze_fabric_vertex_exp_type_t__enumvalues',
    'ze_fabric_vertex_handle_t', 'ze_fabric_vertex_pci_exp_address_t',
    'ze_fence_callbacks_t', 'ze_fence_create_params_t',
    'ze_fence_desc_t', 'ze_fence_destroy_params_t', 'ze_fence_flag_t',
    'ze_fence_flag_t__enumvalues', 'ze_fence_flags_t',
    'ze_fence_handle_t', 'ze_fence_host_synchronize_params_t',
    'ze_fence_query_status_params_t', 'ze_fence_reset_params_t',
    'ze_float_atomic_ext_properties_t',
    'ze_float_atomics_ext_version_t',
    'ze_float_atomics_ext_version_t__enumvalues',
    'ze_global_callbacks_t', 'ze_global_offset_exp_version_t',
    'ze_global_offset_exp_version_t__enumvalues', 'ze_group_count_t',
    'ze_host_mem_alloc_desc_t', 'ze_host_mem_alloc_flag_t',
    'ze_host_mem_alloc_flag_t__enumvalues',
    'ze_host_mem_alloc_flags_t',
    'ze_image_allocation_ext_properties_t', 'ze_image_callbacks_t',
    'ze_image_copy_ext_version_t',
    'ze_image_copy_ext_version_t__enumvalues',
    'ze_image_create_params_t', 'ze_image_desc_t',
    'ze_image_destroy_params_t', 'ze_image_flag_t',
    'ze_image_flag_t__enumvalues', 'ze_image_flags_t',
    'ze_image_format_layout_t',
    'ze_image_format_layout_t__enumvalues',
    'ze_image_format_swizzle_t',
    'ze_image_format_swizzle_t__enumvalues', 'ze_image_format_t',
    'ze_image_format_type_t', 'ze_image_format_type_t__enumvalues',
    'ze_image_get_properties_params_t', 'ze_image_handle_t',
    'ze_image_memory_properties_exp_t',
    'ze_image_memory_properties_exp_version_t',
    'ze_image_memory_properties_exp_version_t__enumvalues',
    'ze_image_properties_t',
    'ze_image_query_alloc_properties_ext_version_t',
    'ze_image_query_alloc_properties_ext_version_t__enumvalues',
    'ze_image_region_t', 'ze_image_sampler_filter_flag_t',
    'ze_image_sampler_filter_flag_t__enumvalues',
    'ze_image_sampler_filter_flags_t', 'ze_image_type_t',
    'ze_image_type_t__enumvalues', 'ze_image_view_exp_version_t',
    'ze_image_view_exp_version_t__enumvalues',
    'ze_image_view_ext_version_t',
    'ze_image_view_ext_version_t__enumvalues',
    'ze_image_view_planar_exp_desc_t',
    'ze_image_view_planar_exp_version_t',
    'ze_image_view_planar_exp_version_t__enumvalues',
    'ze_image_view_planar_ext_desc_t',
    'ze_image_view_planar_ext_version_t',
    'ze_image_view_planar_ext_version_t__enumvalues',
    'ze_init_flag_t', 'ze_init_flag_t__enumvalues', 'ze_init_flags_t',
    'ze_init_params_t', 'ze_ipc_event_pool_handle_t',
    'ze_ipc_mem_handle_t', 'ze_ipc_memory_flag_t',
    'ze_ipc_memory_flag_t__enumvalues', 'ze_ipc_memory_flags_t',
    'ze_ipc_property_flag_t', 'ze_ipc_property_flag_t__enumvalues',
    'ze_ipc_property_flags_t', 'ze_kernel_callbacks_t',
    'ze_kernel_create_params_t', 'ze_kernel_desc_t',
    'ze_kernel_destroy_params_t', 'ze_kernel_flag_t',
    'ze_kernel_flag_t__enumvalues', 'ze_kernel_flags_t',
    'ze_kernel_get_indirect_access_params_t',
    'ze_kernel_get_name_params_t',
    'ze_kernel_get_properties_params_t',
    'ze_kernel_get_source_attributes_params_t', 'ze_kernel_handle_t',
    'ze_kernel_indirect_access_flag_t',
    'ze_kernel_indirect_access_flag_t__enumvalues',
    'ze_kernel_indirect_access_flags_t',
    'ze_kernel_max_group_size_ext_properties_t',
    'ze_kernel_max_group_size_properties_ext_t',
    'ze_kernel_max_group_size_properties_ext_version_t',
    'ze_kernel_max_group_size_properties_ext_version_t__enumvalues',
    'ze_kernel_preferred_group_size_properties_t',
    'ze_kernel_properties_t', 'ze_kernel_set_argument_value_params_t',
    'ze_kernel_set_cache_config_params_t',
    'ze_kernel_set_group_size_params_t',
    'ze_kernel_set_indirect_access_params_t',
    'ze_kernel_suggest_group_size_params_t',
    'ze_kernel_suggest_max_cooperative_group_count_params_t',
    'ze_kernel_timestamp_data_t', 'ze_kernel_timestamp_result_t',
    'ze_kernel_uuid_t', 'ze_latency_unit_t',
    'ze_latency_unit_t__enumvalues',
    'ze_linkage_inspection_ext_desc_t',
    'ze_linkage_inspection_ext_flag_t',
    'ze_linkage_inspection_ext_flag_t__enumvalues',
    'ze_linkage_inspection_ext_flags_t',
    'ze_linkage_inspection_ext_version_t',
    'ze_linkage_inspection_ext_version_t__enumvalues',
    'ze_linkonce_odr_ext_version_t',
    'ze_linkonce_odr_ext_version_t__enumvalues',
    'ze_mem_alloc_device_params_t', 'ze_mem_alloc_host_params_t',
    'ze_mem_alloc_shared_params_t', 'ze_mem_callbacks_t',
    'ze_mem_close_ipc_handle_params_t', 'ze_mem_free_params_t',
    'ze_mem_get_address_range_params_t',
    'ze_mem_get_alloc_properties_params_t',
    'ze_mem_get_ipc_handle_params_t',
    'ze_mem_open_ipc_handle_params_t', 'ze_memory_access_attribute_t',
    'ze_memory_access_attribute_t__enumvalues',
    'ze_memory_access_cap_flag_t',
    'ze_memory_access_cap_flag_t__enumvalues',
    'ze_memory_access_cap_flags_t', 'ze_memory_advice_t',
    'ze_memory_advice_t__enumvalues',
    'ze_memory_allocation_properties_t',
    'ze_memory_atomic_attr_exp_flag_t',
    'ze_memory_atomic_attr_exp_flag_t__enumvalues',
    'ze_memory_atomic_attr_exp_flags_t',
    'ze_memory_compression_hints_ext_desc_t',
    'ze_memory_compression_hints_ext_flag_t',
    'ze_memory_compression_hints_ext_flag_t__enumvalues',
    'ze_memory_compression_hints_ext_flags_t',
    'ze_memory_compression_hints_ext_version_t',
    'ze_memory_compression_hints_ext_version_t__enumvalues',
    'ze_memory_free_ext_desc_t',
    'ze_memory_free_policies_ext_version_t',
    'ze_memory_free_policies_ext_version_t__enumvalues',
    'ze_memory_sub_allocations_exp_properties_t', 'ze_memory_type_t',
    'ze_memory_type_t__enumvalues', 'ze_module_build_log_callbacks_t',
    'ze_module_build_log_destroy_params_t',
    'ze_module_build_log_get_string_params_t',
    'ze_module_build_log_handle_t', 'ze_module_callbacks_t',
    'ze_module_constants_t', 'ze_module_create_params_t',
    'ze_module_desc_t', 'ze_module_destroy_params_t',
    'ze_module_dynamic_link_params_t', 'ze_module_format_t',
    'ze_module_format_t__enumvalues',
    'ze_module_get_function_pointer_params_t',
    'ze_module_get_global_pointer_params_t',
    'ze_module_get_kernel_names_params_t',
    'ze_module_get_native_binary_params_t',
    'ze_module_get_properties_params_t', 'ze_module_handle_t',
    'ze_module_program_exp_desc_t', 'ze_module_program_exp_version_t',
    'ze_module_program_exp_version_t__enumvalues',
    'ze_module_properties_t', 'ze_module_property_flag_t',
    'ze_module_property_flag_t__enumvalues',
    'ze_module_property_flags_t', 'ze_native_kernel_uuid_t',
    'ze_pci_address_ext_t', 'ze_pci_ext_properties_t',
    'ze_pci_properties_ext_version_t',
    'ze_pci_properties_ext_version_t__enumvalues',
    'ze_pci_speed_ext_t', 'ze_pfnCommandListAppendBarrierCb_t',
    'ze_pfnCommandListAppendEventResetCb_t',
    'ze_pfnCommandListAppendImageCopyCb_t',
    'ze_pfnCommandListAppendImageCopyFromMemoryCb_t',
    'ze_pfnCommandListAppendImageCopyRegionCb_t',
    'ze_pfnCommandListAppendImageCopyToMemoryCb_t',
    'ze_pfnCommandListAppendLaunchCooperativeKernelCb_t',
    'ze_pfnCommandListAppendLaunchKernelCb_t',
    'ze_pfnCommandListAppendLaunchKernelIndirectCb_t',
    'ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t',
    'ze_pfnCommandListAppendMemAdviseCb_t',
    'ze_pfnCommandListAppendMemoryCopyCb_t',
    'ze_pfnCommandListAppendMemoryCopyFromContextCb_t',
    'ze_pfnCommandListAppendMemoryCopyRegionCb_t',
    'ze_pfnCommandListAppendMemoryFillCb_t',
    'ze_pfnCommandListAppendMemoryPrefetchCb_t',
    'ze_pfnCommandListAppendMemoryRangesBarrierCb_t',
    'ze_pfnCommandListAppendQueryKernelTimestampsCb_t',
    'ze_pfnCommandListAppendSignalEventCb_t',
    'ze_pfnCommandListAppendWaitOnEventsCb_t',
    'ze_pfnCommandListAppendWriteGlobalTimestampCb_t',
    'ze_pfnCommandListCloseCb_t', 'ze_pfnCommandListCreateCb_t',
    'ze_pfnCommandListCreateImmediateCb_t',
    'ze_pfnCommandListDestroyCb_t', 'ze_pfnCommandListResetCb_t',
    'ze_pfnCommandQueueCreateCb_t', 'ze_pfnCommandQueueDestroyCb_t',
    'ze_pfnCommandQueueExecuteCommandListsCb_t',
    'ze_pfnCommandQueueSynchronizeCb_t', 'ze_pfnContextCreateCb_t',
    'ze_pfnContextDestroyCb_t', 'ze_pfnContextEvictImageCb_t',
    'ze_pfnContextEvictMemoryCb_t', 'ze_pfnContextGetStatusCb_t',
    'ze_pfnContextMakeImageResidentCb_t',
    'ze_pfnContextMakeMemoryResidentCb_t',
    'ze_pfnContextSystemBarrierCb_t', 'ze_pfnDeviceCanAccessPeerCb_t',
    'ze_pfnDeviceGetCachePropertiesCb_t', 'ze_pfnDeviceGetCb_t',
    'ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t',
    'ze_pfnDeviceGetComputePropertiesCb_t',
    'ze_pfnDeviceGetExternalMemoryPropertiesCb_t',
    'ze_pfnDeviceGetImagePropertiesCb_t',
    'ze_pfnDeviceGetMemoryAccessPropertiesCb_t',
    'ze_pfnDeviceGetMemoryPropertiesCb_t',
    'ze_pfnDeviceGetModulePropertiesCb_t',
    'ze_pfnDeviceGetP2PPropertiesCb_t',
    'ze_pfnDeviceGetPropertiesCb_t', 'ze_pfnDeviceGetStatusCb_t',
    'ze_pfnDeviceGetSubDevicesCb_t', 'ze_pfnDriverGetApiVersionCb_t',
    'ze_pfnDriverGetCb_t', 'ze_pfnDriverGetExtensionPropertiesCb_t',
    'ze_pfnDriverGetIpcPropertiesCb_t',
    'ze_pfnDriverGetPropertiesCb_t', 'ze_pfnEventCreateCb_t',
    'ze_pfnEventDestroyCb_t', 'ze_pfnEventHostResetCb_t',
    'ze_pfnEventHostSignalCb_t', 'ze_pfnEventHostSynchronizeCb_t',
    'ze_pfnEventPoolCloseIpcHandleCb_t', 'ze_pfnEventPoolCreateCb_t',
    'ze_pfnEventPoolDestroyCb_t', 'ze_pfnEventPoolGetIpcHandleCb_t',
    'ze_pfnEventPoolOpenIpcHandleCb_t',
    'ze_pfnEventQueryKernelTimestampCb_t',
    'ze_pfnEventQueryStatusCb_t', 'ze_pfnFenceCreateCb_t',
    'ze_pfnFenceDestroyCb_t', 'ze_pfnFenceHostSynchronizeCb_t',
    'ze_pfnFenceQueryStatusCb_t', 'ze_pfnFenceResetCb_t',
    'ze_pfnImageCreateCb_t', 'ze_pfnImageDestroyCb_t',
    'ze_pfnImageGetPropertiesCb_t', 'ze_pfnInitCb_t',
    'ze_pfnKernelCreateCb_t', 'ze_pfnKernelDestroyCb_t',
    'ze_pfnKernelGetIndirectAccessCb_t', 'ze_pfnKernelGetNameCb_t',
    'ze_pfnKernelGetPropertiesCb_t',
    'ze_pfnKernelGetSourceAttributesCb_t',
    'ze_pfnKernelSetArgumentValueCb_t',
    'ze_pfnKernelSetCacheConfigCb_t', 'ze_pfnKernelSetGroupSizeCb_t',
    'ze_pfnKernelSetIndirectAccessCb_t',
    'ze_pfnKernelSuggestGroupSizeCb_t',
    'ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t',
    'ze_pfnMemAllocDeviceCb_t', 'ze_pfnMemAllocHostCb_t',
    'ze_pfnMemAllocSharedCb_t', 'ze_pfnMemCloseIpcHandleCb_t',
    'ze_pfnMemFreeCb_t', 'ze_pfnMemGetAddressRangeCb_t',
    'ze_pfnMemGetAllocPropertiesCb_t', 'ze_pfnMemGetIpcHandleCb_t',
    'ze_pfnMemOpenIpcHandleCb_t', 'ze_pfnModuleBuildLogDestroyCb_t',
    'ze_pfnModuleBuildLogGetStringCb_t', 'ze_pfnModuleCreateCb_t',
    'ze_pfnModuleDestroyCb_t', 'ze_pfnModuleDynamicLinkCb_t',
    'ze_pfnModuleGetFunctionPointerCb_t',
    'ze_pfnModuleGetGlobalPointerCb_t',
    'ze_pfnModuleGetKernelNamesCb_t',
    'ze_pfnModuleGetNativeBinaryCb_t',
    'ze_pfnModuleGetPropertiesCb_t', 'ze_pfnPhysicalMemCreateCb_t',
    'ze_pfnPhysicalMemDestroyCb_t', 'ze_pfnSamplerCreateCb_t',
    'ze_pfnSamplerDestroyCb_t', 'ze_pfnVirtualMemFreeCb_t',
    'ze_pfnVirtualMemGetAccessAttributeCb_t',
    'ze_pfnVirtualMemMapCb_t', 'ze_pfnVirtualMemQueryPageSizeCb_t',
    'ze_pfnVirtualMemReserveCb_t',
    'ze_pfnVirtualMemSetAccessAttributeCb_t',
    'ze_pfnVirtualMemUnmapCb_t', 'ze_physical_mem_callbacks_t',
    'ze_physical_mem_create_params_t', 'ze_physical_mem_desc_t',
    'ze_physical_mem_destroy_params_t', 'ze_physical_mem_flag_t',
    'ze_physical_mem_flag_t__enumvalues', 'ze_physical_mem_flags_t',
    'ze_physical_mem_handle_t', 'ze_power_saving_hint_exp_version_t',
    'ze_power_saving_hint_exp_version_t__enumvalues',
    'ze_power_saving_hint_type_t',
    'ze_power_saving_hint_type_t__enumvalues',
    'ze_raytracing_ext_version_t',
    'ze_raytracing_ext_version_t__enumvalues',
    'ze_raytracing_mem_alloc_ext_desc_t',
    'ze_raytracing_mem_alloc_ext_flag_t',
    'ze_raytracing_mem_alloc_ext_flag_t__enumvalues',
    'ze_raytracing_mem_alloc_ext_flags_t',
    'ze_relaxed_allocation_limits_exp_desc_t',
    'ze_relaxed_allocation_limits_exp_flag_t',
    'ze_relaxed_allocation_limits_exp_flag_t__enumvalues',
    'ze_relaxed_allocation_limits_exp_flags_t',
    'ze_relaxed_allocation_limits_exp_version_t',
    'ze_relaxed_allocation_limits_exp_version_t__enumvalues',
    'ze_result_t', 'ze_result_t__enumvalues', 'ze_rtas_aabb_exp_t',
    'ze_rtas_builder_build_op_exp_desc_t',
    'ze_rtas_builder_build_op_exp_flag_t',
    'ze_rtas_builder_build_op_exp_flag_t__enumvalues',
    'ze_rtas_builder_build_op_exp_flags_t',
    'ze_rtas_builder_build_quality_hint_exp_t',
    'ze_rtas_builder_build_quality_hint_exp_t__enumvalues',
    'ze_rtas_builder_exp_desc_t', 'ze_rtas_builder_exp_flag_t',
    'ze_rtas_builder_exp_flag_t__enumvalues',
    'ze_rtas_builder_exp_flags_t', 'ze_rtas_builder_exp_handle_t',
    'ze_rtas_builder_exp_properties_t',
    'ze_rtas_builder_exp_version_t',
    'ze_rtas_builder_exp_version_t__enumvalues',
    'ze_rtas_builder_geometry_exp_flag_t',
    'ze_rtas_builder_geometry_exp_flag_t__enumvalues',
    'ze_rtas_builder_geometry_exp_flags_t',
    'ze_rtas_builder_geometry_info_exp_t',
    'ze_rtas_builder_geometry_type_exp_t',
    'ze_rtas_builder_geometry_type_exp_t__enumvalues',
    'ze_rtas_builder_input_data_format_exp_t',
    'ze_rtas_builder_input_data_format_exp_t__enumvalues',
    'ze_rtas_builder_instance_exp_flag_t',
    'ze_rtas_builder_instance_exp_flag_t__enumvalues',
    'ze_rtas_builder_instance_exp_flags_t',
    'ze_rtas_builder_instance_geometry_info_exp_t',
    'ze_rtas_builder_packed_geometry_exp_flags_t',
    'ze_rtas_builder_packed_geometry_type_exp_t',
    'ze_rtas_builder_packed_input_data_format_exp_t',
    'ze_rtas_builder_packed_instance_exp_flags_t',
    'ze_rtas_builder_procedural_geometry_info_exp_t',
    'ze_rtas_builder_quads_geometry_info_exp_t',
    'ze_rtas_builder_triangles_geometry_info_exp_t',
    'ze_rtas_device_exp_flag_t',
    'ze_rtas_device_exp_flag_t__enumvalues',
    'ze_rtas_device_exp_flags_t', 'ze_rtas_device_exp_properties_t',
    'ze_rtas_float3_exp_t', 'ze_rtas_format_exp_t',
    'ze_rtas_format_exp_t__enumvalues',
    'ze_rtas_geometry_aabbs_cb_exp_t',
    'ze_rtas_geometry_aabbs_exp_cb_params_t',
    'ze_rtas_parallel_operation_exp_flag_t',
    'ze_rtas_parallel_operation_exp_flag_t__enumvalues',
    'ze_rtas_parallel_operation_exp_flags_t',
    'ze_rtas_parallel_operation_exp_handle_t',
    'ze_rtas_parallel_operation_exp_properties_t',
    'ze_rtas_quad_indices_uint32_exp_t',
    'ze_rtas_transform_float3x4_aligned_column_major_exp_t',
    'ze_rtas_transform_float3x4_column_major_exp_t',
    'ze_rtas_transform_float3x4_row_major_exp_t',
    'ze_rtas_triangle_indices_uint32_exp_t',
    'ze_sampler_address_mode_t',
    'ze_sampler_address_mode_t__enumvalues', 'ze_sampler_callbacks_t',
    'ze_sampler_create_params_t', 'ze_sampler_desc_t',
    'ze_sampler_destroy_params_t', 'ze_sampler_filter_mode_t',
    'ze_sampler_filter_mode_t__enumvalues', 'ze_sampler_handle_t',
    'ze_scheduling_hint_exp_desc_t', 'ze_scheduling_hint_exp_flag_t',
    'ze_scheduling_hint_exp_flag_t__enumvalues',
    'ze_scheduling_hint_exp_flags_t',
    'ze_scheduling_hint_exp_properties_t',
    'ze_scheduling_hints_exp_version_t',
    'ze_scheduling_hints_exp_version_t__enumvalues',
    'ze_srgb_ext_desc_t', 'ze_srgb_ext_version_t',
    'ze_srgb_ext_version_t__enumvalues', 'ze_structure_type_t',
    'ze_structure_type_t__enumvalues', 'ze_sub_allocation_t',
    'ze_sub_allocations_exp_version_t',
    'ze_sub_allocations_exp_version_t__enumvalues',
    'ze_subgroup_ext_version_t',
    'ze_subgroup_ext_version_t__enumvalues',
    'ze_synchronized_timestamp_data_ext_t',
    'ze_synchronized_timestamp_result_ext_t', 'ze_uuid_t',
    'ze_virtual_mem_callbacks_t', 'ze_virtual_mem_free_params_t',
    'ze_virtual_mem_get_access_attribute_params_t',
    'ze_virtual_mem_map_params_t',
    'ze_virtual_mem_query_page_size_params_t',
    'ze_virtual_mem_reserve_params_t',
    'ze_virtual_mem_set_access_attribute_params_t',
    'ze_virtual_mem_unmap_params_t', 'zelLoaderGetVersions',
    'zelLoaderTranslateHandle', 'zelSetDriverTeardown',
    'zel_component_version_t', 'zel_handle_type_t',
    'zel_handle_type_t__enumvalues', 'zel_version_t']
