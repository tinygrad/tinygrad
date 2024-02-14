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





class struct_kfd_ioctl_get_version_args(Structure):
    pass

struct_kfd_ioctl_get_version_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_version_args._fields_ = [
    ('major_version', ctypes.c_uint32),
    ('minor_version', ctypes.c_uint32),
]

class struct_kfd_ioctl_create_queue_args(Structure):
    pass

struct_kfd_ioctl_create_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_create_queue_args._fields_ = [
    ('ring_base_address', ctypes.c_uint64),
    ('write_pointer_address', ctypes.c_uint64),
    ('read_pointer_address', ctypes.c_uint64),
    ('doorbell_offset', ctypes.c_uint64),
    ('ring_size', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('queue_type', ctypes.c_uint32),
    ('queue_percentage', ctypes.c_uint32),
    ('queue_priority', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('eop_buffer_address', ctypes.c_uint64),
    ('eop_buffer_size', ctypes.c_uint64),
    ('ctx_save_restore_address', ctypes.c_uint64),
    ('ctx_save_restore_size', ctypes.c_uint32),
    ('ctl_stack_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_destroy_queue_args(Structure):
    pass

struct_kfd_ioctl_destroy_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_queue_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_update_queue_args(Structure):
    pass

struct_kfd_ioctl_update_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_update_queue_args._fields_ = [
    ('ring_base_address', ctypes.c_uint64),
    ('queue_id', ctypes.c_uint32),
    ('ring_size', ctypes.c_uint32),
    ('queue_percentage', ctypes.c_uint32),
    ('queue_priority', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_cu_mask_args(Structure):
    pass

struct_kfd_ioctl_set_cu_mask_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_cu_mask_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_cu_mask', ctypes.c_uint32),
    ('cu_mask_ptr', ctypes.c_uint64),
]

class struct_kfd_ioctl_get_queue_wave_state_args(Structure):
    pass

struct_kfd_ioctl_get_queue_wave_state_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_queue_wave_state_args._fields_ = [
    ('ctl_stack_address', ctypes.c_uint64),
    ('ctl_stack_used_size', ctypes.c_uint32),
    ('save_area_used_size', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_queue_snapshot_entry(Structure):
    pass

struct_kfd_queue_snapshot_entry._pack_ = 1 # source:False
struct_kfd_queue_snapshot_entry._fields_ = [
    ('exception_status', ctypes.c_uint64),
    ('ring_base_address', ctypes.c_uint64),
    ('write_pointer_address', ctypes.c_uint64),
    ('read_pointer_address', ctypes.c_uint64),
    ('ctx_save_restore_address', ctypes.c_uint64),
    ('queue_id', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('ring_size', ctypes.c_uint32),
    ('queue_type', ctypes.c_uint32),
    ('ctx_save_restore_area_size', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_kfd_dbg_device_info_entry(Structure):
    pass

struct_kfd_dbg_device_info_entry._pack_ = 1 # source:False
struct_kfd_dbg_device_info_entry._fields_ = [
    ('exception_status', ctypes.c_uint64),
    ('lds_base', ctypes.c_uint64),
    ('lds_limit', ctypes.c_uint64),
    ('scratch_base', ctypes.c_uint64),
    ('scratch_limit', ctypes.c_uint64),
    ('gpuvm_base', ctypes.c_uint64),
    ('gpuvm_limit', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('location_id', ctypes.c_uint32),
    ('vendor_id', ctypes.c_uint32),
    ('device_id', ctypes.c_uint32),
    ('revision_id', ctypes.c_uint32),
    ('subsystem_vendor_id', ctypes.c_uint32),
    ('subsystem_device_id', ctypes.c_uint32),
    ('fw_version', ctypes.c_uint32),
    ('gfx_target_version', ctypes.c_uint32),
    ('simd_count', ctypes.c_uint32),
    ('max_waves_per_simd', ctypes.c_uint32),
    ('array_count', ctypes.c_uint32),
    ('simd_arrays_per_engine', ctypes.c_uint32),
    ('num_xcc', ctypes.c_uint32),
    ('capability', ctypes.c_uint32),
    ('debug_prop', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_memory_policy_args(Structure):
    pass

struct_kfd_ioctl_set_memory_policy_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_memory_policy_args._fields_ = [
    ('alternate_aperture_base', ctypes.c_uint64),
    ('alternate_aperture_size', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('default_policy', ctypes.c_uint32),
    ('alternate_policy', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_clock_counters_args(Structure):
    pass

struct_kfd_ioctl_get_clock_counters_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_clock_counters_args._fields_ = [
    ('gpu_clock_counter', ctypes.c_uint64),
    ('cpu_clock_counter', ctypes.c_uint64),
    ('system_clock_counter', ctypes.c_uint64),
    ('system_clock_freq', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_process_device_apertures(Structure):
    pass

struct_kfd_process_device_apertures._pack_ = 1 # source:False
struct_kfd_process_device_apertures._fields_ = [
    ('lds_base', ctypes.c_uint64),
    ('lds_limit', ctypes.c_uint64),
    ('scratch_base', ctypes.c_uint64),
    ('scratch_limit', ctypes.c_uint64),
    ('gpuvm_base', ctypes.c_uint64),
    ('gpuvm_limit', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_process_apertures_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_args._fields_ = [
    ('process_apertures', struct_kfd_process_device_apertures * 7),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_process_apertures_new_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_new_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_new_args._fields_ = [
    ('kfd_process_device_apertures_ptr', ctypes.c_uint64),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_register_args(Structure):
    pass

struct_kfd_ioctl_dbg_register_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_register_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_unregister_args(Structure):
    pass

struct_kfd_ioctl_dbg_unregister_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_unregister_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_address_watch_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_wave_control_args(Structure):
    pass

struct_kfd_ioctl_dbg_wave_control_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_wave_control_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]


# values for enumeration 'kfd_dbg_trap_override_mode'
kfd_dbg_trap_override_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_OVERRIDE_OR',
    1: 'KFD_DBG_TRAP_OVERRIDE_REPLACE',
}
KFD_DBG_TRAP_OVERRIDE_OR = 0
KFD_DBG_TRAP_OVERRIDE_REPLACE = 1
kfd_dbg_trap_override_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_mask'
kfd_dbg_trap_mask__enumvalues = {
    1: 'KFD_DBG_TRAP_MASK_FP_INVALID',
    2: 'KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL',
    4: 'KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    8: 'KFD_DBG_TRAP_MASK_FP_OVERFLOW',
    16: 'KFD_DBG_TRAP_MASK_FP_UNDERFLOW',
    32: 'KFD_DBG_TRAP_MASK_FP_INEXACT',
    64: 'KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    128: 'KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    256: 'KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
    1073741824: 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START',
    -2147483648: 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END',
}
KFD_DBG_TRAP_MASK_FP_INVALID = 1
KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL = 2
KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO = 4
KFD_DBG_TRAP_MASK_FP_OVERFLOW = 8
KFD_DBG_TRAP_MASK_FP_UNDERFLOW = 16
KFD_DBG_TRAP_MASK_FP_INEXACT = 32
KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO = 64
KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH = 128
KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION = 256
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START = 1073741824
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END = -2147483648
kfd_dbg_trap_mask = ctypes.c_int32 # enum

# values for enumeration 'kfd_dbg_trap_wave_launch_mode'
kfd_dbg_trap_wave_launch_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL',
    1: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT',
    3: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG',
}
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL = 0
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT = 1
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG = 3
kfd_dbg_trap_wave_launch_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_address_watch_mode'
kfd_dbg_trap_address_watch_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ',
    1: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD',
    2: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC',
    3: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL',
}
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ = 0
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD = 1
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC = 2
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL = 3
kfd_dbg_trap_address_watch_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_flags'
kfd_dbg_trap_flags__enumvalues = {
    1: 'KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP',
}
KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP = 1
kfd_dbg_trap_flags = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_exception_code'
kfd_dbg_trap_exception_code__enumvalues = {
    0: 'EC_NONE',
    1: 'EC_QUEUE_WAVE_ABORT',
    2: 'EC_QUEUE_WAVE_TRAP',
    3: 'EC_QUEUE_WAVE_MATH_ERROR',
    4: 'EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION',
    5: 'EC_QUEUE_WAVE_MEMORY_VIOLATION',
    6: 'EC_QUEUE_WAVE_APERTURE_VIOLATION',
    16: 'EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    17: 'EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    18: 'EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    19: 'EC_QUEUE_PACKET_RESERVED',
    20: 'EC_QUEUE_PACKET_UNSUPPORTED',
    21: 'EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    22: 'EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    23: 'EC_QUEUE_PACKET_VENDOR_UNSUPPORTED',
    30: 'EC_QUEUE_PREEMPTION_ERROR',
    31: 'EC_QUEUE_NEW',
    32: 'EC_DEVICE_QUEUE_DELETE',
    33: 'EC_DEVICE_MEMORY_VIOLATION',
    34: 'EC_DEVICE_RAS_ERROR',
    35: 'EC_DEVICE_FATAL_HALT',
    36: 'EC_DEVICE_NEW',
    48: 'EC_PROCESS_RUNTIME',
    49: 'EC_PROCESS_DEVICE_REMOVE',
    50: 'EC_MAX',
}
EC_NONE = 0
EC_QUEUE_WAVE_ABORT = 1
EC_QUEUE_WAVE_TRAP = 2
EC_QUEUE_WAVE_MATH_ERROR = 3
EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION = 4
EC_QUEUE_WAVE_MEMORY_VIOLATION = 5
EC_QUEUE_WAVE_APERTURE_VIOLATION = 6
EC_QUEUE_PACKET_DISPATCH_DIM_INVALID = 16
EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = 17
EC_QUEUE_PACKET_DISPATCH_CODE_INVALID = 18
EC_QUEUE_PACKET_RESERVED = 19
EC_QUEUE_PACKET_UNSUPPORTED = 20
EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = 21
EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = 22
EC_QUEUE_PACKET_VENDOR_UNSUPPORTED = 23
EC_QUEUE_PREEMPTION_ERROR = 30
EC_QUEUE_NEW = 31
EC_DEVICE_QUEUE_DELETE = 32
EC_DEVICE_MEMORY_VIOLATION = 33
EC_DEVICE_RAS_ERROR = 34
EC_DEVICE_FATAL_HALT = 35
EC_DEVICE_NEW = 36
EC_PROCESS_RUNTIME = 48
EC_PROCESS_DEVICE_REMOVE = 49
EC_MAX = 50
kfd_dbg_trap_exception_code = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_runtime_state'
kfd_dbg_runtime_state__enumvalues = {
    0: 'DEBUG_RUNTIME_STATE_DISABLED',
    1: 'DEBUG_RUNTIME_STATE_ENABLED',
    2: 'DEBUG_RUNTIME_STATE_ENABLED_BUSY',
    3: 'DEBUG_RUNTIME_STATE_ENABLED_ERROR',
}
DEBUG_RUNTIME_STATE_DISABLED = 0
DEBUG_RUNTIME_STATE_ENABLED = 1
DEBUG_RUNTIME_STATE_ENABLED_BUSY = 2
DEBUG_RUNTIME_STATE_ENABLED_ERROR = 3
kfd_dbg_runtime_state = ctypes.c_uint32 # enum
class struct_kfd_runtime_info(Structure):
    pass

struct_kfd_runtime_info._pack_ = 1 # source:False
struct_kfd_runtime_info._fields_ = [
    ('r_debug', ctypes.c_uint64),
    ('runtime_state', ctypes.c_uint32),
    ('ttmp_setup', ctypes.c_uint32),
]

class struct_kfd_ioctl_runtime_enable_args(Structure):
    pass

struct_kfd_ioctl_runtime_enable_args._pack_ = 1 # source:False
struct_kfd_ioctl_runtime_enable_args._fields_ = [
    ('r_debug', ctypes.c_uint64),
    ('mode_mask', ctypes.c_uint32),
    ('capabilities_mask', ctypes.c_uint32),
]

class struct_kfd_context_save_area_header(Structure):
    pass

class struct_kfd_context_save_area_header_wave_state(Structure):
    pass

struct_kfd_context_save_area_header_wave_state._pack_ = 1 # source:False
struct_kfd_context_save_area_header_wave_state._fields_ = [
    ('control_stack_offset', ctypes.c_uint32),
    ('control_stack_size', ctypes.c_uint32),
    ('wave_state_offset', ctypes.c_uint32),
    ('wave_state_size', ctypes.c_uint32),
]

struct_kfd_context_save_area_header._pack_ = 1 # source:False
struct_kfd_context_save_area_header._fields_ = [
    ('wave_state', struct_kfd_context_save_area_header_wave_state),
    ('debug_offset', ctypes.c_uint32),
    ('debug_size', ctypes.c_uint32),
    ('err_payload_addr', ctypes.c_uint64),
    ('err_event_id', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
]


# values for enumeration 'kfd_dbg_trap_operations'
kfd_dbg_trap_operations__enumvalues = {
    0: 'KFD_IOC_DBG_TRAP_ENABLE',
    1: 'KFD_IOC_DBG_TRAP_DISABLE',
    2: 'KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT',
    3: 'KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED',
    4: 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE',
    5: 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE',
    6: 'KFD_IOC_DBG_TRAP_SUSPEND_QUEUES',
    7: 'KFD_IOC_DBG_TRAP_RESUME_QUEUES',
    8: 'KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH',
    9: 'KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH',
    10: 'KFD_IOC_DBG_TRAP_SET_FLAGS',
    11: 'KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT',
    12: 'KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO',
    13: 'KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT',
    14: 'KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT',
}
KFD_IOC_DBG_TRAP_ENABLE = 0
KFD_IOC_DBG_TRAP_DISABLE = 1
KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT = 2
KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED = 3
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE = 4
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE = 5
KFD_IOC_DBG_TRAP_SUSPEND_QUEUES = 6
KFD_IOC_DBG_TRAP_RESUME_QUEUES = 7
KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH = 8
KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH = 9
KFD_IOC_DBG_TRAP_SET_FLAGS = 10
KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT = 11
KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO = 12
KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT = 13
KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT = 14
kfd_dbg_trap_operations = ctypes.c_uint32 # enum
class struct_kfd_ioctl_dbg_trap_enable_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_enable_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_enable_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('rinfo_ptr', ctypes.c_uint64),
    ('rinfo_size', ctypes.c_uint32),
    ('dbg_fd', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_send_runtime_event_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_send_runtime_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_send_runtime_event_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
]

class struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args._fields_ = [
    ('override_mode', ctypes.c_uint32),
    ('enable_mask', ctypes.c_uint32),
    ('support_request_mask', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args._fields_ = [
    ('launch_mode', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_suspend_queues_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_suspend_queues_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_suspend_queues_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('queue_array_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('grace_period', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_resume_queues_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_resume_queues_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_resume_queues_args._fields_ = [
    ('queue_array_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_node_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_node_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_node_address_watch_args._fields_ = [
    ('address', ctypes.c_uint64),
    ('mode', ctypes.c_uint32),
    ('mask', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_flags_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_flags_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_flags_args._fields_ = [
    ('flags', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_query_debug_event_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_query_debug_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_query_debug_event_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_query_exception_info_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_query_exception_info_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_query_exception_info_args._fields_ = [
    ('info_ptr', ctypes.c_uint64),
    ('info_size', ctypes.c_uint32),
    ('source_id', ctypes.c_uint32),
    ('exception_code', ctypes.c_uint32),
    ('clear_exception', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_queue_snapshot_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_queue_snapshot_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_queue_snapshot_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('snapshot_buf_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('entry_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_device_snapshot_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_device_snapshot_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_device_snapshot_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('snapshot_buf_ptr', ctypes.c_uint64),
    ('num_devices', ctypes.c_uint32),
    ('entry_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_args(Structure):
    pass

class union_kfd_ioctl_dbg_trap_args_0(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('enable', struct_kfd_ioctl_dbg_trap_enable_args),
    ('send_runtime_event', struct_kfd_ioctl_dbg_trap_send_runtime_event_args),
    ('set_exceptions_enabled', struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args),
    ('launch_override', struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args),
    ('launch_mode', struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args),
    ('suspend_queues', struct_kfd_ioctl_dbg_trap_suspend_queues_args),
    ('resume_queues', struct_kfd_ioctl_dbg_trap_resume_queues_args),
    ('set_node_address_watch', struct_kfd_ioctl_dbg_trap_set_node_address_watch_args),
    ('clear_node_address_watch', struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args),
    ('set_flags', struct_kfd_ioctl_dbg_trap_set_flags_args),
    ('query_debug_event', struct_kfd_ioctl_dbg_trap_query_debug_event_args),
    ('query_exception_info', struct_kfd_ioctl_dbg_trap_query_exception_info_args),
    ('queue_snapshot', struct_kfd_ioctl_dbg_trap_queue_snapshot_args),
    ('device_snapshot', struct_kfd_ioctl_dbg_trap_device_snapshot_args),
     ]

struct_kfd_ioctl_dbg_trap_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_args._anonymous_ = ('_0',)
struct_kfd_ioctl_dbg_trap_args._fields_ = [
    ('pid', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('_0', union_kfd_ioctl_dbg_trap_args_0),
]

class struct_kfd_ioctl_create_event_args(Structure):
    pass

struct_kfd_ioctl_create_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_create_event_args._fields_ = [
    ('event_page_offset', ctypes.c_uint64),
    ('event_trigger_data', ctypes.c_uint32),
    ('event_type', ctypes.c_uint32),
    ('auto_reset', ctypes.c_uint32),
    ('node_id', ctypes.c_uint32),
    ('event_id', ctypes.c_uint32),
    ('event_slot_index', ctypes.c_uint32),
]

class struct_kfd_ioctl_destroy_event_args(Structure):
    pass

struct_kfd_ioctl_destroy_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_event_args(Structure):
    pass

struct_kfd_ioctl_set_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_reset_event_args(Structure):
    pass

struct_kfd_ioctl_reset_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_reset_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_memory_exception_failure(Structure):
    pass

struct_kfd_memory_exception_failure._pack_ = 1 # source:False
struct_kfd_memory_exception_failure._fields_ = [
    ('NotPresent', ctypes.c_uint32),
    ('ReadOnly', ctypes.c_uint32),
    ('NoExecute', ctypes.c_uint32),
    ('imprecise', ctypes.c_uint32),
]

class struct_kfd_hsa_memory_exception_data(Structure):
    pass

struct_kfd_hsa_memory_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_memory_exception_data._fields_ = [
    ('failure', struct_kfd_memory_exception_failure),
    ('va', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('ErrorType', ctypes.c_uint32),
]

class struct_kfd_hsa_hw_exception_data(Structure):
    pass

struct_kfd_hsa_hw_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_hw_exception_data._fields_ = [
    ('reset_type', ctypes.c_uint32),
    ('reset_cause', ctypes.c_uint32),
    ('memory_lost', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

class struct_kfd_hsa_signal_event_data(Structure):
    pass

struct_kfd_hsa_signal_event_data._pack_ = 1 # source:False
struct_kfd_hsa_signal_event_data._fields_ = [
    ('last_event_age', ctypes.c_uint64),
]

class struct_kfd_event_data(Structure):
    pass

class union_kfd_event_data_0(Union):
    pass

union_kfd_event_data_0._pack_ = 1 # source:False
union_kfd_event_data_0._fields_ = [
    ('memory_exception_data', struct_kfd_hsa_memory_exception_data),
    ('hw_exception_data', struct_kfd_hsa_hw_exception_data),
    ('signal_event_data', struct_kfd_hsa_signal_event_data),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

struct_kfd_event_data._pack_ = 1 # source:False
struct_kfd_event_data._anonymous_ = ('_0',)
struct_kfd_event_data._fields_ = [
    ('_0', union_kfd_event_data_0),
    ('kfd_event_data_ext', ctypes.c_uint64),
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_wait_events_args(Structure):
    pass

struct_kfd_ioctl_wait_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_wait_events_args._fields_ = [
    ('events_ptr', ctypes.c_uint64),
    ('num_events', ctypes.c_uint32),
    ('wait_for_all', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('wait_result', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_scratch_backing_va_args(Structure):
    pass

struct_kfd_ioctl_set_scratch_backing_va_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_scratch_backing_va_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_tile_config_args(Structure):
    pass

struct_kfd_ioctl_get_tile_config_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_tile_config_args._fields_ = [
    ('tile_config_ptr', ctypes.c_uint64),
    ('macro_tile_config_ptr', ctypes.c_uint64),
    ('num_tile_configs', ctypes.c_uint32),
    ('num_macro_tile_configs', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('gb_addr_config', ctypes.c_uint32),
    ('num_banks', ctypes.c_uint32),
    ('num_ranks', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_trap_handler_args(Structure):
    pass

struct_kfd_ioctl_set_trap_handler_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_trap_handler_args._fields_ = [
    ('tba_addr', ctypes.c_uint64),
    ('tma_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_acquire_vm_args(Structure):
    pass

struct_kfd_ioctl_acquire_vm_args._pack_ = 1 # source:False
struct_kfd_ioctl_acquire_vm_args._fields_ = [
    ('drm_fd', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_alloc_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_alloc_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_memory_of_gpu_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('mmap_offset', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_kfd_ioctl_free_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_free_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_free_memory_of_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
]

class struct_kfd_ioctl_get_available_memory_args(Structure):
    pass

struct_kfd_ioctl_get_available_memory_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_available_memory_args._fields_ = [
    ('available', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_map_memory_to_gpu_args(Structure):
    pass

struct_kfd_ioctl_map_memory_to_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_map_memory_to_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

class struct_kfd_ioctl_unmap_memory_from_gpu_args(Structure):
    pass

struct_kfd_ioctl_unmap_memory_from_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_unmap_memory_from_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

class struct_kfd_ioctl_alloc_queue_gws_args(Structure):
    pass

struct_kfd_ioctl_alloc_queue_gws_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_queue_gws_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_gws', ctypes.c_uint32),
    ('first_gws', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_dmabuf_info_args(Structure):
    pass

struct_kfd_ioctl_get_dmabuf_info_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_dmabuf_info_args._fields_ = [
    ('size', ctypes.c_uint64),
    ('metadata_ptr', ctypes.c_uint64),
    ('metadata_size', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

class struct_kfd_ioctl_import_dmabuf_args(Structure):
    pass

struct_kfd_ioctl_import_dmabuf_args._pack_ = 1 # source:False
struct_kfd_ioctl_import_dmabuf_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

class struct_kfd_ioctl_export_dmabuf_args(Structure):
    pass

struct_kfd_ioctl_export_dmabuf_args._pack_ = 1 # source:False
struct_kfd_ioctl_export_dmabuf_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]


# values for enumeration 'kfd_smi_event'
kfd_smi_event__enumvalues = {
    0: 'KFD_SMI_EVENT_NONE',
    1: 'KFD_SMI_EVENT_VMFAULT',
    2: 'KFD_SMI_EVENT_THERMAL_THROTTLE',
    3: 'KFD_SMI_EVENT_GPU_PRE_RESET',
    4: 'KFD_SMI_EVENT_GPU_POST_RESET',
}
KFD_SMI_EVENT_NONE = 0
KFD_SMI_EVENT_VMFAULT = 1
KFD_SMI_EVENT_THERMAL_THROTTLE = 2
KFD_SMI_EVENT_GPU_PRE_RESET = 3
KFD_SMI_EVENT_GPU_POST_RESET = 4
kfd_smi_event = ctypes.c_uint32 # enum
class struct_kfd_ioctl_smi_events_args(Structure):
    pass

struct_kfd_ioctl_smi_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_smi_events_args._fields_ = [
    ('gpuid', ctypes.c_uint32),
    ('anon_fd', ctypes.c_uint32),
]


# values for enumeration 'kfd_ioctl_spm_op'
kfd_ioctl_spm_op__enumvalues = {
    0: 'KFD_IOCTL_SPM_OP_ACQUIRE',
    1: 'KFD_IOCTL_SPM_OP_RELEASE',
    2: 'KFD_IOCTL_SPM_OP_SET_DEST_BUF',
}
KFD_IOCTL_SPM_OP_ACQUIRE = 0
KFD_IOCTL_SPM_OP_RELEASE = 1
KFD_IOCTL_SPM_OP_SET_DEST_BUF = 2
kfd_ioctl_spm_op = ctypes.c_uint32 # enum
class struct_kfd_ioctl_spm_args(Structure):
    pass

struct_kfd_ioctl_spm_args._pack_ = 1 # source:False
struct_kfd_ioctl_spm_args._fields_ = [
    ('dest_buf', ctypes.c_uint64),
    ('buf_size', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('bytes_copied', ctypes.c_uint32),
    ('has_data_loss', ctypes.c_uint32),
]


# values for enumeration 'kfd_criu_op'
kfd_criu_op__enumvalues = {
    0: 'KFD_CRIU_OP_PROCESS_INFO',
    1: 'KFD_CRIU_OP_CHECKPOINT',
    2: 'KFD_CRIU_OP_UNPAUSE',
    3: 'KFD_CRIU_OP_RESTORE',
    4: 'KFD_CRIU_OP_RESUME',
}
KFD_CRIU_OP_PROCESS_INFO = 0
KFD_CRIU_OP_CHECKPOINT = 1
KFD_CRIU_OP_UNPAUSE = 2
KFD_CRIU_OP_RESTORE = 3
KFD_CRIU_OP_RESUME = 4
kfd_criu_op = ctypes.c_uint32 # enum
class struct_kfd_ioctl_criu_args(Structure):
    pass

struct_kfd_ioctl_criu_args._pack_ = 1 # source:False
struct_kfd_ioctl_criu_args._fields_ = [
    ('devices', ctypes.c_uint64),
    ('bos', ctypes.c_uint64),
    ('priv_data', ctypes.c_uint64),
    ('priv_data_size', ctypes.c_uint64),
    ('num_devices', ctypes.c_uint32),
    ('num_bos', ctypes.c_uint32),
    ('num_objects', ctypes.c_uint32),
    ('pid', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kfd_criu_device_bucket(Structure):
    pass

struct_kfd_criu_device_bucket._pack_ = 1 # source:False
struct_kfd_criu_device_bucket._fields_ = [
    ('user_gpu_id', ctypes.c_uint32),
    ('actual_gpu_id', ctypes.c_uint32),
    ('drm_fd', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_criu_bo_bucket(Structure):
    pass

struct_kfd_criu_bo_bucket._pack_ = 1 # source:False
struct_kfd_criu_bo_bucket._fields_ = [
    ('addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('restored_offset', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('alloc_flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]


# values for enumeration 'kfd_mmio_remap'
kfd_mmio_remap__enumvalues = {
    0: 'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    4: 'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL',
}
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = 0
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = 4
kfd_mmio_remap = ctypes.c_uint32 # enum
class struct_kfd_ioctl_ipc_export_handle_args(Structure):
    pass

struct_kfd_ioctl_ipc_export_handle_args._pack_ = 1 # source:False
struct_kfd_ioctl_ipc_export_handle_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('share_handle', ctypes.c_uint32 * 4),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_kfd_ioctl_ipc_import_handle_args(Structure):
    pass

struct_kfd_ioctl_ipc_import_handle_args._pack_ = 1 # source:False
struct_kfd_ioctl_ipc_import_handle_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('va_addr', ctypes.c_uint64),
    ('mmap_offset', ctypes.c_uint64),
    ('share_handle', ctypes.c_uint32 * 4),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_kfd_memory_range(Structure):
    pass

struct_kfd_memory_range._pack_ = 1 # source:False
struct_kfd_memory_range._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_kfd_ioctl_cross_memory_copy_args(Structure):
    pass

struct_kfd_ioctl_cross_memory_copy_args._pack_ = 1 # source:False
struct_kfd_ioctl_cross_memory_copy_args._fields_ = [
    ('pid', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('src_mem_range_array', ctypes.c_uint64),
    ('src_mem_array_size', ctypes.c_uint64),
    ('dst_mem_range_array', ctypes.c_uint64),
    ('dst_mem_array_size', ctypes.c_uint64),
    ('bytes_copied', ctypes.c_uint64),
]


# values for enumeration 'kfd_ioctl_svm_op'
kfd_ioctl_svm_op__enumvalues = {
    0: 'KFD_IOCTL_SVM_OP_SET_ATTR',
    1: 'KFD_IOCTL_SVM_OP_GET_ATTR',
}
KFD_IOCTL_SVM_OP_SET_ATTR = 0
KFD_IOCTL_SVM_OP_GET_ATTR = 1
kfd_ioctl_svm_op = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_location'
kfd_ioctl_svm_location__enumvalues = {
    0: 'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    4294967295: 'KFD_IOCTL_SVM_LOCATION_UNDEFINED',
}
KFD_IOCTL_SVM_LOCATION_SYSMEM = 0
KFD_IOCTL_SVM_LOCATION_UNDEFINED = 4294967295
kfd_ioctl_svm_location = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_attr_type'
kfd_ioctl_svm_attr_type__enumvalues = {
    0: 'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    1: 'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC',
    2: 'KFD_IOCTL_SVM_ATTR_ACCESS',
    3: 'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    4: 'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    5: 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    6: 'KFD_IOCTL_SVM_ATTR_CLR_FLAGS',
    7: 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
}
KFD_IOCTL_SVM_ATTR_PREFERRED_LOC = 0
KFD_IOCTL_SVM_ATTR_PREFETCH_LOC = 1
KFD_IOCTL_SVM_ATTR_ACCESS = 2
KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE = 3
KFD_IOCTL_SVM_ATTR_NO_ACCESS = 4
KFD_IOCTL_SVM_ATTR_SET_FLAGS = 5
KFD_IOCTL_SVM_ATTR_CLR_FLAGS = 6
KFD_IOCTL_SVM_ATTR_GRANULARITY = 7
kfd_ioctl_svm_attr_type = ctypes.c_uint32 # enum
class struct_kfd_ioctl_svm_attribute(Structure):
    pass

struct_kfd_ioctl_svm_attribute._pack_ = 1 # source:False
struct_kfd_ioctl_svm_attribute._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

class struct_kfd_ioctl_svm_args(Structure):
    pass

struct_kfd_ioctl_svm_args._pack_ = 1 # source:False
struct_kfd_ioctl_svm_args._fields_ = [
    ('start_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('op', ctypes.c_uint32),
    ('nattr', ctypes.c_uint32),
    ('attrs', struct_kfd_ioctl_svm_attribute * 0),
]

class struct_kfd_ioctl_set_xnack_mode_args(Structure):
    pass

struct_kfd_ioctl_set_xnack_mode_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_xnack_mode_args._fields_ = [
    ('xnack_enabled', ctypes.c_int32),
]

__all__ = \
    ['DEBUG_RUNTIME_STATE_DISABLED', 'DEBUG_RUNTIME_STATE_ENABLED',
    'DEBUG_RUNTIME_STATE_ENABLED_BUSY',
    'DEBUG_RUNTIME_STATE_ENABLED_ERROR', 'EC_DEVICE_FATAL_HALT',
    'EC_DEVICE_MEMORY_VIOLATION', 'EC_DEVICE_NEW',
    'EC_DEVICE_QUEUE_DELETE', 'EC_DEVICE_RAS_ERROR', 'EC_MAX',
    'EC_NONE', 'EC_PROCESS_DEVICE_REMOVE', 'EC_PROCESS_RUNTIME',
    'EC_QUEUE_NEW', 'EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    'EC_QUEUE_PACKET_RESERVED', 'EC_QUEUE_PACKET_UNSUPPORTED',
    'EC_QUEUE_PACKET_VENDOR_UNSUPPORTED', 'EC_QUEUE_PREEMPTION_ERROR',
    'EC_QUEUE_WAVE_ABORT', 'EC_QUEUE_WAVE_APERTURE_VIOLATION',
    'EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION', 'EC_QUEUE_WAVE_MATH_ERROR',
    'EC_QUEUE_WAVE_MEMORY_VIOLATION', 'EC_QUEUE_WAVE_TRAP',
    'KFD_CRIU_OP_CHECKPOINT', 'KFD_CRIU_OP_PROCESS_INFO',
    'KFD_CRIU_OP_RESTORE', 'KFD_CRIU_OP_RESUME',
    'KFD_CRIU_OP_UNPAUSE', 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ',
    'KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP',
    'KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    'KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
    'KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    'KFD_DBG_TRAP_MASK_FP_INEXACT',
    'KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL',
    'KFD_DBG_TRAP_MASK_FP_INVALID', 'KFD_DBG_TRAP_MASK_FP_OVERFLOW',
    'KFD_DBG_TRAP_MASK_FP_UNDERFLOW',
    'KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END',
    'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START',
    'KFD_DBG_TRAP_OVERRIDE_OR', 'KFD_DBG_TRAP_OVERRIDE_REPLACE',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL',
    'KFD_IOCTL_SPM_OP_ACQUIRE', 'KFD_IOCTL_SPM_OP_RELEASE',
    'KFD_IOCTL_SPM_OP_SET_DEST_BUF', 'KFD_IOCTL_SVM_ATTR_ACCESS',
    'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    'KFD_IOCTL_SVM_ATTR_CLR_FLAGS', 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
    'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    'KFD_IOCTL_SVM_LOCATION_UNDEFINED', 'KFD_IOCTL_SVM_OP_GET_ATTR',
    'KFD_IOCTL_SVM_OP_SET_ATTR',
    'KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH',
    'KFD_IOC_DBG_TRAP_DISABLE', 'KFD_IOC_DBG_TRAP_ENABLE',
    'KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT',
    'KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT',
    'KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT',
    'KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO',
    'KFD_IOC_DBG_TRAP_RESUME_QUEUES',
    'KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT',
    'KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED',
    'KFD_IOC_DBG_TRAP_SET_FLAGS',
    'KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH',
    'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE',
    'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE',
    'KFD_IOC_DBG_TRAP_SUSPEND_QUEUES',
    'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL',
    'KFD_SMI_EVENT_GPU_POST_RESET', 'KFD_SMI_EVENT_GPU_PRE_RESET',
    'KFD_SMI_EVENT_NONE', 'KFD_SMI_EVENT_THERMAL_THROTTLE',
    'KFD_SMI_EVENT_VMFAULT', 'kfd_criu_op', 'kfd_dbg_runtime_state',
    'kfd_dbg_trap_address_watch_mode', 'kfd_dbg_trap_exception_code',
    'kfd_dbg_trap_flags', 'kfd_dbg_trap_mask',
    'kfd_dbg_trap_operations', 'kfd_dbg_trap_override_mode',
    'kfd_dbg_trap_wave_launch_mode', 'kfd_ioctl_spm_op',
    'kfd_ioctl_svm_attr_type', 'kfd_ioctl_svm_location',
    'kfd_ioctl_svm_op', 'kfd_mmio_remap', 'kfd_smi_event',
    'struct_kfd_context_save_area_header',
    'struct_kfd_context_save_area_header_wave_state',
    'struct_kfd_criu_bo_bucket', 'struct_kfd_criu_device_bucket',
    'struct_kfd_dbg_device_info_entry', 'struct_kfd_event_data',
    'struct_kfd_hsa_hw_exception_data',
    'struct_kfd_hsa_memory_exception_data',
    'struct_kfd_hsa_signal_event_data',
    'struct_kfd_ioctl_acquire_vm_args',
    'struct_kfd_ioctl_alloc_memory_of_gpu_args',
    'struct_kfd_ioctl_alloc_queue_gws_args',
    'struct_kfd_ioctl_create_event_args',
    'struct_kfd_ioctl_create_queue_args',
    'struct_kfd_ioctl_criu_args',
    'struct_kfd_ioctl_cross_memory_copy_args',
    'struct_kfd_ioctl_dbg_address_watch_args',
    'struct_kfd_ioctl_dbg_register_args',
    'struct_kfd_ioctl_dbg_trap_args',
    'struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args',
    'struct_kfd_ioctl_dbg_trap_device_snapshot_args',
    'struct_kfd_ioctl_dbg_trap_enable_args',
    'struct_kfd_ioctl_dbg_trap_query_debug_event_args',
    'struct_kfd_ioctl_dbg_trap_query_exception_info_args',
    'struct_kfd_ioctl_dbg_trap_queue_snapshot_args',
    'struct_kfd_ioctl_dbg_trap_resume_queues_args',
    'struct_kfd_ioctl_dbg_trap_send_runtime_event_args',
    'struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args',
    'struct_kfd_ioctl_dbg_trap_set_flags_args',
    'struct_kfd_ioctl_dbg_trap_set_node_address_watch_args',
    'struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args',
    'struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args',
    'struct_kfd_ioctl_dbg_trap_suspend_queues_args',
    'struct_kfd_ioctl_dbg_unregister_args',
    'struct_kfd_ioctl_dbg_wave_control_args',
    'struct_kfd_ioctl_destroy_event_args',
    'struct_kfd_ioctl_destroy_queue_args',
    'struct_kfd_ioctl_export_dmabuf_args',
    'struct_kfd_ioctl_free_memory_of_gpu_args',
    'struct_kfd_ioctl_get_available_memory_args',
    'struct_kfd_ioctl_get_clock_counters_args',
    'struct_kfd_ioctl_get_dmabuf_info_args',
    'struct_kfd_ioctl_get_process_apertures_args',
    'struct_kfd_ioctl_get_process_apertures_new_args',
    'struct_kfd_ioctl_get_queue_wave_state_args',
    'struct_kfd_ioctl_get_tile_config_args',
    'struct_kfd_ioctl_get_version_args',
    'struct_kfd_ioctl_import_dmabuf_args',
    'struct_kfd_ioctl_ipc_export_handle_args',
    'struct_kfd_ioctl_ipc_import_handle_args',
    'struct_kfd_ioctl_map_memory_to_gpu_args',
    'struct_kfd_ioctl_reset_event_args',
    'struct_kfd_ioctl_runtime_enable_args',
    'struct_kfd_ioctl_set_cu_mask_args',
    'struct_kfd_ioctl_set_event_args',
    'struct_kfd_ioctl_set_memory_policy_args',
    'struct_kfd_ioctl_set_scratch_backing_va_args',
    'struct_kfd_ioctl_set_trap_handler_args',
    'struct_kfd_ioctl_set_xnack_mode_args',
    'struct_kfd_ioctl_smi_events_args', 'struct_kfd_ioctl_spm_args',
    'struct_kfd_ioctl_svm_args', 'struct_kfd_ioctl_svm_attribute',
    'struct_kfd_ioctl_unmap_memory_from_gpu_args',
    'struct_kfd_ioctl_update_queue_args',
    'struct_kfd_ioctl_wait_events_args',
    'struct_kfd_memory_exception_failure', 'struct_kfd_memory_range',
    'struct_kfd_process_device_apertures',
    'struct_kfd_queue_snapshot_entry', 'struct_kfd_runtime_info',
    'union_kfd_event_data_0', 'union_kfd_ioctl_dbg_trap_args_0']
