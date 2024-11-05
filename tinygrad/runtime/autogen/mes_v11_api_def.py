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



__MES_API_DEF_H__ = True # macro
uint32_t = True # macro
uint8_t = True # macro
uint16_t = True # macro
uint64_t = True # macro
bool = True # macro
MES_API_VERSION = 1 # macro

# values for enumeration 'c__Ea_API_FRAME_SIZE_IN_DWORDS'
c__Ea_API_FRAME_SIZE_IN_DWORDS__enumvalues = {
    64: 'API_FRAME_SIZE_IN_DWORDS',
}
API_FRAME_SIZE_IN_DWORDS = 64
c__Ea_API_FRAME_SIZE_IN_DWORDS = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_API_NUMBER_OF_COMMAND_MAX'
c__Ea_API_NUMBER_OF_COMMAND_MAX__enumvalues = {
    32: 'API_NUMBER_OF_COMMAND_MAX',
}
API_NUMBER_OF_COMMAND_MAX = 32
c__Ea_API_NUMBER_OF_COMMAND_MAX = ctypes.c_uint32 # enum

# values for enumeration 'MES_API_TYPE'
MES_API_TYPE__enumvalues = {
    1: 'MES_API_TYPE_SCHEDULER',
    2: 'MES_API_TYPE_MAX',
}
MES_API_TYPE_SCHEDULER = 1
MES_API_TYPE_MAX = 2
MES_API_TYPE = ctypes.c_uint32 # enum

# values for enumeration 'MES_SCH_API_OPCODE'
MES_SCH_API_OPCODE__enumvalues = {
    0: 'MES_SCH_API_SET_HW_RSRC',
    1: 'MES_SCH_API_SET_SCHEDULING_CONFIG',
    2: 'MES_SCH_API_ADD_QUEUE',
    3: 'MES_SCH_API_REMOVE_QUEUE',
    4: 'MES_SCH_API_PERFORM_YIELD',
    5: 'MES_SCH_API_SET_GANG_PRIORITY_LEVEL',
    6: 'MES_SCH_API_SUSPEND',
    7: 'MES_SCH_API_RESUME',
    8: 'MES_SCH_API_RESET',
    9: 'MES_SCH_API_SET_LOG_BUFFER',
    10: 'MES_SCH_API_CHANGE_GANG_PRORITY',
    11: 'MES_SCH_API_QUERY_SCHEDULER_STATUS',
    12: 'MES_SCH_API_PROGRAM_GDS',
    13: 'MES_SCH_API_SET_DEBUG_VMID',
    14: 'MES_SCH_API_MISC',
    15: 'MES_SCH_API_UPDATE_ROOT_PAGE_TABLE',
    16: 'MES_SCH_API_AMD_LOG',
    19: 'MES_SCH_API_SET_HW_RSRC_1',
    255: 'MES_SCH_API_MAX',
}
MES_SCH_API_SET_HW_RSRC = 0
MES_SCH_API_SET_SCHEDULING_CONFIG = 1
MES_SCH_API_ADD_QUEUE = 2
MES_SCH_API_REMOVE_QUEUE = 3
MES_SCH_API_PERFORM_YIELD = 4
MES_SCH_API_SET_GANG_PRIORITY_LEVEL = 5
MES_SCH_API_SUSPEND = 6
MES_SCH_API_RESUME = 7
MES_SCH_API_RESET = 8
MES_SCH_API_SET_LOG_BUFFER = 9
MES_SCH_API_CHANGE_GANG_PRORITY = 10
MES_SCH_API_QUERY_SCHEDULER_STATUS = 11
MES_SCH_API_PROGRAM_GDS = 12
MES_SCH_API_SET_DEBUG_VMID = 13
MES_SCH_API_MISC = 14
MES_SCH_API_UPDATE_ROOT_PAGE_TABLE = 15
MES_SCH_API_AMD_LOG = 16
MES_SCH_API_SET_HW_RSRC_1 = 19
MES_SCH_API_MAX = 255
MES_SCH_API_OPCODE = ctypes.c_uint32 # enum
class union_MES_API_HEADER(Union):
    pass

class struct_MES_API_HEADER_0(Structure):
    pass

struct_MES_API_HEADER_0._pack_ = 1 # source:False
struct_MES_API_HEADER_0._fields_ = [
    ('type', ctypes.c_uint32, 4),
    ('opcode', ctypes.c_uint32, 8),
    ('dwsize', ctypes.c_uint32, 8),
    ('reserved', ctypes.c_uint32, 12),
]

union_MES_API_HEADER._pack_ = 1 # source:False
union_MES_API_HEADER._anonymous_ = ('_0',)
union_MES_API_HEADER._fields_ = [
    ('_0', struct_MES_API_HEADER_0),
    ('u32All', ctypes.c_uint32),
]


# values for enumeration 'MES_AMD_PRIORITY_LEVEL'
MES_AMD_PRIORITY_LEVEL__enumvalues = {
    0: 'AMD_PRIORITY_LEVEL_LOW',
    1: 'AMD_PRIORITY_LEVEL_NORMAL',
    2: 'AMD_PRIORITY_LEVEL_MEDIUM',
    3: 'AMD_PRIORITY_LEVEL_HIGH',
    4: 'AMD_PRIORITY_LEVEL_REALTIME',
    5: 'AMD_PRIORITY_NUM_LEVELS',
}
AMD_PRIORITY_LEVEL_LOW = 0
AMD_PRIORITY_LEVEL_NORMAL = 1
AMD_PRIORITY_LEVEL_MEDIUM = 2
AMD_PRIORITY_LEVEL_HIGH = 3
AMD_PRIORITY_LEVEL_REALTIME = 4
AMD_PRIORITY_NUM_LEVELS = 5
MES_AMD_PRIORITY_LEVEL = ctypes.c_uint32 # enum

# values for enumeration 'MES_QUEUE_TYPE'
MES_QUEUE_TYPE__enumvalues = {
    0: 'MES_QUEUE_TYPE_GFX',
    1: 'MES_QUEUE_TYPE_COMPUTE',
    2: 'MES_QUEUE_TYPE_SDMA',
    3: 'MES_QUEUE_TYPE_MAX',
}
MES_QUEUE_TYPE_GFX = 0
MES_QUEUE_TYPE_COMPUTE = 1
MES_QUEUE_TYPE_SDMA = 2
MES_QUEUE_TYPE_MAX = 3
MES_QUEUE_TYPE = ctypes.c_uint32 # enum
class struct_MES_API_STATUS(Structure):
    pass

struct_MES_API_STATUS._pack_ = 1 # source:False
struct_MES_API_STATUS._fields_ = [
    ('api_completion_fence_addr', ctypes.c_uint64),
    ('api_completion_fence_value', ctypes.c_uint64),
]


# values for enumeration 'c__Ea_MAX_COMPUTE_PIPES'
c__Ea_MAX_COMPUTE_PIPES__enumvalues = {
    8: 'MAX_COMPUTE_PIPES',
}
MAX_COMPUTE_PIPES = 8
c__Ea_MAX_COMPUTE_PIPES = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_GFX_PIPES'
c__Ea_MAX_GFX_PIPES__enumvalues = {
    2: 'MAX_GFX_PIPES',
}
MAX_GFX_PIPES = 2
c__Ea_MAX_GFX_PIPES = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_SDMA_PIPES'
c__Ea_MAX_SDMA_PIPES__enumvalues = {
    2: 'MAX_SDMA_PIPES',
}
MAX_SDMA_PIPES = 2
c__Ea_MAX_SDMA_PIPES = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_COMPUTE_HQD_PER_PIPE'
c__Ea_MAX_COMPUTE_HQD_PER_PIPE__enumvalues = {
    8: 'MAX_COMPUTE_HQD_PER_PIPE',
}
MAX_COMPUTE_HQD_PER_PIPE = 8
c__Ea_MAX_COMPUTE_HQD_PER_PIPE = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_GFX_HQD_PER_PIPE'
c__Ea_MAX_GFX_HQD_PER_PIPE__enumvalues = {
    8: 'MAX_GFX_HQD_PER_PIPE',
}
MAX_GFX_HQD_PER_PIPE = 8
c__Ea_MAX_GFX_HQD_PER_PIPE = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_SDMA_HQD_PER_PIPE'
c__Ea_MAX_SDMA_HQD_PER_PIPE__enumvalues = {
    10: 'MAX_SDMA_HQD_PER_PIPE',
}
MAX_SDMA_HQD_PER_PIPE = 10
c__Ea_MAX_SDMA_HQD_PER_PIPE = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_SDMA_HQD_PER_PIPE_11_0'
c__Ea_MAX_SDMA_HQD_PER_PIPE_11_0__enumvalues = {
    8: 'MAX_SDMA_HQD_PER_PIPE_11_0',
}
MAX_SDMA_HQD_PER_PIPE_11_0 = 8
c__Ea_MAX_SDMA_HQD_PER_PIPE_11_0 = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_QUEUES_IN_A_GANG'
c__Ea_MAX_QUEUES_IN_A_GANG__enumvalues = {
    8: 'MAX_QUEUES_IN_A_GANG',
}
MAX_QUEUES_IN_A_GANG = 8
c__Ea_MAX_QUEUES_IN_A_GANG = ctypes.c_uint32 # enum

# values for enumeration 'VM_HUB_TYPE'
VM_HUB_TYPE__enumvalues = {
    0: 'VM_HUB_TYPE_GC',
    1: 'VM_HUB_TYPE_MM',
    2: 'VM_HUB_TYPE_MAX',
}
VM_HUB_TYPE_GC = 0
VM_HUB_TYPE_MM = 1
VM_HUB_TYPE_MAX = 2
VM_HUB_TYPE = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_VMID_INVALID'
c__Ea_VMID_INVALID__enumvalues = {
    65535: 'VMID_INVALID',
}
VMID_INVALID = 65535
c__Ea_VMID_INVALID = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_VMID_GCHUB'
c__Ea_MAX_VMID_GCHUB__enumvalues = {
    16: 'MAX_VMID_GCHUB',
}
MAX_VMID_GCHUB = 16
c__Ea_MAX_VMID_GCHUB = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MAX_VMID_MMHUB'
c__Ea_MAX_VMID_MMHUB__enumvalues = {
    16: 'MAX_VMID_MMHUB',
}
MAX_VMID_MMHUB = 16
c__Ea_MAX_VMID_MMHUB = ctypes.c_uint32 # enum

# values for enumeration 'SET_DEBUG_VMID_OPERATIONS'
SET_DEBUG_VMID_OPERATIONS__enumvalues = {
    0: 'DEBUG_VMID_OP_PROGRAM',
    1: 'DEBUG_VMID_OP_ALLOCATE',
    2: 'DEBUG_VMID_OP_RELEASE',
}
DEBUG_VMID_OP_PROGRAM = 0
DEBUG_VMID_OP_ALLOCATE = 1
DEBUG_VMID_OP_RELEASE = 2
SET_DEBUG_VMID_OPERATIONS = ctypes.c_uint32 # enum

# values for enumeration 'MES_LOG_OPERATION'
MES_LOG_OPERATION__enumvalues = {
    0: 'MES_LOG_OPERATION_CONTEXT_STATE_CHANGE',
    1: 'MES_LOG_OPERATION_QUEUE_NEW_WORK',
    2: 'MES_LOG_OPERATION_QUEUE_UNWAIT_SYNC_OBJECT',
    3: 'MES_LOG_OPERATION_QUEUE_NO_MORE_WORK',
    4: 'MES_LOG_OPERATION_QUEUE_WAIT_SYNC_OBJECT',
    15: 'MES_LOG_OPERATION_QUEUE_INVALID',
}
MES_LOG_OPERATION_CONTEXT_STATE_CHANGE = 0
MES_LOG_OPERATION_QUEUE_NEW_WORK = 1
MES_LOG_OPERATION_QUEUE_UNWAIT_SYNC_OBJECT = 2
MES_LOG_OPERATION_QUEUE_NO_MORE_WORK = 3
MES_LOG_OPERATION_QUEUE_WAIT_SYNC_OBJECT = 4
MES_LOG_OPERATION_QUEUE_INVALID = 15
MES_LOG_OPERATION = ctypes.c_uint32 # enum

# values for enumeration 'MES_LOG_CONTEXT_STATE'
MES_LOG_CONTEXT_STATE__enumvalues = {
    0: 'MES_LOG_CONTEXT_STATE_IDLE',
    1: 'MES_LOG_CONTEXT_STATE_RUNNING',
    2: 'MES_LOG_CONTEXT_STATE_READY',
    3: 'MES_LOG_CONTEXT_STATE_READY_STANDBY',
    15: 'MES_LOG_CONTEXT_STATE_INVALID',
}
MES_LOG_CONTEXT_STATE_IDLE = 0
MES_LOG_CONTEXT_STATE_RUNNING = 1
MES_LOG_CONTEXT_STATE_READY = 2
MES_LOG_CONTEXT_STATE_READY_STANDBY = 3
MES_LOG_CONTEXT_STATE_INVALID = 15
MES_LOG_CONTEXT_STATE = ctypes.c_uint32 # enum
class struct_MES_LOG_CONTEXT_STATE_CHANGE(Structure):
    pass

struct_MES_LOG_CONTEXT_STATE_CHANGE._pack_ = 1 # source:False
struct_MES_LOG_CONTEXT_STATE_CHANGE._fields_ = [
    ('h_context', ctypes.POINTER(None)),
    ('new_context_state', MES_LOG_CONTEXT_STATE),
]

class struct_MES_LOG_QUEUE_NEW_WORK(Structure):
    pass

struct_MES_LOG_QUEUE_NEW_WORK._pack_ = 1 # source:False
struct_MES_LOG_QUEUE_NEW_WORK._fields_ = [
    ('h_queue', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
]

class struct_MES_LOG_QUEUE_UNWAIT_SYNC_OBJECT(Structure):
    pass

struct_MES_LOG_QUEUE_UNWAIT_SYNC_OBJECT._pack_ = 1 # source:False
struct_MES_LOG_QUEUE_UNWAIT_SYNC_OBJECT._fields_ = [
    ('h_queue', ctypes.c_uint64),
    ('h_sync_object', ctypes.c_uint64),
]

class struct_MES_LOG_QUEUE_NO_MORE_WORK(Structure):
    pass

struct_MES_LOG_QUEUE_NO_MORE_WORK._pack_ = 1 # source:False
struct_MES_LOG_QUEUE_NO_MORE_WORK._fields_ = [
    ('h_queue', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
]

class struct_MES_LOG_QUEUE_WAIT_SYNC_OBJECT(Structure):
    pass

struct_MES_LOG_QUEUE_WAIT_SYNC_OBJECT._pack_ = 1 # source:False
struct_MES_LOG_QUEUE_WAIT_SYNC_OBJECT._fields_ = [
    ('h_queue', ctypes.c_uint64),
    ('h_sync_object', ctypes.c_uint64),
]

class struct_MES_LOG_ENTRY_HEADER(Structure):
    pass

struct_MES_LOG_ENTRY_HEADER._pack_ = 1 # source:False
struct_MES_LOG_ENTRY_HEADER._fields_ = [
    ('first_free_entry_index', ctypes.c_uint32),
    ('wraparound_count', ctypes.c_uint32),
    ('number_of_entries', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64 * 2),
]

class struct_MES_LOG_ENTRY_DATA(Structure):
    pass

class union_MES_LOG_ENTRY_DATA_0(Union):
    pass

union_MES_LOG_ENTRY_DATA_0._pack_ = 1 # source:False
union_MES_LOG_ENTRY_DATA_0._fields_ = [
    ('context_state_change', struct_MES_LOG_CONTEXT_STATE_CHANGE),
    ('queue_new_work', struct_MES_LOG_QUEUE_NEW_WORK),
    ('queue_unwait_sync_object', struct_MES_LOG_QUEUE_UNWAIT_SYNC_OBJECT),
    ('queue_no_more_work', struct_MES_LOG_QUEUE_NO_MORE_WORK),
    ('queue_wait_sync_object', struct_MES_LOG_QUEUE_WAIT_SYNC_OBJECT),
    ('all', ctypes.c_uint64 * 2),
]

struct_MES_LOG_ENTRY_DATA._pack_ = 1 # source:False
struct_MES_LOG_ENTRY_DATA._anonymous_ = ('_0',)
struct_MES_LOG_ENTRY_DATA._fields_ = [
    ('gpu_time_stamp', ctypes.c_uint64),
    ('operation_type', ctypes.c_uint32),
    ('reserved_operation_type_bits', ctypes.c_uint32),
    ('_0', union_MES_LOG_ENTRY_DATA_0),
]

class struct_MES_LOG_BUFFER(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_MES_LOG_ENTRY_HEADER),
    ('entries', struct_MES_LOG_ENTRY_DATA * 1),
     ]


# values for enumeration 'MES_SWIP_TO_HWIP_DEF'
MES_SWIP_TO_HWIP_DEF__enumvalues = {
    8: 'MES_MAX_HWIP_SEGMENT',
}
MES_MAX_HWIP_SEGMENT = 8
MES_SWIP_TO_HWIP_DEF = ctypes.c_uint32 # enum
class union_MESAPI_SET_HW_RESOURCES(Union):
    pass

class struct_MESAPI_SET_HW_RESOURCES_0(Structure):
    pass

class union_MESAPI_SET_HW_RESOURCES_0_0(Union):
    pass

class struct_MESAPI_SET_HW_RESOURCES_0_0_0(Structure):
    pass

struct_MESAPI_SET_HW_RESOURCES_0_0_0._pack_ = 1 # source:False
struct_MESAPI_SET_HW_RESOURCES_0_0_0._fields_ = [
    ('disable_reset', ctypes.c_uint32, 1),
    ('use_different_vmid_compute', ctypes.c_uint32, 1),
    ('disable_mes_log', ctypes.c_uint32, 1),
    ('apply_mmhub_pgvm_invalidate_ack_loss_wa', ctypes.c_uint32, 1),
    ('apply_grbm_remote_register_dummy_read_wa', ctypes.c_uint32, 1),
    ('second_gfx_pipe_enabled', ctypes.c_uint32, 1),
    ('enable_level_process_quantum_check', ctypes.c_uint32, 1),
    ('legacy_sch_mode', ctypes.c_uint32, 1),
    ('disable_add_queue_wptr_mc_addr', ctypes.c_uint32, 1),
    ('enable_mes_event_int_logging', ctypes.c_uint32, 1),
    ('enable_reg_active_poll', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 21),
]

union_MESAPI_SET_HW_RESOURCES_0_0._pack_ = 1 # source:False
union_MESAPI_SET_HW_RESOURCES_0_0._anonymous_ = ('_0',)
union_MESAPI_SET_HW_RESOURCES_0_0._fields_ = [
    ('_0', struct_MESAPI_SET_HW_RESOURCES_0_0_0),
    ('uint32_t_all', ctypes.c_uint32),
]

struct_MESAPI_SET_HW_RESOURCES_0._pack_ = 1 # source:False
struct_MESAPI_SET_HW_RESOURCES_0._anonymous_ = ('_0',)
struct_MESAPI_SET_HW_RESOURCES_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('vmid_mask_mmhub', ctypes.c_uint32),
    ('vmid_mask_gfxhub', ctypes.c_uint32),
    ('gds_size', ctypes.c_uint32),
    ('paging_vmid', ctypes.c_uint32),
    ('compute_hqd_mask', ctypes.c_uint32 * 8),
    ('gfx_hqd_mask', ctypes.c_uint32 * 2),
    ('sdma_hqd_mask', ctypes.c_uint32 * 2),
    ('aggregated_doorbells', ctypes.c_uint32 * 5),
    ('g_sch_ctx_gpu_mc_ptr', ctypes.c_uint64),
    ('query_status_fence_gpu_mc_ptr', ctypes.c_uint64),
    ('gc_base', ctypes.c_uint32 * 8),
    ('mmhub_base', ctypes.c_uint32 * 8),
    ('osssys_base', ctypes.c_uint32 * 8),
    ('api_status', struct_MES_API_STATUS),
    ('_0', union_MESAPI_SET_HW_RESOURCES_0_0),
    ('oversubscription_timer', ctypes.c_uint32),
    ('doorbell_info', ctypes.c_uint64),
    ('event_intr_history_gpu_mc_ptr', ctypes.c_uint64),
]

union_MESAPI_SET_HW_RESOURCES._pack_ = 1 # source:False
union_MESAPI_SET_HW_RESOURCES._anonymous_ = ('_0',)
union_MESAPI_SET_HW_RESOURCES._fields_ = [
    ('_0', struct_MESAPI_SET_HW_RESOURCES_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI_SET_HW_RESOURCES_1(Union):
    pass

class struct_MESAPI_SET_HW_RESOURCES_1_0(Structure):
    pass

class union_MESAPI_SET_HW_RESOURCES_1_0_0(Union):
    pass

class struct_MESAPI_SET_HW_RESOURCES_1_0_0_0(Structure):
    pass

struct_MESAPI_SET_HW_RESOURCES_1_0_0_0._pack_ = 1 # source:False
struct_MESAPI_SET_HW_RESOURCES_1_0_0_0._fields_ = [
    ('enable_mes_info_ctx', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 31),
]

union_MESAPI_SET_HW_RESOURCES_1_0_0._pack_ = 1 # source:False
union_MESAPI_SET_HW_RESOURCES_1_0_0._anonymous_ = ('_0',)
union_MESAPI_SET_HW_RESOURCES_1_0_0._fields_ = [
    ('_0', struct_MESAPI_SET_HW_RESOURCES_1_0_0_0),
    ('uint32_all', ctypes.c_uint32),
]

struct_MESAPI_SET_HW_RESOURCES_1_0._pack_ = 1 # source:False
struct_MESAPI_SET_HW_RESOURCES_1_0._anonymous_ = ('_0',)
struct_MESAPI_SET_HW_RESOURCES_1_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('api_status', struct_MES_API_STATUS),
    ('timestamp', ctypes.c_uint64),
    ('_0', union_MESAPI_SET_HW_RESOURCES_1_0_0),
    ('mes_info_ctx_mc_addr', ctypes.c_uint64),
    ('mes_info_ctx_size', ctypes.c_uint32),
    ('mes_kiq_unmap_timeout', ctypes.c_uint32),
]

union_MESAPI_SET_HW_RESOURCES_1._pack_ = 1 # source:False
union_MESAPI_SET_HW_RESOURCES_1._anonymous_ = ('_0',)
union_MESAPI_SET_HW_RESOURCES_1._fields_ = [
    ('_0', struct_MESAPI_SET_HW_RESOURCES_1_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__ADD_QUEUE(Union):
    pass

class struct_MESAPI__ADD_QUEUE_0(Structure):
    pass

class struct_MESAPI__ADD_QUEUE_0_0(Structure):
    pass

struct_MESAPI__ADD_QUEUE_0_0._pack_ = 1 # source:False
struct_MESAPI__ADD_QUEUE_0_0._fields_ = [
    ('paging', ctypes.c_uint32, 1),
    ('debug_vmid', ctypes.c_uint32, 4),
    ('program_gds', ctypes.c_uint32, 1),
    ('is_gang_suspended', ctypes.c_uint32, 1),
    ('is_tmz_queue', ctypes.c_uint32, 1),
    ('map_kiq_utility_queue', ctypes.c_uint32, 1),
    ('is_kfd_process', ctypes.c_uint32, 1),
    ('trap_en', ctypes.c_uint32, 1),
    ('is_aql_queue', ctypes.c_uint32, 1),
    ('skip_process_ctx_clear', ctypes.c_uint32, 1),
    ('map_legacy_kq', ctypes.c_uint32, 1),
    ('exclusively_scheduled', ctypes.c_uint32, 1),
    ('is_long_running', ctypes.c_uint32, 1),
    ('is_dwm_queue', ctypes.c_uint32, 1),
    ('is_video_blit_queue', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 14),
]

struct_MESAPI__ADD_QUEUE_0._pack_ = 1 # source:False
struct_MESAPI__ADD_QUEUE_0._anonymous_ = ('_0',)
struct_MESAPI__ADD_QUEUE_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('process_id', ctypes.c_uint32),
    ('page_table_base_addr', ctypes.c_uint64),
    ('process_va_start', ctypes.c_uint64),
    ('process_va_end', ctypes.c_uint64),
    ('process_quantum', ctypes.c_uint64),
    ('process_context_addr', ctypes.c_uint64),
    ('gang_quantum', ctypes.c_uint64),
    ('gang_context_addr', ctypes.c_uint64),
    ('inprocess_gang_priority', ctypes.c_uint32),
    ('gang_global_priority_level', MES_AMD_PRIORITY_LEVEL),
    ('doorbell_offset', ctypes.c_uint32),
    ('mqd_addr', ctypes.c_uint64),
    ('wptr_addr', ctypes.c_uint64),
    ('h_context', ctypes.c_uint64),
    ('h_queue', ctypes.c_uint64),
    ('queue_type', MES_QUEUE_TYPE),
    ('gds_base', ctypes.c_uint32),
    ('gds_size', ctypes.c_uint32),
    ('gws_base', ctypes.c_uint32),
    ('gws_size', ctypes.c_uint32),
    ('oa_mask', ctypes.c_uint32),
    ('trap_handler_addr', ctypes.c_uint64),
    ('vm_context_cntl', ctypes.c_uint32),
    ('_0', struct_MESAPI__ADD_QUEUE_0_0),
    ('api_status', struct_MES_API_STATUS),
    ('tma_addr', ctypes.c_uint64),
    ('sch_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint64),
    ('process_context_array_index', ctypes.c_uint32),
    ('gang_context_array_index', ctypes.c_uint32),
    ('pipe_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('alignment_mode_setting', ctypes.c_uint32),
    ('unmap_flag_addr', ctypes.c_uint64),
]

union_MESAPI__ADD_QUEUE._pack_ = 1 # source:False
union_MESAPI__ADD_QUEUE._anonymous_ = ('_0',)
union_MESAPI__ADD_QUEUE._fields_ = [
    ('_0', struct_MESAPI__ADD_QUEUE_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__REMOVE_QUEUE(Union):
    pass

class struct_MESAPI__REMOVE_QUEUE_0(Structure):
    pass

class struct_MESAPI__REMOVE_QUEUE_0_0(Structure):
    pass

struct_MESAPI__REMOVE_QUEUE_0_0._pack_ = 1 # source:False
struct_MESAPI__REMOVE_QUEUE_0_0._fields_ = [
    ('unmap_legacy_gfx_queue', ctypes.c_uint32, 1),
    ('unmap_kiq_utility_queue', ctypes.c_uint32, 1),
    ('preempt_legacy_gfx_queue', ctypes.c_uint32, 1),
    ('unmap_legacy_queue', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 28),
]

struct_MESAPI__REMOVE_QUEUE_0._pack_ = 1 # source:False
struct_MESAPI__REMOVE_QUEUE_0._anonymous_ = ('_0',)
struct_MESAPI__REMOVE_QUEUE_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('doorbell_offset', ctypes.c_uint32),
    ('gang_context_addr', ctypes.c_uint64),
    ('_0', struct_MESAPI__REMOVE_QUEUE_0_0),
    ('api_status', struct_MES_API_STATUS),
    ('pipe_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('tf_addr', ctypes.c_uint64),
    ('tf_data', ctypes.c_uint32),
    ('queue_type', MES_QUEUE_TYPE),
]

union_MESAPI__REMOVE_QUEUE._pack_ = 1 # source:False
union_MESAPI__REMOVE_QUEUE._anonymous_ = ('_0',)
union_MESAPI__REMOVE_QUEUE._fields_ = [
    ('_0', struct_MESAPI__REMOVE_QUEUE_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__SET_SCHEDULING_CONFIG(Union):
    pass

class struct_MESAPI__SET_SCHEDULING_CONFIG_0(Structure):
    pass

struct_MESAPI__SET_SCHEDULING_CONFIG_0._pack_ = 1 # source:False
struct_MESAPI__SET_SCHEDULING_CONFIG_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('grace_period_other_levels', ctypes.c_uint64 * 5),
    ('process_quantum_for_level', ctypes.c_uint64 * 5),
    ('process_grace_period_same_level', ctypes.c_uint64 * 5),
    ('normal_yield_percent', ctypes.c_uint32),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__SET_SCHEDULING_CONFIG._pack_ = 1 # source:False
union_MESAPI__SET_SCHEDULING_CONFIG._anonymous_ = ('_0',)
union_MESAPI__SET_SCHEDULING_CONFIG._fields_ = [
    ('_0', struct_MESAPI__SET_SCHEDULING_CONFIG_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__PERFORM_YIELD(Union):
    pass

class struct_MESAPI__PERFORM_YIELD_0(Structure):
    pass

struct_MESAPI__PERFORM_YIELD_0._pack_ = 1 # source:False
struct_MESAPI__PERFORM_YIELD_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('dummy', ctypes.c_uint32),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__PERFORM_YIELD._pack_ = 1 # source:False
union_MESAPI__PERFORM_YIELD._anonymous_ = ('_0',)
union_MESAPI__PERFORM_YIELD._fields_ = [
    ('_0', struct_MESAPI__PERFORM_YIELD_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__CHANGE_GANG_PRIORITY_LEVEL(Union):
    pass

class struct_MESAPI__CHANGE_GANG_PRIORITY_LEVEL_0(Structure):
    pass

struct_MESAPI__CHANGE_GANG_PRIORITY_LEVEL_0._pack_ = 1 # source:False
struct_MESAPI__CHANGE_GANG_PRIORITY_LEVEL_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('inprocess_gang_priority', ctypes.c_uint32),
    ('gang_global_priority_level', MES_AMD_PRIORITY_LEVEL),
    ('gang_quantum', ctypes.c_uint64),
    ('gang_context_addr', ctypes.c_uint64),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__CHANGE_GANG_PRIORITY_LEVEL._pack_ = 1 # source:False
union_MESAPI__CHANGE_GANG_PRIORITY_LEVEL._anonymous_ = ('_0',)
union_MESAPI__CHANGE_GANG_PRIORITY_LEVEL._fields_ = [
    ('_0', struct_MESAPI__CHANGE_GANG_PRIORITY_LEVEL_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__SUSPEND(Union):
    pass

class struct_MESAPI__SUSPEND_0(Structure):
    pass

class struct_MESAPI__SUSPEND_0_0(Structure):
    pass

struct_MESAPI__SUSPEND_0_0._pack_ = 1 # source:False
struct_MESAPI__SUSPEND_0_0._fields_ = [
    ('suspend_all_gangs', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 31),
]

struct_MESAPI__SUSPEND_0._pack_ = 1 # source:False
struct_MESAPI__SUSPEND_0._anonymous_ = ('_0',)
struct_MESAPI__SUSPEND_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('_0', struct_MESAPI__SUSPEND_0_0),
    ('gang_context_addr', ctypes.c_uint64),
    ('suspend_fence_addr', ctypes.c_uint64),
    ('suspend_fence_value', ctypes.c_uint32),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__SUSPEND._pack_ = 1 # source:False
union_MESAPI__SUSPEND._anonymous_ = ('_0',)
union_MESAPI__SUSPEND._fields_ = [
    ('_0', struct_MESAPI__SUSPEND_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__RESUME(Union):
    pass

class struct_MESAPI__RESUME_0(Structure):
    pass

class struct_MESAPI__RESUME_0_0(Structure):
    pass

struct_MESAPI__RESUME_0_0._pack_ = 1 # source:False
struct_MESAPI__RESUME_0_0._fields_ = [
    ('resume_all_gangs', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 31),
]

struct_MESAPI__RESUME_0._pack_ = 1 # source:False
struct_MESAPI__RESUME_0._anonymous_ = ('_0',)
struct_MESAPI__RESUME_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('_0', struct_MESAPI__RESUME_0_0),
    ('gang_context_addr', ctypes.c_uint64),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__RESUME._pack_ = 1 # source:False
union_MESAPI__RESUME._anonymous_ = ('_0',)
union_MESAPI__RESUME._fields_ = [
    ('_0', struct_MESAPI__RESUME_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__RESET(Union):
    pass

class struct_MESAPI__RESET_0(Structure):
    pass

class struct_MESAPI__RESET_0_0(Structure):
    pass

struct_MESAPI__RESET_0_0._pack_ = 1 # source:False
struct_MESAPI__RESET_0_0._fields_ = [
    ('reset_queue_only', ctypes.c_uint32, 1),
    ('hang_detect_then_reset', ctypes.c_uint32, 1),
    ('hang_detect_only', ctypes.c_uint32, 1),
    ('reset_legacy_gfx', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 28),
]

struct_MESAPI__RESET_0._pack_ = 1 # source:False
struct_MESAPI__RESET_0._anonymous_ = ('_0',)
struct_MESAPI__RESET_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('_0', struct_MESAPI__RESET_0_0),
    ('gang_context_addr', ctypes.c_uint64),
    ('doorbell_offset', ctypes.c_uint32),
    ('doorbell_offset_addr', ctypes.c_uint64),
    ('queue_type', MES_QUEUE_TYPE),
    ('pipe_id_lp', ctypes.c_uint32),
    ('queue_id_lp', ctypes.c_uint32),
    ('vmid_id_lp', ctypes.c_uint32),
    ('mqd_mc_addr_lp', ctypes.c_uint64),
    ('doorbell_offset_lp', ctypes.c_uint32),
    ('wptr_addr_lp', ctypes.c_uint64),
    ('pipe_id_hp', ctypes.c_uint32),
    ('queue_id_hp', ctypes.c_uint32),
    ('vmid_id_hp', ctypes.c_uint32),
    ('mqd_mc_addr_hp', ctypes.c_uint64),
    ('doorbell_offset_hp', ctypes.c_uint32),
    ('wptr_addr_hp', ctypes.c_uint64),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__RESET._pack_ = 1 # source:False
union_MESAPI__RESET._anonymous_ = ('_0',)
union_MESAPI__RESET._fields_ = [
    ('_0', struct_MESAPI__RESET_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__SET_LOGGING_BUFFER(Union):
    pass

class struct_MESAPI__SET_LOGGING_BUFFER_0(Structure):
    pass

struct_MESAPI__SET_LOGGING_BUFFER_0._pack_ = 1 # source:False
struct_MESAPI__SET_LOGGING_BUFFER_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('log_type', MES_QUEUE_TYPE),
    ('logging_buffer_addr', ctypes.c_uint64),
    ('number_of_entries', ctypes.c_uint32),
    ('interrupt_entry', ctypes.c_uint32),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__SET_LOGGING_BUFFER._pack_ = 1 # source:False
union_MESAPI__SET_LOGGING_BUFFER._anonymous_ = ('_0',)
union_MESAPI__SET_LOGGING_BUFFER._fields_ = [
    ('_0', struct_MESAPI__SET_LOGGING_BUFFER_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__QUERY_MES_STATUS(Union):
    pass

class struct_MESAPI__QUERY_MES_STATUS_0(Structure):
    pass

struct_MESAPI__QUERY_MES_STATUS_0._pack_ = 1 # source:False
struct_MESAPI__QUERY_MES_STATUS_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('mes_healthy', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__QUERY_MES_STATUS._pack_ = 1 # source:False
union_MESAPI__QUERY_MES_STATUS._anonymous_ = ('_0',)
union_MESAPI__QUERY_MES_STATUS._fields_ = [
    ('_0', struct_MESAPI__QUERY_MES_STATUS_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__PROGRAM_GDS(Union):
    pass

class struct_MESAPI__PROGRAM_GDS_0(Structure):
    pass

struct_MESAPI__PROGRAM_GDS_0._pack_ = 1 # source:False
struct_MESAPI__PROGRAM_GDS_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('process_context_addr', ctypes.c_uint64),
    ('gds_base', ctypes.c_uint32),
    ('gds_size', ctypes.c_uint32),
    ('gws_base', ctypes.c_uint32),
    ('gws_size', ctypes.c_uint32),
    ('oa_mask', ctypes.c_uint32),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__PROGRAM_GDS._pack_ = 1 # source:False
union_MESAPI__PROGRAM_GDS._anonymous_ = ('_0',)
union_MESAPI__PROGRAM_GDS._fields_ = [
    ('_0', struct_MESAPI__PROGRAM_GDS_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__SET_DEBUG_VMID(Union):
    pass

class struct_MESAPI__SET_DEBUG_VMID_0(Structure):
    pass

class union_MESAPI__SET_DEBUG_VMID_0_0(Union):
    pass

class struct_MESAPI__SET_DEBUG_VMID_0_0_flags(Structure):
    pass

struct_MESAPI__SET_DEBUG_VMID_0_0_flags._pack_ = 1 # source:False
struct_MESAPI__SET_DEBUG_VMID_0_0_flags._fields_ = [
    ('use_gds', ctypes.c_uint32, 1),
    ('operation', ctypes.c_uint32, 2),
    ('reserved', ctypes.c_uint32, 29),
]

union_MESAPI__SET_DEBUG_VMID_0_0._pack_ = 1 # source:False
union_MESAPI__SET_DEBUG_VMID_0_0._fields_ = [
    ('flags', struct_MESAPI__SET_DEBUG_VMID_0_0_flags),
    ('u32All', ctypes.c_uint32),
]

struct_MESAPI__SET_DEBUG_VMID_0._pack_ = 1 # source:False
struct_MESAPI__SET_DEBUG_VMID_0._anonymous_ = ('_0',)
struct_MESAPI__SET_DEBUG_VMID_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('api_status', struct_MES_API_STATUS),
    ('_0', union_MESAPI__SET_DEBUG_VMID_0_0),
    ('reserved', ctypes.c_uint32),
    ('debug_vmid', ctypes.c_uint32),
    ('process_context_addr', ctypes.c_uint64),
    ('page_table_base_addr', ctypes.c_uint64),
    ('process_va_start', ctypes.c_uint64),
    ('process_va_end', ctypes.c_uint64),
    ('gds_base', ctypes.c_uint32),
    ('gds_size', ctypes.c_uint32),
    ('gws_base', ctypes.c_uint32),
    ('gws_size', ctypes.c_uint32),
    ('oa_mask', ctypes.c_uint32),
    ('output_addr', ctypes.c_uint64),
]

union_MESAPI__SET_DEBUG_VMID._pack_ = 1 # source:False
union_MESAPI__SET_DEBUG_VMID._anonymous_ = ('_0',)
union_MESAPI__SET_DEBUG_VMID._fields_ = [
    ('_0', struct_MESAPI__SET_DEBUG_VMID_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]


# values for enumeration 'MESAPI_MISC_OPCODE'
MESAPI_MISC_OPCODE__enumvalues = {
    0: 'MESAPI_MISC__WRITE_REG',
    1: 'MESAPI_MISC__INV_GART',
    2: 'MESAPI_MISC__QUERY_STATUS',
    3: 'MESAPI_MISC__READ_REG',
    4: 'MESAPI_MISC__WAIT_REG_MEM',
    5: 'MESAPI_MISC__SET_SHADER_DEBUGGER',
    6: 'MESAPI_MISC__MAX',
}
MESAPI_MISC__WRITE_REG = 0
MESAPI_MISC__INV_GART = 1
MESAPI_MISC__QUERY_STATUS = 2
MESAPI_MISC__READ_REG = 3
MESAPI_MISC__WAIT_REG_MEM = 4
MESAPI_MISC__SET_SHADER_DEBUGGER = 5
MESAPI_MISC__MAX = 6
MESAPI_MISC_OPCODE = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_MISC_DATA_MAX_SIZE_IN_DWORDS'
c__Ea_MISC_DATA_MAX_SIZE_IN_DWORDS__enumvalues = {
    20: 'MISC_DATA_MAX_SIZE_IN_DWORDS',
}
MISC_DATA_MAX_SIZE_IN_DWORDS = 20
c__Ea_MISC_DATA_MAX_SIZE_IN_DWORDS = ctypes.c_uint32 # enum
class struct_WRITE_REG(Structure):
    pass

struct_WRITE_REG._pack_ = 1 # source:False
struct_WRITE_REG._fields_ = [
    ('reg_offset', ctypes.c_uint32),
    ('reg_value', ctypes.c_uint32),
]

class struct_READ_REG(Structure):
    pass

struct_READ_REG._pack_ = 1 # source:False
struct_READ_REG._fields_ = [
    ('reg_offset', ctypes.c_uint32),
    ('buffer_addr', ctypes.c_uint64),
]


# values for enumeration 'WRM_OPERATION'
WRM_OPERATION__enumvalues = {
    0: 'WRM_OPERATION__WAIT_REG_MEM',
    1: 'WRM_OPERATION__WR_WAIT_WR_REG',
    2: 'WRM_OPERATION__MAX',
}
WRM_OPERATION__WAIT_REG_MEM = 0
WRM_OPERATION__WR_WAIT_WR_REG = 1
WRM_OPERATION__MAX = 2
WRM_OPERATION = ctypes.c_uint32 # enum
class struct_WAIT_REG_MEM(Structure):
    pass

struct_WAIT_REG_MEM._pack_ = 1 # source:False
struct_WAIT_REG_MEM._fields_ = [
    ('op', WRM_OPERATION),
    ('reference', ctypes.c_uint32),
    ('mask', ctypes.c_uint32),
    ('reg_offset1', ctypes.c_uint32),
    ('reg_offset2', ctypes.c_uint32),
]

class struct_INV_GART(Structure):
    pass

struct_INV_GART._pack_ = 1 # source:False
struct_INV_GART._fields_ = [
    ('inv_range_va_start', ctypes.c_uint64),
    ('inv_range_size', ctypes.c_uint64),
]

class struct_QUERY_STATUS(Structure):
    pass

struct_QUERY_STATUS._pack_ = 1 # source:False
struct_QUERY_STATUS._fields_ = [
    ('context_id', ctypes.c_uint32),
]

class struct_SET_SHADER_DEBUGGER(Structure):
    pass

class union_SET_SHADER_DEBUGGER_flags(Union):
    pass

class struct_SET_SHADER_DEBUGGER_0_0(Structure):
    pass

struct_SET_SHADER_DEBUGGER_0_0._pack_ = 1 # source:False
struct_SET_SHADER_DEBUGGER_0_0._fields_ = [
    ('single_memop', ctypes.c_uint32, 1),
    ('single_alu_op', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 29),
    ('process_ctx_flush', ctypes.c_uint32, 1),
]

union_SET_SHADER_DEBUGGER_flags._pack_ = 1 # source:False
union_SET_SHADER_DEBUGGER_flags._anonymous_ = ('_0',)
union_SET_SHADER_DEBUGGER_flags._fields_ = [
    ('_0', struct_SET_SHADER_DEBUGGER_0_0),
    ('u32all', ctypes.c_uint32),
]

struct_SET_SHADER_DEBUGGER._pack_ = 1 # source:False
struct_SET_SHADER_DEBUGGER._fields_ = [
    ('process_context_addr', ctypes.c_uint64),
    ('flags', union_SET_SHADER_DEBUGGER_flags),
    ('spi_gdbg_per_vmid_cntl', ctypes.c_uint32),
    ('tcp_watch_cntl', ctypes.c_uint32 * 4),
    ('trap_en', ctypes.c_uint32),
]

class union_MESAPI__MISC(Union):
    pass

class struct_MESAPI__MISC_0(Structure):
    pass

class union_MESAPI__MISC_0_0(Union):
    pass

union_MESAPI__MISC_0_0._pack_ = 1 # source:False
union_MESAPI__MISC_0_0._fields_ = [
    ('write_reg', struct_WRITE_REG),
    ('inv_gart', struct_INV_GART),
    ('query_status', struct_QUERY_STATUS),
    ('read_reg', struct_READ_REG),
    ('wait_reg_mem', struct_WAIT_REG_MEM),
    ('set_shader_debugger', struct_SET_SHADER_DEBUGGER),
    ('queue_sch_level', MES_AMD_PRIORITY_LEVEL),
    ('data', ctypes.c_uint32 * 20),
]

struct_MESAPI__MISC_0._pack_ = 1 # source:False
struct_MESAPI__MISC_0._anonymous_ = ('_0',)
struct_MESAPI__MISC_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('opcode', MESAPI_MISC_OPCODE),
    ('api_status', struct_MES_API_STATUS),
    ('_0', union_MESAPI__MISC_0_0),
]

union_MESAPI__MISC._pack_ = 1 # source:False
union_MESAPI__MISC._anonymous_ = ('_0',)
union_MESAPI__MISC._fields_ = [
    ('_0', struct_MESAPI__MISC_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI__UPDATE_ROOT_PAGE_TABLE(Union):
    pass

class struct_MESAPI__UPDATE_ROOT_PAGE_TABLE_0(Structure):
    pass

struct_MESAPI__UPDATE_ROOT_PAGE_TABLE_0._pack_ = 1 # source:False
struct_MESAPI__UPDATE_ROOT_PAGE_TABLE_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('page_table_base_addr', ctypes.c_uint64),
    ('process_context_addr', ctypes.c_uint64),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI__UPDATE_ROOT_PAGE_TABLE._pack_ = 1 # source:False
union_MESAPI__UPDATE_ROOT_PAGE_TABLE._anonymous_ = ('_0',)
union_MESAPI__UPDATE_ROOT_PAGE_TABLE._fields_ = [
    ('_0', struct_MESAPI__UPDATE_ROOT_PAGE_TABLE_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

class union_MESAPI_AMD_LOG(Union):
    pass

class struct_MESAPI_AMD_LOG_0(Structure):
    pass

struct_MESAPI_AMD_LOG_0._pack_ = 1 # source:False
struct_MESAPI_AMD_LOG_0._fields_ = [
    ('header', union_MES_API_HEADER),
    ('p_buffer_memory', ctypes.c_uint64),
    ('p_buffer_size_used', ctypes.c_uint64),
    ('api_status', struct_MES_API_STATUS),
]

union_MESAPI_AMD_LOG._pack_ = 1 # source:False
union_MESAPI_AMD_LOG._anonymous_ = ('_0',)
union_MESAPI_AMD_LOG._fields_ = [
    ('_0', struct_MESAPI_AMD_LOG_0),
    ('max_dwords_in_api', ctypes.c_uint32 * 64),
]

__all__ = \
    ['AMD_PRIORITY_LEVEL_HIGH', 'AMD_PRIORITY_LEVEL_LOW',
    'AMD_PRIORITY_LEVEL_MEDIUM', 'AMD_PRIORITY_LEVEL_NORMAL',
    'AMD_PRIORITY_LEVEL_REALTIME', 'AMD_PRIORITY_NUM_LEVELS',
    'API_FRAME_SIZE_IN_DWORDS', 'API_NUMBER_OF_COMMAND_MAX',
    'DEBUG_VMID_OP_ALLOCATE', 'DEBUG_VMID_OP_PROGRAM',
    'DEBUG_VMID_OP_RELEASE', 'MAX_COMPUTE_HQD_PER_PIPE',
    'MAX_COMPUTE_PIPES', 'MAX_GFX_HQD_PER_PIPE', 'MAX_GFX_PIPES',
    'MAX_QUEUES_IN_A_GANG', 'MAX_SDMA_HQD_PER_PIPE',
    'MAX_SDMA_HQD_PER_PIPE_11_0', 'MAX_SDMA_PIPES', 'MAX_VMID_GCHUB',
    'MAX_VMID_MMHUB', 'MESAPI_MISC_OPCODE', 'MESAPI_MISC__INV_GART',
    'MESAPI_MISC__MAX', 'MESAPI_MISC__QUERY_STATUS',
    'MESAPI_MISC__READ_REG', 'MESAPI_MISC__SET_SHADER_DEBUGGER',
    'MESAPI_MISC__WAIT_REG_MEM', 'MESAPI_MISC__WRITE_REG',
    'MES_AMD_PRIORITY_LEVEL', 'MES_API_TYPE', 'MES_API_TYPE_MAX',
    'MES_API_TYPE_SCHEDULER', 'MES_API_VERSION',
    'MES_LOG_CONTEXT_STATE', 'MES_LOG_CONTEXT_STATE_IDLE',
    'MES_LOG_CONTEXT_STATE_INVALID', 'MES_LOG_CONTEXT_STATE_READY',
    'MES_LOG_CONTEXT_STATE_READY_STANDBY',
    'MES_LOG_CONTEXT_STATE_RUNNING', 'MES_LOG_OPERATION',
    'MES_LOG_OPERATION_CONTEXT_STATE_CHANGE',
    'MES_LOG_OPERATION_QUEUE_INVALID',
    'MES_LOG_OPERATION_QUEUE_NEW_WORK',
    'MES_LOG_OPERATION_QUEUE_NO_MORE_WORK',
    'MES_LOG_OPERATION_QUEUE_UNWAIT_SYNC_OBJECT',
    'MES_LOG_OPERATION_QUEUE_WAIT_SYNC_OBJECT',
    'MES_MAX_HWIP_SEGMENT', 'MES_QUEUE_TYPE',
    'MES_QUEUE_TYPE_COMPUTE', 'MES_QUEUE_TYPE_GFX',
    'MES_QUEUE_TYPE_MAX', 'MES_QUEUE_TYPE_SDMA',
    'MES_SCH_API_ADD_QUEUE', 'MES_SCH_API_AMD_LOG',
    'MES_SCH_API_CHANGE_GANG_PRORITY', 'MES_SCH_API_MAX',
    'MES_SCH_API_MISC', 'MES_SCH_API_OPCODE',
    'MES_SCH_API_PERFORM_YIELD', 'MES_SCH_API_PROGRAM_GDS',
    'MES_SCH_API_QUERY_SCHEDULER_STATUS', 'MES_SCH_API_REMOVE_QUEUE',
    'MES_SCH_API_RESET', 'MES_SCH_API_RESUME',
    'MES_SCH_API_SET_DEBUG_VMID',
    'MES_SCH_API_SET_GANG_PRIORITY_LEVEL', 'MES_SCH_API_SET_HW_RSRC',
    'MES_SCH_API_SET_HW_RSRC_1', 'MES_SCH_API_SET_LOG_BUFFER',
    'MES_SCH_API_SET_SCHEDULING_CONFIG', 'MES_SCH_API_SUSPEND',
    'MES_SCH_API_UPDATE_ROOT_PAGE_TABLE', 'MES_SWIP_TO_HWIP_DEF',
    'MISC_DATA_MAX_SIZE_IN_DWORDS', 'SET_DEBUG_VMID_OPERATIONS',
    'VMID_INVALID', 'VM_HUB_TYPE', 'VM_HUB_TYPE_GC',
    'VM_HUB_TYPE_MAX', 'VM_HUB_TYPE_MM', 'WRM_OPERATION',
    'WRM_OPERATION__MAX', 'WRM_OPERATION__WAIT_REG_MEM',
    'WRM_OPERATION__WR_WAIT_WR_REG', '__MES_API_DEF_H__', 'bool',
    'c__Ea_API_FRAME_SIZE_IN_DWORDS',
    'c__Ea_API_NUMBER_OF_COMMAND_MAX',
    'c__Ea_MAX_COMPUTE_HQD_PER_PIPE', 'c__Ea_MAX_COMPUTE_PIPES',
    'c__Ea_MAX_GFX_HQD_PER_PIPE', 'c__Ea_MAX_GFX_PIPES',
    'c__Ea_MAX_QUEUES_IN_A_GANG', 'c__Ea_MAX_SDMA_HQD_PER_PIPE',
    'c__Ea_MAX_SDMA_HQD_PER_PIPE_11_0', 'c__Ea_MAX_SDMA_PIPES',
    'c__Ea_MAX_VMID_GCHUB', 'c__Ea_MAX_VMID_MMHUB',
    'c__Ea_MISC_DATA_MAX_SIZE_IN_DWORDS', 'c__Ea_VMID_INVALID',
    'struct_INV_GART', 'struct_MESAPI_AMD_LOG_0',
    'struct_MESAPI_SET_HW_RESOURCES_0',
    'struct_MESAPI_SET_HW_RESOURCES_0_0_0',
    'struct_MESAPI_SET_HW_RESOURCES_1_0',
    'struct_MESAPI_SET_HW_RESOURCES_1_0_0_0',
    'struct_MESAPI__ADD_QUEUE_0', 'struct_MESAPI__ADD_QUEUE_0_0',
    'struct_MESAPI__CHANGE_GANG_PRIORITY_LEVEL_0',
    'struct_MESAPI__MISC_0', 'struct_MESAPI__PERFORM_YIELD_0',
    'struct_MESAPI__PROGRAM_GDS_0',
    'struct_MESAPI__QUERY_MES_STATUS_0',
    'struct_MESAPI__REMOVE_QUEUE_0',
    'struct_MESAPI__REMOVE_QUEUE_0_0', 'struct_MESAPI__RESET_0',
    'struct_MESAPI__RESET_0_0', 'struct_MESAPI__RESUME_0',
    'struct_MESAPI__RESUME_0_0', 'struct_MESAPI__SET_DEBUG_VMID_0',
    'struct_MESAPI__SET_DEBUG_VMID_0_0_flags',
    'struct_MESAPI__SET_LOGGING_BUFFER_0',
    'struct_MESAPI__SET_SCHEDULING_CONFIG_0',
    'struct_MESAPI__SUSPEND_0', 'struct_MESAPI__SUSPEND_0_0',
    'struct_MESAPI__UPDATE_ROOT_PAGE_TABLE_0',
    'struct_MES_API_HEADER_0', 'struct_MES_API_STATUS',
    'struct_MES_LOG_BUFFER', 'struct_MES_LOG_CONTEXT_STATE_CHANGE',
    'struct_MES_LOG_ENTRY_DATA', 'struct_MES_LOG_ENTRY_HEADER',
    'struct_MES_LOG_QUEUE_NEW_WORK',
    'struct_MES_LOG_QUEUE_NO_MORE_WORK',
    'struct_MES_LOG_QUEUE_UNWAIT_SYNC_OBJECT',
    'struct_MES_LOG_QUEUE_WAIT_SYNC_OBJECT', 'struct_QUERY_STATUS',
    'struct_READ_REG', 'struct_SET_SHADER_DEBUGGER',
    'struct_SET_SHADER_DEBUGGER_0_0', 'struct_WAIT_REG_MEM',
    'struct_WRITE_REG', 'uint16_t', 'uint32_t', 'uint64_t', 'uint8_t',
    'union_MESAPI_AMD_LOG', 'union_MESAPI_SET_HW_RESOURCES',
    'union_MESAPI_SET_HW_RESOURCES_0_0',
    'union_MESAPI_SET_HW_RESOURCES_1',
    'union_MESAPI_SET_HW_RESOURCES_1_0_0', 'union_MESAPI__ADD_QUEUE',
    'union_MESAPI__CHANGE_GANG_PRIORITY_LEVEL', 'union_MESAPI__MISC',
    'union_MESAPI__MISC_0_0', 'union_MESAPI__PERFORM_YIELD',
    'union_MESAPI__PROGRAM_GDS', 'union_MESAPI__QUERY_MES_STATUS',
    'union_MESAPI__REMOVE_QUEUE', 'union_MESAPI__RESET',
    'union_MESAPI__RESUME', 'union_MESAPI__SET_DEBUG_VMID',
    'union_MESAPI__SET_DEBUG_VMID_0_0',
    'union_MESAPI__SET_LOGGING_BUFFER',
    'union_MESAPI__SET_SCHEDULING_CONFIG', 'union_MESAPI__SUSPEND',
    'union_MESAPI__UPDATE_ROOT_PAGE_TABLE', 'union_MES_API_HEADER',
    'union_MES_LOG_ENTRY_DATA_0', 'union_SET_SHADER_DEBUGGER_flags']
