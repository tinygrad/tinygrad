# mypy: ignore-errors
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




import fcntl, functools

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd, **kwargs):
  ret = fcntl.ioctl(__fd, (__idir<<30) | (ctypes.sizeof(made := __user_struct(**kwargs))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)

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
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')


_UAPI_LINUX_ION_H = True # macro
# ION_NUM_HEAP_IDS = (ctypes.sizeof*8) # macro
ION_FLAG_CACHED = 1 # macro
ION_FLAG_CACHED_NEEDS_SYNC = 2 # macro
ION_IOC_MAGIC = 'I' # macro
ion_user_handle_t = ctypes.c_int32

# values for enumeration 'ion_heap_type'
ion_heap_type__enumvalues = {
    0: 'ION_HEAP_TYPE_SYSTEM',
    1: 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    2: 'ION_HEAP_TYPE_CARVEOUT',
    3: 'ION_HEAP_TYPE_CHUNK',
    4: 'ION_HEAP_TYPE_DMA',
    5: 'ION_HEAP_TYPE_CUSTOM',
    16: 'ION_NUM_HEAPS',
}
ION_HEAP_TYPE_SYSTEM = 0
ION_HEAP_TYPE_SYSTEM_CONTIG = 1
ION_HEAP_TYPE_CARVEOUT = 2
ION_HEAP_TYPE_CHUNK = 3
ION_HEAP_TYPE_DMA = 4
ION_HEAP_TYPE_CUSTOM = 5
ION_NUM_HEAPS = 16
ion_heap_type = ctypes.c_uint32 # enum
ION_HEAP_SYSTEM_MASK = ((1<<ION_HEAP_TYPE_SYSTEM)) # macro
ION_HEAP_SYSTEM_CONTIG_MASK = ((1<<ION_HEAP_TYPE_SYSTEM_CONTIG)) # macro
ION_HEAP_CARVEOUT_MASK = ((1<<ION_HEAP_TYPE_CARVEOUT)) # macro
ION_HEAP_TYPE_DMA_MASK = ((1<<ION_HEAP_TYPE_DMA)) # macro
class struct_ion_allocation_data(Structure):
    pass

struct_ion_allocation_data._pack_ = 1 # source:False
struct_ion_allocation_data._fields_ = [
    ('len', ctypes.c_uint64),
    ('align', ctypes.c_uint64),
    ('heap_id_mask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('handle', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

ION_IOC_ALLOC = _IOWR ( 'I' , 0 , struct_ion_allocation_data ) # macro (from list)
class struct_ion_fd_data(Structure):
    pass

struct_ion_fd_data._pack_ = 1 # source:False
struct_ion_fd_data._fields_ = [
    ('handle', ctypes.c_int32),
    ('fd', ctypes.c_int32),
]

ION_IOC_MAP = _IOWR ( 'I' , 2 , struct_ion_fd_data ) # macro (from list)
ION_IOC_SHARE = _IOWR ( 'I' , 4 , struct_ion_fd_data ) # macro (from list)
ION_IOC_IMPORT = _IOWR ( 'I' , 5 , struct_ion_fd_data ) # macro (from list)
ION_IOC_SYNC = _IOWR ( 'I' , 7 , struct_ion_fd_data ) # macro (from list)
class struct_ion_handle_data(Structure):
    pass

struct_ion_handle_data._pack_ = 1 # source:False
struct_ion_handle_data._fields_ = [
    ('handle', ctypes.c_int32),
]

ION_IOC_FREE = _IOWR ( 'I' , 1 , struct_ion_handle_data ) # macro (from list)
class struct_ion_custom_data(Structure):
    pass

struct_ion_custom_data._pack_ = 1 # source:False
struct_ion_custom_data._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('arg', ctypes.c_uint64),
]

ION_IOC_CUSTOM = _IOWR ( 'I' , 6 , struct_ion_custom_data ) # macro (from list)
_UAPI_MSM_ION_H = True # macro
ION_HEAP_TYPE_IOMMU = ION_HEAP_TYPE_SYSTEM # macro
ION_FLAG_CP_TOUCH = (1<<17) # macro
ION_FLAG_CP_BITSTREAM = (1<<18) # macro
ION_FLAG_CP_PIXEL = (1<<19) # macro
ION_FLAG_CP_NON_PIXEL = (1<<20) # macro
ION_FLAG_CP_CAMERA = (1<<21) # macro
ION_FLAG_CP_HLOS = (1<<22) # macro
ION_FLAG_CP_HLOS_FREE = (1<<23) # macro
ION_FLAG_CP_SEC_DISPLAY = (1<<25) # macro
ION_FLAG_CP_APP = (1<<26) # macro
ION_FLAG_ALLOW_NON_CONTIG = (1<<24) # macro
ION_FLAG_FORCE_CONTIGUOUS = (1<<30) # macro
ION_FLAG_POOL_FORCE_ALLOC = (1<<16) # macro
ION_FLAG_POOL_PREFETCH = (1<<27) # macro
ION_FORCE_CONTIGUOUS = (1<<30) # macro
def ION_HEAP(bit):  # macro
   return (1<<(bit))
ION_ADSP_HEAP_NAME = "adsp" # macro
ION_SYSTEM_HEAP_NAME = "system" # macro
ION_VMALLOC_HEAP_NAME = "system" # macro
ION_KMALLOC_HEAP_NAME = "kmalloc" # macro
ION_AUDIO_HEAP_NAME = "audio" # macro
ION_SF_HEAP_NAME = "sf" # macro
ION_MM_HEAP_NAME = "mm" # macro
ION_CAMERA_HEAP_NAME = "camera_preview" # macro
ION_IOMMU_HEAP_NAME = "iommu" # macro
ION_MFC_HEAP_NAME = "mfc" # macro
ION_WB_HEAP_NAME = "wb" # macro
ION_MM_FIRMWARE_HEAP_NAME = "mm_fw" # macro
ION_PIL1_HEAP_NAME = "pil_1" # macro
ION_PIL2_HEAP_NAME = "pil_2" # macro
ION_QSECOM_HEAP_NAME = "qsecom" # macro
ION_SECURE_HEAP_NAME = "secure_heap" # macro
ION_SECURE_DISPLAY_HEAP_NAME = "secure_display" # macro
def ION_SET_CACHED(__cache):  # macro
   return (__cache|1)
def ION_SET_UNCACHED(__cache):  # macro
   return (__cache&~1)
def ION_IS_CACHED(__flags):  # macro
   return ((__flags)&1)
ION_IOC_MSM_MAGIC = 'M' # macro

# values for enumeration 'msm_ion_heap_types'
msm_ion_heap_types__enumvalues = {
    6: 'ION_HEAP_TYPE_MSM_START',
    6: 'ION_HEAP_TYPE_SECURE_DMA',
    7: 'ION_HEAP_TYPE_SYSTEM_SECURE',
    8: 'ION_HEAP_TYPE_HYP_CMA',
}
ION_HEAP_TYPE_MSM_START = 6
ION_HEAP_TYPE_SECURE_DMA = 6
ION_HEAP_TYPE_SYSTEM_SECURE = 7
ION_HEAP_TYPE_HYP_CMA = 8
msm_ion_heap_types = ctypes.c_uint32 # enum

# values for enumeration 'ion_heap_ids'
ion_heap_ids__enumvalues = {
    -1: 'INVALID_HEAP_ID',
    8: 'ION_CP_MM_HEAP_ID',
    9: 'ION_SECURE_HEAP_ID',
    10: 'ION_SECURE_DISPLAY_HEAP_ID',
    12: 'ION_CP_MFC_HEAP_ID',
    16: 'ION_CP_WB_HEAP_ID',
    20: 'ION_CAMERA_HEAP_ID',
    21: 'ION_SYSTEM_CONTIG_HEAP_ID',
    22: 'ION_ADSP_HEAP_ID',
    23: 'ION_PIL1_HEAP_ID',
    24: 'ION_SF_HEAP_ID',
    25: 'ION_SYSTEM_HEAP_ID',
    26: 'ION_PIL2_HEAP_ID',
    27: 'ION_QSECOM_HEAP_ID',
    28: 'ION_AUDIO_HEAP_ID',
    29: 'ION_MM_FIRMWARE_HEAP_ID',
    31: 'ION_HEAP_ID_RESERVED',
}
INVALID_HEAP_ID = -1
ION_CP_MM_HEAP_ID = 8
ION_SECURE_HEAP_ID = 9
ION_SECURE_DISPLAY_HEAP_ID = 10
ION_CP_MFC_HEAP_ID = 12
ION_CP_WB_HEAP_ID = 16
ION_CAMERA_HEAP_ID = 20
ION_SYSTEM_CONTIG_HEAP_ID = 21
ION_ADSP_HEAP_ID = 22
ION_PIL1_HEAP_ID = 23
ION_SF_HEAP_ID = 24
ION_SYSTEM_HEAP_ID = 25
ION_PIL2_HEAP_ID = 26
ION_QSECOM_HEAP_ID = 27
ION_AUDIO_HEAP_ID = 28
ION_MM_FIRMWARE_HEAP_ID = 29
ION_HEAP_ID_RESERVED = 31
ion_heap_ids = ctypes.c_int32 # enum
ION_IOMMU_HEAP_ID = ION_SYSTEM_HEAP_ID # macro
ION_FLAG_SECURE = (1<<ION_HEAP_ID_RESERVED) # macro
ION_SECURE = (1<<ION_HEAP_ID_RESERVED) # macro

# values for enumeration 'ion_fixed_position'
ion_fixed_position__enumvalues = {
    0: 'NOT_FIXED',
    1: 'FIXED_LOW',
    2: 'FIXED_MIDDLE',
    3: 'FIXED_HIGH',
}
NOT_FIXED = 0
FIXED_LOW = 1
FIXED_MIDDLE = 2
FIXED_HIGH = 3
ion_fixed_position = ctypes.c_uint32 # enum

# values for enumeration 'cp_mem_usage'
cp_mem_usage__enumvalues = {
    1: 'VIDEO_BITSTREAM',
    2: 'VIDEO_PIXEL',
    3: 'VIDEO_NONPIXEL',
    4: 'DISPLAY_SECURE_CP_USAGE',
    5: 'CAMERA_SECURE_CP_USAGE',
    6: 'MAX_USAGE',
    2147483647: 'UNKNOWN',
}
VIDEO_BITSTREAM = 1
VIDEO_PIXEL = 2
VIDEO_NONPIXEL = 3
DISPLAY_SECURE_CP_USAGE = 4
CAMERA_SECURE_CP_USAGE = 5
MAX_USAGE = 6
UNKNOWN = 2147483647
cp_mem_usage = ctypes.c_uint32 # enum
class struct_ion_flush_data(Structure):
    pass

struct_ion_flush_data._pack_ = 1 # source:False
struct_ion_flush_data._fields_ = [
    ('handle', ctypes.c_int32),
    ('fd', ctypes.c_int32),
    ('vaddr', ctypes.POINTER(None)),
    ('offset', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

ION_IOC_CLEAN_CACHES = _IOWR ( 'M' , 0 , struct_ion_flush_data ) # macro (from list)
ION_IOC_INV_CACHES = _IOWR ( 'M' , 1 , struct_ion_flush_data ) # macro (from list)
ION_IOC_CLEAN_INV_CACHES = _IOWR ( 'M' , 2 , struct_ion_flush_data ) # macro (from list)
class struct_ion_prefetch_regions(Structure):
    pass

struct_ion_prefetch_regions._pack_ = 1 # source:False
struct_ion_prefetch_regions._fields_ = [
    ('vmid', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('sizes', ctypes.POINTER(ctypes.c_uint64)),
    ('nr_sizes', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_ion_prefetch_data(Structure):
    pass

struct_ion_prefetch_data._pack_ = 1 # source:False
struct_ion_prefetch_data._fields_ = [
    ('heap_id', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('len', ctypes.c_uint64),
    ('regions', ctypes.POINTER(struct_ion_prefetch_regions)),
    ('nr_regions', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ION_IOC_PREFETCH = _IOWR ( 'M' , 3 , struct_ion_prefetch_data ) # macro (from list)
ION_IOC_DRAIN = _IOWR ( 'M' , 4 , struct_ion_prefetch_data ) # macro (from list)
ADSPRPC_SHARED_H = True # macro
FASTRPC_GLINK_GUID = "fastrpcglink-apps-dsp" # macro
FASTRPC_SMD_GUID = "fastrpcsmd-apps-dsp" # macro
DEVICE_NAME = "adsprpc-smd" # macro
FASTRPC_ATTR_NOVA = 0x1 # macro
FASTRPC_ATTR_NON_COHERENT = 0x2 # macro
FASTRPC_ATTR_COHERENT = 0x4 # macro
FASTRPC_ATTR_KEEP_MAP = 0x8 # macro
FASTRPC_ATTR_NOMAP = (16) # macro
FASTRPC_MODE_PARALLEL = 0 # macro
FASTRPC_MODE_SERIAL = 1 # macro
FASTRPC_MODE_PROFILE = 2 # macro
FASTRPC_MODE_SESSION = 4 # macro
FASTRPC_INIT_ATTACH = 0 # macro
FASTRPC_INIT_CREATE = 1 # macro
FASTRPC_INIT_CREATE_STATIC = 2 # macro
FASTRPC_INIT_ATTACH_SENSORS = 3 # macro
def REMOTE_SCALARS_INBUFS(sc):  # macro
   return (((sc)>>16)&0x0ff)
def REMOTE_SCALARS_OUTBUFS(sc):  # macro
   return (((sc)>>8)&0x0ff)
def REMOTE_SCALARS_INHANDLES(sc):  # macro
   return (((sc)>>4)&0x0f)
def REMOTE_SCALARS_OUTHANDLES(sc):  # macro
   return ((sc)&0x0f)
def REMOTE_SCALARS_LENGTH(sc):  # macro
   return (REMOTE_SCALARS_INBUFS(sc)+REMOTE_SCALARS_OUTBUFS(sc)+REMOTE_SCALARS_INHANDLES(sc)+REMOTE_SCALARS_OUTHANDLES(sc))
def REMOTE_SCALARS_MAKE(method, __in, out):  # macro
   return REMOTE_SCALARS_MAKEX(0,method,__in,out,0,0)
# def VERIFY_EPRINTF(format, args):  # macro
#    return (void)0
# def VERIFY_IPRINTF(args):  # macro
#    return (void)0
# def __STR__(x):  # macro
#    return #x":"
def __TOSTR__(x):  # macro
   return __STR__(x)
# __FILE_LINE__ = __FILE__ ":" __TOSTR__ ( __LINE__ ) # macro
# def VERIFY(err, val):  # macro
#    return {VERIFY_IPRINTF(__FILE__":"__TOSTR__(__LINE__)"info: calling: "#val"\n");((val)==0){(err)=(err)==0?-1:(err);VERIFY_EPRINTF(__FILE__":"__TOSTR__(__LINE__)"error: %d: "#val"\n",(err));}{VERIFY_IPRINTF(__FILE__":"__TOSTR__(__LINE__)"info: passed: "#val"\n");}\}(0)
# remote_arg64_t = remote_arg64 # macro
# remote_arg_t = remote_arg # macro
FASTRPC_CONTROL_LATENCY = (1) # macro
FASTRPC_CONTROL_SMMU = (2) # macro
FASTRPC_CONTROL_KALLOC = (3) # macro
class struct_remote_buf64(Structure):
    pass

struct_remote_buf64._pack_ = 1 # source:False
struct_remote_buf64._fields_ = [
    ('pv', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
]

class struct_remote_dma_handle64(Structure):
    pass

struct_remote_dma_handle64._pack_ = 1 # source:False
struct_remote_dma_handle64._fields_ = [
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_uint32),
    ('len', ctypes.c_uint32),
]

class union_remote_arg64(Union):
    pass

union_remote_arg64._pack_ = 1 # source:False
union_remote_arg64._fields_ = [
    ('buf', struct_remote_buf64),
    ('dma', struct_remote_dma_handle64),
    ('h', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 12),
]

class struct_remote_buf(Structure):
    pass

struct_remote_buf._pack_ = 1 # source:False
struct_remote_buf._fields_ = [
    ('pv', ctypes.POINTER(None)),
    ('len', ctypes.c_uint64),
]

class struct_remote_dma_handle(Structure):
    pass

struct_remote_dma_handle._pack_ = 1 # source:False
struct_remote_dma_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_uint32),
]

class union_remote_arg(Union):
    pass

union_remote_arg._pack_ = 1 # source:False
union_remote_arg._fields_ = [
    ('buf', struct_remote_buf),
    ('dma', struct_remote_dma_handle),
    ('h', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 12),
]

class struct_fastrpc_ioctl_invoke(Structure):
    pass

struct_fastrpc_ioctl_invoke._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke._fields_ = [
    ('handle', ctypes.c_uint32),
    ('sc', ctypes.c_uint32),
    ('pra', ctypes.POINTER(union_remote_arg)),
]

FASTRPC_IOCTL_INVOKE = _IOWR ( 'R' , 1 , struct_fastrpc_ioctl_invoke ) # macro (from list)
class struct_fastrpc_ioctl_invoke_fd(Structure):
    pass

struct_fastrpc_ioctl_invoke_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_fd._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
]

FASTRPC_IOCTL_INVOKE_FD = _IOWR ( 'R' , 4 , struct_fastrpc_ioctl_invoke_fd ) # macro (from list)
class struct_fastrpc_ioctl_invoke_attrs(Structure):
    pass

struct_fastrpc_ioctl_invoke_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_attrs._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
]

FASTRPC_IOCTL_INVOKE_ATTRS = _IOWR ( 'R' , 7 , struct_fastrpc_ioctl_invoke_attrs ) # macro (from list)
class struct_fastrpc_ioctl_invoke_crc(Structure):
    pass

struct_fastrpc_ioctl_invoke_crc._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_crc._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
    ('crc', ctypes.POINTER(ctypes.c_uint32)),
]

FASTRPC_IOCTL_INVOKE_CRC = _IOWR ( 'R' , 11 , struct_fastrpc_ioctl_invoke_crc ) # macro (from list)
class struct_fastrpc_ioctl_init(Structure):
    pass

struct_fastrpc_ioctl_init._pack_ = 1 # source:False
struct_fastrpc_ioctl_init._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('file', ctypes.c_uint64),
    ('filelen', ctypes.c_uint32),
    ('filefd', ctypes.c_int32),
    ('mem', ctypes.c_uint64),
    ('memlen', ctypes.c_uint32),
    ('memfd', ctypes.c_int32),
]

FASTRPC_IOCTL_INIT = _IOWR ( 'R' , 6 , struct_fastrpc_ioctl_init ) # macro (from list)
class struct_fastrpc_ioctl_init_attrs(Structure):
    pass

struct_fastrpc_ioctl_init_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_init_attrs._fields_ = [
    ('init', struct_fastrpc_ioctl_init),
    ('attrs', ctypes.c_int32),
    ('siglen', ctypes.c_uint32),
]

FASTRPC_IOCTL_INIT_ATTRS = _IOWR ( 'R' , 10 , struct_fastrpc_ioctl_init_attrs ) # macro (from list)
class struct_fastrpc_ioctl_munmap(Structure):
    pass

struct_fastrpc_ioctl_munmap._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

FASTRPC_IOCTL_MUNMAP = _IOWR ( 'R' , 3 , struct_fastrpc_ioctl_munmap ) # macro (from list)
class struct_fastrpc_ioctl_munmap_64(Structure):
    pass

struct_fastrpc_ioctl_munmap_64._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_64._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

FASTRPC_IOCTL_MUNMAP_64 = _IOWR ( 'R' , 15 , struct_fastrpc_ioctl_munmap_64 ) # macro (from list)
class struct_fastrpc_ioctl_mmap(Structure):
    pass

struct_fastrpc_ioctl_mmap._pack_ = 1 # source:False
struct_fastrpc_ioctl_mmap._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('vaddrin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('vaddrout', ctypes.c_uint64),
]

FASTRPC_IOCTL_MMAP = _IOWR ( 'R' , 2 , struct_fastrpc_ioctl_mmap ) # macro (from list)
class struct_fastrpc_ioctl_mmap_64(Structure):
    pass

struct_fastrpc_ioctl_mmap_64._pack_ = 1 # source:False
struct_fastrpc_ioctl_mmap_64._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('vaddrin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('vaddrout', ctypes.c_uint64),
]

FASTRPC_IOCTL_MMAP_64 = _IOWR ( 'R' , 14 , struct_fastrpc_ioctl_mmap_64 ) # macro (from list)
class struct_fastrpc_ioctl_munmap_fd(Structure):
    pass

struct_fastrpc_ioctl_munmap_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_fd._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('va', ctypes.c_uint64),
    ('len', ctypes.c_int64),
]

FASTRPC_IOCTL_MUNMAP_FD = _IOWR ( 'R' , 13 , struct_fastrpc_ioctl_munmap_fd ) # macro (from list)
class struct_fastrpc_ioctl_perf(Structure):
    pass

struct_fastrpc_ioctl_perf._pack_ = 1 # source:False
struct_fastrpc_ioctl_perf._fields_ = [
    ('data', ctypes.c_uint64),
    ('numkeys', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('keys', ctypes.c_uint64),
]

FASTRPC_IOCTL_GETPERF = _IOWR ( 'R' , 9 , struct_fastrpc_ioctl_perf ) # macro (from list)
class struct_fastrpc_ctrl_latency(Structure):
    pass

struct_fastrpc_ctrl_latency._pack_ = 1 # source:False
struct_fastrpc_ctrl_latency._fields_ = [
    ('enable', ctypes.c_uint32),
    ('level', ctypes.c_uint32),
]

class struct_fastrpc_ctrl_smmu(Structure):
    pass

struct_fastrpc_ctrl_smmu._pack_ = 1 # source:False
struct_fastrpc_ctrl_smmu._fields_ = [
    ('sharedcb', ctypes.c_uint32),
]

class struct_fastrpc_ctrl_kalloc(Structure):
    pass

struct_fastrpc_ctrl_kalloc._pack_ = 1 # source:False
struct_fastrpc_ctrl_kalloc._fields_ = [
    ('kalloc_support', ctypes.c_uint32),
]

class struct_fastrpc_ioctl_control(Structure):
    pass

class union_fastrpc_ioctl_control_0(Union):
    pass

union_fastrpc_ioctl_control_0._pack_ = 1 # source:False
union_fastrpc_ioctl_control_0._fields_ = [
    ('lp', struct_fastrpc_ctrl_latency),
    ('smmu', struct_fastrpc_ctrl_smmu),
    ('kalloc', struct_fastrpc_ctrl_kalloc),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_fastrpc_ioctl_control._pack_ = 1 # source:False
struct_fastrpc_ioctl_control._anonymous_ = ('_0',)
struct_fastrpc_ioctl_control._fields_ = [
    ('req', ctypes.c_uint32),
    ('_0', union_fastrpc_ioctl_control_0),
]

FASTRPC_IOCTL_CONTROL = _IOWR ( 'R' , 12 , struct_fastrpc_ioctl_control ) # macro (from list)
class struct_smq_null_invoke(Structure):
    pass

struct_smq_null_invoke._pack_ = 1 # source:False
struct_smq_null_invoke._fields_ = [
    ('ctx', ctypes.c_uint64),
    ('handle', ctypes.c_uint32),
    ('sc', ctypes.c_uint32),
]

class struct_smq_phy_page(Structure):
    pass

struct_smq_phy_page._pack_ = 1 # source:False
struct_smq_phy_page._fields_ = [
    ('addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_smq_invoke_buf(Structure):
    pass

struct_smq_invoke_buf._pack_ = 1 # source:False
struct_smq_invoke_buf._fields_ = [
    ('num', ctypes.c_int32),
    ('pgidx', ctypes.c_int32),
]

class struct_smq_invoke(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_smq_null_invoke),
    ('page', struct_smq_phy_page),
     ]

class struct_smq_msg(Structure):
    pass

struct_smq_msg._pack_ = 1 # source:False
struct_smq_msg._fields_ = [
    ('pid', ctypes.c_uint32),
    ('tid', ctypes.c_uint32),
    ('invoke', struct_smq_invoke),
]

class struct_smq_invoke_rsp(Structure):
    pass

struct_smq_invoke_rsp._pack_ = 1 # source:False
struct_smq_invoke_rsp._fields_ = [
    ('ctx', ctypes.c_uint64),
    ('retval', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

uint32_t = ctypes.c_uint32
FASTRPC_IOCTL_SETMODE = _IOWR ( 'R' , 5 , uint32_t ) # macro (from list)
FASTRPC_IOCTL_GETINFO = _IOWR ( 'R' , 8 , uint32_t ) # macro (from list)
def REMOTE_SCALARS_MAKEX(attr, method, __in, out, oin, oout):  # macro
   return ((((uint32_t)(attr)&0x7)<<29)|(((uint32_t)(method)&0x1f)<<24)|(((uint32_t)(__in)&0xff)<<16)|(((uint32_t)(out)&0xff)<<8)|(((uint32_t)(oin)&0x0f)<<4)|((uint32_t)(oout)&0x0f))
try:
    smq_invoke_buf_start = _libraries['FIXME_STUB'].smq_invoke_buf_start
    smq_invoke_buf_start.restype = ctypes.POINTER(struct_smq_invoke_buf)
    smq_invoke_buf_start.argtypes = [ctypes.POINTER(union_remote_arg64), uint32_t]
except AttributeError:
    pass
try:
    smq_phy_page_start = _libraries['FIXME_STUB'].smq_phy_page_start
    smq_phy_page_start.restype = ctypes.POINTER(struct_smq_phy_page)
    smq_phy_page_start.argtypes = [uint32_t, ctypes.POINTER(struct_smq_invoke_buf)]
except AttributeError:
    pass
__all__ = \
    ['ADSPRPC_SHARED_H', 'CAMERA_SECURE_CP_USAGE', 'DEVICE_NAME',
    'DISPLAY_SECURE_CP_USAGE', 'FASTRPC_ATTR_COHERENT',
    'FASTRPC_ATTR_KEEP_MAP', 'FASTRPC_ATTR_NOMAP',
    'FASTRPC_ATTR_NON_COHERENT', 'FASTRPC_ATTR_NOVA',
    'FASTRPC_CONTROL_KALLOC', 'FASTRPC_CONTROL_LATENCY',
    'FASTRPC_CONTROL_SMMU', 'FASTRPC_GLINK_GUID',
    'FASTRPC_INIT_ATTACH', 'FASTRPC_INIT_ATTACH_SENSORS',
    'FASTRPC_INIT_CREATE', 'FASTRPC_INIT_CREATE_STATIC',
    'FASTRPC_MODE_PARALLEL', 'FASTRPC_MODE_PROFILE',
    'FASTRPC_MODE_SERIAL', 'FASTRPC_MODE_SESSION', 'FASTRPC_SMD_GUID',
    'FIXED_HIGH', 'FIXED_LOW', 'FIXED_MIDDLE', 'INVALID_HEAP_ID',
    'ION_ADSP_HEAP_ID', 'ION_ADSP_HEAP_NAME', 'ION_AUDIO_HEAP_ID',
    'ION_AUDIO_HEAP_NAME', 'ION_CAMERA_HEAP_ID',
    'ION_CAMERA_HEAP_NAME', 'ION_CP_MFC_HEAP_ID', 'ION_CP_MM_HEAP_ID',
    'ION_CP_WB_HEAP_ID', 'ION_FLAG_ALLOW_NON_CONTIG',
    'ION_FLAG_CACHED', 'ION_FLAG_CACHED_NEEDS_SYNC',
    'ION_FLAG_CP_APP', 'ION_FLAG_CP_BITSTREAM', 'ION_FLAG_CP_CAMERA',
    'ION_FLAG_CP_HLOS', 'ION_FLAG_CP_HLOS_FREE',
    'ION_FLAG_CP_NON_PIXEL', 'ION_FLAG_CP_PIXEL',
    'ION_FLAG_CP_SEC_DISPLAY', 'ION_FLAG_CP_TOUCH',
    'ION_FLAG_FORCE_CONTIGUOUS', 'ION_FLAG_POOL_FORCE_ALLOC',
    'ION_FLAG_POOL_PREFETCH', 'ION_FLAG_SECURE',
    'ION_FORCE_CONTIGUOUS', 'ION_HEAP_CARVEOUT_MASK',
    'ION_HEAP_ID_RESERVED', 'ION_HEAP_SYSTEM_CONTIG_MASK',
    'ION_HEAP_SYSTEM_MASK', 'ION_HEAP_TYPE_CARVEOUT',
    'ION_HEAP_TYPE_CHUNK', 'ION_HEAP_TYPE_CUSTOM',
    'ION_HEAP_TYPE_DMA', 'ION_HEAP_TYPE_DMA_MASK',
    'ION_HEAP_TYPE_HYP_CMA', 'ION_HEAP_TYPE_IOMMU',
    'ION_HEAP_TYPE_MSM_START', 'ION_HEAP_TYPE_SECURE_DMA',
    'ION_HEAP_TYPE_SYSTEM', 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    'ION_HEAP_TYPE_SYSTEM_SECURE', 'ION_IOC_MAGIC',
    'ION_IOC_MSM_MAGIC', 'ION_IOMMU_HEAP_ID', 'ION_IOMMU_HEAP_NAME',
    'ION_KMALLOC_HEAP_NAME', 'ION_MFC_HEAP_NAME',
    'ION_MM_FIRMWARE_HEAP_ID', 'ION_MM_FIRMWARE_HEAP_NAME',
    'ION_MM_HEAP_NAME', 'ION_NUM_HEAPS', 'ION_PIL1_HEAP_ID',
    'ION_PIL1_HEAP_NAME', 'ION_PIL2_HEAP_ID', 'ION_PIL2_HEAP_NAME',
    'ION_QSECOM_HEAP_ID', 'ION_QSECOM_HEAP_NAME', 'ION_SECURE',
    'ION_SECURE_DISPLAY_HEAP_ID', 'ION_SECURE_DISPLAY_HEAP_NAME',
    'ION_SECURE_HEAP_ID', 'ION_SECURE_HEAP_NAME', 'ION_SF_HEAP_ID',
    'ION_SF_HEAP_NAME', 'ION_SYSTEM_CONTIG_HEAP_ID',
    'ION_SYSTEM_HEAP_ID', 'ION_SYSTEM_HEAP_NAME',
    'ION_VMALLOC_HEAP_NAME', 'ION_WB_HEAP_NAME', 'MAX_USAGE',
    'NOT_FIXED', 'UNKNOWN', 'VIDEO_BITSTREAM', 'VIDEO_NONPIXEL',
    'VIDEO_PIXEL', '_IO', '_IOR', '_IOW', '_IOWR',
    '_UAPI_LINUX_ION_H', '_UAPI_MSM_ION_H', 'cp_mem_usage',
    'ion_fixed_position', 'ion_heap_ids', 'ion_heap_type',
    'ion_user_handle_t', 'msm_ion_heap_types', 'smq_invoke_buf_start',
    'smq_phy_page_start', 'struct_fastrpc_ctrl_kalloc',
    'struct_fastrpc_ctrl_latency', 'struct_fastrpc_ctrl_smmu',
    'struct_fastrpc_ioctl_control', 'struct_fastrpc_ioctl_init',
    'struct_fastrpc_ioctl_init_attrs', 'struct_fastrpc_ioctl_invoke',
    'struct_fastrpc_ioctl_invoke_attrs',
    'struct_fastrpc_ioctl_invoke_crc',
    'struct_fastrpc_ioctl_invoke_fd', 'struct_fastrpc_ioctl_mmap',
    'struct_fastrpc_ioctl_mmap_64', 'struct_fastrpc_ioctl_munmap',
    'struct_fastrpc_ioctl_munmap_64',
    'struct_fastrpc_ioctl_munmap_fd', 'struct_fastrpc_ioctl_perf',
    'struct_ion_allocation_data', 'struct_ion_custom_data',
    'struct_ion_fd_data', 'struct_ion_flush_data',
    'struct_ion_handle_data', 'struct_ion_prefetch_data',
    'struct_ion_prefetch_regions', 'struct_remote_buf',
    'struct_remote_buf64', 'struct_remote_dma_handle',
    'struct_remote_dma_handle64', 'struct_smq_invoke',
    'struct_smq_invoke_buf', 'struct_smq_invoke_rsp',
    'struct_smq_msg', 'struct_smq_null_invoke', 'struct_smq_phy_page',
    'uint32_t', 'union_fastrpc_ioctl_control_0', 'union_remote_arg',
    'union_remote_arg64']
