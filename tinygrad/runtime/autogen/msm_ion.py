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



_UAPI_MSM_ION_H = True # macro
# ION_IOMMU_HEAP_ID = ION_SYSTEM_HEAP_ID # macro
# ION_HEAP_TYPE_IOMMU = ION_HEAP_TYPE_SYSTEM # macro
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
# def ION_FLAG_SECURE(1<<ION_HEAP_ID_RESERVED):  # macro
#    return   
ION_FLAG_FORCE_CONTIGUOUS = (1<<30) # macro
ION_FLAG_POOL_FORCE_ALLOC = (1<<16) # macro
ION_FLAG_POOL_PREFETCH = (1<<27) # macro
# ION_SECURE =  # macro
ION_FORCE_CONTIGUOUS = (1<<30) # macro
# def ION_HEAP(bit):  # macro
#    return (1<<(bit))  
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
# def ION_SET_CACHED(__cache):  # macro
#    return (__cache|ION_FLAG_CACHED)  
# def ION_SET_UNCACHED(__cache):  # macro
#    return (__cache&~ION_FLAG_CACHED)  
# def ION_IS_CACHED(__flags):  # macro
#    return ((__flags)&ION_FLAG_CACHED)  
ION_IOC_MSM_MAGIC = 'M' # macro
# ION_IOC_CLEAN_CACHES = _IOWR ( 'M' , 0 , struct ion_flush_data ) # macro
# ION_IOC_INV_CACHES = _IOWR ( 'M' , 1 , struct ion_flush_data ) # macro
# ION_IOC_CLEAN_INV_CACHES = _IOWR ( 'M' , 2 , struct ion_flush_data ) # macro
# ION_IOC_PREFETCH = _IOWR ( 'M' , 3 , struct ion_prefetch_data ) # macro
# ION_IOC_DRAIN = _IOWR ( 'M' , 4 , struct ion_prefetch_data ) # macro

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

__all__ = \
    ['CAMERA_SECURE_CP_USAGE', 'DISPLAY_SECURE_CP_USAGE',
    'FIXED_HIGH', 'FIXED_LOW', 'FIXED_MIDDLE', 'INVALID_HEAP_ID',
    'ION_ADSP_HEAP_ID', 'ION_ADSP_HEAP_NAME', 'ION_AUDIO_HEAP_ID',
    'ION_AUDIO_HEAP_NAME', 'ION_CAMERA_HEAP_ID',
    'ION_CAMERA_HEAP_NAME', 'ION_CP_MFC_HEAP_ID', 'ION_CP_MM_HEAP_ID',
    'ION_CP_WB_HEAP_ID', 'ION_FLAG_ALLOW_NON_CONTIG',
    'ION_FLAG_CP_APP', 'ION_FLAG_CP_BITSTREAM', 'ION_FLAG_CP_CAMERA',
    'ION_FLAG_CP_HLOS', 'ION_FLAG_CP_HLOS_FREE',
    'ION_FLAG_CP_NON_PIXEL', 'ION_FLAG_CP_PIXEL',
    'ION_FLAG_CP_SEC_DISPLAY', 'ION_FLAG_CP_TOUCH',
    'ION_FLAG_FORCE_CONTIGUOUS', 'ION_FLAG_POOL_FORCE_ALLOC',
    'ION_FLAG_POOL_PREFETCH', 'ION_FORCE_CONTIGUOUS',
    'ION_HEAP_ID_RESERVED', 'ION_HEAP_TYPE_HYP_CMA',
    'ION_HEAP_TYPE_MSM_START', 'ION_HEAP_TYPE_SECURE_DMA',
    'ION_HEAP_TYPE_SYSTEM_SECURE', 'ION_IOC_MSM_MAGIC',
    'ION_IOMMU_HEAP_NAME', 'ION_KMALLOC_HEAP_NAME',
    'ION_MFC_HEAP_NAME', 'ION_MM_FIRMWARE_HEAP_ID',
    'ION_MM_FIRMWARE_HEAP_NAME', 'ION_MM_HEAP_NAME',
    'ION_PIL1_HEAP_ID', 'ION_PIL1_HEAP_NAME', 'ION_PIL2_HEAP_ID',
    'ION_PIL2_HEAP_NAME', 'ION_QSECOM_HEAP_ID',
    'ION_QSECOM_HEAP_NAME', 'ION_SECURE_DISPLAY_HEAP_ID',
    'ION_SECURE_DISPLAY_HEAP_NAME', 'ION_SECURE_HEAP_ID',
    'ION_SECURE_HEAP_NAME', 'ION_SF_HEAP_ID', 'ION_SF_HEAP_NAME',
    'ION_SYSTEM_CONTIG_HEAP_ID', 'ION_SYSTEM_HEAP_ID',
    'ION_SYSTEM_HEAP_NAME', 'ION_VMALLOC_HEAP_NAME',
    'ION_WB_HEAP_NAME', 'MAX_USAGE', 'NOT_FIXED', 'UNKNOWN',
    'VIDEO_BITSTREAM', 'VIDEO_NONPIXEL', 'VIDEO_PIXEL',
    '_UAPI_MSM_ION_H', 'cp_mem_usage', 'ion_fixed_position',
    'ion_heap_ids', 'msm_ion_heap_types', 'struct_ion_flush_data',
    'struct_ion_prefetch_data', 'struct_ion_prefetch_regions']
