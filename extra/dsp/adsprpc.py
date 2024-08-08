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


ADSPRPC_SHARED_H = True # macro
# FASTRPC_IOCTL_INVOKE = _IOWR ( 'R' , 1 , struct fastrpc_ioctl_invoke ) # macro
# FASTRPC_IOCTL_MMAP = _IOWR ( 'R' , 2 , struct fastrpc_ioctl_mmap ) # macro
# FASTRPC_IOCTL_MUNMAP = _IOWR ( 'R' , 3 , struct fastrpc_ioctl_munmap ) # macro
# FASTRPC_IOCTL_MMAP_64 = _IOWR ( 'R' , 14 , struct fastrpc_ioctl_mmap_64 ) # macro
# FASTRPC_IOCTL_MUNMAP_64 = _IOWR ( 'R' , 15 , struct fastrpc_ioctl_munmap_64 ) # macro
# FASTRPC_IOCTL_INVOKE_FD = _IOWR ( 'R' , 4 , struct fastrpc_ioctl_invoke_fd ) # macro
# FASTRPC_IOCTL_SETMODE = _IOWR ( 'R' , 5 , uint32_t ) # macro
# FASTRPC_IOCTL_INIT = _IOWR ( 'R' , 6 , struct fastrpc_ioctl_init ) # macro
# FASTRPC_IOCTL_INVOKE_ATTRS = _IOWR ( 'R' , 7 , struct fastrpc_ioctl_invoke_attrs ) # macro
# FASTRPC_IOCTL_GETINFO = _IOWR ( 'R' , 8 , uint32_t ) # macro
# FASTRPC_IOCTL_GETPERF = _IOWR ( 'R' , 9 , struct fastrpc_ioctl_perf ) # macro
# FASTRPC_IOCTL_INIT_ATTRS = _IOWR ( 'R' , 10 , struct fastrpc_ioctl_init_attrs ) # macro
# FASTRPC_IOCTL_INVOKE_CRC = _IOWR ( 'R' , 11 , struct fastrpc_ioctl_invoke_crc ) # macro
# FASTRPC_IOCTL_CONTROL = _IOWR ( 'R' , 12 , struct fastrpc_ioctl_control ) # macro
# FASTRPC_IOCTL_MUNMAP_FD = _IOWR ( 'R' , 13 , struct fastrpc_ioctl_munmap_fd ) # macro
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
# def REMOTE_SCALARS_INBUFS(sc):  # macro
#    return (((sc)>>16)&0x0ff)
# def REMOTE_SCALARS_OUTBUFS(sc):  # macro
#    return (((sc)>>8)&0x0ff)
# def REMOTE_SCALARS_INHANDLES(sc):  # macro
#    return (((sc)>>4)&0x0f)
# def REMOTE_SCALARS_OUTHANDLES(sc):  # macro
#    return ((sc)&0x0f)
# def REMOTE_SCALARS_LENGTH(sc):  # macro
#    return ((((sc)>>16)&0x0ff)(sc)+(((sc)>>8)&0x0ff)(sc)+(((sc)>>4)&0x0f)(sc)+((sc)&0x0f)(sc))
# def REMOTE_SCALARS_MAKEX(attr, method, in, out, oin, oout):  # macro
#    return ((((uint32_t)(attr)&0x7)<<29)|(((uint32_t)(method)&0x1f)<<24)|(((uint32_t)(in)&0xff)<<16)|(((uint32_t)(out)&0xff)<<8)|(((uint32_t)(oin)&0x0f)<<4)|((uint32_t)(oout)&0x0f))
# def REMOTE_SCALARS_MAKE(method, in, out):  # macro
#    return ((((uint32_t)(attr)&0x7)<<29)|(((uint32_t)(method)&0x1f)<<24)|(((uint32_t)(in)&0xff)<<16)|(((uint32_t)(out)&0xff)<<8)|(((uint32_t)(oin)&0x0f)<<4)|((uint32_t)(oout)&0x0f))(0,method,in,out,0,0)
# def VERIFY_EPRINTF(format, args):  # macro
#    return (void)0
# def VERIFY_IPRINTF(args):  # macro
#    return (void)0
# def __STR__(x):  # macro
#    return #x":"
# def __TOSTR__(x):  # macro
#    return #x":"(x)
# __FILE_LINE__ = __FILE__ ":" #x":"(x) ( __LINE__ ) # macro
# def VERIFY(err, val):  # macro
#    return do{(void)0([UndefinedIdentifier(name=__FILE__), '":"', '#x":"(x)', '(', UndefinedIdentifier(name=__LINE__), ')']"info: calling: "#val"\n");if((val)==0){(err)=(err)==0?-1:(err);(void)0([UndefinedIdentifier(name=__FILE__), '":"', '#x":"(x)', '(', UndefinedIdentifier(name=__LINE__), ')']"error: %d: "#val"\n",(err));}else{(void)0([UndefinedIdentifier(name=__FILE__), '":"', '#x":"(x)', '(', UndefinedIdentifier(name=__LINE__), ')']"info: passed: "#val"\n");}\
#}while(0)
# remote_arg64_t = union remote_arg64 # macro
# remote_arg_t = union remote_arg # macro
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

class struct_fastrpc_ioctl_invoke_fd(Structure):
    pass

struct_fastrpc_ioctl_invoke_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_fd._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
]

class struct_fastrpc_ioctl_invoke_attrs(Structure):
    pass

struct_fastrpc_ioctl_invoke_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_attrs._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
]

class struct_fastrpc_ioctl_invoke_crc(Structure):
    pass

struct_fastrpc_ioctl_invoke_crc._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_crc._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
    ('crc', ctypes.POINTER(ctypes.c_uint32)),
]

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

class struct_fastrpc_ioctl_init_attrs(Structure):
    pass

struct_fastrpc_ioctl_init_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_init_attrs._fields_ = [
    ('init', struct_fastrpc_ioctl_init),
    ('attrs', ctypes.c_int32),
    ('siglen', ctypes.c_uint32),
]

class struct_fastrpc_ioctl_munmap(Structure):
    pass

struct_fastrpc_ioctl_munmap._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_fastrpc_ioctl_munmap_64(Structure):
    pass

struct_fastrpc_ioctl_munmap_64._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_64._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

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

class struct_fastrpc_ioctl_munmap_fd(Structure):
    pass

struct_fastrpc_ioctl_munmap_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_fd._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('va', ctypes.c_uint64),
    ('len', ctypes.c_int64),
]

class struct_fastrpc_ioctl_perf(Structure):
    pass

struct_fastrpc_ioctl_perf._pack_ = 1 # source:False
struct_fastrpc_ioctl_perf._fields_ = [
    ('data', ctypes.c_uint64),
    ('numkeys', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('keys', ctypes.c_uint64),
]

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
    ['ADSPRPC_SHARED_H', 'DEVICE_NAME', 'FASTRPC_ATTR_COHERENT',
    'FASTRPC_ATTR_KEEP_MAP', 'FASTRPC_ATTR_NOMAP',
    'FASTRPC_ATTR_NON_COHERENT', 'FASTRPC_ATTR_NOVA',
    'FASTRPC_CONTROL_KALLOC', 'FASTRPC_CONTROL_LATENCY',
    'FASTRPC_CONTROL_SMMU', 'FASTRPC_GLINK_GUID',
    'FASTRPC_INIT_ATTACH', 'FASTRPC_INIT_ATTACH_SENSORS',
    'FASTRPC_INIT_CREATE', 'FASTRPC_INIT_CREATE_STATIC',
    'FASTRPC_MODE_PARALLEL', 'FASTRPC_MODE_PROFILE',
    'FASTRPC_MODE_SERIAL', 'FASTRPC_MODE_SESSION', 'FASTRPC_SMD_GUID',
    'smq_invoke_buf_start', 'smq_phy_page_start',
    'struct_fastrpc_ctrl_kalloc', 'struct_fastrpc_ctrl_latency',
    'struct_fastrpc_ctrl_smmu', 'struct_fastrpc_ioctl_control',
    'struct_fastrpc_ioctl_init', 'struct_fastrpc_ioctl_init_attrs',
    'struct_fastrpc_ioctl_invoke',
    'struct_fastrpc_ioctl_invoke_attrs',
    'struct_fastrpc_ioctl_invoke_crc',
    'struct_fastrpc_ioctl_invoke_fd', 'struct_fastrpc_ioctl_mmap',
    'struct_fastrpc_ioctl_mmap_64', 'struct_fastrpc_ioctl_munmap',
    'struct_fastrpc_ioctl_munmap_64',
    'struct_fastrpc_ioctl_munmap_fd', 'struct_fastrpc_ioctl_perf',
    'struct_remote_buf', 'struct_remote_buf64',
    'struct_remote_dma_handle', 'struct_remote_dma_handle64',
    'struct_smq_invoke', 'struct_smq_invoke_buf',
    'struct_smq_invoke_rsp', 'struct_smq_msg',
    'struct_smq_null_invoke', 'struct_smq_phy_page', 'uint32_t',
    'union_fastrpc_ioctl_control_0', 'union_remote_arg',
    'union_remote_arg64']
