# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
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
if ctypes.sizeof(ctypes.c_longdouble) == 8:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*8



_MACHO_LOADER_H_ = True # macro
MH_MAGIC = 0xfeedface # macro
MH_CIGAM = 0xcefaedfe # macro
MH_MAGIC_64 = 0xfeedfacf # macro
MH_CIGAM_64 = 0xcffaedfe # macro
MH_OBJECT = 0x1 # macro
MH_EXECUTE = 0x2 # macro
MH_FVMLIB = 0x3 # macro
MH_CORE = 0x4 # macro
MH_PRELOAD = 0x5 # macro
MH_DYLIB = 0x6 # macro
MH_DYLINKER = 0x7 # macro
MH_BUNDLE = 0x8 # macro
MH_DYLIB_STUB = 0x9 # macro
MH_DSYM = 0xa # macro
MH_KEXT_BUNDLE = 0xb # macro
MH_FILESET = 0xc # macro
MH_GPU_EXECUTE = 0xd # macro
MH_GPU_DYLIB = 0xe # macro
MH_NOUNDEFS = 0x1 # macro
MH_INCRLINK = 0x2 # macro
MH_DYLDLINK = 0x4 # macro
MH_BINDATLOAD = 0x8 # macro
MH_PREBOUND = 0x10 # macro
MH_SPLIT_SEGS = 0x20 # macro
MH_LAZY_INIT = 0x40 # macro
MH_TWOLEVEL = 0x80 # macro
MH_FORCE_FLAT = 0x100 # macro
MH_NOMULTIDEFS = 0x200 # macro
MH_NOFIXPREBINDING = 0x400 # macro
MH_PREBINDABLE = 0x800 # macro
MH_ALLMODSBOUND = 0x1000 # macro
MH_SUBSECTIONS_VIA_SYMBOLS = 0x2000 # macro
MH_CANONICAL = 0x4000 # macro
MH_WEAK_DEFINES = 0x8000 # macro
MH_BINDS_TO_WEAK = 0x10000 # macro
MH_ALLOW_STACK_EXECUTION = 0x20000 # macro
MH_ROOT_SAFE = 0x40000 # macro
MH_SETUID_SAFE = 0x80000 # macro
MH_NO_REEXPORTED_DYLIBS = 0x100000 # macro
MH_PIE = 0x200000 # macro
MH_DEAD_STRIPPABLE_DYLIB = 0x400000 # macro
MH_HAS_TLV_DESCRIPTORS = 0x800000 # macro
MH_NO_HEAP_EXECUTION = 0x1000000 # macro
MH_APP_EXTENSION_SAFE = 0x02000000 # macro
MH_NLIST_OUTOFSYNC_WITH_DYLDINFO = 0x04000000 # macro
MH_SIM_SUPPORT = 0x08000000 # macro
MH_DYLIB_IN_CACHE = 0x80000000 # macro
LC_REQ_DYLD = 0x80000000 # macro
LC_SEGMENT = 0x1 # macro
LC_SYMTAB = 0x2 # macro
LC_SYMSEG = 0x3 # macro
LC_THREAD = 0x4 # macro
LC_UNIXTHREAD = 0x5 # macro
LC_LOADFVMLIB = 0x6 # macro
LC_IDFVMLIB = 0x7 # macro
LC_IDENT = 0x8 # macro
LC_FVMFILE = 0x9 # macro
LC_PREPAGE = 0xa # macro
LC_DYSYMTAB = 0xb # macro
LC_LOAD_DYLIB = 0xc # macro
LC_ID_DYLIB = 0xd # macro
LC_LOAD_DYLINKER = 0xe # macro
LC_ID_DYLINKER = 0xf # macro
LC_PREBOUND_DYLIB = 0x10 # macro
LC_ROUTINES = 0x11 # macro
LC_SUB_FRAMEWORK = 0x12 # macro
LC_SUB_UMBRELLA = 0x13 # macro
LC_SUB_CLIENT = 0x14 # macro
LC_SUB_LIBRARY = 0x15 # macro
LC_TWOLEVEL_HINTS = 0x16 # macro
LC_PREBIND_CKSUM = 0x17 # macro
LC_LOAD_WEAK_DYLIB = (0x18|0x80000000) # macro
LC_SEGMENT_64 = 0x19 # macro
LC_ROUTINES_64 = 0x1a # macro
LC_UUID = 0x1b # macro
LC_RPATH = (0x1c|0x80000000) # macro
LC_CODE_SIGNATURE = 0x1d # macro
LC_SEGMENT_SPLIT_INFO = 0x1e # macro
LC_REEXPORT_DYLIB = (0x1f|0x80000000) # macro
LC_LAZY_LOAD_DYLIB = 0x20 # macro
LC_ENCRYPTION_INFO = 0x21 # macro
LC_DYLD_INFO = 0x22 # macro
LC_DYLD_INFO_ONLY = (0x22|0x80000000) # macro
LC_LOAD_UPWARD_DYLIB = (0x23|0x80000000) # macro
LC_VERSION_MIN_MACOSX = 0x24 # macro
LC_VERSION_MIN_IPHONEOS = 0x25 # macro
LC_FUNCTION_STARTS = 0x26 # macro
LC_DYLD_ENVIRONMENT = 0x27 # macro
LC_MAIN = (0x28|0x80000000) # macro
LC_DATA_IN_CODE = 0x29 # macro
LC_SOURCE_VERSION = 0x2A # macro
LC_DYLIB_CODE_SIGN_DRS = 0x2B # macro
LC_ENCRYPTION_INFO_64 = 0x2C # macro
LC_LINKER_OPTION = 0x2D # macro
LC_LINKER_OPTIMIZATION_HINT = 0x2E # macro
LC_VERSION_MIN_TVOS = 0x2F # macro
LC_VERSION_MIN_WATCHOS = 0x30 # macro
LC_NOTE = 0x31 # macro
LC_BUILD_VERSION = 0x32 # macro
LC_DYLD_EXPORTS_TRIE = (0x33|0x80000000) # macro
LC_DYLD_CHAINED_FIXUPS = (0x34|0x80000000) # macro
LC_FILESET_ENTRY = (0x35|0x80000000) # macro
LC_ATOM_INFO = 0x36 # macro
SG_HIGHVM = 0x1 # macro
SG_FVMLIB = 0x2 # macro
SG_NORELOC = 0x4 # macro
SG_PROTECTED_VERSION_1 = 0x8 # macro
SG_READ_ONLY = 0x10 # macro
SECTION_TYPE = 0x000000ff # macro
SECTION_ATTRIBUTES = 0xffffff00 # macro
S_REGULAR = 0x0 # macro
S_ZEROFILL = 0x1 # macro
S_CSTRING_LITERALS = 0x2 # macro
S_4BYTE_LITERALS = 0x3 # macro
S_8BYTE_LITERALS = 0x4 # macro
S_LITERAL_POINTERS = 0x5 # macro
S_NON_LAZY_SYMBOL_POINTERS = 0x6 # macro
S_LAZY_SYMBOL_POINTERS = 0x7 # macro
S_SYMBOL_STUBS = 0x8 # macro
S_MOD_INIT_FUNC_POINTERS = 0x9 # macro
S_MOD_TERM_FUNC_POINTERS = 0xa # macro
S_COALESCED = 0xb # macro
S_GB_ZEROFILL = 0xc # macro
S_INTERPOSING = 0xd # macro
S_16BYTE_LITERALS = 0xe # macro
S_DTRACE_DOF = 0xf # macro
S_LAZY_DYLIB_SYMBOL_POINTERS = 0x10 # macro
S_THREAD_LOCAL_REGULAR = 0x11 # macro
S_THREAD_LOCAL_ZEROFILL = 0x12 # macro
S_THREAD_LOCAL_VARIABLES = 0x13 # macro
S_THREAD_LOCAL_VARIABLE_POINTERS = 0x14 # macro
S_THREAD_LOCAL_INIT_FUNCTION_POINTERS = 0x15 # macro
S_INIT_FUNC_OFFSETS = 0x16 # macro
SECTION_ATTRIBUTES_USR = 0xff000000 # macro
S_ATTR_PURE_INSTRUCTIONS = 0x80000000 # macro
S_ATTR_NO_TOC = 0x40000000 # macro
S_ATTR_STRIP_STATIC_SYMS = 0x20000000 # macro
S_ATTR_NO_DEAD_STRIP = 0x10000000 # macro
S_ATTR_LIVE_SUPPORT = 0x08000000 # macro
S_ATTR_SELF_MODIFYING_CODE = 0x04000000 # macro
S_ATTR_DEBUG = 0x02000000 # macro
SECTION_ATTRIBUTES_SYS = 0x00ffff00 # macro
S_ATTR_SOME_INSTRUCTIONS = 0x00000400 # macro
S_ATTR_EXT_RELOC = 0x00000200 # macro
S_ATTR_LOC_RELOC = 0x00000100 # macro
SEG_PAGEZERO = "__PAGEZERO" # macro
SEG_TEXT = "__TEXT" # macro
SECT_TEXT = "__text" # macro
SECT_FVMLIB_INIT0 = "__fvmlib_init0" # macro
SECT_FVMLIB_INIT1 = "__fvmlib_init1" # macro
SEG_DATA = "__DATA" # macro
SECT_DATA = "__data" # macro
SECT_BSS = "__bss" # macro
SECT_COMMON = "__common" # macro
SEG_OBJC = "__OBJC" # macro
SECT_OBJC_SYMBOLS = "__symbol_table" # macro
SECT_OBJC_MODULES = "__module_info" # macro
SECT_OBJC_STRINGS = "__selector_strs" # macro
SECT_OBJC_REFS = "__selector_refs" # macro
SEG_ICON = "__ICON" # macro
SECT_ICON_HEADER = "__header" # macro
SECT_ICON_TIFF = "__tiff" # macro
SEG_LINKEDIT = "__LINKEDIT" # macro
SEG_UNIXSTACK = "__UNIXSTACK" # macro
SEG_IMPORT = "__IMPORT" # macro
INDIRECT_SYMBOL_LOCAL = 0x80000000 # macro
INDIRECT_SYMBOL_ABS = 0x40000000 # macro
PLATFORM_UNKNOWN = 0 # macro
PLATFORM_ANY = 0xFFFFFFFF # macro
PLATFORM_MACOS = 1 # macro
PLATFORM_IOS = 2 # macro
PLATFORM_TVOS = 3 # macro
PLATFORM_WATCHOS = 4 # macro
PLATFORM_BRIDGEOS = 5 # macro
PLATFORM_MACCATALYST = 6 # macro
PLATFORM_IOSSIMULATOR = 7 # macro
PLATFORM_TVOSSIMULATOR = 8 # macro
PLATFORM_WATCHOSSIMULATOR = 9 # macro
PLATFORM_DRIVERKIT = 10 # macro
PLATFORM_FIRMWARE = 13 # macro
PLATFORM_SEPOS = 14 # macro
TOOL_CLANG = 1 # macro
TOOL_SWIFT = 2 # macro
TOOL_LD = 3 # macro
TOOL_LLD = 4 # macro
TOOL_METAL = 1024 # macro
TOOL_AIRLLD = 1025 # macro
TOOL_AIRNT = 1026 # macro
TOOL_AIRNT_PLUGIN = 1027 # macro
TOOL_AIRPACK = 1028 # macro
TOOL_GPUARCHIVER = 1031 # macro
TOOL_METAL_FRAMEWORK = 1032 # macro
REBASE_TYPE_POINTER = 1 # macro
REBASE_TYPE_TEXT_ABSOLUTE32 = 2 # macro
REBASE_TYPE_TEXT_PCREL32 = 3 # macro
REBASE_OPCODE_MASK = 0xF0 # macro
REBASE_IMMEDIATE_MASK = 0x0F # macro
REBASE_OPCODE_DONE = 0x00 # macro
REBASE_OPCODE_SET_TYPE_IMM = 0x10 # macro
REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB = 0x20 # macro
REBASE_OPCODE_ADD_ADDR_ULEB = 0x30 # macro
REBASE_OPCODE_ADD_ADDR_IMM_SCALED = 0x40 # macro
REBASE_OPCODE_DO_REBASE_IMM_TIMES = 0x50 # macro
REBASE_OPCODE_DO_REBASE_ULEB_TIMES = 0x60 # macro
REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB = 0x70 # macro
REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB = 0x80 # macro
BIND_TYPE_POINTER = 1 # macro
BIND_TYPE_TEXT_ABSOLUTE32 = 2 # macro
BIND_TYPE_TEXT_PCREL32 = 3 # macro
BIND_SPECIAL_DYLIB_SELF = 0 # macro
BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE = -1 # macro
BIND_SPECIAL_DYLIB_FLAT_LOOKUP = -2 # macro
BIND_SPECIAL_DYLIB_WEAK_LOOKUP = -3 # macro
BIND_SYMBOL_FLAGS_WEAK_IMPORT = 0x1 # macro
BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION = 0x8 # macro
BIND_OPCODE_MASK = 0xF0 # macro
BIND_IMMEDIATE_MASK = 0x0F # macro
BIND_OPCODE_DONE = 0x00 # macro
BIND_OPCODE_SET_DYLIB_ORDINAL_IMM = 0x10 # macro
BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB = 0x20 # macro
BIND_OPCODE_SET_DYLIB_SPECIAL_IMM = 0x30 # macro
BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM = 0x40 # macro
BIND_OPCODE_SET_TYPE_IMM = 0x50 # macro
BIND_OPCODE_SET_ADDEND_SLEB = 0x60 # macro
BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB = 0x70 # macro
BIND_OPCODE_ADD_ADDR_ULEB = 0x80 # macro
BIND_OPCODE_DO_BIND = 0x90 # macro
BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB = 0xA0 # macro
BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED = 0xB0 # macro
BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB = 0xC0 # macro
BIND_OPCODE_THREADED = 0xD0 # macro
BIND_SUBOPCODE_THREADED_SET_BIND_ORDINAL_TABLE_SIZE_ULEB = 0x00 # macro
BIND_SUBOPCODE_THREADED_APPLY = 0x01 # macro
EXPORT_SYMBOL_FLAGS_KIND_MASK = 0x03 # macro
EXPORT_SYMBOL_FLAGS_KIND_REGULAR = 0x00 # macro
EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL = 0x01 # macro
EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE = 0x02 # macro
EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION = 0x04 # macro
EXPORT_SYMBOL_FLAGS_REEXPORT = 0x08 # macro
EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER = 0x10 # macro
EXPORT_SYMBOL_FLAGS_STATIC_RESOLVER = 0x20 # macro
DICE_KIND_DATA = 0x0001 # macro
DICE_KIND_JUMP_TABLE8 = 0x0002 # macro
DICE_KIND_JUMP_TABLE16 = 0x0003 # macro
DICE_KIND_JUMP_TABLE32 = 0x0004 # macro
DICE_KIND_ABS_JUMP_TABLE32 = 0x0005 # macro
class struct_mach_header(Structure):
    pass

struct_mach_header._pack_ = 1 # source:False
struct_mach_header._fields_ = [
    ('magic', ctypes.c_uint32),
    ('cputype', ctypes.c_int32),
    ('cpusubtype', ctypes.c_int32),
    ('filetype', ctypes.c_uint32),
    ('ncmds', ctypes.c_uint32),
    ('sizeofcmds', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_mach_header_64(Structure):
    pass

struct_mach_header_64._pack_ = 1 # source:False
struct_mach_header_64._fields_ = [
    ('magic', ctypes.c_uint32),
    ('cputype', ctypes.c_int32),
    ('cpusubtype', ctypes.c_int32),
    ('filetype', ctypes.c_uint32),
    ('ncmds', ctypes.c_uint32),
    ('sizeofcmds', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_load_command(Structure):
    pass

struct_load_command._pack_ = 1 # source:False
struct_load_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
]

class union_lc_str(Union):
    pass

union_lc_str._pack_ = 1 # source:False
union_lc_str._fields_ = [
    ('offset', ctypes.c_uint32),
]

class struct_segment_command(Structure):
    pass

struct_segment_command._pack_ = 1 # source:False
struct_segment_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('segname', ctypes.c_char * 16),
    ('vmaddr', ctypes.c_uint32),
    ('vmsize', ctypes.c_uint32),
    ('fileoff', ctypes.c_uint32),
    ('filesize', ctypes.c_uint32),
    ('maxprot', ctypes.c_int32),
    ('initprot', ctypes.c_int32),
    ('nsects', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_segment_command_64(Structure):
    pass

struct_segment_command_64._pack_ = 1 # source:False
struct_segment_command_64._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('segname', ctypes.c_char * 16),
    ('vmaddr', ctypes.c_uint64),
    ('vmsize', ctypes.c_uint64),
    ('fileoff', ctypes.c_uint64),
    ('filesize', ctypes.c_uint64),
    ('maxprot', ctypes.c_int32),
    ('initprot', ctypes.c_int32),
    ('nsects', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_section(Structure):
    pass

struct_section._pack_ = 1 # source:False
struct_section._fields_ = [
    ('sectname', ctypes.c_char * 16),
    ('segname', ctypes.c_char * 16),
    ('addr', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('align', ctypes.c_uint32),
    ('reloff', ctypes.c_uint32),
    ('nreloc', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint32),
]

class struct_section_64(Structure):
    pass

struct_section_64._pack_ = 1 # source:False
struct_section_64._fields_ = [
    ('sectname', ctypes.c_char * 16),
    ('segname', ctypes.c_char * 16),
    ('addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('offset', ctypes.c_uint32),
    ('align', ctypes.c_uint32),
    ('reloff', ctypes.c_uint32),
    ('nreloc', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint32),
    ('reserved3', ctypes.c_uint32),
]

class struct_fvmlib(Structure):
    pass

struct_fvmlib._pack_ = 1 # source:False
struct_fvmlib._fields_ = [
    ('name', union_lc_str),
    ('minor_version', ctypes.c_uint32),
    ('header_addr', ctypes.c_uint32),
]

class struct_fvmlib_command(Structure):
    pass

struct_fvmlib_command._pack_ = 1 # source:False
struct_fvmlib_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('fvmlib', struct_fvmlib),
]

class struct_dylib(Structure):
    pass

struct_dylib._pack_ = 1 # source:False
struct_dylib._fields_ = [
    ('name', union_lc_str),
    ('timestamp', ctypes.c_uint32),
    ('current_version', ctypes.c_uint32),
    ('compatibility_version', ctypes.c_uint32),
]

class struct_dylib_command(Structure):
    pass

struct_dylib_command._pack_ = 1 # source:False
struct_dylib_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('dylib', struct_dylib),
]

class struct_sub_framework_command(Structure):
    pass

struct_sub_framework_command._pack_ = 1 # source:False
struct_sub_framework_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('umbrella', union_lc_str),
]

class struct_sub_client_command(Structure):
    pass

struct_sub_client_command._pack_ = 1 # source:False
struct_sub_client_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('client', union_lc_str),
]

class struct_sub_umbrella_command(Structure):
    pass

struct_sub_umbrella_command._pack_ = 1 # source:False
struct_sub_umbrella_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('sub_umbrella', union_lc_str),
]

class struct_sub_library_command(Structure):
    pass

struct_sub_library_command._pack_ = 1 # source:False
struct_sub_library_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('sub_library', union_lc_str),
]

class struct_prebound_dylib_command(Structure):
    pass

struct_prebound_dylib_command._pack_ = 1 # source:False
struct_prebound_dylib_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('name', union_lc_str),
    ('nmodules', ctypes.c_uint32),
    ('linked_modules', union_lc_str),
]

class struct_dylinker_command(Structure):
    pass

struct_dylinker_command._pack_ = 1 # source:False
struct_dylinker_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('name', union_lc_str),
]

class struct_thread_command(Structure):
    pass

struct_thread_command._pack_ = 1 # source:False
struct_thread_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
]

class struct_routines_command(Structure):
    pass

struct_routines_command._pack_ = 1 # source:False
struct_routines_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('init_address', ctypes.c_uint32),
    ('init_module', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint32),
    ('reserved3', ctypes.c_uint32),
    ('reserved4', ctypes.c_uint32),
    ('reserved5', ctypes.c_uint32),
    ('reserved6', ctypes.c_uint32),
]

class struct_routines_command_64(Structure):
    pass

struct_routines_command_64._pack_ = 1 # source:False
struct_routines_command_64._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('init_address', ctypes.c_uint64),
    ('init_module', ctypes.c_uint64),
    ('reserved1', ctypes.c_uint64),
    ('reserved2', ctypes.c_uint64),
    ('reserved3', ctypes.c_uint64),
    ('reserved4', ctypes.c_uint64),
    ('reserved5', ctypes.c_uint64),
    ('reserved6', ctypes.c_uint64),
]

class struct_symtab_command(Structure):
    pass

struct_symtab_command._pack_ = 1 # source:False
struct_symtab_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('symoff', ctypes.c_uint32),
    ('nsyms', ctypes.c_uint32),
    ('stroff', ctypes.c_uint32),
    ('strsize', ctypes.c_uint32),
]

class struct_dysymtab_command(Structure):
    pass

struct_dysymtab_command._pack_ = 1 # source:False
struct_dysymtab_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('ilocalsym', ctypes.c_uint32),
    ('nlocalsym', ctypes.c_uint32),
    ('iextdefsym', ctypes.c_uint32),
    ('nextdefsym', ctypes.c_uint32),
    ('iundefsym', ctypes.c_uint32),
    ('nundefsym', ctypes.c_uint32),
    ('tocoff', ctypes.c_uint32),
    ('ntoc', ctypes.c_uint32),
    ('modtaboff', ctypes.c_uint32),
    ('nmodtab', ctypes.c_uint32),
    ('extrefsymoff', ctypes.c_uint32),
    ('nextrefsyms', ctypes.c_uint32),
    ('indirectsymoff', ctypes.c_uint32),
    ('nindirectsyms', ctypes.c_uint32),
    ('extreloff', ctypes.c_uint32),
    ('nextrel', ctypes.c_uint32),
    ('locreloff', ctypes.c_uint32),
    ('nlocrel', ctypes.c_uint32),
]

class struct_dylib_table_of_contents(Structure):
    pass

struct_dylib_table_of_contents._pack_ = 1 # source:False
struct_dylib_table_of_contents._fields_ = [
    ('symbol_index', ctypes.c_uint32),
    ('module_index', ctypes.c_uint32),
]

class struct_dylib_module(Structure):
    pass

struct_dylib_module._pack_ = 1 # source:False
struct_dylib_module._fields_ = [
    ('module_name', ctypes.c_uint32),
    ('iextdefsym', ctypes.c_uint32),
    ('nextdefsym', ctypes.c_uint32),
    ('irefsym', ctypes.c_uint32),
    ('nrefsym', ctypes.c_uint32),
    ('ilocalsym', ctypes.c_uint32),
    ('nlocalsym', ctypes.c_uint32),
    ('iextrel', ctypes.c_uint32),
    ('nextrel', ctypes.c_uint32),
    ('iinit_iterm', ctypes.c_uint32),
    ('ninit_nterm', ctypes.c_uint32),
    ('objc_module_info_addr', ctypes.c_uint32),
    ('objc_module_info_size', ctypes.c_uint32),
]

class struct_dylib_module_64(Structure):
    pass

struct_dylib_module_64._pack_ = 1 # source:False
struct_dylib_module_64._fields_ = [
    ('module_name', ctypes.c_uint32),
    ('iextdefsym', ctypes.c_uint32),
    ('nextdefsym', ctypes.c_uint32),
    ('irefsym', ctypes.c_uint32),
    ('nrefsym', ctypes.c_uint32),
    ('ilocalsym', ctypes.c_uint32),
    ('nlocalsym', ctypes.c_uint32),
    ('iextrel', ctypes.c_uint32),
    ('nextrel', ctypes.c_uint32),
    ('iinit_iterm', ctypes.c_uint32),
    ('ninit_nterm', ctypes.c_uint32),
    ('objc_module_info_size', ctypes.c_uint32),
    ('objc_module_info_addr', ctypes.c_uint64),
]

class struct_dylib_reference(Structure):
    pass

struct_dylib_reference._pack_ = 1 # source:False
struct_dylib_reference._fields_ = [
    ('isym', ctypes.c_uint32, 24),
    ('flags', ctypes.c_uint32, 8),
]

class struct_twolevel_hints_command(Structure):
    pass

struct_twolevel_hints_command._pack_ = 1 # source:False
struct_twolevel_hints_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('nhints', ctypes.c_uint32),
]

class struct_twolevel_hint(Structure):
    pass

struct_twolevel_hint._pack_ = 1 # source:False
struct_twolevel_hint._fields_ = [
    ('isub_image', ctypes.c_uint32, 8),
    ('itoc', ctypes.c_uint32, 24),
]

class struct_prebind_cksum_command(Structure):
    pass

struct_prebind_cksum_command._pack_ = 1 # source:False
struct_prebind_cksum_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('cksum', ctypes.c_uint32),
]

class struct_uuid_command(Structure):
    pass

struct_uuid_command._pack_ = 1 # source:False
struct_uuid_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('uuid', ctypes.c_ubyte * 16),
]

class struct_rpath_command(Structure):
    pass

struct_rpath_command._pack_ = 1 # source:False
struct_rpath_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('path', union_lc_str),
]

class struct_linkedit_data_command(Structure):
    pass

struct_linkedit_data_command._pack_ = 1 # source:False
struct_linkedit_data_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('dataoff', ctypes.c_uint32),
    ('datasize', ctypes.c_uint32),
]

class struct_encryption_info_command(Structure):
    pass

struct_encryption_info_command._pack_ = 1 # source:False
struct_encryption_info_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('cryptoff', ctypes.c_uint32),
    ('cryptsize', ctypes.c_uint32),
    ('cryptid', ctypes.c_uint32),
]

class struct_encryption_info_command_64(Structure):
    pass

struct_encryption_info_command_64._pack_ = 1 # source:False
struct_encryption_info_command_64._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('cryptoff', ctypes.c_uint32),
    ('cryptsize', ctypes.c_uint32),
    ('cryptid', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_version_min_command(Structure):
    pass

struct_version_min_command._pack_ = 1 # source:False
struct_version_min_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('version', ctypes.c_uint32),
    ('sdk', ctypes.c_uint32),
]

class struct_build_version_command(Structure):
    pass

struct_build_version_command._pack_ = 1 # source:False
struct_build_version_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('platform', ctypes.c_uint32),
    ('minos', ctypes.c_uint32),
    ('sdk', ctypes.c_uint32),
    ('ntools', ctypes.c_uint32),
]

class struct_build_tool_version(Structure):
    pass

struct_build_tool_version._pack_ = 1 # source:False
struct_build_tool_version._fields_ = [
    ('tool', ctypes.c_uint32),
    ('version', ctypes.c_uint32),
]

class struct_dyld_info_command(Structure):
    pass

struct_dyld_info_command._pack_ = 1 # source:False
struct_dyld_info_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('rebase_off', ctypes.c_uint32),
    ('rebase_size', ctypes.c_uint32),
    ('bind_off', ctypes.c_uint32),
    ('bind_size', ctypes.c_uint32),
    ('weak_bind_off', ctypes.c_uint32),
    ('weak_bind_size', ctypes.c_uint32),
    ('lazy_bind_off', ctypes.c_uint32),
    ('lazy_bind_size', ctypes.c_uint32),
    ('export_off', ctypes.c_uint32),
    ('export_size', ctypes.c_uint32),
]

class struct_linker_option_command(Structure):
    pass

struct_linker_option_command._pack_ = 1 # source:False
struct_linker_option_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

class struct_symseg_command(Structure):
    pass

struct_symseg_command._pack_ = 1 # source:False
struct_symseg_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
]

class struct_ident_command(Structure):
    pass

struct_ident_command._pack_ = 1 # source:False
struct_ident_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
]

class struct_fvmfile_command(Structure):
    pass

struct_fvmfile_command._pack_ = 1 # source:False
struct_fvmfile_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('name', union_lc_str),
    ('header_addr', ctypes.c_uint32),
]

class struct_entry_point_command(Structure):
    pass

struct_entry_point_command._pack_ = 1 # source:False
struct_entry_point_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('entryoff', ctypes.c_uint64),
    ('stacksize', ctypes.c_uint64),
]

class struct_source_version_command(Structure):
    pass

struct_source_version_command._pack_ = 1 # source:False
struct_source_version_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('version', ctypes.c_uint64),
]

class struct_data_in_code_entry(Structure):
    pass

struct_data_in_code_entry._pack_ = 1 # source:False
struct_data_in_code_entry._fields_ = [
    ('offset', ctypes.c_uint32),
    ('length', ctypes.c_uint16),
    ('kind', ctypes.c_uint16),
]

class struct_tlv_descriptor(Structure):
    pass

struct_tlv_descriptor._pack_ = 1 # source:False
struct_tlv_descriptor._fields_ = [
    ('thunk', ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.POINTER(struct_tlv_descriptor))),
    ('key', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

class struct_note_command(Structure):
    pass

struct_note_command._pack_ = 1 # source:False
struct_note_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('data_owner', ctypes.c_char * 16),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_fileset_entry_command(Structure):
    pass

struct_fileset_entry_command._pack_ = 1 # source:False
struct_fileset_entry_command._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('cmdsize', ctypes.c_uint32),
    ('vmaddr', ctypes.c_uint64),
    ('fileoff', ctypes.c_uint64),
    ('entry_id', union_lc_str),
    ('reserved', ctypes.c_uint32),
]

__all__ = \
    ['BIND_IMMEDIATE_MASK', 'BIND_OPCODE_ADD_ADDR_ULEB',
    'BIND_OPCODE_DONE', 'BIND_OPCODE_DO_BIND',
    'BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED',
    'BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB',
    'BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB',
    'BIND_OPCODE_MASK', 'BIND_OPCODE_SET_ADDEND_SLEB',
    'BIND_OPCODE_SET_DYLIB_ORDINAL_IMM',
    'BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB',
    'BIND_OPCODE_SET_DYLIB_SPECIAL_IMM',
    'BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB',
    'BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM',
    'BIND_OPCODE_SET_TYPE_IMM', 'BIND_OPCODE_THREADED',
    'BIND_SPECIAL_DYLIB_FLAT_LOOKUP',
    'BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE', 'BIND_SPECIAL_DYLIB_SELF',
    'BIND_SPECIAL_DYLIB_WEAK_LOOKUP', 'BIND_SUBOPCODE_THREADED_APPLY',
    'BIND_SUBOPCODE_THREADED_SET_BIND_ORDINAL_TABLE_SIZE_ULEB',
    'BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION',
    'BIND_SYMBOL_FLAGS_WEAK_IMPORT', 'BIND_TYPE_POINTER',
    'BIND_TYPE_TEXT_ABSOLUTE32', 'BIND_TYPE_TEXT_PCREL32',
    'DICE_KIND_ABS_JUMP_TABLE32', 'DICE_KIND_DATA',
    'DICE_KIND_JUMP_TABLE16', 'DICE_KIND_JUMP_TABLE32',
    'DICE_KIND_JUMP_TABLE8', 'EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE',
    'EXPORT_SYMBOL_FLAGS_KIND_MASK',
    'EXPORT_SYMBOL_FLAGS_KIND_REGULAR',
    'EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL',
    'EXPORT_SYMBOL_FLAGS_REEXPORT',
    'EXPORT_SYMBOL_FLAGS_STATIC_RESOLVER',
    'EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER',
    'EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION', 'INDIRECT_SYMBOL_ABS',
    'INDIRECT_SYMBOL_LOCAL', 'LC_ATOM_INFO', 'LC_BUILD_VERSION',
    'LC_CODE_SIGNATURE', 'LC_DATA_IN_CODE', 'LC_DYLD_CHAINED_FIXUPS',
    'LC_DYLD_ENVIRONMENT', 'LC_DYLD_EXPORTS_TRIE', 'LC_DYLD_INFO',
    'LC_DYLD_INFO_ONLY', 'LC_DYLIB_CODE_SIGN_DRS', 'LC_DYSYMTAB',
    'LC_ENCRYPTION_INFO', 'LC_ENCRYPTION_INFO_64', 'LC_FILESET_ENTRY',
    'LC_FUNCTION_STARTS', 'LC_FVMFILE', 'LC_IDENT', 'LC_IDFVMLIB',
    'LC_ID_DYLIB', 'LC_ID_DYLINKER', 'LC_LAZY_LOAD_DYLIB',
    'LC_LINKER_OPTIMIZATION_HINT', 'LC_LINKER_OPTION',
    'LC_LOADFVMLIB', 'LC_LOAD_DYLIB', 'LC_LOAD_DYLINKER',
    'LC_LOAD_UPWARD_DYLIB', 'LC_LOAD_WEAK_DYLIB', 'LC_MAIN',
    'LC_NOTE', 'LC_PREBIND_CKSUM', 'LC_PREBOUND_DYLIB', 'LC_PREPAGE',
    'LC_REEXPORT_DYLIB', 'LC_REQ_DYLD', 'LC_ROUTINES',
    'LC_ROUTINES_64', 'LC_RPATH', 'LC_SEGMENT', 'LC_SEGMENT_64',
    'LC_SEGMENT_SPLIT_INFO', 'LC_SOURCE_VERSION', 'LC_SUB_CLIENT',
    'LC_SUB_FRAMEWORK', 'LC_SUB_LIBRARY', 'LC_SUB_UMBRELLA',
    'LC_SYMSEG', 'LC_SYMTAB', 'LC_THREAD', 'LC_TWOLEVEL_HINTS',
    'LC_UNIXTHREAD', 'LC_UUID', 'LC_VERSION_MIN_IPHONEOS',
    'LC_VERSION_MIN_MACOSX', 'LC_VERSION_MIN_TVOS',
    'LC_VERSION_MIN_WATCHOS', 'MH_ALLMODSBOUND',
    'MH_ALLOW_STACK_EXECUTION', 'MH_APP_EXTENSION_SAFE',
    'MH_BINDATLOAD', 'MH_BINDS_TO_WEAK', 'MH_BUNDLE', 'MH_CANONICAL',
    'MH_CIGAM', 'MH_CIGAM_64', 'MH_CORE', 'MH_DEAD_STRIPPABLE_DYLIB',
    'MH_DSYM', 'MH_DYLDLINK', 'MH_DYLIB', 'MH_DYLIB_IN_CACHE',
    'MH_DYLIB_STUB', 'MH_DYLINKER', 'MH_EXECUTE', 'MH_FILESET',
    'MH_FORCE_FLAT', 'MH_FVMLIB', 'MH_GPU_DYLIB', 'MH_GPU_EXECUTE',
    'MH_HAS_TLV_DESCRIPTORS', 'MH_INCRLINK', 'MH_KEXT_BUNDLE',
    'MH_LAZY_INIT', 'MH_MAGIC', 'MH_MAGIC_64',
    'MH_NLIST_OUTOFSYNC_WITH_DYLDINFO', 'MH_NOFIXPREBINDING',
    'MH_NOMULTIDEFS', 'MH_NOUNDEFS', 'MH_NO_HEAP_EXECUTION',
    'MH_NO_REEXPORTED_DYLIBS', 'MH_OBJECT', 'MH_PIE',
    'MH_PREBINDABLE', 'MH_PREBOUND', 'MH_PRELOAD', 'MH_ROOT_SAFE',
    'MH_SETUID_SAFE', 'MH_SIM_SUPPORT', 'MH_SPLIT_SEGS',
    'MH_SUBSECTIONS_VIA_SYMBOLS', 'MH_TWOLEVEL', 'MH_WEAK_DEFINES',
    'PLATFORM_ANY', 'PLATFORM_BRIDGEOS', 'PLATFORM_DRIVERKIT',
    'PLATFORM_FIRMWARE', 'PLATFORM_IOS', 'PLATFORM_IOSSIMULATOR',
    'PLATFORM_MACCATALYST', 'PLATFORM_MACOS', 'PLATFORM_SEPOS',
    'PLATFORM_TVOS', 'PLATFORM_TVOSSIMULATOR', 'PLATFORM_UNKNOWN',
    'PLATFORM_WATCHOS', 'PLATFORM_WATCHOSSIMULATOR',
    'REBASE_IMMEDIATE_MASK', 'REBASE_OPCODE_ADD_ADDR_IMM_SCALED',
    'REBASE_OPCODE_ADD_ADDR_ULEB', 'REBASE_OPCODE_DONE',
    'REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB',
    'REBASE_OPCODE_DO_REBASE_IMM_TIMES',
    'REBASE_OPCODE_DO_REBASE_ULEB_TIMES',
    'REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB',
    'REBASE_OPCODE_MASK', 'REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB',
    'REBASE_OPCODE_SET_TYPE_IMM', 'REBASE_TYPE_POINTER',
    'REBASE_TYPE_TEXT_ABSOLUTE32', 'REBASE_TYPE_TEXT_PCREL32',
    'SECTION_ATTRIBUTES', 'SECTION_ATTRIBUTES_SYS',
    'SECTION_ATTRIBUTES_USR', 'SECTION_TYPE', 'SECT_BSS',
    'SECT_COMMON', 'SECT_DATA', 'SECT_FVMLIB_INIT0',
    'SECT_FVMLIB_INIT1', 'SECT_ICON_HEADER', 'SECT_ICON_TIFF',
    'SECT_OBJC_MODULES', 'SECT_OBJC_REFS', 'SECT_OBJC_STRINGS',
    'SECT_OBJC_SYMBOLS', 'SECT_TEXT', 'SEG_DATA', 'SEG_ICON',
    'SEG_IMPORT', 'SEG_LINKEDIT', 'SEG_OBJC', 'SEG_PAGEZERO',
    'SEG_TEXT', 'SEG_UNIXSTACK', 'SG_FVMLIB', 'SG_HIGHVM',
    'SG_NORELOC', 'SG_PROTECTED_VERSION_1', 'SG_READ_ONLY',
    'S_16BYTE_LITERALS', 'S_4BYTE_LITERALS', 'S_8BYTE_LITERALS',
    'S_ATTR_DEBUG', 'S_ATTR_EXT_RELOC', 'S_ATTR_LIVE_SUPPORT',
    'S_ATTR_LOC_RELOC', 'S_ATTR_NO_DEAD_STRIP', 'S_ATTR_NO_TOC',
    'S_ATTR_PURE_INSTRUCTIONS', 'S_ATTR_SELF_MODIFYING_CODE',
    'S_ATTR_SOME_INSTRUCTIONS', 'S_ATTR_STRIP_STATIC_SYMS',
    'S_COALESCED', 'S_CSTRING_LITERALS', 'S_DTRACE_DOF',
    'S_GB_ZEROFILL', 'S_INIT_FUNC_OFFSETS', 'S_INTERPOSING',
    'S_LAZY_DYLIB_SYMBOL_POINTERS', 'S_LAZY_SYMBOL_POINTERS',
    'S_LITERAL_POINTERS', 'S_MOD_INIT_FUNC_POINTERS',
    'S_MOD_TERM_FUNC_POINTERS', 'S_NON_LAZY_SYMBOL_POINTERS',
    'S_REGULAR', 'S_SYMBOL_STUBS',
    'S_THREAD_LOCAL_INIT_FUNCTION_POINTERS', 'S_THREAD_LOCAL_REGULAR',
    'S_THREAD_LOCAL_VARIABLES', 'S_THREAD_LOCAL_VARIABLE_POINTERS',
    'S_THREAD_LOCAL_ZEROFILL', 'S_ZEROFILL', 'TOOL_AIRLLD',
    'TOOL_AIRNT', 'TOOL_AIRNT_PLUGIN', 'TOOL_AIRPACK', 'TOOL_CLANG',
    'TOOL_GPUARCHIVER', 'TOOL_LD', 'TOOL_LLD', 'TOOL_METAL',
    'TOOL_METAL_FRAMEWORK', 'TOOL_SWIFT', '_MACHO_LOADER_H_',
    'struct_build_tool_version', 'struct_build_version_command',
    'struct_data_in_code_entry', 'struct_dyld_info_command',
    'struct_dylib', 'struct_dylib_command', 'struct_dylib_module',
    'struct_dylib_module_64', 'struct_dylib_reference',
    'struct_dylib_table_of_contents', 'struct_dylinker_command',
    'struct_dysymtab_command', 'struct_encryption_info_command',
    'struct_encryption_info_command_64', 'struct_entry_point_command',
    'struct_fileset_entry_command', 'struct_fvmfile_command',
    'struct_fvmlib', 'struct_fvmlib_command', 'struct_ident_command',
    'struct_linkedit_data_command', 'struct_linker_option_command',
    'struct_load_command', 'struct_mach_header',
    'struct_mach_header_64', 'struct_note_command',
    'struct_prebind_cksum_command', 'struct_prebound_dylib_command',
    'struct_routines_command', 'struct_routines_command_64',
    'struct_rpath_command', 'struct_section', 'struct_section_64',
    'struct_segment_command', 'struct_segment_command_64',
    'struct_source_version_command', 'struct_sub_client_command',
    'struct_sub_framework_command', 'struct_sub_library_command',
    'struct_sub_umbrella_command', 'struct_symseg_command',
    'struct_symtab_command', 'struct_thread_command',
    'struct_tlv_descriptor', 'struct_twolevel_hint',
    'struct_twolevel_hints_command', 'struct_uuid_command',
    'struct_version_min_command', 'union_lc_str']
