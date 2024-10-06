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



_libraries = {}
_libraries['libpciaccess.so'] = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libpciaccess.so')
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





PCIACCESS_H = True # macro
# __deprecated = ((deprecated)) # macro
PCI_DEV_MAP_FLAG_WRITABLE = (1<<0) # macro
PCI_DEV_MAP_FLAG_WRITE_COMBINE = (1<<1) # macro
PCI_DEV_MAP_FLAG_CACHABLE = (1<<2) # macro
PCI_MATCH_ANY = (~0) # macro
def PCI_ID_COMPARE(a, b):  # macro
   return (((a)==(~0)) or ((a)==(b)))  
VGA_ARB_RSRC_NONE = 0x00 # macro
VGA_ARB_RSRC_LEGACY_IO = 0x01 # macro
VGA_ARB_RSRC_LEGACY_MEM = 0x02 # macro
VGA_ARB_RSRC_NORMAL_IO = 0x04 # macro
VGA_ARB_RSRC_NORMAL_MEM = 0x08 # macro
pciaddr_t = ctypes.c_uint64
class struct_pci_device_iterator(Structure):
    pass

class struct_pci_device(Structure):
    pass

try:
    pci_device_has_kernel_driver = _libraries['libpciaccess.so'].pci_device_has_kernel_driver
    pci_device_has_kernel_driver.restype = ctypes.c_int32
    pci_device_has_kernel_driver.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_is_boot_vga = _libraries['libpciaccess.so'].pci_device_is_boot_vga
    pci_device_is_boot_vga.restype = ctypes.c_int32
    pci_device_is_boot_vga.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_read_rom = _libraries['libpciaccess.so'].pci_device_read_rom
    pci_device_read_rom.restype = ctypes.c_int32
    pci_device_read_rom.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    pci_device_map_region = _libraries['libpciaccess.so'].pci_device_map_region
    pci_device_map_region.restype = ctypes.c_int32
    pci_device_map_region.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
try:
    pci_device_unmap_region = _libraries['libpciaccess.so'].pci_device_unmap_region
    pci_device_unmap_region.restype = ctypes.c_int32
    pci_device_unmap_region.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.c_uint32]
except AttributeError:
    pass
try:
    pci_device_map_range = _libraries['libpciaccess.so'].pci_device_map_range
    pci_device_map_range.restype = ctypes.c_int32
    pci_device_map_range.argtypes = [ctypes.POINTER(struct_pci_device), pciaddr_t, pciaddr_t, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    pci_device_unmap_range = _libraries['libpciaccess.so'].pci_device_unmap_range
    pci_device_unmap_range.restype = ctypes.c_int32
    pci_device_unmap_range.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None), pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_map_memory_range = _libraries['libpciaccess.so'].pci_device_map_memory_range
    pci_device_map_memory_range.restype = ctypes.c_int32
    pci_device_map_memory_range.argtypes = [ctypes.POINTER(struct_pci_device), pciaddr_t, pciaddr_t, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    pci_device_unmap_memory_range = _libraries['libpciaccess.so'].pci_device_unmap_memory_range
    pci_device_unmap_memory_range.restype = ctypes.c_int32
    pci_device_unmap_memory_range.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None), pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_probe = _libraries['libpciaccess.so'].pci_device_probe
    pci_device_probe.restype = ctypes.c_int32
    pci_device_probe.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
class struct_pci_agp_info(Structure):
    pass

try:
    pci_device_get_agp_info = _libraries['libpciaccess.so'].pci_device_get_agp_info
    pci_device_get_agp_info.restype = ctypes.POINTER(struct_pci_agp_info)
    pci_device_get_agp_info.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
class struct_pci_bridge_info(Structure):
    pass

try:
    pci_device_get_bridge_info = _libraries['libpciaccess.so'].pci_device_get_bridge_info
    pci_device_get_bridge_info.restype = ctypes.POINTER(struct_pci_bridge_info)
    pci_device_get_bridge_info.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
class struct_pci_pcmcia_bridge_info(Structure):
    pass

try:
    pci_device_get_pcmcia_bridge_info = _libraries['libpciaccess.so'].pci_device_get_pcmcia_bridge_info
    pci_device_get_pcmcia_bridge_info.restype = ctypes.POINTER(struct_pci_pcmcia_bridge_info)
    pci_device_get_pcmcia_bridge_info.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_get_bridge_buses = _libraries['libpciaccess.so'].pci_device_get_bridge_buses
    pci_device_get_bridge_buses.restype = ctypes.c_int32
    pci_device_get_bridge_buses.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    pci_system_init = _libraries['libpciaccess.so'].pci_system_init
    pci_system_init.restype = ctypes.c_int32
    pci_system_init.argtypes = []
except AttributeError:
    pass
try:
    pci_system_init_dev_mem = _libraries['libpciaccess.so'].pci_system_init_dev_mem
    pci_system_init_dev_mem.restype = None
    pci_system_init_dev_mem.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    pci_system_cleanup = _libraries['libpciaccess.so'].pci_system_cleanup
    pci_system_cleanup.restype = None
    pci_system_cleanup.argtypes = []
except AttributeError:
    pass
class struct_pci_slot_match(Structure):
    pass

try:
    pci_slot_match_iterator_create = _libraries['libpciaccess.so'].pci_slot_match_iterator_create
    pci_slot_match_iterator_create.restype = ctypes.POINTER(struct_pci_device_iterator)
    pci_slot_match_iterator_create.argtypes = [ctypes.POINTER(struct_pci_slot_match)]
except AttributeError:
    pass
class struct_pci_id_match(Structure):
    pass

try:
    pci_id_match_iterator_create = _libraries['libpciaccess.so'].pci_id_match_iterator_create
    pci_id_match_iterator_create.restype = ctypes.POINTER(struct_pci_device_iterator)
    pci_id_match_iterator_create.argtypes = [ctypes.POINTER(struct_pci_id_match)]
except AttributeError:
    pass
try:
    pci_iterator_destroy = _libraries['libpciaccess.so'].pci_iterator_destroy
    pci_iterator_destroy.restype = None
    pci_iterator_destroy.argtypes = [ctypes.POINTER(struct_pci_device_iterator)]
except AttributeError:
    pass
try:
    pci_device_next = _libraries['libpciaccess.so'].pci_device_next
    pci_device_next.restype = ctypes.POINTER(struct_pci_device)
    pci_device_next.argtypes = [ctypes.POINTER(struct_pci_device_iterator)]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    pci_device_find_by_slot = _libraries['libpciaccess.so'].pci_device_find_by_slot
    pci_device_find_by_slot.restype = ctypes.POINTER(struct_pci_device)
    pci_device_find_by_slot.argtypes = [uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError:
    pass
try:
    pci_device_get_parent_bridge = _libraries['libpciaccess.so'].pci_device_get_parent_bridge
    pci_device_get_parent_bridge.restype = ctypes.POINTER(struct_pci_device)
    pci_device_get_parent_bridge.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_get_strings = _libraries['libpciaccess.so'].pci_get_strings
    pci_get_strings.restype = None
    pci_get_strings.argtypes = [ctypes.POINTER(struct_pci_id_match), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    pci_device_get_device_name = _libraries['libpciaccess.so'].pci_device_get_device_name
    pci_device_get_device_name.restype = ctypes.POINTER(ctypes.c_char)
    pci_device_get_device_name.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_get_subdevice_name = _libraries['libpciaccess.so'].pci_device_get_subdevice_name
    pci_device_get_subdevice_name.restype = ctypes.POINTER(ctypes.c_char)
    pci_device_get_subdevice_name.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_get_vendor_name = _libraries['libpciaccess.so'].pci_device_get_vendor_name
    pci_device_get_vendor_name.restype = ctypes.POINTER(ctypes.c_char)
    pci_device_get_vendor_name.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_get_subvendor_name = _libraries['libpciaccess.so'].pci_device_get_subvendor_name
    pci_device_get_subvendor_name.restype = ctypes.POINTER(ctypes.c_char)
    pci_device_get_subvendor_name.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_enable = _libraries['libpciaccess.so'].pci_device_enable
    pci_device_enable.restype = None
    pci_device_enable.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_cfg_read = _libraries['libpciaccess.so'].pci_device_cfg_read
    pci_device_cfg_read.restype = ctypes.c_int32
    pci_device_cfg_read.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None), pciaddr_t, pciaddr_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    pci_device_cfg_read_u8 = _libraries['libpciaccess.so'].pci_device_cfg_read_u8
    pci_device_cfg_read_u8.restype = ctypes.c_int32
    pci_device_cfg_read_u8.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(ctypes.c_ubyte), pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_cfg_read_u16 = _libraries['libpciaccess.so'].pci_device_cfg_read_u16
    pci_device_cfg_read_u16.restype = ctypes.c_int32
    pci_device_cfg_read_u16.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(ctypes.c_uint16), pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_cfg_read_u32 = _libraries['libpciaccess.so'].pci_device_cfg_read_u32
    pci_device_cfg_read_u32.restype = ctypes.c_int32
    pci_device_cfg_read_u32.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(ctypes.c_uint32), pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_cfg_write = _libraries['libpciaccess.so'].pci_device_cfg_write
    pci_device_cfg_write.restype = ctypes.c_int32
    pci_device_cfg_write.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None), pciaddr_t, pciaddr_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    pci_device_cfg_write_u8 = _libraries['libpciaccess.so'].pci_device_cfg_write_u8
    pci_device_cfg_write_u8.restype = ctypes.c_int32
    pci_device_cfg_write_u8.argtypes = [ctypes.POINTER(struct_pci_device), uint8_t, pciaddr_t]
except AttributeError:
    pass
uint16_t = ctypes.c_uint16
try:
    pci_device_cfg_write_u16 = _libraries['libpciaccess.so'].pci_device_cfg_write_u16
    pci_device_cfg_write_u16.restype = ctypes.c_int32
    pci_device_cfg_write_u16.argtypes = [ctypes.POINTER(struct_pci_device), uint16_t, pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_cfg_write_u32 = _libraries['libpciaccess.so'].pci_device_cfg_write_u32
    pci_device_cfg_write_u32.restype = ctypes.c_int32
    pci_device_cfg_write_u32.argtypes = [ctypes.POINTER(struct_pci_device), uint32_t, pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_cfg_write_bits = _libraries['libpciaccess.so'].pci_device_cfg_write_bits
    pci_device_cfg_write_bits.restype = ctypes.c_int32
    pci_device_cfg_write_bits.argtypes = [ctypes.POINTER(struct_pci_device), uint32_t, uint32_t, pciaddr_t]
except AttributeError:
    pass
class struct_pci_mem_region(Structure):
    pass

struct_pci_mem_region._pack_ = 1 # source:False
struct_pci_mem_region._fields_ = [
    ('memory', ctypes.POINTER(None)),
    ('bus_addr', ctypes.c_uint64),
    ('base_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('is_IO', ctypes.c_uint32, 1),
    ('is_prefetchable', ctypes.c_uint32, 1),
    ('is_64', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint64, 61),
]

class struct_pci_pcmcia_bridge_info_0(Structure):
    pass

struct_pci_pcmcia_bridge_info_0._pack_ = 1 # source:False
struct_pci_pcmcia_bridge_info_0._fields_ = [
    ('base', ctypes.c_uint32),
    ('limit', ctypes.c_uint32),
]

class struct_pci_pcmcia_bridge_info_1(Structure):
    pass

struct_pci_pcmcia_bridge_info_1._pack_ = 1 # source:False
struct_pci_pcmcia_bridge_info_1._fields_ = [
    ('base', ctypes.c_uint32),
    ('limit', ctypes.c_uint32),
]

try:
    pci_device_vgaarb_init = _libraries['libpciaccess.so'].pci_device_vgaarb_init
    pci_device_vgaarb_init.restype = ctypes.c_int32
    pci_device_vgaarb_init.argtypes = []
except AttributeError:
    pass
try:
    pci_device_vgaarb_fini = _libraries['libpciaccess.so'].pci_device_vgaarb_fini
    pci_device_vgaarb_fini.restype = None
    pci_device_vgaarb_fini.argtypes = []
except AttributeError:
    pass
try:
    pci_device_vgaarb_set_target = _libraries['libpciaccess.so'].pci_device_vgaarb_set_target
    pci_device_vgaarb_set_target.restype = ctypes.c_int32
    pci_device_vgaarb_set_target.argtypes = [ctypes.POINTER(struct_pci_device)]
except AttributeError:
    pass
try:
    pci_device_vgaarb_decodes = _libraries['libpciaccess.so'].pci_device_vgaarb_decodes
    pci_device_vgaarb_decodes.restype = ctypes.c_int32
    pci_device_vgaarb_decodes.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    pci_device_vgaarb_lock = _libraries['libpciaccess.so'].pci_device_vgaarb_lock
    pci_device_vgaarb_lock.restype = ctypes.c_int32
    pci_device_vgaarb_lock.argtypes = []
except AttributeError:
    pass
try:
    pci_device_vgaarb_trylock = _libraries['libpciaccess.so'].pci_device_vgaarb_trylock
    pci_device_vgaarb_trylock.restype = ctypes.c_int32
    pci_device_vgaarb_trylock.argtypes = []
except AttributeError:
    pass
try:
    pci_device_vgaarb_unlock = _libraries['libpciaccess.so'].pci_device_vgaarb_unlock
    pci_device_vgaarb_unlock.restype = ctypes.c_int32
    pci_device_vgaarb_unlock.argtypes = []
except AttributeError:
    pass
try:
    pci_device_vgaarb_get_info = _libraries['libpciaccess.so'].pci_device_vgaarb_get_info
    pci_device_vgaarb_get_info.restype = ctypes.c_int32
    pci_device_vgaarb_get_info.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
class struct_pci_io_handle(Structure):
    pass

try:
    pci_device_open_io = _libraries['libpciaccess.so'].pci_device_open_io
    pci_device_open_io.restype = ctypes.POINTER(struct_pci_io_handle)
    pci_device_open_io.argtypes = [ctypes.POINTER(struct_pci_device), pciaddr_t, pciaddr_t]
except AttributeError:
    pass
try:
    pci_legacy_open_io = _libraries['libpciaccess.so'].pci_legacy_open_io
    pci_legacy_open_io.restype = ctypes.POINTER(struct_pci_io_handle)
    pci_legacy_open_io.argtypes = [ctypes.POINTER(struct_pci_device), pciaddr_t, pciaddr_t]
except AttributeError:
    pass
try:
    pci_device_close_io = _libraries['libpciaccess.so'].pci_device_close_io
    pci_device_close_io.restype = None
    pci_device_close_io.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(struct_pci_io_handle)]
except AttributeError:
    pass
try:
    pci_io_read32 = _libraries['libpciaccess.so'].pci_io_read32
    pci_io_read32.restype = uint32_t
    pci_io_read32.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t]
except AttributeError:
    pass
try:
    pci_io_read16 = _libraries['libpciaccess.so'].pci_io_read16
    pci_io_read16.restype = uint16_t
    pci_io_read16.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t]
except AttributeError:
    pass
try:
    pci_io_read8 = _libraries['libpciaccess.so'].pci_io_read8
    pci_io_read8.restype = uint8_t
    pci_io_read8.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t]
except AttributeError:
    pass
try:
    pci_io_write32 = _libraries['libpciaccess.so'].pci_io_write32
    pci_io_write32.restype = None
    pci_io_write32.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    pci_io_write16 = _libraries['libpciaccess.so'].pci_io_write16
    pci_io_write16.restype = None
    pci_io_write16.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t, uint16_t]
except AttributeError:
    pass
try:
    pci_io_write8 = _libraries['libpciaccess.so'].pci_io_write8
    pci_io_write8.restype = None
    pci_io_write8.argtypes = [ctypes.POINTER(struct_pci_io_handle), uint32_t, uint8_t]
except AttributeError:
    pass
try:
    pci_device_map_legacy = _libraries['libpciaccess.so'].pci_device_map_legacy
    pci_device_map_legacy.restype = ctypes.c_int32
    pci_device_map_legacy.argtypes = [ctypes.POINTER(struct_pci_device), pciaddr_t, pciaddr_t, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    pci_device_unmap_legacy = _libraries['libpciaccess.so'].pci_device_unmap_legacy
    pci_device_unmap_legacy.restype = ctypes.c_int32
    pci_device_unmap_legacy.argtypes = [ctypes.POINTER(struct_pci_device), ctypes.POINTER(None), pciaddr_t]
except AttributeError:
    pass
struct_pci_device._pack_ = 1 # source:False
struct_pci_device._fields_ = [
    ('domain_16', ctypes.c_uint16),
    ('bus', ctypes.c_ubyte),
    ('dev', ctypes.c_ubyte),
    ('func', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('vendor_id', ctypes.c_uint16),
    ('device_id', ctypes.c_uint16),
    ('subvendor_id', ctypes.c_uint16),
    ('subdevice_id', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('device_class', ctypes.c_uint32),
    ('revision', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 3),
    ('regions', struct_pci_mem_region * 6),
    ('rom_size', ctypes.c_uint64),
    ('irq', ctypes.c_int32),
    ('PADDING_3', ctypes.c_ubyte * 4),
    ('user_data', ctypes.c_int64),
    ('vgaarb_rsrc', ctypes.c_int32),
    ('domain', ctypes.c_uint32),
]

struct_pci_agp_info._pack_ = 1 # source:False
struct_pci_agp_info._fields_ = [
    ('config_offset', ctypes.c_uint32),
    ('major_version', ctypes.c_ubyte),
    ('minor_version', ctypes.c_ubyte),
    ('rates', ctypes.c_ubyte),
    ('fast_writes', ctypes.c_uint32, 1),
    ('addr64', ctypes.c_uint32, 1),
    ('htrans', ctypes.c_uint32, 1),
    ('gart64', ctypes.c_uint32, 1),
    ('coherent', ctypes.c_uint32, 1),
    ('sideband', ctypes.c_uint32, 1),
    ('isochronus', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint8, 1),
    ('async_req_size', ctypes.c_uint32, 8),
    ('calibration_cycle_timing', ctypes.c_ubyte),
    ('max_requests', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte),
]

struct_pci_bridge_info._pack_ = 1 # source:False
struct_pci_bridge_info._fields_ = [
    ('primary_bus', ctypes.c_ubyte),
    ('secondary_bus', ctypes.c_ubyte),
    ('subordinate_bus', ctypes.c_ubyte),
    ('secondary_latency_timer', ctypes.c_ubyte),
    ('io_type', ctypes.c_ubyte),
    ('mem_type', ctypes.c_ubyte),
    ('prefetch_mem_type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('secondary_status', ctypes.c_uint16),
    ('bridge_control', ctypes.c_uint16),
    ('io_base', ctypes.c_uint32),
    ('io_limit', ctypes.c_uint32),
    ('mem_base', ctypes.c_uint32),
    ('mem_limit', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('prefetch_mem_base', ctypes.c_uint64),
    ('prefetch_mem_limit', ctypes.c_uint64),
]

struct_pci_pcmcia_bridge_info._pack_ = 1 # source:False
struct_pci_pcmcia_bridge_info._fields_ = [
    ('primary_bus', ctypes.c_ubyte),
    ('card_bus', ctypes.c_ubyte),
    ('subordinate_bus', ctypes.c_ubyte),
    ('cardbus_latency_timer', ctypes.c_ubyte),
    ('secondary_status', ctypes.c_uint16),
    ('bridge_control', ctypes.c_uint16),
    ('io', struct_pci_pcmcia_bridge_info_0 * 2),
    ('mem', struct_pci_pcmcia_bridge_info_1 * 2),
]

struct_pci_slot_match._pack_ = 1 # source:False
struct_pci_slot_match._fields_ = [
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint32),
    ('dev', ctypes.c_uint32),
    ('func', ctypes.c_uint32),
    ('match_data', ctypes.c_int64),
]

struct_pci_id_match._pack_ = 1 # source:False
struct_pci_id_match._fields_ = [
    ('vendor_id', ctypes.c_uint32),
    ('device_id', ctypes.c_uint32),
    ('subvendor_id', ctypes.c_uint32),
    ('subdevice_id', ctypes.c_uint32),
    ('device_class', ctypes.c_uint32),
    ('device_class_mask', ctypes.c_uint32),
    ('match_data', ctypes.c_int64),
]

__all__ = \
    ['PCIACCESS_H', 'PCI_DEV_MAP_FLAG_CACHABLE',
    'PCI_DEV_MAP_FLAG_WRITABLE', 'PCI_DEV_MAP_FLAG_WRITE_COMBINE',
    'PCI_MATCH_ANY', 'VGA_ARB_RSRC_LEGACY_IO',
    'VGA_ARB_RSRC_LEGACY_MEM', 'VGA_ARB_RSRC_NONE',
    'VGA_ARB_RSRC_NORMAL_IO', 'VGA_ARB_RSRC_NORMAL_MEM',
    'pci_device_cfg_read', 'pci_device_cfg_read_u16',
    'pci_device_cfg_read_u32', 'pci_device_cfg_read_u8',
    'pci_device_cfg_write', 'pci_device_cfg_write_bits',
    'pci_device_cfg_write_u16', 'pci_device_cfg_write_u32',
    'pci_device_cfg_write_u8', 'pci_device_close_io',
    'pci_device_enable', 'pci_device_find_by_slot',
    'pci_device_get_agp_info', 'pci_device_get_bridge_buses',
    'pci_device_get_bridge_info', 'pci_device_get_device_name',
    'pci_device_get_parent_bridge',
    'pci_device_get_pcmcia_bridge_info',
    'pci_device_get_subdevice_name', 'pci_device_get_subvendor_name',
    'pci_device_get_vendor_name', 'pci_device_has_kernel_driver',
    'pci_device_is_boot_vga', 'pci_device_map_legacy',
    'pci_device_map_memory_range', 'pci_device_map_range',
    'pci_device_map_region', 'pci_device_next', 'pci_device_open_io',
    'pci_device_probe', 'pci_device_read_rom',
    'pci_device_unmap_legacy', 'pci_device_unmap_memory_range',
    'pci_device_unmap_range', 'pci_device_unmap_region',
    'pci_device_vgaarb_decodes', 'pci_device_vgaarb_fini',
    'pci_device_vgaarb_get_info', 'pci_device_vgaarb_init',
    'pci_device_vgaarb_lock', 'pci_device_vgaarb_set_target',
    'pci_device_vgaarb_trylock', 'pci_device_vgaarb_unlock',
    'pci_get_strings', 'pci_id_match_iterator_create',
    'pci_io_read16', 'pci_io_read32', 'pci_io_read8',
    'pci_io_write16', 'pci_io_write32', 'pci_io_write8',
    'pci_iterator_destroy', 'pci_legacy_open_io',
    'pci_slot_match_iterator_create', 'pci_system_cleanup',
    'pci_system_init', 'pci_system_init_dev_mem', 'pciaddr_t',
    'struct_pci_agp_info', 'struct_pci_bridge_info',
    'struct_pci_device', 'struct_pci_device_iterator',
    'struct_pci_id_match', 'struct_pci_io_handle',
    'struct_pci_mem_region', 'struct_pci_pcmcia_bridge_info',
    'struct_pci_pcmcia_bridge_info_0',
    'struct_pci_pcmcia_bridge_info_1', 'struct_pci_slot_match',
    'uint16_t', 'uint32_t', 'uint8_t']
