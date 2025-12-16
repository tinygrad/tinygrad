# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('libusb', 'usb-1.0')
enum_libusb_class_code = CEnum(ctypes.c_uint32)
LIBUSB_CLASS_PER_INTERFACE = enum_libusb_class_code.define('LIBUSB_CLASS_PER_INTERFACE', 0)
LIBUSB_CLASS_AUDIO = enum_libusb_class_code.define('LIBUSB_CLASS_AUDIO', 1)
LIBUSB_CLASS_COMM = enum_libusb_class_code.define('LIBUSB_CLASS_COMM', 2)
LIBUSB_CLASS_HID = enum_libusb_class_code.define('LIBUSB_CLASS_HID', 3)
LIBUSB_CLASS_PHYSICAL = enum_libusb_class_code.define('LIBUSB_CLASS_PHYSICAL', 5)
LIBUSB_CLASS_IMAGE = enum_libusb_class_code.define('LIBUSB_CLASS_IMAGE', 6)
LIBUSB_CLASS_PTP = enum_libusb_class_code.define('LIBUSB_CLASS_PTP', 6)
LIBUSB_CLASS_PRINTER = enum_libusb_class_code.define('LIBUSB_CLASS_PRINTER', 7)
LIBUSB_CLASS_MASS_STORAGE = enum_libusb_class_code.define('LIBUSB_CLASS_MASS_STORAGE', 8)
LIBUSB_CLASS_HUB = enum_libusb_class_code.define('LIBUSB_CLASS_HUB', 9)
LIBUSB_CLASS_DATA = enum_libusb_class_code.define('LIBUSB_CLASS_DATA', 10)
LIBUSB_CLASS_SMART_CARD = enum_libusb_class_code.define('LIBUSB_CLASS_SMART_CARD', 11)
LIBUSB_CLASS_CONTENT_SECURITY = enum_libusb_class_code.define('LIBUSB_CLASS_CONTENT_SECURITY', 13)
LIBUSB_CLASS_VIDEO = enum_libusb_class_code.define('LIBUSB_CLASS_VIDEO', 14)
LIBUSB_CLASS_PERSONAL_HEALTHCARE = enum_libusb_class_code.define('LIBUSB_CLASS_PERSONAL_HEALTHCARE', 15)
LIBUSB_CLASS_DIAGNOSTIC_DEVICE = enum_libusb_class_code.define('LIBUSB_CLASS_DIAGNOSTIC_DEVICE', 220)
LIBUSB_CLASS_WIRELESS = enum_libusb_class_code.define('LIBUSB_CLASS_WIRELESS', 224)
LIBUSB_CLASS_MISCELLANEOUS = enum_libusb_class_code.define('LIBUSB_CLASS_MISCELLANEOUS', 239)
LIBUSB_CLASS_APPLICATION = enum_libusb_class_code.define('LIBUSB_CLASS_APPLICATION', 254)
LIBUSB_CLASS_VENDOR_SPEC = enum_libusb_class_code.define('LIBUSB_CLASS_VENDOR_SPEC', 255)

enum_libusb_descriptor_type = CEnum(ctypes.c_uint32)
LIBUSB_DT_DEVICE = enum_libusb_descriptor_type.define('LIBUSB_DT_DEVICE', 1)
LIBUSB_DT_CONFIG = enum_libusb_descriptor_type.define('LIBUSB_DT_CONFIG', 2)
LIBUSB_DT_STRING = enum_libusb_descriptor_type.define('LIBUSB_DT_STRING', 3)
LIBUSB_DT_INTERFACE = enum_libusb_descriptor_type.define('LIBUSB_DT_INTERFACE', 4)
LIBUSB_DT_ENDPOINT = enum_libusb_descriptor_type.define('LIBUSB_DT_ENDPOINT', 5)
LIBUSB_DT_INTERFACE_ASSOCIATION = enum_libusb_descriptor_type.define('LIBUSB_DT_INTERFACE_ASSOCIATION', 11)
LIBUSB_DT_BOS = enum_libusb_descriptor_type.define('LIBUSB_DT_BOS', 15)
LIBUSB_DT_DEVICE_CAPABILITY = enum_libusb_descriptor_type.define('LIBUSB_DT_DEVICE_CAPABILITY', 16)
LIBUSB_DT_HID = enum_libusb_descriptor_type.define('LIBUSB_DT_HID', 33)
LIBUSB_DT_REPORT = enum_libusb_descriptor_type.define('LIBUSB_DT_REPORT', 34)
LIBUSB_DT_PHYSICAL = enum_libusb_descriptor_type.define('LIBUSB_DT_PHYSICAL', 35)
LIBUSB_DT_HUB = enum_libusb_descriptor_type.define('LIBUSB_DT_HUB', 41)
LIBUSB_DT_SUPERSPEED_HUB = enum_libusb_descriptor_type.define('LIBUSB_DT_SUPERSPEED_HUB', 42)
LIBUSB_DT_SS_ENDPOINT_COMPANION = enum_libusb_descriptor_type.define('LIBUSB_DT_SS_ENDPOINT_COMPANION', 48)

enum_libusb_endpoint_direction = CEnum(ctypes.c_uint32)
LIBUSB_ENDPOINT_OUT = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_OUT', 0)
LIBUSB_ENDPOINT_IN = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_IN', 128)

enum_libusb_endpoint_transfer_type = CEnum(ctypes.c_uint32)
LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK', 2)
LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT', 3)

enum_libusb_standard_request = CEnum(ctypes.c_uint32)
LIBUSB_REQUEST_GET_STATUS = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_STATUS', 0)
LIBUSB_REQUEST_CLEAR_FEATURE = enum_libusb_standard_request.define('LIBUSB_REQUEST_CLEAR_FEATURE', 1)
LIBUSB_REQUEST_SET_FEATURE = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_FEATURE', 3)
LIBUSB_REQUEST_SET_ADDRESS = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_ADDRESS', 5)
LIBUSB_REQUEST_GET_DESCRIPTOR = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_DESCRIPTOR', 6)
LIBUSB_REQUEST_SET_DESCRIPTOR = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_DESCRIPTOR', 7)
LIBUSB_REQUEST_GET_CONFIGURATION = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_CONFIGURATION', 8)
LIBUSB_REQUEST_SET_CONFIGURATION = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_CONFIGURATION', 9)
LIBUSB_REQUEST_GET_INTERFACE = enum_libusb_standard_request.define('LIBUSB_REQUEST_GET_INTERFACE', 10)
LIBUSB_REQUEST_SET_INTERFACE = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_INTERFACE', 11)
LIBUSB_REQUEST_SYNCH_FRAME = enum_libusb_standard_request.define('LIBUSB_REQUEST_SYNCH_FRAME', 12)
LIBUSB_REQUEST_SET_SEL = enum_libusb_standard_request.define('LIBUSB_REQUEST_SET_SEL', 48)
LIBUSB_SET_ISOCH_DELAY = enum_libusb_standard_request.define('LIBUSB_SET_ISOCH_DELAY', 49)

enum_libusb_request_type = CEnum(ctypes.c_uint32)
LIBUSB_REQUEST_TYPE_STANDARD = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_STANDARD', 0)
LIBUSB_REQUEST_TYPE_CLASS = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_CLASS', 32)
LIBUSB_REQUEST_TYPE_VENDOR = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_VENDOR', 64)
LIBUSB_REQUEST_TYPE_RESERVED = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_RESERVED', 96)

enum_libusb_request_recipient = CEnum(ctypes.c_uint32)
LIBUSB_RECIPIENT_DEVICE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_DEVICE', 0)
LIBUSB_RECIPIENT_INTERFACE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_INTERFACE', 1)
LIBUSB_RECIPIENT_ENDPOINT = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_ENDPOINT', 2)
LIBUSB_RECIPIENT_OTHER = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_OTHER', 3)

enum_libusb_iso_sync_type = CEnum(ctypes.c_uint32)
LIBUSB_ISO_SYNC_TYPE_NONE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_NONE', 0)
LIBUSB_ISO_SYNC_TYPE_ASYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ASYNC', 1)
LIBUSB_ISO_SYNC_TYPE_ADAPTIVE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ADAPTIVE', 2)
LIBUSB_ISO_SYNC_TYPE_SYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_SYNC', 3)

enum_libusb_iso_usage_type = CEnum(ctypes.c_uint32)
LIBUSB_ISO_USAGE_TYPE_DATA = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_DATA', 0)
LIBUSB_ISO_USAGE_TYPE_FEEDBACK = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_FEEDBACK', 1)
LIBUSB_ISO_USAGE_TYPE_IMPLICIT = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_IMPLICIT', 2)

enum_libusb_supported_speed = CEnum(ctypes.c_uint32)
LIBUSB_LOW_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_LOW_SPEED_OPERATION', 1)
LIBUSB_FULL_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_FULL_SPEED_OPERATION', 2)
LIBUSB_HIGH_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_HIGH_SPEED_OPERATION', 4)
LIBUSB_SUPER_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_SUPER_SPEED_OPERATION', 8)

enum_libusb_usb_2_0_extension_attributes = CEnum(ctypes.c_uint32)
LIBUSB_BM_LPM_SUPPORT = enum_libusb_usb_2_0_extension_attributes.define('LIBUSB_BM_LPM_SUPPORT', 2)

enum_libusb_ss_usb_device_capability_attributes = CEnum(ctypes.c_uint32)
LIBUSB_BM_LTM_SUPPORT = enum_libusb_ss_usb_device_capability_attributes.define('LIBUSB_BM_LTM_SUPPORT', 2)

enum_libusb_bos_type = CEnum(ctypes.c_uint32)
LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY', 1)
LIBUSB_BT_USB_2_0_EXTENSION = enum_libusb_bos_type.define('LIBUSB_BT_USB_2_0_EXTENSION', 2)
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_SS_USB_DEVICE_CAPABILITY', 3)
LIBUSB_BT_CONTAINER_ID = enum_libusb_bos_type.define('LIBUSB_BT_CONTAINER_ID', 4)
LIBUSB_BT_PLATFORM_DESCRIPTOR = enum_libusb_bos_type.define('LIBUSB_BT_PLATFORM_DESCRIPTOR', 5)

class struct_libusb_device_descriptor(Struct): pass
uint8_t = ctypes.c_ubyte
uint16_t = ctypes.c_uint16
struct_libusb_device_descriptor.SIZE = 18
struct_libusb_device_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bcdUSB', 'bDeviceClass', 'bDeviceSubClass', 'bDeviceProtocol', 'bMaxPacketSize0', 'idVendor', 'idProduct', 'bcdDevice', 'iManufacturer', 'iProduct', 'iSerialNumber', 'bNumConfigurations']
setattr(struct_libusb_device_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_device_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_device_descriptor, 'bcdUSB', field(2, uint16_t))
setattr(struct_libusb_device_descriptor, 'bDeviceClass', field(4, uint8_t))
setattr(struct_libusb_device_descriptor, 'bDeviceSubClass', field(5, uint8_t))
setattr(struct_libusb_device_descriptor, 'bDeviceProtocol', field(6, uint8_t))
setattr(struct_libusb_device_descriptor, 'bMaxPacketSize0', field(7, uint8_t))
setattr(struct_libusb_device_descriptor, 'idVendor', field(8, uint16_t))
setattr(struct_libusb_device_descriptor, 'idProduct', field(10, uint16_t))
setattr(struct_libusb_device_descriptor, 'bcdDevice', field(12, uint16_t))
setattr(struct_libusb_device_descriptor, 'iManufacturer', field(14, uint8_t))
setattr(struct_libusb_device_descriptor, 'iProduct', field(15, uint8_t))
setattr(struct_libusb_device_descriptor, 'iSerialNumber', field(16, uint8_t))
setattr(struct_libusb_device_descriptor, 'bNumConfigurations', field(17, uint8_t))
class struct_libusb_endpoint_descriptor(Struct): pass
struct_libusb_endpoint_descriptor.SIZE = 32
struct_libusb_endpoint_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bEndpointAddress', 'bmAttributes', 'wMaxPacketSize', 'bInterval', 'bRefresh', 'bSynchAddress', 'extra', 'extra_length']
setattr(struct_libusb_endpoint_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'bEndpointAddress', field(2, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'bmAttributes', field(3, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'wMaxPacketSize', field(4, uint16_t))
setattr(struct_libusb_endpoint_descriptor, 'bInterval', field(6, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'bRefresh', field(7, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'bSynchAddress', field(8, uint8_t))
setattr(struct_libusb_endpoint_descriptor, 'extra', field(16, Pointer(ctypes.c_ubyte)))
setattr(struct_libusb_endpoint_descriptor, 'extra_length', field(24, ctypes.c_int32))
class struct_libusb_interface_association_descriptor(Struct): pass
struct_libusb_interface_association_descriptor.SIZE = 8
struct_libusb_interface_association_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bFirstInterface', 'bInterfaceCount', 'bFunctionClass', 'bFunctionSubClass', 'bFunctionProtocol', 'iFunction']
setattr(struct_libusb_interface_association_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bFirstInterface', field(2, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bInterfaceCount', field(3, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bFunctionClass', field(4, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bFunctionSubClass', field(5, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'bFunctionProtocol', field(6, uint8_t))
setattr(struct_libusb_interface_association_descriptor, 'iFunction', field(7, uint8_t))
class struct_libusb_interface_association_descriptor_array(Struct): pass
struct_libusb_interface_association_descriptor_array.SIZE = 16
struct_libusb_interface_association_descriptor_array._fields_ = ['iad', 'length']
setattr(struct_libusb_interface_association_descriptor_array, 'iad', field(0, Pointer(struct_libusb_interface_association_descriptor)))
setattr(struct_libusb_interface_association_descriptor_array, 'length', field(8, ctypes.c_int32))
class struct_libusb_interface_descriptor(Struct): pass
struct_libusb_interface_descriptor.SIZE = 40
struct_libusb_interface_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bInterfaceNumber', 'bAlternateSetting', 'bNumEndpoints', 'bInterfaceClass', 'bInterfaceSubClass', 'bInterfaceProtocol', 'iInterface', 'endpoint', 'extra', 'extra_length']
setattr(struct_libusb_interface_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bInterfaceNumber', field(2, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bAlternateSetting', field(3, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bNumEndpoints', field(4, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bInterfaceClass', field(5, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bInterfaceSubClass', field(6, uint8_t))
setattr(struct_libusb_interface_descriptor, 'bInterfaceProtocol', field(7, uint8_t))
setattr(struct_libusb_interface_descriptor, 'iInterface', field(8, uint8_t))
setattr(struct_libusb_interface_descriptor, 'endpoint', field(16, Pointer(struct_libusb_endpoint_descriptor)))
setattr(struct_libusb_interface_descriptor, 'extra', field(24, Pointer(ctypes.c_ubyte)))
setattr(struct_libusb_interface_descriptor, 'extra_length', field(32, ctypes.c_int32))
class struct_libusb_interface(Struct): pass
struct_libusb_interface.SIZE = 16
struct_libusb_interface._fields_ = ['altsetting', 'num_altsetting']
setattr(struct_libusb_interface, 'altsetting', field(0, Pointer(struct_libusb_interface_descriptor)))
setattr(struct_libusb_interface, 'num_altsetting', field(8, ctypes.c_int32))
class struct_libusb_config_descriptor(Struct): pass
struct_libusb_config_descriptor.SIZE = 40
struct_libusb_config_descriptor._fields_ = ['bLength', 'bDescriptorType', 'wTotalLength', 'bNumInterfaces', 'bConfigurationValue', 'iConfiguration', 'bmAttributes', 'MaxPower', 'interface', 'extra', 'extra_length']
setattr(struct_libusb_config_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_config_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_config_descriptor, 'wTotalLength', field(2, uint16_t))
setattr(struct_libusb_config_descriptor, 'bNumInterfaces', field(4, uint8_t))
setattr(struct_libusb_config_descriptor, 'bConfigurationValue', field(5, uint8_t))
setattr(struct_libusb_config_descriptor, 'iConfiguration', field(6, uint8_t))
setattr(struct_libusb_config_descriptor, 'bmAttributes', field(7, uint8_t))
setattr(struct_libusb_config_descriptor, 'MaxPower', field(8, uint8_t))
setattr(struct_libusb_config_descriptor, 'interface', field(16, Pointer(struct_libusb_interface)))
setattr(struct_libusb_config_descriptor, 'extra', field(24, Pointer(ctypes.c_ubyte)))
setattr(struct_libusb_config_descriptor, 'extra_length', field(32, ctypes.c_int32))
class struct_libusb_ss_endpoint_companion_descriptor(Struct): pass
struct_libusb_ss_endpoint_companion_descriptor.SIZE = 6
struct_libusb_ss_endpoint_companion_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bMaxBurst', 'bmAttributes', 'wBytesPerInterval']
setattr(struct_libusb_ss_endpoint_companion_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_ss_endpoint_companion_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_ss_endpoint_companion_descriptor, 'bMaxBurst', field(2, uint8_t))
setattr(struct_libusb_ss_endpoint_companion_descriptor, 'bmAttributes', field(3, uint8_t))
setattr(struct_libusb_ss_endpoint_companion_descriptor, 'wBytesPerInterval', field(4, uint16_t))
class struct_libusb_bos_dev_capability_descriptor(Struct): pass
struct_libusb_bos_dev_capability_descriptor.SIZE = 3
struct_libusb_bos_dev_capability_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bDevCapabilityType', 'dev_capability_data']
setattr(struct_libusb_bos_dev_capability_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_bos_dev_capability_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_bos_dev_capability_descriptor, 'bDevCapabilityType', field(2, uint8_t))
setattr(struct_libusb_bos_dev_capability_descriptor, 'dev_capability_data', field(3, Array(uint8_t, 0)))
class struct_libusb_bos_descriptor(Struct): pass
struct_libusb_bos_descriptor.SIZE = 8
struct_libusb_bos_descriptor._fields_ = ['bLength', 'bDescriptorType', 'wTotalLength', 'bNumDeviceCaps', 'dev_capability']
setattr(struct_libusb_bos_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_bos_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_bos_descriptor, 'wTotalLength', field(2, uint16_t))
setattr(struct_libusb_bos_descriptor, 'bNumDeviceCaps', field(4, uint8_t))
setattr(struct_libusb_bos_descriptor, 'dev_capability', field(8, Array(Pointer(struct_libusb_bos_dev_capability_descriptor), 0)))
class struct_libusb_usb_2_0_extension_descriptor(Struct): pass
uint32_t = ctypes.c_uint32
struct_libusb_usb_2_0_extension_descriptor.SIZE = 8
struct_libusb_usb_2_0_extension_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bDevCapabilityType', 'bmAttributes']
setattr(struct_libusb_usb_2_0_extension_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_usb_2_0_extension_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_usb_2_0_extension_descriptor, 'bDevCapabilityType', field(2, uint8_t))
setattr(struct_libusb_usb_2_0_extension_descriptor, 'bmAttributes', field(4, uint32_t))
class struct_libusb_ss_usb_device_capability_descriptor(Struct): pass
struct_libusb_ss_usb_device_capability_descriptor.SIZE = 10
struct_libusb_ss_usb_device_capability_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bDevCapabilityType', 'bmAttributes', 'wSpeedSupported', 'bFunctionalitySupport', 'bU1DevExitLat', 'bU2DevExitLat']
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bDevCapabilityType', field(2, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bmAttributes', field(3, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'wSpeedSupported', field(4, uint16_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bFunctionalitySupport', field(6, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bU1DevExitLat', field(7, uint8_t))
setattr(struct_libusb_ss_usb_device_capability_descriptor, 'bU2DevExitLat', field(8, uint16_t))
class struct_libusb_container_id_descriptor(Struct): pass
struct_libusb_container_id_descriptor.SIZE = 20
struct_libusb_container_id_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bDevCapabilityType', 'bReserved', 'ContainerID']
setattr(struct_libusb_container_id_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_container_id_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_container_id_descriptor, 'bDevCapabilityType', field(2, uint8_t))
setattr(struct_libusb_container_id_descriptor, 'bReserved', field(3, uint8_t))
setattr(struct_libusb_container_id_descriptor, 'ContainerID', field(4, Array(uint8_t, 16)))
class struct_libusb_platform_descriptor(Struct): pass
struct_libusb_platform_descriptor.SIZE = 20
struct_libusb_platform_descriptor._fields_ = ['bLength', 'bDescriptorType', 'bDevCapabilityType', 'bReserved', 'PlatformCapabilityUUID', 'CapabilityData']
setattr(struct_libusb_platform_descriptor, 'bLength', field(0, uint8_t))
setattr(struct_libusb_platform_descriptor, 'bDescriptorType', field(1, uint8_t))
setattr(struct_libusb_platform_descriptor, 'bDevCapabilityType', field(2, uint8_t))
setattr(struct_libusb_platform_descriptor, 'bReserved', field(3, uint8_t))
setattr(struct_libusb_platform_descriptor, 'PlatformCapabilityUUID', field(4, Array(uint8_t, 16)))
setattr(struct_libusb_platform_descriptor, 'CapabilityData', field(20, Array(uint8_t, 0)))
class struct_libusb_control_setup(Struct): pass
struct_libusb_control_setup.SIZE = 8
struct_libusb_control_setup._fields_ = ['bmRequestType', 'bRequest', 'wValue', 'wIndex', 'wLength']
setattr(struct_libusb_control_setup, 'bmRequestType', field(0, uint8_t))
setattr(struct_libusb_control_setup, 'bRequest', field(1, uint8_t))
setattr(struct_libusb_control_setup, 'wValue', field(2, uint16_t))
setattr(struct_libusb_control_setup, 'wIndex', field(4, uint16_t))
setattr(struct_libusb_control_setup, 'wLength', field(6, uint16_t))
class struct_libusb_context(Struct): pass
class struct_libusb_device(Struct): pass
class struct_libusb_device_handle(Struct): pass
class struct_libusb_version(Struct): pass
struct_libusb_version.SIZE = 24
struct_libusb_version._fields_ = ['major', 'minor', 'micro', 'nano', 'rc', 'describe']
setattr(struct_libusb_version, 'major', field(0, uint16_t))
setattr(struct_libusb_version, 'minor', field(2, uint16_t))
setattr(struct_libusb_version, 'micro', field(4, uint16_t))
setattr(struct_libusb_version, 'nano', field(6, uint16_t))
setattr(struct_libusb_version, 'rc', field(8, Pointer(ctypes.c_char)))
setattr(struct_libusb_version, 'describe', field(16, Pointer(ctypes.c_char)))
libusb_context = struct_libusb_context
libusb_device = struct_libusb_device
libusb_device_handle = struct_libusb_device_handle
enum_libusb_speed = CEnum(ctypes.c_uint32)
LIBUSB_SPEED_UNKNOWN = enum_libusb_speed.define('LIBUSB_SPEED_UNKNOWN', 0)
LIBUSB_SPEED_LOW = enum_libusb_speed.define('LIBUSB_SPEED_LOW', 1)
LIBUSB_SPEED_FULL = enum_libusb_speed.define('LIBUSB_SPEED_FULL', 2)
LIBUSB_SPEED_HIGH = enum_libusb_speed.define('LIBUSB_SPEED_HIGH', 3)
LIBUSB_SPEED_SUPER = enum_libusb_speed.define('LIBUSB_SPEED_SUPER', 4)
LIBUSB_SPEED_SUPER_PLUS = enum_libusb_speed.define('LIBUSB_SPEED_SUPER_PLUS', 5)

enum_libusb_error = CEnum(ctypes.c_int32)
LIBUSB_SUCCESS = enum_libusb_error.define('LIBUSB_SUCCESS', 0)
LIBUSB_ERROR_IO = enum_libusb_error.define('LIBUSB_ERROR_IO', -1)
LIBUSB_ERROR_INVALID_PARAM = enum_libusb_error.define('LIBUSB_ERROR_INVALID_PARAM', -2)
LIBUSB_ERROR_ACCESS = enum_libusb_error.define('LIBUSB_ERROR_ACCESS', -3)
LIBUSB_ERROR_NO_DEVICE = enum_libusb_error.define('LIBUSB_ERROR_NO_DEVICE', -4)
LIBUSB_ERROR_NOT_FOUND = enum_libusb_error.define('LIBUSB_ERROR_NOT_FOUND', -5)
LIBUSB_ERROR_BUSY = enum_libusb_error.define('LIBUSB_ERROR_BUSY', -6)
LIBUSB_ERROR_TIMEOUT = enum_libusb_error.define('LIBUSB_ERROR_TIMEOUT', -7)
LIBUSB_ERROR_OVERFLOW = enum_libusb_error.define('LIBUSB_ERROR_OVERFLOW', -8)
LIBUSB_ERROR_PIPE = enum_libusb_error.define('LIBUSB_ERROR_PIPE', -9)
LIBUSB_ERROR_INTERRUPTED = enum_libusb_error.define('LIBUSB_ERROR_INTERRUPTED', -10)
LIBUSB_ERROR_NO_MEM = enum_libusb_error.define('LIBUSB_ERROR_NO_MEM', -11)
LIBUSB_ERROR_NOT_SUPPORTED = enum_libusb_error.define('LIBUSB_ERROR_NOT_SUPPORTED', -12)
LIBUSB_ERROR_OTHER = enum_libusb_error.define('LIBUSB_ERROR_OTHER', -99)

enum_libusb_transfer_type = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_TYPE_CONTROL = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_TRANSFER_TYPE_BULK = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK', 2)
LIBUSB_TRANSFER_TYPE_INTERRUPT = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_INTERRUPT', 3)
LIBUSB_TRANSFER_TYPE_BULK_STREAM = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK_STREAM', 4)

enum_libusb_transfer_status = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_COMPLETED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_COMPLETED', 0)
LIBUSB_TRANSFER_ERROR = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_ERROR', 1)
LIBUSB_TRANSFER_TIMED_OUT = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_TIMED_OUT', 2)
LIBUSB_TRANSFER_CANCELLED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_CANCELLED', 3)
LIBUSB_TRANSFER_STALL = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_STALL', 4)
LIBUSB_TRANSFER_NO_DEVICE = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_NO_DEVICE', 5)
LIBUSB_TRANSFER_OVERFLOW = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_OVERFLOW', 6)

enum_libusb_transfer_flags = CEnum(ctypes.c_uint32)
LIBUSB_TRANSFER_SHORT_NOT_OK = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_SHORT_NOT_OK', 1)
LIBUSB_TRANSFER_FREE_BUFFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_BUFFER', 2)
LIBUSB_TRANSFER_FREE_TRANSFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_TRANSFER', 4)
LIBUSB_TRANSFER_ADD_ZERO_PACKET = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_ADD_ZERO_PACKET', 8)

class struct_libusb_iso_packet_descriptor(Struct): pass
struct_libusb_iso_packet_descriptor.SIZE = 12
struct_libusb_iso_packet_descriptor._fields_ = ['length', 'actual_length', 'status']
setattr(struct_libusb_iso_packet_descriptor, 'length', field(0, ctypes.c_uint32))
setattr(struct_libusb_iso_packet_descriptor, 'actual_length', field(4, ctypes.c_uint32))
setattr(struct_libusb_iso_packet_descriptor, 'status', field(8, enum_libusb_transfer_status))
class struct_libusb_transfer(Struct): pass
libusb_transfer_cb_fn = ctypes.CFUNCTYPE(None, Pointer(struct_libusb_transfer))
struct_libusb_transfer.SIZE = 64
struct_libusb_transfer._fields_ = ['dev_handle', 'flags', 'endpoint', 'type', 'timeout', 'status', 'length', 'actual_length', 'callback', 'user_data', 'buffer', 'num_iso_packets', 'iso_packet_desc']
setattr(struct_libusb_transfer, 'dev_handle', field(0, Pointer(libusb_device_handle)))
setattr(struct_libusb_transfer, 'flags', field(8, uint8_t))
setattr(struct_libusb_transfer, 'endpoint', field(9, ctypes.c_ubyte))
setattr(struct_libusb_transfer, 'type', field(10, ctypes.c_ubyte))
setattr(struct_libusb_transfer, 'timeout', field(12, ctypes.c_uint32))
setattr(struct_libusb_transfer, 'status', field(16, enum_libusb_transfer_status))
setattr(struct_libusb_transfer, 'length', field(20, ctypes.c_int32))
setattr(struct_libusb_transfer, 'actual_length', field(24, ctypes.c_int32))
setattr(struct_libusb_transfer, 'callback', field(32, libusb_transfer_cb_fn))
setattr(struct_libusb_transfer, 'user_data', field(40, ctypes.c_void_p))
setattr(struct_libusb_transfer, 'buffer', field(48, Pointer(ctypes.c_ubyte)))
setattr(struct_libusb_transfer, 'num_iso_packets', field(56, ctypes.c_int32))
setattr(struct_libusb_transfer, 'iso_packet_desc', field(60, Array(struct_libusb_iso_packet_descriptor, 0)))
enum_libusb_capability = CEnum(ctypes.c_uint32)
LIBUSB_CAP_HAS_CAPABILITY = enum_libusb_capability.define('LIBUSB_CAP_HAS_CAPABILITY', 0)
LIBUSB_CAP_HAS_HOTPLUG = enum_libusb_capability.define('LIBUSB_CAP_HAS_HOTPLUG', 1)
LIBUSB_CAP_HAS_HID_ACCESS = enum_libusb_capability.define('LIBUSB_CAP_HAS_HID_ACCESS', 256)
LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER = enum_libusb_capability.define('LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER', 257)

enum_libusb_log_level = CEnum(ctypes.c_uint32)
LIBUSB_LOG_LEVEL_NONE = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_NONE', 0)
LIBUSB_LOG_LEVEL_ERROR = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_ERROR', 1)
LIBUSB_LOG_LEVEL_WARNING = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_WARNING', 2)
LIBUSB_LOG_LEVEL_INFO = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_INFO', 3)
LIBUSB_LOG_LEVEL_DEBUG = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_DEBUG', 4)

enum_libusb_log_cb_mode = CEnum(ctypes.c_uint32)
LIBUSB_LOG_CB_GLOBAL = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_GLOBAL', 1)
LIBUSB_LOG_CB_CONTEXT = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_CONTEXT', 2)

enum_libusb_option = CEnum(ctypes.c_uint32)
LIBUSB_OPTION_LOG_LEVEL = enum_libusb_option.define('LIBUSB_OPTION_LOG_LEVEL', 0)
LIBUSB_OPTION_USE_USBDK = enum_libusb_option.define('LIBUSB_OPTION_USE_USBDK', 1)
LIBUSB_OPTION_NO_DEVICE_DISCOVERY = enum_libusb_option.define('LIBUSB_OPTION_NO_DEVICE_DISCOVERY', 2)
LIBUSB_OPTION_LOG_CB = enum_libusb_option.define('LIBUSB_OPTION_LOG_CB', 3)
LIBUSB_OPTION_MAX = enum_libusb_option.define('LIBUSB_OPTION_MAX', 4)

libusb_log_cb = ctypes.CFUNCTYPE(None, Pointer(struct_libusb_context), enum_libusb_log_level, Pointer(ctypes.c_char))
class struct_libusb_init_option(Struct): pass
class _anonunion0(Union): pass
_anonunion0.SIZE = 8
_anonunion0._fields_ = ['ival', 'log_cbval']
setattr(_anonunion0, 'ival', field(0, ctypes.c_int32))
setattr(_anonunion0, 'log_cbval', field(0, libusb_log_cb))
struct_libusb_init_option.SIZE = 16
struct_libusb_init_option._fields_ = ['option', 'value']
setattr(struct_libusb_init_option, 'option', field(0, enum_libusb_option))
setattr(struct_libusb_init_option, 'value', field(8, _anonunion0))
@dll.bind((Pointer(Pointer(libusb_context)),), ctypes.c_int32)
def libusb_init(ctx): ...
@dll.bind((Pointer(Pointer(libusb_context)), Array(struct_libusb_init_option, 0), ctypes.c_int32,), ctypes.c_int32)
def libusb_init_context(ctx, options, num_options): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_exit(ctx): ...
@dll.bind((Pointer(libusb_context), ctypes.c_int32,), None)
def libusb_set_debug(ctx, level): ...
@dll.bind((Pointer(libusb_context), libusb_log_cb, ctypes.c_int32,), None)
def libusb_set_log_cb(ctx, cb, mode): ...
@dll.bind((), Pointer(struct_libusb_version))
def libusb_get_version(): ...
@dll.bind((uint32_t,), ctypes.c_int32)
def libusb_has_capability(capability): ...
@dll.bind((ctypes.c_int32,), Pointer(ctypes.c_char))
def libusb_error_name(errcode): ...
@dll.bind((Pointer(ctypes.c_char),), ctypes.c_int32)
def libusb_setlocale(locale): ...
@dll.bind((ctypes.c_int32,), Pointer(ctypes.c_char))
def libusb_strerror(errcode): ...
ssize_t = ctypes.c_int64
@dll.bind((Pointer(libusb_context), Pointer(Pointer(Pointer(libusb_device))),), ssize_t)
def libusb_get_device_list(ctx, list): ...
@dll.bind((Pointer(Pointer(libusb_device)), ctypes.c_int32,), None)
def libusb_free_device_list(list, unref_devices): ...
@dll.bind((Pointer(libusb_device),), Pointer(libusb_device))
def libusb_ref_device(dev): ...
@dll.bind((Pointer(libusb_device),), None)
def libusb_unref_device(dev): ...
@dll.bind((Pointer(libusb_device_handle), Pointer(ctypes.c_int32),), ctypes.c_int32)
def libusb_get_configuration(dev, config): ...
@dll.bind((Pointer(libusb_device), Pointer(struct_libusb_device_descriptor),), ctypes.c_int32)
def libusb_get_device_descriptor(dev, desc): ...
@dll.bind((Pointer(libusb_device), Pointer(Pointer(struct_libusb_config_descriptor)),), ctypes.c_int32)
def libusb_get_active_config_descriptor(dev, config): ...
@dll.bind((Pointer(libusb_device), uint8_t, Pointer(Pointer(struct_libusb_config_descriptor)),), ctypes.c_int32)
def libusb_get_config_descriptor(dev, config_index, config): ...
@dll.bind((Pointer(libusb_device), uint8_t, Pointer(Pointer(struct_libusb_config_descriptor)),), ctypes.c_int32)
def libusb_get_config_descriptor_by_value(dev, bConfigurationValue, config): ...
@dll.bind((Pointer(struct_libusb_config_descriptor),), None)
def libusb_free_config_descriptor(config): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_libusb_endpoint_descriptor), Pointer(Pointer(struct_libusb_ss_endpoint_companion_descriptor)),), ctypes.c_int32)
def libusb_get_ss_endpoint_companion_descriptor(ctx, endpoint, ep_comp): ...
@dll.bind((Pointer(struct_libusb_ss_endpoint_companion_descriptor),), None)
def libusb_free_ss_endpoint_companion_descriptor(ep_comp): ...
@dll.bind((Pointer(libusb_device_handle), Pointer(Pointer(struct_libusb_bos_descriptor)),), ctypes.c_int32)
def libusb_get_bos_descriptor(dev_handle, bos): ...
@dll.bind((Pointer(struct_libusb_bos_descriptor),), None)
def libusb_free_bos_descriptor(bos): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_libusb_bos_dev_capability_descriptor), Pointer(Pointer(struct_libusb_usb_2_0_extension_descriptor)),), ctypes.c_int32)
def libusb_get_usb_2_0_extension_descriptor(ctx, dev_cap, usb_2_0_extension): ...
@dll.bind((Pointer(struct_libusb_usb_2_0_extension_descriptor),), None)
def libusb_free_usb_2_0_extension_descriptor(usb_2_0_extension): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_libusb_bos_dev_capability_descriptor), Pointer(Pointer(struct_libusb_ss_usb_device_capability_descriptor)),), ctypes.c_int32)
def libusb_get_ss_usb_device_capability_descriptor(ctx, dev_cap, ss_usb_device_cap): ...
@dll.bind((Pointer(struct_libusb_ss_usb_device_capability_descriptor),), None)
def libusb_free_ss_usb_device_capability_descriptor(ss_usb_device_cap): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_libusb_bos_dev_capability_descriptor), Pointer(Pointer(struct_libusb_container_id_descriptor)),), ctypes.c_int32)
def libusb_get_container_id_descriptor(ctx, dev_cap, container_id): ...
@dll.bind((Pointer(struct_libusb_container_id_descriptor),), None)
def libusb_free_container_id_descriptor(container_id): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_libusb_bos_dev_capability_descriptor), Pointer(Pointer(struct_libusb_platform_descriptor)),), ctypes.c_int32)
def libusb_get_platform_descriptor(ctx, dev_cap, platform_descriptor): ...
@dll.bind((Pointer(struct_libusb_platform_descriptor),), None)
def libusb_free_platform_descriptor(platform_descriptor): ...
@dll.bind((Pointer(libusb_device),), uint8_t)
def libusb_get_bus_number(dev): ...
@dll.bind((Pointer(libusb_device),), uint8_t)
def libusb_get_port_number(dev): ...
@dll.bind((Pointer(libusb_device), Pointer(uint8_t), ctypes.c_int32,), ctypes.c_int32)
def libusb_get_port_numbers(dev, port_numbers, port_numbers_len): ...
@dll.bind((Pointer(libusb_context), Pointer(libusb_device), Pointer(uint8_t), uint8_t,), ctypes.c_int32)
def libusb_get_port_path(ctx, dev, path, path_length): ...
@dll.bind((Pointer(libusb_device),), Pointer(libusb_device))
def libusb_get_parent(dev): ...
@dll.bind((Pointer(libusb_device),), uint8_t)
def libusb_get_device_address(dev): ...
@dll.bind((Pointer(libusb_device),), ctypes.c_int32)
def libusb_get_device_speed(dev): ...
@dll.bind((Pointer(libusb_device), ctypes.c_ubyte,), ctypes.c_int32)
def libusb_get_max_packet_size(dev, endpoint): ...
@dll.bind((Pointer(libusb_device), ctypes.c_ubyte,), ctypes.c_int32)
def libusb_get_max_iso_packet_size(dev, endpoint): ...
@dll.bind((Pointer(libusb_device), ctypes.c_int32, ctypes.c_int32, ctypes.c_ubyte,), ctypes.c_int32)
def libusb_get_max_alt_packet_size(dev, interface_number, alternate_setting, endpoint): ...
@dll.bind((Pointer(libusb_device), uint8_t, Pointer(Pointer(struct_libusb_interface_association_descriptor_array)),), ctypes.c_int32)
def libusb_get_interface_association_descriptors(dev, config_index, iad_array): ...
@dll.bind((Pointer(libusb_device), Pointer(Pointer(struct_libusb_interface_association_descriptor_array)),), ctypes.c_int32)
def libusb_get_active_interface_association_descriptors(dev, iad_array): ...
@dll.bind((Pointer(struct_libusb_interface_association_descriptor_array),), None)
def libusb_free_interface_association_descriptors(iad_array): ...
intptr_t = ctypes.c_int64
@dll.bind((Pointer(libusb_context), intptr_t, Pointer(Pointer(libusb_device_handle)),), ctypes.c_int32)
def libusb_wrap_sys_device(ctx, sys_dev, dev_handle): ...
@dll.bind((Pointer(libusb_device), Pointer(Pointer(libusb_device_handle)),), ctypes.c_int32)
def libusb_open(dev, dev_handle): ...
@dll.bind((Pointer(libusb_device_handle),), None)
def libusb_close(dev_handle): ...
@dll.bind((Pointer(libusb_device_handle),), Pointer(libusb_device))
def libusb_get_device(dev_handle): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_set_configuration(dev_handle, configuration): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_claim_interface(dev_handle, interface_number): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_release_interface(dev_handle, interface_number): ...
@dll.bind((Pointer(libusb_context), uint16_t, uint16_t,), Pointer(libusb_device_handle))
def libusb_open_device_with_vid_pid(ctx, vendor_id, product_id): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32, ctypes.c_int32,), ctypes.c_int32)
def libusb_set_interface_alt_setting(dev_handle, interface_number, alternate_setting): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_ubyte,), ctypes.c_int32)
def libusb_clear_halt(dev_handle, endpoint): ...
@dll.bind((Pointer(libusb_device_handle),), ctypes.c_int32)
def libusb_reset_device(dev_handle): ...
@dll.bind((Pointer(libusb_device_handle), uint32_t, Pointer(ctypes.c_ubyte), ctypes.c_int32,), ctypes.c_int32)
def libusb_alloc_streams(dev_handle, num_streams, endpoints, num_endpoints): ...
@dll.bind((Pointer(libusb_device_handle), Pointer(ctypes.c_ubyte), ctypes.c_int32,), ctypes.c_int32)
def libusb_free_streams(dev_handle, endpoints, num_endpoints): ...
size_t = ctypes.c_uint64
@dll.bind((Pointer(libusb_device_handle), size_t,), Pointer(ctypes.c_ubyte))
def libusb_dev_mem_alloc(dev_handle, length): ...
@dll.bind((Pointer(libusb_device_handle), Pointer(ctypes.c_ubyte), size_t,), ctypes.c_int32)
def libusb_dev_mem_free(dev_handle, buffer, length): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_kernel_driver_active(dev_handle, interface_number): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_detach_kernel_driver(dev_handle, interface_number): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_attach_kernel_driver(dev_handle, interface_number): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_int32,), ctypes.c_int32)
def libusb_set_auto_detach_kernel_driver(dev_handle, enable): ...
@dll.bind((ctypes.c_int32,), Pointer(struct_libusb_transfer))
def libusb_alloc_transfer(iso_packets): ...
@dll.bind((Pointer(struct_libusb_transfer),), ctypes.c_int32)
def libusb_submit_transfer(transfer): ...
@dll.bind((Pointer(struct_libusb_transfer),), ctypes.c_int32)
def libusb_cancel_transfer(transfer): ...
@dll.bind((Pointer(struct_libusb_transfer),), None)
def libusb_free_transfer(transfer): ...
@dll.bind((Pointer(struct_libusb_transfer), uint32_t,), None)
def libusb_transfer_set_stream_id(transfer, stream_id): ...
@dll.bind((Pointer(struct_libusb_transfer),), uint32_t)
def libusb_transfer_get_stream_id(transfer): ...
@dll.bind((Pointer(libusb_device_handle), uint8_t, uint8_t, uint16_t, uint16_t, Pointer(ctypes.c_ubyte), uint16_t, ctypes.c_uint32,), ctypes.c_int32)
def libusb_control_transfer(dev_handle, request_type, bRequest, wValue, wIndex, data, wLength, timeout): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_ubyte, Pointer(ctypes.c_ubyte), ctypes.c_int32, Pointer(ctypes.c_int32), ctypes.c_uint32,), ctypes.c_int32)
def libusb_bulk_transfer(dev_handle, endpoint, data, length, actual_length, timeout): ...
@dll.bind((Pointer(libusb_device_handle), ctypes.c_ubyte, Pointer(ctypes.c_ubyte), ctypes.c_int32, Pointer(ctypes.c_int32), ctypes.c_uint32,), ctypes.c_int32)
def libusb_interrupt_transfer(dev_handle, endpoint, data, length, actual_length, timeout): ...
@dll.bind((Pointer(libusb_device_handle), uint8_t, Pointer(ctypes.c_ubyte), ctypes.c_int32,), ctypes.c_int32)
def libusb_get_string_descriptor_ascii(dev_handle, desc_index, data, length): ...
@dll.bind((Pointer(libusb_context),), ctypes.c_int32)
def libusb_try_lock_events(ctx): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_lock_events(ctx): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_unlock_events(ctx): ...
@dll.bind((Pointer(libusb_context),), ctypes.c_int32)
def libusb_event_handling_ok(ctx): ...
@dll.bind((Pointer(libusb_context),), ctypes.c_int32)
def libusb_event_handler_active(ctx): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_interrupt_event_handler(ctx): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_lock_event_waiters(ctx): ...
@dll.bind((Pointer(libusb_context),), None)
def libusb_unlock_event_waiters(ctx): ...
class struct_timeval(Struct): pass
__time_t = ctypes.c_int64
__suseconds_t = ctypes.c_int64
struct_timeval.SIZE = 16
struct_timeval._fields_ = ['tv_sec', 'tv_usec']
setattr(struct_timeval, 'tv_sec', field(0, ctypes.c_int64))
setattr(struct_timeval, 'tv_usec', field(8, ctypes.c_int64))
@dll.bind((Pointer(libusb_context), Pointer(struct_timeval),), ctypes.c_int32)
def libusb_wait_for_event(ctx, tv): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_timeval),), ctypes.c_int32)
def libusb_handle_events_timeout(ctx, tv): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_timeval), Pointer(ctypes.c_int32),), ctypes.c_int32)
def libusb_handle_events_timeout_completed(ctx, tv, completed): ...
@dll.bind((Pointer(libusb_context),), ctypes.c_int32)
def libusb_handle_events(ctx): ...
@dll.bind((Pointer(libusb_context), Pointer(ctypes.c_int32),), ctypes.c_int32)
def libusb_handle_events_completed(ctx, completed): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_timeval),), ctypes.c_int32)
def libusb_handle_events_locked(ctx, tv): ...
@dll.bind((Pointer(libusb_context),), ctypes.c_int32)
def libusb_pollfds_handle_timeouts(ctx): ...
@dll.bind((Pointer(libusb_context), Pointer(struct_timeval),), ctypes.c_int32)
def libusb_get_next_timeout(ctx, tv): ...
class struct_libusb_pollfd(Struct): pass
struct_libusb_pollfd.SIZE = 8
struct_libusb_pollfd._fields_ = ['fd', 'events']
setattr(struct_libusb_pollfd, 'fd', field(0, ctypes.c_int32))
setattr(struct_libusb_pollfd, 'events', field(4, ctypes.c_int16))
libusb_pollfd_added_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int16, ctypes.c_void_p)
libusb_pollfd_removed_cb = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_void_p)
@dll.bind((Pointer(libusb_context),), Pointer(Pointer(struct_libusb_pollfd)))
def libusb_get_pollfds(ctx): ...
@dll.bind((Pointer(Pointer(struct_libusb_pollfd)),), None)
def libusb_free_pollfds(pollfds): ...
@dll.bind((Pointer(libusb_context), libusb_pollfd_added_cb, libusb_pollfd_removed_cb, ctypes.c_void_p,), None)
def libusb_set_pollfd_notifiers(ctx, added_cb, removed_cb, user_data): ...
libusb_hotplug_callback_handle = ctypes.c_int32
libusb_hotplug_event = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED', 1)
LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT', 2)

libusb_hotplug_flag = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_ENUMERATE = libusb_hotplug_flag.define('LIBUSB_HOTPLUG_ENUMERATE', 1)

libusb_hotplug_callback_fn = ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_libusb_context), Pointer(struct_libusb_device), libusb_hotplug_event, ctypes.c_void_p)
@dll.bind((Pointer(libusb_context), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, libusb_hotplug_callback_fn, ctypes.c_void_p, Pointer(libusb_hotplug_callback_handle),), ctypes.c_int32)
def libusb_hotplug_register_callback(ctx, events, flags, vendor_id, product_id, dev_class, cb_fn, user_data, callback_handle): ...
@dll.bind((Pointer(libusb_context), libusb_hotplug_callback_handle,), None)
def libusb_hotplug_deregister_callback(ctx, callback_handle): ...
@dll.bind((Pointer(libusb_context), libusb_hotplug_callback_handle,), ctypes.c_void_p)
def libusb_hotplug_get_user_data(ctx, callback_handle): ...
@dll.bind((Pointer(libusb_context), enum_libusb_option,), ctypes.c_int32)
def libusb_set_option(ctx, option): ...
LIBUSB_DEPRECATED_FOR = lambda f: __attribute__ ((deprecated))
LIBUSB_API_VERSION = 0x0100010A
LIBUSBX_API_VERSION = LIBUSB_API_VERSION
LIBUSB_DT_DEVICE_SIZE = 18
LIBUSB_DT_CONFIG_SIZE = 9
LIBUSB_DT_INTERFACE_SIZE = 9
LIBUSB_DT_ENDPOINT_SIZE = 7
LIBUSB_DT_ENDPOINT_AUDIO_SIZE = 9
LIBUSB_DT_HUB_NONVAR_SIZE = 7
LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE = 6
LIBUSB_DT_BOS_SIZE = 5
LIBUSB_DT_DEVICE_CAPABILITY_SIZE = 3
LIBUSB_BT_USB_2_0_EXTENSION_SIZE = 7
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE = 10
LIBUSB_BT_CONTAINER_ID_SIZE = 20
LIBUSB_BT_PLATFORM_DESCRIPTOR_MIN_SIZE = 20
LIBUSB_DT_BOS_MAX_SIZE = (LIBUSB_DT_BOS_SIZE + LIBUSB_BT_USB_2_0_EXTENSION_SIZE + LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE + LIBUSB_BT_CONTAINER_ID_SIZE)
LIBUSB_ENDPOINT_ADDRESS_MASK = 0x0f
LIBUSB_ENDPOINT_DIR_MASK = 0x80
LIBUSB_TRANSFER_TYPE_MASK = 0x03
LIBUSB_ISO_SYNC_TYPE_MASK = 0x0c
LIBUSB_ISO_USAGE_TYPE_MASK = 0x30
LIBUSB_ERROR_COUNT = 14
LIBUSB_OPTION_WEAK_AUTHORITY = LIBUSB_OPTION_NO_DEVICE_DISCOVERY
LIBUSB_HOTPLUG_NO_FLAGS = 0
LIBUSB_HOTPLUG_MATCH_ANY = -1