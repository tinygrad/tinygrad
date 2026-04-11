# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('libusb', 'usb-1.0')
class enum_libusb_class_code(ctypes.c_uint32, c.Enum): pass
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

class enum_libusb_descriptor_type(ctypes.c_uint32, c.Enum): pass
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

class enum_libusb_endpoint_direction(ctypes.c_uint32, c.Enum): pass
LIBUSB_ENDPOINT_OUT = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_OUT', 0)
LIBUSB_ENDPOINT_IN = enum_libusb_endpoint_direction.define('LIBUSB_ENDPOINT_IN', 128)

class enum_libusb_endpoint_transfer_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK', 2)
LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT = enum_libusb_endpoint_transfer_type.define('LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT', 3)

class enum_libusb_standard_request(ctypes.c_uint32, c.Enum): pass
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

class enum_libusb_request_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_REQUEST_TYPE_STANDARD = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_STANDARD', 0)
LIBUSB_REQUEST_TYPE_CLASS = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_CLASS', 32)
LIBUSB_REQUEST_TYPE_VENDOR = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_VENDOR', 64)
LIBUSB_REQUEST_TYPE_RESERVED = enum_libusb_request_type.define('LIBUSB_REQUEST_TYPE_RESERVED', 96)

class enum_libusb_request_recipient(ctypes.c_uint32, c.Enum): pass
LIBUSB_RECIPIENT_DEVICE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_DEVICE', 0)
LIBUSB_RECIPIENT_INTERFACE = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_INTERFACE', 1)
LIBUSB_RECIPIENT_ENDPOINT = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_ENDPOINT', 2)
LIBUSB_RECIPIENT_OTHER = enum_libusb_request_recipient.define('LIBUSB_RECIPIENT_OTHER', 3)

class enum_libusb_iso_sync_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_ISO_SYNC_TYPE_NONE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_NONE', 0)
LIBUSB_ISO_SYNC_TYPE_ASYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ASYNC', 1)
LIBUSB_ISO_SYNC_TYPE_ADAPTIVE = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_ADAPTIVE', 2)
LIBUSB_ISO_SYNC_TYPE_SYNC = enum_libusb_iso_sync_type.define('LIBUSB_ISO_SYNC_TYPE_SYNC', 3)

class enum_libusb_iso_usage_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_ISO_USAGE_TYPE_DATA = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_DATA', 0)
LIBUSB_ISO_USAGE_TYPE_FEEDBACK = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_FEEDBACK', 1)
LIBUSB_ISO_USAGE_TYPE_IMPLICIT = enum_libusb_iso_usage_type.define('LIBUSB_ISO_USAGE_TYPE_IMPLICIT', 2)

class enum_libusb_supported_speed(ctypes.c_uint32, c.Enum): pass
LIBUSB_LOW_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_LOW_SPEED_OPERATION', 1)
LIBUSB_FULL_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_FULL_SPEED_OPERATION', 2)
LIBUSB_HIGH_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_HIGH_SPEED_OPERATION', 4)
LIBUSB_SUPER_SPEED_OPERATION = enum_libusb_supported_speed.define('LIBUSB_SUPER_SPEED_OPERATION', 8)

class enum_libusb_usb_2_0_extension_attributes(ctypes.c_uint32, c.Enum): pass
LIBUSB_BM_LPM_SUPPORT = enum_libusb_usb_2_0_extension_attributes.define('LIBUSB_BM_LPM_SUPPORT', 2)

class enum_libusb_ss_usb_device_capability_attributes(ctypes.c_uint32, c.Enum): pass
LIBUSB_BM_LTM_SUPPORT = enum_libusb_ss_usb_device_capability_attributes.define('LIBUSB_BM_LTM_SUPPORT', 2)

class enum_libusb_bos_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY', 1)
LIBUSB_BT_USB_2_0_EXTENSION = enum_libusb_bos_type.define('LIBUSB_BT_USB_2_0_EXTENSION', 2)
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY = enum_libusb_bos_type.define('LIBUSB_BT_SS_USB_DEVICE_CAPABILITY', 3)
LIBUSB_BT_CONTAINER_ID = enum_libusb_bos_type.define('LIBUSB_BT_CONTAINER_ID', 4)
LIBUSB_BT_PLATFORM_DESCRIPTOR = enum_libusb_bos_type.define('LIBUSB_BT_PLATFORM_DESCRIPTOR', 5)

@c.record
class struct_libusb_device_descriptor(c.Struct):
  SIZE = 18
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bcdUSB: 'uint16_t'
  bDeviceClass: 'uint8_t'
  bDeviceSubClass: 'uint8_t'
  bDeviceProtocol: 'uint8_t'
  bMaxPacketSize0: 'uint8_t'
  idVendor: 'uint16_t'
  idProduct: 'uint16_t'
  bcdDevice: 'uint16_t'
  iManufacturer: 'uint8_t'
  iProduct: 'uint8_t'
  iSerialNumber: 'uint8_t'
  bNumConfigurations: 'uint8_t'
uint8_t: TypeAlias = ctypes.c_ubyte
uint16_t: TypeAlias = ctypes.c_uint16
struct_libusb_device_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bcdUSB', uint16_t, 2), ('bDeviceClass', uint8_t, 4), ('bDeviceSubClass', uint8_t, 5), ('bDeviceProtocol', uint8_t, 6), ('bMaxPacketSize0', uint8_t, 7), ('idVendor', uint16_t, 8), ('idProduct', uint16_t, 10), ('bcdDevice', uint16_t, 12), ('iManufacturer', uint8_t, 14), ('iProduct', uint8_t, 15), ('iSerialNumber', uint8_t, 16), ('bNumConfigurations', uint8_t, 17)])
@c.record
class struct_libusb_endpoint_descriptor(c.Struct):
  SIZE = 32
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bEndpointAddress: 'uint8_t'
  bmAttributes: 'uint8_t'
  wMaxPacketSize: 'uint16_t'
  bInterval: 'uint8_t'
  bRefresh: 'uint8_t'
  bSynchAddress: 'uint8_t'
  extra: 'c.POINTER[ctypes.c_ubyte]'
  extra_length: 'ctypes.c_int32'
struct_libusb_endpoint_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bEndpointAddress', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wMaxPacketSize', uint16_t, 4), ('bInterval', uint8_t, 6), ('bRefresh', uint8_t, 7), ('bSynchAddress', uint8_t, 8), ('extra', c.POINTER[ctypes.c_ubyte], 16), ('extra_length', ctypes.c_int32, 24)])
@c.record
class struct_libusb_interface_association_descriptor(c.Struct):
  SIZE = 8
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bFirstInterface: 'uint8_t'
  bInterfaceCount: 'uint8_t'
  bFunctionClass: 'uint8_t'
  bFunctionSubClass: 'uint8_t'
  bFunctionProtocol: 'uint8_t'
  iFunction: 'uint8_t'
struct_libusb_interface_association_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bFirstInterface', uint8_t, 2), ('bInterfaceCount', uint8_t, 3), ('bFunctionClass', uint8_t, 4), ('bFunctionSubClass', uint8_t, 5), ('bFunctionProtocol', uint8_t, 6), ('iFunction', uint8_t, 7)])
@c.record
class struct_libusb_interface_association_descriptor_array(c.Struct):
  SIZE = 16
  iad: 'c.POINTER[struct_libusb_interface_association_descriptor]'
  length: 'ctypes.c_int32'
struct_libusb_interface_association_descriptor_array.register_fields([('iad', c.POINTER[struct_libusb_interface_association_descriptor], 0), ('length', ctypes.c_int32, 8)])
@c.record
class struct_libusb_interface_descriptor(c.Struct):
  SIZE = 40
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bInterfaceNumber: 'uint8_t'
  bAlternateSetting: 'uint8_t'
  bNumEndpoints: 'uint8_t'
  bInterfaceClass: 'uint8_t'
  bInterfaceSubClass: 'uint8_t'
  bInterfaceProtocol: 'uint8_t'
  iInterface: 'uint8_t'
  endpoint: 'c.POINTER[struct_libusb_endpoint_descriptor]'
  extra: 'c.POINTER[ctypes.c_ubyte]'
  extra_length: 'ctypes.c_int32'
struct_libusb_interface_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bInterfaceNumber', uint8_t, 2), ('bAlternateSetting', uint8_t, 3), ('bNumEndpoints', uint8_t, 4), ('bInterfaceClass', uint8_t, 5), ('bInterfaceSubClass', uint8_t, 6), ('bInterfaceProtocol', uint8_t, 7), ('iInterface', uint8_t, 8), ('endpoint', c.POINTER[struct_libusb_endpoint_descriptor], 16), ('extra', c.POINTER[ctypes.c_ubyte], 24), ('extra_length', ctypes.c_int32, 32)])
@c.record
class struct_libusb_interface(c.Struct):
  SIZE = 16
  altsetting: 'c.POINTER[struct_libusb_interface_descriptor]'
  num_altsetting: 'ctypes.c_int32'
struct_libusb_interface.register_fields([('altsetting', c.POINTER[struct_libusb_interface_descriptor], 0), ('num_altsetting', ctypes.c_int32, 8)])
@c.record
class struct_libusb_config_descriptor(c.Struct):
  SIZE = 40
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  wTotalLength: 'uint16_t'
  bNumInterfaces: 'uint8_t'
  bConfigurationValue: 'uint8_t'
  iConfiguration: 'uint8_t'
  bmAttributes: 'uint8_t'
  MaxPower: 'uint8_t'
  interface: 'c.POINTER[struct_libusb_interface]'
  extra: 'c.POINTER[ctypes.c_ubyte]'
  extra_length: 'ctypes.c_int32'
struct_libusb_config_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('wTotalLength', uint16_t, 2), ('bNumInterfaces', uint8_t, 4), ('bConfigurationValue', uint8_t, 5), ('iConfiguration', uint8_t, 6), ('bmAttributes', uint8_t, 7), ('MaxPower', uint8_t, 8), ('interface', c.POINTER[struct_libusb_interface], 16), ('extra', c.POINTER[ctypes.c_ubyte], 24), ('extra_length', ctypes.c_int32, 32)])
@c.record
class struct_libusb_ss_endpoint_companion_descriptor(c.Struct):
  SIZE = 6
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bMaxBurst: 'uint8_t'
  bmAttributes: 'uint8_t'
  wBytesPerInterval: 'uint16_t'
struct_libusb_ss_endpoint_companion_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bMaxBurst', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wBytesPerInterval', uint16_t, 4)])
@c.record
class struct_libusb_bos_dev_capability_descriptor(c.Struct):
  SIZE = 3
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bDevCapabilityType: 'uint8_t'
  dev_capability_data: 'c.Array[uint8_t, Literal[0]]'
struct_libusb_bos_dev_capability_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('dev_capability_data', c.Array[uint8_t, Literal[0]], 3)])
@c.record
class struct_libusb_bos_descriptor(c.Struct):
  SIZE = 8
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  wTotalLength: 'uint16_t'
  bNumDeviceCaps: 'uint8_t'
  dev_capability: 'c.Array[c.POINTER[struct_libusb_bos_dev_capability_descriptor], Literal[0]]'
struct_libusb_bos_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('wTotalLength', uint16_t, 2), ('bNumDeviceCaps', uint8_t, 4), ('dev_capability', c.Array[c.POINTER[struct_libusb_bos_dev_capability_descriptor], Literal[0]], 8)])
@c.record
class struct_libusb_usb_2_0_extension_descriptor(c.Struct):
  SIZE = 8
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bDevCapabilityType: 'uint8_t'
  bmAttributes: 'uint32_t'
uint32_t: TypeAlias = ctypes.c_uint32
struct_libusb_usb_2_0_extension_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bmAttributes', uint32_t, 4)])
@c.record
class struct_libusb_ss_usb_device_capability_descriptor(c.Struct):
  SIZE = 10
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bDevCapabilityType: 'uint8_t'
  bmAttributes: 'uint8_t'
  wSpeedSupported: 'uint16_t'
  bFunctionalitySupport: 'uint8_t'
  bU1DevExitLat: 'uint8_t'
  bU2DevExitLat: 'uint16_t'
struct_libusb_ss_usb_device_capability_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wSpeedSupported', uint16_t, 4), ('bFunctionalitySupport', uint8_t, 6), ('bU1DevExitLat', uint8_t, 7), ('bU2DevExitLat', uint16_t, 8)])
@c.record
class struct_libusb_container_id_descriptor(c.Struct):
  SIZE = 20
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bDevCapabilityType: 'uint8_t'
  bReserved: 'uint8_t'
  ContainerID: 'c.Array[uint8_t, Literal[16]]'
struct_libusb_container_id_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bReserved', uint8_t, 3), ('ContainerID', c.Array[uint8_t, Literal[16]], 4)])
@c.record
class struct_libusb_platform_descriptor(c.Struct):
  SIZE = 20
  bLength: 'uint8_t'
  bDescriptorType: 'uint8_t'
  bDevCapabilityType: 'uint8_t'
  bReserved: 'uint8_t'
  PlatformCapabilityUUID: 'c.Array[uint8_t, Literal[16]]'
  CapabilityData: 'c.Array[uint8_t, Literal[0]]'
struct_libusb_platform_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bReserved', uint8_t, 3), ('PlatformCapabilityUUID', c.Array[uint8_t, Literal[16]], 4), ('CapabilityData', c.Array[uint8_t, Literal[0]], 20)])
@c.record
class struct_libusb_control_setup(c.Struct):
  SIZE = 8
  bmRequestType: 'uint8_t'
  bRequest: 'uint8_t'
  wValue: 'uint16_t'
  wIndex: 'uint16_t'
  wLength: 'uint16_t'
struct_libusb_control_setup.register_fields([('bmRequestType', uint8_t, 0), ('bRequest', uint8_t, 1), ('wValue', uint16_t, 2), ('wIndex', uint16_t, 4), ('wLength', uint16_t, 6)])
class struct_libusb_context(c.Struct): pass
class struct_libusb_device(c.Struct): pass
class struct_libusb_device_handle(c.Struct): pass
@c.record
class struct_libusb_version(c.Struct):
  SIZE = 24
  major: 'uint16_t'
  minor: 'uint16_t'
  micro: 'uint16_t'
  nano: 'uint16_t'
  rc: 'c.POINTER[ctypes.c_char]'
  describe: 'c.POINTER[ctypes.c_char]'
struct_libusb_version.register_fields([('major', uint16_t, 0), ('minor', uint16_t, 2), ('micro', uint16_t, 4), ('nano', uint16_t, 6), ('rc', c.POINTER[ctypes.c_char], 8), ('describe', c.POINTER[ctypes.c_char], 16)])
libusb_context: TypeAlias = struct_libusb_context
libusb_device: TypeAlias = struct_libusb_device
libusb_device_handle: TypeAlias = struct_libusb_device_handle
class enum_libusb_speed(ctypes.c_uint32, c.Enum): pass
LIBUSB_SPEED_UNKNOWN = enum_libusb_speed.define('LIBUSB_SPEED_UNKNOWN', 0)
LIBUSB_SPEED_LOW = enum_libusb_speed.define('LIBUSB_SPEED_LOW', 1)
LIBUSB_SPEED_FULL = enum_libusb_speed.define('LIBUSB_SPEED_FULL', 2)
LIBUSB_SPEED_HIGH = enum_libusb_speed.define('LIBUSB_SPEED_HIGH', 3)
LIBUSB_SPEED_SUPER = enum_libusb_speed.define('LIBUSB_SPEED_SUPER', 4)
LIBUSB_SPEED_SUPER_PLUS = enum_libusb_speed.define('LIBUSB_SPEED_SUPER_PLUS', 5)

class enum_libusb_error(ctypes.c_int32, c.Enum): pass
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

class enum_libusb_transfer_type(ctypes.c_uint32, c.Enum): pass
LIBUSB_TRANSFER_TYPE_CONTROL = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_CONTROL', 0)
LIBUSB_TRANSFER_TYPE_ISOCHRONOUS = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_ISOCHRONOUS', 1)
LIBUSB_TRANSFER_TYPE_BULK = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK', 2)
LIBUSB_TRANSFER_TYPE_INTERRUPT = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_INTERRUPT', 3)
LIBUSB_TRANSFER_TYPE_BULK_STREAM = enum_libusb_transfer_type.define('LIBUSB_TRANSFER_TYPE_BULK_STREAM', 4)

class enum_libusb_transfer_status(ctypes.c_uint32, c.Enum): pass
LIBUSB_TRANSFER_COMPLETED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_COMPLETED', 0)
LIBUSB_TRANSFER_ERROR = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_ERROR', 1)
LIBUSB_TRANSFER_TIMED_OUT = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_TIMED_OUT', 2)
LIBUSB_TRANSFER_CANCELLED = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_CANCELLED', 3)
LIBUSB_TRANSFER_STALL = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_STALL', 4)
LIBUSB_TRANSFER_NO_DEVICE = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_NO_DEVICE', 5)
LIBUSB_TRANSFER_OVERFLOW = enum_libusb_transfer_status.define('LIBUSB_TRANSFER_OVERFLOW', 6)

class enum_libusb_transfer_flags(ctypes.c_uint32, c.Enum): pass
LIBUSB_TRANSFER_SHORT_NOT_OK = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_SHORT_NOT_OK', 1)
LIBUSB_TRANSFER_FREE_BUFFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_BUFFER', 2)
LIBUSB_TRANSFER_FREE_TRANSFER = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_FREE_TRANSFER', 4)
LIBUSB_TRANSFER_ADD_ZERO_PACKET = enum_libusb_transfer_flags.define('LIBUSB_TRANSFER_ADD_ZERO_PACKET', 8)

@c.record
class struct_libusb_iso_packet_descriptor(c.Struct):
  SIZE = 12
  length: 'ctypes.c_uint32'
  actual_length: 'ctypes.c_uint32'
  status: 'enum_libusb_transfer_status'
struct_libusb_iso_packet_descriptor.register_fields([('length', ctypes.c_uint32, 0), ('actual_length', ctypes.c_uint32, 4), ('status', enum_libusb_transfer_status, 8)])
@c.record
class struct_libusb_transfer(c.Struct):
  SIZE = 64
  dev_handle: 'c.POINTER[libusb_device_handle]'
  flags: 'uint8_t'
  endpoint: 'ctypes.c_ubyte'
  type: 'ctypes.c_ubyte'
  timeout: 'ctypes.c_uint32'
  status: 'enum_libusb_transfer_status'
  length: 'ctypes.c_int32'
  actual_length: 'ctypes.c_int32'
  callback: 'libusb_transfer_cb_fn'
  user_data: 'ctypes.c_void_p'
  buffer: 'c.POINTER[ctypes.c_ubyte]'
  num_iso_packets: 'ctypes.c_int32'
  iso_packet_desc: 'c.Array[struct_libusb_iso_packet_descriptor, Literal[0]]'
libusb_transfer_cb_fn: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_libusb_transfer]]]
struct_libusb_transfer.register_fields([('dev_handle', c.POINTER[libusb_device_handle], 0), ('flags', uint8_t, 8), ('endpoint', ctypes.c_ubyte, 9), ('type', ctypes.c_ubyte, 10), ('timeout', ctypes.c_uint32, 12), ('status', enum_libusb_transfer_status, 16), ('length', ctypes.c_int32, 20), ('actual_length', ctypes.c_int32, 24), ('callback', libusb_transfer_cb_fn, 32), ('user_data', ctypes.c_void_p, 40), ('buffer', c.POINTER[ctypes.c_ubyte], 48), ('num_iso_packets', ctypes.c_int32, 56), ('iso_packet_desc', c.Array[struct_libusb_iso_packet_descriptor, Literal[0]], 60)])
class enum_libusb_capability(ctypes.c_uint32, c.Enum): pass
LIBUSB_CAP_HAS_CAPABILITY = enum_libusb_capability.define('LIBUSB_CAP_HAS_CAPABILITY', 0)
LIBUSB_CAP_HAS_HOTPLUG = enum_libusb_capability.define('LIBUSB_CAP_HAS_HOTPLUG', 1)
LIBUSB_CAP_HAS_HID_ACCESS = enum_libusb_capability.define('LIBUSB_CAP_HAS_HID_ACCESS', 256)
LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER = enum_libusb_capability.define('LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER', 257)

class enum_libusb_log_level(ctypes.c_uint32, c.Enum): pass
LIBUSB_LOG_LEVEL_NONE = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_NONE', 0)
LIBUSB_LOG_LEVEL_ERROR = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_ERROR', 1)
LIBUSB_LOG_LEVEL_WARNING = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_WARNING', 2)
LIBUSB_LOG_LEVEL_INFO = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_INFO', 3)
LIBUSB_LOG_LEVEL_DEBUG = enum_libusb_log_level.define('LIBUSB_LOG_LEVEL_DEBUG', 4)

class enum_libusb_log_cb_mode(ctypes.c_uint32, c.Enum): pass
LIBUSB_LOG_CB_GLOBAL = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_GLOBAL', 1)
LIBUSB_LOG_CB_CONTEXT = enum_libusb_log_cb_mode.define('LIBUSB_LOG_CB_CONTEXT', 2)

class enum_libusb_option(ctypes.c_uint32, c.Enum): pass
LIBUSB_OPTION_LOG_LEVEL = enum_libusb_option.define('LIBUSB_OPTION_LOG_LEVEL', 0)
LIBUSB_OPTION_USE_USBDK = enum_libusb_option.define('LIBUSB_OPTION_USE_USBDK', 1)
LIBUSB_OPTION_NO_DEVICE_DISCOVERY = enum_libusb_option.define('LIBUSB_OPTION_NO_DEVICE_DISCOVERY', 2)
LIBUSB_OPTION_LOG_CB = enum_libusb_option.define('LIBUSB_OPTION_LOG_CB', 3)
LIBUSB_OPTION_MAX = enum_libusb_option.define('LIBUSB_OPTION_MAX', 4)

libusb_log_cb: TypeAlias = c.CFUNCTYPE[None, [c.POINTER[struct_libusb_context], enum_libusb_log_level, c.POINTER[ctypes.c_char]]]
@c.record
class struct_libusb_init_option(c.Struct):
  SIZE = 16
  option: 'enum_libusb_option'
  value: 'struct_libusb_init_option_value'
@c.record
class struct_libusb_init_option_value(c.Struct):
  SIZE = 8
  ival: 'ctypes.c_int32'
  log_cbval: 'libusb_log_cb'
struct_libusb_init_option_value.register_fields([('ival', ctypes.c_int32, 0), ('log_cbval', libusb_log_cb, 0)])
struct_libusb_init_option.register_fields([('option', enum_libusb_option, 0), ('value', struct_libusb_init_option_value, 8)])
@dll.bind
def libusb_init(ctx:c.POINTER[c.POINTER[libusb_context]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_init_context(ctx:c.POINTER[c.POINTER[libusb_context]], options:c.Array[struct_libusb_init_option, Literal[0]], num_options:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_exit(ctx:c.POINTER[libusb_context]) -> None: ...
@dll.bind
def libusb_set_debug(ctx:c.POINTER[libusb_context], level:ctypes.c_int32) -> None: ...
@dll.bind
def libusb_set_log_cb(ctx:c.POINTER[libusb_context], cb:libusb_log_cb, mode:ctypes.c_int32) -> None: ...
@dll.bind
def libusb_get_version() -> c.POINTER[struct_libusb_version]: ...
@dll.bind
def libusb_has_capability(capability:uint32_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_error_name(errcode:ctypes.c_int32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind
def libusb_setlocale(locale:c.POINTER[ctypes.c_char]) -> ctypes.c_int32: ...
@dll.bind
def libusb_strerror(errcode:ctypes.c_int32) -> c.POINTER[ctypes.c_char]: ...
ssize_t: TypeAlias = ctypes.c_int64
@dll.bind
def libusb_get_device_list(ctx:c.POINTER[libusb_context], list:c.POINTER[c.POINTER[c.POINTER[libusb_device]]]) -> ssize_t: ...
@dll.bind
def libusb_free_device_list(list:c.POINTER[c.POINTER[libusb_device]], unref_devices:ctypes.c_int32) -> None: ...
@dll.bind
def libusb_ref_device(dev:c.POINTER[libusb_device]) -> c.POINTER[libusb_device]: ...
@dll.bind
def libusb_unref_device(dev:c.POINTER[libusb_device]) -> None: ...
@dll.bind
def libusb_get_configuration(dev:c.POINTER[libusb_device_handle], config:c.POINTER[ctypes.c_int32]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_device_descriptor(dev:c.POINTER[libusb_device], desc:c.POINTER[struct_libusb_device_descriptor]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_active_config_descriptor(dev:c.POINTER[libusb_device], config:c.POINTER[c.POINTER[struct_libusb_config_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_config_descriptor(dev:c.POINTER[libusb_device], config_index:uint8_t, config:c.POINTER[c.POINTER[struct_libusb_config_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_config_descriptor_by_value(dev:c.POINTER[libusb_device], bConfigurationValue:uint8_t, config:c.POINTER[c.POINTER[struct_libusb_config_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_config_descriptor(config:c.POINTER[struct_libusb_config_descriptor]) -> None: ...
@dll.bind
def libusb_get_ss_endpoint_companion_descriptor(ctx:c.POINTER[libusb_context], endpoint:c.POINTER[struct_libusb_endpoint_descriptor], ep_comp:c.POINTER[c.POINTER[struct_libusb_ss_endpoint_companion_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_ss_endpoint_companion_descriptor(ep_comp:c.POINTER[struct_libusb_ss_endpoint_companion_descriptor]) -> None: ...
@dll.bind
def libusb_get_bos_descriptor(dev_handle:c.POINTER[libusb_device_handle], bos:c.POINTER[c.POINTER[struct_libusb_bos_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_bos_descriptor(bos:c.POINTER[struct_libusb_bos_descriptor]) -> None: ...
@dll.bind
def libusb_get_usb_2_0_extension_descriptor(ctx:c.POINTER[libusb_context], dev_cap:c.POINTER[struct_libusb_bos_dev_capability_descriptor], usb_2_0_extension:c.POINTER[c.POINTER[struct_libusb_usb_2_0_extension_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_usb_2_0_extension_descriptor(usb_2_0_extension:c.POINTER[struct_libusb_usb_2_0_extension_descriptor]) -> None: ...
@dll.bind
def libusb_get_ss_usb_device_capability_descriptor(ctx:c.POINTER[libusb_context], dev_cap:c.POINTER[struct_libusb_bos_dev_capability_descriptor], ss_usb_device_cap:c.POINTER[c.POINTER[struct_libusb_ss_usb_device_capability_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_ss_usb_device_capability_descriptor(ss_usb_device_cap:c.POINTER[struct_libusb_ss_usb_device_capability_descriptor]) -> None: ...
@dll.bind
def libusb_get_container_id_descriptor(ctx:c.POINTER[libusb_context], dev_cap:c.POINTER[struct_libusb_bos_dev_capability_descriptor], container_id:c.POINTER[c.POINTER[struct_libusb_container_id_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_container_id_descriptor(container_id:c.POINTER[struct_libusb_container_id_descriptor]) -> None: ...
@dll.bind
def libusb_get_platform_descriptor(ctx:c.POINTER[libusb_context], dev_cap:c.POINTER[struct_libusb_bos_dev_capability_descriptor], platform_descriptor:c.POINTER[c.POINTER[struct_libusb_platform_descriptor]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_platform_descriptor(platform_descriptor:c.POINTER[struct_libusb_platform_descriptor]) -> None: ...
@dll.bind
def libusb_get_bus_number(dev:c.POINTER[libusb_device]) -> uint8_t: ...
@dll.bind
def libusb_get_port_number(dev:c.POINTER[libusb_device]) -> uint8_t: ...
@dll.bind
def libusb_get_port_numbers(dev:c.POINTER[libusb_device], port_numbers:c.POINTER[uint8_t], port_numbers_len:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_port_path(ctx:c.POINTER[libusb_context], dev:c.POINTER[libusb_device], path:c.POINTER[uint8_t], path_length:uint8_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_parent(dev:c.POINTER[libusb_device]) -> c.POINTER[libusb_device]: ...
@dll.bind
def libusb_get_device_address(dev:c.POINTER[libusb_device]) -> uint8_t: ...
@dll.bind
def libusb_get_device_speed(dev:c.POINTER[libusb_device]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_packet_size(dev:c.POINTER[libusb_device], endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_iso_packet_size(dev:c.POINTER[libusb_device], endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_alt_packet_size(dev:c.POINTER[libusb_device], interface_number:ctypes.c_int32, alternate_setting:ctypes.c_int32, endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_interface_association_descriptors(dev:c.POINTER[libusb_device], config_index:uint8_t, iad_array:c.POINTER[c.POINTER[struct_libusb_interface_association_descriptor_array]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_active_interface_association_descriptors(dev:c.POINTER[libusb_device], iad_array:c.POINTER[c.POINTER[struct_libusb_interface_association_descriptor_array]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_interface_association_descriptors(iad_array:c.POINTER[struct_libusb_interface_association_descriptor_array]) -> None: ...
intptr_t: TypeAlias = ctypes.c_int64
@dll.bind
def libusb_wrap_sys_device(ctx:c.POINTER[libusb_context], sys_dev:intptr_t, dev_handle:c.POINTER[c.POINTER[libusb_device_handle]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_open(dev:c.POINTER[libusb_device], dev_handle:c.POINTER[c.POINTER[libusb_device_handle]]) -> ctypes.c_int32: ...
@dll.bind
def libusb_close(dev_handle:c.POINTER[libusb_device_handle]) -> None: ...
@dll.bind
def libusb_get_device(dev_handle:c.POINTER[libusb_device_handle]) -> c.POINTER[libusb_device]: ...
@dll.bind
def libusb_set_configuration(dev_handle:c.POINTER[libusb_device_handle], configuration:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_claim_interface(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_release_interface(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_open_device_with_vid_pid(ctx:c.POINTER[libusb_context], vendor_id:uint16_t, product_id:uint16_t) -> c.POINTER[libusb_device_handle]: ...
@dll.bind
def libusb_set_interface_alt_setting(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32, alternate_setting:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_clear_halt(dev_handle:c.POINTER[libusb_device_handle], endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_reset_device(dev_handle:c.POINTER[libusb_device_handle]) -> ctypes.c_int32: ...
@dll.bind
def libusb_alloc_streams(dev_handle:c.POINTER[libusb_device_handle], num_streams:uint32_t, endpoints:c.POINTER[ctypes.c_ubyte], num_endpoints:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_streams(dev_handle:c.POINTER[libusb_device_handle], endpoints:c.POINTER[ctypes.c_ubyte], num_endpoints:ctypes.c_int32) -> ctypes.c_int32: ...
size_t: TypeAlias = ctypes.c_uint64
@dll.bind
def libusb_dev_mem_alloc(dev_handle:c.POINTER[libusb_device_handle], length:size_t) -> c.POINTER[ctypes.c_ubyte]: ...
@dll.bind
def libusb_dev_mem_free(dev_handle:c.POINTER[libusb_device_handle], buffer:c.POINTER[ctypes.c_ubyte], length:size_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_kernel_driver_active(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_detach_kernel_driver(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_attach_kernel_driver(dev_handle:c.POINTER[libusb_device_handle], interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_set_auto_detach_kernel_driver(dev_handle:c.POINTER[libusb_device_handle], enable:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_alloc_transfer(iso_packets:ctypes.c_int32) -> c.POINTER[struct_libusb_transfer]: ...
@dll.bind
def libusb_submit_transfer(transfer:c.POINTER[struct_libusb_transfer]) -> ctypes.c_int32: ...
@dll.bind
def libusb_cancel_transfer(transfer:c.POINTER[struct_libusb_transfer]) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_transfer(transfer:c.POINTER[struct_libusb_transfer]) -> None: ...
@dll.bind
def libusb_transfer_set_stream_id(transfer:c.POINTER[struct_libusb_transfer], stream_id:uint32_t) -> None: ...
@dll.bind
def libusb_transfer_get_stream_id(transfer:c.POINTER[struct_libusb_transfer]) -> uint32_t: ...
@dll.bind
def libusb_control_transfer(dev_handle:c.POINTER[libusb_device_handle], request_type:uint8_t, bRequest:uint8_t, wValue:uint16_t, wIndex:uint16_t, data:c.POINTER[ctypes.c_ubyte], wLength:uint16_t, timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_bulk_transfer(dev_handle:c.POINTER[libusb_device_handle], endpoint:ctypes.c_ubyte, data:c.POINTER[ctypes.c_ubyte], length:ctypes.c_int32, actual_length:c.POINTER[ctypes.c_int32], timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_interrupt_transfer(dev_handle:c.POINTER[libusb_device_handle], endpoint:ctypes.c_ubyte, data:c.POINTER[ctypes.c_ubyte], length:ctypes.c_int32, actual_length:c.POINTER[ctypes.c_int32], timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_string_descriptor_ascii(dev_handle:c.POINTER[libusb_device_handle], desc_index:uint8_t, data:c.POINTER[ctypes.c_ubyte], length:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_try_lock_events(ctx:c.POINTER[libusb_context]) -> ctypes.c_int32: ...
@dll.bind
def libusb_lock_events(ctx:c.POINTER[libusb_context]) -> None: ...
@dll.bind
def libusb_unlock_events(ctx:c.POINTER[libusb_context]) -> None: ...
@dll.bind
def libusb_event_handling_ok(ctx:c.POINTER[libusb_context]) -> ctypes.c_int32: ...
@dll.bind
def libusb_event_handler_active(ctx:c.POINTER[libusb_context]) -> ctypes.c_int32: ...
@dll.bind
def libusb_interrupt_event_handler(ctx:c.POINTER[libusb_context]) -> None: ...
@dll.bind
def libusb_lock_event_waiters(ctx:c.POINTER[libusb_context]) -> None: ...
@dll.bind
def libusb_unlock_event_waiters(ctx:c.POINTER[libusb_context]) -> None: ...
@c.record
class struct_timeval(c.Struct):
  SIZE = 16
  tv_sec: 'ctypes.c_int64'
  tv_usec: 'ctypes.c_int64'
__time_t: TypeAlias = ctypes.c_int64
__suseconds_t: TypeAlias = ctypes.c_int64
struct_timeval.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_usec', ctypes.c_int64, 8)])
@dll.bind
def libusb_wait_for_event(ctx:c.POINTER[libusb_context], tv:c.POINTER[struct_timeval]) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_timeout(ctx:c.POINTER[libusb_context], tv:c.POINTER[struct_timeval]) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_timeout_completed(ctx:c.POINTER[libusb_context], tv:c.POINTER[struct_timeval], completed:c.POINTER[ctypes.c_int32]) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events(ctx:c.POINTER[libusb_context]) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_completed(ctx:c.POINTER[libusb_context], completed:c.POINTER[ctypes.c_int32]) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_locked(ctx:c.POINTER[libusb_context], tv:c.POINTER[struct_timeval]) -> ctypes.c_int32: ...
@dll.bind
def libusb_pollfds_handle_timeouts(ctx:c.POINTER[libusb_context]) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_next_timeout(ctx:c.POINTER[libusb_context], tv:c.POINTER[struct_timeval]) -> ctypes.c_int32: ...
@c.record
class struct_libusb_pollfd(c.Struct):
  SIZE = 8
  fd: 'ctypes.c_int32'
  events: 'ctypes.c_int16'
struct_libusb_pollfd.register_fields([('fd', ctypes.c_int32, 0), ('events', ctypes.c_int16, 4)])
libusb_pollfd_added_cb: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_int32, ctypes.c_int16, ctypes.c_void_p]]
libusb_pollfd_removed_cb: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_int32, ctypes.c_void_p]]
@dll.bind
def libusb_get_pollfds(ctx:c.POINTER[libusb_context]) -> c.POINTER[c.POINTER[struct_libusb_pollfd]]: ...
@dll.bind
def libusb_free_pollfds(pollfds:c.POINTER[c.POINTER[struct_libusb_pollfd]]) -> None: ...
@dll.bind
def libusb_set_pollfd_notifiers(ctx:c.POINTER[libusb_context], added_cb:libusb_pollfd_added_cb, removed_cb:libusb_pollfd_removed_cb, user_data:ctypes.c_void_p) -> None: ...
libusb_hotplug_callback_handle: TypeAlias = ctypes.c_int32
class libusb_hotplug_event(ctypes.c_uint32, c.Enum): pass
LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED', 1)
LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT', 2)

class libusb_hotplug_flag(ctypes.c_uint32, c.Enum): pass
LIBUSB_HOTPLUG_ENUMERATE = libusb_hotplug_flag.define('LIBUSB_HOTPLUG_ENUMERATE', 1)

libusb_hotplug_callback_fn: TypeAlias = c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_libusb_context], c.POINTER[struct_libusb_device], libusb_hotplug_event, ctypes.c_void_p]]
@dll.bind
def libusb_hotplug_register_callback(ctx:c.POINTER[libusb_context], events:ctypes.c_int32, flags:ctypes.c_int32, vendor_id:ctypes.c_int32, product_id:ctypes.c_int32, dev_class:ctypes.c_int32, cb_fn:libusb_hotplug_callback_fn, user_data:ctypes.c_void_p, callback_handle:c.POINTER[libusb_hotplug_callback_handle]) -> ctypes.c_int32: ...
@dll.bind
def libusb_hotplug_deregister_callback(ctx:c.POINTER[libusb_context], callback_handle:libusb_hotplug_callback_handle) -> None: ...
@dll.bind
def libusb_hotplug_get_user_data(ctx:c.POINTER[libusb_context], callback_handle:libusb_hotplug_callback_handle) -> ctypes.c_void_p: ...
@dll.bind
def libusb_set_option(ctx:c.POINTER[libusb_context], option:enum_libusb_option) -> ctypes.c_int32: ...
LIBUSB_DEPRECATED_FOR = lambda f: __attribute__ ((deprecated)) # type: ignore
LIBUSB_API_VERSION = 0x0100010A # type: ignore
LIBUSBX_API_VERSION = LIBUSB_API_VERSION # type: ignore
LIBUSB_DT_DEVICE_SIZE = 18 # type: ignore
LIBUSB_DT_CONFIG_SIZE = 9 # type: ignore
LIBUSB_DT_INTERFACE_SIZE = 9 # type: ignore
LIBUSB_DT_ENDPOINT_SIZE = 7 # type: ignore
LIBUSB_DT_ENDPOINT_AUDIO_SIZE = 9 # type: ignore
LIBUSB_DT_HUB_NONVAR_SIZE = 7 # type: ignore
LIBUSB_DT_SS_ENDPOINT_COMPANION_SIZE = 6 # type: ignore
LIBUSB_DT_BOS_SIZE = 5 # type: ignore
LIBUSB_DT_DEVICE_CAPABILITY_SIZE = 3 # type: ignore
LIBUSB_BT_USB_2_0_EXTENSION_SIZE = 7 # type: ignore
LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE = 10 # type: ignore
LIBUSB_BT_CONTAINER_ID_SIZE = 20 # type: ignore
LIBUSB_BT_PLATFORM_DESCRIPTOR_MIN_SIZE = 20 # type: ignore
LIBUSB_DT_BOS_MAX_SIZE = (LIBUSB_DT_BOS_SIZE + LIBUSB_BT_USB_2_0_EXTENSION_SIZE + LIBUSB_BT_SS_USB_DEVICE_CAPABILITY_SIZE + LIBUSB_BT_CONTAINER_ID_SIZE) # type: ignore
LIBUSB_ENDPOINT_ADDRESS_MASK = 0x0f # type: ignore
LIBUSB_ENDPOINT_DIR_MASK = 0x80 # type: ignore
LIBUSB_TRANSFER_TYPE_MASK = 0x03 # type: ignore
LIBUSB_ISO_SYNC_TYPE_MASK = 0x0c # type: ignore
LIBUSB_ISO_USAGE_TYPE_MASK = 0x30 # type: ignore
LIBUSB_ERROR_COUNT = 14 # type: ignore
LIBUSB_OPTION_WEAK_AUTHORITY = LIBUSB_OPTION_NO_DEVICE_DISCOVERY # type: ignore
LIBUSB_HOTPLUG_NO_FLAGS = 0 # type: ignore
LIBUSB_HOTPLUG_MATCH_ANY = -1 # type: ignore