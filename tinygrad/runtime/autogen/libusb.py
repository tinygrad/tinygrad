# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('libusb', 'usb-1.0')
enum_libusb_class_code: dict[int, str] = {(LIBUSB_CLASS_PER_INTERFACE:=0): 'LIBUSB_CLASS_PER_INTERFACE', (LIBUSB_CLASS_AUDIO:=1): 'LIBUSB_CLASS_AUDIO', (LIBUSB_CLASS_COMM:=2): 'LIBUSB_CLASS_COMM', (LIBUSB_CLASS_HID:=3): 'LIBUSB_CLASS_HID', (LIBUSB_CLASS_PHYSICAL:=5): 'LIBUSB_CLASS_PHYSICAL', (LIBUSB_CLASS_IMAGE:=6): 'LIBUSB_CLASS_IMAGE', (LIBUSB_CLASS_PTP:=6): 'LIBUSB_CLASS_PTP', (LIBUSB_CLASS_PRINTER:=7): 'LIBUSB_CLASS_PRINTER', (LIBUSB_CLASS_MASS_STORAGE:=8): 'LIBUSB_CLASS_MASS_STORAGE', (LIBUSB_CLASS_HUB:=9): 'LIBUSB_CLASS_HUB', (LIBUSB_CLASS_DATA:=10): 'LIBUSB_CLASS_DATA', (LIBUSB_CLASS_SMART_CARD:=11): 'LIBUSB_CLASS_SMART_CARD', (LIBUSB_CLASS_CONTENT_SECURITY:=13): 'LIBUSB_CLASS_CONTENT_SECURITY', (LIBUSB_CLASS_VIDEO:=14): 'LIBUSB_CLASS_VIDEO', (LIBUSB_CLASS_PERSONAL_HEALTHCARE:=15): 'LIBUSB_CLASS_PERSONAL_HEALTHCARE', (LIBUSB_CLASS_DIAGNOSTIC_DEVICE:=220): 'LIBUSB_CLASS_DIAGNOSTIC_DEVICE', (LIBUSB_CLASS_WIRELESS:=224): 'LIBUSB_CLASS_WIRELESS', (LIBUSB_CLASS_MISCELLANEOUS:=239): 'LIBUSB_CLASS_MISCELLANEOUS', (LIBUSB_CLASS_APPLICATION:=254): 'LIBUSB_CLASS_APPLICATION', (LIBUSB_CLASS_VENDOR_SPEC:=255): 'LIBUSB_CLASS_VENDOR_SPEC'}
enum_libusb_descriptor_type: dict[int, str] = {(LIBUSB_DT_DEVICE:=1): 'LIBUSB_DT_DEVICE', (LIBUSB_DT_CONFIG:=2): 'LIBUSB_DT_CONFIG', (LIBUSB_DT_STRING:=3): 'LIBUSB_DT_STRING', (LIBUSB_DT_INTERFACE:=4): 'LIBUSB_DT_INTERFACE', (LIBUSB_DT_ENDPOINT:=5): 'LIBUSB_DT_ENDPOINT', (LIBUSB_DT_INTERFACE_ASSOCIATION:=11): 'LIBUSB_DT_INTERFACE_ASSOCIATION', (LIBUSB_DT_BOS:=15): 'LIBUSB_DT_BOS', (LIBUSB_DT_DEVICE_CAPABILITY:=16): 'LIBUSB_DT_DEVICE_CAPABILITY', (LIBUSB_DT_HID:=33): 'LIBUSB_DT_HID', (LIBUSB_DT_REPORT:=34): 'LIBUSB_DT_REPORT', (LIBUSB_DT_PHYSICAL:=35): 'LIBUSB_DT_PHYSICAL', (LIBUSB_DT_HUB:=41): 'LIBUSB_DT_HUB', (LIBUSB_DT_SUPERSPEED_HUB:=42): 'LIBUSB_DT_SUPERSPEED_HUB', (LIBUSB_DT_SS_ENDPOINT_COMPANION:=48): 'LIBUSB_DT_SS_ENDPOINT_COMPANION'}
enum_libusb_endpoint_direction: dict[int, str] = {(LIBUSB_ENDPOINT_OUT:=0): 'LIBUSB_ENDPOINT_OUT', (LIBUSB_ENDPOINT_IN:=128): 'LIBUSB_ENDPOINT_IN'}
enum_libusb_endpoint_transfer_type: dict[int, str] = {(LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL:=0): 'LIBUSB_ENDPOINT_TRANSFER_TYPE_CONTROL', (LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS:=1): 'LIBUSB_ENDPOINT_TRANSFER_TYPE_ISOCHRONOUS', (LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK:=2): 'LIBUSB_ENDPOINT_TRANSFER_TYPE_BULK', (LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT:=3): 'LIBUSB_ENDPOINT_TRANSFER_TYPE_INTERRUPT'}
enum_libusb_standard_request: dict[int, str] = {(LIBUSB_REQUEST_GET_STATUS:=0): 'LIBUSB_REQUEST_GET_STATUS', (LIBUSB_REQUEST_CLEAR_FEATURE:=1): 'LIBUSB_REQUEST_CLEAR_FEATURE', (LIBUSB_REQUEST_SET_FEATURE:=3): 'LIBUSB_REQUEST_SET_FEATURE', (LIBUSB_REQUEST_SET_ADDRESS:=5): 'LIBUSB_REQUEST_SET_ADDRESS', (LIBUSB_REQUEST_GET_DESCRIPTOR:=6): 'LIBUSB_REQUEST_GET_DESCRIPTOR', (LIBUSB_REQUEST_SET_DESCRIPTOR:=7): 'LIBUSB_REQUEST_SET_DESCRIPTOR', (LIBUSB_REQUEST_GET_CONFIGURATION:=8): 'LIBUSB_REQUEST_GET_CONFIGURATION', (LIBUSB_REQUEST_SET_CONFIGURATION:=9): 'LIBUSB_REQUEST_SET_CONFIGURATION', (LIBUSB_REQUEST_GET_INTERFACE:=10): 'LIBUSB_REQUEST_GET_INTERFACE', (LIBUSB_REQUEST_SET_INTERFACE:=11): 'LIBUSB_REQUEST_SET_INTERFACE', (LIBUSB_REQUEST_SYNCH_FRAME:=12): 'LIBUSB_REQUEST_SYNCH_FRAME', (LIBUSB_REQUEST_SET_SEL:=48): 'LIBUSB_REQUEST_SET_SEL', (LIBUSB_SET_ISOCH_DELAY:=49): 'LIBUSB_SET_ISOCH_DELAY'}
enum_libusb_request_type: dict[int, str] = {(LIBUSB_REQUEST_TYPE_STANDARD:=0): 'LIBUSB_REQUEST_TYPE_STANDARD', (LIBUSB_REQUEST_TYPE_CLASS:=32): 'LIBUSB_REQUEST_TYPE_CLASS', (LIBUSB_REQUEST_TYPE_VENDOR:=64): 'LIBUSB_REQUEST_TYPE_VENDOR', (LIBUSB_REQUEST_TYPE_RESERVED:=96): 'LIBUSB_REQUEST_TYPE_RESERVED'}
enum_libusb_request_recipient: dict[int, str] = {(LIBUSB_RECIPIENT_DEVICE:=0): 'LIBUSB_RECIPIENT_DEVICE', (LIBUSB_RECIPIENT_INTERFACE:=1): 'LIBUSB_RECIPIENT_INTERFACE', (LIBUSB_RECIPIENT_ENDPOINT:=2): 'LIBUSB_RECIPIENT_ENDPOINT', (LIBUSB_RECIPIENT_OTHER:=3): 'LIBUSB_RECIPIENT_OTHER'}
enum_libusb_iso_sync_type: dict[int, str] = {(LIBUSB_ISO_SYNC_TYPE_NONE:=0): 'LIBUSB_ISO_SYNC_TYPE_NONE', (LIBUSB_ISO_SYNC_TYPE_ASYNC:=1): 'LIBUSB_ISO_SYNC_TYPE_ASYNC', (LIBUSB_ISO_SYNC_TYPE_ADAPTIVE:=2): 'LIBUSB_ISO_SYNC_TYPE_ADAPTIVE', (LIBUSB_ISO_SYNC_TYPE_SYNC:=3): 'LIBUSB_ISO_SYNC_TYPE_SYNC'}
enum_libusb_iso_usage_type: dict[int, str] = {(LIBUSB_ISO_USAGE_TYPE_DATA:=0): 'LIBUSB_ISO_USAGE_TYPE_DATA', (LIBUSB_ISO_USAGE_TYPE_FEEDBACK:=1): 'LIBUSB_ISO_USAGE_TYPE_FEEDBACK', (LIBUSB_ISO_USAGE_TYPE_IMPLICIT:=2): 'LIBUSB_ISO_USAGE_TYPE_IMPLICIT'}
enum_libusb_supported_speed: dict[int, str] = {(LIBUSB_LOW_SPEED_OPERATION:=1): 'LIBUSB_LOW_SPEED_OPERATION', (LIBUSB_FULL_SPEED_OPERATION:=2): 'LIBUSB_FULL_SPEED_OPERATION', (LIBUSB_HIGH_SPEED_OPERATION:=4): 'LIBUSB_HIGH_SPEED_OPERATION', (LIBUSB_SUPER_SPEED_OPERATION:=8): 'LIBUSB_SUPER_SPEED_OPERATION'}
enum_libusb_usb_2_0_extension_attributes: dict[int, str] = {(LIBUSB_BM_LPM_SUPPORT:=2): 'LIBUSB_BM_LPM_SUPPORT'}
enum_libusb_ss_usb_device_capability_attributes: dict[int, str] = {(LIBUSB_BM_LTM_SUPPORT:=2): 'LIBUSB_BM_LTM_SUPPORT'}
enum_libusb_bos_type: dict[int, str] = {(LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY:=1): 'LIBUSB_BT_WIRELESS_USB_DEVICE_CAPABILITY', (LIBUSB_BT_USB_2_0_EXTENSION:=2): 'LIBUSB_BT_USB_2_0_EXTENSION', (LIBUSB_BT_SS_USB_DEVICE_CAPABILITY:=3): 'LIBUSB_BT_SS_USB_DEVICE_CAPABILITY', (LIBUSB_BT_CONTAINER_ID:=4): 'LIBUSB_BT_CONTAINER_ID', (LIBUSB_BT_PLATFORM_DESCRIPTOR:=5): 'LIBUSB_BT_PLATFORM_DESCRIPTOR'}
@c.record
class struct_libusb_device_descriptor(c.Struct):
  SIZE = 18
  bLength: int
  bDescriptorType: int
  bcdUSB: int
  bDeviceClass: int
  bDeviceSubClass: int
  bDeviceProtocol: int
  bMaxPacketSize0: int
  idVendor: int
  idProduct: int
  bcdDevice: int
  iManufacturer: int
  iProduct: int
  iSerialNumber: int
  bNumConfigurations: int
uint8_t: TypeAlias = ctypes.c_ubyte
uint16_t: TypeAlias = ctypes.c_uint16
struct_libusb_device_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bcdUSB', uint16_t, 2), ('bDeviceClass', uint8_t, 4), ('bDeviceSubClass', uint8_t, 5), ('bDeviceProtocol', uint8_t, 6), ('bMaxPacketSize0', uint8_t, 7), ('idVendor', uint16_t, 8), ('idProduct', uint16_t, 10), ('bcdDevice', uint16_t, 12), ('iManufacturer', uint8_t, 14), ('iProduct', uint8_t, 15), ('iSerialNumber', uint8_t, 16), ('bNumConfigurations', uint8_t, 17)])
@c.record
class struct_libusb_endpoint_descriptor(c.Struct):
  SIZE = 32
  bLength: int
  bDescriptorType: int
  bEndpointAddress: int
  bmAttributes: int
  wMaxPacketSize: int
  bInterval: int
  bRefresh: int
  bSynchAddress: int
  extra: ctypes._Pointer[ctypes.c_ubyte]
  extra_length: int
struct_libusb_endpoint_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bEndpointAddress', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wMaxPacketSize', uint16_t, 4), ('bInterval', uint8_t, 6), ('bRefresh', uint8_t, 7), ('bSynchAddress', uint8_t, 8), ('extra', ctypes.POINTER(ctypes.c_ubyte), 16), ('extra_length', ctypes.c_int32, 24)])
@c.record
class struct_libusb_interface_association_descriptor(c.Struct):
  SIZE = 8
  bLength: int
  bDescriptorType: int
  bFirstInterface: int
  bInterfaceCount: int
  bFunctionClass: int
  bFunctionSubClass: int
  bFunctionProtocol: int
  iFunction: int
struct_libusb_interface_association_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bFirstInterface', uint8_t, 2), ('bInterfaceCount', uint8_t, 3), ('bFunctionClass', uint8_t, 4), ('bFunctionSubClass', uint8_t, 5), ('bFunctionProtocol', uint8_t, 6), ('iFunction', uint8_t, 7)])
@c.record
class struct_libusb_interface_association_descriptor_array(c.Struct):
  SIZE = 16
  iad: ctypes._Pointer[struct_libusb_interface_association_descriptor]
  length: int
struct_libusb_interface_association_descriptor_array.register_fields([('iad', ctypes.POINTER(struct_libusb_interface_association_descriptor), 0), ('length', ctypes.c_int32, 8)])
@c.record
class struct_libusb_interface_descriptor(c.Struct):
  SIZE = 40
  bLength: int
  bDescriptorType: int
  bInterfaceNumber: int
  bAlternateSetting: int
  bNumEndpoints: int
  bInterfaceClass: int
  bInterfaceSubClass: int
  bInterfaceProtocol: int
  iInterface: int
  endpoint: ctypes._Pointer[struct_libusb_endpoint_descriptor]
  extra: ctypes._Pointer[ctypes.c_ubyte]
  extra_length: int
struct_libusb_interface_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bInterfaceNumber', uint8_t, 2), ('bAlternateSetting', uint8_t, 3), ('bNumEndpoints', uint8_t, 4), ('bInterfaceClass', uint8_t, 5), ('bInterfaceSubClass', uint8_t, 6), ('bInterfaceProtocol', uint8_t, 7), ('iInterface', uint8_t, 8), ('endpoint', ctypes.POINTER(struct_libusb_endpoint_descriptor), 16), ('extra', ctypes.POINTER(ctypes.c_ubyte), 24), ('extra_length', ctypes.c_int32, 32)])
@c.record
class struct_libusb_interface(c.Struct):
  SIZE = 16
  altsetting: ctypes._Pointer[struct_libusb_interface_descriptor]
  num_altsetting: int
struct_libusb_interface.register_fields([('altsetting', ctypes.POINTER(struct_libusb_interface_descriptor), 0), ('num_altsetting', ctypes.c_int32, 8)])
@c.record
class struct_libusb_config_descriptor(c.Struct):
  SIZE = 40
  bLength: int
  bDescriptorType: int
  wTotalLength: int
  bNumInterfaces: int
  bConfigurationValue: int
  iConfiguration: int
  bmAttributes: int
  MaxPower: int
  interface: ctypes._Pointer[struct_libusb_interface]
  extra: ctypes._Pointer[ctypes.c_ubyte]
  extra_length: int
struct_libusb_config_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('wTotalLength', uint16_t, 2), ('bNumInterfaces', uint8_t, 4), ('bConfigurationValue', uint8_t, 5), ('iConfiguration', uint8_t, 6), ('bmAttributes', uint8_t, 7), ('MaxPower', uint8_t, 8), ('interface', ctypes.POINTER(struct_libusb_interface), 16), ('extra', ctypes.POINTER(ctypes.c_ubyte), 24), ('extra_length', ctypes.c_int32, 32)])
@c.record
class struct_libusb_ss_endpoint_companion_descriptor(c.Struct):
  SIZE = 6
  bLength: int
  bDescriptorType: int
  bMaxBurst: int
  bmAttributes: int
  wBytesPerInterval: int
struct_libusb_ss_endpoint_companion_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bMaxBurst', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wBytesPerInterval', uint16_t, 4)])
@c.record
class struct_libusb_bos_dev_capability_descriptor(c.Struct):
  SIZE = 3
  bLength: int
  bDescriptorType: int
  bDevCapabilityType: int
  dev_capability_data: ctypes.Array[ctypes.c_ubyte]
struct_libusb_bos_dev_capability_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('dev_capability_data', (uint8_t * 0), 3)])
@c.record
class struct_libusb_bos_descriptor(c.Struct):
  SIZE = 8
  bLength: int
  bDescriptorType: int
  wTotalLength: int
  bNumDeviceCaps: int
  dev_capability: ctypes.Array[ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor)]
struct_libusb_bos_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('wTotalLength', uint16_t, 2), ('bNumDeviceCaps', uint8_t, 4), ('dev_capability', (ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor) * 0), 8)])
@c.record
class struct_libusb_usb_2_0_extension_descriptor(c.Struct):
  SIZE = 8
  bLength: int
  bDescriptorType: int
  bDevCapabilityType: int
  bmAttributes: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_libusb_usb_2_0_extension_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bmAttributes', uint32_t, 4)])
@c.record
class struct_libusb_ss_usb_device_capability_descriptor(c.Struct):
  SIZE = 10
  bLength: int
  bDescriptorType: int
  bDevCapabilityType: int
  bmAttributes: int
  wSpeedSupported: int
  bFunctionalitySupport: int
  bU1DevExitLat: int
  bU2DevExitLat: int
struct_libusb_ss_usb_device_capability_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bmAttributes', uint8_t, 3), ('wSpeedSupported', uint16_t, 4), ('bFunctionalitySupport', uint8_t, 6), ('bU1DevExitLat', uint8_t, 7), ('bU2DevExitLat', uint16_t, 8)])
@c.record
class struct_libusb_container_id_descriptor(c.Struct):
  SIZE = 20
  bLength: int
  bDescriptorType: int
  bDevCapabilityType: int
  bReserved: int
  ContainerID: ctypes.Array[ctypes.c_ubyte]
struct_libusb_container_id_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bReserved', uint8_t, 3), ('ContainerID', (uint8_t * 16), 4)])
@c.record
class struct_libusb_platform_descriptor(c.Struct):
  SIZE = 20
  bLength: int
  bDescriptorType: int
  bDevCapabilityType: int
  bReserved: int
  PlatformCapabilityUUID: ctypes.Array[ctypes.c_ubyte]
  CapabilityData: ctypes.Array[ctypes.c_ubyte]
struct_libusb_platform_descriptor.register_fields([('bLength', uint8_t, 0), ('bDescriptorType', uint8_t, 1), ('bDevCapabilityType', uint8_t, 2), ('bReserved', uint8_t, 3), ('PlatformCapabilityUUID', (uint8_t * 16), 4), ('CapabilityData', (uint8_t * 0), 20)])
@c.record
class struct_libusb_control_setup(c.Struct):
  SIZE = 8
  bmRequestType: int
  bRequest: int
  wValue: int
  wIndex: int
  wLength: int
struct_libusb_control_setup.register_fields([('bmRequestType', uint8_t, 0), ('bRequest', uint8_t, 1), ('wValue', uint16_t, 2), ('wIndex', uint16_t, 4), ('wLength', uint16_t, 6)])
class struct_libusb_context(c.Struct): pass
class struct_libusb_device(c.Struct): pass
class struct_libusb_device_handle(c.Struct): pass
@c.record
class struct_libusb_version(c.Struct):
  SIZE = 24
  major: int
  minor: int
  micro: int
  nano: int
  rc: ctypes._Pointer[ctypes.c_char]
  describe: ctypes._Pointer[ctypes.c_char]
struct_libusb_version.register_fields([('major', uint16_t, 0), ('minor', uint16_t, 2), ('micro', uint16_t, 4), ('nano', uint16_t, 6), ('rc', ctypes.POINTER(ctypes.c_char), 8), ('describe', ctypes.POINTER(ctypes.c_char), 16)])
libusb_context: TypeAlias = struct_libusb_context
libusb_device: TypeAlias = struct_libusb_device
libusb_device_handle: TypeAlias = struct_libusb_device_handle
enum_libusb_speed: dict[int, str] = {(LIBUSB_SPEED_UNKNOWN:=0): 'LIBUSB_SPEED_UNKNOWN', (LIBUSB_SPEED_LOW:=1): 'LIBUSB_SPEED_LOW', (LIBUSB_SPEED_FULL:=2): 'LIBUSB_SPEED_FULL', (LIBUSB_SPEED_HIGH:=3): 'LIBUSB_SPEED_HIGH', (LIBUSB_SPEED_SUPER:=4): 'LIBUSB_SPEED_SUPER', (LIBUSB_SPEED_SUPER_PLUS:=5): 'LIBUSB_SPEED_SUPER_PLUS'}
enum_libusb_error: dict[int, str] = {(LIBUSB_SUCCESS:=0): 'LIBUSB_SUCCESS', (LIBUSB_ERROR_IO:=-1): 'LIBUSB_ERROR_IO', (LIBUSB_ERROR_INVALID_PARAM:=-2): 'LIBUSB_ERROR_INVALID_PARAM', (LIBUSB_ERROR_ACCESS:=-3): 'LIBUSB_ERROR_ACCESS', (LIBUSB_ERROR_NO_DEVICE:=-4): 'LIBUSB_ERROR_NO_DEVICE', (LIBUSB_ERROR_NOT_FOUND:=-5): 'LIBUSB_ERROR_NOT_FOUND', (LIBUSB_ERROR_BUSY:=-6): 'LIBUSB_ERROR_BUSY', (LIBUSB_ERROR_TIMEOUT:=-7): 'LIBUSB_ERROR_TIMEOUT', (LIBUSB_ERROR_OVERFLOW:=-8): 'LIBUSB_ERROR_OVERFLOW', (LIBUSB_ERROR_PIPE:=-9): 'LIBUSB_ERROR_PIPE', (LIBUSB_ERROR_INTERRUPTED:=-10): 'LIBUSB_ERROR_INTERRUPTED', (LIBUSB_ERROR_NO_MEM:=-11): 'LIBUSB_ERROR_NO_MEM', (LIBUSB_ERROR_NOT_SUPPORTED:=-12): 'LIBUSB_ERROR_NOT_SUPPORTED', (LIBUSB_ERROR_OTHER:=-99): 'LIBUSB_ERROR_OTHER'}
enum_libusb_transfer_type: dict[int, str] = {(LIBUSB_TRANSFER_TYPE_CONTROL:=0): 'LIBUSB_TRANSFER_TYPE_CONTROL', (LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:=1): 'LIBUSB_TRANSFER_TYPE_ISOCHRONOUS', (LIBUSB_TRANSFER_TYPE_BULK:=2): 'LIBUSB_TRANSFER_TYPE_BULK', (LIBUSB_TRANSFER_TYPE_INTERRUPT:=3): 'LIBUSB_TRANSFER_TYPE_INTERRUPT', (LIBUSB_TRANSFER_TYPE_BULK_STREAM:=4): 'LIBUSB_TRANSFER_TYPE_BULK_STREAM'}
enum_libusb_transfer_status: dict[int, str] = {(LIBUSB_TRANSFER_COMPLETED:=0): 'LIBUSB_TRANSFER_COMPLETED', (LIBUSB_TRANSFER_ERROR:=1): 'LIBUSB_TRANSFER_ERROR', (LIBUSB_TRANSFER_TIMED_OUT:=2): 'LIBUSB_TRANSFER_TIMED_OUT', (LIBUSB_TRANSFER_CANCELLED:=3): 'LIBUSB_TRANSFER_CANCELLED', (LIBUSB_TRANSFER_STALL:=4): 'LIBUSB_TRANSFER_STALL', (LIBUSB_TRANSFER_NO_DEVICE:=5): 'LIBUSB_TRANSFER_NO_DEVICE', (LIBUSB_TRANSFER_OVERFLOW:=6): 'LIBUSB_TRANSFER_OVERFLOW'}
enum_libusb_transfer_flags: dict[int, str] = {(LIBUSB_TRANSFER_SHORT_NOT_OK:=1): 'LIBUSB_TRANSFER_SHORT_NOT_OK', (LIBUSB_TRANSFER_FREE_BUFFER:=2): 'LIBUSB_TRANSFER_FREE_BUFFER', (LIBUSB_TRANSFER_FREE_TRANSFER:=4): 'LIBUSB_TRANSFER_FREE_TRANSFER', (LIBUSB_TRANSFER_ADD_ZERO_PACKET:=8): 'LIBUSB_TRANSFER_ADD_ZERO_PACKET'}
@c.record
class struct_libusb_iso_packet_descriptor(c.Struct):
  SIZE = 12
  length: int
  actual_length: int
  status: int
struct_libusb_iso_packet_descriptor.register_fields([('length', ctypes.c_uint32, 0), ('actual_length', ctypes.c_uint32, 4), ('status', ctypes.c_uint32, 8)])
@c.record
class struct_libusb_transfer(c.Struct):
  SIZE = 64
  dev_handle: ctypes._Pointer[struct_libusb_device_handle]
  flags: int
  endpoint: int
  type: int
  timeout: int
  status: int
  length: int
  actual_length: int
  callback: ctypes._CFunctionType
  user_data: int|None
  buffer: ctypes._Pointer[ctypes.c_ubyte]
  num_iso_packets: int
  iso_packet_desc: ctypes.Array[struct_libusb_iso_packet_descriptor]
libusb_transfer_cb_fn: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_transfer))
struct_libusb_transfer.register_fields([('dev_handle', ctypes.POINTER(libusb_device_handle), 0), ('flags', uint8_t, 8), ('endpoint', ctypes.c_ubyte, 9), ('type', ctypes.c_ubyte, 10), ('timeout', ctypes.c_uint32, 12), ('status', ctypes.c_uint32, 16), ('length', ctypes.c_int32, 20), ('actual_length', ctypes.c_int32, 24), ('callback', libusb_transfer_cb_fn, 32), ('user_data', ctypes.c_void_p, 40), ('buffer', ctypes.POINTER(ctypes.c_ubyte), 48), ('num_iso_packets', ctypes.c_int32, 56), ('iso_packet_desc', (struct_libusb_iso_packet_descriptor * 0), 60)])
enum_libusb_capability: dict[int, str] = {(LIBUSB_CAP_HAS_CAPABILITY:=0): 'LIBUSB_CAP_HAS_CAPABILITY', (LIBUSB_CAP_HAS_HOTPLUG:=1): 'LIBUSB_CAP_HAS_HOTPLUG', (LIBUSB_CAP_HAS_HID_ACCESS:=256): 'LIBUSB_CAP_HAS_HID_ACCESS', (LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER:=257): 'LIBUSB_CAP_SUPPORTS_DETACH_KERNEL_DRIVER'}
enum_libusb_log_level: dict[int, str] = {(LIBUSB_LOG_LEVEL_NONE:=0): 'LIBUSB_LOG_LEVEL_NONE', (LIBUSB_LOG_LEVEL_ERROR:=1): 'LIBUSB_LOG_LEVEL_ERROR', (LIBUSB_LOG_LEVEL_WARNING:=2): 'LIBUSB_LOG_LEVEL_WARNING', (LIBUSB_LOG_LEVEL_INFO:=3): 'LIBUSB_LOG_LEVEL_INFO', (LIBUSB_LOG_LEVEL_DEBUG:=4): 'LIBUSB_LOG_LEVEL_DEBUG'}
enum_libusb_log_cb_mode: dict[int, str] = {(LIBUSB_LOG_CB_GLOBAL:=1): 'LIBUSB_LOG_CB_GLOBAL', (LIBUSB_LOG_CB_CONTEXT:=2): 'LIBUSB_LOG_CB_CONTEXT'}
enum_libusb_option: dict[int, str] = {(LIBUSB_OPTION_LOG_LEVEL:=0): 'LIBUSB_OPTION_LOG_LEVEL', (LIBUSB_OPTION_USE_USBDK:=1): 'LIBUSB_OPTION_USE_USBDK', (LIBUSB_OPTION_NO_DEVICE_DISCOVERY:=2): 'LIBUSB_OPTION_NO_DEVICE_DISCOVERY', (LIBUSB_OPTION_LOG_CB:=3): 'LIBUSB_OPTION_LOG_CB', (LIBUSB_OPTION_MAX:=4): 'LIBUSB_OPTION_MAX'}
libusb_log_cb: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_libusb_context), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char))
@c.record
class struct_libusb_init_option(c.Struct):
  SIZE = 16
  option: int
  value: struct_libusb_init_option_value
@c.record
class struct_libusb_init_option_value(c.Struct):
  SIZE = 8
  ival: int
  log_cbval: ctypes._CFunctionType
struct_libusb_init_option_value.register_fields([('ival', ctypes.c_int32, 0), ('log_cbval', libusb_log_cb, 0)])
struct_libusb_init_option.register_fields([('option', ctypes.c_uint32, 0), ('value', struct_libusb_init_option_value, 8)])
@dll.bind(ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(libusb_context)))
def libusb_init(ctx:ctypes._Pointer[ctypes.POINTER(libusb_context)]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(libusb_context)), (struct_libusb_init_option * 0), ctypes.c_int32)
def libusb_init_context(ctx:ctypes._Pointer[ctypes.POINTER(libusb_context)], options:ctypes.Array[struct_libusb_init_option], num_options:int) -> int: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_exit(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context), ctypes.c_int32)
def libusb_set_debug(ctx:ctypes._Pointer[libusb_context], level:int) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context), libusb_log_cb, ctypes.c_int32)
def libusb_set_log_cb(ctx:ctypes._Pointer[libusb_context], cb:libusb_log_cb, mode:int) -> None: ...
@dll.bind(ctypes.POINTER(struct_libusb_version))
def libusb_get_version() -> ctypes._Pointer[struct_libusb_version]: ...
@dll.bind(ctypes.c_int32, uint32_t)
def libusb_has_capability(capability:uint32_t) -> int: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_int32)
def libusb_error_name(errcode:int) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(ctypes.c_char))
def libusb_setlocale(locale:ctypes._Pointer[ctypes.c_char]) -> int: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_int32)
def libusb_strerror(errcode:int) -> ctypes._Pointer[ctypes.c_char]: ...
ssize_t: TypeAlias = ctypes.c_int64
@dll.bind(ssize_t, ctypes.POINTER(libusb_context), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(libusb_device))))
def libusb_get_device_list(ctx:ctypes._Pointer[libusb_context], list:ctypes._Pointer[ctypes.POINTER(ctypes.POINTER(libusb_device))]) -> ssize_t: ...
@dll.bind(None, ctypes.POINTER(ctypes.POINTER(libusb_device)), ctypes.c_int32)
def libusb_free_device_list(list:ctypes._Pointer[ctypes.POINTER(libusb_device)], unref_devices:int) -> None: ...
@dll.bind(ctypes.POINTER(libusb_device), ctypes.POINTER(libusb_device))
def libusb_ref_device(dev:ctypes._Pointer[libusb_device]) -> ctypes._Pointer[libusb_device]: ...
@dll.bind(None, ctypes.POINTER(libusb_device))
def libusb_unref_device(dev:ctypes._Pointer[libusb_device]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_int32))
def libusb_get_configuration(dev:ctypes._Pointer[libusb_device_handle], config:ctypes._Pointer[ctypes.c_int32]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.POINTER(struct_libusb_device_descriptor))
def libusb_get_device_descriptor(dev:ctypes._Pointer[libusb_device], desc:ctypes._Pointer[struct_libusb_device_descriptor]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor)))
def libusb_get_active_config_descriptor(dev:ctypes._Pointer[libusb_device], config:ctypes._Pointer[ctypes.POINTER(struct_libusb_config_descriptor)]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor)))
def libusb_get_config_descriptor(dev:ctypes._Pointer[libusb_device], config_index:uint8_t, config:ctypes._Pointer[ctypes.POINTER(struct_libusb_config_descriptor)]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor)))
def libusb_get_config_descriptor_by_value(dev:ctypes._Pointer[libusb_device], bConfigurationValue:uint8_t, config:ctypes._Pointer[ctypes.POINTER(struct_libusb_config_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_config_descriptor))
def libusb_free_config_descriptor(config:ctypes._Pointer[struct_libusb_config_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_endpoint_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor)))
def libusb_get_ss_endpoint_companion_descriptor(ctx:ctypes._Pointer[libusb_context], endpoint:ctypes._Pointer[struct_libusb_endpoint_descriptor], ep_comp:ctypes._Pointer[ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor))
def libusb_free_ss_endpoint_companion_descriptor(ep_comp:ctypes._Pointer[struct_libusb_ss_endpoint_companion_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.POINTER(struct_libusb_bos_descriptor)))
def libusb_get_bos_descriptor(dev_handle:ctypes._Pointer[libusb_device_handle], bos:ctypes._Pointer[ctypes.POINTER(struct_libusb_bos_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_bos_descriptor))
def libusb_free_bos_descriptor(bos:ctypes._Pointer[struct_libusb_bos_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor)))
def libusb_get_usb_2_0_extension_descriptor(ctx:ctypes._Pointer[libusb_context], dev_cap:ctypes._Pointer[struct_libusb_bos_dev_capability_descriptor], usb_2_0_extension:ctypes._Pointer[ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor))
def libusb_free_usb_2_0_extension_descriptor(usb_2_0_extension:ctypes._Pointer[struct_libusb_usb_2_0_extension_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor)))
def libusb_get_ss_usb_device_capability_descriptor(ctx:ctypes._Pointer[libusb_context], dev_cap:ctypes._Pointer[struct_libusb_bos_dev_capability_descriptor], ss_usb_device_cap:ctypes._Pointer[ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor))
def libusb_free_ss_usb_device_capability_descriptor(ss_usb_device_cap:ctypes._Pointer[struct_libusb_ss_usb_device_capability_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_container_id_descriptor)))
def libusb_get_container_id_descriptor(ctx:ctypes._Pointer[libusb_context], dev_cap:ctypes._Pointer[struct_libusb_bos_dev_capability_descriptor], container_id:ctypes._Pointer[ctypes.POINTER(struct_libusb_container_id_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_container_id_descriptor))
def libusb_free_container_id_descriptor(container_id:ctypes._Pointer[struct_libusb_container_id_descriptor]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ctypes.POINTER(ctypes.POINTER(struct_libusb_platform_descriptor)))
def libusb_get_platform_descriptor(ctx:ctypes._Pointer[libusb_context], dev_cap:ctypes._Pointer[struct_libusb_bos_dev_capability_descriptor], platform_descriptor:ctypes._Pointer[ctypes.POINTER(struct_libusb_platform_descriptor)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_platform_descriptor))
def libusb_free_platform_descriptor(platform_descriptor:ctypes._Pointer[struct_libusb_platform_descriptor]) -> None: ...
@dll.bind(uint8_t, ctypes.POINTER(libusb_device))
def libusb_get_bus_number(dev:ctypes._Pointer[libusb_device]) -> uint8_t: ...
@dll.bind(uint8_t, ctypes.POINTER(libusb_device))
def libusb_get_port_number(dev:ctypes._Pointer[libusb_device]) -> uint8_t: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.POINTER(uint8_t), ctypes.c_int32)
def libusb_get_port_numbers(dev:ctypes._Pointer[libusb_device], port_numbers:ctypes._Pointer[uint8_t], port_numbers_len:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(libusb_device), ctypes.POINTER(uint8_t), uint8_t)
def libusb_get_port_path(ctx:ctypes._Pointer[libusb_context], dev:ctypes._Pointer[libusb_device], path:ctypes._Pointer[uint8_t], path_length:uint8_t) -> int: ...
@dll.bind(ctypes.POINTER(libusb_device), ctypes.POINTER(libusb_device))
def libusb_get_parent(dev:ctypes._Pointer[libusb_device]) -> ctypes._Pointer[libusb_device]: ...
@dll.bind(uint8_t, ctypes.POINTER(libusb_device))
def libusb_get_device_address(dev:ctypes._Pointer[libusb_device]) -> uint8_t: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device))
def libusb_get_device_speed(dev:ctypes._Pointer[libusb_device]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.c_ubyte)
def libusb_get_max_packet_size(dev:ctypes._Pointer[libusb_device], endpoint:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.c_ubyte)
def libusb_get_max_iso_packet_size(dev:ctypes._Pointer[libusb_device], endpoint:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.c_int32, ctypes.c_int32, ctypes.c_ubyte)
def libusb_get_max_alt_packet_size(dev:ctypes._Pointer[libusb_device], interface_number:int, alternate_setting:int, endpoint:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), uint8_t, ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array)))
def libusb_get_interface_association_descriptors(dev:ctypes._Pointer[libusb_device], config_index:uint8_t, iad_array:ctypes._Pointer[ctypes.POINTER(struct_libusb_interface_association_descriptor_array)]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array)))
def libusb_get_active_interface_association_descriptors(dev:ctypes._Pointer[libusb_device], iad_array:ctypes._Pointer[ctypes.POINTER(struct_libusb_interface_association_descriptor_array)]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_interface_association_descriptor_array))
def libusb_free_interface_association_descriptors(iad_array:ctypes._Pointer[struct_libusb_interface_association_descriptor_array]) -> None: ...
intptr_t: TypeAlias = ctypes.c_int64
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), intptr_t, ctypes.POINTER(ctypes.POINTER(libusb_device_handle)))
def libusb_wrap_sys_device(ctx:ctypes._Pointer[libusb_context], sys_dev:intptr_t, dev_handle:ctypes._Pointer[ctypes.POINTER(libusb_device_handle)]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device), ctypes.POINTER(ctypes.POINTER(libusb_device_handle)))
def libusb_open(dev:ctypes._Pointer[libusb_device], dev_handle:ctypes._Pointer[ctypes.POINTER(libusb_device_handle)]) -> int: ...
@dll.bind(None, ctypes.POINTER(libusb_device_handle))
def libusb_close(dev_handle:ctypes._Pointer[libusb_device_handle]) -> None: ...
@dll.bind(ctypes.POINTER(libusb_device), ctypes.POINTER(libusb_device_handle))
def libusb_get_device(dev_handle:ctypes._Pointer[libusb_device_handle]) -> ctypes._Pointer[libusb_device]: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_set_configuration(dev_handle:ctypes._Pointer[libusb_device_handle], configuration:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_claim_interface(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_release_interface(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int) -> int: ...
@dll.bind(ctypes.POINTER(libusb_device_handle), ctypes.POINTER(libusb_context), uint16_t, uint16_t)
def libusb_open_device_with_vid_pid(ctx:ctypes._Pointer[libusb_context], vendor_id:uint16_t, product_id:uint16_t) -> ctypes._Pointer[libusb_device_handle]: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32, ctypes.c_int32)
def libusb_set_interface_alt_setting(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int, alternate_setting:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte)
def libusb_clear_halt(dev_handle:ctypes._Pointer[libusb_device_handle], endpoint:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle))
def libusb_reset_device(dev_handle:ctypes._Pointer[libusb_device_handle]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), uint32_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32)
def libusb_alloc_streams(dev_handle:ctypes._Pointer[libusb_device_handle], num_streams:uint32_t, endpoints:ctypes._Pointer[ctypes.c_ubyte], num_endpoints:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32)
def libusb_free_streams(dev_handle:ctypes._Pointer[libusb_device_handle], endpoints:ctypes._Pointer[ctypes.c_ubyte], num_endpoints:int) -> int: ...
size_t: TypeAlias = ctypes.c_uint64
@dll.bind(ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(libusb_device_handle), size_t)
def libusb_dev_mem_alloc(dev_handle:ctypes._Pointer[libusb_device_handle], length:size_t) -> ctypes._Pointer[ctypes.c_ubyte]: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.POINTER(ctypes.c_ubyte), size_t)
def libusb_dev_mem_free(dev_handle:ctypes._Pointer[libusb_device_handle], buffer:ctypes._Pointer[ctypes.c_ubyte], length:size_t) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_kernel_driver_active(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_detach_kernel_driver(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_attach_kernel_driver(dev_handle:ctypes._Pointer[libusb_device_handle], interface_number:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_int32)
def libusb_set_auto_detach_kernel_driver(dev_handle:ctypes._Pointer[libusb_device_handle], enable:int) -> int: ...
@dll.bind(ctypes.POINTER(struct_libusb_transfer), ctypes.c_int32)
def libusb_alloc_transfer(iso_packets:int) -> ctypes._Pointer[struct_libusb_transfer]: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(struct_libusb_transfer))
def libusb_submit_transfer(transfer:ctypes._Pointer[struct_libusb_transfer]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(struct_libusb_transfer))
def libusb_cancel_transfer(transfer:ctypes._Pointer[struct_libusb_transfer]) -> int: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_transfer))
def libusb_free_transfer(transfer:ctypes._Pointer[struct_libusb_transfer]) -> None: ...
@dll.bind(None, ctypes.POINTER(struct_libusb_transfer), uint32_t)
def libusb_transfer_set_stream_id(transfer:ctypes._Pointer[struct_libusb_transfer], stream_id:uint32_t) -> None: ...
@dll.bind(uint32_t, ctypes.POINTER(struct_libusb_transfer))
def libusb_transfer_get_stream_id(transfer:ctypes._Pointer[struct_libusb_transfer]) -> uint32_t: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), uint8_t, uint8_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_ubyte), uint16_t, ctypes.c_uint32)
def libusb_control_transfer(dev_handle:ctypes._Pointer[libusb_device_handle], request_type:uint8_t, bRequest:uint8_t, wValue:uint16_t, wIndex:uint16_t, data:ctypes._Pointer[ctypes.c_ubyte], wLength:uint16_t, timeout:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32)
def libusb_bulk_transfer(dev_handle:ctypes._Pointer[libusb_device_handle], endpoint:int, data:ctypes._Pointer[ctypes.c_ubyte], length:int, actual_length:ctypes._Pointer[ctypes.c_int32], timeout:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32)
def libusb_interrupt_transfer(dev_handle:ctypes._Pointer[libusb_device_handle], endpoint:int, data:ctypes._Pointer[ctypes.c_ubyte], length:int, actual_length:ctypes._Pointer[ctypes.c_int32], timeout:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_device_handle), uint8_t, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32)
def libusb_get_string_descriptor_ascii(dev_handle:ctypes._Pointer[libusb_device_handle], desc_index:uint8_t, data:ctypes._Pointer[ctypes.c_ubyte], length:int) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context))
def libusb_try_lock_events(ctx:ctypes._Pointer[libusb_context]) -> int: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_lock_events(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_unlock_events(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context))
def libusb_event_handling_ok(ctx:ctypes._Pointer[libusb_context]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context))
def libusb_event_handler_active(ctx:ctypes._Pointer[libusb_context]) -> int: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_interrupt_event_handler(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_lock_event_waiters(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context))
def libusb_unlock_event_waiters(ctx:ctypes._Pointer[libusb_context]) -> None: ...
@c.record
class struct_timeval(c.Struct):
  SIZE = 16
  tv_sec: ctypes.c_int64
  tv_usec: ctypes.c_int64
__time_t: TypeAlias = ctypes.c_int64
__suseconds_t: TypeAlias = ctypes.c_int64
struct_timeval.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_usec', ctypes.c_int64, 8)])
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval))
def libusb_wait_for_event(ctx:ctypes._Pointer[libusb_context], tv:ctypes._Pointer[struct_timeval]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval))
def libusb_handle_events_timeout(ctx:ctypes._Pointer[libusb_context], tv:ctypes._Pointer[struct_timeval]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval), ctypes.POINTER(ctypes.c_int32))
def libusb_handle_events_timeout_completed(ctx:ctypes._Pointer[libusb_context], tv:ctypes._Pointer[struct_timeval], completed:ctypes._Pointer[ctypes.c_int32]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context))
def libusb_handle_events(ctx:ctypes._Pointer[libusb_context]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(ctypes.c_int32))
def libusb_handle_events_completed(ctx:ctypes._Pointer[libusb_context], completed:ctypes._Pointer[ctypes.c_int32]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval))
def libusb_handle_events_locked(ctx:ctypes._Pointer[libusb_context], tv:ctypes._Pointer[struct_timeval]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context))
def libusb_pollfds_handle_timeouts(ctx:ctypes._Pointer[libusb_context]) -> int: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.POINTER(struct_timeval))
def libusb_get_next_timeout(ctx:ctypes._Pointer[libusb_context], tv:ctypes._Pointer[struct_timeval]) -> int: ...
@c.record
class struct_libusb_pollfd(c.Struct):
  SIZE = 8
  fd: int
  events: int
struct_libusb_pollfd.register_fields([('fd', ctypes.c_int32, 0), ('events', ctypes.c_int16, 4)])
libusb_pollfd_added_cb: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int16, ctypes.c_void_p)
libusb_pollfd_removed_cb: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_void_p)
@dll.bind(ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd)), ctypes.POINTER(libusb_context))
def libusb_get_pollfds(ctx:ctypes._Pointer[libusb_context]) -> ctypes._Pointer[ctypes.POINTER(struct_libusb_pollfd)]: ...
@dll.bind(None, ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd)))
def libusb_free_pollfds(pollfds:ctypes._Pointer[ctypes.POINTER(struct_libusb_pollfd)]) -> None: ...
@dll.bind(None, ctypes.POINTER(libusb_context), libusb_pollfd_added_cb, libusb_pollfd_removed_cb, ctypes.c_void_p)
def libusb_set_pollfd_notifiers(ctx:ctypes._Pointer[libusb_context], added_cb:libusb_pollfd_added_cb, removed_cb:libusb_pollfd_removed_cb, user_data:int|None) -> None: ...
libusb_hotplug_callback_handle: TypeAlias = ctypes.c_int32
libusb_hotplug_event: dict[int, str] = {(LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED:=1): 'LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED', (LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT:=2): 'LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT'}
libusb_hotplug_flag: dict[int, str] = {(LIBUSB_HOTPLUG_ENUMERATE:=1): 'LIBUSB_HOTPLUG_ENUMERATE'}
libusb_hotplug_callback_fn: TypeAlias = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_libusb_context), ctypes.POINTER(struct_libusb_device), ctypes.c_uint32, ctypes.c_void_p)
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, libusb_hotplug_callback_fn, ctypes.c_void_p, ctypes.POINTER(libusb_hotplug_callback_handle))
def libusb_hotplug_register_callback(ctx:ctypes._Pointer[libusb_context], events:int, flags:int, vendor_id:int, product_id:int, dev_class:int, cb_fn:libusb_hotplug_callback_fn, user_data:int|None, callback_handle:ctypes._Pointer[libusb_hotplug_callback_handle]) -> int: ...
@dll.bind(None, ctypes.POINTER(libusb_context), libusb_hotplug_callback_handle)
def libusb_hotplug_deregister_callback(ctx:ctypes._Pointer[libusb_context], callback_handle:libusb_hotplug_callback_handle) -> None: ...
@dll.bind(ctypes.c_void_p, ctypes.POINTER(libusb_context), libusb_hotplug_callback_handle)
def libusb_hotplug_get_user_data(ctx:ctypes._Pointer[libusb_context], callback_handle:libusb_hotplug_callback_handle) -> int|None: ...
@dll.bind(ctypes.c_int32, ctypes.POINTER(libusb_context), ctypes.c_uint32)
def libusb_set_option(ctx:ctypes._Pointer[libusb_context], option:ctypes.c_uint32) -> int: ...
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