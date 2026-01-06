# mypy: ignore-errors
from __future__ import annotations
import ctypes
from typing import Annotated
from tinygrad.runtime.support.c import DLL, record, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
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

@record
class struct_libusb_device_descriptor:
  SIZE = 18
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bcdUSB: Annotated[uint16_t, 2]
  bDeviceClass: Annotated[uint8_t, 4]
  bDeviceSubClass: Annotated[uint8_t, 5]
  bDeviceProtocol: Annotated[uint8_t, 6]
  bMaxPacketSize0: Annotated[uint8_t, 7]
  idVendor: Annotated[uint16_t, 8]
  idProduct: Annotated[uint16_t, 10]
  bcdDevice: Annotated[uint16_t, 12]
  iManufacturer: Annotated[uint8_t, 14]
  iProduct: Annotated[uint8_t, 15]
  iSerialNumber: Annotated[uint8_t, 16]
  bNumConfigurations: Annotated[uint8_t, 17]
uint8_t = ctypes.c_ubyte
uint16_t = ctypes.c_uint16
@record
class struct_libusb_endpoint_descriptor:
  SIZE = 32
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bEndpointAddress: Annotated[uint8_t, 2]
  bmAttributes: Annotated[uint8_t, 3]
  wMaxPacketSize: Annotated[uint16_t, 4]
  bInterval: Annotated[uint8_t, 6]
  bRefresh: Annotated[uint8_t, 7]
  bSynchAddress: Annotated[uint8_t, 8]
  extra: Annotated[ctypes.POINTER(ctypes.c_ubyte), 16]
  extra_length: Annotated[ctypes.c_int32, 24]
@record
class struct_libusb_interface_association_descriptor:
  SIZE = 8
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bFirstInterface: Annotated[uint8_t, 2]
  bInterfaceCount: Annotated[uint8_t, 3]
  bFunctionClass: Annotated[uint8_t, 4]
  bFunctionSubClass: Annotated[uint8_t, 5]
  bFunctionProtocol: Annotated[uint8_t, 6]
  iFunction: Annotated[uint8_t, 7]
@record
class struct_libusb_interface_association_descriptor_array:
  SIZE = 16
  iad: Annotated[ctypes.POINTER(struct_libusb_interface_association_descriptor), 0]
  length: Annotated[ctypes.c_int32, 8]
@record
class struct_libusb_interface_descriptor:
  SIZE = 40
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bInterfaceNumber: Annotated[uint8_t, 2]
  bAlternateSetting: Annotated[uint8_t, 3]
  bNumEndpoints: Annotated[uint8_t, 4]
  bInterfaceClass: Annotated[uint8_t, 5]
  bInterfaceSubClass: Annotated[uint8_t, 6]
  bInterfaceProtocol: Annotated[uint8_t, 7]
  iInterface: Annotated[uint8_t, 8]
  endpoint: Annotated[ctypes.POINTER(struct_libusb_endpoint_descriptor), 16]
  extra: Annotated[ctypes.POINTER(ctypes.c_ubyte), 24]
  extra_length: Annotated[ctypes.c_int32, 32]
@record
class struct_libusb_interface:
  SIZE = 16
  altsetting: Annotated[ctypes.POINTER(struct_libusb_interface_descriptor), 0]
  num_altsetting: Annotated[ctypes.c_int32, 8]
@record
class struct_libusb_config_descriptor:
  SIZE = 40
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  wTotalLength: Annotated[uint16_t, 2]
  bNumInterfaces: Annotated[uint8_t, 4]
  bConfigurationValue: Annotated[uint8_t, 5]
  iConfiguration: Annotated[uint8_t, 6]
  bmAttributes: Annotated[uint8_t, 7]
  MaxPower: Annotated[uint8_t, 8]
  interface: Annotated[ctypes.POINTER(struct_libusb_interface), 16]
  extra: Annotated[ctypes.POINTER(ctypes.c_ubyte), 24]
  extra_length: Annotated[ctypes.c_int32, 32]
@record
class struct_libusb_ss_endpoint_companion_descriptor:
  SIZE = 6
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bMaxBurst: Annotated[uint8_t, 2]
  bmAttributes: Annotated[uint8_t, 3]
  wBytesPerInterval: Annotated[uint16_t, 4]
@record
class struct_libusb_bos_dev_capability_descriptor:
  SIZE = 3
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bDevCapabilityType: Annotated[uint8_t, 2]
  dev_capability_data: Annotated[(uint8_t * 0), 3]
@record
class struct_libusb_bos_descriptor:
  SIZE = 8
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  wTotalLength: Annotated[uint16_t, 2]
  bNumDeviceCaps: Annotated[uint8_t, 4]
  dev_capability: Annotated[(ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor) * 0), 8]
@record
class struct_libusb_usb_2_0_extension_descriptor:
  SIZE = 8
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bDevCapabilityType: Annotated[uint8_t, 2]
  bmAttributes: Annotated[uint32_t, 4]
uint32_t = ctypes.c_uint32
@record
class struct_libusb_ss_usb_device_capability_descriptor:
  SIZE = 10
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bDevCapabilityType: Annotated[uint8_t, 2]
  bmAttributes: Annotated[uint8_t, 3]
  wSpeedSupported: Annotated[uint16_t, 4]
  bFunctionalitySupport: Annotated[uint8_t, 6]
  bU1DevExitLat: Annotated[uint8_t, 7]
  bU2DevExitLat: Annotated[uint16_t, 8]
@record
class struct_libusb_container_id_descriptor:
  SIZE = 20
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bDevCapabilityType: Annotated[uint8_t, 2]
  bReserved: Annotated[uint8_t, 3]
  ContainerID: Annotated[(uint8_t* 16), 4]
@record
class struct_libusb_platform_descriptor:
  SIZE = 20
  bLength: Annotated[uint8_t, 0]
  bDescriptorType: Annotated[uint8_t, 1]
  bDevCapabilityType: Annotated[uint8_t, 2]
  bReserved: Annotated[uint8_t, 3]
  PlatformCapabilityUUID: Annotated[(uint8_t* 16), 4]
  CapabilityData: Annotated[(uint8_t * 0), 20]
@record
class struct_libusb_control_setup:
  SIZE = 8
  bmRequestType: Annotated[uint8_t, 0]
  bRequest: Annotated[uint8_t, 1]
  wValue: Annotated[uint16_t, 2]
  wIndex: Annotated[uint16_t, 4]
  wLength: Annotated[uint16_t, 6]
class struct_libusb_context(ctypes.Structure): pass
class struct_libusb_device(ctypes.Structure): pass
class struct_libusb_device_handle(ctypes.Structure): pass
@record
class struct_libusb_version:
  SIZE = 24
  major: Annotated[uint16_t, 0]
  minor: Annotated[uint16_t, 2]
  micro: Annotated[uint16_t, 4]
  nano: Annotated[uint16_t, 6]
  rc: Annotated[ctypes.POINTER(ctypes.c_char), 8]
  describe: Annotated[ctypes.POINTER(ctypes.c_char), 16]
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

@record
class struct_libusb_iso_packet_descriptor:
  SIZE = 12
  length: Annotated[ctypes.c_uint32, 0]
  actual_length: Annotated[ctypes.c_uint32, 4]
  status: Annotated[enum_libusb_transfer_status, 8]
class struct_libusb_transfer(ctypes.Structure): pass
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

class struct_libusb_init_option(ctypes.Structure): pass
class _anonunion0(ctypes.Union): pass
@dll.bind
def libusb_init(ctx:ctypes.POINTER(ctypes.POINTER(libusb_context))) -> ctypes.c_int32: ...
@dll.bind
def libusb_init_context(ctx:ctypes.POINTER(ctypes.POINTER(libusb_context)), options:(struct_libusb_init_option * 0), num_options:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_exit(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@dll.bind
def libusb_set_debug(ctx:ctypes.POINTER(libusb_context), level:ctypes.c_int32) -> None: ...
@dll.bind
def libusb_get_version() -> ctypes.POINTER(struct_libusb_version): ...
@dll.bind
def libusb_has_capability(capability:uint32_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_error_name(errcode:ctypes.c_int32) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def libusb_setlocale(locale:ctypes.POINTER(ctypes.c_char)) -> ctypes.c_int32: ...
@dll.bind
def libusb_strerror(errcode:ctypes.c_int32) -> ctypes.POINTER(ctypes.c_char): ...
ssize_t = ctypes.c_int64
@dll.bind
def libusb_get_device_list(ctx:ctypes.POINTER(libusb_context), list:ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(libusb_device)))) -> ssize_t: ...
@dll.bind
def libusb_free_device_list(list:ctypes.POINTER(ctypes.POINTER(libusb_device)), unref_devices:ctypes.c_int32) -> None: ...
@dll.bind
def libusb_ref_device(dev:ctypes.POINTER(libusb_device)) -> ctypes.POINTER(libusb_device): ...
@dll.bind
def libusb_unref_device(dev:ctypes.POINTER(libusb_device)) -> None: ...
@dll.bind
def libusb_get_configuration(dev:ctypes.POINTER(libusb_device_handle), config:ctypes.POINTER(ctypes.c_int32)) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_device_descriptor(dev:ctypes.POINTER(libusb_device), desc:ctypes.POINTER(struct_libusb_device_descriptor)) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_active_config_descriptor(dev:ctypes.POINTER(libusb_device), config:ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_config_descriptor(dev:ctypes.POINTER(libusb_device), config_index:uint8_t, config:ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_config_descriptor_by_value(dev:ctypes.POINTER(libusb_device), bConfigurationValue:uint8_t, config:ctypes.POINTER(ctypes.POINTER(struct_libusb_config_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_config_descriptor(config:ctypes.POINTER(struct_libusb_config_descriptor)) -> None: ...
@dll.bind
def libusb_get_ss_endpoint_companion_descriptor(ctx:ctypes.POINTER(libusb_context), endpoint:ctypes.POINTER(struct_libusb_endpoint_descriptor), ep_comp:ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_ss_endpoint_companion_descriptor(ep_comp:ctypes.POINTER(struct_libusb_ss_endpoint_companion_descriptor)) -> None: ...
@dll.bind
def libusb_get_bos_descriptor(dev_handle:ctypes.POINTER(libusb_device_handle), bos:ctypes.POINTER(ctypes.POINTER(struct_libusb_bos_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_bos_descriptor(bos:ctypes.POINTER(struct_libusb_bos_descriptor)) -> None: ...
@dll.bind
def libusb_get_usb_2_0_extension_descriptor(ctx:ctypes.POINTER(libusb_context), dev_cap:ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), usb_2_0_extension:ctypes.POINTER(ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_usb_2_0_extension_descriptor(usb_2_0_extension:ctypes.POINTER(struct_libusb_usb_2_0_extension_descriptor)) -> None: ...
@dll.bind
def libusb_get_ss_usb_device_capability_descriptor(ctx:ctypes.POINTER(libusb_context), dev_cap:ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), ss_usb_device_cap:ctypes.POINTER(ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_ss_usb_device_capability_descriptor(ss_usb_device_cap:ctypes.POINTER(struct_libusb_ss_usb_device_capability_descriptor)) -> None: ...
@dll.bind
def libusb_get_container_id_descriptor(ctx:ctypes.POINTER(libusb_context), dev_cap:ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), container_id:ctypes.POINTER(ctypes.POINTER(struct_libusb_container_id_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_container_id_descriptor(container_id:ctypes.POINTER(struct_libusb_container_id_descriptor)) -> None: ...
@dll.bind
def libusb_get_platform_descriptor(ctx:ctypes.POINTER(libusb_context), dev_cap:ctypes.POINTER(struct_libusb_bos_dev_capability_descriptor), platform_descriptor:ctypes.POINTER(ctypes.POINTER(struct_libusb_platform_descriptor))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_platform_descriptor(platform_descriptor:ctypes.POINTER(struct_libusb_platform_descriptor)) -> None: ...
@dll.bind
def libusb_get_bus_number(dev:ctypes.POINTER(libusb_device)) -> uint8_t: ...
@dll.bind
def libusb_get_port_number(dev:ctypes.POINTER(libusb_device)) -> uint8_t: ...
@dll.bind
def libusb_get_port_numbers(dev:ctypes.POINTER(libusb_device), port_numbers:ctypes.POINTER(uint8_t), port_numbers_len:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_port_path(ctx:ctypes.POINTER(libusb_context), dev:ctypes.POINTER(libusb_device), path:ctypes.POINTER(uint8_t), path_length:uint8_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_parent(dev:ctypes.POINTER(libusb_device)) -> ctypes.POINTER(libusb_device): ...
@dll.bind
def libusb_get_device_address(dev:ctypes.POINTER(libusb_device)) -> uint8_t: ...
@dll.bind
def libusb_get_device_speed(dev:ctypes.POINTER(libusb_device)) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_packet_size(dev:ctypes.POINTER(libusb_device), endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_iso_packet_size(dev:ctypes.POINTER(libusb_device), endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_max_alt_packet_size(dev:ctypes.POINTER(libusb_device), interface_number:ctypes.c_int32, alternate_setting:ctypes.c_int32, endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_interface_association_descriptors(dev:ctypes.POINTER(libusb_device), config_index:uint8_t, iad_array:ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array))) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_active_interface_association_descriptors(dev:ctypes.POINTER(libusb_device), iad_array:ctypes.POINTER(ctypes.POINTER(struct_libusb_interface_association_descriptor_array))) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_interface_association_descriptors(iad_array:ctypes.POINTER(struct_libusb_interface_association_descriptor_array)) -> None: ...
intptr_t = ctypes.c_int64
@dll.bind
def libusb_wrap_sys_device(ctx:ctypes.POINTER(libusb_context), sys_dev:intptr_t, dev_handle:ctypes.POINTER(ctypes.POINTER(libusb_device_handle))) -> ctypes.c_int32: ...
@dll.bind
def libusb_open(dev:ctypes.POINTER(libusb_device), dev_handle:ctypes.POINTER(ctypes.POINTER(libusb_device_handle))) -> ctypes.c_int32: ...
@dll.bind
def libusb_close(dev_handle:ctypes.POINTER(libusb_device_handle)) -> None: ...
@dll.bind
def libusb_get_device(dev_handle:ctypes.POINTER(libusb_device_handle)) -> ctypes.POINTER(libusb_device): ...
@dll.bind
def libusb_set_configuration(dev_handle:ctypes.POINTER(libusb_device_handle), configuration:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_claim_interface(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_release_interface(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_open_device_with_vid_pid(ctx:ctypes.POINTER(libusb_context), vendor_id:uint16_t, product_id:uint16_t) -> ctypes.POINTER(libusb_device_handle): ...
@dll.bind
def libusb_set_interface_alt_setting(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32, alternate_setting:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_clear_halt(dev_handle:ctypes.POINTER(libusb_device_handle), endpoint:ctypes.c_ubyte) -> ctypes.c_int32: ...
@dll.bind
def libusb_reset_device(dev_handle:ctypes.POINTER(libusb_device_handle)) -> ctypes.c_int32: ...
@dll.bind
def libusb_alloc_streams(dev_handle:ctypes.POINTER(libusb_device_handle), num_streams:uint32_t, endpoints:ctypes.POINTER(ctypes.c_ubyte), num_endpoints:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_streams(dev_handle:ctypes.POINTER(libusb_device_handle), endpoints:ctypes.POINTER(ctypes.c_ubyte), num_endpoints:ctypes.c_int32) -> ctypes.c_int32: ...
size_t = ctypes.c_uint64
@dll.bind
def libusb_dev_mem_alloc(dev_handle:ctypes.POINTER(libusb_device_handle), length:size_t) -> ctypes.POINTER(ctypes.c_ubyte): ...
@dll.bind
def libusb_dev_mem_free(dev_handle:ctypes.POINTER(libusb_device_handle), buffer:ctypes.POINTER(ctypes.c_ubyte), length:size_t) -> ctypes.c_int32: ...
@dll.bind
def libusb_kernel_driver_active(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_detach_kernel_driver(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_attach_kernel_driver(dev_handle:ctypes.POINTER(libusb_device_handle), interface_number:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_set_auto_detach_kernel_driver(dev_handle:ctypes.POINTER(libusb_device_handle), enable:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_alloc_transfer(iso_packets:ctypes.c_int32) -> ctypes.POINTER(struct_libusb_transfer): ...
@dll.bind
def libusb_submit_transfer(transfer:ctypes.POINTER(struct_libusb_transfer)) -> ctypes.c_int32: ...
@dll.bind
def libusb_cancel_transfer(transfer:ctypes.POINTER(struct_libusb_transfer)) -> ctypes.c_int32: ...
@dll.bind
def libusb_free_transfer(transfer:ctypes.POINTER(struct_libusb_transfer)) -> None: ...
@dll.bind
def libusb_transfer_set_stream_id(transfer:ctypes.POINTER(struct_libusb_transfer), stream_id:uint32_t) -> None: ...
@dll.bind
def libusb_transfer_get_stream_id(transfer:ctypes.POINTER(struct_libusb_transfer)) -> uint32_t: ...
@dll.bind
def libusb_control_transfer(dev_handle:ctypes.POINTER(libusb_device_handle), request_type:uint8_t, bRequest:uint8_t, wValue:uint16_t, wIndex:uint16_t, data:ctypes.POINTER(ctypes.c_ubyte), wLength:uint16_t, timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_bulk_transfer(dev_handle:ctypes.POINTER(libusb_device_handle), endpoint:ctypes.c_ubyte, data:ctypes.POINTER(ctypes.c_ubyte), length:ctypes.c_int32, actual_length:ctypes.POINTER(ctypes.c_int32), timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_interrupt_transfer(dev_handle:ctypes.POINTER(libusb_device_handle), endpoint:ctypes.c_ubyte, data:ctypes.POINTER(ctypes.c_ubyte), length:ctypes.c_int32, actual_length:ctypes.POINTER(ctypes.c_int32), timeout:ctypes.c_uint32) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_string_descriptor_ascii(dev_handle:ctypes.POINTER(libusb_device_handle), desc_index:uint8_t, data:ctypes.POINTER(ctypes.c_ubyte), length:ctypes.c_int32) -> ctypes.c_int32: ...
@dll.bind
def libusb_try_lock_events(ctx:ctypes.POINTER(libusb_context)) -> ctypes.c_int32: ...
@dll.bind
def libusb_lock_events(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@dll.bind
def libusb_unlock_events(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@dll.bind
def libusb_event_handling_ok(ctx:ctypes.POINTER(libusb_context)) -> ctypes.c_int32: ...
@dll.bind
def libusb_event_handler_active(ctx:ctypes.POINTER(libusb_context)) -> ctypes.c_int32: ...
@dll.bind
def libusb_interrupt_event_handler(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@dll.bind
def libusb_lock_event_waiters(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@dll.bind
def libusb_unlock_event_waiters(ctx:ctypes.POINTER(libusb_context)) -> None: ...
@record
class struct_timeval:
  SIZE = 16
  tv_sec: Annotated[ctypes.c_int64, 0]
  tv_usec: Annotated[ctypes.c_int64, 8]
__time_t = ctypes.c_int64
__suseconds_t = ctypes.c_int64
@dll.bind
def libusb_wait_for_event(ctx:ctypes.POINTER(libusb_context), tv:ctypes.POINTER(struct_timeval)) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_timeout(ctx:ctypes.POINTER(libusb_context), tv:ctypes.POINTER(struct_timeval)) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_timeout_completed(ctx:ctypes.POINTER(libusb_context), tv:ctypes.POINTER(struct_timeval), completed:ctypes.POINTER(ctypes.c_int32)) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events(ctx:ctypes.POINTER(libusb_context)) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_completed(ctx:ctypes.POINTER(libusb_context), completed:ctypes.POINTER(ctypes.c_int32)) -> ctypes.c_int32: ...
@dll.bind
def libusb_handle_events_locked(ctx:ctypes.POINTER(libusb_context), tv:ctypes.POINTER(struct_timeval)) -> ctypes.c_int32: ...
@dll.bind
def libusb_pollfds_handle_timeouts(ctx:ctypes.POINTER(libusb_context)) -> ctypes.c_int32: ...
@dll.bind
def libusb_get_next_timeout(ctx:ctypes.POINTER(libusb_context), tv:ctypes.POINTER(struct_timeval)) -> ctypes.c_int32: ...
@record
class struct_libusb_pollfd:
  SIZE = 8
  fd: Annotated[ctypes.c_int32, 0]
  events: Annotated[ctypes.c_int16, 4]
@dll.bind
def libusb_get_pollfds(ctx:ctypes.POINTER(libusb_context)) -> ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd)): ...
@dll.bind
def libusb_free_pollfds(pollfds:ctypes.POINTER(ctypes.POINTER(struct_libusb_pollfd))) -> None: ...
libusb_hotplug_callback_handle = ctypes.c_int32
libusb_hotplug_event = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED', 1)
LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT = libusb_hotplug_event.define('LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT', 2)

libusb_hotplug_flag = CEnum(ctypes.c_uint32)
LIBUSB_HOTPLUG_ENUMERATE = libusb_hotplug_flag.define('LIBUSB_HOTPLUG_ENUMERATE', 1)

@dll.bind
def libusb_hotplug_deregister_callback(ctx:ctypes.POINTER(libusb_context), callback_handle:libusb_hotplug_callback_handle) -> None: ...
@dll.bind
def libusb_hotplug_get_user_data(ctx:ctypes.POINTER(libusb_context), callback_handle:libusb_hotplug_callback_handle) -> ctypes.POINTER(None): ...
@dll.bind
def libusb_set_option(ctx:ctypes.POINTER(libusb_context), option:enum_libusb_option) -> ctypes.c_int32: ...
init_records()
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