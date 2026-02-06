# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class DLPackVersion(Struct): pass
uint32_t = ctypes.c_uint32
DLPackVersion._fields_ = [
  ('major', uint32_t),
  ('minor', uint32_t),
]
DLDeviceType = CEnum(ctypes.c_uint32)
kDLCPU = DLDeviceType.define('kDLCPU', 1)
kDLCUDA = DLDeviceType.define('kDLCUDA', 2)
kDLCUDAHost = DLDeviceType.define('kDLCUDAHost', 3)
kDLOpenCL = DLDeviceType.define('kDLOpenCL', 4)
kDLVulkan = DLDeviceType.define('kDLVulkan', 7)
kDLMetal = DLDeviceType.define('kDLMetal', 8)
kDLVPI = DLDeviceType.define('kDLVPI', 9)
kDLROCM = DLDeviceType.define('kDLROCM', 10)
kDLROCMHost = DLDeviceType.define('kDLROCMHost', 11)
kDLExtDev = DLDeviceType.define('kDLExtDev', 12)
kDLCUDAManaged = DLDeviceType.define('kDLCUDAManaged', 13)
kDLOneAPI = DLDeviceType.define('kDLOneAPI', 14)
kDLWebGPU = DLDeviceType.define('kDLWebGPU', 15)
kDLHexagon = DLDeviceType.define('kDLHexagon', 16)
kDLMAIA = DLDeviceType.define('kDLMAIA', 17)
kDLTrn = DLDeviceType.define('kDLTrn', 18)

class DLDevice(Struct): pass
int32_t = ctypes.c_int32
DLDevice._fields_ = [
  ('device_type', DLDeviceType),
  ('device_id', int32_t),
]
DLDataTypeCode = CEnum(ctypes.c_uint32)
kDLInt = DLDataTypeCode.define('kDLInt', 0)
kDLUInt = DLDataTypeCode.define('kDLUInt', 1)
kDLFloat = DLDataTypeCode.define('kDLFloat', 2)
kDLOpaqueHandle = DLDataTypeCode.define('kDLOpaqueHandle', 3)
kDLBfloat = DLDataTypeCode.define('kDLBfloat', 4)
kDLComplex = DLDataTypeCode.define('kDLComplex', 5)
kDLBool = DLDataTypeCode.define('kDLBool', 6)
kDLFloat8_e3m4 = DLDataTypeCode.define('kDLFloat8_e3m4', 7)
kDLFloat8_e4m3 = DLDataTypeCode.define('kDLFloat8_e4m3', 8)
kDLFloat8_e4m3b11fnuz = DLDataTypeCode.define('kDLFloat8_e4m3b11fnuz', 9)
kDLFloat8_e4m3fn = DLDataTypeCode.define('kDLFloat8_e4m3fn', 10)
kDLFloat8_e4m3fnuz = DLDataTypeCode.define('kDLFloat8_e4m3fnuz', 11)
kDLFloat8_e5m2 = DLDataTypeCode.define('kDLFloat8_e5m2', 12)
kDLFloat8_e5m2fnuz = DLDataTypeCode.define('kDLFloat8_e5m2fnuz', 13)
kDLFloat8_e8m0fnu = DLDataTypeCode.define('kDLFloat8_e8m0fnu', 14)
kDLFloat6_e2m3fn = DLDataTypeCode.define('kDLFloat6_e2m3fn', 15)
kDLFloat6_e3m2fn = DLDataTypeCode.define('kDLFloat6_e3m2fn', 16)
kDLFloat4_e2m1fn = DLDataTypeCode.define('kDLFloat4_e2m1fn', 17)

class DLDataType(Struct): pass
uint8_t = ctypes.c_ubyte
uint16_t = ctypes.c_uint16
DLDataType._fields_ = [
  ('code', uint8_t),
  ('bits', uint8_t),
  ('lanes', uint16_t),
]
class DLTensor(Struct): pass
int64_t = ctypes.c_int64
uint64_t = ctypes.c_uint64
DLTensor._fields_ = [
  ('data', ctypes.c_void_p),
  ('device', DLDevice),
  ('ndim', int32_t),
  ('dtype', DLDataType),
  ('shape', ctypes.POINTER(int64_t)),
  ('strides', ctypes.POINTER(int64_t)),
  ('byte_offset', uint64_t),
]
class struct_DLManagedTensor(Struct): pass
struct_DLManagedTensor._fields_ = [
  ('dl_tensor', DLTensor),
  ('manager_ctx', ctypes.c_void_p),
  ('deleter', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_DLManagedTensor))),
]
DLManagedTensor = struct_DLManagedTensor
class struct_DLManagedTensorVersioned(Struct): pass
struct_DLManagedTensorVersioned._fields_ = [
  ('version', DLPackVersion),
  ('manager_ctx', ctypes.c_void_p),
  ('deleter', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_DLManagedTensorVersioned))),
  ('flags', uint64_t),
  ('dl_tensor', DLTensor),
]
DLManagedTensorVersioned = struct_DLManagedTensorVersioned
DLPackManagedTensorAllocator = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(DLTensor), ctypes.POINTER(ctypes.POINTER(struct_DLManagedTensorVersioned)), ctypes.c_void_p, ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)))
DLPackManagedTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_DLManagedTensorVersioned)))
DLPackDLTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(DLTensor))
DLPackCurrentWorkStream = ctypes.CFUNCTYPE(ctypes.c_int32, DLDeviceType, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p))
DLPackManagedTensorToPyObjectNoSync = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_DLManagedTensorVersioned), ctypes.POINTER(ctypes.c_void_p))
class struct_DLPackExchangeAPIHeader(Struct): pass
struct_DLPackExchangeAPIHeader._fields_ = [
  ('version', DLPackVersion),
  ('prev_api', ctypes.POINTER(struct_DLPackExchangeAPIHeader)),
]
DLPackExchangeAPIHeader = struct_DLPackExchangeAPIHeader
class struct_DLPackExchangeAPI(Struct): pass
struct_DLPackExchangeAPI._fields_ = [
  ('header', DLPackExchangeAPIHeader),
  ('managed_tensor_allocator', DLPackManagedTensorAllocator),
  ('managed_tensor_from_py_object_no_sync', DLPackManagedTensorFromPyObjectNoSync),
  ('managed_tensor_to_py_object_no_sync', DLPackManagedTensorToPyObjectNoSync),
  ('dltensor_from_py_object_no_sync', DLPackDLTensorFromPyObjectNoSync),
  ('current_work_stream', DLPackCurrentWorkStream),
]
DLPackExchangeAPI = struct_DLPackExchangeAPI
DLPACK_MAJOR_VERSION = 1
DLPACK_MINOR_VERSION = 2
DLPACK_FLAG_BITMASK_READ_ONLY = (1 << 0)
DLPACK_FLAG_BITMASK_IS_COPIED = (1 << 1)
DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED = (1 << 2)