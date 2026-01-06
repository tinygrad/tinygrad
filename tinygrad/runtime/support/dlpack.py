# DLPack support for tinygrad
# See: https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

import ctypes
from typing import Any
from tinygrad.dtype import DType, dtypes
from tinygrad.runtime.autogen.dlpack import (DLDataType, DLManagedTensorVersioned,
                                             kDLCPU, kDLCUDA, kDLROCM, kDLMetal, kDLOpenCL, kDLWebGPU,
                                             kDLInt, kDLUInt, kDLFloat, kDLBfloat, kDLBool)

# Capsule names for DLPack 1.0+
DLPACK_CAPSULE_NAME = b"dltensor_versioned"
DLPACK_CAPSULE_USED_NAME = b"used_dltensor_versioned"

# Mapping from tinygrad device string prefix to DLPack device type
TINYGRAD_TO_DLPACK_DEVICE: dict[str, int] = {
  "CPU": kDLCPU, "CUDA": kDLCUDA, "NV": kDLCUDA, "HIP": kDLROCM, "AMD": kDLROCM,
  "METAL": kDLMetal, "CL": kDLOpenCL, "QCOM": kDLOpenCL, "WEBGPU": kDLWebGPU,
}

# Deleter callback type for use as decorator (must match autogen struct's deleter signature)
DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))

# PyCapsule destructor callback type
PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# PyCapsule APIs
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
pycapsule_new = ctypes.pythonapi.PyCapsule_New
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
pycapsule_isvalid = ctypes.pythonapi.PyCapsule_IsValid
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
pycapsule_getpointer = ctypes.pythonapi.PyCapsule_GetPointer

@PyCapsuleDestructor
def dlpack_capsule_deleter(capsule_ptr):
  capsule = ctypes.cast(capsule_ptr, ctypes.py_object)
  if pycapsule_isvalid(capsule, DLPACK_CAPSULE_USED_NAME):
    return
  managed_ptr = pycapsule_getpointer(capsule, DLPACK_CAPSULE_NAME)
  if not managed_ptr:
    return
  managed = ctypes.cast(managed_ptr, ctypes.POINTER(DLManagedTensorVersioned))
  if managed.contents.deleter:
    managed.contents.deleter(managed)

def get_dlpack_device(device: str) -> tuple[int, int]:
  """Convert tinygrad device string to DLPack device tuple (device_type, device_id)."""
  parts = device.split(":")
  device_type_str = parts[0].upper()
  device_id = int(parts[1]) if len(parts) > 1 else 0
  if device_type_str not in TINYGRAD_TO_DLPACK_DEVICE:
    raise RuntimeError(f"Device '{device_type_str}' is not supported for DLPack export")
  return (TINYGRAD_TO_DLPACK_DEVICE[device_type_str], device_id)

def dtype_to_dl_datatype(dtype: DType) -> DLDataType:
  """Convert tinygrad DType to DLDataType."""
  dtype_map: dict[DType, tuple[int, int, int]] = {
    dtypes.bool: (kDLBool, 8, 1), dtypes.int8: (kDLInt, 8, 1), dtypes.int16: (kDLInt, 16, 1),
    dtypes.int32: (kDLInt, 32, 1), dtypes.int64: (kDLInt, 64, 1), dtypes.uint8: (kDLUInt, 8, 1),
    dtypes.uint16: (kDLUInt, 16, 1), dtypes.uint32: (kDLUInt, 32, 1), dtypes.uint64: (kDLUInt, 64, 1),
    dtypes.float16: (kDLFloat, 16, 1), dtypes.float32: (kDLFloat, 32, 1), dtypes.float64: (kDLFloat, 64, 1),
    dtypes.bfloat16: (kDLBfloat, 16, 1),
  }
  if dtype not in dtype_map:
    raise RuntimeError(f"DType '{dtype}' is not supported for DLPack export")
  code, bits, lanes = dtype_map[dtype]
  return DLDataType(code=code, bits=bits, lanes=lanes)

class DLPackContext:
  """Context object to prevent garbage collection of tensor data while DLPack capsule is in use."""
  def __init__(self, tensor: Any, buffer: Any, shape_array: Any, strides_array: Any = None):
    self.tensor = tensor
    self.buffer = buffer
    self.shape = shape_array
    self.strides = strides_array
    self._prevent_deleter_gc: Any = None  # Prevent deleter callback from being GC'd
    self._prevent_managed_gc: Any = None  # Prevent managed struct from being GC'd
