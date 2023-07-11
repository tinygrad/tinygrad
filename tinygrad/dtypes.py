from tinygrad.DType import DType
from typing import Final, Dict
import inspect
import numpy as np
class dtypes:

  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.int64)
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes._half4, dtypes._float4)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bool: Final[DType] = DType(0, 1, "bool", bool)
  float16: Final[DType] = DType(0, 2, "half", np.float16)
  half = float16
  float32: Final[DType] = DType(4, 4, "float", np.float32)
  float = float32
  int8: Final[DType] = DType(0, 1, "char", np.int8)
  int32: Final[DType] = DType(1, 4, "int", np.int32)
  int64: Final[DType] = DType(2, 8, "long", np.int64)
  uint8: Final[DType] = DType(0, 1, "unsigned char", np.uint8)
  uint32: Final[DType] = DType(1, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(2, 8, "unsigned long", np.uint64)

  # NOTE: these are internal dtypes, should probably check for that
  _half4: Final[DType] = DType(0, 2*4, "half4", None, 4)
  _float2: Final[DType] = DType(4, 4*2, "float2", None, 2)
  _float4: Final[DType] = DType(4, 4*4, "float4", None, 4)

# dtypes.__dict__.items()
# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in inspect.getmembers(dtypes()) if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}
