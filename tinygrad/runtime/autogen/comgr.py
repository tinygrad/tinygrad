# mypy: ignore-errors
import ctypes, os
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
def dll():
  try: return ctypes.CDLL(unwrap(os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamd_comgr.so'))
  except: pass
  try: return ctypes.CDLL(unwrap('/usr/local/lib/libamd_comgr.dylib'))
  except: pass
  try: return ctypes.CDLL(unwrap('/opt/homebrew/lib/libamd_comgr.dylib'))
  except: pass
  return None
dll = dll()

AMD_COMGR_DEPRECATED = lambda msg: __attribute__((deprecated(msg)))
AMD_COMGR_INTERFACE_VERSION_MAJOR = 2
AMD_COMGR_INTERFACE_VERSION_MINOR = 8