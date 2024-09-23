import ctypes
from typing import Any, List, Tuple, cast

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return cast(int, self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self): msg(self, "release")

def load_library(path: str): return ctypes.CDLL(path)

libobjc = load_library("/usr/lib/libobjc.dylib")
libmetal = load_library("/Library/Frameworks/Metal.framework/Metal")
# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
load_library("/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
libdispatch = load_library("/usr/lib/libSystem.dylib")
libobjc.objc_msgSend.restype = objc_id
libobjc.objc_getClass.restype = objc_id
libobjc.sel_registerName.restype = objc_id
libmetal.MTLCreateSystemDefaultDevice.restype = objc_instance
libdispatch.dispatch_data_create.restype = objc_instance

def msg(ptr: objc_instance, selector: str, /, *args: Any, restype: type = objc_id) -> Any:
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  return sender(ptr, libobjc.sel_registerName(selector.encode()), *args)

def to_ns_str(s: str) -> objc_instance: return msg(libobjc.objc_getClass(b"NSString"), "stringWithUTF8String:", s.encode())

def to_ns_array(items: List[Any]): return (objc_instance * len(items))(*items)

def int_tuple_to_struct(t: Tuple[int, ...], _type: type = ctypes.c_ulong):
  class Struct(ctypes.Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

class MTLIndirectCommandType:
  MTLIndirectCommandTypeConcurrentDispatch = 32 # (1 << 5)
  MTLResourceCPUCacheModeDefaultCache = 0

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0

class MTLResourceUsage:
  MTLResourceUsageRead = 0b01
  MTLResourceUsageWrite = 0b10

