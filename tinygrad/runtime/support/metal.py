from ctypes import CDLL, c_void_p, c_char_p, Structure, c_ulong
from ctypes.macholib.dyld import dyld_find  # pyright: ignore
from typing import cast, Any

class objc_id(c_void_p): pass

def load_library(path: str): return CDLL(cast(str,dyld_find(path)))

libobjc = load_library("/usr/lib/libobjc.dylib")
libmetal = load_library("/Library/Frameworks/Metal.framework/Metal")
core_graphics = load_library("/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
libdispatch = load_library("/usr/lib/libSystem.dylib")
libobjc.objc_msgSend.restype = objc_id
libobjc.objc_getClass.restype = objc_id
libobjc.objc_getClass.argtypes = [c_char_p]
libobjc.sel_registerName.restype = objc_id
libobjc.sel_registerName.argtypes = [c_char_p]
libmetal.MTLCreateSystemDefaultDevice.restype = objc_id
libdispatch.dispatch_data_create.restype = objc_id
NSString: objc_id = libobjc.objc_getClass(b"NSString")

def send_message(ptr: objc_id, selector: str, /, *args: Any, restype: type = objc_id) -> Any:
  sender = libobjc["objc_msgSend"]
  sender.restype = restype
  return sender(ptr, libobjc.sel_registerName(selector.encode()), *args)

def to_ns_str(s: str) -> objc_id: return send_message(NSString, "stringWithUTF8String:", s.encode())

def to_ns_array(items: list[Any]): return (objc_id * len(items))(*items)

def int_tuple_to_struct(t: tuple[int, ...], _type: type = c_ulong):
  class Struct(Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

def dedup(items: list[objc_id]) -> list[objc_id]:
  ret = []
  seen = set()
  for item in items:
    if item.value not in seen:
      seen.add(item.value)
      ret.append(item)
  return ret