import os
from ctypes import CDLL, c_void_p, c_char_p, Structure, c_ulong
from ctypes.macholib.dyld import dyld_find  # pyright: ignore
from typing import cast, Any


class objc_id(c_void_p):
    pass


def load_library(path: str):
    return CDLL(
        cast(
            str,
            dyld_find(path),
        )
    )


libobjc = load_library(os.environ.get("LIBOBJC", "/usr/lib/libobjc.dylib"))
libobjc.objc_msgSend.restype = objc_id
libobjc.objc_getClass.restype = objc_id
libobjc.objc_getClass.argtypes = [c_char_p]
libobjc.sel_registerName.restype = objc_id
libobjc.sel_registerName.argtypes = [c_char_p]

metal = load_library(
    os.environ.get("METALLIBPATH", "/Library/Frameworks/Metal.framework/Metal")
)
metal.MTLCreateSystemDefaultDevice.restype = objc_id

core_graphics = load_library(
    os.environ.get(
        "CORE_GRAPHICS",
        "/Library/Frameworks/CoreGraphics.framework/CoreGraphics",
    )
)
libsystem = load_library(
    os.environ.get(
        "LIBSYSTEM",
        "/usr/lib/libSystem.dylib",
    )
)
libdispatch = libsystem
libdispatch.dispatch_data_create.restype = objc_id


def send_message(
    ptr: objc_id, selector: str, /, *args: Any, restype: type = objc_id
) -> objc_id:
    sender = libobjc.objc_msgSend if restype == objc_id else libobjc["objc_msgSend"]
    return sender(ptr, libobjc.sel_registerName(selector.encode()), *args)


NSString: objc_id = libobjc.objc_getClass(b"NSString")
NSConcreteData: objc_id = libobjc.objc_getClass(b"NSConcreteData")
NSData: objc_id = libobjc.objc_getClass(b"NSData")
NSMutableArray: objc_id = libobjc.objc_getClass(b"NSMutableArray")

def to_ns_str(s: str) -> objc_id:
    return send_message(NSString, "stringWithUTF8String:", s.encode())


def to_ns_data(bytes: bytes) -> objc_id:
    return send_message(NSData, "dataWithBytes:length:", bytes, len(bytes))

def to_ns_array(items: list[Any]) -> objc_id:
    c_array = (objc_id * len(items))(*items)
    return c_array

def int_tuple_to_struct(t: tuple[int, ...], _type: type = c_ulong):
    class Struct(Structure):
        pass

    Struct._fields_ = [(f"field{i}", c_ulong) for i in range(len(t))]
    return Struct(*t)
