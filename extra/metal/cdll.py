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
    os.environ.get("METAL", "/Library/Frameworks/Metal.framework/Metal")
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


def _send_message(
    sender: CDLL._FuncPtr, instance_ptr: objc_id, selector: str, *args: Any
):
    return sender(
        instance_ptr,
        libobjc.sel_registerName(selector.encode()),
        *args,
    )


def send_message(
    ptr: objc_id, selector: str, /, *args: Any, restype: type = objc_id
) -> objc_id:
    sender = libobjc.objc_msgSend if restype == objc_id else libobjc["objc_msgSend"]
    return sender(ptr, libobjc.sel_registerName(selector.encode()), *args)


def send_message_with_return_type(
    restype: type, ptr: objc_id, selector: str, *args: Any
) -> Any:
    sender = libobjc["objc_msgSend"]
    sender.restype = bytes
    return _send_message(
        sender,
        ptr,
        selector,
        *args,
    )


NSString: objc_id = libobjc.objc_getClass(b"NSString")
NSConcreteData: objc_id = libobjc.objc_getClass(b"NSConcreteData")


def to_ns_str(s: str) -> objc_id:
    return send_message(NSString, "stringWithUTF8String:", s.encode())


def int_tuple_to_struct(t: tuple[int, ...]):
    class Struct(Structure):
        pass

    Struct._fields_ = [(f"field{i}", c_ulong) for i in range(len(t))]
    return Struct(*t)
