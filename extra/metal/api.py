"""Emulate the pyobjc API"""

from extra.metal.cdll import metal, send_message, libobjc, objc_id, to_ns_str
from extra.metal.cdll import libdispatch as _libdispatch
from typing import Optional, cast, Any
from ctypes import c_ulong, string_at


class MTLCompileOptions:
    @classmethod
    def new(cls):
        return cls(
            send_message(
                libobjc.objc_getClass(b"MTLCompileOptions"),
                "new",
            )
        )

    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def setFastMathEnabled_(self, enabled: bool):
        send_message(self.ptr, "setFastMathEnabled:", enabled)


class Encoder:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(self):
        pass

    def setComputePipelineState_(self, pipeline_state: objc_id):
        send_message(self.ptr, "setComputePipelineState:", pipeline_state)


class CommandBuffer:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def computeCommandEncoder(self):
        return Encoder(send_message(self.ptr, "computeCommandEncoder"))

    def blitCommandEncoder(self):
        return Encoder(send_message(self.ptr, "blitCommandEncoder"))


class MtlQueue:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def commandBuffer(self):
        return CommandBuffer(send_message(self.ptr, "commandBuffer"))


class LibraryDataContentsBytes:
    def __init__(self, ptr: objc_id, length: int):
        self.ptr = ptr
        self.length = length

    def tobytes(self):
        return string_at(self.ptr, self.length)


class LibraryDataContents:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def bytes(self):
        bytes_ptr = send_message(self.ptr, "bytes")
        length = cast(int, send_message(self.ptr, "length", restype=c_ulong))
        return LibraryDataContentsBytes(bytes_ptr, length)


class Library:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def libraryDataContents(self):
        return LibraryDataContents(send_message(self.ptr, "libraryDataContents"))

    def newFunctionWithName_(self, name: str):
        return send_message(self.ptr, "newFunctionWithName:", to_ns_str(name))


class MetalComputePipelineDescriptor:
    @classmethod
    def new(cls):
        return cls(
            send_message(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new")
        )

    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def setComputeFunction_(self, function: objc_id):
        send_message(self.ptr, "setComputeFunction:", function)

    def setSupportIndirectCommandBuffers_(self, support: bool):
        send_message(self.ptr, "setSupportIndirectCommandBuffers:", support)


class MTLSharedEvent:
    def __init__(self, event: objc_id):
        self.event = event


class MTLComputePipelineState:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def maxTotalThreadsPerThreadgroup(self):
        return send_message(self.ptr, "maxTotalThreadsPerThreadgroup", restype=c_ulong)


class MTLBufferContents:
    def __init__(self, ptr: objc_id, length: int):
        self.ptr = ptr
        self.length = length

    def as_buffer(self, _: int):
        return string_at(self.ptr, self.length)


class MTLBuffer:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def contents(self):
        length = cast(int, send_message(self.ptr, "length", restype=c_ulong))
        contents_ptr = send_message(self.ptr, "contents")
        return MTLBufferContents(contents_ptr, length)


class MTLDevice:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def newSharedEvent(self):
        return MTLSharedEvent(send_message(self.ptr, "newSharedEvent"))

    def newCommandQueueWithMaxCommandBufferCount_(self, count: int):
        return MtlQueue(
            send_message(self.ptr, "newCommandQueueWithMaxCommandBufferCount:", count)
        )

    def newBufferWithBytesNoCopy_length_options_deallocator_(self):
        pass

    def newBufferWithLength_options_(self, size: int, mode: int):
        return MTLBuffer(
            send_message(self.ptr, "newBufferWithLength:options:", size, mode)
        )

    def newLibraryWithSource_options_error_(
        self, code: str, options: MTLCompileOptions, error: Optional[objc_id]
    ):
        return Library(
            send_message(
                self.ptr,
                "newLibraryWithSource:options:error:",
                to_ns_str(code),
                options.ptr,
                None,
            )
        )

    def newLibraryWithData_error_(self, data: objc_id, error: Optional[objc_id] = None):
        return Library(
            send_message(
                self.ptr,
                "newLibraryWithData:error:",
                data,
                error,
            )
        )

    def newComputePipelineStateWithDescriptor_options_reflection_error_(
        self, descriptor: objc_id, option: int, reflection: Any, error: Any
    ):
        return MTLComputePipelineState(
            send_message(
                self.ptr,
                "newComputePipelineStateWithDescriptor:options:reflection:error:",
                descriptor,
                option,
                reflection,
                error,
            )
        )


class _Metal:
    def __init__(self):
        self.MTLCompileOptions = MTLCompileOptions
        self.MTLResourceStorageModeShared = 0
        self.MTLComputePipelineDescriptor = MetalComputePipelineDescriptor

    @staticmethod
    def MTLCreateSystemDefaultDevice():
        return MTLDevice(metal.MTLCreateSystemDefaultDevice())

    @staticmethod
    def MTLPipelineOption():
        pass


class _LibDispatch:
    def dispatch_data_create(self, *args: Any) -> objc_id:
        return _libdispatch.dispatch_data_create(*args)


Metal = _Metal()
libdispatch = _LibDispatch()
