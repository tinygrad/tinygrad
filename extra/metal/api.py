# pyright: reportPrivateUsage=false
"""Emulate the pyobjc API"""

from extra.metal.cdll import (
    metal,
    send_message,
    libobjc,
    objc_id,
    to_ns_str,
    int_tuple_to_struct,
)
from extra.metal.cdll import libdispatch as _libdispatch
from typing import Optional, cast, Any
from ctypes import c_ulong, string_at, POINTER, _Pointer, c_float, Structure
import numpy as np


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

    def setComputePipelineState_(self, pipeline_state: "MTLComputePipelineState"):
        send_message(self.ptr, "setComputePipelineState:", pipeline_state.ptr)

    def setBuffer_offset_atIndex_(self, buffer: "MTLBuffer", offset: int, index: int):
        send_message(self.ptr, "setBuffer:offset:atIndex:", buffer.ptr, offset, index)

    def dispatchThreadgroups_threadsPerThreadgroup_(
        self,
        threadgroups: Structure,
        threadPerThreadgroup: Structure,
    ):
        send_message(
            self.ptr,
            "dispatchThreadgroups:threadsPerThreadgroup:",
            threadgroups,
            threadPerThreadgroup,
        )

    def endEncoding(self):
        send_message(self.ptr, "endEncoding")


class CommandBuffer:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr
        self.error = lambda: None
    
    def GPUEndTime(self):
        return send_message(self.ptr, "GPUEndTime", restype=c_ulong)
    
    def GPUStartTime(self):
        return send_message(self.ptr, "GPUStartTime", restype=c_ulong)

    def computeCommandEncoder(self):
        return Encoder(send_message(self.ptr, "computeCommandEncoder"))

    def blitCommandEncoder(self):
        return Encoder(send_message(self.ptr, "blitCommandEncoder"))

    def commit(self):
        send_message(self.ptr, "commit")

    def waitUntilCompleted(self):
        send_message(self.ptr, "waitUntilCompleted")


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
    def __init__(self, buffer: memoryview):
        self.buffer = buffer

    def as_buffer(self, *args: Any):
        return self.buffer


class MTLBuffer:
    def __init__(
        self,
        ptr: objc_id,
        buffer: memoryview,
        length: int,
    ):
        self.ptr = ptr
        self.buffer = buffer
        self.length = length

    def contents(self):
        return MTLBufferContents(self.buffer)

    def readonly_contents(self):
        contents_ptr = send_message(self.ptr, "contents")
        assert contents_ptr.value
        contents = (c_float * self.length).from_address(contents_ptr.value)
        return MTLBufferContents(memoryview(contents))


class MTLDevice:
    def __init__(self, ptr: objc_id):
        self.ptr = ptr

    def newSharedEvent(self):
        return MTLSharedEvent(send_message(self.ptr, "newSharedEvent"))

    def newCommandQueueWithMaxCommandBufferCount_(self, count: int):
        return MtlQueue(
            send_message(self.ptr, "newCommandQueueWithMaxCommandBufferCount:", count)
        )

    def newBufferWithBytesNoCopy_length_options_deallocator_(
        self,
        buffer: "_Pointer[c_float]",
        buffer_memoryview: memoryview,
        length: int,
        options: int,
        deallocator: Any,
    ):
        return MTLBuffer(
            send_message(
                self.ptr,
                "newBufferWithBytesNoCopy:length:options:deallocator:",
                buffer,
                length,
                options,
                deallocator,
            ),
            buffer_memoryview,
            length,
        )

    def newBufferWithLength_options_(self, size: int, mode: int):
        num_elem = size // 4
        buffer = np.zeros(num_elem, dtype=np.float32)
        return self.newBufferWithBytesNoCopy_length_options_deallocator_(
            buffer.ctypes.data_as(POINTER(c_float)),
            buffer.data.cast("B"),
            num_elem,
            mode,
            None,
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
        ), None

    def newComputePipelineStateWithDescriptor_options_reflection_error_(
        self, descriptor: MetalComputePipelineDescriptor, option: int, reflection: Any, error: Any
    ):
        return MTLComputePipelineState(
            send_message(
                self.ptr,
                "newComputePipelineStateWithDescriptor:options:reflection:error:",
                descriptor.ptr,
                option,
                reflection,
                error,
            )
        ), None


class _Metal:
    def __init__(self):
        self.MTLCompileOptions = MTLCompileOptions
        self.MTLResourceStorageModeShared = 0
        self.MTLComputePipelineDescriptor = MetalComputePipelineDescriptor

        def MTLSize(*args: int):
            return int_tuple_to_struct(args)

        self.MTLSize = MTLSize

    @staticmethod
    def MTLCreateSystemDefaultDevice():
        return MTLDevice(metal.MTLCreateSystemDefaultDevice())

    @staticmethod
    def MTLPipelineOption(value: int):
        return value


class _LibDispatch:
    def dispatch_data_create(self, *args: Any) -> objc_id:
        return _libdispatch.dispatch_data_create(*args)


Metal = _Metal()
libdispatch = _LibDispatch()
