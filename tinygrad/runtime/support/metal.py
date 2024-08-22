import ctypes, ctypes.util
from tinygrad.runtime.support.objc import ObjcClass, ObjcInstance

metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
core_graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")  # needed: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc

metal.MTLCreateSystemDefaultDevice.restype, metal.MTLCreateSystemDefaultDevice.argtypes = ctypes.c_void_p, []

libdispatch = ctypes.CDLL(ctypes.util.find_library("dispatch"))
libdispatch.dispatch_data_create.restype, libdispatch.dispatch_data_create.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]

class MTLSize(ctypes.Structure):
  _fields_ = [("width", ctypes.c_ulong),
              ("height", ctypes.c_ulong),
              ("depth", ctypes.c_ulong)]

class IndirectCommandBufferExecutionRange(ctypes.Structure):
  _pack_ = 1
  _fields_ = [("location", ctypes.c_ulong),
              ("length", ctypes.c_ulong)]

class Metal:
  MTLCreateSystemDefaultDevice = lambda: ObjcInstance(metal.MTLCreateSystemDefaultDevice())
  MTLResourceStorageModeShared = 0
  MTLSize = MTLSize
  MTLCompileOptions = ObjcClass("MTLCompileOptions")
  # MTL graph support
  MTLIndirectCommandBufferDescriptor = ObjcClass("MTLIndirectCommandBufferDescriptor")
  MTLIndirectCommandTypeConcurrentDispatch = 32
  MTLResourceUsageRead = 1
  MTLResourceUsageWrite = 2
  MTLResourceUsageSample = 4
  MTLIndirectCommandBufferExecutionRangeMake = IndirectCommandBufferExecutionRange
  MTLComputePipelineDescriptor = ObjcClass("MTLComputePipelineDescriptor")
