import ctypes
import functools
import hashlib
from typing import Any, List, Optional, Set
from tinygrad.codegen.kernel import check
from tinygrad.helpers import init_c_var
import tinygrad.runtime.autogen.opencl as cl
from tinygrad.device import BufferOptions, Compiled, Compiler, LRUAllocator
from tinygrad.renderer.cstyle import IntelRenderer
from tinygrad.runtime.ops_gpu import CLAllocator, CLCompiler, CLDevice, CLProgram, checked

 # not sure if we want to use CL or LRU.
# class IntelAllocator(LRUAllocator):
class IntelAllocator(CLAllocator):
    def __init__(self, device):
        self.device: IntelDevice = device
        self.track_cross_device: Set[IntelDevice] = set()
        super().__init__(device)

    # def free_cache(self):
    #     self.device.synchronize()
    #     for x in self.track_cross_device: x.synchronize()
    #     self.track_cross_device.clear()
    #     return super().free_cache()

    # def _alloc(self, size: int, options: BufferOptions) -> Any:
    #     # metal uses their own lib/APIs. TBD for intel.
    #     return super()._alloc(size, options)

    # def transfer(self, dest:Any, src:Any, sz:int, src_dev, **kwargs):
    #     src_dev.synchronize()
    #     command_buffer = self.device.mtl_queue.commandBuffer()
    # def copyin(self, dest:Any, src:memoryview): self.as_buffer(dest)[:] = src
    # def copyout(self, dest:memoryview, src:Any): dest[:] = self.as_buffer(src)

class IntelCompiler(CLCompiler):
    def __init__(self, device, compile_key):
        super().__init__(device, compile_key)
    pass

class IntelProgram(CLProgram):
    def __init__(self, device, name: str, lib: bytes):
        # self.device, self.name, self.lib = device, name, lib
        # TODO custom intel code needs to be here
        super().__init__(device, name, lib)

class IntelDevice(Compiled):
    device_ids = None
    def __init__(self, device="", *args):
        print ("Intel Device initialized with args: ", args)

        # should be some intel pip package ref
        # might be an OpenCL package instead?
        # CLDevice.device_ids = init_c_var((cl.cl_device_id * num_devices.value)(), lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))  # noqa: E501

        # device_ids = init_c_var((cl.cl_device_id * num_devices.value)(), lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))  # noqa: E501
        # self.device_id = "INTEL"
        # self.device = "INTEL"
        
        # from ops_gpu. Could de-dupe?
        # self.device_name = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_NAME, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501
        # self.driver_version = (cl.clGetDeviceInfo(self.device_id, cl.CL_DRIVER_VERSION, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501
        # compile_key = hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()

        # self.context = checked(cl.clCreateContext(None, 1, self.device_id, cl.clCreateContext.argtypes[3](), None, status := ctypes.c_int32()), status)
        # self.queue = checked(cl.clCreateCommandQueue(self.context, self.device_id, cl.CL_QUEUE_PROFILING_ENABLE, status), status)
        # self.pending_copyin: List[memoryview] = []

        # if IntelDevice.device_ids is None:
        #     # check(cl.clGetPlatformIDs(0, None, num_platforms := ctypes.c_uint32()))
        #     # check(cl.clGetPlatformIDs(num_platforms.value, platform_ids := (cl.cl_platform_id * num_platforms.value)(), None))
        #     for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        #         err = cl.clGetDeviceIDs(platform_ids[0], device_type, 0, None, num_devices := ctypes.c_uint32())
        #         if err == 0 and num_devices.value != 0: break
        #     if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
        #     IntelDevice.device_ids = init_c_var((cl.cl_device_id * num_devices.value)(), lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))  # noqa: E501

        # self.device_id = IntelDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
        # self.device_name = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_NAME, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501

        # print(self.device_id)
        # print(self.device_name)

        # self.driver_version = (cl.clGetDeviceInfo(self.device_id, cl.CL_DRIVER_VERSION, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501
        # self.context = checked(cl.clCreateContext(None, 1, self.device_id, cl.clCreateContext.argtypes[3](), None, status := ctypes.c_int32()), status)
        # self.queue = checked(cl.clCreateCommandQueue(self.context, self.device_id, cl.CL_QUEUE_PROFILING_ENABLE, status), status)
        # self.pending_copyin: List[memoryview] = []

        # compile_key = hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()

        gpu_device = CLDevice(device="GPU")

        # self.device = gpu_device.dname
        self.device = "INTEL"

        self.device_id = gpu_device.device_id
        self.device_name = gpu_device.device_name
        self.driver_version = gpu_device.driver_version
        self.context = gpu_device.context
        self.queue = gpu_device.queue
        self.pending_copyin = gpu_device.pending_copyin

        compile_key = hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()


        super().__init__(
            device=self.device,
            allocator=IntelAllocator(self), # memory allocator
            renderer=IntelRenderer(),
            compiler=IntelCompiler(self, f"compile_cl_{compile_key}"), # compiling code to binary
            runtime=functools.partial(IntelProgram, self),
            graph=None, # TODO
        )