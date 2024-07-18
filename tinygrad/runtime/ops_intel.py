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

class IntelCompiler(CLCompiler):
    def __init__(self, device, compile_key):
        super().__init__(device, compile_key)

class IntelProgram(CLProgram):
    def __init__(self, device, name: str, lib: bytes):
        super().__init__(device, name, lib)

class IntelDevice(Compiled):
    def __init__(self, *args):
        print ("Intel Device initialized with args: ", args)
        gpu_device = CLDevice(device="GPU")
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