import functools
from typing import Any, List, Set
from tinygrad.device import BufferOptions, Compiled, Compiler, LRUAllocator
from tinygrad.runtime.ops_gpu import CLAllocator, CLCompiler

 # not sure if we want to use CL or LRU.
# class IntelAllocator(CLAllocator):
class IntelAllocator(LRUAllocator):

    def __init__(self, device):
        self.device: IntelDevice = device
        self.track_cross_device: Set[IntelDevice] = set()
        super().__init__()

    def free_cache(self):
        self.device.synchronize()
        for x in self.track_cross_device: x.synchronize()
        self.track_cross_device.clear()
        return super().free_cache()
    
    def _alloc(self, size: int, options: BufferOptions) -> Any:
        # metal uses their own lib/APIs. TBD for intel.
        return super()._alloc(size, options)
    
    def transfer(self, dest:Any, src:Any, sz:int, src_dev: IntelDevice, **kwargs):
        src_dev.synchronize()
        command_buffer = self.device.mtl_queue.commandBuffer()

class IntelCompiler(CLCompiler):
    """
    compiler_opts = CompilerOptions("INTEL", has_tensor_cores=True)
    def render(self, name:str, uops) -> str: return IntelRenderer(name, uops)
    """
    pass

class IntelProgram:
    pass



class IntelDevice(Compiled):
    def __init__(self, *args):
        print ("Intel Device initialized with args: ", args)

        # should be some intel pip package ref
        # might be an OpenCL package instead?
        self.device = None

        # also a prop of the metal library
        # self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)

        self.mtl_buffers_in_flight: List[Any] = []
        self.mv_in_metal: List[memoryview] = []
        self.track_cross_buffer: List[Any] = []

        super().__init__(
            device=self.device,
            allocator=IntelAllocator(self),
            renderer=None, # TODO
            compiler=IntelCompiler(self),
            runtime=functools.partial(IntelProgram, self), # TODO
            graph=None, # TODO
        )



