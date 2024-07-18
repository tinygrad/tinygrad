import functools
import hashlib
from typing import Set
from tinygrad.device import Compiled
from tinygrad.renderer.cstyle import IntelRenderer
from tinygrad.runtime.ops_gpu import CLAllocator, CLCompiler, CLDevice, CLProgram

class IntelAllocator(CLAllocator):
  def __init__(self, device):
    self.device = device
    self.track_cross_device: Set[IntelDevice] = set()
    super().__init__(device)

class IntelCompiler(CLCompiler): pass
class IntelProgram(CLProgram): pass

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
      compiler=IntelCompiler(gpu_device, f"compile_cl_{compile_key}"), # compiling code to binary
      runtime=functools.partial(IntelProgram, self),
      graph=None, # TODO
    )