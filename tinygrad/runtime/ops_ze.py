from __future__ import annotations
import ctypes, functools, subprocess, io, atexit, collections, json
from typing import Tuple, TypeVar, List, Dict, Any
import tinygrad.runtime.autogen.ze as ze
from tinygrad.helpers import DEBUG, init_c_var, from_mv, round_up, to_mv, init_c_struct_t, getenv
from tinygrad.device import Compiled, LRUAllocator, BufferOptions, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice

def check(status):
  if status != ze.ZE_RESULT_SUCCESS: raise RuntimeError(f"ZE Error {status}")

def ze_get_all_agents():
  agents = collections.defaultdict(list)
  device_properties = ze.ze_device_properties_t()
  device_properties.stype = ze.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES

  check(ze.zeDeviceGet(ZEDevice.driver, ctypes.byref(device_count := ctypes.c_uint32()), None))
  devices = (ze.ze_device_handle_t*device_count.value)()
  check(ze.zeDeviceGet(ZEDevice.driver, ctypes.byref(device_count), devices))

  for dev in devices:
    check(ze.zeDeviceGetProperties(dev, ctypes.byref(device_properties)))
    agents[device_properties.type].append(dev)
  return agents

class ZECompiler(CLCompiler):
  linearizer_opts = LinearizerOptions("ZE", has_tensor_cores=False, shared_max=65536) # TODO: not sure about shared_max
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(CLDevice(), self.arch)

class ZEProgram:
  def __init__(self, device:ZEDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    module_desc = ze.ze_module_desc_t()
    module_desc.stype = ze.ZE_STRUCTURE_TYPE_MODULE_DESC
    module_desc.pNext = None
    module_desc.format = ze.ZE_MODULE_FORMAT_NATIVE
    module_desc.inputSize = len(lib)
    module_desc.pInputModule = (ctypes.c_uint8*len(lib)).from_buffer(bytearray(lib))
    module_desc.pBuildFlags = None
    module_desc.pConstants = None
    check(ze.zeModuleCreate(self.device.context, self.device.agent, ctypes.byref(module_desc), ctypes.byref(module := ze.ze_module_handle_t()), None))
    self.module = module

    kernel_desc = ze.ze_kernel_desc_t()
    kernel_desc.stype = ze.ZE_STRUCTURE_TYPE_KERNEL_DESC
    kernel_desc.pKernelName = ctypes.create_string_buffer(self.name.encode())
    check(ze.zeKernelCreate(self.module, ctypes.byref(kernel_desc), ctypes.byref(kernel := ze.ze_kernel_handle_t())))
    self.kernel = kernel

    self.groups_count = ze.ze_group_count_t()

  def __del__(self):
    self.device.synchronize()
    # TODO

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    check(ze.zeKernelSetGroupSize(self.kernel, *local_size))
    for i in range(len(args)): check(ze.zeKernelSetArgumentValue(self.kernel, i, 8, ctypes.byref(ctypes.c_uint64(args[i]))))
    for i in range(len(vals)): check(ze.zeKernelSetArgumentValue(self.kernel, len(args)+i, 4, ctypes.byref(ctypes.c_uint32(vals[i]))))

    self.groups_count.groupCountX = global_size[0]
    self.groups_count.groupCountY = global_size[1]
    self.groups_count.groupCountZ = global_size[2]
    check(ze.zeCommandListAppendLaunchKernel(self.device.cmdlist, self.kernel, ctypes.byref(self.groups_count), None, 0, None))
    # signal = self.device.hw_queue.submit_kernel(self, global_size, local_size, kernargs, need_signal=(wait or PROFILE))
    # if wait:
    #   hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    #   check(hsa.hsa_amd_profiling_get_dispatch_time(self.device.agent, signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
    #   return (timings.end - timings.start) * self.device.clocks_to_time

T = TypeVar("T")
class ZEAllocator(LRUAllocator):
  def __init__(self, device:ZEDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    mem_alloc_desc = ze.ze_device_mem_alloc_desc_t()
    mem_alloc_desc.stype = ze.ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    mem_alloc_desc.flags = 0
    mem_alloc_desc.ordinal = 0 # ?
    check(ze.zeMemAllocDevice(self.device.context, ctypes.byref(mem_alloc_desc), size, 0, self.device.agent, ctypes.byref(buf := ctypes.c_void_p())))
    return buf.value

  def _free(self, opaque:T):
    ZEDevice.synchronize_system()
    # check(hsa.hsa_amd_memory_pool_free(opaque))

  def copyin(self, dest:T, src: memoryview):
    check(ze.zeCommandListAppendMemoryCopy(self.device.cmdlist, dest, from_mv(src), src.nbytes, None, 0, None))

  def copyout(self, dest:memoryview, src:T):
    ZEDevice.synchronize_system()
    check(ze.zeCommandListAppendMemoryCopy(self.device.cmdlist, from_mv(dest), src, dest.nbytes, None, 0, None))

class ZEDevice(Compiled):
  devices: List[ZEDevice] = []
  agents: Dict[int, List[hsa.hsa_agent_t]] = {}
  driver = None
  def __init__(self, device:str=""):
    if not ZEDevice.agents:
      check(ze.zeInit(0))
      check(ze.zeDriverGet(ctypes.byref(driver_count := ctypes.c_uint32()), None))
      drivers = (ze.ze_driver_handle_t*driver_count.value)()
      check(ze.zeDriverGet(ctypes.byref(driver_count), drivers))
      ZEDevice.driver = drivers[0]
      ZEDevice.agents = ze_get_all_agents()

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.agent = ZEDevice.agents[ze.ZE_DEVICE_TYPE_GPU][self.device_id]
    
    context_desc = ze.ze_context_desc_t()
    context_desc.stype = ze.ZE_STRUCTURE_TYPE_CONTEXT_DESC
    check(ze.zeContextCreate(ZEDevice.driver, ctypes.byref(context_desc), ctypes.byref(context := ze.ze_context_handle_t())))
    self.context = context

    altdesc = ze.ze_command_queue_desc_t()
    altdesc.stype = ze.ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
    check(ze.zeCommandListCreateImmediate(context, self.agent, ctypes.byref(altdesc), ctypes.byref(imm_cmdlist := ze.ze_command_list_handle_t())))
    self.cmdlist = imm_cmdlist

    ZEDevice.devices.append(self)

    # Create event pool
    ep_desc = ze.ze_event_pool_desc_t()
    ep_desc.stype = ze.ZE_STRUCTURE_TYPE_EVENT_POOL_DESC
    ep_desc.count = 4096
    ep_desc.flags = ze.ZE_EVENT_POOL_FLAG_HOST_VISIBLE
    check(ze.zeEventPoolCreate(context, ctypes.byref(ep_desc), 1, ctypes.byref(self.agent), ctypes.byref(event_pool := ze.ze_event_pool_handle_t())))
    self.event_pool = event_pool

    self.event_desc = ze.ze_event_desc_t()
    self.event_desc.stype = ze.ZE_STRUCTURE_TYPE_EVENT_DESC
    self.event_desc.signal = ze.ZE_EVENT_SCOPE_FLAG_HOST
    self.event_desc.wait = ze.ZE_EVENT_SCOPE_FLAG_HOST

    self.arch = "a770" # TODO: replace this please

    self.delayed_free: List[int] = []
    self.reusable_events: List[ze.ze_event_handle_t] = []

    super().__init__(device, ZEAllocator(self), ZECompiler(self.arch), functools.partial(ZEProgram, self), None)

  def synchronize(self):
    ze.zeCommandListAppendSignalEvent(self.cmdlist, event:=self.alloc_event(reusable=True))
    ze.zeEventHostSynchronize(event, (1<<64)-1)

    for ev in self.reusable_events: check(ze.zeEventDestroy(ev))
    self.reusable_events.clear()

  @staticmethod
  def synchronize_system():
    for d in ZEDevice.devices: d.synchronize()

  def alloc_event(self, reusable=False):
    check(ze.zeEventCreate(self.event_pool, ctypes.byref(self.event_desc), ctypes.byref(event := ze.ze_event_handle_t())))
    # reusable means a event could be reused after synchronize for the device it's allocated from is called.
    if reusable: self.reusable_events.append(event)
    return event
