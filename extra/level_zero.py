import ctypes
import tinygrad.runtime.autogen.ze as ze
from tinygrad.helpers import from_mv, init_c_struct_t


def check(status):
  if status != ze.ZE_RESULT_SUCCESS: raise RuntimeError(f"LEVEL ZERO Error {status}")

check(ze.zeInit(0))
check(ze.zeDriverGet(ctypes.byref(driver_count := ctypes.c_uint32()), None))
print(driver_count)
drivers = (ze.ze_driver_handle_t*driver_count.value)()
check(ze.zeDriverGet(ctypes.byref(driver_count), drivers))
print(driver_count, drivers)
driver = drivers[0]

check(ze.zeDeviceGet(driver, ctypes.byref(device_count := ctypes.c_uint32()), None))
print(device_count)
devices = (ze.ze_device_handle_t*device_count.value)()
check(ze.zeDeviceGet(driver, ctypes.byref(device_count), devices))
print(device_count, devices)

agents = []
for dev in devices:
  device_properties = ze.ze_device_properties_t()
  device_properties.stype = ze.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
  check(ze.zeDeviceGetProperties(dev, ctypes.byref(device_properties)))
  if device_properties.type == ze.ZE_DEVICE_TYPE_GPU: agents.append(dev)

print(agents)

context_desc = ze.ze_context_desc_t()
context_desc.stype = ze.ZE_STRUCTURE_TYPE_CONTEXT_DESC
check(ze.zeContextCreate(driver, ctypes.byref(context_desc), ctypes.byref(context := ze.ze_context_handle_t())))

altdesc = ze.ze_command_queue_desc_t()
altdesc.stype = ze.ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
check(ze.zeCommandListCreateImmediate(context, agents[0], ctypes.byref(altdesc), ctypes.byref(cmdlist := ze.ze_command_list_handle_t())))

# Create an event to be signaled by the device
ep_desc = ze.ze_event_pool_desc_t()
ep_desc.stype = ze.ZE_STRUCTURE_TYPE_EVENT_POOL_DESC
ep_desc.count = 1
ep_desc.flags = ze.ZE_EVENT_POOL_FLAG_HOST_VISIBLE
check(ze.zeEventPoolCreate(context, ctypes.byref(ep_desc), 1, ctypes.byref(agents[0]), ctypes.byref(event_pool := ze.ze_event_pool_handle_t())))
 
ev_desc = ze.ze_event_desc_t()
ev_desc.stype = ze.ZE_STRUCTURE_TYPE_EVENT_DESC
ev_desc.signal = ze.ZE_EVENT_SCOPE_FLAG_HOST
ev_desc.wait = ze.ZE_EVENT_SCOPE_FLAG_HOST
check(ze.zeEventCreate(event_pool, ctypes.byref(ev_desc), ctypes.byref(event := ze.ze_event_handle_t())))

# signal the event from the device and wait for completion


print("trying to create module")
from tinygrad.runtime.ops_gpu import CLCompiler
from tinygrad import Device
spriv_bin = CLCompiler(Device["GPU"], "test").compile(f"""
__attribute__((reqd_sub_group_size(8)))
__kernel void test(__global float* data0, const __global float* data1) {{
  data0[0] = data1[0];
}}
""")

module_desc = ze.ze_module_desc_t()
module_desc.stype = ze.ZE_STRUCTURE_TYPE_MODULE_DESC
module_desc.pNext = None
module_desc.format = ze.ZE_MODULE_FORMAT_NATIVE # should switch to spriv?
module_desc.inputSize = len(spriv_bin)
module_desc.pInputModule = (ctypes.c_uint8*len(spriv_bin)).from_buffer(bytearray(spriv_bin))
module_desc.pBuildFlags = None
module_desc.pConstants = None

check(ze.zeModuleCreate(context, agents[0], ctypes.byref(module_desc), ctypes.byref(module := ze.ze_module_handle_t()), None))

kernel_desc = ze.ze_kernel_desc_t()
kernel_desc.stype = ze.ZE_STRUCTURE_TYPE_KERNEL_DESC
kernel_desc.pKernelName = ctypes.create_string_buffer(b"test")
check(ze.zeKernelCreate(module, ctypes.byref(kernel_desc), ctypes.byref(kernel := ze.ze_kernel_handle_t())))
print("loaded", kernel)

sz = 1024 * 4
mem_alloc_desc = ze.ze_device_mem_alloc_desc_t()
mem_alloc_desc.stype = ze.ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC # can set cached/uncached
mem_alloc_desc.flags = 0
mem_alloc_desc.ordinal = 0 # ?
check(ze.zeMemAllocDevice(context, ctypes.byref(mem_alloc_desc), sz, 0, agents[0], ctypes.byref(buf0 := ctypes.c_void_p())))
check(ze.zeMemAllocDevice(context, ctypes.byref(mem_alloc_desc), sz, 0, agents[0], ctypes.byref(buf1 := ctypes.c_void_p())))
print(buf0, buf1)

import array
ones = memoryview(array.array('f', [1.0] * 1024))
check(ze.zeCommandListAppendMemoryCopy(cmdlist, buf1, from_mv(ones), sz, None, 0, None))

c_args = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(2)]))(buf0, buf1)
# c_args.__setattr__('f1', buf1.value)
global_size, local_size = (32, 1, 1), (1, 1, 1)
check(ze.zeKernelSetGroupSize(kernel, *local_size))
# check(ze.zeKernelSetGroupSize(kernel, 32, 1, ctypes.byref(c_args, 8)))
check(ze.zeKernelSetArgumentValue(kernel, 0, 8, ctypes.byref(c_args, 0)))
check(ze.zeKernelSetArgumentValue(kernel, 1, 8, ctypes.byref(c_args, 8)))

groups_count = ze.ze_group_count_t(*global_size)
check(ze.zeCommandListAppendLaunchKernel(cmdlist, kernel, ctypes.byref(groups_count), None, 0, None))

result = memoryview(array.array('f', [0.0] * 1024))
check(ze.zeCommandListAppendMemoryCopy(cmdlist, from_mv(result), buf0, sz, None, 0, None))

ze.zeCommandListAppendSignalEvent(cmdlist, event)
ze.zeEventHostSynchronize(event, (1<<64)-1)
print("done")

assert result[0] == 1.0

check(ze.zeContextDestroy(context))
