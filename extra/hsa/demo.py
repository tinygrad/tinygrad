import ctypes, functools
import gpuctypes.hsa as hsa
from tinygrad.helpers import init_c_var
from tinygrad.helpers import Timing, from_mv

def check(status: hsa.hsa_status_t):
  assert status == 0, f"has status is {status}"

def check_ex(status: hsa.hsa_status_t):
  assert status == 0, f"has status is {status}"

@ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
def filter_amdgpu_agent(agent, data):
  status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
  if status == 0 and device_type.value == hsa.HSA_DEVICE_TYPE_GPU:
    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
    ret[0] = agent
    return hsa.HSA_STATUS_INFO_BREAK
  return hsa.HSA_STATUS_SUCCESS

@ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_region_t, ctypes.c_void_p)
def filter_shared_memtype(region, data):
  check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_region_segment_t())))
  if segment.value != hsa.HSA_REGION_SEGMENT_GLOBAL:
    return hsa.HSA_STATUS_SUCCESS
  
  check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_GLOBAL_FLAGS, ctypes.byref(flags := hsa.hsa_region_global_flag_t())))
  if flags.value & hsa.HSA_REGION_GLOBAL_FLAG_KERNARG:
    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_region_t))
    ret[0] = region
    return hsa.HSA_STATUS_INFO_BREAK
  return hsa.HSA_STATUS_SUCCESS

@functools.lru_cache(None)
def find_agent(typ, device_id):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_amdgpu_agent(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == typ:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      print(ret[0].handle)
      if ret[0].handle < device_id:
        ret[0].handle = ret[0].handle + 1
        return hsa.HSA_STATUS_SUCCESS

      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      ret[0] = agent
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  agent = hsa.hsa_agent_t()
  agent.handle = 0
  hsa.hsa_iterate_agents(__filter_amdgpu_agent, ctypes.byref(agent))
  return agent

# @functools.lru_cache(None)
def get_amd_memory_pool(agent, segtyp=-1, flags=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    if segtyp != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
      if segment.value != segtyp:
        return hsa.HSA_STATUS_SUCCESS

    if flags != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(fgs := hsa.hsa_amd_memory_pool_global_flag_t())))
      # print(fgs.value, flags)
      if fgs.value != flags:
        return hsa.HSA_STATUS_SUCCESS

    if location != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc := hsa.hsa_amd_memory_pool_location_t())))
      # print(fgs.value, flags)
      if loc.value != location:
        return hsa.HSA_STATUS_SUCCESS

    # print("")
    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  region = hsa.hsa_amd_memory_pool_t()
  region.handle = 0
  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region))
  return region

  # // Get the kernel code handle
  # hsa_status_t hsaStatus;
  # hsa_executable_symbol_t symbol;
  # hsa_agent_t agent = program()->rocDevice().getBackendDevice();
  # hsaStatus = hsa_executable_get_symbol_by_name(program()->hsaExecutable(),
  #                                               symbolName().c_str(),
  #                                               &agent, &symbol);
  # if (hsaStatus != HSA_STATUS_SUCCESS) {
  #   DevLogPrintfError("Cannot Get Symbol : %s, failed with hsa_status: %d \n",
  #                     symbolName().c_str(), hsaStatus);
  #   return false;
  # }

  # hsaStatus = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
  #                                            &kernelCodeHandle_);
  # if (hsaStatus != HSA_STATUS_SUCCESS) {
  #   DevLogPrintfError(" Cannot Get Symbol Info: %s, failed with hsa_status: %d \n ",
  #                     symbolName().c_str(), hsaStatus);
  #   return false;
  # }

  # hsaStatus = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK,
  #                                            &kernelHasDynamicCallStack_);
  # if (hsaStatus != HSA_STATUS_SUCCESS) {
  #   DevLogPrintfError(" Cannot Get Dynamic callstack info, failed with hsa_status: %d \n ", hsaStatus);
  #   return false;
  # }

class Kernel():
  def __init__(self, agent, binary, kernel_name):
    self.info = self.get_binary_info(binary)
    assert self.info['amdhsa.version'] == [1, 2] # This is Code Object V5

    bin_size = len(binary)

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x))))
    check(hsa.hsa_code_object_reader_create_from_memory(binary, bin_size, ctypes.byref(code_reader := hsa.hsa_code_object_reader_t())))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, agent, code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    # kernel_name = "__" + kernel_name
    sym = kernel_name + ".kd"
    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, sym.encode("utf-8"), ctypes.byref(agent), ctypes.byref(x))))
    self.kernel_handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x))))
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x))))
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x))))
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x))))
    print("Kernel handle:", self.kernel_handle)

  def handle(self): return self.kernel_handle

  def get_binary_info(self, binary):
    # with open("/home/nimlgen/amd.elf", 'wb') as file:
    #   file.write(binary)

    from io import BytesIO
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import NoteSection
    import msgpack
    with BytesIO(binary) as f:
      elffile = ELFFile(f)
      print(elffile.header['e_type'])

      dynsym = elffile.get_section_by_name('.dynsym')
      if not dynsym:
        print("Dynamic symbol table not found.")
        return

      kern_info = None
      for section in elffile.iter_sections():
        if isinstance(section, NoteSection):
          for note in section.iter_notes():
            kern_info = msgpack.unpackb(note['n_descdata'])

      assert kern_info is not None
      return kern_info

class GPUBuffer():
  def __init__(self, sz, agent):
    pass

def compile_ast_to_hip(out):
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.helpers import DEBUG, to_function_name
  from tinygrad.codegen.linearizer import Linearizer
  from tinygrad.renderer.cstyle import HIPRenderer
  from tinygrad.runtime.ops_hip import compile_hip
  # lin = Linearizer(out.lazydata.schedule()[-1].ast)
  # lin.hand_coded_optimizations()
  # lin.linearize()
  # code = HIPRenderer(to_function_name(lin.name), lin.uops)[0]
  code = """
#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (32, 1) E_2_32_4(float* data0) {
  int gidx0 = blockIdx.x; /* 2 */
  int lidx1 = threadIdx.x; /* 32 */
  int alu0 = ((gidx0*128)+(lidx1*4));
  float4 val0 = float4(3.0);
  float4 val1 = float4(5.0);
  *((float4*)(data0+alu0)) = float4(5.0);
}"""
  if DEBUG >= 4: print(code)
  return compile_hip(code), "E_2_32_4"

# AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7
if __name__ == "__main__":
  print("***** import tinygrad")
  from tinygrad import Tensor, Device, TinyJit
  print("***** init HSA")
  check(hsa.hsa_init())

  # print("***** call HSAKMT")
  # hsa.hsaKmtGetVersion(ctypes.byref(hsakmt_ver := hsa.struct__HsaVersionInfo()))
  # print(hsakmt_ver)

  print("***** agent HSA")
  agent_cpu = find_agent(hsa.HSA_DEVICE_TYPE_CPU, device_id=0)
  agent = agent_0 = find_agent(hsa.HSA_DEVICE_TYPE_GPU, device_id=0)
  agent_1 = find_agent(hsa.HSA_DEVICE_TYPE_GPU, device_id=1)

  # Check that we got different devs with real handle ptr
  assert agent_0.handle != agent_1.handle and agent_0.handle > 0xffff and agent_1.handle > 0xffff


  check(hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_NAME, ctypes.byref(buf := ctypes.create_string_buffer(256))))
  print("dev:", ctypes.string_at(buf).decode()) # gfx1100 is here!

  check(hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(queue_size := ctypes.c_uint32())))
  check(hsa.hsa_agent_get_info(agent, hsa.HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES, ctypes.byref(coop_queues := ctypes.c_uint32())))
  print("max queue size:", queue_size.value)
  print("support coop queues:", coop_queues.value) # gfx1100 supports coops queue

  check(hsa.hsa_agent_get_info(agent, hsa.HSA_AMD_AGENT_INFO_NUM_SDMA_ENG, ctypes.byref(num_sdma := ctypes.c_uint32())))
  print("num_sdma", num_sdma.value) # 2

  check(hsa.hsa_agent_get_info(agent, hsa.HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG, ctypes.byref(num_xdgi_sdma := ctypes.c_uint32())))
  print("num_xdgi_sdma", num_xdgi_sdma.value) # 0 -- seems this is CDNA only

  UINT32_MAX = (1 << 32) - 1
  hsa_queue_ptr_t = ctypes.POINTER(hsa.hsa_queue_t)
  null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
  check(hsa.hsa_queue_create(agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, UINT32_MAX, UINT32_MAX, ctypes.byref(queue := hsa_queue_ptr_t())))

  print("***** create HSA kernel")
  a = Tensor([1.,2.]*128, device="HIP").realize()
  b = Tensor([6.,5.]*128, device="HIP").realize()
  binary, name = compile_ast_to_hip((a + b).realize()) # elf
  kern = Kernel(agent, binary, name)

  print("***** create simple args")
  # char* in=(char*)malloc(1024*1024*4)
  # memset(in, 1, 1024*1024*4)
  # err=hsa_memory_register(in, 1024*1024*4)

  # def simple_memory(sz, val):
  #   mv = memoryview(bytearray(sz))
  #   ctypes.memset(from_mv(mv), 0, len(mv))
  #   check(hsa.hsa_memory_register())
  #   return 

  print("***** prep buffers")
  # hsa_memory_allocate -> (internal)MemoryRegion::AllocateImpl -> hsaKmtAllocMemory.
  # They seem to use caching allocator (fragment_allocator_, 2MB blocks) to reduce number of calls to hsaKmtAllocMemory.

  # hsa_memory_copy -- bad, use amd ext
  # CPU-CPU is memcpy
  # Same GPU is DmaCopy, BlitDevToDev always engages a Blit Kernel.
  #   BlitDevToDev always engages a Blit Kernel
  #   BlitHostToDev (H2D) engage either a Blit Kernel or sDMA.
  #   BlitDevToHost (D2H and P2P) engage either a Blit Kernel or sDMA.
  #   XGMI engines
  # GPU-GPU using host memory here.
  sz = 4 << 30
  agents_cnt = 3
  agents = (hsa.hsa_agent_t * agents_cnt)(agent_cpu, agent_0, agent_1)

  gpu_0_memory_pool = get_amd_memory_pool(agent_0, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_GPU)
  print("memory pool 0", gpu_0_memory_pool.handle)

  gpu_1_memory_pool = get_amd_memory_pool(agent_1, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_GPU)
  print("memory pool 1", gpu_1_memory_pool.handle)

  cpu_memory_pool = get_amd_memory_pool(agent_cpu, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_CPU)
  print("memory pool cpu", cpu_memory_pool.handle)

  check(hsa.hsa_amd_memory_pool_allocate(gpu_0_memory_pool, sz, 0, ctypes.byref(buf0 := ctypes.c_void_p())))
  check(hsa.hsa_amd_agents_allow_access(agents_cnt, agents, None, buf0))
  # print(buf0)

  check(hsa.hsa_amd_memory_pool_allocate(gpu_0_memory_pool, sz, 0, ctypes.byref(buf01 := ctypes.c_void_p())))
  check(hsa.hsa_amd_agents_allow_access(agents_cnt, agents, None, buf01))
  # print(buf01)

  check(hsa.hsa_amd_memory_pool_allocate(gpu_1_memory_pool, sz, 0, ctypes.byref(buf1 := ctypes.c_void_p())))
  check(hsa.hsa_amd_agents_allow_access(agents_cnt, agents, None, buf1))
  # print(buf1)

  check(hsa.hsa_amd_memory_pool_allocate(cpu_memory_pool, sz, 0, ctypes.byref(cpu_buf := ctypes.c_void_p())))
  check(hsa.hsa_amd_agents_allow_access(agents_cnt, agents, None, cpu_buf))
  
  # cpu_memory = memoryview(bytearray(sz))
  # check(hsa.hsa_amd_memory_lock(from_mv(cpu_memory), sz, agents, agents_cnt, ctypes.byref(cpu_buf := ctypes.c_void_p())))
  # # print("cpu", cpu_buf)

  for i in range(3):
    with Timing("transfer copy (p2p) ", lambda x: f" {sz/x:.2f} GB/s"): # 19.19 GB/s
      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
      check(hsa.hsa_amd_memory_async_copy(buf1, agent_1, buf0, agent_0, sz, 0, None, copy_signal))
      sigval = hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  for i in range(3):
    with Timing("engine copy (p2p) ", lambda x: f" {sz/x:.2f} GB/s"): # 19.19 GB/s
      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
      check(hsa.hsa_amd_memory_async_copy_on_engine(buf1, agent_1, buf0, agent_0, sz, 0, None, copy_signal, hsa.HSA_AMD_SDMA_ENGINE_1, True))
      sigval = hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  for i in range(3):
    with Timing("bidirectional engine copy (p2p) ", lambda x: f" {sz/x:.2f} GB/s"): # ~ 37.55 GB/s
      first_half = sz // 2
      second_half = sz - first_half

      second_buf0 = ctypes.c_void_p(buf0.value + first_half)
      second_buf1 = ctypes.c_void_p(buf1.value + first_half)

      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal0 := hsa.hsa_signal_t())))     
      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal1 := hsa.hsa_signal_t())))
      check(hsa.hsa_amd_memory_async_copy_on_engine(buf1, agent_1, buf0, agent_0, first_half, 0, None, copy_signal0, hsa.HSA_AMD_SDMA_ENGINE_0, True))
      check(hsa.hsa_amd_memory_async_copy_on_engine(second_buf0, agent_0, second_buf1, agent_1, second_half, 0, None, copy_signal1, hsa.HSA_AMD_SDMA_ENGINE_0, True))
      sigval = hsa.hsa_signal_wait_acquire(copy_signal0, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)
      sigval = hsa.hsa_signal_wait_acquire(copy_signal1, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)


  print("***** play with copies")
  ca = Tensor([1.,2.]*128, device="HIP:0").realize()
  cb = Tensor([6.,5.]*128, device="HIP:1").realize()

  # hsa.hsa_amd_agents_allow_access(agent_0)

  print("***** prep kernel")
  check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t())))
  
  
  kernarg_region = hsa.hsa_region_t()
  kernarg_region.handle = -1
  hsa.hsa_agent_iterate_regions(agent, filter_shared_memtype, ctypes.byref(kernarg_region))
  print(kernarg_region.handle)

  hip_dev = Device["HIP"]
  hip_buffer = hip_dev.allocator.alloc(2 * 32 * 4 * 4)
  hip_dev.synchronize()

  class Args(ctypes.Structure): # This is what we've put in extra[1], seems that...
    _fields_ = [("l1", ctypes.c_void_p)]
    _align_ = 16
  args = Args()
  args.l1 = hip_buffer

  check(hsa.hsa_memory_allocate(kernarg_region, ctypes.c_size_t(kern.kernargs_segment_size.value), ctypes.byref(kernarg_address := ctypes.c_void_p())))
  ctypes.memmove(kernarg_address, ctypes.addressof(args), ctypes.sizeof(args))
  print(kernarg_address)

  print("***** prepearing kernel")
  index = hsa.hsa_queue_load_write_index_relaxed(queue)
  print("q indx:", index)

  base_address = ctypes.cast(queue.contents.base_address, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
  dispatch_packet_addr = ctypes.addressof(base_address[index & (queue.contents.size - 1)])
  dispatch_packet_ptr = ctypes.pointer(base_address[index & (queue.contents.size - 1)])
  print("dispatch_packet_addr", dispatch_packet_addr)

  dispatch_packet_ptr.contents.setup |= 1 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS

  dispatch_packet_ptr.contents.workgroup_size_x = 32
  dispatch_packet_ptr.contents.workgroup_size_y = 1
  dispatch_packet_ptr.contents.workgroup_size_z = 1
  dispatch_packet_ptr.contents.grid_size_x = 2 * 32 # opencl-like
  dispatch_packet_ptr.contents.grid_size_y = 1
  dispatch_packet_ptr.contents.grid_size_z = 1
  dispatch_packet_ptr.contents.completion_signal = signal
  dispatch_packet_ptr.contents.kernel_object = kern.handle()
  dispatch_packet_ptr.contents.kernarg_address = kernarg_address
  dispatch_packet_ptr.contents.private_segment_size = kern.private_segment_size
  dispatch_packet_ptr.contents.group_segment_size = kern.group_segment_size

  header = 0
  header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE
  header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE
  header |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

  dispatch_packet_ptr.contents.header = header # atomic?
  hsa.hsa_queue_store_write_index_relaxed(queue, index + 1)
  hsa.hsa_signal_store_relaxed(queue.contents.doorbell_signal, index)

  print("***** waiting kernel")

  sigval = hsa.hsa_signal_wait_acquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  
  ans = memoryview(bytearray(2 * 32 * 4 * 4))
  hip_dev.synchronize()
  hip_dev.allocator.copyout(ans, hip_buffer)
  
  import numpy as np
  ans = np.frombuffer(ans, np.float32)
  print(ans) # got 3 + 5 = 8.0

  print("***** stop HSA")
  check(hsa.hsa_shut_down())
