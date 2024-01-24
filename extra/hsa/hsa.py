import ctypes, functools, time
import gpuctypes.hsa as hsa
from tinygrad.helpers import init_c_var

def check(status: hsa.hsa_status_t):
  assert status == 0, f"has status is {status}"

@functools.lru_cache(None)
def find_gpu_agent(device_id):
  assert device_id == 0, "FIXME"

  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_amdgpu_agent(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == hsa.HSA_DEVICE_TYPE_GPU:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      ret[0] = agent
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  hsa.hsa_iterate_agents(__filter_amdgpu_agent, ctypes.byref(agent := hsa.hsa_agent_t()))
  return agent

@functools.lru_cache(None)
def find_mem_zone(device_id, typ, req_flags):
  agent = find_gpu_agent(device_id)

  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_region_t, ctypes.c_void_p)
  def filter_shared_memtype(region, data):
    check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_region_segment_t())))
    if segment.value != typ:
      return hsa.HSA_STATUS_SUCCESS
    
    check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_GLOBAL_FLAGS, ctypes.byref(flags := hsa.hsa_region_global_flag_t())))
    if flags.value & req_flags:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_region_t))
      ret[0] = region
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  region = hsa.hsa_region_t()
  region.handle = -1
  hsa.hsa_agent_iterate_regions(agent, filter_shared_memtype, ctypes.byref(region))
  return region

class Kernel:
  def __init__(self, device_id, binary, kernel_name):
    self.device_id = device_id
    agent = find_gpu_agent(self.device_id)
    bin_size = len(binary)

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x))))
    check(hsa.hsa_code_object_reader_create_from_memory(binary, bin_size, ctypes.byref(code_reader := hsa.hsa_code_object_reader_t())))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, agent, code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    sym = kernel_name + ".kd"
    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, sym.encode("utf-8"), ctypes.byref(agent), ctypes.byref(x))))
    self.handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x))))
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x))))
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x))))
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x))))

exec_header = 0
exec_header |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
exec_header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
exec_header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
exec_header |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE
exec_setup = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS

barrier_header = 0
barrier_header |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
barrier_header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
barrier_header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
barrier_header |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE
barrier_setup = 0

empty_signal = hsa.hsa_signal_t()

class Queue:
  def __init__(self, device_id, sz=-1):
    check(hsa.hsa_init())
    self.agent = find_gpu_agent(device_id)
    
    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size, sz) if sz != -1 else max_queue_size

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(hsa.hsa_queue_create(self.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))
    self.write_addr = self.hw_queue.contents.base_address
    # self.hw_base_address = ctypes.cast(self.hw_queue.contents.base_address, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
    # self.write_addr = self.hw_queue.contents.base_address
    # print(self.hw_base_address, hex(self.write_addr))
    # print(type(self.write_addr))
    self.next_doorbell_index = -1

    self.hw_base_address = ctypes.cast(self.hw_queue.contents.base_address, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
    
    self.last_signal = None
    self.all_signals = []

    hsa.hsa_amd_queue_set_priority(self.hw_queue, hsa.HSA_AMD_QUEUE_PRIORITY_HIGH) # calls hsa.hsaKmtUpdateQueue()

    kernarg_region = find_mem_zone(device_id, hsa.HSA_REGION_SEGMENT_GLOBAL, hsa.HSA_REGION_GLOBAL_FLAG_KERNARG)
    self.kernargs = init_c_var(ctypes.c_void_p(), lambda x: check(hsa.hsa_memory_allocate(kernarg_region, 16 << 20, ctypes.byref(x)))).value

    # profiling, do not see any performance loss
    check(hsa.hsa_system_get_info(hsa.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.byref(gpu_freq := ctypes.c_uint64())))
    self.clocks_to_time = 1 / gpu_freq.value # TODO: double check
    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))

  def alloc_kernargs(self, sz):
    alignment = 16
    res = self.kernargs
    self.kernargs = (self.kernargs + sz + alignment - 1) & ~(alignment - 1)
    return res

  def submit_kernel(self, prg, global_size, local_size, kernargs, profile):
    if profile: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t())))

    # index = hsa.hsa_queue_load_write_index_scacquire(self.hw_queue)
    
    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.group_segment_size = prg.group_segment_size
    packet.private_segment_size = prg.private_segment_size
    packet.setup = exec_setup
    packet.header = exec_header
    if profile: packet.completion_signal = signal

    self.write_addr += 64
    self.next_doorbell_index += 1
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    if profile:
      hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(self.agent, signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      return (timings.end - timings.start) * self.clocks_to_time

  def submit_barrier(self, need_signal=False):
    check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t())))

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.dep_signal[0] = self.all_signals.pop() if len(self.all_signals) > 0 else empty_signal
    packet.dep_signal[1] = self.all_signals.pop() if len(self.all_signals) > 0 else empty_signal
    packet.dep_signal[2] = self.all_signals.pop() if len(self.all_signals) > 0 else empty_signal
    packet.dep_signal[3] = self.all_signals.pop() if len(self.all_signals) > 0 else empty_signal
    packet.dep_signal[4] = self.all_signals.pop() if len(self.all_signals) > 0 else empty_signal
    packet.completion_signal = signal
    packet.header = barrier_header

    self.all_signals.append(signal)
    self.write_addr += 64
    self.next_doorbell_index += 1
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

  def submit(self, cmds, need_signal=False):
    # index = None
    for cmd in cmds:
      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(self.last_signal))) # TODO: Better sync

      # self.st = ctypes.cast(self.write_addr, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
      # print(self.write_addr)

      # index = hsa.hsa_queue_add_write_index_screlease(self.hw_queue, 1)
      # print(index, self.next_doorbell_index)
      # dispatch_packet_ptr = ctypes.pointer(self.hw_base_address[index & (self.hw_queue.contents.size - 1)])
      # # print(self.hw_base_address[index & (self.hw_queue.contents.size - 1)])
      # print(self.hw_base_address[index & (self.hw_queue.contents.size - 1)])
      # print(ctypes.c_void_p(self.write_addr))
      # # dispatch_packet_ptr = ctypes.cast(ctypes.byref(self.hw_base_address[index & (self.hw_queue.contents.size - 1)]), ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
      # print(hex(self.write_addr), dispatch_packet_ptr)

      # print(self.hw_base_address[(self.next_doorbell_index + 1)])
      # print(hex(self.write_addr))
      # print(hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr))
      # dispatch_packet_ptr = ctypes.pointer(self.hw_base_address[(self.next_doorbell_index + 1)])
      # dispatch_packet_ptr_2 = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
      # dispatch_packet_ptr_2 = ctypes.byref(self.hw_base_address[(self.next_doorbell_index + 1) & (self.hw_queue.contents.size - 1)])
      # print(dispatch_packet_ptr, dispatch_packet_ptr.contents, dispatch_packet_ptr_2)
      # ptr = ctypes.c_void_p(self.write_addr)
      # dispatch_packet_ptr = ctypes.pointer(ctypes.c_void_p(self.write_addr))
      # dispatch_packet_ptr_2 = hsa_kernel_dispatch_packet_ptr(ctypes.c_void_p(self.write_addr))
      # print(dispatch_packet_ptr, dispatch_packet_ptr.contents, dispatch_packet_ptr_2)

      # dispatch_packet_ptr_2 = hsa_kernel_dispatch_packet_ptr.from_address(self.write_addr)

      # print(dispatch_packet_ptr, dispatch_packet_ptr_2)
      # print(hex(self.hw_base_address[(self.next_doorbell_index + 1) & (self.hw_queue.contents.size - 1)]))
      # print(dispatch_packet_ptr.contents, hex(self.write_addr))

      # Fill the packet for the given command.
      # if need_signal: self.hw_base_address[self.next_doorbell_index + 1].completion_signal = signal
      # cmd.fill_aql_packet(self, dispatch_packet_ptr)

      # dispatch_packet_ptr = ctypes.pointer(self.hw_base_address[self.next_doorbell_index + 1])
      # ctypes.cast(self.hw_queue.contents.base_address, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
      # print(dispatch_packet_ptr)
      packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
      # print(packet, hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr))
      packet.workgroup_size_x = cmd.local_size[0]
      packet.workgroup_size_y = cmd.local_size[1]
      packet.workgroup_size_z = cmd.local_size[2]
      packet.grid_size_x = cmd.global_size[0] * cmd.local_size[0]
      packet.grid_size_y = cmd.global_size[1] * cmd.local_size[1]
      packet.grid_size_z = cmd.global_size[2] * cmd.local_size[2]
      packet.kernel_object = cmd.prg.handle
      packet.kernarg_address = cmd.kernargs
      packet.group_segment_size = cmd.prg.group_segment_size
      packet.private_segment_size = cmd.prg.private_segment_size
      packet.setup = exec_setup
      packet.header = exec_header
      packet.completion_signal = self.last_signal

      self.write_addr += 64
      self.next_doorbell_index += 1

    hsa.hsa_signal_store_relaxed(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)
    # st = time.perf_counter()
    # self.wait() # FIXME
    # print(time.perf_counter()-st)

  # TODO: Better sync
  def __wait_for_last_signal(self):
    # if not self.all_signals: return
    for sig in self.all_signals:
      hsa.hsa_signal_wait_scacquire(sig, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    self.all_signals.clear()

  def wait(self):
    # while len(self.all_signals) > 1:
    #   self.submit_barrier(need_signal=True)
    self.submit_barrier(need_signal=True)
    self.__wait_for_last_signal()

class Command:
  def __init__(self): pass
  def fill_aql_packet(self, packet_ptr): pass

class ExecCommand(Command):
  def __init__(self, prg, global_size, local_size, kernargs):
    self.prg, self.global_size, self.local_size, self.kernargs = prg, global_size, local_size, kernargs

  # def fill_aql_packet(self, q, packet_ptr):
  #   # grid_size = tuple(int(g*l) for g,l in zip(self.global_size, self.local_size))

  #   # local = 

  #   packet = packet_ptr.contents
  #   # packet = hsa.hsa_kernel_dispatch_packet_t()
    
  #   packet.workgroup_size_x = self.local_size[0]
  #   packet.workgroup_size_y = self.local_size[1]
  #   packet.workgroup_size_z = self.local_size[2]
  #   packet.grid_size_x = self.global_size[0] * self.local_size[0]
  #   packet.grid_size_y = self.global_size[1] * self.local_size[1]
  #   packet.grid_size_z = self.global_size[2] * self.local_size[2]
  #   packet.kernel_object = self.prg.handle
  #   packet.kernarg_address = self.kernargs
  #   packet.group_segment_size = self.prg.group_segment_size
  #   packet.private_segment_size = self.prg.private_segment_size

  #   # header = 0
  #   # header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
  #   # header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  #   # header |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

  #   # cnt = max(1, sum(g > 1 for g in grid_size))
  #   packet.setup = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
  #   packet.header = exec_header

  #   # print(ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t))
  #   # ctypes.memmove(packet_ptr, ctypes.byref(packet), 64)

class CopyCommand(Command):
  def __init__(self): pass

def launch_kernel(dev, prg, global_size, local_size, args, vals, args_struct_t, profile=False):
  kernargs = None
  if prg.kernargs_segment_size.value > 0:
    kernargs = dev.hsa_queue.alloc_kernargs(prg.kernargs_segment_size.value)
    args_st = args_struct_t.from_address(kernargs)
    for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i])
    for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

  return dev.hsa_queue.submit_kernel(prg, global_size, local_size, kernargs, profile=profile)
  # if wait:
  #   dev.hsa_queue.wait()
  #   return float("inf")
