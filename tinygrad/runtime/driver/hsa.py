import ctypes, functools, time
import gpuctypes.hsa as hsa
from tinygrad.helpers import init_c_var, getenv

DEBUG_HSA = getenv("DEBUG_HSA", 0)

def check(status):
  if status != 0: raise RuntimeError(f"HSA Error {status}")

# Precalulated AQL info
AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
EMPTY_SIGNAL = hsa.hsa_signal_t()

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER = 0
DISPATCH_KERNEL_HEADER |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE # HSA_FENCE_SCOPE_AGENT?
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER = 0
BARRIER_HEADER |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

PM4_HDR_IT_OPCODE_NOP = 0x10
PM4_HDR_IT_OPCODE_INDIRECT_BUFFER = 0x3F
PM4_HDR_IT_OPCODE_RELEASE_MEM = 0x49
PM4_HDR_IT_OPCODE_ACQUIRE_MEM = 0x58

def PM4_HDR_SHADER_TYPE(x): return (x & 0x1) << 1
def PM4_HDR_IT_OPCODE(x): return (x & 0xFF) << 8
def PM4_HDR_COUNT(x): return (x & 0x3FFF) << 16
def PM4_HDR_TYPE(x): return (x & 0x3) << 30
def PM4_HDR(it_opcode, pkt_size_dw, gfxip_ver):
    return (PM4_HDR_SHADER_TYPE(1 if gfxip_ver == 7 else 0) |
            PM4_HDR_IT_OPCODE(it_opcode) |
            PM4_HDR_COUNT(pkt_size_dw - 2) |
            PM4_HDR_TYPE(3))
def PM4_INDIRECT_BUFFER_DW1_IB_BASE_LO(x): return (x & 0x3FFFFFFF) << 2
def PM4_INDIRECT_BUFFER_DW2_IB_BASE_HI(x): return (x & 0xFFFF) << 0
def PM4_INDIRECT_BUFFER_DW3_IB_SIZE(x): return (x & 0xFFFFF) << 0
def PM4_INDIRECT_BUFFER_DW3_IB_VALID(x): return (x & 0x1) << 23

def PM4_ACQUIRE_MEM_DW1_COHER_CNTL(x): return (x & 0x7FFFFFFF) << 0
PM4_ACQUIRE_MEM_COHER_CNTL_TC_WB_ACTION_ENA = 1 << 18
PM4_ACQUIRE_MEM_COHER_CNTL_TC_ACTION_ENA = 1 << 23
PM4_ACQUIRE_MEM_COHER_CNTL_SH_KCACHE_ACTION_ENA = 1 << 27
PM4_ACQUIRE_MEM_COHER_CNTL_SH_ICACHE_ACTION_ENA = 1 << 29
def PM4_ACQUIRE_MEM_DW2_COHER_SIZE(x): return (x & 0xFFFFFFFF) << 0
def PM4_ACQUIRE_MEM_DW3_COHER_SIZE_HI(x): return (x & 0xFF) << 0
def PM4_ACQUIRE_MEM_DW7_GCR_CNTL(x): return (x & 0x7FFFF) << 0
def PM4_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x): return (x & 0x3) << 0
PM4_ACQUIRE_MEM_GCR_CNTL_GLK_INV = 1 << 7
PM4_ACQUIRE_MEM_GCR_CNTL_GLV_INV = 1 << 8
PM4_ACQUIRE_MEM_GCR_CNTL_GL1_INV = 1 << 9
PM4_ACQUIRE_MEM_GCR_CNTL_GL2_INV = 1 << 14
def PM4_RELEASE_MEM_DW1_EVENT_INDEX(x): return (x & 0xF) << 8
PM4_RELEASE_MEM_EVENT_INDEX_AQL = 0x7

class AmdAqlPm4Ib(ctypes.Structure):
    _fields_ = [
        ("header", ctypes.c_uint16),
        ("amd_format", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("ib_jump_cmd", ctypes.c_uint32 * 4),
        ("dw_cnt_remain", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 8),
        ("completion_signal", hsa.hsa_signal_t)
    ]
    _pack_ = 1
amd_aql_pm4_ib_t = AmdAqlPm4Ib


class HWQueue:
  def __init__(self, dev, cpu_agent, cpu_memory_pool, sz=-1):
    self.dev = dev
    self.agent = dev.agent
    self.signals = []

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size.value, sz) if sz != -1 else max_queue_size.value

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(hsa.hsa_queue_create(self.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))

    self.write_addr = self.hw_queue.contents.base_address
    self.write_end = self.hw_queue.contents.base_address + (64 * self.hw_queue.contents.size) - 1
    self.next_doorbell_index = -1

    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))
    check(hsa.hsa_system_get_info(hsa.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.byref(gpu_freq := ctypes.c_uint64())))
    self.clocks_to_time = 1 / gpu_freq.value # TODO: double check

    agents = (hsa.hsa_agent_t * 2)(self.agent, cpu_agent)
    check(hsa.hsa_amd_memory_pool_allocate(cpu_memory_pool, 0x1000, 0, ctypes.byref(pm4_buf := ctypes.c_void_p())))
    check(hsa.hsa_amd_agents_allow_access(2, agents, None, pm4_buf))
    self.pm4_buf = ctypes.cast(pm4_buf, ctypes.POINTER(ctypes.c_uint32))
    self.pm4_buf_address = pm4_buf.value

  def __del__(self): pass # TODO

  def submit_kernel(self, prg, global_size, local_size, kernargs, signals=None):
    if DEBUG_HSA >= 2: print(f"queue.submit_kernel({global_size=}, {local_size=})")

    if signals is not None and len(signals):
      for i in range(0, len(signals), 5): self.submit_barrier(wait_signals=signals[i:i+5], need_signal=False)

    signal = self.dev.alloc_signal()

    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.reserved0 = 0
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.private_segment_size = prg.private_segment_size
    packet.group_segment_size = prg.group_segment_size
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER

    self.next_doorbell_index += 1
    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_end:
      self.write_addr = self.hw_queue.contents.base_address
    hsa.hsa_queue_store_write_index_relaxed(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    return signal

  def submit_barrier(self, wait_signals=None, need_signal=True):
    if DEBUG_HSA >= 2: print(f"queue.submit_barrier({wait_signals=})")

    assert wait_signals is None or len(wait_signals) < 5
    if need_signal: signal = self.dev.alloc_signal()

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.reserved0 = 0
    packet.reserved1 = 0
    packet.dep_signal[0] = wait_signals[0] if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.dep_signal[1] = wait_signals[1] if wait_signals and len(wait_signals) > 1 else EMPTY_SIGNAL
    packet.dep_signal[2] = wait_signals[2] if wait_signals and len(wait_signals) > 2 else EMPTY_SIGNAL
    packet.dep_signal[3] = wait_signals[3] if wait_signals and len(wait_signals) > 3 else EMPTY_SIGNAL
    packet.dep_signal[4] = wait_signals[4] if wait_signals and len(wait_signals) > 4 else EMPTY_SIGNAL
    packet.reserved2 = 0
    if need_signal: packet.completion_signal = signal
    packet.header = BARRIER_HEADER

    if need_signal: self.signals.append(signal)
    self.next_doorbell_index += 1
    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_end: 
      self.write_addr = self.hw_queue.contents.base_address
    hsa.hsa_queue_store_write_index_screlease(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

  def submit_pm4(self, cmd, cmd_size_dw):
    # TODO: check isa version, this works only for isa >= 9
    for i in range(cmd_size_dw): self.pm4_buf[i] = cmd[i]

    signal = self.dev.alloc_signal()

    # self.next_doorbell_index = hsa.hsa_queue_load_write_index_relaxed(self.hw_queue)

    ctypes.memset(self.write_addr, 0, 64)
    packet = amd_aql_pm4_ib_t.from_address(self.write_addr)
    packet.header = hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE
    packet.amd_format = 0x1 # AMD_AQL_FORMAT_PM4_IB

    packet.ib_jump_cmd[0] = PM4_HDR(PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, 4, 11)
    packet.ib_jump_cmd[1] = PM4_INDIRECT_BUFFER_DW1_IB_BASE_LO(self.pm4_buf_address >> 2)
    packet.ib_jump_cmd[2] = PM4_INDIRECT_BUFFER_DW2_IB_BASE_HI(self.pm4_buf_address >> 32)
    packet.ib_jump_cmd[3] = PM4_INDIRECT_BUFFER_DW3_IB_SIZE(cmd_size_dw) | PM4_INDIRECT_BUFFER_DW3_IB_VALID(1)
    packet.dw_cnt_remain = 0xA
    packet.completion_signal.handle = signal.handle

    self.next_doorbell_index += 1
    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_end: 
      self.write_addr = self.hw_queue.contents.base_address
    hsa.hsa_queue_store_write_index_screlease(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    self.dev.acquired_signals.append(signal)

    for i in range(cmd_size_dw): self.pm4_buf[i] = 0

  def wait(self):
    if DEBUG_HSA >= 2: print("queue.wait()")
    self.submit_barrier()
    for sig in self.signals:
      if DEBUG_HSA >= 4:
        # Debug active wait
        while hsa.hsa_signal_load_scacquire(sig) != 0:
          print("rw", hsa.hsa_queue_load_read_index_scacquire(self.hw_queue), hsa.hsa_queue_load_write_index_scacquire(self.hw_queue))
      else:
        hsa.hsa_signal_wait_scacquire(sig, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      self.dev.acquired_signals.append(sig)
    self.signals.clear()

  def blit(self, packet_addr, packet_cnt):
    # TODO: support overflow
    ctypes.memmove(self.write_addr, packet_addr, AQL_PACKET_SIZE * packet_cnt)

    self.next_doorbell_index += packet_cnt
    self.write_addr += AQL_PACKET_SIZE * packet_cnt
    hsa.hsa_queue_store_write_index_screlease(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

@functools.lru_cache(None)
def find_hsa_agent(typ, device_id):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_agent(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == typ:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      if ret[0].handle < device_id:
        ret[0].handle = ret[0].handle + 1
        return hsa.HSA_STATUS_SUCCESS

      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      ret[0] = agent
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  agent = hsa.hsa_agent_t()
  agent.handle = 0
  hsa.hsa_iterate_agents(__filter_agent, ctypes.byref(agent))
  return agent

def find_memory_pool(agent, segtyp=-1, flags=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    if segtyp != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
      if segment.value != segtyp: return hsa.HSA_STATUS_SUCCESS

    if flags != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(fgs := hsa.hsa_amd_memory_pool_global_flag_t())))
      if (fgs.value & flags) == flags: return hsa.HSA_STATUS_SUCCESS

    if location != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc := hsa.hsa_amd_memory_pool_location_t())))
      if loc.value != location: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SIZE, ctypes.byref(sz := ctypes.c_size_t())))
    if sz == 0: return hsa.HSA_STATUS_SUCCESS

    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  region = hsa.hsa_amd_memory_pool_t()
  region.handle = 0
  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region))
  return region

def find_old_mem_zone(agent, typ, req_flags):
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
