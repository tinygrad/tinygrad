from __future__ import annotations
import ctypes, functools, subprocess, io, atexit
from typing import Tuple, TypeVar, List, Any, cast, Set
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import DEBUG, getenv, init_c_var
from tinygrad.helpers import from_mv, round_up, to_mv, colored, init_c_struct_t
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator, BufferOptions, JITRunner, Device, Buffer, update_stats, Compiler
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.runtime.compiler.hip_comgr import compile_hip

import gpuctypes.hsa as hsa
from tinygrad.runtime.driver.hsa import *

USAGE_EXEC = 0
USAGE_MEM = 1

class HSACompiler(Compiler):
  linearizer_opts = LinearizerOptions("HIP")
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

class HSAProgram:
  def __init__(self, device:HSADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x))))
    check(hsa.hsa_code_object_reader_create_from_memory(lib, len(lib), ctypes.byref(code_reader := hsa.hsa_code_object_reader_t())))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, self.device.agent, code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, (name+".kd").encode("utf-8"), ctypes.byref(self.device.agent), ctypes.byref(x))))
    self.handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x))))
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x))))
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x))))
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x))))

    check(hsa.hsa_code_object_reader_destroy(code_reader))

  def __del__(self):
    if hasattr(self, 'exec'): check(hsa.hsa_executable_destroy(self.exec))

  # @profile
  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] +
                                            [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      assert ctypes.sizeof(self.args_struct_t) == self.kernargs_segment_size.value, f"{ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size.value}"

    kernargs, wait_signals = None, [] if self.device.last_copy_signal is None else [self.device.last_copy_signal]
    if self.kernargs_segment_size.value > 0:
      kernargs = self.device.alloc_kernargs(self.kernargs_segment_size.value)
      args_st = self.args_struct_t.from_address(kernargs)
      for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i].value)
      for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])
    signal = self.device.hw_queue.submit_kernel(self, global_size, local_size, kernargs, signals=wait_signals)
    self.device.acquired_signals.extend(wait_signals)
    if wait:
      hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(self.device.agent, self.device.last_exec_signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      self.device.acquired_signals.append(signal)
      self.device.last_exec_signal = None
      return (timings.end - timings.start) * self.device.hw_queue.clocks_to_time
    else: self.device.last_exec_signal = signal


T = TypeVar("T")
class HSAAllocator(LRUAllocator):
  def __init__(self, device:HSADevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    check(hsa.hsa_amd_memory_pool_allocate(self.device.gpu_memory_pool, size, 0, ctypes.byref(buf := ctypes.c_void_p())))
    return buf

  def _free(self, opaque:T):
    self.device.synchronize()
    check(hsa.hsa_amd_memory_pool_free(opaque))

  def copyin(self, dest:T, src: memoryview):
    sdma_engine = self.device.select_sdma(len(src))
    agents = [HSADevice.cpu_agent, self.device.agent]
    c_agents = (hsa.hsa_agent_t * len(agents))(*agents)

    wait_signal, c_wait_signal = [], None
    if self.device.last_copy_signal:
      wait_signal.append(self.device.last_copy_signal)
      self.device.last_copy_signal = None
    if self.device.last_exec_signal:
      wait_signal.append(self.device.last_exec_signal)
      self.device.last_exec_signal = None
    if wait_signal: c_wait_signal = (hsa.hsa_signal_t * len(wait_signal))(*wait_signal)

    check(hsa.hsa_amd_memory_lock_to_pool(from_mv(src), src.nbytes, c_agents, len(agents), HSADevice.cpu_memory_pool, 0, ctypes.byref(src_addr := ctypes.c_void_p())))
    copy_signal = self.device.alloc_signal()
    check(hsa.hsa_amd_memory_async_copy_on_engine(dest, self.device.agent, src_addr, HSADevice.cpu_agent, src.nbytes, len(wait_signal), c_wait_signal, copy_signal, sdma_engine, True))
    self.device.last_copy_signal = copy_signal
    self.device.pending_copyin.append(src)
    self.device.acquired_signals.extend(wait_signal)

  def copyout(self, dest:memoryview, src:T):
    self.device.synchronize()
    agents = [HSADevice.cpu_agent, self.device.agent]
    c_agents = (hsa.hsa_agent_t * len(agents))(*agents)
    check(hsa.hsa_amd_memory_lock_to_pool(from_mv(dest), dest.nbytes, c_agents, len(agents), HSADevice.cpu_memory_pool, 0, ctypes.byref(addr := ctypes.c_void_p())))
    copy_signal = self.device.alloc_signal()
    check(hsa.hsa_amd_memory_async_copy(addr, HSADevice.cpu_agent, src, self.device.agent, dest.nbytes, 0, None, copy_signal))
    hsa.hsa_signal_wait_scacquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    check(hsa.hsa_amd_memory_unlock(from_mv(dest)))
    self.device.acquired_signals.append(copy_signal)

  def transfer(self, dest:T, src:T, sz:int): assert False, "not supported atm"

class HSADevice(Compiled):
  cpu_agent = None
  cpu_memory_pool = None
  def __init__(self, device:str=""):
    if not HSADevice.cpu_agent:
      check(hsa.hsa_init())
      atexit.register(lambda: hsa.hsa_shut_down())
      HSADevice.cpu_agent = find_hsa_agent(hsa.HSA_DEVICE_TYPE_CPU, device_id=0)
      HSADevice.cpu_memory_pool = find_memory_pool(HSADevice.cpu_agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_CPU)

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.agent = find_hsa_agent(hsa.HSA_DEVICE_TYPE_GPU, device_id=self.device_id)
    self.gpu_memory_pool = find_memory_pool(self.agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_GPU)
    self.kernargs_memory_pool = find_memory_pool(self.agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, flags=hsa.HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
    self.hw_queue = HWQueue(self, HSADevice.cpu_agent, HSADevice.cpu_memory_pool)

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_NAME, ctypes.byref(agent_name_buf := ctypes.create_string_buffer(256))))
    self.arch = ctypes.string_at(agent_name_buf).decode()

    self.kernarg_pool_sz = 64 << 20
    self.kernarg_ptr = init_c_var(ctypes.c_void_p(), lambda x: check(hsa.hsa_amd_memory_pool_allocate(self.kernargs_memory_pool, self.kernarg_pool_sz, 0, ctypes.byref(x)))).value
    check(hsa.hsa_amd_agents_allow_access(1, ctypes.byref(self.agent), None, self.kernarg_ptr))
    self.kernarg_next = self.kernarg_ptr

    self.signal_pool = []
    for _ in range(4096):
      check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(signal := hsa.hsa_signal_t())))
      self.signal_pool.append(signal)

    self.pending_copyin = []
    self.acquired_signals = []
    self.delayed_free = []
    self.next_sdma = 0

    # Pretend that we execute copy and exec in different streams, so we need to synchronise them.
    self.last_exec_signal = None
    self.last_copy_signal = None

    super().__init__(device, HSAAllocator(self), HSACompiler(self.arch), functools.partial(HSAProgram, self), None)

  def synchronize(self):
    if self.last_exec_signal: self.hw_queue.signals.append(self.last_exec_signal)
    if self.last_copy_signal: self.hw_queue.signals.append(self.last_copy_signal)
    self.hw_queue.wait()
    for opaque in self.pending_copyin: check(hsa.hsa_amd_memory_unlock(from_mv(opaque)))
    for opaque in self.delayed_free: check(hsa.hsa_amd_memory_pool_free(opaque))
    for sig in self.acquired_signals: hsa.hsa_signal_store_relaxed(sig, 1)
    self.signal_pool.extend(self.acquired_signals)
    self.pending_copyin.clear()
    self.delayed_free.clear()
    self.acquired_signals.clear()
    self.kernarg_next = self.kernarg_ptr

    self.last_exec_signal = None
    self.last_copy_signal = None

  def select_sdma(self, sz):
    self.next_sdma ^= 1
    return hsa.HSA_AMD_SDMA_ENGINE_0 if self.next_sdma == 0 else hsa.HSA_AMD_SDMA_ENGINE_1

  def alloc_signal(self):
    if len(self.signal_pool): return self.signal_pool.pop()
    check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(signal := hsa.hsa_signal_t())))
    return signal

  def alloc_kernargs(self, sz):
    # TODO: need something smarter here...
    result = self.kernarg_next
    self.kernarg_next = (self.kernarg_next + sz + 15) & (~15) # align to 16 bytes
    assert self.kernarg_next <= self.kernarg_ptr + self.kernarg_pool_sz, "no space for kernargs"
    return result
