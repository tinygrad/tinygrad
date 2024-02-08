from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2, GraphException
from tinygrad.device import Buffer, CompiledASTRunner, update_stats
from tinygrad.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice

import gpuctypes.hsa as hsa
from tinygrad.runtime.driver.hsa import *

class HSAGraph:
  def __init__(self, device:HSADevice, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    if not all(isinstance(ji.prg, CompiledASTRunner) for ji in jit_cache): raise GraphException # only execution for now

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.device: HSADevice = device

    self.packets_count = len(jit_cache) + 1
    self.c_aql_packets = (hsa.hsa_kernel_dispatch_packet_t * self.packets_count)()
    self.c_aql_packets_addr = ctypes.addressof(self.c_aql_packets)
    self.packet_off = ctypes.addressof(self.c_aql_packets)
    
    # Allocate one memory for all kern args of the graph
    kernarg_sz = 0
    for ji in self.jit_cache:
      kernarg_sz += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15 if isinstance(ji.prg, CompiledASTRunner) else 0

    self.kernarg_ptr = init_c_var(ctypes.c_void_p(), lambda x: check(hsa.hsa_amd_memory_pool_allocate(self.device.kernargs_memory_pool, kernarg_sz, 0, ctypes.byref(x)))).value
    check(hsa.hsa_amd_agents_allow_access(1, ctypes.byref(self.device.agent), None, self.kernarg_ptr))

    # Build initial arguments
    self.kernel_args_addr = {}
    self.kernel_args_structs = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledASTRunner): continue
      self.kernel_args_addr[j] = self.kernarg_ptr
      self.kernarg_ptr += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15

      args_st = ji.prg.clprg.args_struct_t.from_address(self.kernel_args_addr[j])
      self.kernel_args_structs[j] = args_st
      for i in range(len(ji.rawbufs)): args_st.__setattr__(f'f{i}', ji.rawbufs[i]._buf.value)
      for i in range(len(ji.prg.vars)): args_st.__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Prepare signals
    self.finish_signal = self.device.alloc_signal() # This is a special signal, we cannot run this instance while it's running.
    hsa.hsa_signal_store_relaxed(self.finish_signal, 0) # reset it

    # Build packets
    self.packets = []

    print("packet off", self.packet_off)

    self.barrier_packet = hsa.hsa_barrier_and_packet_t.from_address(self.packet_off)
    self.barrier_packet.reserved0 = 0
    self.barrier_packet.reserved1 = 0
    self.barrier_packet.dep_signal[0] = EMPTY_SIGNAL
    self.barrier_packet.dep_signal[1] = EMPTY_SIGNAL
    self.barrier_packet.dep_signal[2] = EMPTY_SIGNAL
    self.barrier_packet.dep_signal[3] = EMPTY_SIGNAL
    self.barrier_packet.dep_signal[4] = EMPTY_SIGNAL
    self.barrier_packet.reserved2 = 0
    self.barrier_packet.completion_signal = EMPTY_SIGNAL
    self.barrier_packet.header = BARRIER_HEADER
    self.packet_off += 64

    for j,ji in enumerate(self.jit_cache):
      packet = None
      if isinstance(ji.prg, CompiledASTRunner):
        global_size, local_size = ji.prg.launch_dims(var_vals)
        packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.packet_off)
        packet.workgroup_size_x = local_size[0]
        packet.workgroup_size_y = local_size[1]
        packet.workgroup_size_z = local_size[2]
        packet.reserved0 = 0
        packet.grid_size_x = global_size[0] * local_size[0]
        packet.grid_size_y = global_size[1] * local_size[1]
        packet.grid_size_z = global_size[2] * local_size[2]
        packet.private_segment_size = ji.prg.clprg.private_segment_size
        packet.group_segment_size = ji.prg.clprg.group_segment_size
        packet.kernel_object = ji.prg.clprg.handle
        packet.kernarg_address = self.kernel_args_addr[j]
        packet.reserved2 = 0
        packet.completion_signal = self.finish_signal if j == len(self.jit_cache)-1 else EMPTY_SIGNAL
        packet.setup = DISPATCH_KERNEL_SETUP
        packet.header = DISPATCH_KERNEL_HEADER
      else:
        assert False, "not now"
      self.packets.append(packet)
      self.packet_off += 64

    print("done init")
    self.signals_pool = []

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Wait and restore signal
    hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    hsa.hsa_signal_store_relaxed(self.finish_signal, 1)

    # Wait for signals
    wait_signal = []
    if self.device.last_copy_signal:
      wait_signal.append(self.device.last_copy_signal)
      self.device.acquired_signals.append(self.device.last_copy_signal)
      self.device.last_copy_signal = None
    if self.device.last_transfer_signal:
      wait_signal.append(self.device.last_transfer_signal)
      self.device.last_transfer_signal = None
    if self.device.last_exec_signal:
      wait_signal.append(self.device.last_exec_signal)
      self.device.acquired_signals.append(self.device.last_exec_signal)
      self.device.last_exec_signal = None
    for i in range(3): self.barrier_packet.dep_signal[i] = wait_signal[i] if i < len(wait_signal) else EMPTY_SIGNAL

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.kernel_args_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf.value)

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        self.kernel_args_structs[j].__setattr__(f'v{i}', var_vals[v])

    # Update launch dims
    for j in self.jc_idxs_with_updatable_launch_dims:
      gl, lc = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.packets[j].workgroup_size_x = lc[0]
      self.packets[j].workgroup_size_y = lc[1]
      self.packets[j].workgroup_size_z = lc[2]
      self.packets[j].grid_size_x = gl[0] * lc[0]
      self.packets[j].grid_size_y = gl[1] * lc[1]
      self.packets[j].grid_size_z = gl[2] * lc[2]

    self.device.last_transfer_signal = self.finish_signal # HACK to not return the signal to device
    self.device.hw_queue.blit(self.c_aql_packets_addr, self.packets_count)
    et = None
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device=f"<GPU>:{self.device}")
