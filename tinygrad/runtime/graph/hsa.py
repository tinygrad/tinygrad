from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2, GraphException
from tinygrad.device import Buffer, CompiledASTRunner, BufferXfer, update_stats
from tinygrad.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice

import gpuctypes.hsa as hsa
from tinygrad.runtime.driver.hsa import *

class HSAGraph:
  def __init__(self, device:HSADevice, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    # if not all(isinstance(ji.prg, CompiledASTRunner) for ji in jit_cache): raise GraphException # only execution for now

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.device: HSADevice = device

    self.packets_count = len(jit_cache) + 300
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
      for i in range(len(ji.rawbufs)): args_st.__setattr__(f'f{i}', ji.rawbufs[i]._buf)
      for i in range(len(ji.prg.vars)): args_st.__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Prepare signals
    self.finish_signal = self.device.alloc_signal() # This is a special signal, we cannot run this instance while it's running.
    # hsa.hsa_signal_store_relaxed(self.finish_signal, 0) # reset it

    # Build packets
    self.packets = []
    self.w_buffer_to_signal = {}
    self.r_buffer_to_signal = {}
    # print(self.buffer_to_signal)
    self.transfers = []
    self.signals_to_reset = [self.finish_signal]

    self.barrier_packet = self.add_barrier_packet([], EMPTY_SIGNAL)

    for j,ji in enumerate(self.jit_cache):
      packet = None
      if isinstance(ji.prg, CompiledASTRunner):
        wait_signals = []
        for i,rb in enumerate(ji.rawbufs):
          if i == 0 and rb._buf in self.r_buffer_to_signal:
            if isinstance(self.r_buffer_to_signal[rb._buf], hsa.hsa_signal_t):
              wait_signals.append(self.r_buffer_to_signal[rb._buf])
              self.r_buffer_to_signal.pop(rb._buf)
          if rb._buf in self.w_buffer_to_signal:
            if isinstance(self.w_buffer_to_signal[rb._buf], hsa.hsa_signal_t):
              wait_signals.append(self.w_buffer_to_signal[rb._buf])
              self.w_buffer_to_signal.pop(rb._buf)

        if len(wait_signals) > 0:
          for i in range(0, len(wait_signals), 5):
            self.add_barrier_packet(wait_signals[i:i+5], EMPTY_SIGNAL)

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
        packet.completion_signal = EMPTY_SIGNAL
        packet.setup = DISPATCH_KERNEL_SETUP
        packet.header = DISPATCH_KERNEL_HEADER

        for rb in ji.rawbufs:
          if i == 0: self.w_buffer_to_signal[rb._buf] = self.packet_off
          else: self.r_buffer_to_signal[rb._buf] = self.packet_off

        self.packets.append(packet)
        self.packet_off += 64
      elif isinstance(ji.prg, BufferXfer):
        dest, src = ji.rawbufs[0:2]
        assert dest.d != src.d
        device_is_source = (src.d == self.device)
        if ji.prg.reset_device == self.device: # TODO: might have sync problems in some cases.
          self.signals_to_reset.append(ji.prg.sync_signal)
          self.signals_to_reset.append(ji.prg.completion_signal)

        # q0 (src)                    q1 (dest)
        # wait <--------------------- sync when dest is ready
        # enqueue copy src->dest
        # barrier on src as output    barrier on dest (all cases)
        if device_is_source:
          # Wait for dest to be ready to accept signal
          wait_signal, c_wait_signal = [], None
          wait_signal.append(ji.prg.sync_signal)
          if src._buf in self.w_buffer_to_signal:
            # now we need a signal, patch it
            if isinstance(self.w_buffer_to_signal[src._buf], int):
              off = self.w_buffer_to_signal[src._buf]
              packet = hsa.hsa_kernel_dispatch_packet_t.from_address(off)
              if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
                self.w_buffer_to_signal[src._buf] = self.device.alloc_signal()
                self.signals_to_reset.append(self.w_buffer_to_signal[src._buf])
                packet.completion_signal = self.w_buffer_to_signal[src._buf]
              else: assert False
            else:
              self.w_buffer_to_signal[src._buf] = packet.completion_signal

            wait_signal.append(self.w_buffer_to_signal[src._buf])
            self.w_buffer_to_signal.pop(src._buf)

          sdma_engine = self.device.select_sdma(0)
          c_wait_signal = (hsa.hsa_signal_t * len(wait_signal))(*wait_signal)
          self.transfers.append((dest._buf, dest.d.agent, src._buf, src.d.agent, dest.nbytes, len(wait_signal), c_wait_signal, ji.prg.completion_signal, sdma_engine))
          self.r_buffer_to_signal[src._buf] = ji.prg.completion_signal
        else:
          wait_signal = []
          has_embed = False
          
          need_barrier = -1
          if dest._buf in self.r_buffer_to_signal:
            if isinstance(self.r_buffer_to_signal[dest._buf], int):
              need_barrier = max(need_barrier, self.r_buffer_to_signal[dest._buf])
              self.r_buffer_to_signal.pop(dest._buf)
            else:
              wait_signal.append(self.r_buffer_to_signal[dest._buf])
              self.r_buffer_to_signal.pop(dest._buf)
          if dest._buf in self.w_buffer_to_signal:
            if isinstance(self.w_buffer_to_signal[dest._buf], int):
              need_barrier = max(need_barrier, self.w_buffer_to_signal[dest._buf])
            else:
              wait_signal.append(self.w_buffer_to_signal[dest._buf])
              self.w_buffer_to_signal.pop(dest._buf)

          if need_barrier > 0:
            packet = hsa.hsa_kernel_dispatch_packet_t.from_address(need_barrier)
            if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
              packet.completion_signal = ji.prg.sync_signal
              has_embed = True

          if has_embed and len(wait_signal) == 0:
            pass
          else:
            self.add_barrier_packet(wait_signal, ji.prg.sync_signal)
          self.w_buffer_to_signal[dest._buf] = ji.prg.completion_signal


          # self.buffer_to_signal[dest._buf] = ji.prg.completion_signal
          # if dest._buf in self.buffer_to_signal:
          #   if isinstance(self.buffer_to_signal[dest._buf], int):
          #     off = self.buffer_to_signal[dest._buf]
          #     packet = hsa.hsa_kernel_dispatch_packet_t.from_address(off)
          #     if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
          #       packet.completion_signal = ji.prg.sync_signal
          #       has_embed = True
          #   else:
          #     wait_signal.append(self.buffer_to_signal[dest._buf])
          # if not has_embed: self.add_barrier_packet(wait_signal, ji.prg.sync_signal)
          # self.buffer_to_signal[dest._buf] = ji.prg.completion_signal

        self.packets.append(None) # so packet is easy to find with jit index.

      else:
        assert False, "not now"

    final_wait = []
    last_addr = 0
    rem = list(self.w_buffer_to_signal.values()) + list(self.r_buffer_to_signal.values())
    for v in rem:
      if isinstance(v, hsa.hsa_signal_t):
        final_wait.append(v)
      else:
        last_addr = max(last_addr, v)
    if len(final_wait) > 0:
      setj = False
      for i in range(0, len(final_wait), 5):
        if i+5 >= len(final_wait):
          assert setj == False
          setj = True
        self.add_barrier_packet(final_wait[i:i+5], self.finish_signal if i+5 >= len(final_wait) else EMPTY_SIGNAL)
      assert setj == True
    else:
      # assert last_addr != 0
      # packet = hsa.hsa_kernel_dispatch_packet_t.from_address(last_addr)
      # packet.completion_signal = self.finish_signal
      self.add_barrier_packet([], self.finish_signal)

    self.packets_count = (self.packet_off - self.c_aql_packets_addr) // 64
    for sig in self.signals_to_reset: hsa.hsa_signal_store_relaxed(sig, 0)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Wait and restore signal
    # hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    for sig in self.signals_to_reset:
      hsa.hsa_signal_wait_scacquire(sig, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      hsa.hsa_signal_store_relaxed(sig, 1)

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
      self.kernel_args_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf)

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

    for x in self.transfers:
      check(hsa.hsa_amd_memory_async_copy_on_engine(*x, True))

    et = None
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device=f"<GPU>:{self.device.device_id}")

  def add_barrier_packet(self, wait_signal, completion_signal):
    barrier_packet = hsa.hsa_barrier_and_packet_t.from_address(self.packet_off)
    barrier_packet.reserved0 = 0
    barrier_packet.reserved1 = 0
    barrier_packet.dep_signal[0] = wait_signal.pop() if len(wait_signal) > 0 else EMPTY_SIGNAL
    barrier_packet.dep_signal[1] = wait_signal.pop() if len(wait_signal) > 0 else EMPTY_SIGNAL
    barrier_packet.dep_signal[2] = wait_signal.pop() if len(wait_signal) > 0 else EMPTY_SIGNAL
    barrier_packet.dep_signal[3] = wait_signal.pop() if len(wait_signal) > 0 else EMPTY_SIGNAL
    barrier_packet.dep_signal[4] = wait_signal.pop() if len(wait_signal) > 0 else EMPTY_SIGNAL
    barrier_packet.reserved2 = 0
    barrier_packet.completion_signal = completion_signal
    barrier_packet.header = BARRIER_HEADER
    self.packet_off += 64
    assert len(wait_signal) == 0
    return barrier_packet