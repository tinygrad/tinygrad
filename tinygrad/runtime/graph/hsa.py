import ctypes, collections, time
from typing import List, Any, Dict, cast, Optional, Union
from tinygrad.helpers import GraphException, init_c_var
from tinygrad.device import Compiled, Buffer, CompiledASTRunner, BufferXfer, MultiDeviceJITGraph, update_stats
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.runtime.driver.hsa import check, AQLQueue, AQL_PACKET_SIZE, EMPTY_SIGNAL

def dedup_signals(signals): return [hsa.hsa_signal_t(hndl) for hndl in set([x.handle for x in signals if isinstance(x, hsa.hsa_signal_t)])]

class VirtAQLQueue(AQLQueue):
  def __init__(self, device, sz):
    self.device = device
    self.virt_queue = (hsa.hsa_kernel_dispatch_packet_t * sz)()
    self.queue_base = self.write_addr = ctypes.addressof(self.virt_queue)
    self.packets_count = 0
    self.available_packet_slots = sz
  def _wait_queue(self, need_packets=1): assert False, f"VirtQueue is too small to handle {self.packets_count+need_packets} packets!"
  def _submit_packet(self):
    self.write_addr += AQL_PACKET_SIZE
    self.packets_count += 1
    self.available_packet_slots -= 1
  def _alloc_signal(self, reusable=False): return init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(x))))

class HSAGraph(MultiDeviceJITGraph):
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache) #type:ignore
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): compiled_devices.add(ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.rawbufs[0:2]: compiled_devices.add(cast(Buffer, x).d)
      else: raise GraphException
    if any(not isinstance(d, HSADevice) for d in compiled_devices): raise GraphException

    self.devices: List[HSADevice] = list(compiled_devices) #type:ignore

    # Allocate kernel args.
    kernargs_size: Dict[HSADevice, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): kernargs_size[cast(HSADevice, ji.prg.device)] += (ctypes.sizeof(ji.prg.clprg.args_struct_t)+15) & ~15

    kernargs_ptrs: Dict[Compiled, int] = {}
    for dev,sz in kernargs_size.items():
      kernargs_ptrs[dev] = init_c_var(ctypes.c_void_p(),
                                      lambda x: check(hsa.hsa_amd_memory_pool_allocate(dev.kernargs_pool, sz, 0, ctypes.byref(x)))).value
      check(hsa.hsa_amd_agents_allow_access(1, ctypes.byref(dev.agent), None, kernargs_ptrs[dev]))

    # Fill initial arguments.
    self.ji_kargs_structs: Dict[int, ctypes.Structure] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledASTRunner): continue
      self.ji_kargs_structs[j] = ji.prg.clprg.args_struct_t.from_address(kernargs_ptrs[ji.prg.device])
      kernargs_ptrs[ji.prg.device] += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15
      for i in range(len(ji.rawbufs)): self.ji_kargs_structs[j].__setattr__(f'f{i}', cast(Buffer, ji.rawbufs[i])._buf)
      for i in range(len(ji.prg.vars)): self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Build queues.
    self.virt_aql_queues: Dict[Compiled, VirtAQLQueue] = {dev:VirtAQLQueue(dev, 2*len(self.jit_cache)+16) for dev in self.devices}
    self.packets = {}
    self.transfers = []
    self.signals_to_reset: List[hsa.hsa_signal_t] = []
    self.w_dependency_map: Dict[Any, Union[hsa.hsa_signal_t, hsa.hsa_agent_dispatch_packet_t]] = {}
    self.r_dependency_map: Dict[Any, Union[hsa.hsa_signal_t, hsa.hsa_agent_dispatch_packet_t]] = {}
    signals_to_devices: Dict[ctypes.c_uint64, List[HSADevice]] = {}

    # Special packet to wait for the world.
    self.kickoff_signals: Dict[HSADevice, hsa.hsa_signal_t] = {}
    for dev in self.devices: self.kickoff_signals[dev] = self.virt_aql_queues[dev].submit_barrier(need_signal=True)
    self.signals_to_reset += list(self.kickoff_signals.values())

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledASTRunner):
        wait_signals = self.access_resources(read=ji.rawbufs[1:], write=ji.rawbufs[0:1], new_dependency=j, sync_with_aql_packets=False)
        for i in range(0, len(wait_signals), 5):
          self.virt_aql_queues[ji.prg.device].submit_barrier(wait_signals=wait_signals[i:i+5])
        self.packets[j] = hsa.hsa_kernel_dispatch_packet_t.from_address(self.virt_aql_queues[ji.prg.device].write_addr)
        self.virt_aql_queues[ji.prg.device].submit_kernel(ji.prg.clprg, *ji.prg.launch_dims(var_vals), ctypes.addressof(self.ji_kargs_structs[j])) #type:ignore
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.rawbufs[0:2]]
        dest_dev, src_dev = cast(HSADevice, dest.d), cast(HSADevice, src.d)
        sync_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(sync_signal)
        signals_to_devices[sync_signal.handle] = [dest_dev, src_dev]

        wait_signals = self.access_resources(read=[src], write=[dest], new_dependency=sync_signal, sync_with_aql_packets=True)
        self.transfers.append((dest._buf, dest_dev.agent, src._buf, src_dev.agent, dest.nbytes, len(wait_signals),
                              (hsa.hsa_signal_t*len(wait_signals))(*wait_signals), sync_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True))

    # Wait for all active signals to finish the graph
    wait_signals_to_finish: Dict[HSADevice, List[hsa.hsa_signal_t]] = collections.defaultdict(list)
    for v in dedup_signals([s for s in list(self.w_dependency_map.values())+list(self.r_dependency_map.values()) if isinstance(s, hsa.hsa_signal_t)]):
      for dev in signals_to_devices[v.handle]:
        wait_signals_to_finish[dev].append(v)

    self.finish_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
    for dev in self.devices:
      wait_signals = wait_signals_to_finish[dev]
      for i in range(0, max(1, len(wait_signals)), 5):
        self.virt_aql_queues[dev].submit_barrier(wait_signals[i:i+5], need_signal=(i+5>=len(wait_signals)), completion_signal=self.finish_signal)

    # Zero signals to allow graph to start and execute.
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 0)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, 0)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Wait and restore signals
    hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, len(self.devices))

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.ji_kargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    # Update launch dims
    for j in self.jc_idxs_with_updatable_launch_dims:
      gl, lc = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.packets[j].workgroup_size_x = lc[0]
      self.packets[j].workgroup_size_y = lc[1]
      self.packets[j].workgroup_size_z = lc[2]
      self.packets[j].grid_size_x = gl[0] * lc[0]
      self.packets[j].grid_size_y = gl[1] * lc[1]
      self.packets[j].grid_size_z = gl[2] * lc[2]

    for dev in self.devices:
      dev.hw_queue.blit_packets(self.virt_aql_queues[dev].queue_base, self.virt_aql_queues[dev].packets_count)

    for transfer_data in self.transfers:
      check(hsa.hsa_amd_memory_async_copy_on_engine(*transfer_data))

    et = None
    if wait:
      st = time.perf_counter()
      hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      et = time.perf_counter() - st

    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device="HSA")
    return et

  def dependency_as_signal(self, dep, sync_with_aql_packets) -> Optional[hsa.hsa_signal_t]:
    if isinstance(dep, hsa.hsa_signal_t): return dep
    elif sync_with_aql_packets and isinstance(packet := self.packets.get(dep), hsa.hsa_kernel_dispatch_packet_t):
      if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
        packet.completion_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(packet.completion_signal)
      return packet.completion_signal
    return None

  def access_resources(self, read, write, new_dependency=None, sync_with_aql_packets=False):
    wait_signals = []
    for rawbuf in read:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
      if new_dependency: self.r_dependency_map[rawbuf._buf] = new_dependency
    for rawbuf in write:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
      wait_signals.append(self.dependency_as_signal(self.r_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
      if new_dependency: self.w_dependency_map[rawbuf._buf] = new_dependency
    if sync_with_aql_packets: wait_signals += [self.kickoff_signals[rawbuf.d] for rawbuf in read+write]
    return dedup_signals(wait_signals)
