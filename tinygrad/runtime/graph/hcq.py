import ctypes, collections, time, itertools
from typing import List, Any, Dict, cast, Optional, Union, Tuple
from tinygrad.helpers import GraphException, init_c_var, round_up, colored
from tinygrad.buffer import Buffer, BufferOptions
from tinygrad.device import Compiled, CompiledRunner, BufferXfer, MultiDeviceJITGraph, Device, Runner
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem
from tinygrad.engine.jit import get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

class HCQGraph(Runner):
  def __init__(self, device_t, comp_q_t, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledRunner): compiled_devices.add(ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.bufs[0:2]: compiled_devices.add(Device[cast(Buffer, x).device])
      else: raise GraphException
    if any(not isinstance(d, device_t) for d in compiled_devices): raise GraphException

    self.devices: List[Compiled] = list(set(list(compiled_devices))) #type:ignore

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledRunner): kernargs_size[ji.prg.device] += round_up(ji.prg.clprg.kernargs_segment_size, 16)
    kernargs_ptrs: Dict[Compiled, int] = {dev:dev.allocator._alloc(sz, BufferOptions()).va_addr for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.kargs_addrs: Dict[int, ctypes.Structure] = {}
    self.ji_kargs_structs: Dict[int, ctypes.Structure] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue
      self.kargs_addrs[j] = kernargs_ptrs[ji.prg.device]
      kernargs_ptrs[ji.prg.device] += round_up(ji.prg.clprg.kernargs_segment_size, 16)

      self.ji_kargs_structs[j] = ji.prg.clprg.args_struct_t.from_address(self.kargs_addrs[j] + ji.prg.clprg.program_args_offset)
      for i in range(len(ji.bufs)): self.ji_kargs_structs[j].__setattr__(f'f{i}', cast(Buffer, ji.bufs[i])._buf.va_addr)
      for i in range(len(ji.prg.vars)): self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Build queues.
    self.comp_queues: Dict[Compiled, comp_q_t] = collections.defaultdict(comp_q_t)
    # self.copy_queues: Dict[Compiled, HWCopyQueue] = {dev:[] for dev in self.devices}

    # signal_size = ctypes.sizeof(hsa.amd_signal_t)
    # signals = device_t.signals_page = dev._gpu_alloc(SIGNAL_SIZE*SIGNAL_COUNT, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    # self.signals_pool = [hsa.amd_signal_t.from_address(device_t.signals_page.va_addr + SIGNAL_SIZE*i) for i in range(SIGNAL_COUNT)]
    # device_t.signals_page = self._gpu_alloc(SIGNAL_SIZE*64, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
    # device_t.signal_pool = [hsa.amd_signal_t.from_address(device_t.signals_page.va_addr + SIGNAL_SIZE*i) for i in range(SIGNAL_COUNT)]
    self.comp_queues_list: List[Any] = []
    self.chain_signal = device_t._get_signal()

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        # self.comp_queues_list.append((j,ji))
        if j in self.jc_idxs_with_updatable_launch_dims:
          if ji.prg.device in self.comp_queues: self.comp_queues_list.append((ji.prg.device, self.comp_queues.pop(ji.prg.device)))
          self.comp_queues_list.append((j,ji))
        else:
          self.comp_queues[ji.prg.device].wait(signal=self.chain_signal, value=j)
          self.comp_queues[ji.prg.device].exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.launch_dims(var_vals))
          self.comp_queues[ji.prg.device].signal(signal=self.chain_signal, value=j+1)
      # elif isinstance(ji.prg, BufferXfer):
      #   dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
      #   self.comp_queues[Device[src.device]].copy(dest._buf, src._buf, dest.nbytes)

    for dev in self.devices:
      self.comp_queues[dev].wait(signal=self.chain_signal, value=len(self.jit_cache)) # need this?
      if dev in self.comp_queues: self.comp_queues_list.append((dev, self.comp_queues.pop(dev)))

    # clear jit inputs to allow their memory to be freed/reused
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].bufs[i] = None
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), "HCQ", *get_jit_stats(jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Kick graph
    self.chain_signal.value = 0

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.ji_kargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf.va_addr)

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledRunner, self.jit_cache[j].prg).vars):
        self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    comp_q_t().wait(self.devices[0].compute_progress_signal, self.devices[0].compute_put_value).submit(self.devices[0])
    for dev,q in self.comp_queues_list:
      if not isinstance(q, comp_q_t):
        j,ji = dev,q
        q = comp_q_t()
        q.wait(signal=self.chain_signal, value=j)
        q.exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.launch_dims(var_vals))
        q.signal(signal=self.chain_signal, value=j+1)
        dev = ji.prg.device
      q.submit(dev)

    et = None
    return et
