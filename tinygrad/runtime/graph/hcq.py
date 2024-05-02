import ctypes, collections, time, itertools
from typing import List, Any, Dict, cast, Optional, Union, Tuple
from tinygrad.helpers import GraphException, init_c_var, round_up, colored
from tinygrad.buffer import Buffer, BufferOptions
from tinygrad.device import Compiled, CompiledRunner, BufferXfer, Device
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, device_t, comp_hcq_t, copy_hcq_t, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.device_t, self.comp_hcq_t, self.copy_hcq_t = device_t, comp_hcq_t, copy_hcq_t
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledRunner): compiled_devices.add(ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.bufs[0:2]: compiled_devices.add(Device[cast(Buffer, x).device])
      else: raise GraphException
    if any(not isinstance(d, self.device_t) for d in compiled_devices): raise GraphException

    self.devices: List[Compiled] = list(set(list(compiled_devices))) #type:ignore

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledRunner): kernargs_size[ji.prg.device] += round_up(ji.prg.clprg.kernargs_segment_size, 16)
    kernargs_ptrs: Dict[Compiled, int] = {dev:dev.allocator._alloc(sz, BufferOptions(host=True)).va_addr for dev,sz in kernargs_size.items()}

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
    self.queues_list: List[Any] = []
    self.comp_queues: Dict[Compiled, self.comp_hcq_t] = collections.defaultdict(self.comp_hcq_t)
    self.comp_signal = {dev: dev._get_signal() for dev in self.devices}
    self.comp_signal_val = {dev: 0 for dev in self.devices}

    self.copy_queues: Dict[Compiled, self.copy_hcq_t] = collections.defaultdict(self.copy_hcq_t)
    self.copy_signal = {dev: dev._get_signal() for dev in self.devices}
    self.copy_signal_val = {dev: 0 for dev in self.devices}

    # Signal dma to allow execution.
    for dev in self.devices: self.comp_queues[dev].signal(self.copy_signal[dev], value=0)

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        deps = self.access_resources(ji.bufs[(outs:=ji.prg.outcount):], ji.bufs[:outs], (self.comp_signal[ji.prg.device], sig_val:=j+1))
        deps.append((self.comp_signal[ji.prg.device], self.comp_signal_val[ji.prg.device]))
        if j in self.jc_idx_with_updatable_launch_dims:
          # Rebuilt this runner dynamicaly.
          if ji.prg.device in self.comp_queues: self.queues_list.append((self.comp_queues.pop(ji.prg.device), ji.prg.device))
          self.queues_list.append((j, ji, deps))
        else:
          for sig, val in deps: self.comp_queues[ji.prg.device].wait(sig, val)
          self.comp_queues[ji.prg.device].exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.launch_dims(var_vals))
          self.comp_queues[ji.prg.device].signal(self.comp_signal[ji.prg.device], value=sig_val)
        self.comp_signal_val[ji.prg.device] = sig_val
      elif isinstance(ji.prg, BufferXfer):        
        # pass
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        src_dev = Device[src.device]
        deps = self.access_resources([src], [dest], (self.copy_signal[src_dev], sig_val:=j+1))
        deps.append((self.copy_signal[src_dev], self.copy_signal_val[src_dev]))
        for sig,val in deps: self.copy_queues[src_dev].wait(sig, val)
        self.copy_queues[src_dev].copy(dest._buf.va_addr, src._buf.va_addr, dest.nbytes)
        self.copy_queues[src_dev].signal(self.copy_signal[src_dev], value=sig_val)
        self.copy_signal_val[src_dev] = sig_val

    for dev in self.devices:
      self.comp_queues[dev].wait(self.comp_signal[dev], value=self.comp_signal_val[dev])
      if dev in self.copy_queues: self.comp_queues[dev].wait(self.copy_signal[dev], value=self.copy_signal_val[dev])
      self.queues_list.append((self.comp_queues.pop(dev), dev))
      if dev in self.copy_queues: self.queues_list.append((self.copy_queues.pop(dev), dev))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Kick graph
    for dev in self.devices: dev._set_signal_value(self.comp_signal[dev], 0)

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.ji_kargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf.va_addr)

    # Update var_vals
    for j in self.jc_idx_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledRunner, self.jit_cache[j].prg).vars):
        self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    for dev in self.devices: self.comp_hcq_t().wait(dev.compute_progress_signal, dev.compute_put_value).submit(dev)
    for pack in self.queues_list:
      if not isinstance(pack[0], self.comp_hcq_t) and not isinstance(pack[0], self.copy_hcq_t):
        j, ji, deps = pack
        q = self.comp_hcq_t()
        for sig, val in deps: q.wait(sig, val)
        q.exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.launch_dims(var_vals))
        q.signal(self.comp_signal[ji.prg.device], value=j+1)
        dev = ji.prg.device
      else: q, dev = pack
      q.submit(dev)

    for dev in self.devices: dev.synchronize()

    et = None
    return et

  def access_resources(self, read, write, new_dependency):
    deps = self._access_resources(read, write, new_dependency)
    return [(k, max(v for x, v in deps if id(x) == idk)) for idk, k in {id(x[0]): x[0] for x in deps}.items()]
