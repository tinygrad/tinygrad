import collections, array, time
from typing import List, Any, Dict, cast, Optional, Tuple, Set
from tinygrad.helpers import GraphException, round_up, to_mv
from tinygrad.device import Buffer, BufferOptions, Compiled, Device
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, device_t, comp_hcq_t, copy_hcq_t, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.device_t, self.comp_hcq_t, self.copy_hcq_t = device_t, comp_hcq_t, copy_hcq_t

    # Check all jit items are compatible.
    self.devices = list(set(cast(self.device_t, d) for ji in jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs])) #type: ignore
    if any(not isinstance(d, self.device_t) for d in self.devices): raise GraphException

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.device] += round_up(ji.prg.clprg.kernargs_alloc_size, 16)
    kernargs_ptrs: Dict[Compiled, int] = {dev:dev.allocator._alloc(sz, BufferOptions(cpu_access=True)).va_addr for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.kargs_addrs: Dict[int, int] = {}
    self.ji_args_bufs: Dict[int, memoryview] = {}
    self.ji_args_vars: Dict[int, memoryview] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue
      self.kargs_addrs[j] = kernargs_ptrs[ji.prg.device]
      kernargs_ptrs[ji.prg.device] += round_up(ji.prg.clprg.kernargs_alloc_size, 16)

      self.ji_args_bufs[j] = to_mv(self.kargs_addrs[j] + ji.prg.clprg.kernargs_offset, len(ji.bufs) * 8).cast('Q')
      self.ji_args_vars[j] = to_mv(self.kargs_addrs[j] + ji.prg.clprg.kernargs_offset + len(ji.bufs) * 8, len(ji.prg.p.vars) * 4).cast('I')
      for i in range(len(ji.bufs)): self.ji_args_bufs[j][i] = cast(Buffer, ji.bufs[i])._buf.va_addr
      for i in range(len(ji.prg.p.vars)): self.ji_args_vars[j][i] = var_vals[ji.prg.p.vars[i]]

      # NV needs constbuffer to be set
      if ji.prg.device.dname.startswith("NV"): to_mv(self.kargs_addrs[j], 0x160).cast('I')[:] = array.array('I', ji.prg.clprg.constbuffer_0)

    # Build queues.
    self.comp_queues: Dict[Compiled, Any] = collections.defaultdict(self.comp_hcq_t)
    self.comp_signal = {dev: dev._get_signal(value=0) for dev in self.devices}
    self.comp_signal_val = {dev: 0 for dev in self.devices}

    self.copy_queues: Dict[Compiled, Any] = collections.defaultdict(self.copy_hcq_t)
    self.copy_signal = {dev: dev._get_signal(value=0) for dev in self.devices}
    self.copy_signal_val = {dev: 0 for dev in self.devices}

    self.kickoff_signal = self.devices[0]._get_signal(value=0)
    self.kickoff_value = 0
    self.graph_timeline = {dev: 0 for dev in self.devices}

    self.exec_ptrs: Dict[int, Tuple[Any, int]] = {}
    self.copy_to_devs: Dict[Compiled, Set[Compiled]] = {dev: set() for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(dev.timeline_signal, dev.timeline_value - 1).wait(self.kickoff_signal, self.kickoff_value)
      self.copy_queues[dev].wait(dev.timeline_signal, dev.timeline_value - 1).wait(self.kickoff_signal, self.kickoff_value)

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        exec_params = {}
        deps = self.access_resources(ji.bufs[(outs:=ji.prg.p.outcount):], ji.bufs[:outs], (self.comp_signal[ji.prg.device], sig_val:=j+1))
        deps = [x for x in deps if id(x[0]) != id(self.comp_signal[ji.prg.device])]

        # On NV, to synchronize kernel execution, we must either issue a wait or chain executions to schedule them in order.
        # Chaining executions is preferred when possible, as it is faster.
        if ji.prg.device.dname.startswith("NV"):
          if len(deps) == 0 and self.comp_signal_val[ji.prg.device] > 0:
            exec_params['chain_exec_ptr'] = self.exec_ptrs[self.comp_signal_val[ji.prg.device] - 1][1]
          else: deps.append((self.comp_signal[ji.prg.device], self.comp_signal_val[ji.prg.device]))

        for sig, val in deps: self.comp_queues[ji.prg.device].wait(sig, val)

        self.exec_ptrs[j] = (self.comp_queues[ji.prg.device], len(self.comp_queues[ji.prg.device]))
        self.comp_queues[ji.prg.device].exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.p.launch_dims(var_vals),
                                             signal=self.comp_signal[ji.prg.device], signal_value=sig_val, **exec_params)
        self.comp_signal_val[ji.prg.device] = sig_val
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        Device[src.device]._gpu_map(dest._buf) #type: ignore

        deps = self.access_resources([src], [dest], (self.copy_signal[Device[src.device]], sig_val:=j+1))
        deps.append((self.copy_signal[Device[src.device]], self.copy_signal_val[Device[src.device]]))
        self.copy_signal_val[Device[src.device]] = sig_val

        for sig,val in deps: self.copy_queues[Device[src.device]].wait(sig, val)
        self.copy_queues[Device[src.device]].copy(dest._buf.va_addr, src._buf.va_addr, dest.nbytes) \
                                            .signal(self.copy_signal[Device[src.device]], sig_val)
        self.copy_to_devs[Device[dest.device]].add(Device[src.device])

    for dev in self.devices:
      if self.copy_signal_val[dev] > 0: self.comp_queues[dev].wait(self.copy_signal[dev], self.copy_signal_val[dev])
      for dep_dev in self.copy_to_devs[dev]: self.comp_queues[dev].wait(self.copy_signal[dep_dev], self.copy_signal_val[dep_dev])

      self.comp_queues[dev].signal(dev.timeline_signal, dev.timeline_value)
      if hasattr(self.comp_queues[dev], 'bind'): self.comp_queues[dev].bind(dev)
      if hasattr(self.copy_queues[dev], 'bind') and self.copy_signal_val[dev] > 0: self.copy_queues[dev].bind(dev)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: dev._wait_signal(dev.timeline_signal, self.graph_timeline[dev])
    for dev in self.devices:
      dev._set_signal(self.comp_signal[dev], 0)
      dev._set_signal(self.copy_signal[dev], 0)
    dev._set_signal(self.kickoff_signal, self.kickoff_value)

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items(): self.ji_args_bufs[j][i] = input_rawbuffers[input_idx]._buf.va_addr

    # Update var_vals
    for j in self.jc_idx_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledRunner, self.jit_cache[j].prg).p.vars): self.ji_args_vars[j][i] = var_vals[v]

    for j in self.jc_idx_with_updatable_launch_dims:
      queue, cmd_ptr = self.exec_ptrs[j]
      queue.update_exec(cmd_ptr, *cast(CompiledRunner, self.jit_cache[j].prg).p.launch_dims(var_vals))

    for dev in self.devices:
      self.comp_queues[dev].update_wait(1, dev.timeline_signal, dev.timeline_value - 1).update_wait(2, value=self.kickoff_value) \
                           .update_signal(len(self.comp_queues[dev]) - 1, dev.timeline_signal, dev.timeline_value).submit(dev)

      if self.copy_signal_val[dev] > 0:
        self.copy_queues[dev].update_wait(0, dev.timeline_signal, dev.timeline_value - 1).update_wait(1, value=self.kickoff_value).submit(dev)

      self.graph_timeline[dev] = dev.timeline_value
      dev.timeline_value += 1

    if wait:
      st = time.perf_counter()
      for dev in self.devices: dev._wait_signal(dev.timeline_signal, self.graph_timeline[dev])
      return time.perf_counter() - st
    return None

  def access_resources(self, read, write, new_dependency):
    deps = self._access_resources(read, write, new_dependency)
    return [(k, max(v for x, v in deps if id(x) == idk)) for idk, k in {id(x[0]): x[0] for x in deps}.items()]
