import collections, array, time
from typing import List, Any, Dict, cast, Optional, Tuple, Set
from tinygrad.helpers import round_up, to_mv, PROFILE
from tinygrad.device import Buffer, BufferOptions, Compiled, Device
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.devices = list(set(cast(Any, d) for ji in jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs]))

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.device] += round_up(ji.prg.clprg.kernargs_alloc_size, 16)
    self.kernargs_bufs: Dict[Compiled, Any] = {dev:dev.allocator._alloc(sz, BufferOptions(cpu_access=True)) for dev,sz in kernargs_size.items()}
    kernargs_ptrs: Dict[Compiled, int] = {dev:buf.va_addr for dev,buf in self.kernargs_bufs.items()}

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
    self.comp_queues: Dict[Compiled, Any] = {dev: dev.hw_compute_queue_t() for dev in self.devices}
    self.comp_signal = {dev: dev._get_signal(value=0) for dev in self.devices}
    self.comp_signal_val = {dev: 0 for dev in self.devices}

    self.copy_queues: Dict[Compiled, Any] = {dev: dev.hw_copy_queue_t() for dev in self.devices}
    self.copy_signal = {dev: dev._get_signal(value=0) for dev in self.devices}
    self.copy_signal_val = {dev: 0 for dev in self.devices}

    self.kickoff_signal = self.devices[0]._get_signal(value=0)
    self.copy_kickoff_signal = {dev: dev._get_signal(value=0) for dev in self.devices}
    self.kickoff_value = 0

    self.signal_sched: Dict[int, Tuple[List, Optional[int], Optional[Tuple]]] = {} # Dict[ji_idx, (deps, output sigval, (prof_st_sig, prof_en_sig))]
    self.exec_ptrs: Dict[int, Tuple[Any, int]] = {}
    self.copy_to_devs: Dict[Compiled, Set[Compiled]] = {dev: set() for dev in self.devices}
    self.graph_timeline = {dev: 0 for dev in self.devices}
    signal_save_devs: Dict[int, Set] = collections.defaultdict(set)
    self.queue_last_prg = collections.defaultdict(lambda : None)

    # Schedule dependencies
    for j,ji in enumerate(self.jit_cache):
      enqueue_queue = self.copy_queues[Device[ji.bufs[1].device]] if isinstance(ji.prg, BufferXfer) else self.comp_queues[ji.prg.device] #type:ignore
      if isinstance(ji.prg, CompiledRunner):
        deps = self.access_resources(ji.bufs[(outs:=ji.prg.p.outcount):], ji.bufs[:outs], (self.comp_signal[(dev:=ji.prg.device)], j + 1))
        if (last_j:=self.queue_last_prg[self.comp_queues[dev]]) is not None:
          deps = [x for x in deps if id(x[0]) != id(self.comp_signal[dev])] + [(self.comp_signal[dev], last_j + 1)]

        # Remove self-dependency for AMD or NV with only 1 same-queue dep, since NV chains 2+ execs in this case, eliminating dep need.
        if (dname:=dev.dname.split(":", 1)[0]) == "AMD" or (dname == "NV" and len(deps) == 1 and id(deps[0][0]) == id(self.comp_signal[dev])):
          deps = [x for x in deps if id(x[0]) != id(self.comp_signal[dev])]
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        deps = self.access_resources([src], [dest], (self.copy_signal[(dev:=Device[src.device])], j + 1))
        deps = [x for x in deps if id(x[0]) != id(self.copy_signal[Device[src.device]])]

      # When running compute, set up lazy signals, since no dependencies might be there. Copies always have signals to sync.
      signal_save_devs[j].update(((list(signal_save_devs[prev_j])+[dev]) if (prev_j:=self.queue_last_prg[enqueue_queue]) is not None else [dev]))
      for sig, val in deps: signal_save_devs[j].update(signal_save_devs[val - 1])

      # Update previous schedules in case we need thier signal now.
      for sig, val in deps: self.signal_sched[val - 1] = (self.signal_sched[val - 1][0], val, self.signal_sched[val - 1][2])

      # Transfer op accesses 2 buffers, src buffer is save, since it's on the device where op in enqueued, but also
      # need to be sure that dest buffer is save access.
      if isinstance(ji.prg, BufferXfer) and (dest_dev:=Device[ji.bufs[0].device]) not in signal_save_devs[j]:
        deps = [(self.copy_kickoff_signal[dest_dev], self.kickoff_value)] + deps
        signal_save_devs[j].update([dest_dev])
        # print(signal_save_devs[j])
      
      # Fill current signal sched entry.
      prof_ji_desc = ji.prg.clprg.name if isinstance(ji.prg, CompiledRunner) else f"{ji.bufs[1].device} -> {ji.bufs[0].device}" # type: ignore
      prof_info = (dev._get_signal(), dev._get_signal(), dev, prof_ji_desc, isinstance(ji.prg, BufferXfer)) if PROFILE else None
      self.signal_sched[j] = (deps, None if isinstance(ji.prg, CompiledRunner) else j + 1, prof_info)

      self.queue_last_prg[enqueue_queue] = j

    # Building hardware queues
    self.exec_ptrs: Dict[int, Tuple[Any, int]] = {}
    self.copy_to_devs: Dict[Compiled, Set[Compiled]] = {dev: set() for dev in self.devices}
    self.waits_cmds: Dict[Compiled, List] = {self.copy_queues[dev]: [0] for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(dev.timeline_signal, dev.timeline_value - 1).wait(self.kickoff_signal, self.kickoff_value) \
                           .signal(self.copy_kickoff_signal[dev], self.kickoff_value)
      self.copy_queues[dev].wait(self.copy_kickoff_signal[dev], self.kickoff_value)

    for j,ji in enumerate(self.jit_cache):
      deps, signal_value, prof_info = self.signal_sched[j]
      enqueue_queue = self.copy_queues[Device[ji.bufs[1].device]] if isinstance(ji.prg, BufferXfer) else self.comp_queues[ji.prg.device] #type:ignore

      # Encode waits and start profile timestamp (if needed).
      for sig, val in deps:
        enqueue_queue.wait(sig, val)
        if id(sig) in [id(x) for x in self.copy_kickoff_signal.values()]: self.waits_cmds[enqueue_queue].append(len(enqueue_queue) - 1)
      if prof_info: enqueue_queue.timestamp(prof_info[0])

      # Encode main commands based on ji type.
      if isinstance(ji.prg, CompiledRunner):
        self.comp_queues[ji.prg.device].exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.p.launch_dims(var_vals),
                                             signal=self.comp_signal[ji.prg.device] if signal_value is not None else None, signal_value=signal_value)
        self.exec_ptrs[j] = (self.comp_queues[ji.prg.device], len(self.comp_queues[ji.prg.device]) - 1)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        Device[src.device]._gpu_map(dest._buf) #type: ignore
        self.copy_queues[Device[src.device]].copy(dest._buf.va_addr, src._buf.va_addr, dest.nbytes) \
                                            .signal(self.copy_signal[Device[src.device]], signal_value)

      # Encode finish profile timestamp (if needed).
      if prof_info: enqueue_queue.timestamp(prof_info[1])

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        if (last_j:=self.queue_last_prg[self.copy_queues[dep_dev]]) is None: continue
        self.comp_queues[dev].wait(self.copy_signal[dep_dev], self.signal_sched[last_j][1])

      self.comp_queues[dev].signal(dev.timeline_signal, dev.timeline_value)
      if hasattr(self.comp_queues[dev], 'bind'): self.comp_queues[dev].bind(dev)
      if hasattr(self.copy_queues[dev], 'bind') and self.queue_last_prg[self.copy_queues[dev]] is not None: self.copy_queues[dev].bind(dev)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: dev._wait_signal(dev.timeline_signal, self.graph_timeline[dev])
    for dev in self.devices:
      dev._set_signal(self.comp_signal[dev], 0)
      dev._set_signal(self.copy_signal[dev], 0)
    self.devices[0]._set_signal(self.kickoff_signal, self.kickoff_value)

    if PROFILE and self.kickoff_value > 1:
      for _,_,(st,en,dev,desc,is_cp) in self.signal_sched.values(): #type: ignore
        dev.raw_prof_records += [(dev._read_timestamp(st), dev._read_timestamp(en), desc, is_cp)]

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
                           .update_signal(3, value=self.kickoff_value) \
                           .update_signal(len(self.comp_queues[dev]) - 1, dev.timeline_signal, dev.timeline_value).submit(dev)

      if self.queue_last_prg[(cp_queue:=self.copy_queues[dev])] is not None:
        # print(dev.device_id, self.copy_waits_cmds[dev])
        for cmd_idx in self.waits_cmds[cp_queue]: cp_queue.update_wait(cmd_idx, value=self.kickoff_value)
        cp_queue.submit(dev)

      self.graph_timeline[dev] = dev.timeline_value
      dev.timeline_value += 1
    # print()

    if wait:
      st = time.perf_counter()
      for dev in self.devices: dev._wait_signal(dev.timeline_signal, self.graph_timeline[dev])
      return time.perf_counter() - st
    return None

  def access_resources(self, read, write, new_dependency):
    deps = self._access_resources(read, write, new_dependency)
    return [(k, max(v for x, v in deps if id(x) == idk)) for idk, k in {id(x[0]): x[0] for x in deps}.items()]

  def __del__(self):
    # Graph is destructed. No need to keep signals any more, so return them as part of profiling.
    if PROFILE and self.kickoff_value > 1:
      for _,_,(st,en,dev,desc,is_cp) in self.signal_sched.values(): dev.sig_prof_records += [(st, en, desc, is_cp)] #type: ignore

    self.devices[0].signals_pool += [self.kickoff_signal] + list(self.copy_signal.values()) + list(self.comp_signal.values()) # type: ignore
    for dev, buf in self.kernargs_bufs.items(): dev.allocator._free(buf, BufferOptions(cpu_access=True))
