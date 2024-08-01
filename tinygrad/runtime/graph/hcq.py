import collections, time
from typing import List, Any, Dict, cast, Optional, Tuple, Set
from tinygrad.helpers import round_up, to_mv, PROFILE
from tinygrad.device import HCQCompiled, HCQAllocator, HCQSignal, HCQBuffer, HWCommandQueue, HWComputeQueue, HWCopyQueue, \
                            Buffer, BufferOptions, Compiled, Device
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.devices = list(set(cast(HCQCompiled, d) for ji in jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs]))

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.device] += round_up(ji.prg.clprg.kernargs_alloc_size, 16)
    self.kernargs_bufs: Dict[Compiled, HCQBuffer] = {dev:dev.allocator._alloc(sz, BufferOptions(cpu_access=True)) for dev,sz in kernargs_size.items()}
    kernargs_ptrs: Dict[Compiled, int] = {dev:buf.va_addr for dev,buf in self.kernargs_bufs.items()}

    # Fill initial arguments.
    self.kargs_addrs: Dict[int, int] = {}
    self.ji_args_bufs: Dict[int, memoryview] = {}
    self.ji_args_vars: Dict[int, memoryview] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue
      self.kargs_addrs[j] = kernargs_ptrs[ji.prg.device]
      kernargs_ptrs[ji.prg.device] += round_up(ji.prg.clprg.kernargs_alloc_size, 16)

      ji.prg.clprg.fill_kernargs([cast(Buffer, b)._buf for b in ji.bufs], [var_vals[v] for v in ji.prg.p.vars], self.kargs_addrs[j])
      self.ji_args_bufs[j] = to_mv(self.kargs_addrs[j] + ji.prg.clprg.kernargs_args_offset, len(ji.bufs) * 8).cast('Q')
      self.ji_args_vars[j] = to_mv(self.kargs_addrs[j] + ji.prg.clprg.kernargs_args_offset + len(ji.bufs) * 8, len(ji.prg.p.vars) * 4).cast('I')

    # Schedule Dependencies.
    # There are two types of queues on each device: copy and compute. Both must synchronize with all external operations before launching any
    # graph-related tasks. This synchronization uses a global timeline signal per device. Within the graph, the compute queue coordinates with
    # global operations and sets a kickoff signal. Any queue accessing a buffer from another device waits for this signal from the device’s
    # compute queue to ensure exclusive access. The compute queue signals the completion of the graph, synchronizing with the device's copy queue.
    self.comp_queues: Dict[Compiled, HWComputeQueue] = {dev: dev.hw_compute_queue_t() for dev in self.devices}
    self.copy_queues: Dict[Compiled, HWCopyQueue] = {dev: dev.hw_copy_queue_t() for dev in self.devices}

    self.signal_sched: Dict[int, Tuple[List, HCQSignal, Optional[int], Optional[List]]] = {} # Dict[ji_idx, (deps, signal, sigval, prof_info)]
    self.signals = {q: dev.signal_t(value=0) for queues in (self.comp_queues, self.copy_queues) for dev,q in queues.items()} #type:ignore
    self.dev_kickoff_signal = {**{dev.dname: dev.signal_t(value=0) for dev in self.devices}, **{"CPU": self.devices[0].signal_t(value=0)}}
    self.kickoff_value = 0

    self.save_devs: Dict[HWCommandQueue, Set] = {q: set() for q in list(self.comp_queues.values()) + list(self.copy_queues.values())}
    for dev in self.devices: self.save_devs[self.comp_queues[dev]].add(dev)

    self.last_timeline: Dict[HCQCompiled, Tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}
    self.last_ji: Dict[HWCommandQueue, Optional[int]] = {q: None for q in list(self.comp_queues.values()) + list(self.copy_queues.values())}
    self.prof_signals: List[HCQSignal] = [self.devices[0].signal_t() for i in range(len(self.jit_cache) * 2)] if PROFILE else []
    self.prof_records: List

    for j,ji in enumerate(self.jit_cache):
      enqueue_dev = ji.prg.device if isinstance(ji.prg, CompiledRunner) else Device[ji.bufs[1].device] #type:ignore
      enqueue_queue = self.comp_queues[enqueue_dev] if isinstance(ji.prg, CompiledRunner) else self.copy_queues[enqueue_dev]
      out_signal = self.signals[enqueue_queue] #type:ignore
      writable_buffers = ji.prg.p.outcount if isinstance(ji.prg, CompiledRunner) else 1
      deps = self.access_resources(enqueue_queue, ji.bufs[writable_buffers:], ji.bufs[:writable_buffers], j + 1)

      # Profiler related info
      prof_ji_desc = ji.prg.clprg.name if isinstance(ji.prg, CompiledRunner) else f"{ji.bufs[1].device} -> {ji.bufs[0].device}" # type: ignore
      prof_info = ([(j * 2, True), (j * 2 + 1, True), enqueue_dev, prof_ji_desc, isinstance(ji.prg, BufferXfer)]) if PROFILE else None

      if isinstance(ji.prg, CompiledRunner):
        # Update signal on compute kernel to depend on the previous kernel.
        if (last_j:=self.last_ji[enqueue_queue]) is not None: deps = [x for x in deps if id(x[0]) != id(out_signal)] + [(out_signal, last_j + 1)]

        # Remove self-dependency for AMD or NV with only 1 same-queue dep, since NV chains 2+ execs in this case, eliminating dep need.
        if (dname:=enqueue_dev.dname.split(":", 1)[0]) == "AMD" or (dname == "NV" and len(deps) == 1 and id(deps[0][0]) == id(out_signal)):
          deps = [x for x in deps if id(x[0]) != id(out_signal)]
          if last_j is not None and prof_info is not None: prof_info = [(self.signal_sched[last_j][3][1][0], False)] + prof_info[1:] # type: ignore

      elif isinstance(ji.prg, BufferXfer): deps = [x for x in deps if id(x[0]) != id(out_signal)]

      # Go through all dependencies and, if we need the signal from that ji, enable it by setting the signal value in the signal schedule.
      for sig, val in deps:
        if id(sig) in [id(x) for x in self.signals.values()]:
          self.signal_sched[val - 1] = self.signal_sched[val - 1][:2] + (val,) + self.signal_sched[val - 1][3:]

      self.signal_sched[j] = (deps, out_signal, None if isinstance(ji.prg, CompiledRunner) else (j + 1), prof_info)
      self.last_ji[enqueue_queue] = j

    # Build hardware queues.
    self.op_cmd_idx: Dict[int, Tuple[Any, int]] = {}
    self.copy_to_devs: Dict[Compiled, Set[Compiled]] = {dev: set() for dev in self.devices}
    self.kickoff_wait_cmds: Dict[HWCommandQueue, List] = {q: list() for q in list(self.comp_queues.values()) + list(self.copy_queues.values())}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(dev.timeline_signal, dev.timeline_value - 1) \
                           .wait(self.dev_kickoff_signal['CPU'], self.kickoff_value).signal(self.dev_kickoff_signal[dev.dname], self.kickoff_value)

    for j,ji in enumerate(self.jit_cache):
      deps, signal, signal_val, prof_info = self.signal_sched[j]
      enqueue_queue = self.copy_queues[Device[ji.bufs[1].device]] if isinstance(ji.prg, BufferXfer) else self.comp_queues[ji.prg.device] #type:ignore

      # Encode waits and start profile timestamp (if needed).
      for sig, val in deps:
        enqueue_queue.wait(sig, val)
        if id(sig) in [id(x) for x in self.dev_kickoff_signal.values()]: self.kickoff_wait_cmds[enqueue_queue].append(len(enqueue_queue) - 1)
      if prof_info and prof_info[0][1]: enqueue_queue.timestamp(self.prof_signals[prof_info[0][0]])

      # Encode main commands based on ji type.
      if isinstance(ji.prg, CompiledRunner):
        cast(HWComputeQueue, enqueue_queue).exec(ji.prg.clprg, self.kargs_addrs[j], *ji.prg.p.launch_dims(var_vals))
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        cast(HCQAllocator, Device[src.device].allocator).map(dest._buf)
        cast(HWCopyQueue, enqueue_queue).copy(dest._buf.va_addr, src._buf.va_addr, dest.nbytes)
        self.copy_to_devs[Device[dest.device]].add(Device[src.device])
      self.op_cmd_idx[j] = (enqueue_queue, len(enqueue_queue) - 1)

      # Encode finish profile timestamp (if needed).
      if prof_info and prof_info[1][1]: enqueue_queue.timestamp(self.prof_signals[prof_info[1][0]])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        if (last_j:=self.last_ji[self.copy_queues[dep_dev]]) is None: continue
        self.comp_queues[dev].wait(self.signals[self.copy_queues[dep_dev]], self.signal_sched[last_j][2])

      self.comp_queues[dev].signal(dev.timeline_signal, dev.timeline_value)
      self.comp_queues[dev].bind(dev)
      if self.last_ji[self.copy_queues[dev]] is not None: self.copy_queues[dev].bind(dev)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    for comp_queue in self.comp_queues.values(): self.signals[comp_queue].value = 0
    for copy_queue in self.copy_queues.values(): self.signals[copy_queue].value = 0
    self.dev_kickoff_signal['CPU'].value = self.kickoff_value

    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      if j in self.ji_args_bufs: self.ji_args_bufs[j][i] = input_rawbuffers[input_idx]._buf.va_addr
      else: self.op_cmd_idx[j][0].update_copy(self.op_cmd_idx[j][1], **{('dest' if i == 0 else 'src'): input_rawbuffers[input_idx]._buf.va_addr})

    # Update var_vals
    for j, i, v in self.updated_vars(var_vals): self.ji_args_vars[j][i] = v

    # Update launch dims
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      queue, cmd_ptr = self.op_cmd_idx[j]
      queue.update_exec(cmd_ptr, global_dims, local_dims)

    for dev in self.devices:
      comp_queue, copy_queue, need_sig_upd = self.comp_queues[dev], self.copy_queues[dev], dev.timeline_signal != self.last_timeline[dev][0]
      comp_queue.update_wait(1, dev.timeline_signal if need_sig_upd else None, dev.timeline_value - 1) \
                .update_wait(2, value=self.kickoff_value).update_signal(3, value=self.kickoff_value) \
                .update_signal(len(comp_queue)-1, dev.timeline_signal if need_sig_upd else None, dev.timeline_value).submit(dev)

      if self.last_ji[copy_queue] is not None:
        for cmd_idx in self.kickoff_wait_cmds[copy_queue]: copy_queue.update_wait(cmd_idx, value=self.kickoff_value)
        copy_queue.submit(dev)

      self.last_timeline[dev] = (dev.timeline_signal, dev.timeline_value)
      dev.timeline_value += 1

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def access_resources(self, queue, read, write, new_val):
    deps = self._access_resources(read, write, (queue, new_val))

    sync_signals = []
    for dep_queue,_ in deps: self.save_devs[queue].update(self.save_devs[dep_queue])
    for buf in read+write:
      if buf.device not in self.save_devs[queue]:
        self.save_devs[queue].add(buf.device)
        sync_signals += [(self.dev_kickoff_signal[Device[buf.device].dname], self.kickoff_value)]

    return [(self.signals[k], max(v for x, v in deps if id(x) == idk)) for idk, k in {id(x[0]): x[0] for x in deps}.items()] + sync_signals

  def collect_timestamps(self):
    timestamps = [s.timestamp for s in self.prof_signals]
    for _,_,_,((st,_),(en,_),dev,desc,is_cp) in self.signal_sched.values(): #type: ignore
      dev.raw_prof_records += [(timestamps[st], timestamps[en], desc, is_cp)]

  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferOptions(cpu_access=True))
