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
    self.copy_queues: Dict[Compiled, HWCopyQueue] = {} # lazy allocation
    self.ji_schedule: Dict[int, Tuple[HCQCompiled, HWCommandQueue, List, List, HCQSignal, Optional[int]]] = {}
    self.signals: Dict[Any, HCQSignal] = {**{dev.dname: dev.signal_t(value=0) for dev in self.devices}, **{"CPU": self.devices[0].signal_t(value=0)}}
    self.kickoff_value = 0

    dev_access: Dict[HWCommandQueue, List[HCQCompiled]] = collections.defaultdict(set)
    queue_access: Dict[HWCommandQueue, Dict[HWCommandQueue, Optional[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: None))

    for j,ji in enumerate(self.jit_cache):
      enqueue_dev = ji.prg.device if isinstance(ji.prg, CompiledRunner) else Device[ji.bufs[1].device] #type:ignore
      enqueue_queue = self.copy_queues.setdefault(enqueue_dev, enqueue_dev.hw_copy_queue_t()) if isinstance(ji.prg, BufferXfer) else self.comp_queues[enqueue_dev]
      out_signal = self.signals.setdefault(enqueue_queue, enqueue_dev.signal_t(value=0))

      deps = self._access_resources(ji.bufs[(wb:=1 if isinstance(ji.prg, BufferXfer) else ji.prg.p.outcount):], ji.bufs[:wb], (enqueue_queue, j + 1))

      # Update signal on compute kernel to depend on the previous kernel.
      opt_deps, deps2 = [], deps + ([(enqueue_queue, prev_ji + 1)] if (prev_ji:=queue_access[enqueue_queue][enqueue_queue]) is not None else [])

      # Optimize dependencies removing all dependencies which are already know to be correct.
      for dep_queue, dep_val in sorted(deps2, key=lambda x: x[1], reverse=True):
        if (qa:=queue_access[enqueue_queue][dep_queue]) is None or qa < dep_val:
          opt_deps.append((self.signals[dep_queue], dep_val))
          queue_access[enqueue_queue][dep_queue] = dep_val

      # Need to be sure that device is ready to be used in this context, so issue a device kickoff sync.
      for dep_queue, _ in opt_deps: dev_access[enqueue_queue].update(dev_access[dep_queue])
      sync_signals = [(self.signals[buf.device], self.kickoff_value) for buf in ji.bufs if buf.device not in dev_access[enqueue_queue]]
      dev_access[enqueue_queue].update(buf.device for buf in ji.bufs)

      if isinstance(ji.prg, CompiledRunner):
        # Remove self-dependency for AMD or NV with only 1 same-queue dep, since NV chains 2+ execs in this case, eliminating dep need.
        if (dname:=enqueue_dev.dname.split(":", 1)[0]) == "AMD" or (dname == "NV" and len(opt_deps + sync_signals) == 1 and id(opt_deps[0][0]) == id(out_signal)):
          opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]
      elif isinstance(ji.prg, BufferXfer): opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]

      # Go through all dependencies and, if we need the signal from that ji, enable it by setting the signal value in the signal schedule.
      for sig, val in opt_deps: self.ji_schedule[val - 1] = self.ji_schedule[val - 1][:5] + (val,)
      self.ji_schedule[j] = (enqueue_dev, enqueue_queue, sync_signals, opt_deps, out_signal, None if isinstance(ji.prg, CompiledRunner) else (j + 1))

      # print(sync_signals, opt_deps)

      queue_access[enqueue_queue][enqueue_queue] = j
      # assert queue_access[enqueue_queue][enqueue_queue] == j + 1, f"{queue_access[enqueue_queue][enqueue_queue]} != {j + 1}"
      
      # Collect profile info as well
      if PROFILE:
        pass

    # Build hardware queues.
    self.op_cmd_idx: Dict[int, Tuple[Any, int]] = {}
    self.copy_to_devs: Dict[Compiled, Set[Compiled]] = {dev: set() for dev in self.devices}
    self.kickoff_wait_cmds: Dict[HWCommandQueue, List] = {q: list() for q in list(self.comp_queues.values()) + list(self.copy_queues.values())}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(dev.timeline_signal, dev.timeline_value - 1) \
                           .wait(self.signals['CPU'], self.kickoff_value).signal(self.signals[dev.dname], self.kickoff_value)

    for j,ji in enumerate(self.jit_cache):
      enqueue_dev, enqueue_queue, sync_signals, deps, signal, signal_val = self.ji_schedule[j]

      for i in range(len(sync_signals)): self.kickoff_wait_cmds[enqueue_queue].append(len(enqueue_queue) + i)
      for sig, val in sync_signals + deps: enqueue_queue.wait(sig, val)

      # Encode waits and start profile timestamp (if needed).
      # if prof_info and prof_info[0][1]: enqueue_queue.timestamp(self.prof_signals[prof_info[0][0]])

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
      # if prof_info and prof_info[1][1]: enqueue_queue.timestamp(self.prof_signals[prof_info[1][0]])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        if dep_dev in self.copy_queues:
          self.comp_queues[dev].wait(self.signals[(copy_q:=self.copy_queues[dep_dev])], queue_access[copy_q][copy_q] + 1)

      # for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
      #   if (last_j:=self.last_ji[self.copy_queues[dep_dev]]) is None: continue
      #   self.comp_queues[dev].wait(self.signals[self.copy_queues[dep_dev]], self.signal_sched[last_j][2])

      self.comp_queues[dev].signal(dev.timeline_signal, dev.timeline_value).bind(dev)
      if dev in self.copy_queues: copy_q.bind(dev)

    self.last_timeline: Dict[HCQCompiled, Tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    for comp_queue in self.comp_queues.values(): self.signals[comp_queue].value = 0
    for copy_queue in self.copy_queues.values(): self.signals[copy_queue].value = 0
    self.signals['CPU'].value = self.kickoff_value

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
      comp_queue, copy_queue, need_sig_upd = self.comp_queues[dev], self.copy_queues.get(dev, None), dev.timeline_signal != self.last_timeline[dev][0]
      comp_queue.update_wait(1, dev.timeline_signal if need_sig_upd else None, dev.timeline_value - 1) \
                .update_wait(2, value=self.kickoff_value).update_signal(3, value=self.kickoff_value) \
                .update_signal(len(comp_queue)-1, dev.timeline_signal if need_sig_upd else None, dev.timeline_value).submit(dev)

      if copy_queue is not None:
        for cmd_idx in self.kickoff_wait_cmds[copy_queue]: copy_queue.update_wait(cmd_idx, value=self.kickoff_value)
        copy_queue.submit(dev)

      self.last_timeline[dev] = (dev.timeline_signal, dev.timeline_value)
      dev.timeline_value += 1

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def collect_timestamps(self):
    timestamps = [s.timestamp for s in self.prof_signals]

    for _,_,_,((st,_),(en,_),dev,desc,is_cp) in self.signal_sched.values(): # type: ignore
      dev.raw_prof_records += [(timestamps[st], timestamps[en], desc, is_cp)]

    for ((a_st,_), (a_en,_), a_dev, _, a_is_cp), ((b_st,_), (b_en,_), b_dev, _, b_is_cp) in self.prof_deps:
      b_dev.dep_prof_records += [(timestamps[a_st], timestamps[a_en], a_dev, a_is_cp, timestamps[b_st], timestamps[b_en], b_dev, b_is_cp)]

  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value >= 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferOptions(cpu_access=True))
