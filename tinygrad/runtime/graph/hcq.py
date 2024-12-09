import collections, time
from typing import List, Any, Dict, cast, Optional, Tuple, Set
from tinygrad.helpers import round_up, PROFILE, memsize_to_str
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQSignal, HCQBuffer, HWQueue, HCQArgsState, BumpAllocator
from tinygrad.device import Buffer, BufferSpec, Compiled, Device
from tinygrad import Variable, dtypes
from tinygrad.ops import Variable as VariableT
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[VariableT, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.devices = list(set(cast(HCQCompiled, d) for ji in jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs]))

    # Replace input buffers with variables.
    self.hcq_bufs = [[cast(Buffer, x)._buf for x in ji.bufs] for ji in jit_cache]
    self.input_replace_to_var: Dict[Tuple[int, int], VariableT] = {}

    for (j,i), input_idx in self.input_replace.items():
      x = self.input_replace_to_var.setdefault((j,i), Variable(f"input_{input_idx}", 0, 0xffffffffffffffff, dtype=dtypes.uint64))
      self.hcq_bufs[j][i] = HCQBuffer(x, self.hcq_bufs[j][i].size, texture_info=self.hcq_bufs[j][i].texture_info) # Create fake buffer with variable

    # Allocate kernel args.
    kernargs_size: Dict[Compiled, int] = collections.defaultdict(int)
    for ji in jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.dev] += round_up(ji.prg._prg.kernargs_alloc_size, 16)
    self.kernargs_bufs: Dict[Compiled, HCQBuffer] = {dev:dev.allocator._alloc(sz, BufferSpec(cpu_access=True)) for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_args: Dict[int, HCQArgsState] = {}

    kargs_alloc: Dict[Compiled, BumpAllocator] = {dev:BumpAllocator(buf.size, start=cast(int, buf.va_addr)) for dev,buf in self.kernargs_bufs.items()}
    for j,ji in enumerate(jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue

      self.ji_args[j] = ji.prg._prg.fill_kernargs(self.hcq_bufs[j], ji.prg.p.vars, kargs_alloc[ji.prg.dev].alloc(ji.prg._prg.kernargs_alloc_size, 16))

    # Schedule Dependencies.
    # There are two types of queues on each device: copy and compute. Both must synchronize with all external operations before launching any
    # graph-related tasks. This synchronization uses a global timeline signal per device. Within the graph, the compute queue coordinates with
    # global operations and sets a kickoff signal. Any queue accessing a buffer from another device waits for this signal from the device’s
    # compute queue to ensure exclusive access. The compute queue signals the completion of the graph, synchronizing with the device's copy queue.
    self.ji_schedule: Dict[int, Tuple[HCQCompiled, HWQueue, List, List, HCQSignal, Optional[int]]] = {}

    self.comp_queues: Dict[HCQCompiled, HWQueue] = {dev: dev.hw_compute_queue_t() for dev in self.devices}
    self.copy_queues: Dict[HCQCompiled, HWQueue] = {} # lazy allocation

    self.signals: Dict[Any, HCQSignal] = {**{dev: dev.signal_t(value=0) for dev in self.devices}, **{"CPU": self.devices[0].signal_t(value=0)}}
    self.kickoff_value: int = 0
    self.kickoff_var = Variable("kickoff_var", 0, 0xffffffff, dtype=dtypes.uint32)

    self.prof_signals: List[HCQSignal] = [self.devices[0].signal_t() for i in range(len(jit_cache) * 2)] if PROFILE else []
    self.prof_records: List[Tuple[Tuple[int, bool], Tuple[int, bool], HCQCompiled, str, bool, List[int], Optional[Dict]]] = []

    last_j: Dict[HWQueue, Optional[int]] = collections.defaultdict(lambda: None)
    queue_access: Dict[HWQueue, Dict[HWQueue, Optional[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    dev_access: Dict[HWQueue, Set[HCQCompiled]] = collections.defaultdict(set)

    for dev, queue in self.comp_queues.items(): dev_access[queue].add(dev)

    for j,ji in enumerate(jit_cache):
      enqueue_dev: HCQCompiled = ji.prg.dev if (is_exec_prg:=isinstance(ji.prg, CompiledRunner)) else Device[ji.bufs[1].device] #type:ignore

      if is_exec_prg:
        enqueue_queue = self.comp_queues[enqueue_dev]
      else:
        assert (enqueue_dev.hw_copy_queue_t is not None), "device must implement a copy queue"
        enqueue_queue = self.copy_queues.setdefault(enqueue_dev, enqueue_dev.hw_copy_queue_t())

      out_signal = self.signals.setdefault(enqueue_queue, enqueue_dev.signal_t(value=0))

      # Get dependencies based on input and output buffers.
      rdeps = self._access_resources(ji.bufs, ji.prg.p.outs if is_exec_prg else [0], (enqueue_queue, j + 1)) #type:ignore

      # Update dependencies to include previous kernel in queue. This is required for timeline signals.
      opt_deps, deps = [], rdeps + ([(enqueue_queue, prev_ji + 1)] if (prev_ji:=last_j[enqueue_queue]) is not None else [])

      # Optimize dependencies by removing redundant ones. Remove waiting for the value of the queue which is known to be already
      # synced with the current queue.
      for dep_queue, dep_val in sorted(deps, key=lambda x: x[1], reverse=True):
        if (qa:=queue_access[enqueue_queue][dep_queue]) is None or qa < dep_val:
          opt_deps.append((self.signals[dep_queue], dep_val))
          queue_access[enqueue_queue][dep_queue] = dep_val

      # Ensure device is ready for use in current context: the graph has initialized the device and it's safe to operate on it within this graph.
      for dep_queue, _ in opt_deps: dev_access[enqueue_queue].update(dev_access[dep_queue])
      sync_signals = [(self.signals[d], self.kickoff_var) for b in ji.bufs if (d:=Device[cast(Buffer, b).device]) not in dev_access[enqueue_queue]]
      dev_access[enqueue_queue].update(cast(HCQCompiled, Device[cast(Buffer, b).device]) for b in ji.bufs)

      # Remove self-dependency for compute and copy queues.
      # For compute, in case of NV, optimize when only 1 same-queue dependency exists, since NV chains 2+ executions in this case,
      # eliminating dependency need.
      dname = enqueue_dev.device.split(":", 1)[0]
      can_opt = dname in {"AMD", "QCOM"} or (dname == "NV" and len(sync_signals) == 0 and len(opt_deps) == 1 and id(opt_deps[0][0]) == id(out_signal))
      if can_opt or isinstance(ji.prg, BufferXfer): opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]

      # Enable necessary signals in the schedule by setting the signal value.
      for sig, val in opt_deps: self.ji_schedule[val - 1] = self.ji_schedule[val - 1][:5] + (val,)
      self.ji_schedule[j] = (enqueue_dev, enqueue_queue, sync_signals, opt_deps[::-1], out_signal, None if is_exec_prg else (j + 1))

      # Collect profile information if profiling is enabled.
      if PROFILE:
        prof_ji_desc = ji.prg._prg.name if is_exec_prg else f"{ji.bufs[1].device} -> {ji.bufs[0].device}" # type: ignore

        sig_st, sig_en = (j * 2, True), (j * 2 + 1, True)
        if len(opt_deps) == 0 and (prev_ji:=last_j[enqueue_queue]) is not None: sig_st = (prev_ji * 2 + 1, False)

        if is_exec_prg: prof_args = None
        else: prof_args = {"Size": memsize_to_str(ji.bufs[0].nbytes), "GB/S": lambda dur, b=ji.bufs[0].nbytes: f"{b/1e3/dur:.2f}"} # type: ignore

        self.prof_records.append((sig_st, sig_en, enqueue_dev, prof_ji_desc, not is_exec_prg, [d - 1 for _, d in rdeps], prof_args))

      last_j[enqueue_queue] = j

    # Build hardware queues.
    self.copy_to_devs: Dict[HCQCompiled, Set[HCQCompiled]] = {dev: set() for dev in self.devices}

    # Create variable timeline signals for each device.
    timeline_sigaddrs = {dev: Variable(f"timeline_sig_{dev.device_id}", 0, 0xffffffffffffffff, dtype=dtypes.uint64) for dev in self.devices}
    self.virt_timeline_vals = {dev: Variable(f"timeline_var_{dev.device_id}", 0, 0xffffffff, dtype=dtypes.uint32) for dev in self.devices}
    self.virt_timeline_signals = {dev: dev.signal_t(base_addr=timeline_sigaddrs[dev], timeline_for_device=dev) for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev]) \
                           .wait(self.signals['CPU'], self.kickoff_var).signal(self.signals[dev], self.kickoff_var)

    for j,ji in enumerate(jit_cache):
      enqueue_dev, enqueue_queue, sync_signals, deps, signal, signal_val = self.ji_schedule[j]

      for sig, val in sync_signals + deps: enqueue_queue.wait(sig, val)

      # Encode waits and start profile timestamp (if needed).
      if PROFILE and self.prof_records[j][0][1]: enqueue_queue.timestamp(self.prof_signals[self.prof_records[j][0][0]])

      # Encode main commands based on ji type.
      if isinstance(ji.prg, CompiledRunner):
        enqueue_queue.exec(ji.prg._prg, self.ji_args[j], tuple(ji.prg.p.global_size or (1,1,1)), tuple(ji.prg.p.local_size or (1,1,1)))
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        cast(HCQAllocator, Device[src.device].allocator).map(dest._buf)

        enqueue_queue.copy(self.hcq_bufs[j][0].va_addr, self.hcq_bufs[j][1].va_addr, dest.nbytes)
        self.copy_to_devs[cast(HCQCompiled, Device[dest.device])].add(cast(HCQCompiled, Device[src.device]))

      # Encode finish profile timestamp (if needed).
      if PROFILE and self.prof_records[j][1][1]: enqueue_queue.timestamp(self.prof_signals[self.prof_records[j][1][0]])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        if dep_dev in self.copy_queues: self.comp_queues[dev].wait(self.signals[(copy_q:=self.copy_queues[dep_dev])], cast(int, last_j[copy_q]) + 1)

      self.comp_queues[dev].signal(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev] + 1).bind(dev)
      if dev in self.copy_queues: self.copy_queues[dev].bind(dev)

    self.last_timeline: Dict[HCQCompiled, Tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}
    self.queue_signals_to_reset = [self.signals[q] for q in list(self.comp_queues.values()) + list(self.copy_queues.values()) if q in self.signals]

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[VariableT, int], wait=False) -> Optional[float]:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    for sig in self.queue_signals_to_reset: sig.value = 0
    self.signals['CPU'].value = self.kickoff_value

    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    hcq_var_vals = {self.kickoff_var: self.kickoff_value, **var_vals,
                    **{var: dev.timeline_value - 1 for dev, var in self.virt_timeline_vals.items()},
                    **{sig.base_addr: dev.timeline_signal.base_addr for dev, sig in self.virt_timeline_signals.items()}}

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items(): hcq_var_vals[self.input_replace_to_var.get((j,i))] = input_rawbuffers[input_idx]._buf.va_addr

    for dev in self.devices:
      self.comp_queues[dev].submit(dev, hcq_var_vals)
      if (copy_queue:=self.copy_queues.get(dev, None)) is not None: copy_queue.submit(dev, hcq_var_vals)

      self.last_timeline[dev] = (dev.timeline_signal, dev.timeline_value)
      dev.timeline_value += 1

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def collect_timestamps(self):
    timestamps = [s.timestamp for s in self.prof_signals]

    for (st,_), (en,_), dev, desc, is_cp, deps, args in self.prof_records:
      dev.raw_prof_records += [(timestamps[st], timestamps[en], desc, is_cp, args)]

      for x in deps:
        (b_st,_), (b_en,_), b_dev, _, b_is_cp, _, _ = self.prof_records[x]
        dev.dep_prof_records += [(timestamps[b_st], timestamps[b_en], b_dev, b_is_cp, timestamps[st], timestamps[en], dev, is_cp)]

  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value >= 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferSpec(cpu_access=True))
