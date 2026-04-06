import collections, time
from typing import Any, cast
from tinygrad.helpers import round_up, PROFILE, ALL2ALL, merge_dicts, getenv, suppress_finalizing, TracingKey, unwrap
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQSignal, HCQBuffer, HWQueue, HCQArgsState, BumpAllocator, MMIOInterface
from tinygrad.device import Buffer, BufferSpec, Compiled, Device, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, Variable
from tinygrad.engine.realize import BufferXfer, CompiledRunner, BufferCopy
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner
from tinygrad.runtime.ops_rdma import RDMACopyQueue

class HCQGraph(MultiGraphRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.devices = list(set(cast(HCQCompiled, d) for ji in self.jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs]))

    # CPU Device is always last
    self.devices = sorted(self.devices, key=lambda x: 1 if x._is_cpu() else 0)

    # Replace input buffers with variables.
    self.hcq_bufs = [[cast(Buffer, x)._buf for x in ji.bufs] for ji in self.jit_cache]
    self.input_replace_to_var: dict[tuple[int, int], Variable] = {}

    for (j,i), input_idx in self.input_replace.items():
      x = self.input_replace_to_var.setdefault((j,i), UOp.variable(f"input_{input_idx}", 0, 0xffffffffffffffff, dtype=dtypes.uint64))
      self.hcq_bufs[j][i] = HCQBuffer(x, self.hcq_bufs[j][i].size) # Create fake buffer with variable

    # Allocate kernel args.
    kernargs_size: dict[Compiled, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.dev] += round_up(ji.prg._prg.kernargs_alloc_size, 16)
    self.kernargs_bufs: dict[Compiled, HCQBuffer] = {d:d.allocator._alloc(max(sz, 1), BufferSpec(cpu_access=True)) for d,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_args: dict[int, HCQArgsState] = {}

    kargs_alloc: dict[Compiled, BumpAllocator] = {dev:BumpAllocator(buf.size) for dev,buf in self.kernargs_bufs.items()}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue

      argsbuf = self.kernargs_bufs[ji.prg.dev].offset(kargs_alloc[ji.prg.dev].alloc(ji.prg._prg.kernargs_alloc_size, 16))
      self.ji_args[j] = ji.prg._prg.fill_kernargs(self.hcq_bufs[j], ji.prg.p.vars, argsbuf)

    # Schedule Dependencies.
    # There are two types of queues on each device: copy and compute. Both must synchronize with all external operations before launching any
    # graph-related tasks. This synchronization uses a global timeline signal per device. Within the graph, the compute queue coordinates with
    # global operations and sets a kickoff signal. Any queue accessing a buffer from another device waits for this signal from the device’s
    # compute queue to ensure exclusive access. The compute queue signals the completion of the graph, synchronizing with the device's copy queue.
    self.ji_schedule: dict[int, tuple[HCQCompiled, HWQueue, list, list, HCQSignal, int|None]] = {}

    self.comp_queues: dict[HCQCompiled, HWQueue] = {dev: unwrap(dev.hw_compute_queue_t)() for dev in self.devices}
    self.copy_queues: dict[tuple[HCQCompiled, int], HWQueue] = {} # lazy allocation, keyed by (device, queue_idx)
    self.rdma_queues: dict[tuple[HCQCompiled, HCQCompiled], RDMACopyQueue] = {} # lazy allocation, keyed by device pair
    self.num_copy_queues: int = getenv("HCQ_NUM_SDMA", min(len(self.devices), 8) if ALL2ALL >= 1 else 1)
    self.num_rdma_ops: dict[tuple[HCQCompiled, HCQCompiled], int] = collections.defaultdict(int)

    self.rdma_vars: dict[tuple[HCQCompiled, HCQCompiled], tuple[Variable, Any]] = {} # value is variable and src_qp
    self.rdma_deps: dict[int, tuple[HWQueue, list[tuple[HCQSignal, int]], HCQSignal, int]] = {}

    # Per-peer-group representative device for signal allocation. Prefer non-CPU devices, fall back to devices[0].
    self.pg_dev: dict[Any, HCQCompiled] = {dev.peer_group: self.devices[0] for dev in self.devices if dev._is_cpu()} | \
                                           {dev.peer_group: dev for dev in self.devices if not dev._is_cpu()}

    self.kick_signals: dict[Any, HCQSignal] = {pg: pg_dev.new_signal(value=0) for pg, pg_dev in self.pg_dev.items()}
    self.signals: dict[Any, HCQSignal] = {**{dev: dev.new_signal(value=0) for dev in self.devices if not dev._is_cpu()},
      **{dev: self.pg_dev[dev.peer_group].new_signal(value=0) for dev in self.devices if dev._is_cpu()}}
    self.kickoff_value: int = 0
    self.kickoff_var = UOp.variable("kickoff_var", 0, 0xffffffff, dtype=dtypes.uint32)

    # When profiling allocate 2 signals for each jit item to measure speed. The jth jit item have signals at 2*j and 2*j+1.
    # TODO: This logic might allocate a few extra signals...
    self.prof_signals: list[HCQSignal] = []
    self.prof_graph_deps: list[list[int]] = []
    self.prof_graph_entries: list[ProfileGraphEntry] = []

    self.last_j: dict[HWQueue, int|None] = collections.defaultdict(lambda: None)
    self.queue_access: dict[HWQueue, dict[HWQueue, int|None]] = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    self.dev_access: dict[HWQueue, set[HCQCompiled]] = collections.defaultdict(set)

    for dev, queue in self.comp_queues.items(): self.dev_access[queue].add(dev)

    self.input_replace_map: dict[HCQCompiled, set[int]] = collections.defaultdict(set)
    self.device_vars: dict[HCQCompiled, dict[str, int]] = {}

    for j,ji in enumerate(self.jit_cache):
      ji_devs = [cast(HCQCompiled, Device[cast(Buffer, b).device]) for b in ji.bufs] if isinstance(ji.prg, BufferXfer) else []
      is_rdma = len(ji_devs) > 0 and not any(d._is_cpu() for d in ji_devs) and len(set(d.peer_group for d in ji_devs)) > 1

      if is_exec_prg:=isinstance(ji.prg, CompiledRunner): enqueue_dev: HCQCompiled = ji.prg.dev
      else:
        # For copy ops prioritize enqeueuing on the src device, so reverse the buffers.
        for b in cast(list[Buffer], ji.bufs[::-1]):
          if (enqueue_dev:=cast(HCQCompiled, Device[b.device])).hw_copy_queue_t is not None: break

      # set any fixedvars on the device
      self.device_vars[enqueue_dev] = merge_dicts([self.device_vars.get(enqueue_dev, {}), ji.fixedvars])
      if is_exec_prg: self.device_vars[enqueue_dev] = merge_dicts([self.device_vars[enqueue_dev], cast(CompiledRunner, ji.prg).p.runtimevars])

      if is_exec_prg:
        enqueue_queue = self.comp_queues[enqueue_dev]
      elif is_rdma:
        enqueue_queue = self.comp_queues[enqueue_dev]
        rdma_key = (cast(HCQCompiled, Device[cast(Buffer, ji.bufs[0]).device]), cast(HCQCompiled, Device[cast(Buffer, ji.bufs[1]).device]))
        self.rdma_queues.setdefault(rdma_key, RDMACopyQueue(enqueue_dev.rdma_dev()))
      else:
        assert (enqueue_dev.hw_copy_queue_t is not None), "device must implement a copy queue"
        queue_idx = self.devices.index(cast(HCQCompiled, Device[cast(Buffer, ji.bufs[0]).device])) % self.num_copy_queues
        enqueue_queue = self.copy_queues.setdefault((enqueue_dev, queue_idx),
          enqueue_dev.hw_copy_queue_t(queue_idx=queue_idx).wait(self.kick_signals[enqueue_dev.peer_group], self.kickoff_var))

      out_signal = self.signals.setdefault(enqueue_queue, self.pg_dev[enqueue_dev.peer_group].new_signal(value=0))

      # Get dependencies based on input and output buffers.
      if is_rdma:
        sync_signals, opt_deps, rdeps = self._resolve_deps(ji.bufs[1:], [], enqueue_queue, enqueue_dev, out_signal, j,
                                                           is_copy=isinstance(ji.prg, BufferXfer), is_rdma=is_rdma)
        peer_queue = self.comp_queues[peer_dev:=cast(HCQCompiled, Device[cast(Buffer, ji.bufs[0]).device])]
        peer_out_signal = self.signals.setdefault(peer_queue, self.pg_dev[peer_dev.peer_group].new_signal(value=0))
        peer_sync_signals, peer_opt_deps, peer_rdeps = self._resolve_deps(ji.bufs[:1], [0], peer_queue, peer_dev, peer_out_signal, j,
                                                                          is_copy=isinstance(ji.prg, BufferXfer), is_rdma=is_rdma)
        self.rdma_deps[j] = (peer_queue, peer_sync_signals + peer_opt_deps, peer_out_signal, j + 1)
        self.last_j[peer_queue] = j
      else:
        sync_signals, opt_deps, rdeps = self._resolve_deps(ji.bufs, cast(CompiledRunner, ji.prg).p.outs if is_exec_prg else [0], enqueue_queue,
          enqueue_dev, out_signal, j, is_copy=isinstance(ji.prg, BufferXfer), is_rdma=is_rdma)

      self.ji_schedule[j] = (enqueue_dev, enqueue_queue, sync_signals, opt_deps[::-1], out_signal, None if is_exec_prg else (j + 1))

      # Collect profile information if profiling is enabled.
      if PROFILE:
        # When execution are chained, we can reuse the end timestamp from the previous command as the start timestamp for the current command.
        sig_st = prev_ji * 2 + 1 if len(opt_deps) == 0 and (prev_ji:=self.last_j[enqueue_queue]) is not None else j * 2

        # Description based on the command.
        prof_ji_desc = ji.prg._prg.name if is_exec_prg else TracingKey(f"{ji.bufs[1].device} -> {ji.bufs[0].device}", ret=ji.bufs[0].nbytes) # type: ignore

        prof_name = f"{enqueue_dev.device}:SDMA:{queue_idx}" if not is_exec_prg else enqueue_dev.device
        self.prof_graph_entries.append(ProfileGraphEntry(prof_name, prof_ji_desc, sig_st, j * 2 + 1))
        self.prof_graph_deps.append([d - 1 for _, d in rdeps])

      self.last_j[enqueue_queue] = j

    # Check which signals are used in the profile graph.
    self.prof_signal_is_used = [any(ent.st_id == j or ent.en_id == j for ent in self.prof_graph_entries) for j in range(len(self.jit_cache) * 2)]

    # Build hardware queues.
    self.copy_to_devs: dict[HCQCompiled, set[HCQCompiled]] = {dev: set() for dev in self.devices}

    # Create variable timeline signals for each device.
    timeline_sigaddrs = {dev: UOp.variable(f"timeline_sig_{self.dev_name(dev)}", 0, 0xffffffffffffffff, dtype=dtypes.uint64) for dev in self.devices}
    self.virt_timeline_vals = {dev: UOp.variable(f"timeline_var_{self.dev_name(dev)}", 0, 0xffffffff, dtype=dtypes.uint32) for dev in self.devices}
    self.virt_timeline_signals = {dev: unwrap(dev.signal_t)(HCQBuffer(timeline_sigaddrs[dev], 16),owner=dev,is_timeline=True) for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev]) \
                           .wait(self.kick_signals[dev.peer_group], self.kickoff_var).signal(self.signals[dev], self.kickoff_var)

    for j,ji in enumerate(self.jit_cache):
      enqueue_dev, enqueue_queue, sync_signals, deps, signal, signal_val = self.ji_schedule[j]

      # Lazy allocate signals
      if PROFILE: self.prof_signals += [enqueue_dev.new_signal(value=0) for _ in range(2)]

      for sig, val in sync_signals + deps: enqueue_queue.wait(sig, val)

      # Encode waits and start profile timestamp (if needed).
      if PROFILE and self.prof_signal_is_used[j * 2]: enqueue_queue.timestamp(self.prof_signals[j * 2])

      # Encode main commands based on ji type.
      if isinstance(ji.prg, CompiledRunner):
        enqueue_queue.exec(ji.prg._prg, self.ji_args[j], tuple(ji.prg.p.global_size or (1,1,1)), tuple(ji.prg.p.local_size or (1,1,1)))
      elif isinstance(ji.prg, BufferXfer) and len(set(cast(HCQCompiled, Device[cast(Buffer, b).device]).peer_group for b in ji.bufs)) > 1:
        dest_queue, dest_deps, dest_out_signal, dest_out_val = self.rdma_deps[j]
        for sig, val in dest_deps: dest_queue.wait(sig, val)

        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        dest_dev, src_dev = cast(HCQCompiled, Device[dest.device]), cast(HCQCompiled, Device[src.device])
        dest_rdma, src_rdma = dest_dev.rdma_dev(), src_dev.rdma_dev()

        # get qp info
        src_qp, dest_qp, src_cq_buf, dest_cq_buf = src_rdma.iface.connect(dest_rdma)

        # use var for head
        head_var = self.rdma_vars.setdefault((dest_rdma, src_rdma), (UOp.variable(f"rdma_var_{j}", 0, 0xffffffff, dtype=dtypes.uint32), src_qp))[0]
        next_head = self.num_rdma_ops[(dest_rdma, src_rdma)]

        rdma_queue = self.rdma_queues[(dest_dev, src_dev)]
        rdma_queue.copy(self.hcq_bufs[j][0], self.hcq_bufs[j][1], dest.nbytes) \
                  .encode_ring(enqueue_queue, src_dev, src_rdma.iface, src_qp, src_cq_buf, head_var + next_head, ring_uar=True) \
                  .encode_ring(self.comp_queues[dest_dev], dest_dev, dest_rdma.iface, dest_qp, dest_cq_buf, head_var + next_head)

        dest_queue.signal(dest_out_signal, dest_out_val)
        self.num_rdma_ops[(dest_rdma, src_rdma)] += 1
      elif isinstance(ji.prg, (BufferXfer, BufferCopy)):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        for bufid, src in enumerate(cast(list[Buffer], ji.bufs)):
          if (inprep_idx:=self.input_replace.get((j, bufid))) is not None: self.input_replace_map[enqueue_dev].add(inprep_idx)
          else: cast(HCQAllocator, enqueue_dev.allocator).map(self.hcq_bufs[j][bufid])
        enqueue_queue.copy(self.hcq_bufs[j][0], self.hcq_bufs[j][1], dest.nbytes)
        self.copy_to_devs[cast(HCQCompiled, Device[dest.device])].add(cast(HCQCompiled, Device[src.device]))

      # Encode finish profile timestamp (if needed).
      if PROFILE and self.prof_signal_is_used[j * 2 + 1]: enqueue_queue.timestamp(self.prof_signals[j * 2 + 1])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        for copy_q in self._dev_copy_queues(dep_dev):
          if copy_q in self.signals: self.comp_queues[dev].wait(self.signals[copy_q], cast(int, self.last_j[copy_q]) + 1)

      self.comp_queues[dev].signal(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev] + 1).bind(dev)
      for copy_q in self._dev_copy_queues(dev): copy_q.bind(dev)

    self.last_timeline: dict[HCQCompiled, tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}
    self.queue_signals_to_reset = [self.signals[q] for q in list(self.comp_queues.values()) + list(self.copy_queues.values()) if q in self.signals]

  def _resolve_deps(self, bufs, outs, enqueue_queue, enqueue_dev, out_signal, j, is_copy, is_rdma):
    rdeps = self._access_resources(bufs, outs, (enqueue_queue, j + 1)) #type:ignore

    # Update dependencies to include previous kernel in queue. This is required for timeline signals.
    opt_deps, deps = [], rdeps + ([(enqueue_queue, prev_ji + 1)] if (prev_ji:=self.last_j[enqueue_queue]) is not None else [])

    # Optimize dependencies by removing redundant ones. Remove waiting for the value of the queue which is known to be already
    # synced with the current queue.
    for dep_queue, dep_val in sorted(deps, key=lambda x: x[1], reverse=True):
      if (qa:=self.queue_access[enqueue_queue][dep_queue]) is None or qa < dep_val:
        opt_deps.append((self.signals[dep_queue], dep_val))
        self.queue_access[enqueue_queue][dep_queue] = dep_val
        self.dev_access[enqueue_queue].update(self.dev_access[dep_queue])

    # Ensure device is ready for use in current context: the graph has initialized the device and it's safe to operate on it within this graph.
    # Only sync with same-peer-group devices; cross-peer-group sync is handled by RDMA.
    sync_signals = [(self.signals[d], self.kickoff_var) for b in bufs
      if (d:=cast(HCQCompiled, Device[cast(Buffer, b).device])) not in self.dev_access[enqueue_queue]
      and (d.peer_group == enqueue_dev.peer_group or not is_rdma)]
    self.dev_access[enqueue_queue].update(cast(HCQCompiled, Device[cast(Buffer, b).device]) for b in bufs)

    # Remove self-dependency for compute and copy queues.
    # For compute, in case of NV, optimize when only 1 same-queue dependency exists, since NV chains 2+ executions in this case,
    # eliminating dependency need. For RDMA, keep self-dependency to flush cache.
    dname = enqueue_dev.device.split(":", 1)[0]
    can_opt = dname in {"AMD", "QCOM"} or (dname == "NV" and len(sync_signals) == 0 and len(opt_deps) == 1 and id(opt_deps[0][0]) == id(out_signal))
    if (can_opt or is_copy) and not is_rdma: opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]

    # Enable necessary signals in the schedule by setting the signal value.
    for sig, val in opt_deps: self.ji_schedule[val - 1] = self.ji_schedule[val - 1][:5] + (val,)

    return sync_signals, opt_deps, rdeps

  def _dev_copy_queues(self, dev): return [q for (d, _), q in self.copy_queues.items() if d == dev]

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    # Map input buffers
    for dev in self.devices:
      for idx_to_map in self.input_replace_map[dev]: cast(HCQAllocator, dev.allocator).map(input_buffers[idx_to_map]._buf)

    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    hcq_var_vals = {self.kickoff_var.expr: self.kickoff_value, **var_vals,
                    **{var.expr: dev.timeline_value - 1 for dev, var in self.virt_timeline_vals.items()},
                    **{sig.base_buf.va_addr.expr: dev.timeline_signal.base_buf.va_addr for dev, sig in self.virt_timeline_signals.items()}}

    # Update buffers
    for (j,i),input_idx in self.input_replace.items():
      hcq_var_vals[self.input_replace_to_var[(j,i)].expr] = input_buffers[input_idx]._buf.va_addr

    for (var, qp) in self.rdma_vars.values(): hcq_var_vals[var.expr] = qp.head
    for q in self.rdma_queues.values(): q.submit(q.dev, hcq_var_vals)

    for dev in self.devices:
      self.comp_queues[dev].submit(dev, hcq_var_vals_local:=hcq_var_vals|self.device_vars.get(dev, {}))
      for copy_queue in self._dev_copy_queues(dev): copy_queue.submit(dev, hcq_var_vals_local)
      self.last_timeline[dev] = (dev.timeline_signal, dev.next_timeline())

    # Launch graph
    for sig in self.queue_signals_to_reset: sig.value = 0
    for sig in self.kick_signals.values(): sig.value = self.kickoff_value

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def collect_timestamps(self):
    # NOTE: Append to any device is fine...
    self.devices[0].profile_events += [ProfileGraphEvent(self.prof_graph_entries, self.prof_graph_deps, [s.timestamp for s in self.prof_signals])]

  def dev_name(self, dev) -> str: return dev.device.replace(":", "_")

  @suppress_finalizing
  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value >= 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferSpec(cpu_access=True))

  @staticmethod
  def supports_exec_item(batch_devs:list[Compiled], new_call:UOp) -> bool:
    # Check if all devices are HCQ
    all_devs = cast(list[HCQCompiled], GraphRunner._all_devs(batch_devs, new_call))
    if not all(issubclass(type(d), HCQCompiled) for d in all_devs): return False

    # If all of devices are mapped into CPU address space, can use CPU inside the peer group.
    cpu_support = all(type(d.timeline_signal.base_buf.view) is MMIOInterface for d in all_devs)

    # Check if all devices are within the same peer group. Allow cross-peer-group if all peer groups have RDMA devices.
    if len(set(d.peer_group for d in all_devs if not (cpu_support and d._is_cpu()))) > 1:
      try: [d.rdma_dev() for d in all_devs if not d._is_cpu()]
      except RuntimeError: return False

    if new_call.src[0].op is Ops.COPY:
      # MOCKGPU is not supported, since it can't execute commands in parallel
      is_xfer = len(set(type(d) for d in all_devs)) == 1 and hasattr(alc:=all_devs[0].allocator, '_transfer') and alc.supports_transfer
      return is_xfer or (all_devs[0].hw_copy_queue_t is not None and not getenv("MOCKGPU"))
    return new_call.src[0].op in (Ops.SINK, Ops.PROGRAM)
