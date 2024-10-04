from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Any, cast, Protocol, Type, Union
import contextlib, decimal, statistics, random, json, atexit, time, array, ctypes
from tinygrad.helpers import PROFILEPATH, PROFILE, from_mv
from tinygrad.renderer import Renderer
from tinygrad.device import BufferOptions, Allocator, Compiler, Compiled, LRUAllocator

# **************** for HCQ Compatible Devices ****************

def hcq_command(func):
  """
  Decorator for HWCommandQueue commands. Enables command indexing and stores metadata for command updates.

  For example:
    ```python
      @hcq_command
      def command_method(self, ...): ...
    ```
  """
  def __wrapper(self, *args, **kwargs):
    self.cmds_offset.append(len(self.q))
    func(self, *args, **kwargs)
    self.cmds_len.append(len(self.q) - self.cmds_offset[-1])
    self.cmds_meta.append(func.__name__)
    return self
  return __wrapper

class HWCommandQueue:
  """
  A base class for hardware command queues in the HCQ (Hardware Command Queue) API.
  Both compute and copy queues should have the following commands implemented.
  """

  def __init__(self): self.q, self.binded_device, self.cmds_offset, self.cmds_len, self.cmds_meta = [], None, [], [], []
  def __len__(self): return len(self.cmds_offset)
  def _patch(self, cmd_idx, offset, data): self.q[(st:=self.cmds_offset[cmd_idx]+offset):st+len(data)] = array.array('I', data)
  def _cur_cmd_idx(self) -> int:
    """
    Returns the index of the command currently being enqueued.
    Should be called only within functions that enqueue commands and are decorated with `@hcq_command`.
    """
    return len(self) - 1

  @hcq_command
  def signal(self, signal:HCQSignal, value:int):
    """
    Enqueues a signal command which sets the signal to the given value, ensuring all previous operations are completed.

    Args:
      signal: The signal to set
      value: The value to set the signal to
    """
    self._signal(signal, value)
  def _signal(self, signal:HCQSignal, value:int): raise NotImplementedError("backend should overload this function")

  @hcq_command
  def wait(self, signal:HCQSignal, value:int):
    """
    Enqueues a wait command which halts execution until the signal is greater than or equal to a specific value.

    Args:
      signal: The signal to wait on
      value: The value to wait for
    """
    self._wait(signal, value)
  def _wait(self, signal, value): raise NotImplementedError("backend should overload this function")

  @hcq_command
  def timestamp(self, signal:HCQSignal):
    """
    Enqueues a timestamp command which records the current time in a signal after all previously enqueued commands are completed.

    Args:
      signal: The signal to store the timestamp
    """
    self._timestamp(signal)
  def _timestamp(self, signal): raise NotImplementedError("backend should overload this function")

  def update_signal(self, cmd_idx:int, signal:Optional[Any]=None, value:Optional[int]=None):
    """
    Updates a previously queued signal command.

    Args:
      cmd_idx: Index of the signal command to update
      signal: New signal to set (if None, keeps the original)
      value: New value to set (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "signal": raise RuntimeError("called update_signal not on a signal command")
    self._update_signal(cmd_idx, signal, value)
    return self
  def _update_signal(self, cmd_idx:int, signal:Optional[Any], value:Optional[int]): raise NotImplementedError("backend should overload this function")

  def update_wait(self, cmd_idx:int, signal:Optional[Any]=None, value:Optional[int]=None):
    """
    Updates a previously queued wait command.

    Args:
      cmd_idx: Index of the wait command to update
      signal: New signal to wait on (if None, keeps the original)
      value: New value to wait for (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "wait": raise RuntimeError("called update_wait not on a wait command")
    self._update_wait(cmd_idx, signal, value)
    return self
  def _update_wait(self, cmd_idx:int, signal:Optional[Any], value:Optional[int]): raise NotImplementedError("backend should overload this function")

  def bind(self, device:HCQCompiled):
    """
    Associates the queue with a specific device for optimized execution.

    This optional method allows backend implementations to tailor the queue for efficient use on the given device. When implemented, it can eliminate
    the need to copy queues into the device, thereby enhancing performance.

    Args:
      device: The target device for queue optimization.

    Note:
      Implementing this method is optional but recommended for performance gains.
    """

  def submit(self, device:HCQCompiled):
    """
    Submits the command queue to a specific device for execution.

    Args:
      device: The device to submit the queue to
    """
    if self.q: self._submit(device)
    return self
  def _submit(self, device:HCQCompiled): raise NotImplementedError("backend should overload this function")

class HWComputeQueue(HWCommandQueue):
  @hcq_command
  def memory_barrier(self):
    """
    Enqueues a memory barrier command to ensure memory coherence between agents.
    """
    self._memory_barrier()
  def _memory_barrier(self): pass

  @hcq_command
  def exec(self, prg:HCQProgram, args_state:HCQArgsState, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int]):
    """
    Enqueues an execution command for a kernel program.

    Args:
      prg: The program to execute
      args_state: The args state to execute program with
      global_size: The global work size
      local_size: The local work size
    """
    self._exec(prg, args_state, global_size, local_size)
  def _exec(self, prg, args_state, global_size, local_size): raise NotImplementedError("backend should overload this function")

  def update_exec(self, cmd_idx:int, global_size:Optional[Tuple[int,int,int]]=None, local_size:Optional[Tuple[int,int,int]]=None):
    """
    Updates a previously queued execution command.

    Args:
      cmd_idx: Index of the execution command to update
      global_size: New global work size (if None, keeps the original)
      local_size: New local work size (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "exec": raise RuntimeError("called update_exec not on an exec command")
    self._update_exec(cmd_idx, global_size, local_size)
    return self
  def _update_exec(self, cmd_idx, global_size, local_size): raise NotImplementedError("backend should overload this function")

class HWCopyQueue(HWCommandQueue):
  @hcq_command
  def copy(self, dest:HCQBuffer, src:HCQBuffer, copy_size:int):
    """
    Enqueues a copy command to transfer data.

    Args:
      dest: The destination of the copy
      src: The source of the copy
      copy_size: The size of data to copy
    """
    self._copy(dest, src, copy_size)
  def _copy(self, dest:HCQBuffer, src:HCQBuffer, copy_size:int): raise NotImplementedError("backend should overload this function")

  def update_copy(self, cmd_idx:int, dest:Optional[HCQBuffer]=None, src:Optional[HCQBuffer]=None):
    """
    Updates a previously queued copy command.

    Args:
      cmd_idx: Index of the copy command to update
      dest: New destination of the copy (if None, keeps the original)
      src: New source of the copy (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "copy": raise RuntimeError("called update_copy not on an copy command")
    self._update_copy(cmd_idx, dest, src)
    return self
  def _update_copy(self, cmd_idx, dest, src): raise NotImplementedError("backend should overload this function")

class HCQSignal:
  def __init__(self, value:int=0): self._set_value(value)

  @property
  def value(self) -> int: return self._get_value()

  @value.setter
  def value(self, new_value:int): self._set_value(new_value)

  def _get_value(self) -> int: raise NotImplementedError("_get_value() method must be implemented")
  def _set_value(self, new_value:int): raise NotImplementedError("_set_value() method must be implemented")

  @property
  def timestamp(self) -> decimal.Decimal:
    """
    Get the timestamp field of the signal.

    This property provides read-only access to the signal's timestamp.

    Returns:
      The timestamp in microseconds.
    """
    return self._get_timestamp()
  def _get_timestamp(self) -> decimal.Decimal: raise NotImplementedError("_get_timestamp() method must be implemented")

  def wait(self, value:int, timeout:int=10000):
    """
    Waits the signal is greater than or equal to a specific value.

    Args:
      value: The value to wait for.
      timeout: Maximum time to wait in milliseconds. Defaults to 10s.
    """
    start_time = time.time() * 1000
    while time.time() * 1000 - start_time < timeout:
      if self.value >= value: return
    raise RuntimeError(f"Wait timeout: {timeout} ms! (the signal is not set to {value}, but {self.value})")

@contextlib.contextmanager
def hcq_profile(dev, enabled, desc, queue_type=None, queue=None):
  st, en = (dev.signal_t(), dev.signal_t()) if enabled else (None, None)

  if enabled and queue is not None: queue.timestamp(st)
  elif enabled:
    queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(st).signal(dev.timeline_signal, dev.timeline_value).submit(dev)
    dev.timeline_value += 1

  try: yield (st, en)
  finally:
    if enabled and queue is not None: queue.timestamp(en)
    elif enabled:
      queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(en).signal(dev.timeline_signal, dev.timeline_value).submit(dev)
      dev.timeline_value += 1

    if enabled and PROFILE: dev.sig_prof_records.append((st, en, desc, queue_type is dev.hw_copy_queue_t))

class HCQArgsState:
  def __init__(self, ptr:int, prg:HCQProgram, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=()): self.ptr, self.prg = ptr, prg
  def update_buffer(self, index:int, buf:HCQBuffer): raise NotImplementedError("need update_buffer")
  def update_var(self, index:int, val:int): raise NotImplementedError("need update_var")

class HCQProgram:
  def __init__(self, args_state_t:Type[HCQArgsState], device:HCQCompiled, name:str, kernargs_alloc_size:int):
    self.args_state_t, self.device, self.name, self.kernargs_alloc_size = args_state_t, device, name, kernargs_alloc_size

  def fill_kernargs(self, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=(), kernargs_ptr:Optional[int]=None) -> HCQArgsState:
    """
    Fills arguments for the kernel, optionally allocating space from the device if `kernargs_ptr` is not provided.
    Args:
      bufs: Buffers to be written to kernel arguments.
      vals: Values to be written to kernel arguments.
      kernargs_ptr: Optional pointer to pre-allocated kernel arguments memory.
    Returns:
      Arguments state with the given buffers and values set for the program.
    """
    return self.args_state_t(kernargs_ptr or self.device._alloc_kernargs(self.kernargs_alloc_size), self, bufs, vals=vals)

  def __call__(self, *bufs:HCQBuffer, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1),
               vals:Tuple[int, ...]=(), wait:bool=False) -> Optional[float]:
    """
    Enqueues the program for execution with the given arguments and dimensions.

    Args:
      bufs: Buffer arguments to execute the kernel with.
      global_size: Specifies the global work size for kernel execution (equivalent to CUDA's grid size).
      local_size: Specifies the local work size for kernel execution (equivalent to CUDA's block size).
      vals: Value arguments to execute the kernel with.
      wait: If True, waits for the kernel to complete execution.

    Returns:
      Execution time of the kernel if 'wait' is True, otherwise None.
    """

    kernargs = self.fill_kernargs(bufs, vals)
    q = self.device.hw_compute_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1).memory_barrier()

    with hcq_profile(self.device, queue=q, desc=self.name, enabled=wait or PROFILE) as (sig_st, sig_en):
      q.exec(self, kernargs, global_size, local_size)

    q.signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
    self.device.timeline_value += 1

    if wait: self.device.timeline_signal.wait(self.device.timeline_value - 1)
    return (float(sig_en.timestamp - sig_st.timestamp) / 1e6) if wait else None

class ProfileLogger:
  writers: int = 0
  mjson: List[Dict] = []
  actors: Dict[Union[str, Tuple[str, str]], int] = {}

  def __init__(self): self.events, self.deps, ProfileLogger.writers = [], [], ProfileLogger.writers + 1

  def add_event(self, ev_name, ev_start, ev_end, actor, subactor=None, args=None): self.events += [(ev_name, ev_start, ev_end, actor, subactor, args)]

  def _ensure_actor(self, actor_name, subactor_name):
    if actor_name not in self.actors:
      self.actors[actor_name] = (pid:=len(self.actors))
      self.mjson.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": actor_name}})

    if (subactor_key:=(actor_name,subactor_name)) not in self.actors:
      self.actors[subactor_key] = (tid:=len(self.actors))
      self.mjson.append({"name": "thread_name", "ph": "M", "pid": self.actors[actor_name], "tid":tid, "args": {"name": subactor_name}})

    return self.actors[actor_name], self.actors.get(subactor_key, -1)

  def __del__(self):
    # perfetto json docs: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    for name, st, et, actor_name, subactor_name, args in self.events:
      pid, tid = self._ensure_actor(actor_name,subactor_name)
      args = {k: (v if v.__class__ is str else v(et-st)) for k, v in args.items()} if args is not None else None
      self.mjson.append({"name": name, "ph": "X", "pid": pid, "tid": tid, "ts": st, "dur": et-st, "args": args})

    for en,st,dep_actor_name,dep_subactor_name,actor_name,subactor_name in self.deps:
      dep_pid, dep_tid = self._ensure_actor(dep_actor_name,dep_subactor_name)
      pid, tid = self._ensure_actor(actor_name,subactor_name)
      self.mjson.append({"ph": "s", "pid": dep_pid, "tid": dep_tid, "id": len(self.mjson), "ts": en, "bp": "e"})
      self.mjson.append({"ph": "f", "pid": pid, "tid": tid, "id": len(self.mjson)-1, "ts": st, "bp": "e"})

    ProfileLogger.writers -= 1
    if ProfileLogger.writers == 0 and len(self.mjson) > 0:
      with open(PROFILEPATH.value, "w") as f: f.write(json.dumps({"traceEvents": self.mjson}))
      print(f"Saved profile to {PROFILEPATH.value}. Use https://ui.perfetto.dev/ to open it.")

class HCQCompiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  devices: List[HCQCompiled] = []
  gpu2cpu_copy_time_diff: decimal.Decimal = decimal.Decimal('nan')
  gpu2cpu_compute_time_diff: decimal.Decimal = decimal.Decimal('nan')

  def __init__(self, device:str, allocator:Allocator, renderer:Renderer, compiler:Compiler, runtime, signal_t:Type[HCQSignal],
               comp_queue_t:Type[HWComputeQueue], copy_queue_t:Optional[Type[HWCopyQueue]], timeline_signals:Tuple[HCQSignal, HCQSignal]):
    self.signal_t, self.hw_compute_queue_t, self.hw_copy_queue_t = signal_t, comp_queue_t, copy_queue_t
    self.timeline_value:int = 1
    self.timeline_signal, self._shadow_timeline_signal = timeline_signals
    self.sig_prof_records:List[Tuple[HCQSignal, HCQSignal, str, bool]] = []
    self.raw_prof_records:List[Tuple[decimal.Decimal, decimal.Decimal, str, bool, Optional[Dict]]] = []
    self.dep_prof_records:List[Tuple[decimal.Decimal, decimal.Decimal, HCQCompiled, bool, decimal.Decimal, decimal.Decimal, HCQCompiled, bool]] = []
    if PROFILE: self._prof_setup()

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, allocator, renderer, compiler, runtime, HCQGraph)

    self.kernargs_page:HCQBuffer = self.allocator.alloc(16 << 20, BufferOptions(cpu_access=True))
    self.kernargs_ptr:int = self.kernargs_page.va_addr
    self.devices.append(self)

  def synchronize(self):
    self.timeline_signal.wait(self.timeline_value - 1) if not hasattr(self, '_syncdev') else self._syncdev()

    if self.timeline_value > (1 << 31): self._wrap_timeline_signal()
    if PROFILE:
      self.raw_prof_records += [(st.timestamp, en.timestamp, name, is_cp, None) for st, en, name, is_cp in self.sig_prof_records]
      self.sig_prof_records = []

  def _alloc_kernargs(self, alloc_size:int) -> int:
    """
    Allocates space for arguments passed to the kernel.
    """
    if self.kernargs_ptr >= (self.kernargs_page.va_addr + self.kernargs_page.size - alloc_size): self.kernargs_ptr = self.kernargs_page.va_addr
    self.kernargs_ptr = (res:=self.kernargs_ptr) + alloc_size
    return res

  def _ensure_shared_time_base(self):
    if not self.gpu2cpu_compute_time_diff.is_nan(): return

    def _sync_cpu_queue(d, q_t):
      q_t().timestamp(d.timeline_signal).signal(d.timeline_signal, d.timeline_value).submit(d)
      d.timeline_value += 1
      st = time.perf_counter_ns()
      d.timeline_signal.wait(d.timeline_value - 1)  # average of the two
      et = time.perf_counter_ns()
      return (decimal.Decimal(et+st) / 2000) - d.timeline_signal.timestamp

    # randomly sample the timing from GPU to CPU
    choices: List = [(d, d.hw_compute_queue_t, []) for d in self.devices]
    choices += [(d, d.hw_copy_queue_t, []) for d in self.devices if d.hw_copy_queue_t is not None]
    for _ in range(100*len(self.devices)):
      d,q,l = random.choice(choices)
      l.append(_sync_cpu_queue(d,q))
    for d,q,l in choices:
      if q == d.hw_compute_queue_t: d.gpu2cpu_compute_time_diff = statistics.median(l)
      if q == d.hw_copy_queue_t: d.gpu2cpu_copy_time_diff = statistics.median(l)

    def _sync_gpu_to_gpu_queue(d1, d2, q1_t, q2_t):
      q1_t().signal(d1.timeline_signal, d1.timeline_value).wait(d2.timeline_signal, d2.timeline_value) \
            .timestamp(d1.timeline_signal).signal(d1.timeline_signal, d1.timeline_value+1).submit(d1)
      q2_t().signal(d2.timeline_signal, d2.timeline_value).wait(d1.timeline_signal, d1.timeline_value) \
            .timestamp(d2.timeline_signal).signal(d2.timeline_signal, d2.timeline_value+1).submit(d2)
      d1.timeline_value += 2
      d2.timeline_value += 2
      d1.timeline_signal.wait(d1.timeline_value - 1)
      d2.timeline_signal.wait(d2.timeline_value - 1)
      return d2.timeline_signal.timestamp - d1.timeline_signal.timestamp

    # then test it by timing the GPU to GPU times
    jitter_matrix = [[float('nan')]*len(self.devices) for _ in range(len(self.devices))]
    for i1, d1 in enumerate(self.devices):
      for i2, d2 in enumerate(self.devices):
        if d1 == d2: continue
        d1_to_d2 = statistics.median(_sync_gpu_to_gpu_queue(d1, d2, d1.hw_compute_queue_t, d2.hw_compute_queue_t) - \
                                     _sync_gpu_to_gpu_queue(d2, d1, d2.hw_compute_queue_t, d1.hw_compute_queue_t) for _ in range(20)) / 2
        jitter_matrix[i1][i2] = d1_to_d2 - (d1.gpu2cpu_compute_time_diff - d2.gpu2cpu_compute_time_diff)
    print("pairwise clock jitter matrix (us):\n" + '\n'.join([''.join([f'{float(item):8.3f}' for item in row]) for row in jitter_matrix]))

  def _gpu2cpu_time(self, gpu_time:decimal.Decimal, is_copy:bool) -> float:
    """
    Translates local gpu time (timestamp) into global cpu time.
    """
    self._ensure_shared_time_base()
    return float(gpu_time + (self.gpu2cpu_copy_time_diff if is_copy else self.gpu2cpu_compute_time_diff))

  def _prof_setup(self):
    if hasattr(self, 'profile_logger'): return
    atexit.register(self._prof_finalize)
    self.profile_logger = ProfileLogger()

  def _prof_finalize(self):
    qname = ["COMPUTE", "DMA"]

    # Sync to be sure all events on the device are recorded.
    self.synchronize()

    for st, en, name, is_cp, args in self.raw_prof_records:
      self.profile_logger.events += [(name, self._gpu2cpu_time(st, is_cp), self._gpu2cpu_time(en, is_cp), self.dname, qname[is_cp], args)]
    for a_st, a_en, a_dev, a_is_copy, b_st, b_en, b_dev, b_is_copy in self.dep_prof_records:
      # Perfetto connects nodes based on timing data, ensuring every choice is valid by averaging times to a midpoint.
      a_tm, b_tm = a_dev._gpu2cpu_time((a_st+a_en)/decimal.Decimal(2), a_is_copy), b_dev._gpu2cpu_time((b_st+b_en)/decimal.Decimal(2), b_is_copy)
      self.profile_logger.deps += [(a_tm, b_tm, a_dev.dname, qname[a_is_copy], b_dev.dname, qname[b_is_copy])]
    self.raw_prof_records, self.dep_prof_records = [], []

    # Remove the logger, this flushes all data written by the device.
    del self.profile_logger

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self.timeline_signal.value = 0
    cast(HCQAllocator, self.allocator).b_timeline = [0] * len(cast(HCQAllocator, self.allocator).b)

# Protocol for hcq compatible allocators for allocated buffers to contain VA address and it's size.
class HCQBuffer(Protocol): va_addr:int; size:int # noqa: E702

class HCQAllocator(LRUAllocator): # pylint: disable=abstract-method
  """
  A base allocator class compatible with the HCQ (Hardware Command Queue) API.

  This class implements basic copy operations following the HCQ API, utilizing both `HWComputeQueue` and `HWCopyQueue`.
  """

  def __init__(self, device:HCQCompiled, batch_size:int=(2 << 20), batch_cnt:int=32):
    self.device:Any = device
    self.b = [self._alloc(batch_size, BufferOptions(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next = [0] * len(self.b), 0
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer: raise NotImplementedError("need hcq compat alloc")

  def copyin(self, dest:HCQBuffer, src:memoryview):
    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"CPU -> {self.device.dname}", enabled=PROFILE):
      for i in range(0, src.nbytes, self.b[0].size):
        self.b_next = (self.b_next + 1) % len(self.b)
        self.device.timeline_signal.wait(self.b_timeline[self.b_next])
        ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].size, src.nbytes-i))
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.b_timeline[self.b_next] = self.device.timeline_value
        self.device.timeline_value += 1

  def copy_from_disk(self, dest:HCQBuffer, src, size):
    def _get_temp_buf():
      # Check if the next buffer is safe to be used (its signal has passed) and reserve it.
      if self.b_timeline[(self.b_next + 1) % len(self.b)] <= self.device.timeline_signal.value:
        self.b_timeline[(self.b_next + 1) % len(self.b)], self.b_next = (1 << 64), (self.b_next + 1) % len(self.b)
        return (self.b[self.b_next].va_addr, self.b_next)
      return None

    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"DISK -> {self.device.dname}", enabled=PROFILE):
      for (batch_info, dst_off, src_off, copy_size) in src.device.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=self.b[0].size):
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(dest.va_addr + dst_off, batch_info[0] + src_off, copy_size) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.b_timeline[batch_info[1]] = self.device.timeline_value
        self.device.timeline_value += 1

  def copyout(self, dest:memoryview, src:HCQBuffer):
    self.device.synchronize()

    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"{self.device.dname} -> CPU", enabled=PROFILE):
      for i in range(0, dest.nbytes, self.b[0].size):
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i)) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.device.timeline_signal.wait(self.device.timeline_value)
        self.device.timeline_value += 1

        ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest:HCQBuffer, src:HCQBuffer, sz:int, src_dev, dest_dev):
    src_dev.allocator.map(dest)

    with hcq_profile(src_dev, queue_type=src_dev.hw_copy_queue_t, desc=f"{src_dev.dname} -> {dest_dev.dname}", enabled=PROFILE):
      src_dev.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                               .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                               .copy(dest.va_addr, src.va_addr, sz) \
                               .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
      src_dev.timeline_value += 1

    if src_dev != dest_dev:
      dest_dev.hw_compute_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                   .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                   .signal(dest_dev.timeline_signal, dest_dev.timeline_value).submit(dest_dev)
      dest_dev.timeline_value += 1

  def map(self, buf:HCQBuffer): pass

  def offset(self, buf, size:int, offset:int) -> HCQBuffer:
    return type(buf)(va_addr=buf.va_addr + offset, size=size, **{k:v for k,v in buf.__dict__.items() if k not in ['va_addr', 'size']},
                     **{x[0]:getattr(buf, x[0]) for x in getattr(buf, '_fields_', []) if x[0] not in ['va_addr', 'size']}, _base=buf)
