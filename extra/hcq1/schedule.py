"""Optional static compute/copy scheduling using conservative buffer hazards and fixed cost estimates."""
import functools, heapq
from tinygrad.device import DepsTracker, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import ALL2ALL, getenv
from tinygrad.uop.ops import Ops

def write_slots(ast, bufs):
  # Some opaque native kernels expose only parameters, despite writing outputs. Never treat those as read-only.
  return (ast.arg.outs or tuple(range(len(bufs)))) if ast.op is Ops.PROGRAM else (0,)

def copy_queue_index(devices, source, target, nbytes, num_copy, small_copies=True):
  # Eight-way peer transfers normally leave the sender's own destination queue unused.
  # Keep short transfers off the long-copy queues, without changing their buffer hazards.
  if small_copies and 0 < nbytes <= 512*1024 and num_copy == len(devices) == 8 and source is not target and \
      len({d.peer_group for d in devices}) == 1 and all(d.device.startswith("AMD") and not d._is_cpu() for d in devices):
    return devices.index(source)
  return devices.index(target) % num_copy

def schedule(deps, queues, durations, mode):
  successors = [[] for _ in deps]
  for i, predecessors in enumerate(deps):
    assert all(j < i for j in predecessors)
    for j in predecessors: successors[j].append(i)
  finish, free = [0.0] * len(deps), {}
  if mode == "original":
    for i, predecessors in enumerate(deps):
      finish[i] = max(free.get(queues[i], 0.0), max((finish[j] for j in predecessors), default=0.0)) + durations[i]
      free[queues[i]] = finish[i]
    return list(range(len(deps))), max(finish, default=0.0)
  assert mode in ("index", "critical")
  critical = [0.0] * len(deps)
  for i in reversed(range(len(deps))): critical[i] = durations[i] + max((critical[j] for j in successors[i]), default=0.0)
  counts, earliest, ready, order = list(map(len, deps)), [0.0] * len(deps), [], []
  for i, count in enumerate(counts):
    if not count: heapq.heappush(ready, (0.0, -critical[i] if mode == "critical" else i, i))
  while ready:
    start, priority, i = heapq.heappop(ready)
    if start < free.get(queues[i], 0.0):
      heapq.heappush(ready, (free[queues[i]], priority, i))
      continue
    finish[i] = start + durations[i]
    free[queues[i]] = finish[i]
    order.append(i)
    for j in successors[i]:
      earliest[j] = max(earliest[j], finish[i])
      counts[j] -= 1
      if not counts[j]: heapq.heappush(ready, (max(earliest[j], free.get(queues[j], 0.0)), -critical[j] if mode == "critical" else j, j))
  assert len(order) == len(deps)
  return order, max(finish, default=0.0)

def eager_ready_order(deps, order, eager):
  """Finish selected ready operations promptly, preserving every hazard and all other relative order."""
  successors, pending = [[] for _ in deps], list(map(len, deps))
  for i, predecessors in enumerate(deps):
    for j in predecessors: successors[j].append(i)
  positions, emitted, result = {i:p for p,i in enumerate(order)}, set(), []
  ready = [(positions[i], i) for i in order if eager[i] and not pending[i]]
  heapq.heapify(ready)
  for root in order:
    if root not in emitted: heapq.heappush(ready, (len(order), root))
    while ready:
      _, i = heapq.heappop(ready)
      if i in emitted: continue
      assert pending[i] == 0
      emitted.add(i)
      result.append(i)
      for j in successors[i]:
        pending[j] -= 1
        if not pending[j] and eager[j]: heapq.heappush(ready, (positions[j], j))
  assert [i for i in result if not eager[i]] == [i for i in order if not eager[i]]
  return result

@functools.cache
def is_storage_copy(ast):
  """Inspect the actual kernel dataflow, not its generated name or tracing metadata."""
  stores = [u for u in ast.src[0].toposort() if u.op is Ops.STORE]
  return bool(stores) and all(s.src[1].op is Ops.LOAD and s.src[0].dtype == s.src[1].dtype and
    s.src[0].buf_uop.op is Ops.PARAM and s.src[0].buf_uop.arg.slot == 0 and
    s.src[1].src[0].buf_uop.op is Ops.PARAM and s.src[1].src[0].buf_uop.arg.slot == 1 for s in stores)

def fc1_completion_mask(calls, deps, staging=False):
  # Exact native entrypoint from fused_adam_mxfp8's GPT-OSS main/tail ABI. Whole q shards wait for their tails.
  tail = "fused_adam_mxfp8_72351744_fc1pipe1_rowskip1_split_tail_rawclip1"
  names = {tail, tail+"_compactq1", "gptoss_gather_local_copy"}
  eager = [ast.op is Ops.PROGRAM and ast.arg.name in names for _, ast, _, _ in calls]
  if staging:
    for i, (_, ast, bufs, _) in enumerate(calls):
      # Only continue through actual storage copies of the completed FC1 shard. Keep scale/reduction math unchanged.
      if any(eager[j] for j in deps[i]) and (ast.op is Ops.COPY or
          len(bufs) == 2 and bufs[0].nbytes == bufs[1].nbytes and bufs[0].nbytes in (72351744, 66355200) and is_storage_copy(ast)):
        eager[i] = True
  return eager

@functools.cache
def is_bf16_round_copy(ast):
  """Only a float32 load rounded to BF16 and stored; no optimizer arithmetic."""
  stores = [u for u in ast.src[0].toposort() if u.op is Ops.STORE]
  return bool(stores) and all(s.src[0].dtype == dtypes.bfloat16 and s.src[1].op is Ops.CAST and
    s.src[1].src[0].op is Ops.LOAD and s.src[1].src[0].dtype == dtypes.float32 and
    s.src[0].buf_uop.op is Ops.PARAM and s.src[0].buf_uop.arg.slot == 0 and
    s.src[1].src[0].src[0].buf_uop.op is Ops.PARAM and s.src[1].src[0].src[0].buf_uop.arg.slot == 1 for s in stores)

@functools.cache
def is_bf16_vocab_gather(ast):
  stores = [u for u in ast.src[0].toposort() if u.op is Ops.STORE]
  if not stores or not all(s.src[0].dtype == s.src[1].dtype == dtypes.bfloat16 and s.src[1].op is Ops.LOAD and
      s.src[0].buf_uop.op is Ops.PARAM and s.src[0].buf_uop.arg.slot == 0 and
      s.src[1].src[0].buf_uop.op is Ops.PARAM for s in stores): return False
  return {s.src[1].src[0].buf_uop.arg.slot for s in stores} == set(range(1, 9))

def is_vocab_transport(ast, bufs):
  if len(bufs) != 2 or bufs[0].nbytes != 92344320: return False
  if ast.op is Ops.COPY: return bufs[1].nbytes == 92344320
  return ast.op is Ops.PROGRAM and ((bufs[1].nbytes == 92344320 and is_storage_copy(ast)) or
                                    (bufs[1].nbytes == 184688640 and is_bf16_round_copy(ast)))

def vocab_completion_mask(calls, deps):
  # Inspected fused_adam_bf16_vocab ABI, shared by embedding and LM-head. Advance only its output transport,
  # not BF16 gradient reductions of the same size. Both vocabulary consumers retain all their hazards.
  eager = [ast.op is Ops.PROGRAM and ast.arg.name == "fused_adam_bf16_vocab_46172160_rawclip1" for _, ast, _, _ in calls]
  for i, (_, ast, bufs, _) in enumerate(calls):
    if any(eager[j] for j in deps[i]) and is_vocab_transport(ast, bufs): eager[i] = True
  # A graph boundary can separate the native update from its transfers. Trace backwards from the actual
  # eight-shard BF16 assembly too, stopping at math. Do not move the assembly itself on the compute queue.
  needed = [ast.op is Ops.PROGRAM and len(bufs) == 9 and bufs[0].nbytes == 738754560 and
            all(b.nbytes == 92344320 for b in bufs[1:]) and is_bf16_vocab_gather(ast) for _, ast, bufs, _ in calls]
  for i in reversed(range(len(calls))):
    if not needed[i]: continue
    for j in deps[i]:
      _, ast, bufs, _ = calls[j]
      if is_vocab_transport(ast, bufs): eager[j] = needed[j] = True
  return eager

def reorder_graph(graph):
  _reorder_graph(graph)
  if getenv("GPTOSS_EMBEDDING_READY", 0):
    from extra.hcq1.gptoss import reorder_embedding
    reorder_embedding(graph)

def _reorder_graph(graph):
  if len(graph.calls) < 1000 or any(d._is_cpu() or not d.device.startswith("AMD") for d in graph.devices): return
  if len({d.peer_group for d in graph.devices}) != 1: return
  tracker, deps, queues, durations = DepsTracker(), [], [], []
  num_copy = getenv("HCQ_NUM_SDMA", min(len(graph.devices), 8) if ALL2ALL >= 1 else 1)
  for i, ((_, ast, bufs, _), runtime) in enumerate(zip(graph.calls, graph.runtimes)):
    assert ast.op in (Ops.COPY, Ops.PROGRAM)
    deps.append(sorted(set(tracker.access_resources(bufs, write_slots(ast, bufs), i))))
    if runtime is not None:
      queues.append((runtime.dev.device, "compute"))
      durations.append(0.1)
    else:
      enqueue = next(Device[b.device] for b in bufs[::-1] if Device[b.device].hw_copy_queue_t is not None)
      queues.append((enqueue.device, copy_queue_index(graph.devices, enqueue, Device[bufs[0].device], bufs[0].nbytes, num_copy)))
      durations.append(max(0.002, bufs[0].nbytes / 56e6))
  options = {mode: schedule(deps, queues, durations, mode) for mode in ("original", "index", "critical")}
  mode = min(options, key=lambda m: options[m][1])
  if options[mode][1] >= options["original"][1] - 0.05: mode = "original"
  order = options[mode][0]
  if (completion := getenv("GPTOSS_FC1_READY_TAIL", 0)):
    assert completion in (1, 2)
    eager = fc1_completion_mask(graph.calls, deps, staging=completion == 2)
    order = eager_ready_order(deps, order, eager)
  if getenv("GPTOSS_VOCAB_READY", 0):
    order = eager_ready_order(deps, order, vocab_completion_mask(graph.calls, deps))
  if order == list(range(len(deps))): return
  positions = {old: new for new, old in enumerate(order)}
  assert set(positions) == set(range(len(deps)))
  assert all(positions[d] < positions[i] for i, predecessors in enumerate(deps) for d in predecessors)
  for field in ("calls", "runtimes", "uop_replace"):
    old = getattr(graph, field)
    setattr(graph, field, [old[i] for i in order])
  for field in ("var_vals_replace", "launch_dims_replace", "launch_dims_base"):
    setattr(graph, field, {positions[k]: v for k, v in getattr(graph, field).items()})
