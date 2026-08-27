from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, Sequence, Iterable
import struct, functools, time, collections, itertools, decimal, statistics
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, JIT_BATCH_SIZE, unwrap, PROFILE
from tinygrad.helpers import to_tuple, round_up, partition, panic, ContextVar, perf_counter_us, Context
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer, DepsTracker
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, rewrite_group, GroupOp
from tinygrad.uop.symbolic import symbolic
from tinygrad.dtype import dtypes, truncate, DType
from tinygrad.runtime.support.hcq import MMIOInterface, HCQBuffer
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop
from tinygrad.engine.realize import pm_flatten_linear, lower_and_compile

# *****************
# 0. helpers

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "CPU")

HCQ_DEVS = frozenset(("AMD", "CPU"))
HCQ_CACHE_TAGS = frozenset(("program", "systems"))

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]
  estimates:Estimates = Estimates()

  inputs:int|None = None
  input_addrs:tuple[tuple[str, UOp], ...] = () # (device, lane arg uop)
  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = ()

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_mstack(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.MSTACK: return tuple(x for s in u.src for x in unwrap_mstack(s))
  return unwrap_mstack(u.src[0]) if u.op is Ops.MSELECT else (u,)

def unwrap_view(v:UOp) -> tuple[UOp, int]:
  return unwrap_view(v.src[0]) if v.op is Ops.BITCAST else (v.src[0], v.src[1].val) if v.op is Ops.SHRINK else (v, 0)

def _lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

# patches

def is_value_known_at_link(val:UOp) -> bool:
  runtime_reads = [u for u in val.toposort() if u.op in (Ops.LOAD, Ops.INDEX)]
  addressed_bufs = [b for g in val.toposort() if g.op is Ops.GETADDR for b in unwrap_mstack(g.buf_uop)]

  # addr of input params is not known at link time
  return not val.variables() and not runtime_reads and all(b.op is not Ops.PARAM or b.tag is not None for b in addressed_bufs)

def make_patches(buf:UOp, patches:Sequence[tuple[sint, UOp]]) -> tuple[UOp, ...]:
  # group patches into stacks: (tag, type, offset). offset is used for shrink later
  groups:dict[tuple[str|None, DType, sint], list[tuple[sint, UOp]]] = collections.defaultdict(list)
  for off, val in patches:
    tag = "link" if is_value_known_at_link(val) else "inputs" if val.op is Ops.GETADDR else None
    groups[(tag, (v:=(val.bitcast(buf.dtype) if val.dtype.itemsize == buf.dtype.itemsize else val)).dtype, off % v.dtype.itemsize)].append((off, v))

  ret, bit = [], buf.dtype.itemsize
  for (tag, dt, r), ps in groups.items():
    view = buf.shrink(((r // bit, (max(off for off,_ in ps) + dt.itemsize) // bit),)).bitcast(dt)
    offs = UOp(Ops.STACK, src=tuple(UOp.const((off - r) // dt.itemsize, dtypes.int) for off,_ in ps))
    ret.append(view.index(offs).store(UOp(Ops.STACK, src=tuple(val for _,val in ps))).rtag(tag))
  return tuple(ret)

def make_binary_patch(buf:UOp, blob:bytes) -> UOp: return buf.store(UOp(Ops.BINARY, src=(), arg=blob).bitcast(buf.dtype)).rtag("link")

def make_cmdbuf(lin, devs, buf:UOp|None=None, dep:tuple[UOp, ...]=()):
  blob, patches = bytearray(), []
  for s in (s for ins in lin.src for s in ins.src):
    if not (is_const:=(s.op is Ops.CAST and s.src[0].op is Ops.CONST)): patches.append((len(blob), s))
    blob.extend(struct.pack(f'<{s.dtype.fmt}', s.val if is_const else 0x0))
  cmdbuf = buf if buf is not None else UOp.placeholder((len(blob) // 4,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("cmdbuf")
  return cmdbuf.after(*dep, make_binary_patch(cmdbuf, bytes(blob)), *make_patches(cmdbuf, patches))

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  return UOp.custom_function("submit_cmdbuf", UOp(Ops.LINEAR, src=tuple(cmds), arg=(to_tuple(devs), queue)))
def get_submit(ast:UOp) -> UOp: return next(u for u in ast.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_cmdbuf")

def make_call(name:str, body:UOp, info:HCQInfo) -> UOp: return UOp.custom_function("hcq", body).call(name=name, aux=info)

def encode_kernargs_clike(call:UOp, prg:UOp, devs:str|tuple[str, ...]) -> UOp:
  data, info = prg.arg
  buf = UOp.placeholder((data.kernargs_alloc_size // 4,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("kernargs")
  words = [get_call_arg_uops(call)[gi].getaddr(devs) for gi in info.globals] + list(info.vars)
  return buf.after(*make_patches(buf, list(zip(itertools.accumulate((w.dtype.itemsize for w in words), initial=0), words))))

def make_buf(devs, slot:int=0, tag:str="signal") -> UOp: return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True, tag=tag)

# *****************
# 0.1. prep: replace buffers with params

def replace_call_buffers(ctx:tuple[list[UOp], dict[UOp, int]], call:UOp) -> UOp|None:
  bufs, slots = ctx
  for s in call.src[1:]:
    if s.op is not Ops.PARAM and not s.is_bound_var and slots.setdefault(s, len(bufs)) == len(bufs): bufs.append(s)
  return call.replace(src=call.src[:1] + tuple(s if s.op is Ops.PARAM or s.is_bound_var else s.param_like(slots[s]) for s in call.src[1:]))
pm_replace_buffers = PatternMatcher([(UPat(Ops.CALL, name="call"), replace_call_buffers)])

# *****************
# 1.1. prep: staging copies

STAGING_SIZE, STAGING_SLOTS = 128 << 20, 2

@functools.cache
def _staging() -> Buffer: return Buffer("CPU", STAGING_SIZE, dtypes.uint8, preallocate=True)

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS - {"CPU"}) and not all_devices_in(b.device, HCQ_DEVS)

def stage_copy_ext(call:UOp) -> UOp|None:
  if (d:=next((d for b in call.src[1:] for d in to_tuple(b.device) if not d.startswith("CPU")), None)) is None: return None
  return pm.rewrite(call) if (pm:=getattr(Device[d], "pm_stage_copy", None)) is not None else None

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  assert src.dtype.itemsize == dst.dtype.itemsize, "staged copies must be dtype-size matched"
  base, it, copies = UOp.from_buffer(_staging()), src.dtype.itemsize, []
  chunk = (STAGING_SIZE // STAGING_SLOTS) // it
  for i, off in enumerate(range(0, src.max_numel(), chunk)):
    stage = base[(so:=(i % STAGING_SLOTS) * chunk * it):so + (n:=min(chunk, src.max_numel() - off)) * it]
    copies += [src[off:off+n].copy_to_device("CPU").call(stage, src[off:off+n]), stage.copy_to_device(dst.device).call(dst[off:off+n], stage)]
  return UOp(Ops.LINEAR, src=tuple(copies))

# *****************
# 1.2. prep: kernel copies

def _get_enqueue_devs(call:UOp) -> Any|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=call.src[1:]) or not all(all_devices_in(b.device, HCQ_DEVS) for b in bufs): return None
  if call.src[0].op is Ops.COPY: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: to_tuple(b.device)[0].startswith("CPU")).device # prio to enqueue on not CPU device
  return devs if all_devices_in(devs, HCQ_DEVS) else None

def copy_with_kernel(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  if (devs:=_get_enqueue_devs(call)) is None or Device[(dev:=to_tuple(devs)[0])].has_copy_queue: return None
  d, s = (UOp.param(i, dst.dtype, (n:=dst.max_numel(),), device=devs) for i in range(2))
  ast = d.index(r:=UOp.range(n, 0)).store(s.index(r).load()).end(r).sink(arg=KernelInfo(name="copy"), tag=1)
  return call.replace(src=(to_program(ast, Device[dev].renderer), dst, src))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY),), name="call", allow_any_len=True), stage_copy_ext),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src")), name="call"), copy_with_kernel)
])

# *****************
# 2. deps

class HCQDepsTracker(DepsTracker):
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    if isinstance(buf, UOp) and buf.op is Ops.MSELECT: buf = buf.src[0]
    return (buf.arg.slot, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

@dataclass(frozen=True)
class Dep: dev:str; queue:str; tag:int; lane:int # producer submit (dev, queue, tag) awaited by consumer lane # noqa: E702

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...]]]; batch_info:list[tuple[tuple[str, ...], str]]; profile:bool # noqa: E702
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker); signal_tags:set[int] = field(default_factory=set) # noqa: E702
  slots:dict[str, int] = field(default_factory=lambda: collections.defaultdict(lambda: next(UOp.unique_num)))

def _get_call_bufs_by_lane(call:UOp, devices:tuple[str, ...]) -> list[list[Any]]:
  return [[b if (b:=_lane(a, lane)).op is Ops.PARAM or (b.op is Ops.MSELECT and b.src[0].op is Ops.PARAM) else b.buffer
           for a in get_call_arg_uops(call)] for lane in range(len(devices))]

def _wait_ins(ctx:BatchCtx, bufs_by_lane:list[list[Any]], write, devices:tuple[str, ...], queue:str, tag:int) -> list[UOp]:
  deps:list[Dep] = []
  for lane, bufs in enumerate(bufs_by_lane):
    written = write if write is not None else list(range(len(bufs)))
    deps += [Dep(d, q, t, lane) for d, q, t in ctx.tracker.access_resources(bufs, written, (devices[lane], queue, tag)) if t < tag]

  # same-queue submits are fifo-ordered, no wait needed
  if devices[0].split(":")[0] in {"AMD", "QCOM", "CPU"} or queue.startswith("COPY"):
    deps = [d for d in deps if (d.dev, d.queue) != (devices[d.lane], queue)]
  latest = {(d.dev, d.queue, d.lane): d for d in sorted(deps, key=lambda d: d.tag)}

  # keep only the latest signal
  rows:dict[tuple[str, int], dict[int, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
  for d in latest.values(): rows[(d.queue, d.tag)][d.lane].append(d.dev)
  waits = []
  for (dqueue, dtag), by_lane in rows.items():
    for ds in itertools.zip_longest(*(by_lane[lane] for lane in range(len(devices)))):
      sig = UOp.mstack(*[make_buf(d, tag="sentinel_signal") if dd is None else make_buf(dd, ctx.slots[dqueue]) for dd, d in zip(ds, devices)])
      waits.append(UOp(Ops.INS, arg="wait", src=(sig, UOp.const(dtag + 1, dtypes.uint64))))
  ctx.signal_tags |= {t for _, t in rows}
  return waits

def _merge_submits(calls:list[UOp]) -> UOp: # TODO: simplify?
  if len(calls) == 1: return calls[0]
  devs, queue = get_submit(calls[0]).src[0].arg
  body = make_submit(*[cmd for c in calls for cmd in get_submit(c).src[0].src], devs=devs, queue=queue).sink()
  return make_call(f"submit {queue} ({len(calls)})", body, replace(calls[0].arg.aux,
    estimates=sum((c.arg.aux.estimates for c in calls), start=Estimates()).simplify()))

def _merge_queues(submits:list[UOp]) -> list[UOp]:
  merged:list[UOp] = []
  opened:dict[tuple[tuple[str, ...], str], list[UOp]] = {} # (devs, queue) -> hcq calls in submit order
  limits:dict[tuple[tuple[str, ...], str], int] = collections.defaultdict(lambda: JIT_BATCH_SIZE.value)
  for call in submits:
    devs, queue = key = get_submit(call).src[0].arg
    if (group:=opened.pop(key, None)) is None:
      # first submit on this queue: close open groups on the same queue with shared devices, so submit order is kept
      for k in [k for k in opened if k[1] == queue and set(k[0]) & set(devs)]: merged.append(_merge_submits(opened.pop(k)))
      group = []
    elif limits[key] and len(group) >= limits[key]: merged, group, limits[key] = merged + [_merge_submits(group)], [], limits[key] * 2
    opened[key] = group + [call]
  return merged + [_merge_submits(g) for g in opened.values()]

def _make_finalizers(ctx:BatchCtx) -> tuple[list[UOp], list[UOp], list[UOp]]:
  # collect all buffers which belong to devices
  dev_bufs:dict[str, dict[int, Any]] = collections.defaultdict(dict)
  for call, devices in ctx.batch:
    for b in itertools.chain.from_iterable(_get_call_bufs_by_lane(call, devices)):
      for bd in to_tuple(b.device): dev_bufs[bd][id(b)] = b

  n, fences, resets, fins = len(ctx.batch_info), [], [], []
  for _, devgroup in itertools.groupby(sorted(dev_bufs), key=lambda d: d.split(":")[0]):
    sched_epoch = make_buf(devs:=tuple(devgroup), next(UOp.unique_num), tag="epoch")
    tl_signal, tl_value = make_buf(devs, tag="timeline_signal"), make_buf(devs, tag="timeline_value")

    # fence: spin until the device timeline reaches this schedule's previous epoch
    done = tl_signal.after(loop:=UOp.loop(0)).index(0).load()
    fences.append(make_call("hcq_fence", UOp.sink(done.end(loop, done < sched_epoch.index(0).load())), HCQInfo(devs)))

    # reset: queues of other groups wait on this group's signals, zero them only after every group reached its epoch
    qs = dedup([qn for bdevs, qn in ctx.batch_info if set(bdevs) & set(devs)])
    rst:tuple[UOp, ...] = ()
    for q in qs: rst += (make_buf(devs, ctx.slots[q]).after(*rst[-1:]).index(0).store(0),)
    if rst: resets.append(make_call("hcq_reset", UOp.sink(*rst), HCQInfo(devs)))

    # finalizer: bump the host timeline and remember this schedule's epoch for the next fence
    waits = _wait_ins(ctx, [list(dev_bufs[d].values()) for d in devs], None, devs, "COMPUTE:0", n)
    fin_submit = make_submit(*waits, UOp(Ops.INS, arg="store", src=(tl_signal, tl_value.index(0))), devs=devs, queue="COMPUTE:0")
    epoch = (epoch_slot:=tl_value.after(fin_submit).index(0)).load()
    fins.append(make_call("hcq_finalizer", UOp.sink(epoch_slot.store(epoch + 1), sched_epoch.after(fin_submit).index(0).store(epoch)), HCQInfo(devs)))
  return fences, resets, fins

def _emit_submits(ctx:BatchCtx, call_waits:list[list[UOp]]) -> tuple[list[UOp], list[tuple]]:
  # one submit per call: timeline sync on first queue use, timestamps, the call, and a signal if someone waits on it
  src, kerns, seen_queues = [], [], set()
  for tag, ((call, _), (devices, queue), q) in enumerate(zip(ctx.batch, ctx.batch_info, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if (devices, queue) not in seen_queues:
      seen_queues.add((devices, queue))
      epoch = make_buf(devices, tag="timeline_value").index(0) - 1
      q = [UOp(Ops.INS, arg="barrier", src=()), UOp(Ops.INS, arg="wait", src=(make_buf(devices, tag="timeline_signal"), epoch))] + q

    # and make hcq call
    name, info = get_call_name(call, get_call_arg_uops(call)), HCQInfo(devices, estimate_uop(call))
    ts_ids = [next(UOp.unique_num) for _ in range(2)] if ctx.profile else []
    kerns.append((devices, name, info.estimates, tuple(ts_ids), make_call(name, call.src[0], info).key))

    ts_ins = [UOp(Ops.INS, arg="timestamp", src=(make_buf(devices, s),)) for s in ts_ids]
    q += ts_ins[:1] + [call.replace(arg=replace(call.arg, aux=info))] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags: q += [UOp(Ops.INS, arg="store", src=(make_buf(devices, ctx.slots[queue]), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_call(f"submit {name}", make_submit(*q, devs=devices, queue=queue).sink(), info))
  return src, kerns

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...]]], profile:bool) -> list[UOp]:
  ctx = BatchCtx(batch, [(devices, "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0") for call, devices in batch], profile)

  call_waits = [_wait_ins(ctx, _get_call_bufs_by_lane(call, devices), get_call_outs_ins(call)[0], devices, queue, tag)
                for tag, ((call, _), (devices, queue)) in enumerate(zip(ctx.batch, ctx.batch_info))]
  fences, resets, fins = _make_finalizers(ctx)
  submits, kerns = _emit_submits(ctx, call_waits)

  # append batch timestamps to finalizers
  fins = [f.replace(arg=replace(f.arg, aux=replace(a:=f.arg.aux, kernels=tuple(x for x in kerns if set(x[0]) & set(a.device))))) for f in fins]
  return fences + resets + _merge_queues(submits) + fins

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...]]] = []
  for call in l.src:
    if (devs:=_get_enqueue_devs(call)) is not None: batch.append((call, to_tuple(devs)))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 4.2. hcq lowering: ops to ir

def encode_host_call(call:UOp) -> UOp|None:
  if (pm:=getattr(Device[call.arg.aux.device[0]], "pm_host_lower", None)) is None: return None
  body = graph_rewrite(call.src[0], pm, name="lower host access", enter_calls=True)
  return None if body is call.src[0] else call.replace(src=(body, *call.src[1:]))

def encode_cmdbuf(submit:UOp, lin:UOp) -> UOp|None:
  if (pm:=Device.get_class(lin.arg[0][0]).pm_lower) is None: return None
  return graph_rewrite(submit, pm, name=f"encode {lin.arg[0]}", enter_calls=True)
pm_encode_cmdbufs = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), encode_cmdbuf),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), encode_host_call)])

# *****************

def get_getaddrs(p:UOp) -> list[UOp]: return [u for u in p.toposort(gate=lambda u: u.op is not Ops.AFTER) if u.op is Ops.GETADDR]

def trim_link_patches(ctx:tuple[list[UOp], list[UOp]], a:UOp) -> UOp|None:
  links, kept = partition(a.src[1:], lambda p: p.tag == "link")
  ctx[0].extend(kept)

  # keep all patches from the link-time patches' subtrees in the C code
  afters = [u for u in UOp.sink(*links).toposort() if u.op is Ops.AFTER]
  ctx[1].extend(UOp.sink(*links).substitute({p: p.src[0] for p in afters}).src)
  return a.src[0].after(*kept, *[d for p in afters for d in p.src[1:]]) if links else None
pm_trim_link_patches = PatternMatcher([(UPat(Ops.AFTER, src=(UPat((Ops.PARAM, Ops.MSTACK)),), allow_any_len=True, name="a"), trim_link_patches)])

def _dnum(stride:int) -> UOp: return UOp.variable("_device_num", 0, stride - 1, dtypes.int, param=True) if stride > 1 else UOp.const(0, dtypes.int)

def make_addr_table(call:UOp, gaddrs:list[UOp], name:str, stride:int=1) -> tuple[UOp, dict[UOp, UOp], tuple[UOp, ...], dict[UOp, int]]:
  bare = {g: g.replace(src=(g.src[0].without_after,)) for g in gaddrs}

  # slot-major layout: slot i of lane j lives at i*stride+j, every lane reads through the same table base
  slots = {g:i*stride for i,g in enumerate(sorted(dedup(bare.values()), key=lambda g: g.key))}
  table = UOp.placeholder((len(slots)*stride,), dtypes.uint64, next(UOp.unique_num), device=call.arg.aux.device).rtag(name)

  reads = {g: table.after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ()).index(_dnum(stride) + slots[bare[g]]).load() for g in gaddrs}
  fills = (table.after(*make_patches(table, [(i*table.dtype.itemsize, addr) for addr, i in slots.items()])),) if slots and stride == 1 else ()
  return table, reads, fills, {g:slots[bare[g]] for g in gaddrs}

def make_gather_loop(patches:list[UOp], table:UOp, slots:dict[UOp, int], lt_patches:list[UOp], stride:int) -> dict[UOp, UOp]:
  (dst,), words = dedup(p.buf_uop for p in patches), [(unwrap_view(p.src[0].src[0])[1] + off.val*(val.dtype.itemsize//p.buf_uop.dtype.itemsize),
                                                       slots[val]) for p in patches for off,val in zip(p.src[0].src[1].src, p.src[1].src)]

  # build a runtime loop that writes every input address
  pairs = UOp.placeholder((2*len(words),), dtypes.uint32, next(UOp.unique_num), device=dst.device).rtag("systems")
  lt_patches.append(make_binary_patch(pairs, struct.pack(f'<{2*len(words)}I', *itertools.chain(*words))))
  r = UOp.range(len(words), next(UOp.unique_num), dtype=dtypes.int, src=(pairs, dst))
  off, slot = ((pairs.index(2*r+i).load() % bound).cast(dtypes.int) for i, bound in ((0, dst.max_numel()-1), (1, table.max_numel()-(stride-1))))
  # SHRINK(offset, length): a const length keeps the end bound from becoming an expression the program spec rejects
  patch = UOp(Ops.SHRINK, src=(dst, off, off.const_like(table.dtype.itemsize//dst.dtype.itemsize))).bitcast(table.dtype).index(0) \
    .store(table.index(slot + _dnum(stride)).load()).end(r)
  return {p: UOp(Ops.NOOP) for p in patches} | {patches[0]: patch}

def is_input_addr(g:UOp) -> bool: return any(x.op is Ops.PARAM and x.tag is None for x in unwrap_mstack(g.buf_uop))

def split_patches(call:UOp) -> UOp|None:
  rt_patches:list[UOp] = []
  lt_patches:list[UOp] = []
  body = graph_rewrite(call.src[0], pm_trim_link_patches, ctx=(rt_patches, lt_patches), name=f"trim link-time patches ({call.arg.name})")

  # split patches. addresses read in the body go through the tables too
  lanes = len(to_tuple(call.arg.aux.device))
  inputs, internals = partition(dedup(get_getaddrs(UOp.sink(body, *rt_patches))), is_input_addr)
  runtimes, systems = partition(internals, lambda g: any(x.tag in {"program", "kernargs", "cmdbuf"} for x in unwrap_mstack(g.buf_uop)))
  tables = [make_addr_table(call, gs, n, lanes if n == "inputs" else 1) for gs,n in ((inputs, "inputs"), (runtimes, "runtime"), (systems, "systems"))]
  reads, fills = {k:v for _,r,_,_ in tables for k,v in r.items()}, [f for t in tables[1:] for f in t[2]] # inputs table is filled by exec

  ipatches = [p for p in rt_patches if p.tag == "inputs" and all(v in tables[0][3] for v in p.src[1].src)] # only getaddrs go to the table
  gathers = make_gather_loop(ipatches, tables[0][0], tables[0][3], lt_patches, lanes) if ipatches else {}
  body = body.substitute({p:p.substitute(gathers | reads) for p in rt_patches} | reads, walk=True)

  lt_srcs = collections.defaultdict(list)
  for p in lt_patches: lt_srcs[p.buf_uop].append(p)

  bufs = [u for _, u in sorted(dedup([(i, g.src[0].without_after) for g, i in tables[0][3].items()]))]
  aux = replace(call.arg.aux, input_addrs=tuple((d, _lane(u, j)) for u in bufs for j,d in enumerate(call.arg.aux.device))) if inputs else call.arg.aux
  return call.replace(src=(body, *call.src[1:], *[b.after(*ps) for b,ps in lt_srcs.items()], *fills), arg=replace(call.arg, aux=aux))
pm_split_patches = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), split_patches)])

# *****************

def _rank_ranges(uops:Iterable[UOp]) -> dict[UOp, UOp]:
  return {r: r.replace(arg=(i,)+r.arg[1:]) for i,r in enumerate(sorted([u for u in uops if u.op is Ops.RANGE], key=lambda r: r.arg))}

def replace_params(call:UOp) -> UOp|None:
  body, variables, param_ops = call.src[0], call.src[0].variables(), {Ops.PARAM, Ops.MSTACK}
  tops = body.toposort(gate=lambda u: u.op not in param_ops)
  args = dedup([s for u in tops for s in u.src if s.op in param_ops and s not in variables])

  patched, refhold = partition(call.src[1:], lambda x: x.src[0] in args)
  by_root = {p.src[0]: p for p in patched}
  c_args = [by_root.get(a, a) for a in args]

  # keep buffers whose addresses become link-time constants alive and mapped
  held = args + [r.without_after for r in refhold]
  addrs = dedup([g.src[0].without_after for g in call.toposort() if g.op is Ops.GETADDR])
  refhold += [a for a in addrs if a not in held and all(b.op is not Ops.PARAM or b.tag is not None for b in unwrap_mstack(a))]

  sub = {(b:=u.without_after): UOp.param(i, u.dtype, shape=b.shape, device=HCQ_RUNTIME_DEV.value, volatile=b.op is Ops.PARAM and b.arg.volatile)
         for i,u in enumerate(c_args)} | {v: v.replace(arg=replace(v.arg, slot=-1)) for v in variables if v.op is Ops.PARAM} | _rank_ranges(tops)
  info = replace(call.arg.aux, inputs=next((i for i,u in enumerate(c_args + refhold) if u.without_after.tag == "inputs"), None))
  prg_sink = body.src[0].substitute(sub).replace(arg=KernelInfo("hcq_submit"), tag=1)
  return call.replace(src=(body.replace(src=(prg_sink,)), *c_args, *refhold), arg=replace(call.arg, aux=info))
pm_replace_params = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq", src=(UPat(Ops.SINK),)),), name="call", allow_any_len=True), replace_params)])

# *****************

def resolve_getaddr_view(bv:UOp, g:UOp) -> UOp:
  base = bv.src[0].after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ())
  addr = UOp(Ops.GETADDR, src=(base,), arg=g.arg)
  return addr if bv.op is Ops.BITCAST else addr + UOp.const(bv.src[1].val * bv.dtype.itemsize, dtypes.uint64)

pm_early_simplify = PatternMatcher([
  (UPat(Ops.GETADDR, src=(UPat((Ops.SHRINK, Ops.BITCAST), name="bv").or_after(),), name="g"), resolve_getaddr_view),
  (UPat(Ops.SHRINK, src=(UPat(Ops.SHRINK, name="bv"), UPat(), UPat()), name="x"),
   lambda bv,x: bv.src[0].shrink(((start:=bv.src[1]+x.src[1], start+x.src[2]),))),
])

# *****************
# 5.3. pack placeholders buffers

def pack_hcq_placeholders(call:UOp) -> UOp|None:
  bufs = [b for b in call.src[0].toposort() if b.op is Ops.PARAM and b.tag in {"scratch", "kernargs"}]
  offs:dict[UOp, int] = {}
  sizes:dict[Any, int] = {}
  for b in bufs:
    if b.tag == "scratch": sizes[b.tag] = max(sizes.get(b.tag, 0), b.max_numel())
    else:
      offs[b] = round_up(sizes.get(b.tag, 0), 128 // b.dtype.itemsize)
      sizes[b.tag] = offs[b] + b.max_numel()
  counts = collections.Counter(b.tag for b in bufs)
  bases = {b.tag:UOp.placeholder((sizes[b.tag],), b.dtype, next(UOp.unique_num), device=b.device).rtag(b.tag) for b in bufs if counts[b.tag] > 1}
  subs = {b:bases[b.tag][(off:=offs.get(b, 0)):off+b.max_numel()] for b in bufs if b.tag in bases}
  return call.replace(src=(call.src[0].substitute(subs, walk=True), *call.src[1:])) if subs else None
pm_pack_placeholders = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), pack_hcq_placeholders)])

# *****************
# 9. merge submitters

def _lane_arg(a:UOp, lane:int, table:UOp) -> UOp: return table if a.tag == "inputs" else a.mselect(lane) if len(to_tuple(a.device)) > 1 else a

def merge_batch(batch:list[UOp]) -> UOp:
  tables = UOp.variable("hcq_inputs_ptr", 0, 2**64-1, dtypes.uint64, param=True)
  offs = itertools.accumulate((8 * len(c.arg.aux.input_addrs) for c in batch), initial=0) # every call owns the next table of the region
  cmds = [c.src[0].src[0].call(*[_lane_arg(a.without_after, j, tables + off) for a in c.src[1:]], UOp.variable("_device_num", 0, 1 << 30).bind(j))
          for c, off in zip(batch, offs) for j in range(len(c.arg.aux.device))]

  info = HCQInfo((HCQ_RUNTIME_DEV.value,), sum((c.arg.aux.estimates for c in batch), start=Estimates()).simplify(),
                 input_addrs=tuple(x for c in batch for x in c.arg.aux.input_addrs), kernels=tuple(k for c in batch for k in c.arg.aux.kernels))
  body = UOp.custom_function("hcq", make_submit(*cmds, devs=HCQ_RUNTIME_DEV.value, queue="SUBMIT:0").sink())
  return body.call(*[s for c in batch for s in c.src[1:] if s.without_after.tag != "inputs"], name=f"hcq_submitter ({len(batch)})", aux=info)

def merge_submitters(linear:UOp) -> UOp:
  batches = [(k, list(g)) for k, g in itertools.groupby(linear.src, key=lambda c: isinstance(c.arg.aux, HCQInfo))]
  return linear.replace(src=tuple(c for is_hcq, b in batches for c in ([merge_batch(b)] if is_hcq else b)))

# *****************
# hcq schedule

hcq_compile_cache:dict[tuple[bytes, bool], UOp] = {}

def hcq_lower(linear:UOp, pm_encode:PatternMatcher) -> UOp:
  # lowering to hcq ir
  linear = graph_rewrite(linear, pm_encode, walk=True, name="encode and pack", enter_calls=True)

  # patches and runtime uops
  linear = graph_rewrite(linear, pm_early_simplify+symbolic, bottom_up=False, name="simplify patches", enter_calls=True)
  linear = graph_rewrite(linear, pm_split_patches, walk=True, name="split patches")

  # and compile it
  return lower_and_compile(graph_rewrite(linear, pm_replace_params, walk=True, name="replace params"))

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear.key, profile)))) is None:
    # prep
    linear = linear.substitute(back_map:={s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}, walk=True)
    linear = graph_rewrite(linear, pm_insert_copy_staging+pm_flatten_linear, name="insert copy staging")

    # schedule on real buffers
    linear = sched_batches(linear, profile).substitute({s:p for p,s in back_map.items()}, walk=True, enter_calls=True)

    # lower to hcq programs, then pack the programs of every batch into one C submitter (needs a C runtime device for the program addresses)
    linear = hcq_lower(linear, pm_encode_cmdbufs+pm_pack_placeholders)
    final_linear = hcq_compile_cache[cache_key] = hcq_lower(merge_submitters(linear), pm_encode_cmdbufs) if HCQ_RUNTIME_DEV.value == "CPU" else linear

  return final_linear

# *****************
# 6. bufferize placeholders: replace placeholders with real buffers.

def bufferize_buf(ctx:bool, buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.mstack(*(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=(dv, ctx)), HCQ_RUNTIME_DEV.value)
                      for dev in to_tuple(buf.device)))
pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="buf"), bufferize_buf)])

# *****************
# 7. resolve patches

def push_stack(op, s): return UOp(Ops.STACK,
  src=tuple(op.replace(src=tuple(x if y is s else y for y in op.src)) for x in s.src))

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in (m.bufs if isinstance(m:=buf.buffer, MultiBuffer) else (m,)):
    b.ensure_allocated()._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(view:UOp, off:UOp, val:UOp) -> UOp:
  buf, start = unwrap_view(view)
  for off,val in zip(off.src, val.src):
    for b,v in zip((bs:=mb.bufs if isinstance((mb:=buf.buffer), MultiBuffer) else (mb,)), val.src if val.op is Ops.STACK else (val,)*len(bs)):
      data = struct.pack(f'<{v.dtype.fmt}', truncate[v.dtype](v.val))
      bo = start*buf.dtype.itemsize + off.val*val.dtype.itemsize
      b.ensure_allocated()._buf.cpu_view().view(fmt='B')[bo:bo+len(data)] = data
  return UOp(Ops.NOOP)

def resolve_getaddr(buf:UOp, g:UOp) -> UOp:
  assert buf.op in (Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), f"{buf.op}"

  devs, b = g.arg, buf.buffer
  bufs = tuple(cast(Buffer, x.buffer) for x in buf.src) if buf.op is Ops.MSTACK else tuple(b.bufs if isinstance(b, MultiBuffer) else (b,)*len(devs))
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(x.get_buf(d).va_addr, dtypes.uint64) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, src=addrs)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU, src=[UPat(Ops.STACK, name="s"), UPat.any(UPat(Ops.CONST), UPat(Ops.CAST, src=(UPat(Ops.CONST),)))], name="op"), push_stack),
  (UPat(Ops.CAST, src=(UPat(Ops.STACK, name="s"),), name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(name="buf"),), name="g"), resolve_getaddr),

  # folders
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat((Ops.BITCAST, Ops.SHRINK, Ops.BUFFER, Ops.MSTACK), name="view")
    .index(UPat(Ops.STACK, name="off")).store(UPat(Ops.STACK, name="val")), fold_const_store),
])

pm_assert_no_afters = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: panic(RuntimeError, f"AFTER left at hcq_link: {a.src[0].op}"))])

def link_buf_key(a:UOp): return a.key, to_tuple(a.device)
link_buf_cache:dict[tuple[bytes, tuple[str, ...]], UOp] = {}
link_linear_cache:dict[bytes, UOp] = {}

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear_key:=linear.key)) is not None: return linked

  bufs = {(j,i):a for j,c in enumerate(linear.src) for i,a in enumerate(c.src[1:], 1)
          if a.op is Ops.AFTER and unwrap_mstack(a.src[0])[0].tag in HCQ_CACHE_TAGS}
  linear = linear.substitute({x:link_buf_cache[k] for a in bufs.values() if (k:=link_buf_key(a)) in link_buf_cache for x in (a, a.src[0])}, walk=True)
  linear = graph_rewrite(linear, pm_resolve_patches+symbolic+pm_assert_no_afters, bpm=pm_bufferize, ctx=cache, bottom_up=False,
                         name="resolve patches")
  for (j,i),a in bufs.items(): link_buf_cache.setdefault(link_buf_key(a), linear.src[j].src[i])
  if cache: link_linear_cache[linear_key] = linear
  return linear

# *****************
# Device classes

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0
  wait_timeout_ms: float = 30000.0
  rt_nbytes: int = 64 << 20 # scratch that single-run placeholders are carved out of

  def __init__(self, device:str, allocator:HCQAllocator, compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.can_recover = can_recover

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].signal("timeline")),
      (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].signal("value", 1, device="CPU")),
      (UPat(Ops.PARAM, tag="epoch", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot, device="CPU")),
      (UPat(Ops.PARAM, tag="signal", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if b.tag is None else ctx[0].new_buffer(b, cache=ctx[1]))
    ])

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.rt_allocator = BumpAllocator(self.rt_nbytes)
    self.prof_ents:dict[int, ProfileGraphEntry] = {}

  def collect_prof(self):
    if PROFILE:
      es = list(self.prof_ents.values())
      sigs = [self.signal(i)._buf.cpu_view().view(fmt='Q')[0]/decimal.Decimal(self.timestamp_divider) for e in es for i in (e.st_id, e.en_id)]
      Compiled.profile_events.append(ProfileGraphEvent([replace(e, st_id=2*i, en_id=2*i+1) for i,e in enumerate(es)], [], sigs))
    self.prof_ents.clear()

  def _at_profile_finalize(self):
    from tinygrad.tensor import Tensor
    tdiffs = []
    for _ in range(5):
      with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
      if not (ents:=list(self.prof_ents.values())): return
      self.prof_ents.clear()
      st = perf_counter_us()
      self.synchronize()
      gpu = max(self.signal(e.en_id)._buf.cpu_view().view(fmt='Q')[0] for e in ents)/decimal.Decimal(self.timestamp_divider)
      tdiffs.append((st+perf_counter_us())/2 - gpu)
    Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), self.device_props()))

  @functools.cache
  def rt_buffer(self, uncached:bool=True) -> Buffer:
    return Buffer(self.device, self.rt_allocator.size, dtypes.uint8, options=BufferSpec(uncached=uncached, cpu_access=True), preallocate=True)

  def new_buffer(self, b:UOp, cache:bool) -> Buffer:
    if cache or b.tag in HCQ_CACHE_TAGS:
      return Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(uncached=b.tag not in ("program","kernargs"), cpu_access=True,nolru=True))
    return self.rt_buffer(uncached=b.tag!="kernargs").view(b.max_numel(), b.dtype,
      self.rt_allocator.alloc(b.max_numel() * b.dtype.itemsize, alignment=128))

  @functools.cache
  def signal(self, name:str|int, init_value:int=0, device:str|None=None) -> Buffer:
    buf = Buffer(device or self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf._buf.cpu_view().view(fmt='Q')[0] = init_value
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done = time.perf_counter(), sig[0]
    while done < value:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif time.perf_counter() - st > (timeout or self.wait_timeout_ms) / 1000: self.on_device_hang()

  def synchronize(self, timeout:int|None=None):
    if HCQ_RUNTIME_DEV.value != self.device: Device[HCQ_RUNTIME_DEV.value].synchronize()

    sig = self.signal("timeline")._buf.cpu_view().view(fmt='Q')
    tl = self.signal("value", 1, device="CPU")._buf.cpu_view().view(fmt='Q')
    self._wait_signal(sig, tl[0] - 1, timeout)
    if self.prof_ents: self.collect_prof()

  def on_device_hang(self): raise RuntimeError(f"{self.device} hang detected")

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    super().finalize()

@dataclass
class HCQ2Buffer:
  va_addr:sint
  meta:Any=None
  view:MMIOInterface|None=None

  def offset(self, offset:int, size:int) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, meta=self.meta, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _as_buffer(self, buf:HCQBuffer) -> memoryview:
    return unwrap(buf.view).mv

  def _map(self, buf:HCQBuffer) -> HCQBuffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  def _do_unmap(self, mb): self.dev.iface.free(mb)

  @suppress_finalizing
  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    self.dev.synchronize()
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self._do_unmap(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer: return buf.offset(offset=offset, size=size)
