from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any
import struct, functools, time, collections, itertools
from dataclasses import replace, dataclass
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, dedup, pluralize
from tinygrad.helpers import to_tuple, round_up, partition, data64_le, panic, ContextVar
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites, GroupOp
from tinygrad.uop.symbolic import symbolic
from tinygrad.dtype import dtypes, truncate
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop, pm_flatten_linear
from tinygrad.engine.jit import DepsTracker

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

# *****************
# 0. helpers

HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "CPU")

HCQ_DEVS = frozenset(("AMD",))
HCQ_P2P_DEVS = HCQ_DEVS | frozenset(("CPU",))
HCQ_CACHE_TAGS = frozenset(("program", "systems", "template"))

@dataclass(frozen=True)
class HCQInfo:
  name:str
  estimates:Estimates
  device:tuple[str, ...]
  queue:str

  input_idxs:tuple[int, ...] = () # indexes into input_uops used by this call
  inputs:int|None = None

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_after(uop):
  while uop.op is Ops.AFTER: uop = uop.src[0]
  return uop

def unwrap_mstack(u):
  return tuple(x for s in u.src for x in unwrap_mstack(s)) if u.op is Ops.MSTACK else (unwrap_mstack(u.src[0]) if u.op in {Ops.MSELECT, Ops.SLICE} else (u,))

def make_getaddr(u, device=None):
  if unwrap_after(u).op not in (Ops.BUFFER, Ops.SLICE, Ops.BINARY, Ops.MSTACK, Ops.MSELECT, Ops.PARAM): return u
  return UOp(Ops.GETADDR, dtypes.uint64, src=(u,), arg=device or to_tuple(u.device)[0])

def make_ins(op, *srcs):
  return UOp(Ops.INS, dtypes.void, tuple(UOp.const(dtypes.uint32, s) if isinstance(s, int) else s.cast(dtypes.uint32) for s in srcs), op)

def make_placeholder(devs, size:int, dtype, name=None, unique=True) -> UOp:
  return UOp.param(next(UOp.unique_num) if unique else 0, dtype, shape=(size,), device=devs).rtag(name or "temp")

def make_patch(buf:UOp, off:sint, val:UOp, dtype=None) -> UOp:
  return buf.index(UOp.const(dtypes.int, off // buf.dtype.itemsize)).store(val.simplify().cast(dtype or buf.dtype))

def make_binary_patch(buf:UOp, blob:bytes) -> UOp:
  data = UOp(Ops.BINARY, src=(), arg=blob).bitcast(buf.dtype)
  r = UOp.range(len(blob) // buf.dtype.itemsize, 0, dtype=dtypes.int, src=(buf, data))
  return buf.index(r).store(data.index(r).load()).end(r)

def make_cmdbuf(lin, devs):
  blob, patches = b'', []
  for s in (s for ins in lin.src for s in ins.src):
    if (ssimp:=s.simplify()).op is not Ops.CONST: patches.append((len(blob), ssimp))
    blob += struct.pack(f'<{ssimp.dtype.fmt}', ssimp.arg if ssimp.op is Ops.CONST else 0x0)
  cmdbuf = make_placeholder(devs, len(blob) // 4, dtypes.uint32, name="cmdbuf")
  return cmdbuf.after(make_binary_patch(cmdbuf, blob), *[make_patch(cmdbuf, off, s) for off, s in patches])

def make_mstack(uops): return uops[0] if len(uops) == 1 else UOp(Ops.MSTACK, uops[0].dtype, tuple(uops))

def make_signal(devs, queue=None, sentinel=False):
  return make_placeholder(devs, 1, dtypes.uint64, "sentinel_signal" if sentinel else (queue, "timeline_signal") if queue else "timeline_signal", unique=False)
def make_signal_value(devs, queue=None):
  return make_placeholder(devs, 1, dtypes.uint64, (queue, "timeline_value") if queue else "timeline_value", unique=False)

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  return UOp.custom_function("submit_cmdbuf", UOp(Ops.LINEAR, src=tuple(cmds), arg=(to_tuple(devs), queue)))
def get_submit(ast:UOp) -> UOp: return next(u for u in ast.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_cmdbuf")

# *****************
# 0.1. prep: replace buffers with params

def replace_call_buffers(ctx:list[UOp], call:UOp) -> UOp|None:
  ctx += [s for s in call.src[1:] if s not in ctx and s.op not in (Ops.PARAM, Ops.BIND)]
  return call.replace(src=call.src[:1] + tuple(s if s.op in (Ops.PARAM, Ops.BIND) else s.param_like(ctx.index(s)) for s in call.src[1:]))
pm_replace_buffers = PatternMatcher([(UPat(Ops.CALL, name="call"), replace_call_buffers)])

# *****************
# 1.1. prep: staging copies

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS) and not all_devices_in(b.device, HCQ_P2P_DEVS)

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  stage = UOp.new_buffer("CPU", src.max_numel() * src.dtype.itemsize, dtypes.uint8)
  return UOp(Ops.LINEAR, src=(src.copy_to_device("CPU").call(stage, src), stage.copy_to_device(dst.device).call(dst, stage)))
pm_insert_copy_staging = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy)])

# *****************
# 2.1. tag hcq calls

def tag_hcq_call(ctx:itertools.count, call:UOp) -> UOp:
  if (hcq_devs:=next((b.device for b in call.src[1:] if all_devices_in(b.device, HCQ_DEVS)), None)) is None: return call

  queue = "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0"
  info = HCQInfo(get_call_name(call, get_call_arg_uops(call)), estimate_uop(call), to_tuple(hcq_devs), queue)
  return call.replace(arg=replace(call.arg, aux=info)).rtag(next(ctx))
pm_tag_hcq_calls = PatternMatcher([(UPat(Ops.LINEAR, name="l"), lambda ctx, l: l.replace(src=tuple(tag_hcq_call(ctx, s) for s in l.src)))])

# *****************
# 2.2. deps tracking
# device.timeline_signal/value are the per-device schedule epoch. Before a schedule queue accesses memory owned by device N for the first time,
# it waits for device[N].timeline_signal >= device[N].timeline_value - 1. This orders the schedule after all prior schedules that touched device N.
#
# queue.timeline_signal/value are per-queue progress counters used only inside a schedule.
# Only the owner queue signals its queue.timeline_signal. Values are monotonic.
#
# At schedule end, one finalizer queue per touched device[N] waits for every active queue on device[N] to reach its schedule-local
# final queue.timeline value, then signals device[N].timeline_signal with the schedule's reserved device epoch. After that, buffers/transients
# for device N from this schedule are safe for the next schedule
#
# C programs reserve and bump timeline values, then patch command buffers with the concrete wait/signal values.

class HCQDepsTracker(DepsTracker):
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    return (buf.arg.slot, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

def make_deps(u:UOp, dep_lanes:list[tuple[UOp, int, int]], nlanes:int) -> UOp:
  deps:dict[UOp, list[int|None]] = collections.defaultdict(lambda: [None]*nlanes)
  for dep, dlane, lane in dep_lanes: deps[dep][lane] = dlane
  return u.after(*deps, arg=tuple(tuple(v) for v in deps.values()))

def sched_sync(ctx:DepsTracker, call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo): return None

  refs = get_call_arg_uops(call)
  outs, _ = get_call_outs_ins(call)
  devices, queue = call.arg.aux.device, call.arg.aux.queue

  dep_lanes:list[tuple[UOp, int, int]] = []
  for lane, d in enumerate(devices):
    lane_refs = [b if b.op is Ops.PARAM else mb.bufs[lane] if isinstance(mb:=b.buffer, MultiBuffer) else mb for b in refs]
    for dep, dlane in ctx.access_resources(lane_refs, outs, (call, lane)): dep_lanes.append((dep, dlane, lane))

  if devices[0].split(":")[0] in {"AMD", "QCOM"} or queue.startswith("COPY"):
    dep_lanes = [(dep, dlane, lane) for dep, dlane, lane in dep_lanes if (dep.arg.aux.device[dlane], dep.arg.aux.queue) != (devices[lane], queue)]

  # keep latest dep per (dep device, queue, cur lane)
  latest = {((dep.arg.aux.device[dlane], dep.arg.aux.queue), lane): (dep, dlane) for dep, dlane, lane in sorted(dep_lanes, key=lambda x: x[0].tag)}
  return make_deps(call, [(dep, dlane, lane) for (_, lane), (dep, dlane) in latest.items()], len(devices))
pm_sched_sync = PatternMatcher([(UPat(Ops.CALL, name="call"), sched_sync)])

# *****************
# 2.3. merge into queues

def _merged_hcq_call(calls:list[UOp]):
  info = replace(unwrap_after(calls[0]).arg.aux, estimates=sum((unwrap_after(c).arg.aux.estimates for c in calls), start=Estimates()))
  cmdbuf = make_submit(*calls, devs=info.device, queue=info.queue)
  return UOp.custom_function("hcq", cmdbuf.sink()).call(name="hcq", aux=info)

def merge_queues(linear:UOp) -> UOp:
  new_src:list[UOp] = []
  opened_qs:dict[tuple[tuple[str, ...], str], list[UOp]] = {} # (devs, queue) -> list of calls, kept in submit order

  for call in linear.src:
    if not isinstance(unwrap_after(call).arg.aux, HCQInfo):
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in list(opened_qs)] + [call]
      continue

    devices, queue = unwrap_after(call).arg.aux.device, unwrap_after(call).arg.aux.queue

    if (old:=opened_qs.pop((devices, queue), None)) is not None: new_rec = old + [call]
    else:
      # no such queue opened: close every open submit on this queue that shares a device, so submit order is kept
      closing = [k for k in opened_qs if k[1] == queue and set(k[0]) & set(devices)]
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in closing]
      new_rec = [call]
    opened_qs[(devices, queue)] = new_rec
  return linear.replace(src=tuple(new_src + [_merged_hcq_call(c) for c in opened_qs.values()]))
pm_merge_queues = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), merge_queues)])

# *****************
# 2.4. finalizer

def add_finalizer(ctx:itertools.count, linear:UOp) -> UOp:
  # collect by device type
  parts:dict[str, list[UOp]] = collections.defaultdict(list)
  for call in linear.src:
    if (c:=unwrap_after(call)).src[0].op is not Ops.CUSTOM_FUNCTION or c.src[0].arg != "hcq": continue
    parts[c.arg.aux.device[0].split(':')[0]].append(unwrap_after(get_submit(call).src[0].src[0]))

  nbump = next(ctx)
  finalizers = []
  for calls in parts.values():
    devs = tuple(dedup(d for call in calls for d in unwrap_after(call).arg.aux.device))
    zero = UOp.const(dtypes.int, 0)
    tl = make_signal_value(devs)

    # split each (multi-device) call into per-device deps, then store the device timeline value into the device signal after them
    dep_lanes = [(call, dlane, devs.index(d)) for call in calls for dlane, d in enumerate(unwrap_after(call).arg.aux.device)]
    store = make_deps(make_signal(devs).store(tl.index(zero)), dep_lanes, len(devs))
    submit = make_submit(store, devs=devs, queue="COMPUTE:0")

    upd = [(tl, 1)] + [(make_signal_value(devs, queue=qn), nbump) for qn in dedup([unwrap_after(call).arg.aux.queue for call in calls])]
    patches = [s.after(submit).index(zero, dtype=s.dtype).store(s.index(zero) + inc) for s, inc in upd]
    finalizers.append(UOp.custom_function("hcq", UOp.barrier(*patches).sink()).call(aux=HCQInfo("hcq finalizer", Estimates(), devs, "COMPUTE:0")))
  return linear.replace(src=linear.src + tuple(finalizers))
pm_add_finalizer = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), add_finalizer)])

# *****************
# 2.5. global sync

def add_global_sync(ctx:set[tuple[str, ...]], submit:UOp, q:UOp) -> UOp|None:
  if (devs:=q.arg[0]) in ctx: return None
  ctx.add(devs)

  # some devices from a command buffer might be used for the first time this schedule, so we wait for their global timeline epoch.
  wait = (make_signal(devs).index(zero:=UOp.const(dtypes.int, 0)).load() >= make_signal_value(devs).index(zero) - 1).wait()
  return submit.replace(src=(q.replace(src=(UOp(Ops.BARRIER, dtypes.void), wait, *q.src)),))
pm_add_global_sync = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_global_sync)])

# *****************
# 3.1. lower loads/stores

def add_loads(ctx:set[int], submit:UOp, q:UOp) -> UOp|None:
  cur_devs = q.arg[0]
  new_src:list[UOp] = []
  for s in q.src:
    if s.op is Ops.AFTER:
      for lanes, dep in zip(s.arg, s.src[1:]):
        devs, queue = dep.arg.aux.device, dep.arg.aux.queue
        ctx.add(dep.tag) # mark op to update signal.

        sig = make_mstack([make_signal(d if dl is None else devs[dl], queue=queue, sentinel=dl is None) for dl, d in zip(lanes, cur_devs)])
        val = make_mstack([make_signal_value(d if dl is None else devs[dl], queue=queue) for dl, d in zip(lanes, cur_devs)]).index(UOp.const(dtypes.int, 0))
        new_src.append((sig.index(UOp.const(dtypes.int, 0)).load() >= val + dep.tag).wait())
      s = s.src[0]
    new_src.append(s)
  return submit.replace(src=(q.replace(src=tuple(new_src)),))
pm_add_inner_loads = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_loads)])

def add_stores(ctx:set[int], submit:UOp, q:UOp) -> UOp|None:
  devs, queue = q.arg
  new_src:list[UOp] = []
  for op in q.src:
    new_src.append(op)
    if (sigval:=unwrap_after(op).tag) in ctx:
      new_src.append(make_signal(devs, queue=queue).store(make_signal_value(devs, queue=queue).index(UOp.const(dtypes.int, 0)) + sigval))
  return submit.replace(src=(q.replace(src=tuple(new_src)),))
pm_add_inner_stores = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_stores)])

# *****************
# 4.1. hcq lowering: programs

def encode_kernargs_clike(call:UOp, prg:UOp, devs:str|tuple[str, ...]) -> UOp:
  data, info = prg.arg
  buf = make_placeholder(devs, data.kernargs_alloc_size // 4, dtypes.uint32, name="kernargs")
  words = [w for gi in info.globals for w in data64_le(make_getaddr(get_call_arg_uops(call)[gi], devs))] + list(info.vars)
  return buf.after(*[make_patch(buf, i * 4, w) for i, w in enumerate(words)])

# *****************
# 4.2. hcq lowering: ops to ir

def encode_cmdbuf(submit:UOp, lin:UOp) -> UOp|None:
  if (pm:=Device.get_class(lin.arg[0][0]).pm_lower) is None: return None
  return graph_rewrite(submit, pm, name=f"encode {lin.arg[0]}", enter_calls=True)
pm_encode_cmdbufs = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), encode_cmdbuf)])

# *****************

def is_value_known_at_link(val:UOp) -> bool:
  runtime_reads = [u for u in val.toposort() if u.op in (Ops.LOAD, Ops.INDEX)]
  addressed_bufs = [b for g in val.toposort() if g.op is Ops.GETADDR for b in unwrap_mstack(g.buf_uop)]

  # addr of input params is not known at link time
  return not runtime_reads and all(b.op is not Ops.PARAM or b.tag is not None for b in addressed_bufs)

def is_link_patch(p:UOp, jit:bool) -> bool:
  store = p.src[0] if (is_binary_patch:=p.op is Ops.END) else p
  if not jit: return store.buf_uop.tag == "program"
  return is_binary_patch or (store.op is Ops.STORE and is_value_known_at_link(store.src[1]))

def trim_link_patches(ctx:tuple[bool, list[UOp]], a:UOp) -> UOp|None:
  links, kept = partition(a.src[1:], lambda p: is_link_patch(p, ctx[0]))

  # keep all patches from the link-time patches' subtrees in the C code
  afters = [u for u in UOp.sink(*links).toposort() if u.op is Ops.AFTER]
  ctx[1].extend(UOp.sink(*links).substitute({p: p.src[0] for p in afters}).src)
  return a.src[0].after(*kept, *[d for p in afters for d in p.src[1:]]) if links else None
pm_trim_link_patches = PatternMatcher([(UPat(Ops.AFTER, src=(UPat((Ops.PARAM, Ops.MSTACK)),), allow_any_len=True, name="a"), trim_link_patches)])

def split_patches(ctx:bool, call:UOp) -> UOp|None:
  body = graph_rewrite(call.src[0], pm_trim_link_patches, ctx=(ctx, lt_patches:=[]), name=f"trim link-time patches ({call.arg.aux.name})")

  lt_srcs = collections.defaultdict(list)
  for p in lt_patches: lt_srcs[p.buf_uop].append(p)
  return call.replace(src=(body, *call.src[1:], *[b.after(*ps) for b,ps in lt_srcs.items()]))
pm_split_patches = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), split_patches)])

# *****************

def make_addr_table(call:UOp, gaddrs:list[UOp], name:str) -> tuple[dict[UOp, UOp], tuple[UOp, ...]]:
  bare = {g: g.replace(src=(unwrap_after(g.src[0]),)) for g in gaddrs}

  order = sorted(dedup(bare.values()), key=lambda g: ((b:=unwrap_mstack(g.buf_uop)[0]).arg.slot, to_tuple(b.tag)))
  slots, table = {g:i for i,g in enumerate(order)}, make_placeholder(call.arg.aux.device, len(order), dtypes.uint64, name)

  reads = {g: table.after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ()).index(UOp.const(dtypes.int, slots[bare[g]])).load() for g in gaddrs}
  return reads, (table.after(*[make_patch(table, i * table.dtype.itemsize, addr) for addr, i in slots.items()]),) if slots else ()

def make_blob_bufs(call:UOp, blobs:list[UOp]) -> tuple[dict[UOp, UOp], tuple[UOp, ...]]:
  bufs = {b: make_placeholder(call.arg.aux.device, b.max_numel(), b.dtype, "template") for b in blobs}
  return bufs, tuple(buf.after(make_binary_patch(buf, b.src[0].arg)) for b,buf in bufs.items())

def rm_rt_uops(call:UOp) -> UOp|None:
  if not (rt_uops:=[u for u in call.src[0].toposort() if u.op is Ops.GETADDR or (u.op is Ops.BITCAST and u.src[0].op is Ops.BINARY)]): return None
  gaddrs, blobs = partition(rt_uops, lambda u: u.op is Ops.GETADDR)
  inputs, internals = partition(gaddrs, lambda g: all(x.op is Ops.PARAM and x.tag is None for x in unwrap_mstack(g.buf_uop)))
  runtimes, systems = partition(internals, lambda g: any(x.tag in {"program", "kernargs", "cmdbuf"} for x in unwrap_mstack(g.buf_uop)))

  # exec fills the inputs table with the input addresses every run, so it has no fill patches
  (reads, _), *tables = [make_addr_table(call, gs, n) for gs,n in ((inputs, "inputs"), (runtimes, "runtime"), (systems, "systems"))] + \
                        [make_blob_bufs(call, blobs)]
  reads, fills = reads | {k:v for r,_ in tables for k,v in r.items()}, [f for _,fs in tables for f in fs]
  return call.replace(src=(call.src[0].substitute(reads), *call.src[1:], *fills),
                      arg=replace(call.arg, aux=replace(call.arg.aux, input_idxs=tuple(sorted(dedup(g.buf_uop.arg.slot for g in inputs))))))
pm_rm_rt_uops = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), rm_rt_uops)])

# *****************

def replace_params(call:UOp) -> UOp|None:
  body, variables, param_ops = call.src[0], call.src[0].variables(), {Ops.PARAM, Ops.MSTACK}
  args = dedup([s for u in body.toposort(gate=lambda u: u.op not in param_ops) for s in u.src if s.op in param_ops and s not in variables])

  patched, refhold = partition(call.src[1:], lambda x: x.src[0] in args)
  by_root = {p.src[0]: p for p in patched}
  c_args = [by_root.get(a, a) for a in args]

  sub = {unwrap_after(u): UOp.param(i, u.dtype, shape=unwrap_after(u).shape, device=u.device) for i,u in enumerate(c_args)} | \
        {v: v.replace(arg=replace(v.arg, slot=-1)) for v in variables if v.op is Ops.PARAM}
  info = replace(call.arg.aux, inputs=next((i for i,u in enumerate(c_args) if u.tag == "inputs"), None))
  return call.replace(src=(body.substitute(sub), *c_args, *refhold), arg=replace(call.arg, aux=info)) # TODO: call.after(*refhold)?
pm_replace_params = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), replace_params)])

# *****************

def resolve_getaddr_slice(bv:UOp, g:UOp) -> UOp:
  base = bv.src[0].after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ())
  itemsize = bv.src[0].dtype.itemsize if unwrap_after(bv.src[0]).op in (Ops.BUFFER, Ops.SLICE, Ops.MSTACK, Ops.MSELECT) else bv.dtype.itemsize
  return UOp(Ops.GETADDR, dtypes.uint64, src=(base,), arg=g.arg) + UOp.const(dtypes.uint64, bv.src[1].arg * itemsize)

pm_early_simplify = PatternMatcher([
  (UPat(Ops.GETADDR, src=(UPat.any(sl:=UPat(Ops.SLICE, name="bv"), sl.after(allow_any_len=True)),), name="g"), resolve_getaddr_slice),
  (UPat(Ops.INDEX, src=(UPat(Ops.SLICE, name="bv"),), allow_any_len=True, name="x"),
   lambda bv,x: x.replace(src=(bv.src[0], x.src[1] + bv.src[1].cast(x.src[1].dtype), *x.src[2:]))),
])

# *****************
# 5.3. pack placeholders buffers

def pack_hcq_placeholders(call:UOp) -> UOp|None:
  bufs = [b for b in call.src[0].toposort() if b.op is Ops.PARAM and b.tag in {"scratch", "kernargs"}]
  offs, sizes = {}, {}
  for b in bufs:
    if b.tag == "scratch": sizes[b.tag] = max(sizes.get(b.tag, 0), b.max_numel())
    else:
      offs[b] = round_up(sizes.get(b.tag, 0), 128 // b.dtype.itemsize)
      sizes[b.tag] = offs[b] + b.max_numel()
  counts = collections.Counter(b.tag for b in bufs)
  bases = {b.tag:make_placeholder(b.device, sizes[b.tag], b.dtype, b.tag) for b in bufs if counts[b.tag] > 1}
  subs = {b:UOp(Ops.SLICE, b.dtype, (bases[b.tag], UOp.const(dtypes.index, offs.get(b, 0))), b.max_numel()) for b in bufs if b.tag in bases}
  return call.replace(src=(call.src[0].substitute(subs, walk=True), *call.src[1:])) if subs else None
pm_pack_placeholders = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), pack_hcq_placeholders)])

# *****************
# 8. callify hcq programs

pm_callify_hcq = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="hcq", src=(UPat(Ops.SINK),), name="cf"),
  lambda cf: cf.replace(src=(to_program(cf.src[0].replace(arg=KernelInfo("hcq_submit"), tag=1), Device[HCQ_RUNTIME_DEV.value].renderer),)))])

hcq_compile_cache:dict[tuple[bytes, bool], UOp] = {}

@track_rewrites(lambda linear,input_uops,jit,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None=None, jit=False) -> UOp:
  if input_uops is not None: linear = graph_rewrite(linear, pm_replace_buffers, ctx=input_uops, walk=True, enter_calls=True, name="replace buffer")

  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear.key, jit)))) is None:
    # schedule
    linear = linear.substitute(back_map:={s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}, walk=True)
    linear = graph_rewrite(linear, pm_insert_copy_staging + pm_flatten_linear, name="insert copy staging")
    linear = graph_rewrite(linear, pm_tag_hcq_calls, ctx=(enumerator:=itertools.count(0)), walk=True, name="tag hcq calls")
    linear = graph_rewrite(linear, pm_sched_sync, ctx=HCQDepsTracker(), walk=True, name="schedule sync")
    linear = linear.substitute({s: p for p, s in back_map.items()}, walk=True)
    linear = graph_rewrite(linear, pm_merge_queues, walk=True, name="merge queues")
    linear = graph_rewrite(linear, pm_add_finalizer, ctx=enumerator, walk=True, name="add finalizer")
    linear = graph_rewrite(linear, pm_add_global_sync, ctx=set(), walk=True, name="add global sync", enter_calls=True)

    # lowering to hcq ir
    linear = graph_rewrite(linear, pm_add_inner_loads, ctx=(waited:=set()), walk=True, name="add loads", enter_calls=True)
    linear = graph_rewrite(linear, pm_add_inner_stores, ctx=waited, walk=True, name="add stores", enter_calls=True)
    linear = graph_rewrite(linear, pm_encode_cmdbufs, walk=True, name="encode cmdbufs", enter_calls=True)
    linear = graph_rewrite(linear, pm_pack_placeholders, walk=True, name="pack placeholders")

    # pie
    linear = graph_rewrite(linear, pm_split_patches, ctx=jit, walk=True, name="split rt/lt patches")
    linear = graph_rewrite(linear, pm_early_simplify + symbolic, bottom_up=False, name="simplify packed placeholders", enter_calls=True)
    linear = graph_rewrite(linear, pm_rm_rt_uops, walk=True, name="replace rt uops")
    linear = graph_rewrite(linear, pm_replace_params, walk=True, name="replace with args")

    # and compile it
    final_linear = hcq_compile_cache[cache_key] = graph_rewrite(linear, pm_callify_hcq, name="callify hcq", enter_calls=True)

  return final_linear

# *****************
# 6. bufferize placeholders: replace placeholders with real buffers.

def bufferize_buf(ctx:bool, buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return make_mstack(tuple(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=(dv, ctx)), HCQ_RUNTIME_DEV.value) for dev in to_tuple(buf.device)))
pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="buf"), bufferize_buf)])

# *****************
# 7. resolve patches

def push_stack(op, s): return UOp(Ops.STACK, op.dtype.scalar(),
  tuple(op.replace(dtype=op.dtype.scalar(), src=tuple(x if y is s else y for y in op.src)) for x in s.src))

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in (m.bufs if isinstance(m:=buf.buffer, MultiBuffer) else (m,)): b.ensure_allocated()._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(buf:UOp, off:UOp, val:UOp) -> UOp:
  for b, v in zip((bs:=mb.bufs if isinstance((mb:=buf.buffer), MultiBuffer) else (mb,)), val.src if val.op is Ops.STACK else (val,)*len(bs)):
    data = struct.pack(f'<{v.dtype.fmt}', truncate[v.dtype](v.arg))
    b.ensure_allocated()._buf.cpu_view().view(offset=off.arg * buf.dtype.itemsize, size=len(data), fmt='B')[:] = data
  return UOp(Ops.NOOP)

def resolve_getaddr(buf:UOp, g:UOp) -> UOp:
  assert buf.op in (Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), f"{buf.op}"

  devs, b = g.arg, buf.buffer
  bufs = tuple(cast(Buffer, x.buffer) for x in buf.src) if buf.op is Ops.MSTACK else tuple(b.bufs if isinstance(b, MultiBuffer) else (b,)*len(devs))
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(dtypes.uint64, x.get_buf(d).va_addr) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, dtypes.uint64, addrs)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU, src=[UPat(Ops.STACK, name="s"), UPat(Ops.CONST)], name="op"), push_stack),
  (UPat(Ops.CAST, src=(UPat(Ops.STACK, name="s"),), name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(name="buf"),), name="g"), resolve_getaddr),

  # folders
  (UPat(name="buf").index(UPat(Ops.RANGE), allow_any_len=True)
    .store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast()).index(UPat(Ops.RANGE), allow_any_len=True).load())
    .end(UPat(Ops.RANGE)), fold_binary),
  (UPat({Ops.BUFFER, Ops.SLICE, Ops.MSTACK}, name="buf").index(UPat.cvar("off"))
    .store(UPat.any(UPat.cvar("val"), UPat(Ops.STACK, name="val"))), fold_const_store),
])

pm_assert_no_afters = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: panic(RuntimeError, f"AFTER left at hcq_link: {a.src[0].op}"))])

hcq_link_cache:dict[tuple[bytes, tuple[str, ...]], UOp] = {}

def link_cache_key(a:UOp): return a.key, to_tuple(a.device)
pm_link_cache = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: hcq_link_cache.get(link_cache_key(a)))])

@track_rewrites(lambda _,jit,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, jit=False) -> UOp:
  cacheable = {(j,i):a for j,c in enumerate(linear.src) for i,a in enumerate(c.src[1:], 1)
               if a.op is Ops.AFTER and unwrap_mstack(a.src[0])[0].tag in HCQ_CACHE_TAGS}
  hits = {a.src[0]:hcq_link_cache[key] for a in cacheable.values() if (key:=link_cache_key(a)) in hcq_link_cache}
  linear = graph_rewrite(linear, pm_link_cache, name="apply link cache").substitute(hits, walk=True)
  linear = graph_rewrite(linear, pm_bufferize, ctx=jit, bottom_up=True, walk=True, name="bufferize placeholders")
  linear = graph_rewrite(linear, pm_resolve_patches + symbolic, bottom_up=False, name="simplify patches")
  linear = graph_rewrite(linear, pm_assert_no_afters, name="assert no afters")
  for (j,i),a in cacheable.items(): hcq_link_cache.setdefault(link_cache_key(a), linear.src[j].src[i])
  return linear

# *****************
# Device classes

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0

  def __init__(self, device:str, allocator:HCQAllocator, compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].timeline_signal()),
      (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].timeline_value()),
      (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].timeline_signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if b.tag is None else ctx[0].new_buffer(b, jit=ctx[1]))
    ])

    super().__init__(device, allocator, compilers, lambda *a, **kw: None, None, arch=arch)

    self.rt_buffer = Buffer(self.device, 64 << 20, dtypes.uint8, options=BufferSpec(uncached=True, cpu_access=True))
    self.rt_allocator = BumpAllocator(64 << 20, wrap=False)

  def new_buffer(self, b:UOp, jit:bool) -> Buffer:
    if jit or b.tag in HCQ_CACHE_TAGS: return Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True))
    return self.rt_buffer.view(b.max_numel(), b.dtype, self.rt_allocator.alloc(b.max_numel() * b.dtype.itemsize, alignment=128))

  @functools.cache
  def timeline_signal(self, queue:str|None=None, init_value:int=0) -> Buffer:
    buf = Buffer(self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf._buf.cpu_view().mv.cast('Q')[0] = init_value
    return buf

  @functools.cache
  def timeline_value(self, queue:str|None=None, init_value:int=1) -> Buffer:
    buf = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    buf.as_memoryview(force_zero_copy=True).cast('Q')[0] = init_value
    return buf

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal()._buf.cpu_view().mv.cast('Q')
    tl = self.timeline_value().as_memoryview(force_zero_copy=True).cast('Q')
    st = time.perf_counter()
    while sig[0] < tl[0] - 1:
      if time.perf_counter() - st > (timeout or 3000) / 1000: self.on_device_hang()

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def _select_iface(self):
    assert (v:=getenv(k:=f'{type(self).__name__[:-6].upper()}_IFACE', "")) == "",  \
      f"{k}={v} is deprecated, use DEV={replace(DEV.target(type(self).__name__[:-6]), interface=v)} instead"
    assert hasattr(self, "ifaces"), "must have ifaces to select an iface"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fall back to mock ifaces
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # if the device has an interface, call device_fini to clean up resources
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

class HCQ2Buffer:
  def __init__(self, va_addr:sint, size:int, meta:Any=None, _base:HCQ2Buffer|None=None, view:MMIOInterface|None=None, owner:HCQ2Compiled|None=None):
    self.va_addr, self.size, self.meta, self._base, self.view, self.owner = va_addr, size, meta, _base, view, owner

  def offset(self, offset:int=0, size:int|None=None) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def base(self) -> HCQ2Buffer: return self._base or self

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _as_buffer(self, buf:HCQ2Buffer) -> memoryview:
    self.dev.synchronize()
    return buf.cpu_view().mv

  def _map(self, buf:HCQ2Buffer) -> HCQ2Buffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    self.dev.synchronize()
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self.dev.iface.free(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)
