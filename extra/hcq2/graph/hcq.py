from __future__ import annotations
import time
from typing import cast
from tinygrad.device import Buffer, BufferSpec, Compiled, Device, MultiBuffer
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import get_call_outs_ins, get_runtime
from tinygrad.helpers import round_up, ceildiv
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite
from extra.hcq2.hcq2 import HCQ2Compiled, hcq_realize
# from extra.hcq2.hcq2 import pm_split_into_queues, pm_add_barriers, pm_add_signals
# from extra.hcq2.hcq2 import pm_bufferize, pm_lift_patches_to_cmdbuf, pm_resolve_patches, pm_parametrize_host_buffers
# from extra.hcq2.hcq2 import pm_add_timeline_inc, pm_callify, pm_calc_kernargs_sizes

# **************** insert deps ****************

# def insert_deps(ctx:HCQ2Graph, linear:UOp) -> UOp:
#   src = []
#   for j, call in enumerate(linear.src):
#     call = call.replace(tag=j)
#     _, _, bufs, _ = ctx.calls[j]
#     outs, ins = get_call_outs_ins(call)
#     deps = ctx._access_resources([bufs[i] for i in outs + ins], list(range(len(outs))), call)
#     src.append(UOp(Ops.AFTER, call.dtype, (call, *deps), tag=call.tag))
#   return linear.replace(src=tuple(src))
# pm_insert_deps = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), insert_deps)])

# pm_replace_params = PatternMatcher([
#   (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.input_addrs_uop.index(UOp.const(dtypes.int, p.arg))),
#   (UPat(Ops.SLICE, src=(UPat(Ops.INDEX, name="addr"), UPat(Ops.CONST, dtype=dtypes.weakint, name="off")), name="bv"),
#     lambda ctx, bv, addr, off: addr.cast(dtypes.uint64) + UOp.const(dtypes.uint64, off.arg * ctx.input_uops[addr.src[1].arg].dtype.itemsize)),
# ])

# # **************** graph-only passes ****************

# def alloc_queue_sig(ctx:HCQ2Graph, q:UOp) -> None:
#   if q.arg in ctx.queue_sigs: return None
#   dev = q.arg[0][0]  # TODO: multi device
#   buf = Buffer(dev, 0x100, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
#   ctx.queue_sig_bufs.append(buf)
#   ctx.queue_sigs[q.arg] = UOp.from_buffer(buf, dev)
#   return None
# pm_alloc_queue_sigs = PatternMatcher([(UPat(Ops.LINEAR, src=UPat({Ops.PROGRAM, Ops.COPY}), name="q"), alloc_queue_sig)])

# def lower_queue_deps(ctx:HCQ2Graph, after:UOp) -> UOp:
#   wrapper, deps, call_idx = after.src[0], after.src[1:], after.tag
#   def store(q_arg, v): return ctx.queue_sigs[q_arg].store(UOp.const(dtypes.uint32, v))
#   waits = tuple(UOp(Ops.WAIT, dtypes.void, (ctx.queue_sigs[dep.src[0].arg], UOp.const(dtypes.uint32, dep.tag),
#                                             store(dep.src[0].arg, dep.tag))) for dep in deps)
#   return wrapper.replace(src=tuple(q.replace(src=(*waits, *q.src, store(q.arg, call_idx))) for q in wrapper.src))
# pm_lower_queue_deps = PatternMatcher([(UPat(Ops.AFTER, src=UPat(Ops.LINEAR), name="after"), lower_queue_deps)])

# def optimize_queue_deps(ctx:HCQ2Graph, queue:UOp) -> UOp|None:
#   src, seen, pending, queue_sig = [], {}, {}, ctx.queue_sigs[queue.arg]
#   for x in queue.src:
#     if x.op is Ops.WAIT:
#       sig, val = x.src[0], x.src[1]
#       if sig is queue_sig or seen.get(sig, -1) >= val.arg: continue
#       if (old:=pending.get(sig)) is None or old.src[1].arg < val.arg: pending[sig] = x
#       continue
#     for wait in pending.values():
#       src.append(wait)
#       seen[wait.src[0]] = wait.src[1].arg
#     pending.clear()
#     src.append(x)
#   src += pending.values()
#   return queue.replace(src=tuple(src)) if tuple(src) != queue.src else None
# pm_optimize_queue_deps = PatternMatcher([
#   (UPat(Ops.LINEAR, src=UPat({Ops.BARRIER, Ops.WAIT, Ops.STORE, Ops.PROGRAM, Ops.COPY}), name="queue"), optimize_queue_deps),
# ])

# def drop_dead_stores(ctx:HCQ2Graph, outer:UOp) -> UOp:
#   live = {u.src[2] for u in outer.toposort() if u.op is Ops.WAIT}
#   return outer.replace(src=tuple(q.replace(src=tuple(x for x in q.src if x.op is not Ops.STORE or x in live)) for q in outer.src))
# pm_drop_dead_stores = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"), drop_dead_stores)])

# def add_queue_sig_resets(ctx:HCQ2Graph, x:UOp, cmdbuf:UOp) -> UOp|None:
#   if not ctx.queue_sig_bufs or cmdbuf.tag not in ("compute", "copy"): return None
#   resets = tuple((b:=UOp.from_buffer(sig)).index(UOp.const(dtypes.int, 0), dtype=b.dtype.ptr())
#                  .cast(dtypes.uint64.ptr()).store(UOp.const(dtypes.uint64, 0)) for sig in ctx.queue_sig_bufs)
#   return x.replace(src=x.src + resets)
# pm_add_queue_sig_resets = PatternMatcher([(UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="cmdbuf"),), allow_any_len=True, name="x"),
#                                            add_queue_sig_resets)])

# **************** Graph ****************

class HCQ2Graph(GraphRunner):
  def __init__(self, linear:UOp, input_uops:tuple[UOp, ...]=()):
    super().__init__(linear, input_uops)
    self.linear = hcq_realize(linear)

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None:
    # addrs = self.input_addrs.as_memoryview(force_zero_copy=True).cast('Q')
    # for i, u in enumerate(input_uops):
    #   buf = next(b for b in u.buffer.bufs if b.device == self.dev.device) if isinstance(u.buffer, MultiBuffer) else u.buffer
    #   addrs[i] = buf._buf.va_addr
    # self.host_rt(*[self.hcq_ctx.inputs[i].get_buf("CPU") for i in self.host_globals], vals=self.host_call.src[0].arg.vals(var_vals), wait=True)
    # if wait:
    #   st = time.perf_counter()
    #   self.dev.synchronize()
    #   return time.perf_counter() - st
    return None

  @staticmethod
  def supports_uop(batch_devs:list[Compiled], new_call:UOp) -> bool:
    all_devs = GraphRunner._all_devs(batch_devs, new_call)
    return new_call.src[0].op in (Ops.PROGRAM, Ops.COPY) and len(all_devs) == 1 and isinstance(all_devs[0], HCQ2Compiled)
