#!/usr/bin/env python3
"""
peer to peer read bandwidth of compute kernels (a peer's buffer mapped on the reader, see seq_views) against the copy engine, to see what caps
the exchanges at ~20 GB/s a device: the link, the copy engine or the reader. GB/s of the reader(s):
  cu1:    device 0 reads MB from device 1 (a kernel)
  cuP:    device 0 reads MB from P peers at once (MB/P from each, one kernel), P in PEERS
  cuall:  every device reads MB from all its peers at once (the all_gather pattern)
  sdmaP:  device 0 gets MB from P peers at once by copies (the copy queues: AM_COPY_QUEUES)
  pushP/pushall: the same as writes (kernels on the sender storing into the peers' buffers)
  sdmaall: every device gets MB/(N-1) from every peer by copies at once (one-shot all_gather), sdmarot: in rotated order
run: DEV=AMD N=14 MB=256 PYTHONPATH=. python3 extra/glm_p2p_cu_bench.py
"""
import time
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.device import Buffer, BufferSpec, BufferStorage
import functools
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.helpers import getenv, Context

N, MB, K = getenv("N", 14), getenv("MB", 256), getenv("K", 10)
devs = [Device.canonicalize(f"{Device.DEFAULT}:{i}") for i in range(N)]
E = (MB << 20) // 4
src = [Tensor.full((E,), i + 1, dtype=dtypes.uint32, device=d).contiguous().realize() for i, d in enumerate(devs)]

def buf(t:Tensor) -> Buffer: return [u.buffer for u in t.uop.toposort() if u.op is Ops.BUFFER][0]
def alias(i:int, d:int) -> Tensor:
  # device i's src mapped on device d
  b = buf(src[i])
  with Context(ALLOW_DEVICE_USAGE=1):
    nb = Buffer(devs[d], b.nbytes, opaque=BufferStorage(va:=b.get_buf(devs[d])), options=BufferSpec(external_ptr=va)).ensure_allocated()
  return Tensor(UOp.from_buffer(nb, dtypes.uint32, devs[d]))

def sync():
  for d in devs: Device[d].synchronize()

def bench(name:str, fn, nbytes:int, readers:int=1):
  jit = TinyJit(fn)
  for _ in range(3): jit()
  sync()
  st = time.perf_counter()
  for _ in range(K): jit()
  sync()
  dt = (time.perf_counter() - st) / K
  print(f"{name:10s} {dt*1e3:8.3f} ms  {nbytes/dt/1e9:6.1f} GB/s a reader ({readers} reader(s), {nbytes/1e6:.0f} MB each)", flush=True)

def peers_read(d:int, peers:list[int]) -> Tensor:
  # one kernel on d reading E//len(peers) words of each peer at once
  m = E // len(peers)
  views = [alias(p, d)[:m] if p != d else src[d][:m] for p in peers]
  acc = views[0]
  for v in views[1:]: acc = acc ^ v
  return (acc ^ 7).contiguous()

dst = [Tensor.empty(E, dtype=dtypes.uint32, device=d).contiguous().realize() for d in devs]
def dalias(i:int, d:int) -> Tensor:
  # device i's dst mapped on device d
  b = buf(dst[i])
  with Context(ALLOW_DEVICE_USAGE=1):
    nb = Buffer(devs[d], b.nbytes, opaque=BufferStorage(va:=b.get_buf(devs[d])), options=BufferSpec(external_ptr=va)).ensure_allocated()
  return Tensor(UOp.from_buffer(nb, dtypes.uint32, devs[d]))
_dal:dict = {}
@functools.cache
def _push_kernel(*bufs:UOp, P:int, me:int, m:int) -> UOp:
  # the first m words of x (the last buf) ^ 7 into words me*m.. of each of the P outs (peers' buffers mapped here): all the peers at once
  x = bufs[P]
  g, l = UOp.range(m//256, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  i = g*256 + l
  v = x[i].load() ^ 7
  return UOp.group(*[o[me*m + i].store(v) for o in bufs[:P]]).end(g, l).sink(arg=KernelInfo(name=f"push{P}", opts_to_apply=()))
def peers_write(d:int, peers:list[int]) -> list[Tensor]:
  # one kernel on d writing E//N words to each peer (slice d of the peer's dst: no two writers on one)
  for p in peers:
    if (p, d) not in _dal: _dal[(p, d)] = dalias(p, d)
  return list(Tensor.custom_kernel(*[_dal[(p, d)] for p in peers], src[d], fxn=functools.partial(_push_kernel, P=len(peers), me=d, m=E//N//256*256)))

modes = getenv("MODES", "local,cu1,cuP,cuall,push1,pushP,pushall,sdmaP,sdmaall,sdmarot").split(",")
if "local" in modes: bench("local", lambda: (src[0] ^ 7).contiguous().realize(), E*4)
if "cu1" in modes: bench("cu1", lambda: peers_read(0, [1]).realize(), E*4)
if "cuP" in modes:
  for P in [int(p) for p in getenv("PEERS", "2,4,8,13").split(",") if int(p) < N]:
    bench(f"cu{P}", lambda P=P: peers_read(0, list(range(1, P+1))).realize(), E//P*P*4)
if "cuall" in modes:
  bench("cuall", lambda: Tensor.realize(*[peers_read(d, [p for p in range(N) if p != d]) for d in range(N)]), E//(N-1)*(N-1)*4, N)
if "push1" in modes: bench("push1", lambda: Tensor.realize(*peers_write(0, [1])), E//N//256*256*4)
if "pushP" in modes:
  for P in [int(p) for p in getenv("PEERS", "2,4,8,13").split(",") if int(p) < N]:
    bench(f"push{P}", lambda P=P: Tensor.realize(*peers_write(0, list(range(1, P+1)))), E//N//256*256*P*4)
if "pushall" in modes:
  bench("pushall", lambda: Tensor.realize(*[o for d in range(N) for o in peers_write(d, [p for p in range(N) if p != d])]), E//N//256*256*(N-1)*4, N)
if "sdmaP" in modes:
  for P in [1] + [int(p) for p in getenv("PEERS", "2,4,8,13").split(",") if int(p) < N]:
    bench(f"sdma{P}", lambda P=P: Tensor.realize(*[src[p][:E//P].to(devs[0]) for p in range(1, P+1)]), E//P*P*4)
if "sdmaall" in modes:
  m = E // (N-1)
  bench("sdmaall", lambda: Tensor.realize(*[src[p][:m].to(devs[d]) for d in range(N) for p in range(N) if p != d]), m*(N-1)*4, N)
if "sdmarot" in modes:
  # (the same copies, a device's in rotated order: from d+1, d+2, .. (no source read by all at once; AM_COPY_QUEUES=2: k alternates engines))
  m = E // (N-1)
  bench("sdmarot", lambda: Tensor.realize(*[src[(d+k) % N][:m].to(devs[d]) for k in range(1, N) for d in range(N)]), m*(N-1)*4, N)
