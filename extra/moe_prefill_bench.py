#!/usr/bin/env python3
"""
the grouped experts of a prefill chunk on one device (a device's share of qwen4exp's 512 experts, 10 used a token), timed SERIALLY over L
layers of distinct cold weights: moe_route, the fused gate|up grouped kernel (its activation in the kernel), the down grouped kernel.
us a layer, the weights' GB/s and the useful TOP/s (2*tokens*k_local*(3*D*F)).
env: TOKENS=576,1152,4096,13824 E=86 (local experts) NE=512 K=10 D=2560 F=640 TYPE=IQ3_S|IQ4_XS|Q8_0 DTYPE=IQ4_NL TILE=32 L=6 PARTS=gateup,down
MODE=grouped|bundle (batched decoding's bundled experts)
"""
import hashlib, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm.kernels.amd_qwen import moe_route, moe_grouped_linear, moe_linear, QUANT_SIZES

dev, rng, L = Device.DEFAULT, np.random.default_rng(0), getenv("L", 6)
E, NE, K, D, F, TILE = getenv("E", 86), getenv("NE", 512), getenv("K", 10), getenv("D", 2560), getenv("F", 640), getenv("TILE", 32)
TYPE = getattr(amd_qwen, getenv("TYPE", "IQ3_S"))
# (the C4 IQ4_XS file: gate and up IQ3_S, down IQ4_NL (a few Q8_0))
DTYPE = getattr(amd_qwen, getenv("DTYPE", "IQ4_NL"))
def weights(rows:int, cols:int, typ:int=TYPE) -> Tensor:
  nbytes = E * rows * cols // 256 * QUANT_SIZES[typ]
  b = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
  # (the blocks' fp16 scales sane: their first 2 bytes (every format here starts a block with d) a small positive half)
  # (IQ4_NL: QUANT_SIZES has 8 blocks of 32 (18 bytes), each with its d)
  b.reshape(-1, 18 if typ == amd_qwen.IQ4_NL else QUANT_SIZES[typ])[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
  return Tensor(b.view(np.uint16), device=dev).realize()
gus, dns = [weights(2*F, D) for _ in range(L)], [weights(D, F, DTYPE) for _ in range(L)]
first = Tensor([0], dtype=dtypes.int32, device=dev).realize()
parts = getenv("PARTS", "gateup,down").split(",")

def timeit(fn, *args) -> float:
  run = TinyJit(fn)
  for _ in range(3): run(*args)
  Device[dev].synchronize()
  ts = []
  for _ in range(5):
    t0 = time.perf_counter()
    run(*args)
    Device[dev].synchronize()
    ts.append(time.perf_counter() - t0)
  return min(ts) / L * 1e6

for T in [int(t) for t in getenv("TOKENS", "576,1152,4096,13824").split(",")]:
  sel = Tensor(np.argsort(rng.random((1, T, NE)), axis=-1)[..., :K].astype(np.int32), device=dev).realize()
  x = Tensor(rng.normal(size=(T, D)).astype(np.float32), device=dev).realize()
  # (the routing in numpy, as moe_route's counting sort: every local expert's selections in its rows padded to whole tiles. ROUTE=1: moe_route
  # on the device, timed in the chain)
  sn = sel.numpy().reshape(-1)
  R = -(-(len(sn) + (TILE-1)*E) // TILE) * TILE
  ent, til, at = np.full(R, -1, np.int32), np.full(R // TILE, -1, np.int32), 0
  for e in range(E):
    idx = np.nonzero(sn == e)[0]
    ent[at:at+len(idx)] = idx
    nt = -(-len(idx) // TILE)
    til[at // TILE:at // TILE + nt] = e
    at += nt * TILE
  ent0, til0 = Tensor(ent, device=dev).realize(), Tensor(til, device=dev).realize()
  act0 = moe_grouped_linear(gus[0], TYPE, E, x, ent0, til0, slots=K, act=(1.0, 0.0, 0.0)).contiguous().realize()
  def layer(i:int, x:Tensor, sel:Tensor) -> Tensor:
    if getenv("MODE", "grouped") == "bundle":
      # (batched decoding's path: the rows of an expert in bundles that share its weight loads, the decode kernels)
      gu = moe_linear(gus[i], TYPE, E, x.reshape(1, T, 1, D), sel, first)
      gu = gu.reshape(1, T, K, -1, 2, 16)
      act = (gu[..., 0, :].flatten(-2).silu() * gu[..., 1, :].flatten(-2)) if "gateup" in parts else gu[..., 0, :].flatten(-2)
      return moe_linear(dns[i], DTYPE, E, act, sel, first).reshape(T*K, D) if "down" in parts else act.reshape(T*K, F)
    if "route" in parts: entries, tiles = moe_route(Tensor(sel.uop.after(x.uop)), first, E, TILE)[1:3]
    else: entries, tiles = ent0, til0
    if "gateup" not in parts: return moe_grouped_linear(dns[i], DTYPE, E, Tensor(act0.uop.after(x.uop)), entries, tiles, selections=T*K,
                                                        sorted_out=True)
    act = moe_grouped_linear(gus[i], TYPE, E, x, entries, tiles, slots=K, act=(1.0, 0.0, 0.0))
    if "down" not in parts: return act.float()[:, :D].pad(((0, 0), (0, max(0, D - int(act.shape[1])))))[:T]
    return moe_grouped_linear(dns[i], DTYPE, E, act.contiguous(), entries, tiles, selections=T*K, sorted_out=True)
  def chain(x:Tensor, sel:Tensor) -> Tensor:
    dep = None
    for i in range(L):
      xi = Tensor(x.uop.after(dep.uop)) if dep is not None else x
      dep = layer(i, xi, sel)
    return dep.realize()
  t = timeit(chain, x, sel)
  # (the rows of the selections only: the padding rows aren't written)
  o = layer(L-1, x, sel).numpy()
  outhash = hashlib.md5(o[ent[:len(o)] >= 0].tobytes() if o.shape[0] == R else o.tobytes()).hexdigest()[:12]
  local = T * K * E / NE
  wbytes = E * D * F / 256 * (2 * QUANT_SIZES[TYPE] + QUANT_SIZES[DTYPE])
  print(f"{Device[dev].arch} {getenv('TYPE', 'IQ3_S')} {','.join(parts)} T={T:6d} tile={TILE}: {t:9.1f} us a layer ({local:7.0f} local selections, "
        f"{local/E:5.1f}/expert) weights {wbytes/t/1e3:6.0f} GB/s once, useful {2*local*3*D*F/t/1e6:6.1f} TOP/s out {outhash}", flush=True)
