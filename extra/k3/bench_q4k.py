#!/usr/bin/env python3
# PROFILE=1 DEV=AMD python3 extra/k3/bench_q4k.py: one gpu's streaming read bandwidth and the q4_k decode gemv of the trunk's shapes,
# the gpu time of each from the profile (graphs of distinct buffers, 512 MB: nothing in cache)
import collections, functools, re
from tinygrad import Tensor, TinyJit, Device, dtypes, UOp
from tinygrad.uop.ops import AxisType, KernelInfo, AddrSpace
from tinygrad.device import Compiled, ProfileGraphEvent
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import Linear, Q4_K, QUANT_SIZES

def _read_kernel(out:UOp, buf:UOp, iters:int, vec:int) -> UOp:  # every thread sums vec floats a step, the steps a grid apart
  G = int(out.shape[0]) // 256
  g, l, it = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL), UOp.range(iters, 2, AxisType.LOOP)
  tid = g*256 + l
  acc = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(UOp.const(0, dtypes.float)))
  vals = [buf[(it*G*256 + tid)*vec + j].load() for j in range(vec)]
  acc = acc.after(acc[0].store(sum(vals[1:], acc.after(it)[0].load() + vals[0])).end(it))
  return out[tid].store(acc[0].load()).end(g, l).sink(arg=KernelInfo(name=f"read_{vec}", opts_to_apply=()))

def kernel_us(fn, runs=4) -> dict[str, float]:
  for _ in range(runs): fn()
  Device[Device.DEFAULT].synchronize()
  ev = [e for e in Compiled.profile_events if isinstance(e, ProfileGraphEvent) and e.ents[0].device in (Device.DEFAULT, f"{Device.DEFAULT}:0")][-1]
  dur, cnt = collections.Counter(), collections.Counter()
  for e in ev.ents:
    n = re.sub(r"\x1b\[[0-9;]*m", "", str(e.name))
    dur[n], cnt[n] = dur[n] + float(ev.sigs[e.en_id]) - float(ev.sigs[e.st_id]), cnt[n] + 1
  return {n: dur[n] / cnt[n] for n in dur}

def q4k(i:int, o:int) -> Linear:  # (random blocks: the gemv's time doesn't depend on them)
  l = Linear(i, o, bias=False)
  l.weight, l.ggml_type = Tensor.randint(o*i*QUANT_SIZES[Q4_K]//1024, low=0, high=1 << 30, dtype=dtypes.uint32).realize(), Q4_K
  return l

if __name__ == "__main__":
  MB = 256
  for vec in (1, 4):
    for G in (1024, 4096):
      # GPUS=n: the same on n gpus at once
      devs = [f"{Device.DEFAULT}:{d}" for d in range(getenv("GPUS", 1))]
      bufs = [Tensor.empty(MB << 18, dtype=dtypes.float, device=d).realize() for _ in range(4) for d in devs]
      fxn = functools.partial(_read_kernel, iters=(MB << 18) // (G*256*vec), vec=vec)
      jit = TinyJit(lambda: Tensor.realize(*[Tensor.custom_kernel(Tensor.empty(G*256, device=b.device), b, fxn=fxn)[0] for b in bufs]))
      us = kernel_us(jit)[f"read_{vec}"]
      print(f"read {MB} MB, {vec} floats a load, {G} workgroups of 256: {us:.0f} us = {(MB << 20) / us / 1e3:.0f} GB/s")
  for i, o in ((7168, 3072), (3072, 7168), (7168, 1536), (1536, 7168), (7168, 896), (896, 7168)):  # (a gpu's shapes of the trunk)
    nbytes = o*i*QUANT_SIZES[Q4_K]//256
    ls = [q4k(i, o) for _ in range(-(-512_000_000 // nbytes))]  # (past the 64 MB infinity cache)
    x = Tensor.randn(1, i).realize()
    jit = TinyJit(lambda: [l(x).realize() for l in ls])
    us = kernel_us(jit)["linear_q4_k"]
    print(f"q4_k {i} -> {o}: {nbytes/1e6:.1f} MB in {us:.1f} us = {nbytes / us / 1e3:.0f} GB/s")
  # a chain: each gemv's input the one before's output (a barrier between them, as in the model)
  for i, o in ((7168, 3072), (7168, 1536), (7168, 896)):
    n = -(-512_000_000 // (2*o*i*QUANT_SIZES[Q4_K]//256))
    pairs = [(q4k(i, o), q4k(o, i)) for _ in range(n)]
    x = Tensor.randn(1, i).realize()
    def chain():
      h = x
      for a, b in pairs: h = (b(a(h).float().clip(-1, 1)).float().clip(-1, 1))
      return h.realize()
    d = kernel_us(TinyJit(chain))
    print(f"chain {i} <-> {o}: q4_k {d['linear_q4_k']:.1f} us = {o*i*QUANT_SIZES[Q4_K]//256 / d['linear_q4_k'] / 1e3:.0f} GB/s, " +
          ", ".join(f"{k} {v:.1f}" for k, v in d.items() if k != "linear_q4_k"))
  # the mla absorb matmuls of a gpu (24 heads)
  from tinygrad.llm.kernels.amd import mla_bmm
  for I, O in ((128, 512), (512, 128)):
    ws = [Tensor.empty(24, O, I, dtype=dtypes.half).realize() for _ in range(-(-512_000_000 // (24*O*I*2)))]
    x = Tensor.randn(1, 24, 1, I).realize()
    us = kernel_us(TinyJit(lambda: Tensor.realize(*[mla_bmm(x, w) for w in ws])))["mla_bmm"]
    print(f"mla_bmm 24 x {I} -> {O}: {24*O*I*2/1e6:.1f} MB in {us:.1f} us = {24*O*I*2 / us / 1e3:.0f} GB/s")
