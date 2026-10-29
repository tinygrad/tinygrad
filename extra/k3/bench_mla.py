# PROFILE=1: mla_decode_attention of a gpu (24 heads, D 576, DV 512) on a half latent cache at S keys, T tokens: a chain (each call's q the
# one before's output), the gpu time of its kernels a call and the cache's bytes over it
from tinygrad import Tensor, TinyJit, UOp, dtypes
from extra.k3.bench_q4k import kernel_us
from tinygrad.llm.kernels.amd import mla_decode_attention
H, D, DV, N = 24, 576, 512, 131072
cache = Tensor.randn(1, 1, N, D, dtype=dtypes.half).realize()
w = Tensor.randn(DV, D).realize() * 0.05
for T in (1, 2):
  for S in eval(__import__("os").environ.get("SS", "(4096, 24576, 131072)")):
    valid = UOp.variable("valid", 1, N)
    q = Tensor.randn(1, H, T, D).realize()
    def chain(v):
      h = q
      for _ in range(8): h = mla_decode_attention(h, cache, v, DV, 192 ** -0.5) @ w
      return h.realize()
    jit = TinyJit(chain)
    d = kernel_us(lambda: jit(valid.bind(S)))
    us = sum(t for n, t in d.items() if "mla" in n or "combine" in n)
    print(f"T={T} S={S}: " + ", ".join(f"{n} {t:.1f}" for n, t in d.items() if "mla" in n or "combine" in n) +
          f" us = {us:.1f} us a call, {S*D*2/us/1e3:.0f} GB/s of cache", flush=True)
