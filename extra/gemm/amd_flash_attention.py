from tinygrad import Tensor, getenv
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, Context
from tinygrad.llm.amd import amd_flash_attention, amd_flash_attention_causal

if __name__ == "__main__":
  B, H, N, D = getenv("B", 1), getenv("H", 32), getenv("N", 1024), getenv("D", 64)
  M, causal = getenv("M", N), getenv("CAUSAL", 0)
  q = Tensor.rand(B, H, M, D).cast(dtypes.half)
  k = Tensor.rand(B, H, N, D).cast(dtypes.half)
  v = Tensor.rand(B, H, N, D).cast(dtypes.half)
  o = Tensor.empty(B, H, M, D, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(q, k, v)

  q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, M, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, M, D)
  ets = []
  with Context(DEBUG=2):
    for _ in range(getenv("CNT", 5)):
      GlobalCounters.reset()
      tst = Tensor.custom_kernel(o_flat, q_flat, k_flat, v_flat,
                                 fxn=amd_flash_attention_causal if causal else amd_flash_attention)[0].realize()
      ets.append(GlobalCounters.time_sum_s)
  print(f"best time: {min(ets)*1e3:.2f}ms")

  if getenv("VERIFY", 1):
    with Context(DEBUG=0):
      mask = Tensor.full((1, 1, M, N), float("-inf"), buffer=False).triu(N-M+1) if causal else None
      ref = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask).reshape(B*H, M, D).realize()
      diff = (ref - tst).abs()
      err, max_err = diff.square().mean().item(), diff.max().item()
    print(f"mean squared error {err}, max error {max_err}")
    if err > 1e-2: raise RuntimeError("flash attention is wrong!")
    print("flash attention is correct!")
