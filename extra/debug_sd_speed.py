# NOTE: this is written in a way that checkout back to old commit still works
# fast SD 297ms step on M1 Max, 4444e6d  https://github.com/tinygrad/tinygrad/pull/2129
# lazy rewrite, 1765849  https://github.com/tinygrad/tinygrad/pull/2878
# SD 415ms step on M1 Max on master around 11/15/2024

import time
from typing import Optional
try: from tinygrad.jit import TinyJit
except ImportError: from tinygrad import TinyJit
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import GlobalCounters
from tinygrad.nn import Linear, LayerNorm
from tinygrad.nn.state import get_parameters

class CrossAttention:
  def __init__(self, query_dim:int, ctx_dim:int, n_heads:int, d_head:int):
    self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = Linear(ctx_dim,   n_heads*d_head, bias=False)
    self.to_v = Linear(ctx_dim,   n_heads*d_head, bias=False)
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    ctx = x if ctx is None else ctx
    q,k,v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
    q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
    attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
    h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
    return h_.sequential(self.to_out)

class GEGLU:
  def __init__(self, dim_in:int, dim_out:int):
    self.proj = Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def __call__(self, x:Tensor) -> Tensor:
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * gate.gelu()

class FeedForward:
  def __init__(self, dim:int, mult:int=4):
    self.net = [
      GEGLU(dim, dim*mult),
      lambda x: x,  # needed for weights loading code to work
      Linear(dim*mult, dim)
    ]

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.net)

class BasicTransformerBlock:
  def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff    = FeedForward(dim)
    self.attn2 = CrossAttention(dim, ctx_dim, n_heads, d_head)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.norm3 = LayerNorm(dim)

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    x = x + self.attn1(self.norm1(x))  # 5.4 before, # 6.8 master
    x = x + self.attn2(self.norm2(x), ctx=ctx)  # 12 before, 12 master
    x = x + self.ff(self.norm3(x))  # 23 before, # 27 master
    return x

def helper_test(gen, model):
  tms = []
  for _ in range(5):
    early_gen = [x.realize() if isinstance(x, Tensor) else x for x in gen()]
    GlobalCounters.reset()
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter_ns()
    model(*early_gen)
    Device[Device.DEFAULT].synchronize()
    tms.append(time.perf_counter_ns() - st)
  print(f"{min(tms)/1e6=:.2f} ms")

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype).lazydata
    p.realize()

def test_transformer_block():
  # dim, d_head, x = 320, 40, (4096, 320)  # 137ms 4444e6d 115ms master
  # dim, d_head, x = 640, 80, (1024, 640)  #  36ms 4444e6d, 31ms master
  dim, d_head, x = 1280, 160, (256, 1280)  # 23ms 4444e6d, 28ms master, 31ms on 176584993

  model = [BasicTransformerBlock(dim, 768, 8, d_head) for _ in range(4)]

  derandomize_model(model)
  @TinyJit
  def test(t, t2):
    for l in model: t = l(t, t2)
    return t.realize()
  helper_test(lambda: (Tensor.empty(2, *x), Tensor.empty(2, 77, 768)), test)

if __name__ == "__main__":
  test_transformer_block()
