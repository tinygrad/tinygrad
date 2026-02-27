import os
os.environ.setdefault("DEVICE_IN_FUNCTION_BUG", "1")
os.environ.setdefault("HK_FLASH_ATTENTION", "1")
os.environ.setdefault("EMULATE", "AMD_CDNA4")
os.environ.setdefault("DEV", "NULL")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")

from tinygrad import Tensor, TinyJit, function

@function
def attn(q:Tensor, k:Tensor, v:Tensor) -> Tensor:
  return q.scaled_dot_product_attention(k, v, is_causal=True)

# use separate tensors for q, k, v
q = Tensor.randn(1, 1, 16, 128).requires_grad_(True)
k = Tensor.randn(1, 1, 16, 128).requires_grad_(True)
v = Tensor.randn(1, 1, 16, 128).requires_grad_(True)
for p in [q, k, v]:
  p.grad = p.zeros_like().contiguous().realize()

@TinyJit
def step():
  out = attn(q, k, v)
  out.sum().backward()
  Tensor.realize(q.grad, k.grad, v.grad)

step()
step()
print("PASS")
