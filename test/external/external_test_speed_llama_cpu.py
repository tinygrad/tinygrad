#!/usr/bin/env python3
# CPU speed test: TinyLlama-1.1B (TinyLlama 1.1B config, random weights, no download) single token forward pass.
# This is the memory bound decode step, where every weight is read exactly once. tinygrad is compared against
# a hand written torch reference on the same machine, single threaded, in a separate process.
import os, sys, json, time, gc, subprocess, unittest
os.environ.setdefault("MAX_BUFFER_SIZE", "0")  # 1.1B fp32 weights are 4.4GB, over the default buffer cap
os.environ.setdefault("OMP_NUM_THREADS", "1")  # tinygrad CI runs single threaded
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# TinyLlama 1.1B
LLAMA_1B = {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4,
            "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 5632}
WARMUP, ITERS = 3, 10

# **** torch reference (mirrors extra/models/llama.py) ****
TORCH_REF = r'''
import os, json, time, gc
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import torch, torch.nn.functional as F
torch.set_num_threads(1)

P = {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4,
     "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 5632}

class RMSNorm(torch.nn.Module):
  def __init__(self, dim, eps):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.eps = eps
  def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:(dim // 2)] / dim))
  freqs = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
  return torch.stack((freqs.cos(), freqs.sin()), dim=-1)  # (end, dim//2, 2)

def apply_rotary(x, freqs_cis):
  bs, seqlen, n_heads, head_dim = x.shape
  x = x.reshape(bs, seqlen, n_heads, head_dim // 2, 2)
  c, d = freqs_cis[None, :, None, :, 0:1], freqs_cis[None, :, None, :, 1:2]
  a, b = x[..., 0:1], x[..., 1:2]
  return torch.cat((a * c - b * d, a * d + b * c), dim=-1).reshape(bs, seqlen, n_heads, head_dim)

class Attention(torch.nn.Module):
  def __init__(self, dim, n_heads, n_kv_heads):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, dim // n_heads
    self.wq = torch.nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.wk = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wv = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wo = torch.nn.Linear(n_heads * self.head_dim, dim, bias=False)
  def forward(self, x, freqs_cis):
    bs, seqlen, _ = x.shape
    xq = self.wq(x).reshape(bs, seqlen, self.n_heads, self.head_dim)
    xk = self.wk(x).reshape(bs, seqlen, self.n_kv_heads, self.head_dim)
    xv = self.wv(x).reshape(bs, seqlen, self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary(xq, freqs_cis), apply_rotary(xk, freqs_cis)
    attn = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
                                         is_causal=True, enable_gqa=True)
    return self.wo(attn.transpose(1, 2).reshape(bs, seqlen, -1))

class FeedForward(torch.nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
  def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.attention = Attention(p["dim"], p["n_heads"], p["n_kv_heads"])
    self.feed_forward = FeedForward(p["dim"], p["hidden_dim"])
    self.attention_norm, self.ffn_norm = RMSNorm(p["dim"], p["norm_eps"]), RMSNorm(p["dim"], p["norm_eps"])
  def forward(self, x, freqs_cis):
    h = x + self.attention(self.attention_norm(x), freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

class Transformer(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.layers = torch.nn.ModuleList([Block(p) for _ in range(p["n_layers"])])
    self.norm = RMSNorm(p["dim"], p["norm_eps"])
    self.tok_embeddings = torch.nn.Embedding(p["vocab_size"], p["dim"])
    self.output = torch.nn.Linear(p["dim"], p["vocab_size"], bias=False)
    self.freqs_cis = precompute_freqs_cis(p["dim"] // p["n_heads"], 2048, 10000.0)
  def forward(self, tokens, start_pos=0):
    h = self.tok_embeddings(tokens)
    freqs = self.freqs_cis[start_pos:start_pos + tokens.shape[1]]
    for layer in self.layers: h = layer(h, freqs)
    return self.output(self.norm(h))

torch.manual_seed(0)
model = Transformer(P).eval()
toks = torch.randint(0, P["vocab_size"], (1, 1))
def run():
  with torch.no_grad(): model(toks, 1).argmax().item()
for _ in range(3): run()
ts = []
for _ in range(10):
  gc.collect(); t0 = time.perf_counter(); run(); ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()
print(json.dumps({"min": ts[0], "med": ts[len(ts)//2], "all": [round(x, 2) for x in ts]}))
'''

def torch_bench() -> dict:
  # a subprocess so the 4.4GB of torch weights are released before tinygrad allocates its own
  out = subprocess.run([sys.executable, "-c", TORCH_REF], capture_output=True, text=True, check=True)
  return json.loads(out.stdout.strip().splitlines()[-1])

class TestLlamaCPUSpeed(unittest.TestCase):
  def test_llama_1b_faster_than_torch(self):
    from tinygrad import Tensor, Device
    from extra.models.llama import Transformer
    print(f"\nDEV={Device.DEFAULT}, torch reference in subprocess")
    ref = torch_bench()
    print(f"torch   : min {ref['min']:8.2f} ms  med {ref['med']:8.2f} ms  all {ref['all']}")

    Tensor.manual_seed(0)
    p = LLAMA_1B
    model = Transformer(dim=p["dim"], hidden_dim=p["hidden_dim"], n_heads=p["n_heads"], n_layers=p["n_layers"],
                        norm_eps=p["norm_eps"], vocab_size=p["vocab_size"], n_kv_heads=p["n_kv_heads"],
                        max_context=1024, disable_kv_cache=True, jit=True)
    toks = Tensor([[1]])
    # no kv cache: one static graph for the decode step, identical work to the torch reference
    def run(): model.forward_jit(toks, 0, 0.0, 0, 0.8, 0.0, 0.0).realize()
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(ITERS):
      gc.collect()
      t0 = time.perf_counter()
      run()
      ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    tg_min, tg_med = ts[0], ts[len(ts)//2]
    print(f"tinygrad: min {tg_min:8.2f} ms  med {tg_med:8.2f} ms  all {[round(x, 2) for x in ts]}")
    print(f"ratio tinygrad/torch: min {tg_min/ref['min']:.3f}x  med {tg_med/ref['med']:.3f}x")
    if os.getenv("ASSERT_FASTER", "0") == "1":
      self.assertLess(tg_min, ref["min"], "tinygrad is slower than torch on a single token llama-1B forward pass")
    else:
      print("ASSERT_FASTER=0: reporting the ratio only")

if __name__ == '__main__': unittest.main()
