import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)
import time
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.helpers import colorize_float, getenv, CI
from tinygrad import nn as tinygrad_nn

# LLaMA 1B config
DIM = 2048
N_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 8192
VOCAB_SIZE = 128256
HEAD_DIM = DIM // N_HEADS
N_REP = N_HEADS // N_KV_HEADS
NORM_EPS = 1e-5
MAX_CONTEXT = 128
SEQ_LEN = 1  # single token decode

torch_dt = torch.float16 if getenv("HALF", 0) else torch.float32
torch_device = torch.device('mps' if getenv("MPS", 0) else ('cuda' if getenv("TORCHCUDA", 0) else 'cpu'))
if str(torch_device) == "mps":
  import torch.mps
  def sync(): torch.mps.synchronize()
elif str(torch_device) == "cuda":
  import torch.cuda
  def sync(): torch.cuda.synchronize()
else:
  def sync(): pass

save_ops, save_mem = 0, 0
CNT = getenv("CNT", 8)

# +----- Torch Port of extras/models/llama.py -----+

def torch_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
  freqs = torch.arange(end).unsqueeze(1) * freqs.unsqueeze(0)
  return torch.stack([freqs.cos(), freqs.sin()], dim=-1).reshape(1, end, 1, dim//2, 2)

def torch_complex_mult(A, c, d):
  a, b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return torch.cat([ro, co], dim=-1)

def torch_apply_rotary_emb(xq, xk, freqs_cis):
  xq = xq.reshape(*xq.shape[:-1], -1, 2)
  xk = xk.reshape(*xk.shape[:-1], -1, 2)
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = torch_complex_mult(xq, c, d)
  xk_out = torch_complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def torch_repeat_kv(x, n_rep):
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  return x.repeat(1, 1, 1, n_rep).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class TorchRMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps
  def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class TorchAttention(nn.Module):
  def __init__(self, dim, n_heads, n_kv_heads, max_context):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    self.n_rep = n_heads // n_kv_heads
    self.max_context = max_context

    self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    self.register_buffer('cache_kv', torch.zeros(2, 1, max_context, n_kv_heads, self.head_dim))

  def forward(self, x, start_pos, freqs_cis):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = torch_apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :] = torch.stack([xk, xv])
    keys = self.cache_kv[0, :, :start_pos+seqlen, :, :]
    values = self.cache_kv[1, :, :start_pos+seqlen, :, :]

    keys, values = torch_repeat_kv(keys, self.n_rep), torch_repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

    mask = None
    if seqlen > 1:
      mask = torch.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), device=x.device).triu(start_pos+1)

    attn = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
    attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn)

class TorchFeedForward(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TorchTransformerBlock(nn.Module):
  def __init__(self, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context):
    super().__init__()
    self.attention = TorchAttention(dim, n_heads, n_kv_heads, max_context)
    self.feed_forward = TorchFeedForward(dim, hidden_dim)
    self.attention_norm = TorchRMSNorm(dim, norm_eps)
    self.ffn_norm = TorchRMSNorm(dim, norm_eps)

  def forward(self, x, start_pos, freqs_cis):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

# +----- End of Torch Port -----+

from extra.models.llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv, TransformerBlock as TinyTransformerBlock

class TinyRMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)
  def __call__(self, x:Tensor) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight

def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048, 2048))
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # force syncing
    [x.numpy() if isinstance(x, Tensor) or str(torch_device) == "cpu" else x.cpu().numpy() for x in args if x is not None]

    # clear 32MB global memory cache
    cache_defeat += 1

    # manual pre sync
    if isinstance(args[0], Tensor):
      local_device = Device[args[0].device]
      local_device.synchronize()
    else: sync()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor): local_device.synchronize()
    else: sync()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

def helper_test_generic(name, f1, f1_args, f2, f2_args):
  with torch.no_grad():
    val_torch, et_torch = helper_test_speed(f1, *f1_args)
  val_tinygrad, et_tinygrad = helper_test_speed(f2, *f2_args)

  desc = "faster" if et_torch > et_tinygrad else "slower"
  flops = save_ops*1e-6
  mem = save_mem*1e-6
  print(("\r" if not CI else "")+f"{name:50s} {et_torch:7.2f} ms ({flops/et_torch:8.2f} GFLOPS {mem/et_torch:7.2f} GB/s) in torch, {et_tinygrad:7.2f} ms ({flops/et_tinygrad:8.2f} GFLOPS {mem/et_tinygrad:7.2f} GB/s) in tinygrad, {colorize_float(et_tinygrad/et_torch)} {desc}")  # noqa: E501
  atol, rtol = (1e-2, 1e-2) if torch_dt == torch.float16 else (1e-3, 1e-3)
  np.testing.assert_allclose(val_tinygrad, val_torch, atol=atol, rtol=rtol)

@unittest.skipIf(getenv("BIG") == 1, "only big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestBlockLayers(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.manual_seed(0)
    # shared inputs for attention tests
    cls.torch_x = (torch.rand(1, SEQ_LEN, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    cls.tiny_x = Tensor(cls.torch_x.cpu().numpy())

    # freqs_cis for RoPE
    cls.torch_freqs = torch_precompute_freqs_cis(HEAD_DIM, MAX_CONTEXT * 2).to(torch_dt).to(torch_device)
    cls.tiny_freqs = precompute_freqs_cis(HEAD_DIM, MAX_CONTEXT * 2)

    # KV cache
    cls.start_pos = 2
    cls.torch_cache_k = (torch.rand(1, cls.start_pos, N_KV_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    cls.torch_cache_v = (torch.rand(1, cls.start_pos, N_KV_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    cls.tiny_cache_k = Tensor(cls.torch_cache_k.cpu().numpy())
    cls.tiny_cache_v = Tensor(cls.torch_cache_v.cpu().numpy())

  def test_00_embedding_lookup(self):
    vocab_size = 10000
    torch_emb = nn.Embedding(vocab_size, DIM).to(torch_dt).to(torch_device)
    tiny_emb = tinygrad_nn.Embedding(vocab_size, DIM)
    tiny_emb.weight = Tensor(torch_emb.weight.detach().cpu().numpy())

    indices = torch.tensor([[1, 23, 456, 7890]], dtype=torch.long).to(torch_device)
    tiny_indices = Tensor(indices.cpu().numpy().astype(np.int32))

    def f1(x): return torch_emb(x)
    def f2(x): return tiny_emb(x)
    helper_test_generic(f"embedding_lookup vocab={vocab_size} dim={DIM}", f1, (indices,), TinyJit(f2), (tiny_indices,))

  def test_01_rmsnorm_attention(self):
    torch_norm = TorchRMSNorm(DIM, NORM_EPS).to(torch_dt).to(torch_device)
    tiny_norm = TinyRMSNorm(DIM, NORM_EPS)
    tiny_norm.weight = Tensor(torch_norm.weight.detach().cpu().numpy())

    def f1(x): return torch_norm(x)
    def f2(x): return tiny_norm(x)
    helper_test_generic("attention_norm (RMSNorm)", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_02_wq_projection(self):
    torch_wq = torch.nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_wq = tinygrad_nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
    tiny_wq.weight = Tensor(torch_wq.weight.detach().cpu().numpy())

    def f1(x): return torch_wq(x)
    def f2(x): return tiny_wq(x)
    helper_test_generic(f"wq projection {DIM}->{N_HEADS*HEAD_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_03_wk_projection(self):
    torch_wk = torch.nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_wk = tinygrad_nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
    tiny_wk.weight = Tensor(torch_wk.weight.detach().cpu().numpy())

    def f1(x): return torch_wk(x)
    def f2(x): return tiny_wk(x)
    helper_test_generic(f"wk projection {DIM}->{N_KV_HEADS*HEAD_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_04_wv_projection(self):
    torch_wv = torch.nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_wv = tinygrad_nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
    tiny_wv.weight = Tensor(torch_wv.weight.detach().cpu().numpy())

    def f1(x): return torch_wv(x)
    def f2(x): return tiny_wv(x)
    helper_test_generic(f"wv projection {DIM}->{N_KV_HEADS*HEAD_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_05_rope(self):
    torch.manual_seed(0)
    torch_xq = (torch.rand(1, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_xk = (torch.rand(1, SEQ_LEN, N_KV_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_xq = Tensor(torch_xq.cpu().numpy())
    tiny_xk = Tensor(torch_xk.cpu().numpy())

    start_pos = self.start_pos
    torch_freqs_slice = self.torch_freqs[:, start_pos:start_pos+SEQ_LEN, :, :, :]
    tiny_freqs_slice = self.tiny_freqs[:, start_pos:start_pos+SEQ_LEN, :, :, :]

    def f1(xq, xk):
      xq_out, xk_out = torch_apply_rotary_emb(xq, xk, torch_freqs_slice)
      return xq_out
    def f2(xq, xk):
      xq_out, xk_out = apply_rotary_emb(xq, xk, tiny_freqs_slice)
      return xq_out
    helper_test_generic(f"rope heads={N_HEADS} kv_heads={N_KV_HEADS}", f1, (torch_xq, torch_xk), TinyJit(f2), (tiny_xq, tiny_xk))

  def test_06_kv_cache_update(self):
    torch.manual_seed(0)
    torch_new_k = (torch.rand(1, SEQ_LEN, N_KV_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_new_k = Tensor(torch_new_k.cpu().numpy())

    def f1(cache, new): return torch.cat([cache, new], dim=1)
    def f2(cache, new): return cache.cat(new, dim=1)
    helper_test_generic(f"kv_cache_update {self.start_pos}+{SEQ_LEN}", f1, (self.torch_cache_k, torch_new_k),
                        TinyJit(f2), (self.tiny_cache_k, tiny_new_k))

  def test_07_repeat_kv(self):
    torch.manual_seed(0)
    torch_k = (torch.rand(1, self.start_pos, N_KV_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_k = Tensor(torch_k.cpu().numpy())

    def f1(k): return torch_repeat_kv(k, N_REP)
    def f2(k): return repeat_kv(k, N_REP).contiguous()  # force materialization
    helper_test_generic(f"repeat_kv n_rep={N_REP}", f1, (torch_k,), TinyJit(f2), (tiny_k,))

  def test_08_sdpa(self):
    torch.manual_seed(0)
    # single query attending to cached keys/values
    torch_q = (torch.rand(1, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_k = (torch.rand(1, N_HEADS, self.start_pos + SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_v = (torch.rand(1, N_HEADS, self.start_pos + SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_q, tiny_k, tiny_v = Tensor(torch_q.cpu().numpy()), Tensor(torch_k.cpu().numpy()), Tensor(torch_v.cpu().numpy())

    def f1(q, k, v): return F.scaled_dot_product_attention(q, k, v)
    def f2(q, k, v): return q.scaled_dot_product_attention(k, v)
    helper_test_generic(f"sdpa q=1 kv={self.start_pos+SEQ_LEN} heads={N_HEADS}", f1, (torch_q, torch_k, torch_v),
                        TinyJit(f2), (tiny_q, tiny_k, tiny_v))

  def test_09_wo_projection(self):
    torch_wo = torch.nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_wo = tinygrad_nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)
    tiny_wo.weight = Tensor(torch_wo.weight.detach().cpu().numpy())

    def f1(x): return torch_wo(x)
    def f2(x): return tiny_wo(x)
    helper_test_generic(f"wo projection {N_HEADS*HEAD_DIM}->{DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_10_residual_add(self):
    torch.manual_seed(0)
    torch_a = (torch.rand(1, SEQ_LEN, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_b = (torch.rand(1, SEQ_LEN, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_a, tiny_b = Tensor(torch_a.cpu().numpy()), Tensor(torch_b.cpu().numpy())

    def f(a, b): return a + b
    helper_test_generic(f"residual_add {DIM}", f, (torch_a, torch_b), TinyJit(f), (tiny_a, tiny_b))

  def test_11_rmsnorm_ffn(self):
    torch_norm = TorchRMSNorm(DIM, NORM_EPS).to(torch_dt).to(torch_device)
    tiny_norm = TinyRMSNorm(DIM, NORM_EPS)
    tiny_norm.weight = Tensor(torch_norm.weight.detach().cpu().numpy())

    def f1(x): return torch_norm(x)
    def f2(x): return tiny_norm(x)
    helper_test_generic("ffn_norm (RMSNorm)", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_12_ffn_w1(self):
    torch_w1 = torch.nn.Linear(DIM, HIDDEN_DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_w1 = tinygrad_nn.Linear(DIM, HIDDEN_DIM, bias=False)
    tiny_w1.weight = Tensor(torch_w1.weight.detach().cpu().numpy())

    def f1(x): return torch_w1(x)
    def f2(x): return tiny_w1(x)
    helper_test_generic(f"ffn w1 (gate) {DIM}->{HIDDEN_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_13_ffn_w3(self):
    torch_w3 = torch.nn.Linear(DIM, HIDDEN_DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_w3 = tinygrad_nn.Linear(DIM, HIDDEN_DIM, bias=False)
    tiny_w3.weight = Tensor(torch_w3.weight.detach().cpu().numpy())

    def f1(x): return torch_w3(x)
    def f2(x): return tiny_w3(x)
    helper_test_generic(f"ffn w3 (up) {DIM}->{HIDDEN_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_14_silu(self):
    torch.manual_seed(0)
    torch_h = (torch.rand(1, SEQ_LEN, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_h = Tensor(torch_h.cpu().numpy())

    def f1(x): return F.silu(x)
    def f2(x): return x.silu()
    helper_test_generic(f"silu {HIDDEN_DIM}", f1, (torch_h,), TinyJit(f2), (tiny_h,))

  def test_15_ffn_gate_mul(self):
    torch.manual_seed(0)
    torch_a = (torch.rand(1, SEQ_LEN, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_b = (torch.rand(1, SEQ_LEN, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_a, tiny_b = Tensor(torch_a.cpu().numpy()), Tensor(torch_b.cpu().numpy())

    def f(a, b): return a * b
    helper_test_generic(f"ffn gate_mul {HIDDEN_DIM}", f, (torch_a, torch_b), TinyJit(f), (tiny_a, tiny_b))

  def test_16_ffn_w2(self):
    torch.manual_seed(0)
    torch_h = (torch.rand(1, SEQ_LEN, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_h = Tensor(torch_h.cpu().numpy())

    torch_w2 = torch.nn.Linear(HIDDEN_DIM, DIM, bias=False).to(torch_dt).to(torch_device)
    tiny_w2 = tinygrad_nn.Linear(HIDDEN_DIM, DIM, bias=False)
    tiny_w2.weight = Tensor(torch_w2.weight.detach().cpu().numpy())

    def f1(x): return torch_w2(x)
    def f2(x): return tiny_w2(x)
    helper_test_generic(f"ffn w2 (down) {HIDDEN_DIM}->{DIM}", f1, (torch_h,), TinyJit(f2), (tiny_h,))

  def test_17_output_norm(self):
    torch_norm = TorchRMSNorm(DIM, NORM_EPS).to(torch_dt).to(torch_device)
    tiny_norm = TinyRMSNorm(DIM, NORM_EPS)
    tiny_norm.weight = Tensor(torch_norm.weight.detach().cpu().numpy())

    def f1(x): return torch_norm(x)
    def f2(x): return tiny_norm(x)
    helper_test_generic("output_norm (RMSNorm)", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_18_output_projection(self):
    torch_out = torch.nn.Linear(DIM, VOCAB_SIZE, bias=False).to(torch_dt).to(torch_device)
    tiny_out = tinygrad_nn.Linear(DIM, VOCAB_SIZE, bias=False)
    tiny_out.weight = Tensor(torch_out.weight.detach().cpu().numpy())

    def f1(x): return torch_out(x)
    def f2(x): return tiny_out(x)
    helper_test_generic(f"output projection {DIM}->{VOCAB_SIZE}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

  def test_19_argmax(self):
    torch.manual_seed(0)
    torch_logits = (torch.rand(VOCAB_SIZE, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_logits = Tensor(torch_logits.cpu().numpy())

    def f1(x): return x.argmax()
    def f2(x): return x.argmax()
    helper_test_generic(f"argmax vocab={VOCAB_SIZE}", f1, (torch_logits,), TinyJit(f2), (tiny_logits,))

  def test_20_transformer_block(self):
    torch.manual_seed(0)
    torch_block = TorchTransformerBlock(DIM, HIDDEN_DIM, N_HEADS, N_KV_HEADS, NORM_EPS, MAX_CONTEXT).to(torch_dt).to(torch_device)
    tiny_block = TinyTransformerBlock(DIM, HIDDEN_DIM, N_HEADS, N_KV_HEADS, NORM_EPS, MAX_CONTEXT)

    tiny_block.attention_norm.weight = Tensor(torch_block.attention_norm.weight.detach().cpu().numpy())
    tiny_block.attention.wq.weight = Tensor(torch_block.attention.wq.weight.detach().cpu().numpy())
    tiny_block.attention.wk.weight = Tensor(torch_block.attention.wk.weight.detach().cpu().numpy())
    tiny_block.attention.wv.weight = Tensor(torch_block.attention.wv.weight.detach().cpu().numpy())
    tiny_block.attention.wo.weight = Tensor(torch_block.attention.wo.weight.detach().cpu().numpy())
    tiny_block.ffn_norm.weight = Tensor(torch_block.ffn_norm.weight.detach().cpu().numpy())
    tiny_block.feed_forward.w1.weight = Tensor(torch_block.feed_forward.w1.weight.detach().cpu().numpy())
    tiny_block.feed_forward.w2.weight = Tensor(torch_block.feed_forward.w2.weight.detach().cpu().numpy())
    tiny_block.feed_forward.w3.weight = Tensor(torch_block.feed_forward.w3.weight.detach().cpu().numpy())

    start_pos = self.start_pos
    torch_freqs_slice = self.torch_freqs[:, start_pos:start_pos+SEQ_LEN, :, :, :]
    tiny_freqs_slice = self.tiny_freqs[:, start_pos:start_pos+SEQ_LEN, :, :, :]

    def f1(x): return torch_block(x, start_pos, torch_freqs_slice)
    def f2(x): return tiny_block(x, start_pos, tiny_freqs_slice, None)
    helper_test_generic(f"transformer_block dim={DIM} hidden={HIDDEN_DIM}", f1, (self.torch_x,), TinyJit(f2), (self.tiny_x,))

if __name__ == '__main__':
  unittest.main()
