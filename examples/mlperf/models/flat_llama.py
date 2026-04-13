import math, os, functools
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  if "DEV" not in os.environ: os.environ["DEV"] = "NULL"
  # CDNA
  os.environ["EMULATE"] = "AMD_CDNA4"
  os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"
  os.environ["ALL2ALL"] = "1"
  os.environ["USE_ATOMICS"] = "1"
  if "HK_FLASH_ATTENTION" not in os.environ:
    os.environ["HK_FLASH_ATTENTION"] = "1"
    if "ASM_GEMM" not in os.environ:
      os.environ["ASM_GEMM"] = "1"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters, profile_marker
from tinygrad.uop.ops import Ops, UOp
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis

FP8 = getenv("FP8", 0)

FP8_DTYPE = dtypes.fp8e4m3
FP8_GRAD_DTYPE = dtypes.fp8e5m2
FP8_MAX = 448.0

def quantize_fp8(x:Tensor, amax_state:Tensor|None=None):
  new_amax = x.abs().max().detach()
  scale = FP8_MAX / ((amax_state if amax_state is not None else new_amax) + 1e-8)
  x_scaled = x * scale
  x_clamped = x_scaled + (x_scaled.detach().clamp(-FP8_MAX, FP8_MAX) - x_scaled.detach())  # STE
  return x_clamped.cast(FP8_DTYPE), scale.float().reciprocal(), new_amax

def matmul(x:Tensor, w:Tensor, fp8=FP8, amax_x:Tensor|None=None, amax_w:Tensor|None=None) -> tuple[Tensor,...]:
  if not fp8:
    if getenv("ASM_GEMM"):
      from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
      if can_use_asm_gemm(x, w.T): return (asm_gemm(x, w.T),)
    return (x @ w.T,)
  x_fp8, x_scale, x_new_amax = quantize_fp8(x, amax_state=amax_x)
  w_fp8, w_scale, w_new_amax = quantize_fp8(w, amax_state=amax_w)
  combined_scale = x_scale * w_scale
  if getenv("ASM_GEMM"):
    from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
    if can_use_asm_gemm(x_fp8, w_fp8.T): return asm_gemm(x_fp8, w_fp8.T, combined_scale=combined_scale), x_new_amax, w_new_amax, x_fp8, w_fp8
  return x_fp8.dot(w_fp8.T, dtype=dtypes.float) * combined_scale, x_new_amax, w_new_amax, x_fp8, w_fp8

def _rmsnorm_fwd(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return (x * rrms).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_fwd_fxn(x_in_p, eps, device):
  return _rmsnorm_fwd(Tensor(x_in_p, device=device), eps)

def _rmsnorm_bwd(grad:UOp, call:UOp) -> tuple:
  x_normed = Tensor(call.gettuple(0)).float()
  do_float = Tensor(grad).float()
  d_x = Tensor(call.gettuple(1)) * (do_float - x_normed * (do_float * x_normed).mean(-1, keepdim=True))
  return (d_x.cast(call.src[1].dtype).uop,)

def rmsnorm(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_fwd_fxn(x_in.as_param(0).uop, eps, x_in.device)
  call = UOp.maketuple(fxn[0].uop, fxn[1].uop).call(x_in.uop, grad_fxn=_rmsnorm_bwd)
  return Tensor(call.gettuple(0)), Tensor(call.gettuple(1))

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024):
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    scaled_std = 0.02 / math.sqrt(2 * n_layers)

    # Attention
    self.wqkv = self.lin_per_layer(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2)
    self.wo = self.lin_per_layer(self.n_heads * self.head_dim, dim, std=scaled_std)

    # FeedForward
    self.w1 = self.lin_per_layer(dim, hidden_dim)
    self.w2 = self.lin_per_layer(hidden_dim, dim, std=scaled_std)
    self.w3 = self.lin_per_layer(dim, hidden_dim)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(1, vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

    if FP8:
      def _amax(): return Tensor.full((), FP8_MAX).contiguous().requires_grad_(False)
      names = ["xqkv", "wqkv", "xo", "wo", "x1", "w1", "x2", "w2", "x3", "w3"]
      # _fp8_amax[name][layer_idx] = scalar amax tensor
      self._fp8_amax = {name: [_amax() for _ in range(n_layers)] for name in names}
      self._fp8_amax["xout"] = [_amax()]
      self._fp8_amax["wout"] = [_amax()]

  def lin_per_layer(self, in_features:int, out_features:int, std:float=0.02):
    if getenv("ZEROS"): return Tensor.zeros(self.n_layers, out_features, in_features)
    return Tensor.normal(self.n_layers, out_features, in_features, mean=0.0, std=std)

  def attention(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                amax_xqkv=None, amax_wqkv=None, amax_xo=None, amax_wo=None):
    bsz, seqlen, _ = x.shape
    new_amaxs, saves = [], []

    x, rrms = rmsnorm(x, self.norm_eps)
    saves.extend([x, rrms])
    x = x * attention_norm

    xqkv, *ret = matmul(x, wqkv, amax_x=amax_xqkv, amax_w=amax_wqkv)
    new_amaxs.extend(ret[:2])
    saves.extend(ret[2:] + [xqkv])
    xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    if FP8: xq, xk, xv = xq.cast(dtypes.bfloat16), xk.cast(dtypes.bfloat16), xv.cast(dtypes.bfloat16)
    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    if getenv("HK_FLASH_ATTENTION"):
      from extra.thunder.amd.fa import flash_attention
      attn, *save = flash_attention(xq, xk, xv, is_causal=True)
      saves.extend(save)
    else:
      attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)

    out, *ret = matmul(attn, wo, amax_x=amax_xo, amax_w=amax_wo)
    new_amaxs.extend(ret[:2])
    saves.extend(ret[2:] + [out])
    return (out, *new_amaxs, *saves)

  def feed_forward(self, x:Tensor, ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor,
                   amax_x1=None, amax_w1=None, amax_x2=None, amax_w2=None, amax_x3=None, amax_w3=None):
    new_amaxs, saves = [], []

    x, rrms = rmsnorm(x, self.norm_eps)
    saves.extend([x, rrms])
    x = x * ffn_norm

    x_w1, *ret = matmul(x, w1, amax_x=amax_x1, amax_w=amax_w1)
    new_amaxs.extend(ret[:2])
    saves.extend(ret[2:] + [x_w1])
    x_w3, *ret = matmul(x.contiguous_backward(), w3, amax_x=amax_x3, amax_w=amax_w3)
    new_amaxs.extend(ret[:2])
    saves.extend(ret[2:] + [x_w3])
    out, *ret = matmul(x_w1.silu() * x_w3, w2, amax_x=amax_x2, amax_w=amax_w2)
    new_amaxs.extend(ret[:2])
    saves.extend(ret[2:] + [out])
    return (out, *new_amaxs, *saves)

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor,
                attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor,
                amax_xqkv=None, amax_wqkv=None, amax_xo=None, amax_wo=None,
                amax_x1=None, amax_w1=None, amax_x2=None, amax_w2=None, amax_x3=None, amax_w3=None):
    attn, *attn_ret = self.attention(x, freqs_cis, attention_norm, wqkv, wo,
                                     amax_xqkv=amax_xqkv, amax_wqkv=amax_wqkv, amax_xo=amax_xo, amax_wo=amax_wo)
    attn_amaxs, attn_saves = attn_ret[:4], attn_ret[4:]
    h = x + attn
    ffn, *ffn_ret = self.feed_forward(h, ffn_norm, w1, w2, w3,
                                      amax_x1=amax_x1, amax_w1=amax_w1, amax_x2=amax_x2, amax_w2=amax_w2, amax_x3=amax_x3, amax_w3=amax_w3)
    ffn_amaxs, ffn_saves = ffn_ret[:6], ffn_ret[6:]
    h = h + ffn
    return (h, *attn_amaxs, *ffn_amaxs, *attn_saves, *ffn_saves)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    from tinygrad.nn.state import get_parameters
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      self.wqkv.shard_(device, axis=1).realize()          # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2).realize()             # (n_layers, dim, in) shard in
      self.w1.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.w2.shard_(device, axis=2).realize()             # (n_layers, dim, hidden) shard in
      self.w3.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.shard_(device, axis=1).realize()
      self.freqs_cis.shard_(device, axis=None).realize()

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    a = self._fp8_amax if FP8 else None
    for i in range(self.n_layers):
      amax_layer = {"amax_xqkv": a["xqkv"][i], "amax_wqkv": a["wqkv"][i],
                    "amax_xo": a["xo"][i], "amax_wo": a["wo"][i],
                    "amax_x1": a["x1"][i], "amax_w1": a["w1"][i],
                    "amax_x2": a["x2"][i], "amax_w2": a["w2"][i],
                    "amax_x3": a["x3"][i], "amax_w3": a["w3"][i]} if a else {}
      h, *ret = self.run_layer(h, freqs_cis,
                               self.attention_norm[i], self.wqkv[i], self.wo[i],
                               self.ffn_norm[i], self.w1[i], self.w2[i], self.w3[i],
                               **amax_layer)
      if a:
        amaxs = ret[:10]
        amax_names = ["xqkv", "wqkv", "xo", "wo", "x1", "w1", "x3", "w3", "x2", "w2"]
        for name, new_val in zip(amax_names, amaxs):
          a[name][i].assign(new_val)

    logits = matmul(self.norm(h).contiguous().contiguous_backward(), self.output[0], fp8=False)[0].contiguous_backward()
    return logits

def _get_pads(uop:UOp) -> list[UOp]:
  if uop.op == Ops.ADD: return _get_pads(uop.src[0]) + _get_pads(uop.src[1])
  return [uop]

def apply_grad(grad_buf:Tensor, new_grad:UOp):
  pads = _get_pads(new_grad)
  new_grad = new_grad.cast(grad_buf.dtype)
  if len(pads) <= 1:
    store = grad_buf.uop.store(grad_buf.uop + new_grad)
    grad_buf.uop = grad_buf.uop.after(store)
    return
  sorted_pads = sorted(pads, key=lambda p: p.marg[0][0] if p.op == Ops.PAD else 0)
  inners = [Tensor(p.src[0] if p.op == Ops.PAD else p, device=grad_buf.device).cast(grad_buf.dtype) for p in sorted_pads]
  grad_buf.assign(grad_buf + inners[0].cat(*inners[1:], dim=0))

if __name__ == "__main__":
  config = {}
  BS                 = config["BS"]                     = getenv("BS", 16)
  SEQLEN             = config["SEQLEN"]                 = getenv("SEQLEN", 8192)

  from examples.llama3 import MODEL_PARAMS
  model_params = MODEL_PARAMS[getenv("LLAMA3_SIZE", "8B")]["args"]
  if (llama_layers:=getenv("LLAMA_LAYERS")) != 0: model_params['n_layers'] = llama_layers
  model = FlatTransformer(**model_params, max_context=SEQLEN)
  state = nn.state.get_state_dict(model)
  print("tensor count:", len(state))

  # shard the model
  from tinygrad import Device
  if (DP := getenv("DP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)))
  if (MP := getenv("MP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)), mp=True)

  # preallocate all the grad buffers and zero them out
  grads = {x:Tensor.zeros(x.shape, dtype=x.dtype, device=x.device).contiguous()
           for x in state.values() if x.requires_grad is None}

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {str(v.dtype):20s} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=model.vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if DP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)), axis=0)
  if MP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)))

  @TinyJit
  def jit_step(tokens:Tensor):
    with Timing("python forward: "): loss = model(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[t], g.uop)
    with Timing("run step: "): loss.realize(*grads.values())

  for i in range(6):
    GlobalCounters.reset()
    profile_marker(f"step {i}")
    with Timing(colored(f"*** step {i}: ", "red")):
      jit_step(tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
