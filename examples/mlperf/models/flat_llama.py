import math, os
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
from extra.llama_kernels.rmsnorm import rmsnorm
from extra.llama_kernels import FP8_MAX, local_abs_max

ASM_GEMM = getenv("ASM_GEMM", 0)
FUSED_INPUT_QUANTIZE = getenv("FUSED_INPUT_QUANTIZE", 0)
FUSED_ADD_NORM_MUL_QUANTIZE = getenv("FUSED_ADD_NORM_MUL_QUANTIZE", 0)
FUSED_SILU_W13 = getenv("FUSED_SILU_W13", 0)

FP8_DTYPE = dtypes.fp8e4m3
FP8_GRAD_DTYPE = dtypes.fp8e5m2

def quantize_fp8(x:Tensor, amax_state:Tensor|None=None):
  new_amax = (local_abs_max(x) if isinstance(x.device, tuple) else x.abs().max()).detach().cast(dtypes.float32)
  scale = FP8_MAX / ((amax_state if amax_state is not None else new_amax) + 1e-8)
  x_scaled = x * scale
  x_clamped = x_scaled + (x_scaled.detach().clamp(-FP8_MAX, FP8_MAX) - x_scaled.detach())  # STE
  return x_clamped.cast(FP8_DTYPE), scale.float().reciprocal(), new_amax

def matmul(x:Tensor, w:Tensor, fp8:bool=True, amax_x:Tensor|None=None, w_inv_scale:Tensor|None=None,
           x_fp8:Tensor|None=None, x_scale:Tensor|None=None, x_new_amax:Tensor|None=None,
           grad_amax_state:Tensor|None=None) -> tuple[Tensor,...]:
  if not fp8:
    if ASM_GEMM:
      from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
      if can_use_asm_gemm(x, w.T): return (asm_gemm(x, w.T),)
    return (x @ w.T,)
  assert w_inv_scale is not None, "fp8 matmul requires w_inv_scale (weights must be stored in fp8 with per-tensor scale)"
  if x_fp8 is None:
    if FUSED_INPUT_QUANTIZE and amax_x is not None:
      from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed
      x_fp8, x_scale, x_new_amax, _ = quantize_fp8_delayed(x, amax_x, FP8_DTYPE)
    else:
      x_fp8, x_scale, x_new_amax = quantize_fp8(x, amax_state=amax_x)
  if ASM_GEMM:
    from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
    if can_use_asm_gemm(x_fp8, w.T):
      return asm_gemm(x_fp8, w.T, x_scale=x_scale, w_scale=w_inv_scale, grad_amax_state=grad_amax_state), x_new_amax, x_fp8, w
  return x_fp8.dot(w.T, dtype=dtypes.float) * x_scale * w_inv_scale, x_new_amax, x_fp8, w

def norm_quantize_matmul(x:Tensor, norm:Tensor, w:Tensor, w_inv_scale:Tensor, eps:float, amax_x:Tensor, grad_amax_state:Tensor):
  if FUSED_ADD_NORM_MUL_QUANTIZE:
    from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import fused_rmsnorm_mul_quantize_fp8
    x_fp8, x_inv_scale, new_amax, x_normed, rrms = fused_rmsnorm_mul_quantize_fp8(x, norm, amax_x, eps, FP8_DTYPE)
    out, *ret = matmul(None, w, w_inv_scale=w_inv_scale, x_fp8=x_fp8, x_scale=x_inv_scale, x_new_amax=new_amax, grad_amax_state=grad_amax_state)
    return out, x_normed, rrms, ret
  x_normed, rrms = rmsnorm(x, eps)
  out, *ret = matmul(x_normed * norm, w, amax_x=amax_x, w_inv_scale=w_inv_scale, grad_amax_state=grad_amax_state)
  return out, x_normed, rrms, ret

def add_norm_quantize_matmul(x:Tensor, residual:Tensor, norm:Tensor, w:Tensor, w_inv_scale:Tensor, eps:float, amax_x:Tensor):
  if FUSED_ADD_NORM_MUL_QUANTIZE:
    from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import fused_add_rmsnorm_mul_quantize_fp8
    x_fp8, x_inv_scale, new_amax, h, x_normed, rrms = fused_add_rmsnorm_mul_quantize_fp8(x, residual, norm, amax_x, eps, FP8_DTYPE)
    out, *ret = matmul(None, w, w_inv_scale=w_inv_scale, x_fp8=x_fp8, x_scale=x_inv_scale, x_new_amax=new_amax)
    return out, h, x_normed, rrms, ret
  h = x + residual
  x_normed, rrms = rmsnorm(h, eps)
  out, *ret = matmul(x_normed * norm, w, amax_x=amax_x, w_inv_scale=w_inv_scale)
  return out, h, x_normed, rrms, ret

def silu_w13_quantize_matmul(x_w13:Tensor, w2:Tensor, s_2:Tensor,
                             amax_x2:Tensor,
                             grad_amax_xw13:Tensor, grad_amax_xout:Tensor):
  if FUSED_SILU_W13:
    from extra.llama_kernels.cast_amax import fused_quantize_fp8_w13
    x2_fp8, x2_inv_scale, new_amax_x2 = fused_quantize_fp8_w13(x_w13, amax_x2, FP8_DTYPE, grad_amax_state=grad_amax_xw13)
    out, *ret = matmul(None, w2, w_inv_scale=s_2, x_fp8=x2_fp8, x_scale=x2_inv_scale, x_new_amax=new_amax_x2, grad_amax_state=grad_amax_xout)
    return out, ret
  hidden = x_w13.shape[-1] // 2
  x_w1, x_w3 = x_w13[..., :hidden], x_w13[..., hidden:]
  out, *ret = matmul(x_w1.silu() * x_w3, w2, amax_x=amax_x2, w_inv_scale=s_2, grad_amax_state=grad_amax_xout)
  return out, ret

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024):
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.hidden_dim = hidden_dim

    scaled_std = 0.02 / math.sqrt(2 * n_layers)

    # Attention
    self._init_inv_scales = []  # populated by lin_per_layer
    self.wqkv = self.lin_per_layer(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2)
    self.wo = self.lin_per_layer(self.n_heads * self.head_dim, dim, std=scaled_std)

    # FeedForward
    self.w13 = self.lin_per_layer(dim, hidden_dim * 2)
    self.w2 = self.lin_per_layer(hidden_dim, dim, std=scaled_std)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(1, vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

    def _amax(): return Tensor.full((), FP8_MAX, dtype=dtypes.float32).contiguous().requires_grad_(False)
    names = ["xqkv", "xo", "x13", "x2"]
    self._fp8_amax = {name: [_amax() for _ in range(n_layers)] for name in names}
    grad_names = ["xqkv", "xo", "xw13", "xout"]
    self._fp8_grad_amax = {name: [_amax() for _ in range(n_layers)] for name in grad_names}
    w_names = ["wqkv", "wo", "w13", "w2"]
    self._fp8_inv_scale = {wname: inv_scales.float().contiguous().requires_grad_(False)
                           for wname, inv_scales in zip(w_names, self._init_inv_scales)}
    del self._init_inv_scales

  def lin_per_layer(self, in_features:int, out_features:int, std:float=0.02):
    if getenv("ZEROS"): w = Tensor.zeros(self.n_layers, out_features, in_features)
    else: w = Tensor.normal(self.n_layers, out_features, in_features, mean=0.0, std=std)
    amax = w.abs().flatten(1).max(1).detach()
    scale = FP8_MAX / (amax + 1e-8)
    self._init_inv_scales.append((amax + 1e-8) / FP8_MAX)
    return (w * scale.reshape(-1, 1, 1)).clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE)

  def attention(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                amax_xqkv:Tensor, amax_xo:Tensor, s_qkv:Tensor, s_o:Tensor,
                grad_amax_xqkv:Tensor, grad_amax_xo:Tensor):
    bsz, seqlen, _ = x.shape
    new_amaxs, saves = [], []

    xqkv, x_normed, rrms, ret = norm_quantize_matmul(x, attention_norm, wqkv, s_qkv, self.norm_eps,
                                                     amax_x=amax_xqkv, grad_amax_state=grad_amax_xqkv)
    saves.extend([x_normed, rrms])
    new_amaxs.extend(ret[:1])
    saves.extend(ret[1:] + [xqkv])
    xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xq, xk, xv = xq.cast(dtypes.bfloat16), xk.cast(dtypes.bfloat16), xv.cast(dtypes.bfloat16)
    if getenv("HK_FLASH_ATTENTION"):
      from extra.thunder.amd.fa import flash_attention
      attn, *save = flash_attention(xq, xk, xv, is_causal=True)
      saves.extend(save)
    else:
      xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
      attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)

    out, *ret = matmul(attn, wo, amax_x=amax_xo, w_inv_scale=s_o, grad_amax_state=grad_amax_xo)
    new_amaxs.extend(ret[:1])
    saves.extend(ret[1:] + [out])
    return (out, *new_amaxs, *saves)

  def feed_forward(self, x:Tensor, residual:Tensor, ffn_norm:Tensor, w13:Tensor, w2:Tensor,
                   amax_x13:Tensor, amax_x2:Tensor, s_13:Tensor, s_2:Tensor,
                   grad_amax_xw13:Tensor, grad_amax_xout:Tensor):
    new_amaxs, saves = [], []

    x_w13, h, x_normed, rrms, ret = add_norm_quantize_matmul(x, residual, ffn_norm, w13, s_13, self.norm_eps,
                                                             amax_x=amax_x13)
    saves.extend([x_normed, rrms])
    new_amaxs.extend(ret[:1])
    saves.extend(ret[1:] + [x_w13])

    out, ret = silu_w13_quantize_matmul(x_w13, w2, s_2, amax_x2=amax_x2, grad_amax_xw13=grad_amax_xw13, grad_amax_xout=grad_amax_xout)
    new_amaxs.extend(ret[:1])
    saves.extend(ret[1:] + [out])
    return (out, h, *new_amaxs, *saves)

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor,
                attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                ffn_norm:Tensor, w13:Tensor, w2:Tensor,
                amax_xqkv:Tensor, amax_xo:Tensor,
                amax_x13:Tensor, amax_x2:Tensor,
                s_qkv:Tensor, s_o:Tensor, s_13:Tensor, s_2:Tensor,
                grad_amax_xqkv:Tensor, grad_amax_xo:Tensor,
                grad_amax_xw13:Tensor, grad_amax_xout:Tensor):
    attn, *attn_ret = self.attention(x, freqs_cis, attention_norm, wqkv, wo,
                                     amax_xqkv=amax_xqkv, amax_xo=amax_xo, s_qkv=s_qkv, s_o=s_o,
                                     grad_amax_xqkv=grad_amax_xqkv, grad_amax_xo=grad_amax_xo)
    attn_amaxs, attn_saves = attn_ret[:2], attn_ret[2:]
    ffn, h, *ffn_ret = self.feed_forward(x, attn, ffn_norm, w13, w2,
                                                   amax_x13=amax_x13, amax_x2=amax_x2, s_13=s_13, s_2=s_2,
                                                   grad_amax_xw13=grad_amax_xw13, grad_amax_xout=grad_amax_xout)
    ffn_amaxs, ffn_saves = ffn_ret[:2], ffn_ret[2:]
    h = h + ffn
    return (h, *attn_amaxs, *ffn_amaxs, *attn_saves, *ffn_saves)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    from tinygrad.nn.state import get_parameters
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      self.wqkv.shard_(device, axis=1).realize()          # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2).realize()            # (n_layers, dim, in) shard in
      self.w13.shard_(device, axis=1).realize()           # (n_layers, hidden*2, dim) shard out
      self.w2.shard_(device, axis=2).realize()            # (n_layers, dim, hidden) shard in
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.shard_(device, axis=1).realize()
      self.freqs_cis.shard_(device, axis=None).realize()
      for amax_dict in (self._fp8_amax, self._fp8_grad_amax):
        for name in amax_dict:
          for i in range(len(amax_dict[name])):
            amax_dict[name][i] = amax_dict[name][i].to(device).contiguous().requires_grad_(False)
      for name in self._fp8_inv_scale:
        self._fp8_inv_scale[name] = self._fp8_inv_scale[name].to(device).contiguous().requires_grad_(False)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    a, ga, s = self._fp8_amax, self._fp8_grad_amax, self._fp8_inv_scale
    for i in range(self.n_layers):
      h, *ret = self.run_layer(h, freqs_cis,
                               self.attention_norm[i], self.wqkv[i], self.wo[i],
                               self.ffn_norm[i], self.w13[i], self.w2[i],
                               amax_xqkv=a["xqkv"][i], amax_xo=a["xo"][i],
                               amax_x13=a["x13"][i], amax_x2=a["x2"][i],
                               s_qkv=s["wqkv"][i], s_o=s["wo"][i],
                               s_13=s["w13"][i], s_2=s["w2"][i],
                               grad_amax_xqkv=ga["xqkv"][i], grad_amax_xo=ga["xo"][i],
                               grad_amax_xw13=ga["xw13"][i], grad_amax_xout=ga["xout"][i])
      for name, new_val in zip(["xqkv", "xo", "x13", "x2"], ret[:5]):
        a[name][i].assign(new_val)

    logits = matmul(self.norm(h), self.output[0], fp8=False)[0]
    return logits

def _get_pads(uop:UOp) -> list[UOp]:
  if uop.op == Ops.ADD: return _get_pads(uop.src[0]) + _get_pads(uop.src[1])
  return [uop]

def apply_grad(grad_buf:Tensor, new_grad:UOp):
  pads = _get_pads(new_grad)
  if len(pads) <= 1:
    new_grad = new_grad.cast(grad_buf.dtype)
    store = grad_buf.uop.store(grad_buf.uop + new_grad)
    grad_buf.uop = grad_buf.uop.after(store)
    return
  sorted_pads = sorted(pads, key=lambda p: p.marg[0][0] if p.op == Ops.PAD else 0)
  inners_raw = [Tensor(p.src[0] if p.op == Ops.PAD else p, device=grad_buf.device) for p in sorted_pads]
  if getenv("FUSED_PAD_GRAD_ACCUM", 0):
    from extra.llama_kernels.fused_pad_grad_accum import fused_pad_grad_accum, can_fused_pad_grad_accum
    if can_fused_pad_grad_accum(grad_buf, inners_raw):
      grad_buf.uop = fused_pad_grad_accum(grad_buf, inners_raw).uop
      return
  inners = [t.cast(grad_buf.dtype) for t in inners_raw]
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
