import math, os, functools
from dataclasses import replace
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  if "DEV" not in os.environ: os.environ["DEV"] = "NULL::gfx950"
  # CDNA
  os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"
  os.environ["ALL2ALL"] = "1"
  os.environ["USE_ATOMICS"] = "1"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters, profile_marker, TRAINING
from tinygrad.uop.ops import Ops, UOp
from extra.models.llama import apply_rotary_emb
from extra.llama_kernels.rmsnorm import rmsnorm
from extra.gptoss_kernels.embedding import GPTOSSEmbedding
from extra.llama_kernels import owned_empty
from extra.gemm.cdna_asm_gemm import _mx_block_scale, _mx_block_scale_3d, quantize_mxfp8
from extra.gemm.moe_gemm import grouped_mx_gemm, grouped_mx_gemm_swiglu, mx_pack_3d, FUSED_FC1_COLW
from extra.gemm.moe_routing import route, dispatch, dispatch_fp8, combine, router_mfma, router_quantize, Routing, BLOCK_ROW

def gptoss_layer_backward(grad:UOp, call:UOp) -> tuple:
  """Differentiate the layer's logical inputs, retaining DP-owned activation gradients.

  Only the first output is differentiable; the remaining outputs are forward checkpoints.
  Targeting a sharded input's flat PARAM instead would differentiate its UNSHARD,
  gathering an already-owned gradient before selecting the same local shard again.
  """
  from tinygrad.mixin.gradient import compute_gradient
  from tinygrad.function import renumber_invalid_outputs
  args = call.src[1:]
  stores = {st.src[0].unsharded_base.arg.slot:st for st in call.body.src}
  params = {p.arg.slot:p for p in call.body.toposort(enter_calls=False) if p.op is Ops.PARAM}
  # Replicated parameters can be read through several reshapes: retain their common flat target.
  inputs = {i:p.view_as(args[i].shard_shape, args[i].axis) if args[i].axis is not None else p
            for i,p in params.items() if i not in stores}
  root = next(iter(stores.values())).src[1]
  gradients = compute_gradient(root, grad.param_like(len(args)), set(inputs.values()))
  slots = [i for i,t in inputs.items() if t in gradients]
  body = UOp.sink(*(gradients[inputs[i]] if args[i].axis is not None else
                   gradients[inputs[i]].view_as(args[i].shard_shape, None) for i in slots))
  if call.arg.precompile:
    body = body.substitute({st.src[1]:st.src[0] for st in stores.values()}, walk=True)
    args = tuple(a.after(call) if i in stores else a for i,a in enumerate(args))
  args += (grad,)
  body = renumber_invalid_outputs(body)
  used = sorted((p for p in body.toposort(enter_calls=False) if p.op is Ops.PARAM), key=lambda p:p.arg.slot)
  body = body.substitute({p:p.replace(arg=replace(p.arg, slot=i)) for i,p in enumerate(used)}, walk=True)
  bound_args = tuple(args[p.arg.slot] for p in used)
  values = (gptoss_backward_outputs(body.src, bound_args, name=(call.arg.name or "")+"_backward", precompile=call.arg.precompile_backward)
            if getenv("GPTOSS_DEFER_FC1_REDUCE", 0) else
            UOp.call_with_outputs(body.src, *bound_args, name=(call.arg.name or "")+"_backward", precompile=call.arg.precompile_backward))
  outputs = dict(zip(slots, values))
  return tuple(outputs.get(i) for i in range(len(call.src)-1) if i not in stores)

def _return_storage(v:UOp) -> UOp:
  # Present the exact owned output, with producer dependencies outside its view.
  # An AFTER buried under UNSHARD otherwise prevents stock output forwarding.
  if v.storage_base.op is not Ops.ALLOC or not v.has_buffer_identity(after_ok=True): return v
  node, deps = v, []
  while node.op in (Ops.AFTER, Ops.RESHAPE, Ops.UNSHARD):
    if node.op is Ops.AFTER: deps.extend(node.src[1:])
    node = node.src[0]
  assert node is v.storage_base
  return node.view_as(v.shard_shape, v.axis).after(*deps)

def gptoss_backward_outputs(values:tuple[UOp, ...], args:tuple[UOp, ...], *, name:str, precompile:bool) -> tuple[UOp, ...]:
  """Keep FC1's collective outside the precompiled activation-backward boundary."""
  markers = [u for u in UOp.sink(*values).toposort(enter_calls=False)
             if u.op is Ops.CALL and u.arg.aux == "gptoss_fc1_reduce"]
  if not markers: return UOp.call_with_outputs(values, *args, name=name, precompile=precompile)
  frontier = list(dict.fromkeys(x for c in markers for x in c.src[1:-1]))
  deferred = [any(c in v.backward_slice_with_self for c in markers) for v in values]
  direct = [v for v,d in zip(values,deferred) if not d]
  outputs = UOp.call_with_outputs(tuple(direct+[_return_storage(v) for v in frontier]), *args, name=name, precompile=precompile)
  replacements = dict(zip(frontier,outputs[len(direct):]))
  normal = iter(outputs[:len(direct)])
  result = tuple(v.substitute(replacements, walk=True) if d else next(normal) for v,d in zip(values,deferred))
  # A deferred expression must be fully bound to caller buffers, never inner PARAMs.
  for v,d in zip(result,deferred):
    if d: assert not any(u.op is Ops.PARAM for u in v.toposort(enter_calls=False)), "unbound deferred gradient"
  return result

FP8_DTYPE = dtypes.fp8e4m3
FP8_MAX = 448.0
INIT_STD = 0.02
ASM_GEMM = getenv("ASM_GEMM", 0)
PRESTORE_WT = getenv("PRESTORE_WT", 0)

def _quant_dequant_fwd(x:Tensor) -> Tensor:
  # x (2d bf16) -> bf16 value after an mxfp8 round-trip (1x32 block scaling on the last axis)
  M, K = x.shape
  scale_K = K // 32
  amax = x.float().reshape(M, scale_K, 32).abs().max(axis=-1)
  e8 = (amax.maximum(1e-38).log2().floor() + 127).clamp(0, 254).cast(dtypes.uint8)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(M, scale_K, 1).expand(M, scale_K, 32).reshape(M, K)
  x_fp8 = (x.float() * qscale).clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE).cast(dtypes.float32)
  return (x_fp8 * _mx_block_scale(e8)).cast(dtypes.bfloat16)

@functools.cache
def _quant_dequant_fwd_fxn(x_p, device):
  return _quant_dequant_fwd(Tensor(x_p, device=device))

def _quant_dequant_bwd(grad:UOp, call:UOp) -> tuple:
  return (Tensor(grad).cast(dtypes.bfloat16).uop,)

def quant_dequant_mx(x:Tensor) -> Tensor:
  fxn = _quant_dequant_fwd_fxn(x.as_param(0).uop, x.device)
  return Tensor(fxn.uop.call_with_output(x.uop, grad_fxn=_quant_dequant_bwd))

def _mx_scale(e8:Tensor) -> Tensor:
  return _mx_block_scale(e8) if e8.ndim == 2 else _mx_block_scale_3d(e8)

def _dequant_fwd(w_q:Tensor, w_scale:Tensor) -> Tensor:
  return w_q.cast(dtypes.bfloat16) * _mx_scale(w_scale)

@functools.cache
def _dequant_fwd_fxn(wq_p, ws_p, device):
  return _dequant_fwd(Tensor(wq_p, device=device), Tensor(ws_p, device=device))

def _dequant_bwd(grad:UOp, call:UOp) -> tuple:
  return (Tensor(grad).cast(dtypes.bfloat16).uop, None)

def dequant_weight(w_q:Tensor, w_scale:Tensor) -> Tensor:
  fxn = _dequant_fwd_fxn(w_q.as_param(0).uop, w_scale.as_param(1).uop, w_q.device)
  return Tensor(fxn.uop.call_with_output(w_q.uop, w_scale.uop, grad_fxn=_dequant_bwd))

def matmul_mx(x:Tensor|tuple[Tensor, Tensor], w_q:Tensor, w_scale:Tensor) -> Tensor:
  if isinstance(x, tuple):
    assert ASM_GEMM, "pre-quantized MXFP8 input requires ASM_GEMM"
    from extra.gemm.cdna_asm_gemm import asm_gemm, can_use_asm_gemm, mx_pack
    x_q, x_e8 = x
    l_shape, padded = x_q.shape[:-1], x_q.shape[-1]
    x_q, x_e8 = x_q.reshape(-1, padded), x_e8.reshape(-1, padded // 32)
    K, N = w_q.shape[1], w_q.shape[0]
    assert padded >= K and (padded - K) % 32 == 0 and x_e8.shape[-1] == padded // 32
    wq, ws = w_q, w_scale
    if (pad := padded - K):
      wq = wq.pad(((0, 0), (0, pad)))
      ws = ws.pad(((0, 0), (0, pad // 32)), value=127).cast(dtypes.uint8)
    if (npad := (-N) % 256):
      wq = wq.pad(((0, npad), (0, 0)))
      ws = ws.pad(((0, npad), (0, 0)), value=127).cast(dtypes.uint8)
    assert can_use_asm_gemm(x_q, wq.T)
    out = asm_gemm(x_q, wq.T, mx=True, mx_scales=(mx_pack(x_e8), x_e8, mx_pack(ws), ws), mx_w_stored=True)
    return (out[:, :N] if npad else out).reshape(*l_shape, N).cast(dtypes.bfloat16)
  l_shape = x.shape[:-1]
  if ASM_GEMM:
    from extra.gemm.cdna_asm_gemm import asm_gemm, can_use_asm_gemm, mx_pack
    x2, K, N = x.reshape(-1, x.shape[-1]), x.shape[-1], w_q.shape[0]
    wq, ws = w_q, w_scale
    if (pad := (-K) % 256):
      x2 = x2.pad(((0, 0), (0, pad)))
      wq = wq.pad(((0, 0), (0, pad)))
      ws = ws.pad(((0, 0), (0, pad // 32)), value=127).cast(dtypes.uint8)
    if (npad := (-N) % 256):
      wq = wq.pad(((0, npad), (0, 0)))
      ws = ws.pad(((0, npad), (0, 0)), value=127).cast(dtypes.uint8)
    local_rows = x2.shape[0] // len(x2.device) if isinstance(x2.device, tuple) and x2.uop.axis == 0 else x2.shape[0]
    if getenv("FUSED_ATTN_QE8", 0) and (local_rows, x2.shape[1]) == (16384, 4096) and w_q.shape == (2880, 4096):
      from extra.gptoss_kernels.quantize_mxfp8 import quantize_mxfp8_fused_qe8
      x_q, x_e8 = quantize_mxfp8_fused_qe8(x2)
      x_si = mx_pack(x_e8)
    else: x_q, x_e8, x_si = quantize_mxfp8(x2)
    if x_si is not None and can_use_asm_gemm(x_q, wq.T):
      out = asm_gemm(x_q, wq.T, mx=True, mx_scales=(x_si, x_e8, mx_pack(ws), ws), mx_w_stored=True)
      return (out[:, :N] if npad else out).reshape(*l_shape, N).cast(dtypes.bfloat16)
  x_phys = quant_dequant_mx(x.reshape(-1, x.shape[-1])).reshape(*l_shape, x.shape[-1])
  w_phys = dequant_weight(w_q, w_scale)
  return (x_phys @ w_phys.T).cast(dtypes.bfloat16)

def dense_bias_add(x:Tensor, bias:Tensor) -> Tensor:
  if getenv("FUSED_DENSE_BIAS_GRAD", 0):
    from extra.llama_kernels.dense_bias import dense_bias_add as fused_dense_bias_add
    return fused_dense_bias_add(x, bias)
  return x + bias

def _pad_to_mult(t:Tensor, axis:int, mult:int=256) -> Tensor:
  if (r := (-t.shape[axis]) % mult) == 0: return t
  pads = [(0, 0)] * t.ndim
  pads[axis] = (0, r)
  return t.pad(tuple(pads))

def _pad_cols(t:Tensor) -> Tensor: return _pad_to_mult(t, -1)
def _pad_rows(t:Tensor) -> Tensor: return _pad_to_mult(t, -2)

def swiglu(x:Tensor, limit:float=7.0, alpha:float=1.702) -> Tensor:
  x_glu, x_linear = x[..., ::2], x[..., 1::2]
  x_glu = x_glu.clamp(max_=limit)
  x_linear = x_linear.clamp(-limit, limit)
  return (x_glu * (alpha * x_glu).sigmoid()) * (x_linear + 1)

def _moe_bias_tile(bias:Tensor, r:Routing) -> Tensor:
  tile_bias = r.tile_e.one_hot(bias.shape[0]).float() @ bias.float()
  return tile_bias.reshape(-1, 1, bias.shape[1]).expand(-1, BLOCK_ROW, -1).reshape(-1, bias.shape[1])

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float32)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end, dtype=dtypes.float32).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).cast(dtypes.default_float).reshape(1, end, 1, dim//2, 2)

class GPTOSS:
  def __init__(self, dim:int, n_layers:int, n_heads:int, n_kv_heads:int, head_dim:int, n_experts:int, experts_per_tok:int,
               intermediate_size:int, vocab_size:int, norm_eps:float=1e-5, rope_theta:int=150000, sliding_window:int=128,
               swiglu_limit:float=7.0, max_context:int=8192):
    self.dim, self.n_layers, self.n_heads, self.n_kv_heads, self.head_dim = dim, n_layers, n_heads, n_kv_heads, head_dim
    self.n_rep = n_heads // n_kv_heads
    self.n_experts, self.experts_per_tok, self.intermediate_size = n_experts, experts_per_tok, intermediate_size
    self.vocab_size, self.norm_eps, self.sliding_window, self.swiglu_limit = vocab_size, norm_eps, sliding_window, swiglu_limit
    self.sm_scale = 1.0 / math.sqrt(head_dim)

    scaled_std = INIT_STD / math.sqrt(2 * n_layers)
    q_dim, qkv_dim = n_heads * head_dim, head_dim * (n_heads + 2 * n_kv_heads)

    # attn
    self.wqkv, self.wqkv_scale = self._quant_weight(n_layers, qkv_dim, dim)
    self.wqkv_bias = Tensor.zeros(n_layers, qkv_dim, dtype=dtypes.bfloat16).contiguous()
    self.wo, self.wo_scale = self._quant_weight(n_layers, dim, q_dim, std=scaled_std)
    self.wo_bias = Tensor.zeros(n_layers, dim, dtype=dtypes.bfloat16).contiguous()
    self.sinks = Tensor.zeros(n_layers, n_heads, dtype=dtypes.bfloat16).contiguous()
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()

    # moe ffn
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()
    self.gate = Tensor.normal(n_layers, n_experts, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.gate_bias = Tensor.zeros(n_layers, n_experts, dtype=dtypes.bfloat16).contiguous()
    self.w_gate_up, self.w_gate_up_scale = self._quant_weight(n_layers, n_experts, intermediate_size * 2, dim, moe=True)
    self.w_gate_up_si = [mx_pack_3d(s).is_param_(False) for s in self.w_gate_up_scale] if getenv("PREPACK_FC1_WSI", 0) else None
    self.w_gate_up_bias = Tensor.zeros(n_layers, n_experts, intermediate_size * 2, dtype=dtypes.bfloat16).contiguous()
    self.w_down, self.w_down_scale = self._quant_weight(n_layers, n_experts, dim, intermediate_size, std=scaled_std, moe=True)
    self.w_down_bias = Tensor.zeros(n_layers, n_experts, dim, dtype=dtypes.bfloat16).contiguous()
    if PRESTORE_WT:
      self.w_gate_up_wT, self.w_gate_up_wT_scale = self._make_wT(self.w_gate_up, self.w_gate_up_scale)
      self.w_down_wT, self.w_down_wT_scale = self._make_wT(self.w_down, self.w_down_scale)

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = (GPTOSSEmbedding if getenv("GPTOSS_EMBEDDING", 0) else nn.Embedding)(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(vocab_size, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_context * 2, rope_theta).contiguous().is_param_(False)

  def _quant_weight(self, *shape:int, std:float=INIT_STD, moe:bool=False):
    def _one(*s:int):
      w = Tensor.zeros(*s) if getenv("ZEROS") else Tensor.normal(*s, mean=0.0, std=std)
      w_q, w_e8, _ = quantize_mxfp8(_pad_cols(_pad_rows(w)) if moe else w)
      return w_q, w_e8.is_param_(False)
    if moe:
      qs = [_one(*shape[1:]) for _ in range(shape[0])]
      ws = [q[0] for q in qs]
      for w in ws: w._zero2_moe = True   # mark: grad is reduce-scattered on the expert axis under ZeRO-2
      return ws, [q[1] for q in qs]
    return _one(*shape)

  def _make_wT(self, weights:list[Tensor], scales:list[Tensor]):
    wtq, wte = [], []
    for w_q, w_e8 in zip(weights, scales):
      wq, we, _ = quantize_mxfp8((w_q.cast(dtypes.bfloat16) * _mx_block_scale_3d(w_e8).cast(dtypes.bfloat16)).transpose(1, 2))
      wq, we = wq.is_param_(False), we.is_param_(False)
      wq._prestore_wT = we._prestore_wT = True
      wtq.append(wq); wte.append(we)
    return wtq, wte

  def _attn_mask(self, seqlen:int, dtype) -> Tensor:
    i, j = Tensor.arange(seqlen).reshape(seqlen, 1), Tensor.arange(seqlen).reshape(1, seqlen)
    return (j <= i).where(0.0, -1e30).cast(dtype).contiguous()

  def _sliding_attention(self, xq:Tensor, xk:Tensor, xv:Tensor, sinks:Tensor) -> Tensor:
    bsz, seqlen, H, hd = xq.shape
    KV, R, W = self.n_kv_heads, self.n_rep, self.sliding_window
    assert seqlen % W == 0, f"seqlen {seqlen} must be a multiple of sliding_window {W} for banded attention"
    nb = seqlen // W
    q = xq.reshape(bsz, seqlen, KV, R, hd).permute(0, 2, 3, 1, 4).reshape(bsz, KV, R, nb, W, hd).float()
    k, v = (x.permute(0, 2, 1, 3).reshape(bsz, KV, 1, nb, W, hd).float() for x in (xk, xv))
    kk, vv = (x.pad((None, None, None, (1, 0), None, None))[:, :, :, :nb].cat(x, dim=-2) for x in (k, v))
    sc = (q @ kk.transpose(-1, -2)) * self.sm_scale   # (B,KV,R,nb,W,2W)
    i, j, pv = Tensor.arange(W).reshape(W, 1), Tensor.arange(2 * W).reshape(1, 2 * W), Tensor.arange(nb).reshape(nb, 1, 1) >= 1
    sc = ((j > i) & (j <= i + W) & (pv | (j >= W))).where(sc, -float("inf"))
    sink = sinks.reshape(1, KV, R, 1, 1, 1).float()
    m = sc.max(-1, keepdim=True).maximum(sink)
    e = (sc - m).exp()
    p = (e / (e.sum(-1, keepdim=True) + (sink - m).exp())).cast(dtypes.bfloat16)
    attn = p @ vv.cast(dtypes.bfloat16)
    return attn.reshape(bsz, KV, R, seqlen, hd).permute(0, 3, 1, 2, 4).reshape(bsz, seqlen, H * hd)

  def attention(self, x:Tensor, freqs_cis:Tensor, mask:Tensor, sliding:bool, *, attention_norm:Tensor, wqkv:Tensor,
                wqkv_scale:Tensor, wqkv_bias:Tensor, wo:Tensor, wo_scale:Tensor, wo_bias:Tensor, sinks:Tensor,
                qkv_rope_out:tuple[Tensor, Tensor, Tensor]|None=None):
    bsz, seqlen, _ = x.shape
    if getenv("FUSED_RMSNORM_MX", 0):
      from extra.gptoss_kernels.rmsnorm import rmsnorm_mul_quantize_mxfp8
      x_q, x_e8, rrms = rmsnorm_mul_quantize_mxfp8(x, attention_norm, self.norm_eps)
      qkv = matmul_mx((x_q, x_e8), wqkv, wqkv_scale)
      norm_saves = [x_q, x_e8, rrms]
    elif getenv("FUSED_RMSNORM_MUL", 0):
      from extra.gptoss_kernels.rmsnorm import rmsnorm_mul
      x_normed, rrms = rmsnorm_mul(x, attention_norm, self.norm_eps)   # folds the *attention_norm into rmsnorm
      qkv = matmul_mx(x_normed, wqkv, wqkv_scale)
      norm_saves = [x_normed, rrms]
    else:
      x_normed, rrms = rmsnorm(x, self.norm_eps)
      qkv = matmul_mx(x_normed * attention_norm, wqkv, wqkv_scale)
      norm_saves = [x_normed, rrms]
    if getenv("FUSED_QKV_ROPE", 0):
      # one kernel fuses the qkv split + rope + bf16 cast + packed->per-head reindex (llama's fused_qkv_rope)
      from extra.thunder.amd.fa import fused_qkv_rope
      xq, xk, xv = fused_qkv_rope(qkv, freqs_cis, self.n_heads, self.n_kv_heads, self.head_dim, out=qkv_rope_out, bias=wqkv_bias)
    else:
      qkv = dense_bias_add(qkv, wqkv_bias)
      qkv = qkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = qkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
      xk, xv = qkv[:, :, :, self.n_rep], qkv[:, :, :, self.n_rep + 1]
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
      xq, xk, xv = xq.cast(dtypes.bfloat16), xk.cast(dtypes.bfloat16), xv.cast(dtypes.bfloat16)  # (B,N,H,D)/(B,N,KV,D)

    fa_saves = []  # save flash fwd inputs + LSE so the precompiled backward substitutes them instead of re-running custom_fa_forward + qkv/rope
    if getenv("HK_FLASH_ATTENTION") and (not sliding or getenv("HK_FLASH_SLIDING", 1)):
      # CAUSAL and SLIDING both use flash (window=0 -> pure causal). Sliding's peaked bf16 backward is corrected in
      # fa.py by recomputing the softmax LSE (_windowed_lse) and delta (_windowed_delta) in exact fp32; the window
      # mask itself lives in the fwd/bwd kernels (-DWINDOW).
      from extra.thunder.amd.fa import flash_attention
      attn, _, l_vec = flash_attention(xq, xk, xv, is_causal=True, write_flat=True, sinks=sinks,
                                       window=self.sliding_window if sliding else 0)
      attn = attn.reshape(bsz, seqlen, self.n_heads * self.head_dim)
      fa_saves = [xq, xk, xv, l_vec]  # exact ker.src inputs + l_vec output (must be un-wrapped or the fwd_subs key misses)
    elif sliding:
      attn = self._sliding_attention(xq, xk, xv, sinks)
    else:
      xqm = xq.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).permute(0, 2, 3, 1, 4)
      xkm, xvm = xk.permute(0, 2, 1, 3).unsqueeze(2), xv.permute(0, 2, 1, 3).unsqueeze(2)
      scores = (xqm @ xkm.transpose(-2, -1)).float() * self.sm_scale + mask
      sink = sinks.reshape(1, self.n_kv_heads, self.n_rep, 1, 1).float()
      m = scores.max(-1, keepdim=True).maximum(sink)
      e = (scores - m).exp()
      w = (e / (e.sum(-1, keepdim=True) + (sink - m).exp())).cast(dtypes.bfloat16)
      attn = (w @ xvm).permute(0, 3, 1, 2, 4).reshape(bsz, seqlen, self.n_heads * self.head_dim)

    out = dense_bias_add(matmul_mx(attn, wo, wo_scale), wo_bias)
    return out, [*norm_saves, attn] + fa_saves

  def feed_forward(self, x:Tensor, *, ffn_norm:Tensor, gate:Tensor, gate_bias:Tensor,
                   w_gate_up:Tensor, w_gate_up_scale:Tensor, w_gate_up_bias:Tensor,
                   w_down:Tensor, w_down_scale:Tensor, w_down_bias:Tensor,
                   w_gate_up_wT=None, w_gate_up_wT_scale=None, w_down_wT=None, w_down_wT_scale=None,
                   w_gate_up_si=None,
                   fc1_h_out:Tensor|None=None, dispatch_colw_out:tuple[Tensor, Tensor]|None=None,
                   router_topk_out:tuple[Tensor, Tensor]|None=None, normed_input:tuple[Tensor, Tensor]|None=None):
    # PRESTORE_WT: pass W^T alongside the fwd weight as a 4-tuple so the gemm wrapper registers it for the dgrad.
    if w_gate_up_wT is not None:
      w_gu = (w_gate_up, w_gate_up_scale, w_gate_up_wT, w_gate_up_wT_scale, w_gate_up_si) if w_gate_up_si is not None else \
             (w_gate_up, w_gate_up_scale, w_gate_up_wT, w_gate_up_wT_scale)
    else:
      w_gu = (w_gate_up, w_gate_up_scale, w_gate_up_si) if w_gate_up_si is not None else (w_gate_up, w_gate_up_scale)
    w_dn = (w_down, w_down_scale, w_down_wT, w_down_wT_scale) if w_down_wT is not None else (w_down, w_down_scale)
    if normed_input is not None:
      inp, rrms = normed_input
      x_normed = inp
    elif getenv("FUSED_RMSNORM_MUL", 0):
      from extra.gptoss_kernels.rmsnorm import rmsnorm_mul
      inp, rrms = rmsnorm_mul(x, ffn_norm, self.norm_eps)   # folds the *ffn_norm into rmsnorm
      x_normed = inp                                        # save the scaled value (backward via rmsnorm_mul grad_fxn)
    else:
      x_normed, rrms = rmsnorm(x, self.norm_eps)
      inp = x_normed * ffn_norm

    dim, inter = self.dim, self.intermediate_size
    grouped_moe = getenv("GROUPED_MOE", 0)
    # The grouped router consumes flat tokens. Make that the custom kernel's physical output shape too, so direct
    # saved-output slots do not acquire a shape-fixing materialization at the function boundary.
    if grouped_moe: inp = inp.reshape(-1, dim)
    dispatch_quantized = None
    if TRAINING and getenv("FUSED_ROUTER_DGRAD", 0):
      assert grouped_moe and getenv("ROUTER_MFMA", 0) and getenv("FP8_DISPATCH", 0)
      logits, q, e8 = router_quantize(inp, gate, gate_bias)
      dispatch_quantized = (q, e8)
    else:
      logits = router_mfma(inp, gate, gate_bias) if getenv("ROUTER_MFMA", 0) else \
        inp.float() @ gate.float().T + gate_bias.float()

    if grouped_moe:
      bsz, seqlen = x.shape[:2]
      r = route(logits, self.experts_per_tok, self.n_experts, router_topk_out)
      inp_pd = _pad_cols(inp.cast(dtypes.bfloat16))
      # FP8_DISPATCH: quantize before the permute so the scatter moves fp8+e8 (not bf16) and the fc1 reads the
      # already-quantized grouped rows (tuple path, no re-quantize). Byte-identical operands to the bf16 path.
      xg = dispatch_fp8(inp_pd, r, dispatch_colw_out, quantized=dispatch_quantized) if getenv("FP8_DISPATCH", 0) else dispatch(inp_pd, r)
      fused_fc1 = getenv("FUSED_FC1", 0)
      fp8_y_path = fused_fc1 or getenv("FUSED_SWIGLU", 0)
      # FUSED_DOWN_EPILOGUE folds the down-gemm bias + router-weighted combine scatter into the down-gemm epilogue
      # (requires the fp8 y path). It consumes the per-expert w_down_bias directly, so dn_bias is not materialized.
      fused_down = getenv("FUSED_DOWN_EPILOGUE", 0) and fp8_y_path
      # FUSED_DOWN_BIAS folds ONLY the per-expert down-gemm bias into the down-gemm epilogue (no combine, no quantize) --
      # a cheap per-column register add that deletes the separate `+dn_bias` bf16 kernel + its z round-trip. Combine stays
      # separate. It consumes w_down_bias directly, so dn_bias is not materialized.
      fused_down_bias = getenv("FUSED_DOWN_BIAS", 0) and not fused_down
      skip_dn_bias = fused_down or fused_down_bias
      if getenv("FUSED_MOE_BIAS", 0):
        dn_bias = None if skip_dn_bias else _moe_bias_tile(w_down_bias, r)
        gu_bias = None if fused_fc1 else _moe_bias_tile(w_gate_up_bias, r)
      else:
        onehot = r.rows_e.one_hot(self.n_experts).float() if not (skip_dn_bias and fused_fc1) else None
        dn_bias = None if skip_dn_bias else onehot @ w_down_bias.float()
        gu_bias = None if fused_fc1 else onehot @ w_gate_up_bias.float()
      if fused_fc1:
        # fused FC1: gate_up gemm + per-expert bias fold + clamped SwiGLU + mxfp8-quantize in ONE kernel (fwd). The
        # kernel ALSO emits h (pre-swiglu gate_up+bias). Save BOTH h AND y_e8: the precompiled backward reads the saved
        # h (grad_fxn -> _custom_swiglu_bwd_fp8) AND the saved y_e8 (the DOWN-gemm's bwd needs y_e8 via its xe_in). If
        # y_e8 is unsaved, that reference to the fused kernel's y_e8 OUTPUT keeps the fused CALL live and re-runs the
        # whole fused fwd in the backward (x48). Saving y_e8 lets @function substitute it -> fused CALL dead in the
        # backward -> fused fwd runs ONCE (x24).
        if FUSED_FC1_COLW:
          # ALSO emit + SAVE the pre-emitted COLUMNWISE mxfp8 of y (byte-exact with transpose_quantize(dequant(y_fp8)),
          # the down/FC2 wgrad's operand). SAVING y_colw/y_colw_e8 (like y_e8) lets @function substitute them so the
          # down-wgrad's reconstructed reference does NOT re-run the fused fwd (x24, not x48). See _fc1_colw_wgrad_operand.
          fc1_ret = grouped_mx_gemm_swiglu(xg, w_gu, r.off, w_gate_up_bias, real_inter=inter, out_h=fc1_h_out,
                                           padded_inter=w_down.shape[2], expert_counts=r.counts)
          h, y_fp8, y_e8_full, y_colw, y_colw_e8 = fc1_ret[:5]
          y_e8 = y_e8_full                                 # already padded to the down-gemm's physical K
          y, h_save = y_fp8, [h, y_e8_full, y_colw, y_colw_e8]
        else:
          h, y_fp8, y_e8_full = grouped_mx_gemm_swiglu(
            xg, w_gu, r.off, w_gate_up_bias, real_inter=inter, out_h=fc1_h_out, padded_inter=w_down.shape[2], expert_counts=r.counts)
          y_e8 = y_e8_full                                 # already padded to the down-gemm's physical K
          y, h_save = y_fp8, [h, y_e8_full]               # SAVE the FULL y_e8 -> grad_fxn reads the checkpoint (no recompute)
      else:
        h = grouped_mx_gemm(xg, w_gu, r.off)[:, :2*inter] + gu_bias.cast(dtypes.bfloat16)
        h_save = [h]
        if getenv("FUSED_SWIGLU", 0):
          from extra.llama_kernels.fused_swiglu_quantize_gptoss import fused_swiglu_quantize
          y_fp8, y_e8 = fused_swiglu_quantize(h)  # swiglu + mxfp8-quantize fused: h (M,2*inter) -> fp8 y (M,inter) + e8
          y = y_fp8                               # save the down-gemm fp8 input (bwd substitutes it; grad chains via the pre-quant path)
        else:
          y = swiglu(h, self.swiglu_limit)
      if fused_down:
        # ONE kernel: down gemm + per-expert bias + router-weighted combine scatter. z (byte-exact) is emitted for
        # the backward's d_weights=<d_out,z>; `out` is the fp32-atomic-scattered token output. Eliminates the separate
        # `+dn_bias` elementwise and the `combine` gather (z re-read).
        from extra.gemm.moe_gemm import grouped_mx_gemm_down_combine
        out_flat, z = grouped_mx_gemm_down_combine((y_fp8, y_e8), w_dn, w_down_bias, r,
                                                   inp.shape[0], self.experts_per_tok, real_out=dim)
        out = out_flat.reshape(bsz, seqlen, dim)
      else:
        if fused_down_bias:
          from extra.gemm.moe_gemm import grouped_mx_gemm_down_bias
          gemm_in = (y_fp8, y_e8) if fp8_y_path else _pad_cols(y.cast(dtypes.bfloat16))
          z = grouped_mx_gemm_down_bias(gemm_in, w_dn, r.off, w_down_bias, dim)
        elif fp8_y_path:
          z = grouped_mx_gemm((y_fp8, y_e8), w_dn, r.off)[:, :dim] + dn_bias.cast(dtypes.bfloat16)
        else:
          z = grouped_mx_gemm(_pad_cols(y.cast(dtypes.bfloat16)), w_dn, r.off)[:, :dim] \
              + dn_bias.cast(dtypes.bfloat16)
        if getenv("FUSED_COMBINE", 0):
          from extra.llama_kernels.fused_combine_gptoss import fused_combine
          out = fused_combine(z, r, inp.shape[0], self.experts_per_tok, real_dim=dim).reshape(bsz, seqlen, dim)
        else:
          if z.shape[-1] != dim: z = z[:, :dim]
          out = combine(z, r, inp.shape[0], self.experts_per_tok).reshape(bsz, seqlen, dim)
      # save the MoE fwd chain + routing metadata so the precompiled backward substitutes them (no route/dispatch/gemm/swiglu recompute)
      # Save every dispatch output, including the fresh column-layout buffers, for the precompiled backward.
      xg_save = list(xg) if isinstance(xg, tuple) else [xg]
      # FC1 wgrad consumes the exact per-expert counts from its forward custom call. Save them explicitly at the
      # precompiled-function boundary just like the other routing metadata: rebuilding counts in run_layer_backward
      # makes the one-hot reduction cross FUNCTION parameter substitution, where its int topi INDEX can acquire the
      # float comparison dtype of the router-logit path. The saved int32 buffer also avoids that redundant reduction.
      assert r.counts is not None
      # Match the buffer-backed identity passed to the fused FC1 call. Its backward mailboxes recover this exact
      # checkpoint instead of pulling the lazy routing reduction back across the precompiled-function boundary.
      return out, [x_normed, rrms, *xg_save, *h_save, y, z, r.weights, r.topi, r.dest_row, r.off, r.counts]
    else:
      logits = router_mfma(inp, gate, gate_bias) if getenv("ROUTER_MFMA", 0) else inp.float() @ gate.float().T + gate_bias.float()
      thresh = logits.topk(self.experts_per_tok)[0][..., -1:]
      weights = (logits >= thresh).where(logits, -float("inf")).softmax(-1)

      out = None
      for e in range(self.n_experts):
        gu_q, gu_s = w_gate_up[e][:2*inter, :dim].contiguous(), w_gate_up_scale[e][:2*inter, :dim//32].contiguous()
        dn_q, dn_s = w_down[e][:dim, :inter].contiguous(), w_down_scale[e][:dim, :inter//32].contiguous()
        gate_up = matmul_mx(inp, gu_q, gu_s) + w_gate_up_bias[e]
        y = (matmul_mx(swiglu(gate_up, self.swiglu_limit), dn_q, dn_s) + w_down_bias[e]).contiguous()
        contrib = weights[..., e:e+1].cast(y.dtype) * y
        out = contrib if out is None else out + contrib
    return out, [x_normed, rrms]

  @function(precompile=True, precompile_backward=True, grad_fxn=gptoss_layer_backward)
  def run_layer(self, x:Tensor, freqs_cis:Tensor, mask:Tensor, sliding:bool, attn_kwargs:dict, ffn_kwargs:dict, save:bool=True):
    # Own scratch inside this boundary so normal return binding saves it without caller-input aliases.
    if getenv("GPTOSS_QKV_ROPE_DIRECT_SAVE", 0):
      from extra.thunder.amd.fa import gptoss_qkv_rope_saved_outputs
      if (qkv_rope_out := gptoss_qkv_rope_saved_outputs(x, self.n_heads, self.n_kv_heads, self.head_dim, sliding=sliding, save=save)) is not None:
        attn_kwargs = {**attn_kwargs, "qkv_rope_out": qkv_rope_out}
    attn, attn_saves = self.attention(x, freqs_cis, mask, sliding, **attn_kwargs)
    h = x + attn
    bsz, seqlen = x.shape[:2]
    ffn_kwargs = dict(ffn_kwargs)
    if getenv("DIRECT_ROUTER_TOPK_OUT", 0) and getenv("FUSED_ROUTER_TOPK", 0) and getenv("GROUPED_MOE", 0):
      from extra.gemm.moe_routing import _sharded_invalids
      ffn_kwargs["router_topk_out"] = (
        owned_empty(_sharded_invalids((bsz * seqlen, self.experts_per_tok), dtypes.float32, h.device)),
        owned_empty(_sharded_invalids((bsz * seqlen, self.experts_per_tok), dtypes.int32, h.device)))
    if getenv("DIRECT_FC1_H_OUT", 0) and getenv("GROUPED_MOE", 0) and getenv("FUSED_FC1", 0):
      from extra.gemm.moe_routing import _sharded_invalids, m_max_for, n_groups_of
      G = n_groups_of(h)
      m_l = m_max_for((bsz * seqlen) // G, self.experts_per_tok, self.n_experts)
      Npad = ffn_kwargs['w_gate_up'].shape[1]
      ffn_kwargs["fc1_h_out"] = _sharded_invalids((G, m_l, Npad), dtypes.bfloat16, h.device).reshape(G * m_l, Npad).contiguous()
    saved_h = h
    if getenv("GPTOSS_RESIDUAL_RMSNORM_FWD", 0) and getenv("FUSED_RMSNORM_MUL", 0):
      from extra.gptoss_kernels.rmsnorm import gptoss_residual_rmsnorm_mul
      if (normed := gptoss_residual_rmsnorm_mul(h, ffn_kwargs['ffn_norm'], self.norm_eps)) is not None:
        ffn_kwargs['normed_input'], saved_h = normed[:2], normed[2]
    ffn, ffn_saves = self.feed_forward(h, **ffn_kwargs)
    # Preserve the exact norm input UOp so backward can reuse it without rematerializing WO.
    if save and getenv("GPTOSS_SAVE_FFN_INPUT", 0): ffn_saves.append(saved_h)
    if getenv("GPTOSS_RESIDUAL_HIP", 0) and x.shape[-1] == 2880 and x.dtype == dtypes.bfloat16:
      from extra.llama_kernels.gptoss_residual import gptoss_residual_join
      h = gptoss_residual_join(x, attn, ffn)
    else:
      h = h + ffn
    if save: return (h, *attn_saves, *ffn_saves)
    return (h,)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    assert not mp, "MP not supported"
    from tinygrad.nn.state import get_parameters
    for v in get_parameters(self): v.shard_(device, axis=None)
    Tensor.realize(*get_parameters(self))

  def __call__(self, tokens:Tensor, save:bool=True, targets:Tensor|None=None):
    h = self.tok_embeddings(tokens)
    bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :seqlen, :, :, :]
    mask_full = None if getenv("HK_FLASH_ATTENTION") else self._attn_mask(seqlen, dtypes.float32)
    for i in range(self.n_layers):
      attn_kwargs = dict(attention_norm=self.attention_norm[i], wqkv=self.wqkv[i], wqkv_scale=self.wqkv_scale[i],
                         wqkv_bias=self.wqkv_bias[i], wo=self.wo[i], wo_scale=self.wo_scale[i], wo_bias=self.wo_bias[i],
                         sinks=self.sinks[i])
      ffn_kwargs = dict(ffn_norm=self.ffn_norm[i], gate=self.gate[i], gate_bias=self.gate_bias[i],
                        w_gate_up=self.w_gate_up[i], w_gate_up_scale=self.w_gate_up_scale[i], w_gate_up_bias=self.w_gate_up_bias[i],
                        w_down=self.w_down[i], w_down_scale=self.w_down_scale[i], w_down_bias=self.w_down_bias[i])
      if self.w_gate_up_si is not None: ffn_kwargs["w_gate_up_si"] = self.w_gate_up_si[i]
      if PRESTORE_WT:
        ffn_kwargs.update(w_gate_up_wT=self.w_gate_up_wT[i], w_gate_up_wT_scale=self.w_gate_up_wT_scale[i],
                          w_down_wT=self.w_down_wT[i], w_down_wT_scale=self.w_down_wT_scale[i])
      h, *_ = self.run_layer(h, freqs_cis, mask_full, i % 2 == 0, attn_kwargs, ffn_kwargs, save=save)
      if i == 0 and TRAINING and getattr(self, '_deferred_lmhead', None) is not None:
        self._deferred_lmhead.prefetch_after(h)

    if getenv("FAST_FINAL_RMSNORM", 0):
      from extra.gptoss_kernels.rmsnorm import fast_final_rmsnorm
      h_normed = fast_final_rmsnorm(h, self.norm.weight, self.norm_eps)
    else: h_normed = self.norm(h)

    if targets is not None and getenv("FUSED_LINEAR_CE", 0):
      from extra.llama_kernels.fused_linear_ce import fused_linear_cross_entropy
      return fused_linear_cross_entropy(h_normed, self.output, targets)
    if getenv("FP8_LMHEAD", 0) and ASM_GEMM:
      # mxfp8 lm_head: quantize hidden + output on-the-fly, mxfp8 gemm (2x bf16 peak). w NOT stored fp8
      # (output stays a bf16 master param) -> mx_w_stored=False so the wgrad is d/d the physical bf16 weight.
      from extra.gemm.cdna_asm_gemm import asm_gemm, can_use_asm_gemm, quantize_mxfp8, mx_pack
      pad = (-self.dim) % 256
      h2 = h_normed.reshape(-1, self.dim).pad(((0, 0), (0, pad)))
      w2 = self.output.pad(((0, 0), (0, pad)))
      hq, he8, hsi = quantize_mxfp8(h2)
      oq, oe8, _ = quantize_mxfp8(w2)
      if hsi is not None and can_use_asm_gemm(hq, oq.T):
        logits = asm_gemm(hq, oq.T, mx=True, mx_scales=(hsi, he8, mx_pack(oe8), oe8), mx_w_stored=False)
        logits = logits.reshape(bsz, seqlen, self.vocab_size).cast(dtypes.bfloat16)
      else:
        logits = h_normed @ self.output.T
    elif ASM_GEMM:
      from extra.gemm.cdna_asm_gemm import asm_gemm, can_use_asm_gemm
      # pad hidden 2880 -> 3072 so fwd K%64, dgrad N%256 and wgrad M%256 all hold for the asm bf16 kernels;
      # backward through the pads is a shrink, so output grads still land in the (vocab, dim) buffer
      pad = (-self.dim) % 256
      h_padded, w_padded = h_normed.pad((None, None, (0, pad))), self.output.pad(((0, 0), (0, pad)))
      logits = asm_gemm(h_padded, w_padded.T) if can_use_asm_gemm(h_padded, w_padded.T) and getenv("VOCAB_ASM", 1) else h_normed @ self.output.T
    else:
      logits = h_normed @ self.output.T
    return logits

def _get_pads(uop:UOp) -> list[UOp]:
  if uop.op == Ops.ADD: return _get_pads(uop.src[0]) + _get_pads(uop.src[1])
  return [uop]

def apply_grad(grad_buf:Tensor, new_grad:UOp, accumulate:bool=True):
  if not accumulate and getenv("GPTOSS_DEFER_FC1_REDUCE", 0) and new_grad.shape == (32,5888,3072) and \
     new_grad.dtype == dtypes.bfloat16 and new_grad.axis == 0:
    new_grad = _return_storage(new_grad)
  pads = _get_pads(new_grad)
  if len(pads) <= 1:
    new_grad = new_grad.cast(grad_buf.dtype)
    if not accumulate and getenv("DIRECT_GRAD_ALIAS", 0):
      grad_buf.uop = new_grad
      return
    stored = grad_buf.uop + new_grad if accumulate else new_grad
    grad_buf.uop = grad_buf.uop.after(grad_buf.uop.store(stored))
    return

  if not accumulate:
    assert all(pad.op == Ops.PAD for pad in pads), "direct gradient writes require disjoint padded slices"
    regions = [tuple((p[0], s+p[0]) for s,p in zip(pad.src[0].shape, pad.marg)) for pad in pads]
    assert sum(math.prod(pad.src[0].shape) for pad in pads) == math.prod(grad_buf.shape), "direct gradient writes must cover the buffer"
    assert all(any(a[1] <= b[0] or b[1] <= a[0] for a,b in zip(regions[i], regions[j]))
               for i in range(len(regions)) for j in range(i)), "direct gradient writes must not overlap"

  cur = grad_buf.uop
  for pad in sorted(pads, key=lambda p: p.marg[0][0] if p.op == Ops.PAD else 0, reverse=True):
    if pad.op == Ops.PAD:
      grad_shrink = tuple((p[0], s+p[0]) for s,p in zip(pad.src[0].shape, pad.marg))
      buf_slice = cur.shrink(grad_shrink)
      stored = buf_slice + pad.src[0].cast(cur.dtype) if accumulate else pad.src[0].cast(cur.dtype)
      cur = cur.after(buf_slice.store(stored))
    else:
      cur = cur.after(cur.store(cur + pad.cast(cur.dtype)))
  grad_buf.uop = cur

GPT_OSS_20B = dict(dim=2880, n_layers=24, n_heads=64, n_kv_heads=8, head_dim=64, n_experts=32, experts_per_tok=4,
                   intermediate_size=2880, vocab_size=128256, norm_eps=1e-5, rope_theta=150000, sliding_window=128,
                   swiglu_limit=7.0)

if __name__ == "__main__":
  config = {}
  BS      = config["BS"]      = getenv("BS", 16)
  SEQLEN  = config["SEQLEN"]  = getenv("SEQLEN", 8192)

  model_params = GPT_OSS_20B
  real_vocab_size = model_params["vocab_size"]
  if (layers := getenv("LAYERS")) != 0: model_params["n_layers"] = layers

  model = GPTOSS(**model_params, max_context=SEQLEN)

  state = nn.state.get_state_dict(model)
  print("tensor count:", len(state))

  from tinygrad import Device
  is_dp = (DP := getenv("DP", 1)) > 1
  device_count = DP
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(device_count))

  if is_dp: model.shard(device)

  # preallocate all the grad buffers and zero them out
  grad_dtype = lambda x: dtypes.bfloat16 if x.dtype in dtypes.fp8s else x.dtype
  grads = {x:x.zeros_like(dtype=grad_dtype(x)).contiguous() for x in state.values() if x.is_param}

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {str(v.dtype):20s} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=real_vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if is_dp: tokens = tokens.shard(device, axis=0)

  @TinyJit
  def fwd_bwd(tokens:Tensor):
    with Timing("python forward: "):
      logits = model(tokens[:, :-1], save=True)
      loss = logits.sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[t], g.uop)
    with Timing("run fwd_bwd: "): loss.realize(*grads.values())

  @TinyJit
  def optim_step():
    for g in grads.values(): g.assign(g.zeros_like())
    Tensor.realize(*grads.values())

  for i in range(6):
    GlobalCounters.reset()
    profile_marker(f"step {i}")
    with Timing(colored(f"*** step {i}: ", "red")):
      fwd_bwd(tokens)
      optim_step()
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
