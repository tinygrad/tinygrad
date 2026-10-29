#!/usr/bin/env python3
# the EAGLE-3.1 draft of K3 (lightseekorg/kimi-k3-eagle3.1-mla): one MLA decoder layer on the target's hidden states of three layers.
# a position j takes the next token t[j+1] and the target's hidden states after processing t[j], and guesses t[j+2]. its MLA keys (the
# normed latent and the roped key) in a cache: positions go in one or a few at a time.
import json, math
from typing import Callable
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.llm.model import precompute_freqs_cis, apply_rope, YaRNConfig
from tinygrad.llm.kernels.amd import Linear, Q8_0, mla_decode_attention

def q8_0(w:np.ndarray) -> np.ndarray:
  """a (rows, cols) weight in ggml's q8_0 blocks (32 values: an f16 scale, 32 int8), the halfwords Linear takes"""
  b = w.astype(np.float32).reshape(-1, 32)
  d = np.abs(b).max(1) / 127
  blocks = np.empty(len(b), [("d", "<f2"), ("q", "i1", 32)])
  blocks["d"], blocks["q"] = d, np.round(b / np.where(d == 0, 1, d)[:, None])
  return blocks.view(np.uint16)

def q8_0_linear(w:np.ndarray, device:str|tuple[str, ...], axis:int|None=None) -> Linear:
  """a q8_0 Linear of w (out, in) on device, on several split like the trunk's: its outputs (axis 0), its inputs (axis 1) or not (None)"""
  lin = Linear(w.shape[1], w.shape[0], bias=False)
  if not isinstance(device, tuple) or axis is None: lin.weight = Tensor(q8_0(w)).to(device)
  else: lin.weight = Tensor(np.concatenate([q8_0(p) for p in np.split(w, len(device), axis=axis)])).shard(device, axis=0)
  lin.weight.realize()
  lin.ggml_type, lin.shard_axis = Q8_0, axis
  return lin

class Eagle3Draft:
  """embed: tokens -> their embeddings (the target's token_embd). on the target's gpus devs, the linears in q8_0 split over them (heads,
  the mlp's units, the head's vocab)"""
  def __init__(self, path:str, embed:Callable[[Tensor], Tensor], devs:tuple[str, ...], max_context:int=8192):
    cfg = json.load(open(f"{path}/config.json"))
    self.H, self.nope, self.rope, self.vd, self.lora = (cfg[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                                                                          "v_head_dim", "kv_lora_rank"))
    self.eps, self.embed = cfg["rms_norm_eps"], embed
    rs = cfg["rope_scaling"]
    yarn = YaRNConfig(factor=rs["factor"], orig_ctx_len=rs["original_max_position_embeddings"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"])
    # deepseek yarn: plain cos/sin (precompute_freqs_cis scales them by the concentration), the scores scaled by mscale^2 instead
    self.mscale = 0.1 * rs["mscale_all_dim"] * math.log(rs["factor"]) + 1.0
    self.freqs_cis = (precompute_freqs_cis(self.rope, max_context, cfg["rope_theta"], yarn=yarn) / self.mscale).to(devs).realize()
    st = nn.state.safe_load(f"{path}/model.safetensors")
    def w(name:str) -> np.ndarray:  # bf16 -> float32
      return (st[name].to("CPU").bitcast("uint16").numpy().astype(np.uint32) << 16).view(np.float32)
    def lin(name:str, axis:int|None) -> Linear: return q8_0_linear(w(name), devs, axis)
    p, a = "layers.0.", "layers.0.self_attn."
    self.norms = {k[:-len(".weight")]: Tensor(w(k)).to(devs).realize() for k in st if k.endswith("norm.weight") or k.startswith("fc_norm")}
    self.fc, self.o, self.down = lin("fc.weight", 1), lin(a + "o_proj.weight", 1), lin(p + "mlp.down_proj.weight", 1)
    self.q_a, self.kv_a = lin(a + "q_a_proj.weight", None), lin(a + "kv_a_proj_with_mqa.weight", None)
    self.q_b, self.gate, self.up = lin(a + "q_b_proj.weight", 0), lin(p + "mlp.gate_proj.weight", 0), lin(p + "mlp.up_proj.weight", 0)
    self.head = lin("lm_head.weight", 0)
    # kv_b by head: the up projections from the latent of the nope keys (absorbed into the queries) and of the values
    kv_b = w(a + "kv_b_proj.weight").reshape(self.H, self.nope + self.vd, self.lora)
    self.k_b, self.v_b = (Tensor(np.ascontiguousarray(x)).shard(devs, axis=0).realize() for x in (kv_b[:, :self.nope], kv_b[:, self.nope:]))
    self.cache = Tensor.empty(1, 1, max_context, self.lora + self.rope, dtype=dtypes.half, device=devs).realize()

  def norm(self, x:Tensor, name:str) -> Tensor: return x.float().rmsnorm(eps=self.eps) * self.norms[name]

  def __call__(self, tokens:Tensor, hidden:Tensor, start_pos:int|UOp=0, head:bool=True) -> Tensor:
    """(1, T) tokens at positions start_pos.., (1, T, 3, D) the target's hidden states -> (1, T, vocab) logits (not head: the states before
    the head, a prompt's chunk filling the cache)"""
    T, p, a = tokens.shape[1], "layers.0.", "layers.0.self_attn."
    hid = self.fc(Tensor.cat(*[self.norm(hidden[:, :, i], f"fc_norm.{i}") for i in range(hidden.shape[2])], dim=-1))
    h = self.norm(self.embed(tokens), p + "input_layernorm").cat(self.norm(hid, p + "hidden_norm"), dim=-1)
    q = self.q_b(self.norm(self.q_a(h), a + "q_a_layernorm")).reshape(1, T, self.H, self.nope + self.rope).transpose(1, 2)
    kv = self.kv_a(h)
    fc = self.freqs_cis[start_pos:start_pos + T]
    q_pe = apply_rope(q[..., self.nope:], fc, interleaved=True)
    k_pe = apply_rope(kv[..., self.lora:].reshape(1, 1, T, self.rope), fc, interleaved=True)
    store = self.norm(kv[..., :self.lora], a + "kv_a_layernorm").reshape(1, 1, T, self.lora).cat(k_pe, dim=-1)
    cache = Tensor(self.cache.uop.after(self.cache[:, :, start_pos:start_pos + T].uop.store(store.half().uop)))
    # absorbed: the nope scores (q_nope k_b) . latent, the values v_b (attention . latent), in the target's flash decoding kernels
    qa, scale = (q[..., :self.nope] @ self.k_b).cat(q_pe, dim=-1), self.mscale ** 2 / math.sqrt(self.nope + self.rope)
    if T <= 8: attn = mla_decode_attention(qa, cache, start_pos + T, self.lora, scale)
    else:  # (a prompt's chunk: causal)
      k = cache[:, :, :start_pos + T]
      mask = Tensor.full((1, 1, T, start_pos + T), float("-inf"), buffer=False).triu(start_pos + 1)
      attn = ((qa.float() @ k.float().transpose(-1, -2)) * scale + mask).softmax(-1) @ k[..., :self.lora].float()
    o = attn @ self.v_b.transpose(-1, -2)
    x = self.o(o.transpose(1, 2).reshape(1, T, self.H * self.vd)) + hid
    m = self.norm(x, p + "post_attention_layernorm")
    x = x + self.down(self.gate(m).silu() * self.up(m))
    return self.head(self.norm(x, "norm")) if head else x
