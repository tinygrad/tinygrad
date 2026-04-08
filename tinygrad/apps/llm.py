from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, itertools, urllib.request
from dataclasses import dataclass
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.uop.ops import resolve
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3", backend_tokenizer=None):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","gemma4"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {tok.encode(): tid for tok, tid in normal_tokens.items()} if preset == "gemma4" else {
      bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()
    }
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self._backend_tokenizer = backend_tokenizer

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(
      dict(normal_tokens), dict(special_tokens),
      kv.get("tokenizer.ggml.pre", kv.get("tokenizer.ggml.model", "llama3")), _get_backend_tokenizer(kv))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    if self._backend_tokenizer is not None: return self._backend_tokenizer.encode(text, add_special_tokens=False).ids
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str:
    if self._backend_tokenizer is not None: return self._backend_tokenizer.decode(ids)
    return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'gemma4': return self.encode("<|turn>" + ("model" if role == "assistant" else role) + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'qwen2': return [eos_id] + self.encode("\n")
    if self.preset == 'gemma4': return self.encode("<turn|>\n")
    return [eos_id]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

class ScaledLinear:
  def __init__(self, in_features:int, out_features:int):
    self.weight = Tensor.zeros(out_features, in_features)
    self.scale = Tensor.ones(in_features)
  def __call__(self, x:Tensor) -> Tensor: return (x * self.scale) @ self.weight.transpose(-1, -2)

class ScaledExpertWeights(ExpertWeights):
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    super().__init__(num_experts, in_features, out_features)
    self.scale = Tensor.ones(num_experts)

class ScalarWeight:
  def __init__(self): self.weight = Tensor.ones(1)

@functools.cache
def _get_gemma4_backend_tokenizer():
  try:
    from tokenizers import Tokenizer
    with urllib.request.urlopen("https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer.json") as r:
      return Tokenizer.from_str(r.read().decode())
  except Exception:
    return None

def _get_backend_tokenizer(kv:dict):
  return _get_gemma4_backend_tokenizer() if kv.get("tokenizer.ggml.model") == "gemma4" else None

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def rms_norm_no_weight(x:Tensor, eps:float) -> Tensor:
  return x * (x.square().mean(axis=-1, keepdim=True) + eps).rsqrt()

def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  return x if n_rep == 1 else x.unsqueeze(2).expand(
    x.shape[0], x.shape[1], n_rep, x.shape[2], x.shape[3]).reshape(x.shape[0], x.shape[1] * n_rep, x.shape[2], x.shape[3])

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = Tensor.zeros_like(x).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int|tuple[int, ...]
  n_heads: int
  n_kv_heads: int|tuple[int, ...]
  norm_eps: float
  vocab_size: int
  head_dim: int|tuple[int, ...]
  rope_theta: float|tuple[float, ...]
  max_context: int = 0
  qk_norm: int|tuple[int, ...] = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  sliding_window: int = 0
  sliding_window_pattern: tuple[bool, ...] = ()
  per_layer_input_dim: int = 0
  final_logit_softcap: float = 0.0
  num_kv_shared_layers: int = 0
  gemma4: bool = False
  expert_hidden_dim: int = 0
  moe_dense_branch: bool = False

class TransformerBlock:
  def __init__(self, config:TransformerConfig, layer_idx:int):
    self.config = config
    self.hidden_dim = config.hidden_dim[layer_idx] if isinstance(config.hidden_dim, tuple) else config.hidden_dim
    self.head_dim = config.head_dim[layer_idx] if isinstance(config.head_dim, tuple) else config.head_dim
    self.rope_theta = config.rope_theta[layer_idx] if isinstance(config.rope_theta, tuple) else config.rope_theta
    self.qk_norm = config.qk_norm[layer_idx] if isinstance(config.qk_norm, tuple) else config.qk_norm
    self.n_kv_heads = config.n_kv_heads[layer_idx] if isinstance(config.n_kv_heads, tuple) else config.n_kv_heads
    self.is_sliding = config.sliding_window > 0 and bool(config.sliding_window_pattern) and config.sliding_window_pattern[layer_idx]
    self.use_alternative_attention = config.moe_dense_branch and not self.is_sliding
    self.store_full_length_kv = False
    self.shared_kv_src_idx: int|None = None
    self.full_kv_cache: Tensor|None = None

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = self.head_dim * config.n_heads
    kv_proj_out      = self.head_dim * self.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=False)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=False)
    if not self.use_alternative_attention: self.attn_v = nn.Linear(config.dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(q_proj_out, config.dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    if self.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(self.qk_norm, config.norm_eps), nn.RMSNorm(self.qk_norm, config.norm_eps)
    if config.gemma4:
      self.post_attention_norm = nn.RMSNorm(config.dim, config.norm_eps)
      self.post_ffw_norm = nn.RMSNorm(config.dim, config.norm_eps)
      self.layer_output_scale = ScalarWeight()
    if config.per_layer_input_dim:
      self.inp_gate = nn.Linear(config.dim, config.per_layer_input_dim, bias=False)
      self.proj = nn.Linear(config.per_layer_input_dim, config.dim, bias=False)
      self.post_norm = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    self.ffn_gate_inp: nn.Linear|ScaledLinear
    self.ffn_down_exps: ExpertWeights|ScaledExpertWeights
    if config.moe_dense_branch or config.num_experts == 0:
      self.ffn_gate    = nn.Linear(config.dim, self.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, self.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(self.hidden_dim, config.dim, bias=False)
    if config.moe_dense_branch:
      self.ffn_gate_inp = ScaledLinear(config.dim, config.num_experts)
      self.ffn_gate_up_exps = ExpertWeights(config.num_experts, config.dim, config.expert_hidden_dim * 2)
      self.ffn_down_exps = ScaledExpertWeights(config.num_experts, config.expert_hidden_dim, config.dim)
      self.post_ffw_norm_1 = nn.RMSNorm(config.dim, config.norm_eps)
      self.pre_ffw_norm_2 = nn.RMSNorm(config.dim, config.norm_eps)
      self.post_ffw_norm_2 = nn.RMSNorm(config.dim, config.norm_eps)
    elif config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, self.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, self.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, self.hidden_dim, config.dim)

  def _attention(self, x:Tensor, start_pos:int|UOp, shared_kv_cache:Tensor|None=None) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k = self.attn_q(x_norm), self.attn_k(x_norm)
    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.config.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    if self.qk_norm == self.head_dim: q = self.attn_q_norm(q)
    q = apply_rope(q, self.freqs_cis[start_pos:start_pos+T])
    if shared_kv_cache is not None:
      k = shared_kv_cache[0, :, :, 0:start_pos+T, :]
      v = shared_kv_cache[1, :, :, 0:start_pos+T, :]
    else:
      raw_k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
      k = raw_k.transpose(1, 2)  # (B,KvH,T,Hd)
      if self.qk_norm == self.head_dim: k = self.attn_k_norm(k)
      raw_v = raw_k if self.use_alternative_attention else self.attn_v(x_norm).reshape(B, T, self.n_kv_heads, self.head_dim)
      v = rms_norm_no_weight(raw_v, self.config.norm_eps).transpose(1, 2)  # (B,KvH,T,Hd)
      k = apply_rope(k, self.freqs_cis[start_pos:start_pos+T])

      # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      if self.store_full_length_kv: self.full_kv_cache = assigned_kv
      k = assigned_kv[0, :, :, 0:start_pos+T, :]
      v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = None
    if resolve(T != 1) or self.is_sliding:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1)
      if self.is_sliding:
        mask = mask + Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).tril(start_pos-self.config.sliding_window)
    if self.config.gemma4:
      k, v = repeat_kv(k, self.config.n_heads // self.n_kv_heads), repeat_kv(v, self.config.n_heads // self.n_kv_heads)
      attn = ((q @ k.transpose(-1, -2)) + (mask if mask is not None else 0)).softmax(-1) @ v
    else:
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn)

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if self.config.moe_dense_branch:
      ffn_gate_inp = typing.cast(ScaledLinear, self.ffn_gate_inp)
      ffn_down_exps = typing.cast(ScaledExpertWeights, self.ffn_down_exps)
      dense = self.post_ffw_norm_1(self.ffn_down((self.ffn_gate(h_norm).gelu().contiguous()) * self.ffn_up(h_norm)))
      router_probs = (
        rms_norm_no_weight(h, self.config.norm_eps) * (self.config.dim ** -0.5) * ffn_gate_inp.scale
      ) @ ffn_gate_inp.weight.transpose(-1, -2)
      vals, sel = pairwise_topk(router_probs.softmax(-1), self.config.num_experts_per_tok)
      probs = vals / vals.sum(axis=-1, keepdim=True) * ffn_down_exps.scale[sel]
      x = self.pre_ffw_norm_2(h).unsqueeze(2)
      gate_up = self.ffn_gate_up_exps(sel, x)
      gate, up = gate_up.chunk(2, dim=-1)
      moe = self.post_ffw_norm_2((ffn_down_exps(sel, gate.gelu().contiguous() * up) * probs.unsqueeze(-1)).sum(axis=2))
      return dense + moe
    if self.config.num_experts > 0:
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(h_norm)
      vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
      probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      return (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
    # TODO: remove the need for this contiguous
    act = self.ffn_gate(h_norm).gelu() if self.config.gemma4 else self.ffn_gate(h_norm).silu()
    gated  = act.contiguous() * self.ffn_up(h_norm)
    return self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp, per_layer_input:Tensor|None=None, shared_kv_cache:Tensor|None=None):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.empty(2, x.shape[0], self.n_kv_heads, self.config.max_context, self.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.head_dim, self.config.max_context, self.rope_theta)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, per_layer_input:Tensor|None=None, shared_kv_cache:Tensor|None=None):
      attn = self._attention(x, start_pos, shared_kv_cache)
      h = x + (self.post_attention_norm(attn) if self.config.gemma4 else attn)
      ffn = self._feed_forward(h)
      h = h + (self.post_ffw_norm(ffn) if self.config.gemma4 else ffn)
      if per_layer_input is not None:
        h = h + self.post_norm(self.proj(self.inp_gate(h).gelu() * per_layer_input))
      if self.config.gemma4: h = h * self.layer_output_scale.weight
      return h.contiguous()
    return _run(x, start_pos, per_layer_input, shared_kv_cache)

class Transformer:
  def __init__(self, config:TransformerConfig):
    self.blk = [TransformerBlock(config, i) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    if config.per_layer_input_dim:
      self.per_layer_model_proj = nn.Linear(config.dim, config.num_blocks * config.per_layer_input_dim, bias=False)
      self.per_layer_proj_norm = nn.RMSNorm(config.per_layer_input_dim, config.norm_eps)
      self.per_layer_token_embd = nn.Embedding(config.vocab_size, config.num_blocks * config.per_layer_input_dim)
    self.max_context = config.max_context
    self.embed_scale = config.dim ** 0.5 if config.gemma4 else 1.0
    self.per_layer_embed_scale = config.per_layer_input_dim ** 0.5 if config.per_layer_input_dim else 1.0
    self.per_layer_input_scale = 2 ** -0.5
    self.per_layer_model_projection_scale = config.dim ** -0.5 if config.gemma4 else 1.0
    self.final_logit_softcap = config.final_logit_softcap
    if config.num_kv_shared_layers:
      first_shared = config.num_blocks - config.num_kv_shared_layers
      last_of_type = {False: max(i for i in range(first_shared) if not config.sliding_window_pattern[i]),
                      True: max(i for i in range(first_shared) if config.sliding_window_pattern[i])}
      for idx, block in enumerate(self.blk[:first_shared]):
        if bool(config.sliding_window_pattern) and idx == last_of_type[config.sliding_window_pattern[idx]]: block.store_full_length_kv = True
      for idx, block in enumerate(self.blk[first_shared:], start=first_shared):
        block.shared_kv_src_idx = last_of_type[config.sliding_window_pattern[idx]]
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float() * self.embed_scale                    # (B, T, D)
    per_layer_inputs = None
    if hasattr(self, 'per_layer_token_embd'):
      B, T, _ = x.shape
      per_layer_inputs = self.per_layer_proj_norm(
        (self.per_layer_model_proj(x) * self.per_layer_model_projection_scale).reshape(B, T, len(self.blk), -1))
      per_layer_inputs = (
        per_layer_inputs + self.per_layer_token_embd(tokens).float().reshape(B, T, len(self.blk), -1) * self.per_layer_embed_scale
      ) * self.per_layer_input_scale
    for i, block in enumerate(self.blk):
      shared_kv_cache = self.blk[block.shared_kv_src_idx].full_kv_cache if block.shared_kv_src_idx is not None else None
      x = block(x, start_pos, None if per_layer_inputs is None else per_layer_inputs[:,:,i,:], shared_kv_cache)
    logits = self.output(self.output_norm(x))[:, -1, :]
    if self.final_logit_softcap: logits = (logits / self.final_logit_softcap).tanh() * self.final_logit_softcap
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens, start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None).realize())

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in state_dict:
        if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    hidden_dim = tuple(kv[f'{arch}.feed_forward_length']) if isinstance(kv.get(f'{arch}.feed_forward_length'), list) else (
      kv[f'{arch}.feed_forward_length'] if arch == 'gemma4' else kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length']))
    if arch == 'gemma4':
      sliding_window_pattern = tuple(kv[f'{arch}.attention.sliding_window_pattern'])
      n_kv_heads = tuple(n_kv_heads) if isinstance(n_kv_heads, list) else n_kv_heads
      head_dim = tuple(kv[f'{arch}.attention.key_length_swa'] if is_sliding else kv[f'{arch}.attention.key_length']
                       for is_sliding in sliding_window_pattern)
      rope_theta = tuple(kv.get(f'{arch}.rope.freq_base_swa', kv[f'{arch}.rope.freq_base']) if is_sliding else kv[f'{arch}.rope.freq_base']
                         for is_sliding in sliding_window_pattern)
    else:
      sliding_window_pattern = ()
      head_dim = kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads)
      rope_theta = kv[f'{arch}.rope.freq_base']

    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
      hidden_dim=hidden_dim,
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim, rope_theta=rope_theta, max_context=max_context,
      qk_norm=tuple(head_dim) if arch == 'gemma4' else (
        int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0),
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=arch == 'qwen3moe',
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0), sliding_window_pattern=sliding_window_pattern,
      per_layer_input_dim=kv.get(f'{arch}.embedding_length_per_layer_input', 0),
      final_logit_softcap=kv.get(f'{arch}.final_logit_softcapping', 0.0), num_kv_shared_layers=kv.get(f'{arch}.attention.shared_kv_layers', 0),
      gemma4=arch == 'gemma4', expert_hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', 0),
      moe_dense_branch=arch == 'gemma4' and kv.get(f'{arch}.expert_count', 0) > 0)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]):
    return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor(temperature).contiguous()
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the kv cache
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "gemma4:e2b-q4": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
  "gemma4:26b-a4b-q4": "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
}

# *** simple OpenAI API compatible server with web interface on http://localhost:8000/ ***

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>tinygrad chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything" autofocus></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'llama', messages: msgs, stream: true, temperature: 0.7})});
    let buf = '';
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const ln of lines)
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":model_name,"object":"model"}]}).encode())
    else: self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0):
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = time.perf_counter()
    for next_id in model.generate(ids, temperature=temperature):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens, last assistant message is treated as prefill
      ids: list[int] = [bos_id] if bos_id is not None else []
      for i, msg in enumerate(body["messages"]):
        ids += tok.role(msg["role"])
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        if msg["role"] == "assistant" and i == len(body["messages"]) - 1: break
        ids += tok.end_turn(eos_id)
      else: ids += tok.role("assistant")

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, finish_reason = [], "stop"
        for c in chunks:
          if c["choices"] and c["choices"][0].get("delta", {}).get("content"): out.append(c["choices"][0]["delta"]["content"])
          if c["choices"] and c["choices"][0].get("finish_reason"): finish_reason = c["choices"][0]["finish_reason"]
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models.get(args.model, args.model))
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  print(f"using model \"{model_name}\" with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) or tok.preset == 'gemma4' else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # warmup the JIT
  if args.warmup or args.serve:
    # run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # interactive chat
  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
