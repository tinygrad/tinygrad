from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, itertools
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.uop.ops import resolve
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","deepseek"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
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

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"])

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
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'deepseek': return self.encode("<|" + role + "|>\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'qwen2': return [eos_id] + self.encode("\n")
    if self.preset == 'deepseek': return [eos_id]
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

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, norm_topk_prob:bool=False,
               scoring_func:str="softmax", routed_scaling_factor:float=1.0, n_shared_experts:int=0, shared_expert_hidden_dim:int=0,
               kv_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm

    # --- MLA (Multi-head Latent Attention) config -------------------------
    self.kv_lora_rank      = kv_lora_rank
    self.qk_nope_head_dim  = qk_nope_head_dim
    self.qk_rope_head_dim  = qk_rope_head_dim
    self.v_head_dim        = v_head_dim

    # --- attention projections (all linear, bias-free) ------------------
    if kv_lora_rank > 0:
      # MLA: query projects to n_heads * (nope + rope), KV uses compressed latent
      q_head_dim = qk_nope_head_dim + qk_rope_head_dim
      self.attn_q      = nn.Linear(dim, n_heads * q_head_dim, bias=False)
      self.attn_kv_a_mqa = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
      self.attn_kv_a_norm = nn.RMSNorm(kv_lora_rank, norm_eps)
      self.attn_kv_b   = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False)
      self.attn_output = nn.Linear(n_heads * v_head_dim, dim, bias=False)
    else:
      q_proj_out       = self.head_dim * n_heads
      kv_proj_out      = self.head_dim * n_kv_heads
      self.attn_q      = nn.Linear(dim, q_proj_out,  bias=False)
      self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)
      self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
      self.attn_output = nn.Linear(q_proj_out, dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm    = nn.RMSNorm(dim, norm_eps)
    if qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if num_experts > 0:
      self.norm_topk_prob = norm_topk_prob
      self.num_experts_per_tok = num_experts_per_tok
      self.scoring_func = scoring_func
      self.routed_scaling_factor = routed_scaling_factor
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)  # router
      if scoring_func == "sigmoid": self.exp_probs_b = Tensor.zeros(num_experts)  # e_score_correction_bias for noaux_tc
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
      # shared experts (always active, separate dense MLP)
      if n_shared_experts > 0:
        self.ffn_gate_shexp = nn.Linear(dim, shared_expert_hidden_dim, bias=False)
        self.ffn_up_shexp   = nn.Linear(dim, shared_expert_hidden_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(shared_expert_hidden_dim, dim, bias=False)
    else:
      self.ffn_gate    = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(hidden_dim, dim, bias=False)

  @function
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    x_norm = self.attn_norm(x)                       # (B,T,D)

    if self.kv_lora_rank > 0:
      # --- MLA (Multi-head Latent Attention) path ---
      q = self.attn_q(x_norm)                        # (B,T, n_heads*(nope+rope))
      q = q.reshape(B, T, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2)  # (B,H,T,nope+rope)
      q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

      # compress KV into latent + rope key
      compressed_kv = self.attn_kv_a_mqa(x_norm)     # (B,T, kv_lora_rank + rope)
      compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
      k_pe = k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)   # (B,1,T,rope)

      # decompress latent to per-head K_nope and V
      kv = self.attn_kv_b(self.attn_kv_a_norm(compressed_kv))  # (B,T, n_heads*(nope+v))
      kv = kv.reshape(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)  # (B,H,T,nope+v)
      k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

      # apply RoPE only to the rope portion of Q and K
      q_pe = apply_rope(q_pe, self.freqs_cis[start_pos:start_pos+T])
      k_pe = apply_rope(k_pe, self.freqs_cis[start_pos:start_pos+T])

      # reassemble full Q and K: [nope, rope]
      q = q_nope.cat(q_pe, dim=-1)                   # (B,H,T,nope+rope)
      k = k_nope.cat(k_pe.expand(-1, self.n_heads, -1, -1), dim=-1)  # (B,H,T,nope+rope)

      # KV cache — K and V have different last dims, pad V to match K for stacking
      q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
      if self.v_head_dim < q_head_dim:
        v_padded = v.pad((0, q_head_dim - self.v_head_dim))  # pad last dim
      else:
        v_padded = v
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v_padded).uop)))
      k = assigned_kv[0, :, :, 0:start_pos+T, :]
      v = assigned_kv[1, :, :, 0:start_pos+T, :self.v_head_dim]  # unpad V

      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask)  # (B,H,T,v_head_dim)
      attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.v_head_dim)
    else:
      # --- standard GQA/MHA path ---
      q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
      if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

      q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
      k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
      v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
      if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

      q = apply_rope(q, self.freqs_cis[start_pos:start_pos+T])
      k = apply_rope(k, self.freqs_cis[start_pos:start_pos+T])

      # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      k = assigned_kv[0, :, :, 0:start_pos+T, :]
      v = assigned_kv[1, :, :, 0:start_pos+T, :]

      # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
      # TODO: this if statement should be removed and it shouldn't generate extra kernels
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)

    attn = self.attn_output(attn)
    return x + attn

  @function
  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      if self.scoring_func == "sigmoid":
        scores = self.ffn_gate_inp(h_norm).sigmoid()  # (B, T, num_experts)
        # noaux_tc: add learned bias for top-k selection, then select top-k from original scores
        scores_for_choice = scores + self.exp_probs_b.reshape(1, 1, -1)
        _, sel = scores_for_choice.topk(self.num_experts_per_tok)  # (B, T, k)
        probs = scores.gather(2, sel)  # (B, T, k) - gather original scores at selected indices
      else:
        probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)  # (B, T, k) each
      if self.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      # add shared expert output if present
      if hasattr(self, 'ffn_gate_shexp'):
        moe_out = moe_out + self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
      return h + moe_out
    # TODO: remove the need for this contiguous
    gated  = self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    if not hasattr(self, "cache_kv"):
      if self.kv_lora_rank > 0:
        # MLA: K cache stores full decompressed K (nope+rope), V cache stores full decompressed V
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.cache_kv = Tensor.empty(2, x.shape[0], self.n_heads, self.max_context, max(q_head_dim, self.v_head_dim), device=x.device)
        self.freqs_cis = precompute_freqs_cis(self.qk_rope_head_dim, self.max_context, self.rope_theta)
      else:
        # TODO: how is the dtype of this determined?
        self.cache_kv = Tensor.empty(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device)
        self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_context, self.rope_theta)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp): return self._feed_forward(self._attention(x, start_pos)).contiguous()
    return _run(x, start_pos)

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, norm_topk_prob:bool=False,
               scoring_func:str="softmax", routed_scaling_factor:float=1.0, n_shared_experts:int=0, shared_expert_hidden_dim:int=0,
               first_k_dense_replace:int=0, dense_hidden_dim:int=0,
               kv_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0):
    def make_block(i):
      is_moe = num_experts > 0 and i >= first_k_dense_replace
      return TransformerBlock(dim, hidden_dim if is_moe else (dense_hidden_dim or hidden_dim), n_heads, n_kv_heads, norm_eps,
                              head_dim, rope_theta, max_context, qk_norm,
                              num_experts if is_moe else 0, num_experts_per_tok if is_moe else 0, norm_topk_prob,
                              scoring_func if is_moe else "softmax", routed_scaling_factor if is_moe else 1.0,
                              n_shared_experts if is_moe else 0, shared_expert_hidden_dim if is_moe else 0,
                              kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    self.blk = [make_block(i) for i in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    logits = self.output(self.output_norm(x))[:, -1, :]
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

    # MLA-specific GGUF metadata (deepseek2 architecture)
    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    qk_rope_head_dim = kv.get(f'{arch}.rope.dimension_count', 0) if kv_lora_rank else 0
    # key_length_mla = qk_nope_head_dim + qk_rope_head_dim
    qk_nope_head_dim = kv.get(f'{arch}.attention.key_length_mla', 0) - qk_rope_head_dim if kv_lora_rank else 0
    v_head_dim = kv.get(f'{arch}.attention.value_length_mla', 0) if kv_lora_rank else 0
    # deepseek2 MoE config
    scoring_func = "sigmoid" if 'blk.1.exp_probs_b.bias' in state_dict else "softmax"
    routed_scaling_factor = kv.get(f'{arch}.expert_weights_scale', 1.0)
    n_shared_experts = kv.get(f'{arch}.expert_shared_count', 0)
    expert_ff_len = kv.get(f'{arch}.expert_feed_forward_length', 0)
    shared_expert_hidden_dim = expert_ff_len * n_shared_experts if n_shared_experts else 0
    first_k_dense_replace = kv.get(f'{arch}.leading_dense_block_count', 0)

    # for MLA, GGUF stores kv_b as a single tensor; we need to handle that during weight loading
    # llama.cpp conversion may split kv_b into attn_k_b + attn_v_b, or keep as attn_kv_b
    # reassemble split k_b/v_b into single kv_b for our model if needed
    if kv_lora_rank:
      for name in list(state_dict.keys()):
        if 'attn_k_b.weight' in name:
          prefix = name.replace('attn_k_b.weight', '')
          k_b = state_dict.pop(name)                 # (nope, kv_lora_rank, n_heads) in GGUF 3D
          v_b = state_dict.pop(f'{prefix}attn_v_b.weight')  # (kv_lora_rank, v_head_dim, n_heads)
          # k_b was transposed during conversion: (qk_nope_head_dim, kv_lora_rank, n_heads)
          # undo transpose: -> (kv_lora_rank, qk_nope_head_dim, n_heads) -> reshape to 2D Linear weight
          k_b_2d = k_b.permute(2, 1, 0).reshape(-1, kv_lora_rank)   # (n_heads*qk_nope_head_dim, kv_lora_rank)
          v_b_2d = v_b.permute(2, 1, 0).reshape(-1, kv_lora_rank)   # (n_heads*v_head_dim, kv_lora_rank)
          state_dict[f'{prefix}attn_kv_b.weight'] = k_b_2d.cat(v_b_2d, dim=0)  # (n_heads*(nope+v), kv_lora_rank)

    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
                        hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length']),
                        n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
                        vocab_size=len(kv['tokenizer.ggml.tokens']),
                        head_dim=kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads),
                        rope_theta=kv[f'{arch}.rope.freq_base'], max_context=max_context,
                        qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
                        num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
                        norm_topk_prob=True if arch in ('qwen3moe', 'deepseek2') else False,
                        scoring_func=scoring_func, routed_scaling_factor=routed_scaling_factor,
                        n_shared_experts=n_shared_experts, shared_expert_hidden_dim=shared_expert_hidden_dim,
                        first_k_dense_replace=first_k_dense_replace,
                        dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0),
                        kv_lora_rank=kv_lora_rank, qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim)
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
      self._cached_tokens = tokens[:]
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
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
}

# *** simple OpenAI compatible server on 11434 to match ollama ***
# OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama uvx --from gpt-command-line gpt

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
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
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
      # extract tokens
      ids: list[int] = [bos_id] if bos_id is not None else []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        # content can be a str or a list
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        ids += tok.end_turn(eos_id)
      ids += tok.role("assistant")

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
  parser.add_argument("--model", "-m", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models[args.model])
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  if DEBUG >= 1 or args.benchmark:
    print(f"using model {args.model} with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  # extract some metadata
  model_name = args.model
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # do benchmark
  if args.benchmark:
    gen = model.generate(toks:=[bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # start server
  if args.serve:
    # warmup: run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))
    TCPServerWithReuse(('', args.serve), Handler).serve_forever()

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
