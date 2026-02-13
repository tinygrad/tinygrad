from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, gc, pathlib
from tinygrad import Tensor, nn, UOp, TinyJit, getenv
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

from tinygrad.nn.rope import precompute_freqs_cis, apply_rope, load_yarn_params_from_gguf
from tinygrad.nn.mla import MLATransformerBlock, load_mla_params_from_gguf, split_mla_kv_weights
from tinygrad.nn.moe import ExpertWeights, finalize_moe_weights
from tinygrad.nn.quantized import replace_quantized_modules

# DeepSeek tokenizer patterns (used when preset="deepseek-llm")
_DEEPSEEK_LETTERS = (
  r"[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯ"
  r"Ա-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟὡὣὥὧὩὫὭὯὰάὲέὴήὶίὸόὺύὼώ"
  r"ᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐῑῒΐῖῗῘῙῚΊῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹ"
  r"ℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗ"
  r"Ａ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"
)
_DEEPSEEK_PUNCT = r"\s?[!-/:-~！-／：-～'-‟　-。]+"

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in (
      "llama3","llama-v3","llama-bpe","qwen2","olmo","glm4","deepseek-llm"
    ):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    if preset == "deepseek-llm":
      self._split_to_word = re.compile("|".join([
        r"[\r\n]",
        r"\s?" + _DEEPSEEK_LETTERS,
        _DEEPSEEK_PUNCT,
        r"\s+$",
        r"[一-龥ࠀ-一가-퟿]+",
        f"[{r_p_N}]+",
      ]))
    else:
      self._split_to_word = re.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}|"
        f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|"
        f"[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
      )
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self._bpe_ranks: dict[tuple[bytes, bytes], int] = {}
    self.preset = preset
    # Initialize tokenizer attributes with defaults (can be overridden by from_gguf_kv)
    self.add_bos_token = True
    self.add_eos_token = True
    self.add_space_prefix = False
    self.clean_spaces = False
    self.ignore_merges = False
    self.byte_fallback = False
    self.merges: list[str] = []

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    tok = SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"])
    for attr, default in [("add_bos_token", True), ("add_eos_token", True), ("add_space_prefix", False),
                          ("clean_spaces", False), ("ignore_merges", False), ("byte_fallback", False)]:
      setattr(tok, attr, kv.get(f"tokenizer.ggml.{attr}", default))
    tok.merges = kv.get("tokenizer.ggml.merges", [])
    if tok.merges:
      for idx, merge in enumerate(tok.merges):
        parts = merge.split()
        if len(parts) != 2: continue
        a = bytes(tok._byte_decoder[c] for c in parts[0])
        b = bytes(tok._byte_decoder[c] for c in parts[1])
        tok._bpe_ranks[(a, b)] = idx
    return tok

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # Merge by rank (BPE) or by token presence (legacy)
    while len(parts) > 1:
      best_score, best_i = sys.maxsize, -1
      for i in range(len(parts)-1):
        score = self._bpe_ranks.get((parts[i], parts[i+1]), sys.maxsize) if self._bpe_ranks else self._normal_tokens.get(parts[i]+parts[i+1], sys.maxsize)
        if score < best_score: best_score, best_i = score, i
      if best_i == -1: break
      parts[best_i:best_i+2] = [parts[best_i] + parts[best_i+1]]
    # Handle parts with optional byte fallback
    out: list[int] = []
    for p in parts:
      if (tid := self._normal_tokens.get(p)) is not None:
        out.append(tid)
      elif self.byte_fallback:
        for b in p:
          if (bt := self._normal_tokens.get(bytes([b]))) is None: raise RuntimeError("token not found")
          out.append(bt)
      else:
        raise RuntimeError("token not found")
    return out
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    if self.add_space_prefix and text and not text.startswith(" "): text = " " + text
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
    if self.preset == 'glm4': return self.encode("<|" + role + "|>\n")
    if self.preset == 'deepseek-llm': return self.encode(role.capitalize() + ": ")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'qwen2': return [eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []  # GLM4 doesn't use end turn tokens between messages
    if self.preset == 'deepseek-llm': return self.encode("\n\n")
    return [eos_id]
  def build_chat_ids(self, messages: list[dict], bos_id: int|None, eos_id: int, add_generation_prompt: bool=True) -> list[int]:
    ids: list[int] = [bos_id] if bos_id is not None else []
    if self.preset == 'glm4': ids += self.encode("<sop>")
    for msg in messages:
      ids += self.role(msg["role"])
      content = msg["content"]
      if isinstance(content, str): ids += self.encode(content)
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": ids += self.encode(c["text"])
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      else: raise RuntimeError(f"unknown content type: {type(content)}")
      ids += self.end_turn(eos_id)
    if add_generation_prompt:
      ids += self.role("assistant")
      if self.preset in ('glm4',): ids += self.encode("<think>")
    return ids

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm

    # --- attention projections (all linear, bias-free) ------------------
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
      self.num_experts_per_tok = num_experts_per_tok
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)  # router
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    else:
      self.ffn_gate    = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(hidden_dim, dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs_cis = precompute_freqs_cis(self.head_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.empty(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_output(attn)
    return x + attn

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)  # (B, T, k) each
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      return h + (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
    # TODO: remove the need for this contiguous
    gated  = self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0,
               # MLA parameters (for deepseek2 architecture)
               q_lora_rank:int=0, kv_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0,
               n_shared_experts:int=0, moe_hidden_dim:int=0, leading_dense_blocks:int=0,
               expert_gating_func:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0,
               mscale:float=1.0, yarn_scaling_factor:float=1.0, yarn_params=None):
    if kv_lora_rank > 0:  # MLA architecture (use when kv_lora_rank is present, q_lora_rank is optional)
      self.blk = []
      for i in range(num_blocks):
        # First leading_dense_blocks use dense FFN, rest use MoE
        is_dense = i < leading_dense_blocks
        blk = MLATransformerBlock(dim, hidden_dim, n_heads, norm_eps, max_context,
                                            q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                                            0 if is_dense else num_experts, num_experts_per_tok, n_shared_experts, moe_hidden_dim,
                                            expert_gating_func, expert_weights_norm, expert_weights_scale, mscale,
                                            rope_theta, yarn_scaling_factor, yarn_params)
        self.blk.append(blk)
    else:  # Standard attention
      self.blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, qk_norm,
                                   num_experts, num_experts_per_tok) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=True, quantized:bool=False) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    # TODO: ugly, remove variable returns
    if quantized: kv, state_dict, quantized_tensors = nn.state.gguf_load(gguf.to(None), quantized=True)
    else: kv, state_dict, quantized_tensors = nn.state.gguf_load(gguf.to(None)), None

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict and (quantized_tensors is None or 'output.weight' not in quantized_tensors):
      if 'token_embd.weight' in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']
      elif quantized_tensors and 'token_embd.weight' in quantized_tensors: quantized_tensors['output.weight'] = quantized_tensors['token_embd.weight']
    # Remap GGUF exp_probs_b tensors to tinygrad exp_probs_b.bias naming
    for d in [state_dict] + ([quantized_tensors] if quantized_tensors else []):
      for k in list(d.keys()):
        if re.match(r"blk\.\d+\.exp_probs_b$", k): d[f"{k}.bias"] = d.pop(k)

    # Extract architecture metadata
    arch = kv['general.architecture']
    if DEBUG >= 1: print(f"architecture: {arch}")
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in state_dict:
        if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    # Load architecture-specific parameters
    mla = load_mla_params_from_gguf(kv, arch)
    rope_theta = kv[f'{arch}.rope.freq_base']
    yarn_params, mscale, yarn_scaling_factor = load_yarn_params_from_gguf(kv, arch, rope_theta)
    qk_norm = int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0
    num_experts = kv.get(f'{arch}.expert_count', 0)

    # Create model with extracted parameters
    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
                        hidden_dim=kv.get(f'{arch}.feed_forward_length', kv.get(f'{arch}.expert_feed_forward_length', 0)),
                        n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
                        vocab_size=len(kv['tokenizer.ggml.tokens']),
                        head_dim=kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads),
                        rope_theta=rope_theta, max_context=max_context, qk_norm=qk_norm,
                        num_experts=num_experts, num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
                        mscale=mscale, yarn_scaling_factor=yarn_scaling_factor, yarn_params=yarn_params, **mla)

    # Apply quantization if requested
    if quantized_tensors:
      q_linear, q_expert, q_dequant = replace_quantized_modules(model, quantized_tensors, state_dict)
      if DEBUG >= 1: print(f"quantized replaced linear={q_linear} expert={q_expert}, dequantized={q_dequant}")

    # Split MLA KV weights if needed
    if mla['kv_lora_rank'] > 0:
      split_mla_kv_weights(state_dict, quantized_tensors, len(model.blk), n_heads, mla)

    # Load state dict and finalize
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False, strict=False)
    if quantized_tensors:
      del state_dict, quantized_tensors
      gc.collect()

    finalize_moe_weights(model.blk)

    params = nn.state.get_parameters(model)
    if DEBUG >= 1: print(f"total params: {len(params)}, total bytes: {sum(p.nbytes() for p in params)/1e9:.2f} GB")
    if not quantized:
      # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
      for s in params: s.replace(s.contiguous())
    if realize:
      for i in range(0, len(params), 50): Tensor.realize(*params[i:i+50])
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

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
  "glm-4.7:flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_0.gguf",
  "deepseek-v2-lite": "https://huggingface.co/zhentaoyu/DeepSeek-V2-Lite-Chat-Q4_0-GGUF/resolve/main/deepseek-v2-lite-chat-q4_0.gguf",
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
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'llama', messages: msgs, stream: true})});
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
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id in stop_tokens: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens
      ids = tok.build_chat_ids(body["messages"], bos_id, eos_id, add_generation_prompt=True)

      # reply
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--quantized", action="store_true", default=None, help="Keep weights quantized for lower memory (slower inference)")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run (non-interactive mode)")
  parser.add_argument("--raw-prompt", action="store_true", help="Use raw prompt (no chat formatting)")
  parser.add_argument("--count", type=int, default=1000, help="Max tokens to generate (with --prompt)")
  args = parser.parse_args()

  if args.quantized is None:
    args.quantized = any(args.model.startswith(x) for x in ["glm-", "deepseek-"])

  # load the model
  if DEBUG >= 1: print(f"loading {args.model} (quantized={args.quantized})")
  model_src = models.get(args.model, args.model)
  if isinstance(model_src, str) and model_src.startswith("http"):
    local = pathlib.Path("models") / pathlib.Path(model_src).name
    if local.is_file(): model_src = str(local.resolve())
  elif pathlib.Path(args.model).exists():
    model_src = str(pathlib.Path(args.model).resolve())
  model, kv = Transformer.from_gguf(Tensor.from_url(model_src), args.max_context, quantized=args.quantized)
  if DEBUG >= 1: print(f"using model {args.model}")

  # do benchmark
  if args.benchmark:
    param_bytes = sum(x.nbytes() for x in nn.state.get_parameters(model))
    gen = model.generate([0], 0)
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s, param {param_bytes/x:7.2f} GB/s"): next(gen)
    exit(0)

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']
  stop_tokens = [eos_id]

  if kv['general.architecture'] == 'deepseek2':  # GLM-4.7-Flash uses deepseek2 architecture
    user_tok = [tok.encode("<|user|>")[0]] if tok.encode("<|user|>") else []
    if user_tok: stop_tokens += user_tok

  if DEBUG >= 1: print(f"bos_id={bos_id}, eos_id={eos_id}, stop_tokens={stop_tokens}")

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  # Single prompt mode
  if args.prompt:
    if args.raw_prompt:
      # Raw prompt: just encode the text with BOS
      ids = ([bos_id] if bos_id is not None else []) + tok.encode(args.prompt)
    else:
      # Chat formatted prompt
      ids = tok.build_chat_ids([{"role":"user", "content": args.prompt}], bos_id, eos_id, add_generation_prompt=True)
    generated = 0
    for next_id in model.generate(ids, 0):
      sys.stdout.write(tok.decode([next_id]) if next_id not in stop_tokens else "\n")
      sys.stdout.flush()
      generated += 1
      if next_id in stop_tokens or generated >= args.count: break
    print()
    exit(0)

  # Interactive mode
  messages: list[dict] = []
  start_pos = 0  # Position for KV cache reuse
  while 1:
    try:
      messages.append({"role":"user", "content": input('>>> ')})
      ids = tok.build_chat_ids(messages, bos_id, eos_id, add_generation_prompt=True)
    except EOFError:
      break
    out_txt: list[str] = []
    for next_id in model.generate(ids, start_pos):
      if next_id in stop_tokens:
        sys.stdout.write("\n\n")
        break
      tok_txt = tok.decode([next_id])
      sys.stdout.write(tok_txt)
      out_txt.append(tok_txt)
      sys.stdout.flush()
    if out_txt:
      messages.append({"role":"assistant", "content": "".join(out_txt)})
      start_pos = len(ids)
