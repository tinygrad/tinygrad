from tinygrad import Tensor, nn, UOp, getenv

class SimpleLlamaTokenizer:
  def __init__(self, vocab: list[str]):
    self.vocab: list[str] = vocab
    self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
    self.add_prefix_space = "Ġ"

  def encode(self, text:str) -> list[int]:
    spm_str = text.replace(" ", self.add_prefix_space)

    out: list[int] = []
    i = 0
    while i < len(spm_str):
      for j in range(len(spm_str), i, -1):
        tid = self.token_to_id.get(spm_str[i:j])
        if tid is not None:
          out.append(tid)
          i = j
          break
      else: raise RuntimeError("unmatched token")
    return out

  def decode(self, ids: list[int]) -> str:
    ret = ''.join(self.vocab[tid] for tid in ids)
    ret = ret.replace(self.add_prefix_space, " ")
    ret = ret.replace("Ċ", "\n")
    return ret

# --------------------------------------------------------------------------- #
# rotary-embedding helpers
# --------------------------------------------------------------------------- #

def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000):
  half_dim = head_dim // 2
  freq = base ** (-Tensor.arange(0, half_dim, dtype='float32') / half_dim)
  t = Tensor.arange(seq_len, dtype='float32')[:, None]          # (T, 1)
  angles = t * freq                                             # (T, Hd/2)
  return Tensor.stack(angles.cos(), angles.sin(), dim=-1)     # (T, Hd/2, 2)

def apply_rope(x:Tensor, freqs_cis: Tensor, start_pos: int | Tensor):
  B, H, T, Hd = x.shape
  cos = freqs_cis[None, None, start_pos:start_pos+T, :, 0]     # (T, half)
  sin = freqs_cis[None, None, start_pos:start_pos+T, :, 1]     # (T, half)
  x = x.reshape(B, H, T, Hd // 2, 2)    # split into pairs
  y1 = x[..., 0] * cos - x[..., 1] * sin
  y2 = x[..., 0] * sin + x[..., 1] * cos
  return Tensor.stack(y1, y2, dim=-1).reshape(B, H, T, Hd)

class TransformerBlock:
  def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int = 0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = dim // n_heads
    self.max_context  = max_context

    # --- attention projections (all linear, bias-free) ------------------
    kv_proj_out      = self.head_dim * n_kv_heads    # Llama-3 uses the same dim for K/V
    self.attn_q      = nn.Linear(dim, dim,            bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out,    bias=False)
    self.attn_v      = nn.Linear(dim, kv_proj_out,    bias=False)
    self.attn_output = nn.Linear(dim, dim,            bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm  = nn.RMSNorm(dim, norm_eps)

    # --- feed-forward ----------------------------------------------------
    self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up   = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

  # ------------------------------------------------------------------ #
  # helpers
  # ------------------------------------------------------------------ #
  def _attention(self, x: Tensor, start_pos: int|UOp, freqs_cis: Tensor, mask: Tensor|None) -> Tensor:
    """
    RMS-norm → QKV proj → RoPE → SDPA → output proj
    Returns the *residual-added* tensor (x + attn_out).
    """
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)

    q = apply_rope(q, freqs_cis, start_pos)
    k = apply_rope(k, freqs_cis, start_pos)

    if self.max_context:
      if not hasattr(self, "cache_kv"):
        self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=x.dtype).contiguous().realize()
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
      k = self.cache_kv[0, :, :, 0:start_pos+T, :]
      v = self.cache_kv[1, :, :, 0:start_pos+T, :]
    else:
      assert start_pos == 0

    if self.n_heads != self.n_kv_heads:               # MQA replication
      rep = self.n_heads // self.n_kv_heads
      k = k.repeat_interleave(rep, dim=1)
      v = v.repeat_interleave(rep, dim=1)

    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                   # back to (B,T,D)
    attn = self.attn_output(attn)
    return x + attn                                   # residual-add

  def _feed_forward(self, h: Tensor) -> Tensor:
    """
    RMS-norm → gated SiLU MLP → residual add.
    Accepts and returns shape (B,T,D).
    """
    h_norm = self.ffn_norm(h)
    gated  = self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm)
    return (h + self.ffn_down(gated)).contiguous()

  def __call__(self, x: Tensor, start_pos: int|UOp, freqs_cis: Tensor, mask: Tensor|None):
    return self._feed_forward(self._attention(x, start_pos, freqs_cis, mask))

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, max_context):
    self.blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.max_context, self.head_dim = max_context, dim // n_heads

  def __call__(self, tokens: Tensor, start_pos: int|UOp = 0):
    #print(tokens.tolist(), start_pos)
    B, T = tokens.shape
    if not hasattr(self, '_rope_cache'):
      self._rope_cache = build_rope_cache(self.max_context, self.head_dim)  # pre-compute the base RoPE table once
    x = self.token_embd(tokens)                           # (B, T, D)
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    for block in self.blk: x = block(x, start_pos, self._rope_cache, mask)
    return (self.output_norm(x) @ self.token_embd.weight.T)[:, -1, :].softmax(-1).argmax().item()

if __name__ == "__main__":
  gguf_tensor = Tensor.from_url("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf")
  kv, state_dict = nn.state.gguf_load(gguf_tensor.to(None))
  tok = SimpleLlamaTokenizer(kv["tokenizer.ggml.tokens"])

  model = Transformer(num_blocks=kv['llama.block_count'], dim=kv['llama.embedding_length'], hidden_dim=kv['llama.feed_forward_length'],
                      n_heads=kv['llama.attention.head_count'], n_kv_heads=kv['llama.attention.head_count_kv'],
                      norm_eps=kv['llama.attention.layer_norm_rms_epsilon'],
                      vocab_size=kv['llama.vocab_size'], max_context=kv['llama.context_length'])
  nn.state.load_state_dict(model, state_dict, consume=True, realize=False)

  # rope_freqs.weight (32,) is unused?
  #for k,v in state_dict.items(): print(k, v.shape)

  bos_id = kv.get("tokenizer.ggml.bos_token_id")
  eos_id = kv.get("tokenizer.ggml.eos_token_id")

  prompt_ids = [bos_id] + tok.encode("What's the sqrt of 4? Tell a story slowly meandering to the answer.")

  max_new_tokens = 256
  ids = prompt_ids.copy()

  start_pos = 0
  v_start_pos = UOp.variable("start_pos", 1, model.max_context-1)
  while len(ids) < model.max_context and max_new_tokens > 0:
    next_id = model(Tensor([ids[start_pos:]], dtype="int32"), v_start_pos.bind(start_pos) if getenv("SYM") and start_pos != 0 else start_pos)
    ids.append(next_id)

    start_pos = len(ids) - 1
    max_new_tokens -= 1

    # Stream the freshly produced token
    token_str = tok.decode(ids)
    print(token_str)
    if next_id == eos_id: break

