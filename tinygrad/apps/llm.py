import re
from tinygrad import Tensor, nn, UOp
from typing import Dict, List, Sequence

class SimpleLlamaTokenizer:
    """
    A *pure-Python*, greedy tokenizer that matches the behaviour of
    `llama.cpp`’s `llama_tokenize()` in longest-match mode.

    ▸ Supports:
        • “normal” SentencePiece pieces (token type = 0)
        • Byte-fallback tokens  <0x00> … <0xFF>  (ids 3 … 258)

    ▸ Example
        tok = SimpleLlamaTokenizer(kv_data)
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
    """

    BYTE_FALLBACK_OFFSET = 3                    # id 3 → 0x00 … id 258 → 0xFF
    _ws_re = re.compile(r"\s+")

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(self, kv_data: Dict):
        pieces: Sequence[str] = kv_data["tokenizer.ggml.tokens"]

        # core vocab and lookup table
        self.vocab: List[str] = list(pieces)
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}

        # pre-build full byte-fallback maps (robust to sparse token_type vectors)
        self._byte_to_id = {b: self.BYTE_FALLBACK_OFFSET + b for b in range(256)}
        self._id_to_byte = {i: b for b, i in self._byte_to_id.items()}

        # SentencePiece inserts a leading underline on each new word
        self.add_prefix_space = "▁"

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """
        Minimal SentencePiece pipeline:

        1. Collapse all whitespace to single spaces.
        2. Replace each space with U+2581 “▁”.
        3. Greedy longest-match over the whole string.
        4. Byte-fallback for anything unmatched.
        """
        # 1) basic pre-tokenisation
        text = self._ws_re.sub(" ", text).strip()
        if not text:
            return []

        # e.g. "▁Hello▁world"
        spm_str = self.add_prefix_space + text.replace(" ", self.add_prefix_space)

        # 2-4) greedy BPE with byte fallback
        out: List[int] = []
        i = 0
        while i < len(spm_str):
            for j in range(len(spm_str), i, -1):
                piece = spm_str[i:j]
                tid = self.token_to_id.get(piece)
                if tid is not None:            # found a match
                    out.append(tid)
                    i = j
                    break
            else:                              # no match → UTF-8 bytes
                for b in spm_str[i].encode("utf-8", errors="replace"):
                    out.append(self._byte_to_id[b])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode(self, ids: Sequence[int]) -> str:
        pieces: List[str] = []
        for tid in ids:
            if tid in self._id_to_byte:        # byte-fallback id
                pieces.append(bytes([self._id_to_byte[tid]]).decode("utf-8", "replace"))
            else:                              # normal vocab id
                pieces.append(self.vocab[tid])

        # remove leading "▁" and convert underlines back to spaces
        return "".join(pieces).replace(self.add_prefix_space, " ").lstrip(" ")

# --------------------------------------------------------------------------- #
# rotary-embedding helpers
# --------------------------------------------------------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000):
  half_dim = head_dim // 2
  freq = base ** (-Tensor.arange(0, half_dim, dtype='float32') / half_dim)
  t = Tensor.arange(seq_len, dtype='float32')[:, None]          # (T, 1)
  angles = t * freq                                             # (T, Hd/2)
  return Tensor.stack(angles.cos(), angles.sin(), dim=-1)     # (T, Hd/2, 2)

def apply_rope(q: Tensor, k: Tensor, freqs_cis: Tensor, start_pos: int | Tensor):
  B, H, T, Hd = q.shape
  half        = Hd // 2
  cos = freqs_cis[start_pos : start_pos + T, :, 0]     # (T, half)
  sin = freqs_cis[start_pos : start_pos + T, :, 1]     # (T, half)
  cos, sin = cos[None, None, ...], sin[None, None, ...]
  def _rope(x):
    x = x.reshape(B, H, T, half, 2)                  # split into pairs
    x1, x2 = x[..., 0], x[..., 1]                    # even / odd dims
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return Tensor.stack(y1, y2, dim=-1).reshape(B, H, T, Hd)
  return _rope(q), _rope(k)

class TransformerBlock:
  def __init__(self, *, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int = 0):
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
    self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)  # SiLU gate
    self.ffn_up   = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

  def __call__(self, x: Tensor, start_pos: UOp | int, freqs_cis: Tensor, mask: Tensor | None):
    B, T, _ = x.shape

    # --- attention ------------------------------------------------------
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)

    # reshape into heads
    q = q.reshape(B, T, self.n_heads,     self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
    k = k.reshape(B, T, self.n_kv_heads,  self.head_dim).transpose(1, 2)  # (B, KvH, T, Hd)
    v = v.reshape(B, T, self.n_kv_heads,  self.head_dim).transpose(1, 2)  # (B, KvH, T, Hd)

    # repeat k/v if MQA (n_heads > n_kv_heads)
    if self.n_heads != self.n_kv_heads:
      k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
      v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    # rotary positional embedding (in-place)
    q, k = apply_rope(q, k, freqs_cis, start_pos)

    # tinygrad’s native SDPA: (B,H,T,Hd) → (B,H,T,Hd)
    out = q.scaled_dot_product_attention(k, v, attn_mask=mask, is_causal=True)
    out = self.attn_output(out.transpose(1, 2).reshape(B, T, -1))

    # --- feed forward ------------------------------------------------------
    h = x + out
    h_norm = self.ffn_norm(h)
    return (h + self.ffn_down(self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm))).contiguous()

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, eps, vocab_size, max_context):
    self.blk = [TransformerBlock(dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                                 norm_eps=eps, max_context=max_context) for _ in range(num_blocks)]
    self.token_embd   = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, eps)
    self.max_context, self.head_dim = max_context, dim // n_heads

  def __call__(self, tokens: Tensor, start_pos: int | UOp = 0, mask: Tensor | None = None):
    if not hasattr(self, '_rope_cache'): self._rope_cache = build_rope_cache(self.max_context, self.head_dim) # pre-compute the base RoPE table once
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos, self._rope_cache, mask)
    return self.output_norm(x) @ self.token_embd.weight.T

if __name__ == "__main__":
  gguf_tensor = Tensor.from_url("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf").to(None)
  kv, state_dict = nn.state.gguf_load(gguf_tensor)

  model = Transformer(num_blocks=kv['llama.block_count'], dim=kv['llama.embedding_length'],
                      hidden_dim=kv['llama.feed_forward_length'], n_heads=kv['llama.attention.head_count'],
                      n_kv_heads=kv['llama.attention.head_count_kv'], eps=kv['llama.attention.layer_norm_rms_epsilon'],
                      vocab_size=kv['llama.vocab_size'], max_context=kv['llama.context_length'])
  nn.state.load_state_dict(model, state_dict, consume=True, realize=False)

  for k,v in state_dict.items(): print(k, v.shape)

  # TODO: add tokenizer and call to model
  print(kv.keys())

  tok = SimpleLlamaTokenizer(kv)

  prompt = "Hello"
  input_ids = Tensor(tok.encode(prompt), dtype="int32")[None, :]   # (1, T)

  logits = model(input_ids)
  next_id = logits[:, -1, :].softmax(-1).argmax().item()
  print("Assistant starts with:", tok.decode([next_id]), next_id)

