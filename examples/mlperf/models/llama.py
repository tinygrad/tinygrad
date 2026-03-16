from tinygrad import Tensor, nn
from tinygrad.helpers import getenv
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis

class Attention:
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int|None=None, linear=nn.Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    if getenv("WQKV"):
      self.wqkv = linear(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2, bias=False)
    else:
      self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
      self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
      self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)

    self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x:Tensor, freqs_cis:Tensor) -> Tensor:
    if getenv("WQKV"):
      xqkv = self.wqkv(x)
      xqkv = xqkv.reshape(xqkv.shape[0], xqkv.shape[1], self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = xqkv[:, :, :, :self.n_rep].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xk = xqkv[:, :, :, self.n_rep:self.n_rep+1].reshape(xqkv.shape[0], xqkv.shape[1], -1)
      xv = xqkv[:, :, :, self.n_rep+1:self.n_rep+2].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)

    attn = attn.reshape(bsz, seqlen, -1)
    return self.wo(attn)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear):
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    w1 = self.w1(x).silu()
    w3 = self.w3(x)
    return self.w2(w1 * w3)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int|None, norm_eps:float, linear=nn.Linear):
    self.attention = Attention(dim, n_heads, n_kv_heads, linear)
    self.feed_forward = FeedForward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, freqs_cis:Tensor):
    h = x + self.attention(self.attention_norm(x), freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

class Transformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024, linear=nn.Linear, embedding=nn.Embedding):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, linear) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False) if embedding == nn.Embedding else linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for layer in self.layers: h = layer(h, freqs_cis)
    logits = self.output(self.norm(h))
    return logits
