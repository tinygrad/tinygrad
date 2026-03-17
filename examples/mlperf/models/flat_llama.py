import math, os
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  os.environ["DEV"] = "NULL"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis

def rmsnorm(x_in:Tensor, eps:float):
  x = x_in.float()
  x = x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return x.cast(x_in.dtype)

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024):
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    # Attention
    self.wqkv = self.lin_per_layer(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2)
    self.wo = self.lin_per_layer(dim, self.n_heads * self.head_dim)

    # FeedForward
    self.w1 = self.lin_per_layer(dim, hidden_dim)
    self.w2 = self.lin_per_layer(hidden_dim, dim)
    self.w3 = self.lin_per_layer(dim, hidden_dim)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim)
    self.ffn_norm = Tensor.ones(n_layers, dim)

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)

  def lin_per_layer(self, in_features:int, out_features:int):
    bound = 1 / math.sqrt(in_features)
    if getenv("ZEROS"): return Tensor.zeros(self.n_layers, out_features, in_features)
    return Tensor.uniform(self.n_layers, out_features, in_features, low=-bound, high=bound)

  def attention(self, x:Tensor, freqs_cis:Tensor, attention_norm:Tensor, wqkv:Tensor, wo:Tensor):
    x = rmsnorm(x, self.norm_eps) * attention_norm
    xqkv = x @ wqkv.T

    # reshapes
    xqkv = xqkv.reshape(xqkv.shape[0], xqkv.shape[1], self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = xqkv[:, :, :, :self.n_rep].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    xk = xqkv[:, :, :, self.n_rep:self.n_rep+1].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    xv = xqkv[:, :, :, self.n_rep+1:self.n_rep+2].reshape(xqkv.shape[0], xqkv.shape[1], -1)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    return attn @ wo.T

  def feed_forward(self, x:Tensor, ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor):
    x = rmsnorm(x, self.norm_eps) * ffn_norm
    w1 = (x @ w1.T).silu()
    w3 =  x @ w3.T
    return (w1 * w3) @ w2.T

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor,
                attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor):
    h = x + self.attention(x, freqs_cis, attention_norm, wqkv, wo)
    return h + self.feed_forward(h, ffn_norm, w1, w2, w3)

  def __call__(self, tokens:Tensor):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :tokens.shape[1], :, :, :]
    for i in range(self.n_layers):
      h = self.run_layer(h, freqs_cis,
                         self.attention_norm[i], self.wqkv[i], self.wo[i],
                         self.ffn_norm[i], self.w1[i], self.w2[i], self.w3[i])
    logits = self.output(self.norm(h))
    return logits

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
  sz = 0
  for k,v in state.items():
    if v.requires_grad is None: v.requires_grad_(True)
    print(f"{k:30s} {str(v.shape):30s} {v.dtype} {v.device}")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("realize weights: "): Tensor.realize(*state.values())
  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=model.vocab_size, dtype=dtypes.int).realize()

  @TinyJit
  def jit_step(tokens:Tensor):
    GlobalCounters.reset()
    print(colored("*** step", "red"))
    with Timing("python forward: "): loss = model(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "): loss.backward()
    with Timing("run step: "): loss.realize(*[x.grad for x in state.values() if x.requires_grad])

  jit_step(tokens)
  jit_step(tokens)
  jit_step(tokens)
  print(f"mem used: {GlobalCounters.mem_used/1e9:.2f} GB")
