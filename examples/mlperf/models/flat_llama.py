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
from tinygrad.helpers import Timing, colored, GlobalCounters
from tinygrad.uop.ops import Ops, UOp
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
    self.wo = self.lin_per_layer(self.n_heads * self.head_dim, dim)

    # FeedForward
    self.w1 = self.lin_per_layer(dim, hidden_dim)
    self.w2 = self.lin_per_layer(hidden_dim, dim)
    self.w3 = self.lin_per_layer(dim, hidden_dim)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

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

    bsz, seqlen, _ = xqkv.shape
    # interleaved layout: each kv group has [n_rep q heads, 1 k head, 1 v head] for clean MP sharding
    xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    return attn @ wo.T

  def feed_forward(self, x:Tensor, ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor):
    x = rmsnorm(x, self.norm_eps) * ffn_norm
    x_w1 = (x @ w1.T).silu()
    x_w3 =  x.contiguous_backward() @ w3.T
    return (x_w1 * x_w3) @ w2.T

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor,
                attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                ffn_norm:Tensor, w1:Tensor, w2:Tensor, w3:Tensor):
    h = x + self.attention(x, freqs_cis, attention_norm, wqkv, wo)
    return h + self.feed_forward(h, ffn_norm, w1, w2, w3)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    from tinygrad.nn.state import get_parameters
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      self.wqkv.shard_(device, axis=1).realize()          # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2).realize()             # (n_layers, dim, in) shard in
      self.w1.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.w2.shard_(device, axis=2).realize()             # (n_layers, dim, hidden) shard in
      self.w3.shard_(device, axis=1).realize()             # (n_layers, hidden, dim) shard out
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.weight.shard_(device, axis=0).realize()
      self.freqs_cis.shard_(device, axis=None).realize()

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

  # shard the model
  from tinygrad import Device
  if (DP := getenv("DP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)))
  if (MP := getenv("MP", 1)) > 1:
    model.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)), mp=True)

  # preallocate all the grad buffers and zero them out
  grads = {x:Tensor.zeros_like(x).contiguous() for x in state.values() if x.requires_grad is None}

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {v.dtype} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=model.vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if DP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)), axis=0)
  if MP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)))

  # TODO: this shouldn't be needed, but it prevents a copy of the grads. CAT can help
  def apply_grad(old_grad:UOp, new_grad:UOp) -> list[UOp]:
    if new_grad.op == Ops.ADD:
      return apply_grad(old_grad, new_grad.src[0])+apply_grad(old_grad, new_grad.src[1])
    elif new_grad.op == Ops.PAD:
      grad_shrink = tuple([(p[0], s+p[0]) for s,p in zip(new_grad.src[0].shape, new_grad.marg)])
      return apply_grad(old_grad.shrink(grad_shrink), new_grad.src[0])
    else:
      return [old_grad.store(old_grad + new_grad)]

  @TinyJit
  def jit_step(tokens:Tensor):
    GlobalCounters.reset()
    with Timing("python forward: "): loss = model(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        grads[t] = Tensor(grads[t].uop.after(UOp.group(*apply_grad(grads[t].uop, g.uop))), device=t.device)
    with Timing("run step: "): loss.realize(*grads.values())

  for i in range(5):
    with Timing(colored(f"*** step {i}: ", "red")):
      jit_step(tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
