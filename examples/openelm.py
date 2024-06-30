import json, pprint
from typing import Optional
from tinygrad import fetch, nn, Tensor
from extra.models.llama import RMSNorm    # TODO: move to nn
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis

class FeedForward:
  def __init__(self, model_dim, intermediate_dim):
    self.proj_1 = nn.Linear(model_dim, 2*intermediate_dim, bias=False)
    self.proj_2 = nn.Linear(intermediate_dim, model_dim, bias=False)

  def __call__(self, x):
    y_12 = self.proj_1(x)
    y_1, y_2 = y_12.chunk(2, dim=-1)
    return self.proj_2(y_1.silu() * y_2)

MAX_CONTEXT = 1024
class Attention:
  def __init__(self, model_dim, num_query_heads, num_kv_heads, head_dim):
    self.qkv_proj = nn.Linear(model_dim, (num_query_heads + num_kv_heads*2) * head_dim, bias=False)
    self.num_query_heads, self.num_kv_heads = num_query_heads, num_kv_heads
    self.head_dim = head_dim
    self.q_norm = RMSNorm(head_dim)
    self.k_norm = RMSNorm(head_dim)
    self.out_proj = nn.Linear(num_query_heads * head_dim, model_dim, bias=False)
    #self.freqs_cis = None

  def __call__(self, x:Tensor):
    batch_size, seq_len, embed_dim = x.shape
    qkv = self.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, self.num_query_heads+self.num_kv_heads*2, self.head_dim).transpose(1, 2)
    xq,xk,xv = qkv.split([self.num_query_heads, self.num_kv_heads, self.num_kv_heads], 1)
    xq = self.q_norm(xq)
    xk = self.k_norm(xk)

    # grouped-query attention
    num_groups = self.num_query_heads // self.num_kv_heads
    xk = xk.repeat_interleave(num_groups, dim=1)
    xv = xv.repeat_interleave(num_groups, dim=1)

    # Add positional embedding (NOTE: jit avoid independent would avoid this None hack)
    #if self.freqs_cis is None:
    #  self.freqs_cis = precompute_freqs_cis(embed_dim // self.num_kv_heads, MAX_CONTEXT*1)
    #xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)  # TODO: why aren't types going through here?

    attn_output = xq.scaled_dot_product_attention(xk, xv)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_query_heads * self.head_dim)
    return self.out_proj(attn_output)

class Layer:
  def __init__(self, model_dim, intermediate_dim, num_query_heads, num_kv_heads, head_dim):
    self.ffn = FeedForward(model_dim, intermediate_dim)
    self.attn = Attention(model_dim, num_query_heads, num_kv_heads, head_dim)
    self.ffn_norm = RMSNorm(model_dim)
    self.attn_norm = RMSNorm(model_dim)

  def __call__(self, x):
    # (batch, seq_len, embed_dim)
    x = self.attn_norm(x)
    x = self.attn(x)
    res = x
    x = self.ffn_norm(x)
    x = self.ffn(x)
    return res + x

# stupidly complex
def make_divisible(v, divisor):
  new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
  if new_v < 0.9 * v: new_v += divisor
  return new_v

class Transformer:
  def __init__(self, cfg):
    #pprint.pp(cfg)
    self.layers = [Layer(cfg['model_dim'], make_divisible(int(cfg["model_dim"] * cfg['ffn_multipliers'][i]), cfg['ffn_dim_divisor']),
                         cfg['num_query_heads'][i], cfg['num_kv_heads'][i], cfg['head_dim']) for i in range(cfg['num_transformer_layers'])]
    self.norm = RMSNorm(cfg['model_dim'])
    self.token_embeddings = nn.Embedding(cfg['vocab_size'], cfg['model_dim'])

  def __call__(self, tokens:Tensor):
    # _bsz, seqlen = tokens.shape
    x = self.token_embeddings(tokens)
    for l in self.layers: x = l(x)
    return self.norm(x) @ self.token_embeddings.weight.T

if __name__ == "__main__":
  model = Transformer(json.loads(fetch("https://huggingface.co/apple/OpenELM-270M-Instruct/resolve/main/config.json?download=true").read_bytes()))
  weights = nn.state.safe_load(fetch("https://huggingface.co/apple/OpenELM-270M-Instruct/resolve/main/model.safetensors?download=true"))
  for k, v in weights.items(): print(k, v.shape)
  nn.state.load_state_dict(model, {k.removeprefix("transformer."):v for k,v in weights.items()})

  toks = Tensor([[0]])
  out = model(toks).realize()
  print(out.shape)

