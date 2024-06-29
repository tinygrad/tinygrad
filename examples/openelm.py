import json, pprint
from tinygrad import fetch, nn, Tensor
from extra.models.llama import RMSNorm    # TODO: move to nn

class FeedForward:
  def __init__(self, model_dim, intermediate_dim):
    self.proj_1 = nn.Linear(model_dim, 2*intermediate_dim, bias=False)
    self.proj_2 = nn.Linear(intermediate_dim, model_dim, bias=False)

  def __call__(self, x):
    y_12 = self.proj_1(x)
    y_1, y_2 = y_12.chunk(2, dim=-1)
    return self.proj_2(y_1.silu() * y_2)

class Attention:
  def __init__(self, model_dim, num_query_heads, num_kv_heads, head_dim):
    self.qkv_proj = nn.Linear(model_dim, (num_query_heads + num_kv_heads*2) * head_dim, bias=False)

  def __call__(self, x):
    pass

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

  def __call__(self, x:Tensor):
    return self.token_embeddings(self.norm(x))

if __name__ == "__main__":
  model = Transformer(json.loads(fetch("https://huggingface.co/apple/OpenELM-270M-Instruct/resolve/main/config.json?download=true").read_bytes()))
  weights = nn.state.safe_load(fetch("https://huggingface.co/apple/OpenELM-270M-Instruct/resolve/main/model.safetensors?download=true"))
  #for k, v in weights.items(): print(k, v.shape)
  nn.state.load_state_dict(model, {k.removeprefix("transformer."):v for k,v in weights.items()})

