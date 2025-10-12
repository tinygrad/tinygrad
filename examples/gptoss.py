# https://huggingface.co/blog/faster-transformers
# https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the
import argparse, functools, os, sys
from pathlib import Path
from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.helpers import fetch, getenv, Timing, GlobalCounters
from tinygrad.nn.state import load_state_dict, ggml_data_to_tensor, get_parameters
from examples.llama3 import load
from extra.models.llama import Transformer
from transformers import AutoTokenizer

from icecream import install
install()

MODELS = {
  "20B": {
    "params": {"dim": 2880, "hidden_dim": 2880, "n_heads": 64, "n_layers": 24, "norm_eps": 1e-5, "vocab_size": 201088, "n_kv_heads": 8, "head_dim": 64, "rope_theta": 150000, "max_context": 4096, "num_experts": 32, "activated_experts": 4},
    "total_num_weights": 3,
    "model": "openai/gpt-oss-20b",
    "tokenizer": "openai/gpt-oss-20b",
  }
}

def swiglu(x:Tensor, alpha:float=1.702, limit:float=7.0) -> Tensor:
  gate, up = x[..., ::2], x[..., 1::2]
  return (up.clip(-limit, limit) + 1) * gate.clip(None, limit) * (gate.clip(None, limit) * alpha).sigmoid()

class MixtureFeedForward:
  def __init__(self, num_experts:int, activated_experts:int, dim:int, hidden_dim:int, linear=nn.Linear):
    self.activated_experts = activated_experts
    self.gate = nn.Linear(dim, num_experts, bias=True)
    self.up_proj = Tensor.zeros(num_experts, hidden_dim * 2, dim, dtype=dtypes.bfloat16)
    self.up_proj_bias = Tensor.zeros(num_experts, hidden_dim * 2, dtype=dtypes.bfloat16)
    self.down_proj = Tensor.zeros(num_experts, dim, hidden_dim, dtype=dtypes.bfloat16)
    self.down_proj_bias = Tensor.zeros(num_experts, dim, dtype=dtypes.bfloat16)

  # def __call__(self, x:Tensor) -> Tensor:
  #   assert x.shape[0] == 1 and x.shape[1] == 1, "expected BS=1 and seqlen=1 but got BS={x.shape[0]} and seqlen={x.shape[1]}"
  #   ic(x.shape)

  #   # Select top-k experts
  #   g = self.gate(x).softmax(-1) # (B,T,D) -> (B,T,E)
  #   g = g.squeeze() # (B,T,E) -> (E,)
  #   probs, sel = g.topk(self.activated_experts) # (E,) -> (E,) (E,)
  #   ic(probs.shape, sel.shape)

  #   # expert weights
  #   w1, b1 = self.up_proj[sel].transpose(1, 2), self.up_proj_bias[sel] # (E,D,D2) (E,D2)
  #   w2, b2 = self.down_proj[sel].transpose(1, 2), self.down_proj_bias[sel] # (E,D,D) (E,D)
  #   ic(w1.shape, b1.shape, w2.shape, b2.shape)



  #   # out = (swiglu(x @ w1 + b1) @ w2 + b2).reshape(1, 1, -1)
  #   out = swiglu(Tensor.einsum("beck,bk->bec", w1, g) + b1)
  #   ic(out.shape)
  #   out = Tensor.einsum("beck,bek->bec", w2, out) + b2
  #   ic(out.shape)
  #   out = Tensor.einsum("bec,be->bc", out, probs)
  #   ic(out.shape)
  #   return out

  def __call__(self, x: Tensor) -> Tensor:
    assert x.shape[0] == 1 and x.shape[1] == 1, "expected BS=1 and seqlen=1 but got BS={x.shape[0]} and seqlen={x.shape[1]}"

    # Select top-k experts
    g = self.gate(x).softmax(-1) # (B,T,D) -> (B,T,E)
    g = g.squeeze() # (B,T,E) -> (E,)
    probs, sel = g.topk(self.activated_experts) # (E,) -> (E,) (E,)

    # reshape
    w1 = self.up_proj[sel].unsqueeze(0) # (1,E,D2,D)
    w2 = self.down_proj[sel].unsqueeze(0) # (1,E,D,D)
    b1 = self.up_proj_bias[sel].unsqueeze(0) # (1,E,D2)
    b2 = self.down_proj_bias[sel].unsqueeze(0) # (1,E,D)

    # MLP forward
    t = x.squeeze(1)  # (B,T,D) -> (B,1,T,D)
    t = swiglu(Tensor.einsum("bk,beck->bec", t, w1) + b1)  # (B,1,T,D) (1,E,D2,D) -> ?
    t = Tensor.einsum("bek,beck->bec", t, w2) + b2  # (1, 4, 2880)

    # Weighted sum over experts
    return (t * probs.reshape(1, -1, 1)).sum(1, keepdim=True)  # (1, 1, 2880)

def download_weights(model:str, total_num_weights:int) -> Path:
  model = fetch(f"https://huggingface.co/{model}/resolve/main/model.safetensors.index.json", "model.safetensors.index.json", subdir=(subdir:=model.split('/')[-1]))
  for i in range(total_num_weights):
    filename = f"model-{i:05d}-of-{total_num_weights-1:05d}.safetensors"
    fetch(f"https://huggingface.co/{model}/resolve/main/{filename}?download=true", filename, subdir=subdir)
  return Path(os.path.dirname(model))

def convert_from_huggingface(weights:dict[str, Tensor], n_layers: int, n_heads: int, n_kv_heads: int, permute_layers: bool = True):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_norm.weight": f"layers.{l}.attention.{x}_norm.weight" for x in ["q", "k"] for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"layers.{l}.attention.w{x}.bias" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.sinks": f"layers.{l}.attention.sinks" for l in range(n_layers)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.router.weight": f"layers.{l}.feed_forward.gate.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.router.bias": f"layers.{l}.feed_forward.gate.bias" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.gate_up_proj_bias": f"layers.{l}.feed_forward.gate_up_proj_bias" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.down_proj_bias": f"layers.{l}.feed_forward.gate_down_proj_bias" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.gate_up_proj_blocks": f"layers.{l}.feed_forward.gate_up_proj_blocks" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.gate_up_proj_scales": f"layers.{l}.feed_forward.gate_up_proj_scales" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.down_proj_blocks": f"layers.{l}.feed_forward.gate_down_proj_blocks" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.experts.down_proj_scales": f"layers.{l}.feed_forward.gate_down_proj_scales" for l in range(n_layers)},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }

  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if ("q_proj" in k or "q_norm" in k) and permute_layers: v = permute(v, n_heads)
      elif ("k_proj" in k or "k_norm" in k) and permute_layers: v = permute(v, n_kv_heads)
    sd[keymap[k]] = v
  return sd

def fix_mxfp4(weights, n_layers) -> Tensor:
    def dequantize_mxfp4(blocks: Tensor, scales: Tensor) -> Tensor:
        """Dequantize MXFP4 to float32. blocks: (*batch, num_blocks, 16), scales: (*batch, num_blocks) -> (*batch, num_blocks*32)"""
        assert blocks.shape[:-1] == scales.shape and blocks.shape[-1] == 16
        mxfp4_data = Tensor.cat(scales.unsqueeze(-1), blocks, dim=-1).flatten()  # interleave and flatten to 1D
        # ic(blocks, scales, mxfp4_data)
        return ggml_data_to_tensor(mxfp4_data, scales.numel() * 32, 39).reshape(*scales.shape[:2], -1)

    for l in range(n_layers):
        for proj in ['gate_up_proj', 'gate_down_proj']:
            blocks = f'layers.{l}.feed_forward.{proj}_blocks'
            scales = f'layers.{l}.feed_forward.{proj}_scales'
            proj = dequantize_mxfp4(weights.pop(blocks), weights.pop(scales))
            # ic(proj)
            weights[f'layers.{l}.feed_forward.{proj}.weights'] = proj
    return weights

def load_model(path:Path, params:dict[str, int|float]) -> Transformer:
  # build model
  feed_forward = functools.partial(MixtureFeedForward, params.pop("num_experts"), params.pop("activated_experts"))
  # todo: fix jit=True
  model = Transformer(**params, jit=False, feed_forward=feed_forward)

  # set head dim and add bias to each attention projection
  for i in range(len(model.layers)):
    model.layers[i].attention.wq = nn.Linear(params["dim"], params["n_heads"] * params["head_dim"], bias=True)
    model.layers[i].attention.wk = nn.Linear(params["dim"], params["n_kv_heads"] * params["head_dim"], bias=True)
    model.layers[i].attention.wv = nn.Linear(params["dim"], params["n_kv_heads"] * params["head_dim"], bias=True)
    model.layers[i].attention.wo = nn.Linear(params["n_heads"] * params["head_dim"], params["dim"], bias=True)

  # add sliding attention to all even attention layers
  for i in range(0, len(model.layers), 2): model.layers[i].sliding_window = 128

  # add attention sinks to all attention layers
  for i in range(len(model.layers)): model.layers[i].sinks = Tensor.empty(params['n_heads'], dtype=dtypes.bfloat16)

  # load weights
  weights = convert_from_huggingface(load(str(path / "model.safetensors.index.json")), params["n_layers"], params["n_heads"], params["n_kv_heads"], permute_layers=True)
  weights = fix_mxfp4(weights, params["n_layers"])
#   for k, v in weights.items():
#       if 'model.layers.0' in k: ic(k, v.dtype, v.shape)

  # replace weights in model
  load_state_dict(model, weights, strict=False, consume=True)
  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run gpt-oss in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--size", choices=["20B"], default="20B", help="Model size")
  parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--prompt", type=str, default="Hi", help="Phrase to start with")
  parser.add_argument("--weights", type=str, default=None, help="Path to the downloaded weights")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  args = parser.parse_args()

  model_info = MODELS[args.size]

  if getenv("TORCH"):
    from transformers import GptOssForCausalLM
    model = GptOssForCausalLM.from_pretrained(model_info["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    inputs = tokenizer("Hello", return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    exit(0)

  model_path = Path(args.weights) if args.weights else download_weights(model_info["model"], model_info["total_num_weights"])
  transformer = load_model(model_path, model_info["params"])
  tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(transformer))

  if args.seed is not None: Tensor.manual_seed(args.seed)

  outputted = args.prompt
  start_pos, toks = 0, tokenizer(outputted)["input_ids"]
  print(outputted, end="", flush=True)

  tok_tensor = None
  for i in range(args.count):
    GlobalCounters.reset()

    if args.timing: print("")
    st = GlobalCounters.time_sum_s
    next_tok = Tensor([toks[start_pos:]]) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
    with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
      with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on {Device.DEFAULT}") +
                  f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB" +
                  (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s")), enabled=args.timing):
        tok_tensor = transformer(next_tok, start_pos, args.temperature)
      tok = tok_tensor.item()

    # use the kv cache
    start_pos = len(toks)

    # add the new token
    toks.append(tok)

    cur = tokenizer.decode(toks, skip_special_tokens=True)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur
