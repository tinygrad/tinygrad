import argparse, os, functools
from pathlib import Path
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.state import load_state_dict
from examples.llama3 import load
# from examples.olmoe import MixtureFeedForward
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16

from icecream import install
install()

MODELS = {
  "20B": {
    "params": {"dim": 2880, "hidden_dim": 2880, "n_heads": 64, "n_kv_heads": 8, "n_layers": 24, "norm_eps": 1e-5, "rope_theta": 150000, "vocab_size": 201088, "max_context": 4096, "num_experts": 32, "activated_experts": 4},
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
    self.gate = nn.Linear(dim, num_experts, bias=False)
    self.gate_up_proj = Tensor.zeros(num_experts, dim, hidden_dim * 2, dtype=dtypes.bfloat16)
    self.gate_up_proj_bias = Tensor.zeros(num_experts, hidden_dim * 2, dtype=dtypes.bfloat16)
    self.down_proj = Tensor.zeros(num_experts, hidden_dim, dim, dtype=dtypes.bfloat16)
    self.down_proj_bias = Tensor.zeros(num_experts, dim, dtype=dtypes.bfloat16)

  def __call__(self, x:Tensor) -> Tensor:
    assert x.shape[0] == 1 and x.shape[1] == 1, "only BS=1 and seqlen=1"
    x = x.squeeze()

    # Select top-k experts
    g = self.gate(x.unsqueeze(0)).squeeze()
    probs, sel = g.topk(self.activated_experts)
    probs = probs.softmax(dim=-1)

    # Up projection + SwiGLU
    up = x.dot(self.gate_up_proj[sel].transpose(1, 2)) + self.gate_up_proj_bias[sel]
    up = swiglu(up)

    # Down projection
    down = up.dot(self.down_proj[sel].transpose(1, 2)) + self.down_proj_bias[sel]

    # Weighted sum
    return (down * probs.unsqueeze(-1)).sum(0).unsqueeze(0).unsqueeze(0)

def download_weights(model:str, total_num_weights:int) -> Path:
  model = fetch(f"https://huggingface.co/{model}/resolve/main/model.safetensors.index.json", "model.safetensors.index.json", subdir=(subdir:=model.split('/')[-1]))
  for i in range(total_num_weights):
    filename = f"model-{i:05d}-of-{total_num_weights-1:05d}.safetensors"
    fetch(f"https://huggingface.co/{model}/resolve/main/{filename}?download=true", filename, subdir=subdir)
  return Path(os.path.dirname(model))

def load_model(path:Path, params:dict[str, int|float]) -> Transformer:
  # build model
  ffn = functools.partial(MixtureFeedForward, params.pop("num_experts"), params.pop("activated_experts"))
  model = Transformer(**params, feed_forward=ffn)

  # update layers to add bias
  updated_layers = []
  for layer in model.layers:
    head_dim = params["dim"] // params["n_heads"]
    layer.attention.wq = nn.Linear(params["dim"], params["n_heads"] * head_dim, bias=True)
    layer.attention.wk = nn.Linear(params["dim"], params["n_kv_heads"] * head_dim, bias=True)
    layer.attention.wv = nn.Linear(params["dim"], params["n_kv_heads"] * head_dim, bias=True)
    updated_layers.append(layer)
  model.layers = updated_layers

  # update layers to add sliding attention
  for i in range(0, len(model.layers), 2): model.layers[i].sliding_window = 128

  # load weights
  l = load(str(path / "model.safetensors.index.json"))
  weights = fix_bf16(convert_from_huggingface(l, params["n_layers"], params["n_heads"], params["n_kv_heads"], permute_layers=False))

  # replace weights in model
  load_state_dict(model, weights, strict=False, consume=True)
  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run gpt-oss in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--size", choices=["20B"], default="20B", help="Model size")
  parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--prompt", type=str, default="Hello.", help="Phrase to start with")
  parser.add_argument("--weights", type=str, default=None, help="Path to the downloaded weights")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  args = parser.parse_args()

  model_info = MODELS[args.size]

  if getenv("TORCH"):
    from transformers import GptOssForCausalLM, AutoTokenizer
    model = GptOssForCausalLM.from_pretrained(model_info["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    inputs = tokenizer("Hello", return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    exit(0)

  # download weights
#   with Timing("download weights ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
  model_path = Path(args.weights) if args.weights else download_weights(model_info["model"], model_info["total_num_weights"])
  # load weights to GPU
  transformer = load_model(model_path, model_info["params"])
  tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
  assert tokenizer.vocab_size() == model_info["params"]["vocab_size"], f"{tokenizer.vocab_size()=} not equal to {model_info['params']['vocab_size']}"

  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(transformer))

  if args.seed is not None: Tensor.manual_seed(args.seed)
