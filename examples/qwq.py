import argparse
import os
import sys

from transformers import AutoTokenizer
from pathlib import Path

from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from examples.llama3 import load
from tinygrad import nn, Tensor
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict

MODELS = {
  "32B": {
    "model_params": {"dim": 5120, "n_heads": 40, "n_kv_heads": 8, "n_layers": 64, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 152064, "hidden_dim": 27648},
    "total_num_weights": 17
  }
}

def download_weights(total_num_weights:int):
  model = fetch("https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model.safetensors.index.json?download=true", "model.safetensors.index.json", subdir="qwq_32b_preview")

  for i in range(1, total_num_weights + 1):
    filename = f"model-{str(i).zfill(5)}-of-{str(total_num_weights).zfill(5)}.safetensors"
    fetch(f"https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/{filename}?download=true", filename, subdir="qwq_32b_preview")

  return model

def load_model(model_path, model_params):
  # build model
  model = Transformer(**model_params, linear=nn.Linear, max_context=32000)

  new_layers = []
  for layer in model.layers:
    head_dim = model_params["dim"] // model_params["n_heads"]
    layer.attention.wq = nn.Linear(model_params["dim"], model_params["n_heads"] * head_dim, bias=True)
    layer.attention.wk = nn.Linear(model_params["dim"], model_params["n_kv_heads"] * head_dim, bias=True)
    layer.attention.wv = nn.Linear(model_params["dim"], model_params["n_kv_heads"] * head_dim, bias=True)
    new_layers.append(layer)
  model.layers = new_layers

  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, model_params["n_heads"], model_params["n_kv_heads"])
  weights = fix_bf16(weights)

  # replace weights in model
  load_state_dict(model, weights, strict=False, consume=True)

  return model
  

if __name__ == "__main__":
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description="Run QwQ in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--size", choices=["32B"], default="32B", help="Model size")
  parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  # parser.add_argument("--weights", type=str, default=None,
  #                     help="Path to the downloaded weights")
  args = parser.parse_args()

  model_info = MODELS[args.size]

  model_path = download_weights(model_info["total_num_weights"])
  transformer = load_model(Path(os.path.dirname(model_path)), model_info["model_params"])
  tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")

  outputted = "Hello."
  start_pos, toks = 0, tokenizer(outputted)["input_ids"]
  print(outputted, end='', flush=True)

  new_toks = tokenizer(outputted)["input_ids"]
  assert toks == new_toks[:len(toks)]
  toks = new_toks
  assert outputted == tokenizer.decode(toks)

  tok_tensor = None
  for i in range(args.count):
    next_tok = Tensor([toks[start_pos:]]) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
    tok_tensor = transformer(next_tok, start_pos, args.temperature)
    tok = tok_tensor.item()

    # use the kv cache
    start_pos = len(toks)

    # add the new token
    toks.append(tok)

    # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
    cur = tokenizer.decode(toks, skip_special_tokens=True)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur