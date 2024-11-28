import os

from transformers import Qwen2Tokenizer
from pathlib import Path

from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16
from examples.llama3 import load
from tinygrad import nn, Context
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict


class QwQ:
  def __init__(self, model, tokenizer):
    self.model, self.tokenizer = model, tokenizer

  @staticmethod
  def from_pretrained(model_path:str):
    pass

def _download_weights(total_parts:int = 17):
  model = fetch("https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model.safetensors.index.json?download=true", "model.safetensors.index.json", subdir="qwq_32b_preview")

  for i in range(1, total_parts + 1):
    filename = f"model-{str(i).zfill(5)}-of-{str(total_parts).zfill(5)}.safetensors"
    fetch(f"https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/{filename}?download=true", filename, subdir="qwq_32b_preview")

  return model


def load_model(model_path, model_params, device=None):
  # build model
  model = Transformer(**model_params, linear=nn.Linear, max_context=8192, jit=True)

  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, model_params["n_heads"], model_params["n_kv_heads"])
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # quantize
    # if quantize is not None:
    #   weights = linear.quantize(weights, device)
    #   for _,v in weights.items(): v.realize()

    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if 'scale' in k: v.shard_(device, axis=None)  # from quantized
        elif '.attention.' in k: v.shard_(device, axis=-1)
        elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=0)
        else: v.shard_(device, axis=None)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)

  return model
  

if __name__ == "__main__":
  model_params = {"dim": 5120, "n_heads": 40, "n_kv_heads": 8, "n_layers": 64, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 152064, "hidden_dim": 27648}
  model_path = _download_weights()
  model = load_model(Path(os.path.dirname(model_path)), model_params)
  # tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/QwQ-32B-Preview")