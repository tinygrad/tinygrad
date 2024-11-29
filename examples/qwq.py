import argparse
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe
from tinygrad import Tensor, nn
from tinygrad.nn.state import load_state_dict, get_parameters, get_state_dict
from tinygrad.helpers import fetch
from extra.models.llama import Transformer
from examples.llama3 import concat_weights, load

# https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/config.json
MODEL_PARAMS = dict(dim=5120, hidden_dim=27648, n_heads=40, n_layers=64, norm_eps=1e-05, vocab_size=152064, n_kv_heads=8,  rope_theta=1000000.0)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py
# https://huggingface.co/yzsydlc/qwen2/blob/main/tokenization_qwen.py
class Tokenizer:
  pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_token_start, special_tokens = 151643, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"<|extra_{i}|>" for i in range(205)]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens, start=special_token_start)}
    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

def build_transformer(model_path: Path):
  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
    else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device)
  else:
    weights = load(str(model_path))

  # qwq arch is like the Llama arch but modified with 1) biases in wkqv 2) sliding window attn
  model = Transformer(**MODEL_PARAMS, jit=True)
  new_layers = []
  for layer in model.layers:
    head_dim = MODEL_PARAMS['dim'] // MODEL_PARAMS['n_heads']
    layer.attention.wq = nn.Linear(MODEL_PARAMS['dim'], MODEL_PARAMS['n_heads'] * head_dim, bias=True)
    layer.attention.wk = nn.Linear(MODEL_PARAMS['dim'], MODEL_PARAMS['n_kv_heads'] * head_dim, bias=True)
    layer.attention.wv = nn.Linear(MODEL_PARAMS['dim'], MODEL_PARAMS['n_kv_heads'] * head_dim, bias=True)
    new_layers.append(layer)
  model.layers = new_layers

  # replace weights in model
  load_state_dict(model, weights, strict=False, consume=True)
  return model


def main():
  Tensor.no_grad = True
  parser = argparse.ArgumentParser()
  parser.add_argument("--download_model", action="store_true", help="Download a model")
  parser.add_argument("--model", type=Path, help="Model path")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=int, default=0.85, help="Temperature")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  args = parser.parse_args()

  # download model maybe
  assert (args.model and not args.download_model) or (not args.model and args.download_model), "either download or provide model"
  if args.download_model:
    subdir = 'qwq-32b-preview'
    args.model = fetch('https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model.safetensors.index.json', "model.safetensors.index.json", subdir=subdir)
    for i  in range(17): fetch(f'https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model-{i+1:05d}-of-00017.safetensors', f'model-{i+1:05d}-of-00017.safetensors', subdir=subdir)
    for tokenizer_obj in ['vocab.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json']:
      fetch(f'https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/{tokenizer_obj}', tokenizer_obj, subdir=subdir)
  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  model = build_transformer(args.model)
  tokenizer = Tokenizer(str((args.model if args.model.is_dir() else args.model.parent) / "merges.txt"))

if __name__ == '__main__':
  main()
