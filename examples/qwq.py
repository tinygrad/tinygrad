import argparse
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import load_state_dict, get_parameters, get_state_dict
from tinygrad.helpers import fetch, tqdm
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from examples.llama3 import concat_weights, load

from icecream import ic, install
from transformers import AutoTokenizer
install()

# https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/config.json
MODEL_PARAMS = {
  '32B': {
    'args': dict(dim=5120, hidden_dim=27648, n_heads=40, n_layers=64, norm_eps=1e-05, vocab_size=152064, n_kv_heads=8,  rope_theta=1000000.0),
    'files': 17,
  },
  'test': {
    'args': dict(dim=5120, hidden_dim=27648, n_heads=40, n_layers=1, norm_eps=1e-05, vocab_size=152064, n_kv_heads=8,  rope_theta=1000000.0),
    'files': 17,
  }
}

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

def build_transformer(model_path: Path, model_size="32B"):

  # build model
  # qwq arch is like the Llama arch but modified with 1) biases in wkqv 2) sliding window attn
  model = Transformer(**(args := MODEL_PARAMS[model_size]['args']), jit=True)
  new_layers = []
  for layer in model.layers:
    head_dim = args['dim'] // args['n_heads']
    layer.attention.wq = nn.Linear(args['dim'],args['n_heads'] * head_dim, bias=True)
    layer.attention.wk = nn.Linear(args['dim'], args['n_kv_heads'] * head_dim, bias=True)
    layer.attention.wv = nn.Linear(args['dim'], args['n_kv_heads'] * head_dim, bias=True)
    new_layers.append(layer)
  model.layers = new_layers

  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
    else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device)
  else:
    weights = load(str(model_path))
  # only look at first `n_layers`
  weights = {k: v for k, v in weights.items() if 'model.layers' not in k or int(k.split('.')[2]) < args['n_layers']}
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  # weights = fix_bf16(weights) # on mac, need to run with `SUPPORT_BF16=1` flag

  # weights = {k: v.cast(dtypes.float) for k, v in weights.items()}

  # replace weights in model
  load_state_dict(model, weights, strict=False, consume=True)
  return model




  toks = [spp.bos_id()]
  start_pos = 0
  for i in range(args.count):
    GlobalCounters.reset()
    with Profiling(sort="time", frac=0.1, enabled=args.profile):
      with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
        tok = model(Tensor([toks[start_pos:]]), 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos), args.temperature).item()
    toks.append(tok)
    start_pos += 1
    print(spp.decode(toks))


def generate(model, tokenizer, prompt: str, n_tokens_to_gen: int = 10, temp: bool = 1.0, sample: bool = False, top_k: int = None):
  tks = tokenizer(prompt)["input_ids"]
  while len(tks) < 4:
    tks = [50279] + tks

  # Loading in the prompt tokens
  logits = model.forward(Tensor([tks]))[:, -1, :]
  for _ in tqdm(range(n_tokens_to_gen), desc="Speed Gen"):
    if sample:
      tok_Tens = (logits/temp).softmax().multinomial()
    else:
      tok_Tens = logits.argmax(axis=-1).unsqueeze(0)
    tok = tok_Tens.item()
    tks.append(tok)
    logits = model.forward_jit(tok_Tens)[:, -1, :]

  output_completions = ''.join([tokenizer.decode(output) for output in tks])
  return output_completions

def main():
  Tensor.no_grad = True
  parser = argparse.ArgumentParser()
  parser.add_argument("--download_model", action="store_true", help="Download a model")
  parser.add_argument("--model", type=Path, help="Model path")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument('--model_size', type=str, choices=['32B', 'test'], default='32B', help="Model size")
  parser.add_argument("--temperature", type=int, default=0.85, help="Temperature")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  parser.add_argument('--max_tokens', type=int, help='maximum tokens outputted')
  args = parser.parse_args()

  # download model maybe
  assert (args.model and not args.download_model) or (not args.model and args.download_model), "either download or provide model"
  if args.download_model:
    subdir = 'qwq-32b-preview'
    args.model = fetch('https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model.safetensors.index.json', "model.safetensors.index.json", subdir=subdir)
    for i  in range(MODEL_PARAMS[args.model_size]["files"]): fetch(f'https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/model-{i+1:05d}-of-00017.safetensors', f'model-{i+1:05d}-of-00017.safetensors', subdir=subdir)
    for tokenizer_obj in ['vocab.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json']:
      fetch(f'https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main/{tokenizer_obj}', tokenizer_obj, subdir=subdir)
  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  model = build_transformer(args.model, args.model_size)
  # tokenizer = Tokenizer(str((args.model if args.model.is_dir() else args.model.parent) / "merges.txt"))
  tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")
  msg = ["How are you?"]
  start_pos, tokens = 0, Tensor(tokenizer(msg)['input_ids'])
  ic(tokens, tokens.numpy())
  out = model(tokens, start_pos)


if __name__ == '__main__':
  main()
