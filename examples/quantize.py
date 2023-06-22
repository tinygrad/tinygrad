from pathlib import Path
import argparse
import numpy as np
np.set_printoptions(linewidth=200)

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from tinygrad.state import safe_save, torch_load

WEIGHTS_DIR = Path(__file__).parent.parent / "weights/LLaMA/"
TOKENIZER_FILENAME = WEIGHTS_DIR / "tokenizer.model"
VOCAB_SIZE = 32000

LLAMA_STANDARD_CONFIGS = {
    '3B': {
        'vocab_size': VOCAB_SIZE,
        'dim': 3200,
        'interm_size': 8640,
        'n_layers': 26,
        'n_heads': 32,
        'max_seq_len': 2048,
        'norm_eps': 1e-6,
    },
    '7B': {
        'vocab_size': VOCAB_SIZE,
        'dim': 4096,
        'interm_size': 11008,
        'n_layers': 32,
        'n_heads': 32,
        'max_seq_len': 2048,
        'norm_eps': 1e-6,
    },
    '13B': {
        'vocab_size': VOCAB_SIZE,
        'dim': 5120,
        'interm_size': 13824,
        'n_layers': 40,
        'n_heads': 40,
        'max_seq_len': 2048,
        'norm_eps': 1e-6,
    },
    '30B': {
        'vocab_size': VOCAB_SIZE,
        'dim': 6656,
        'interm_size': 17920,
        'n_layers': 60,
        'n_heads': 52,
        'max_seq_len': 2048,
        'norm_eps': 1e-6,
    },
    '65B': {
        'vocab_size': VOCAB_SIZE,
        'dim': 8192,
        'interm_size': 22016,
        'n_layers': 80,
        'n_heads': 64,
        'max_seq_len': 2048,
        'norm_eps': 1e-5,
    },
    'fake': {
        'vocab_size': VOCAB_SIZE,
        'dim': 128,
        'interm_size': 256,
        'n_layers': 8,
        'n_heads': 8,
        'max_seq_len': 2048,
        'norm_eps': 1e-6,
    },
}

def absmax_quantize(x:Tensor):
  x = x.numpy()
  xmax = np.max(np.abs(x),1)
  qm = np.round(x.T/xmax*127.)
  return Tensor(qm).T.cast(dtypes.int8), Tensor(xmax).cast(dtypes.float16)

if __name__ == "__main__":
  Tensor.no_grad = True
  parser = argparse.ArgumentParser(description='Quantize LLaMa to (almost) 8-bit', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', type=str, default="7B", help="Path to model weights")

  args = parser.parse_args()

  #for now quantization doesn't work with anything other than LLaMa 7B since it's the only official model that fits in one file
  #technically works with OpenLLaMa 3B, OpenLLaMa 3B outputs gibberish even with full precision weights, need to investigate
  #TODO: make it work with all sizes, perhaps even other models
  # load the torch model
  state_dict = torch_load(WEIGHTS_DIR/args.model/"consolidated.00.pth")
  #state_dict = safe_load(WEIGHTS_DIR/args.model/"consolidated.safetensors")
  print(f"quantizing {args.model} model")
  qstate_dict = {}
  qstate_dict = state_dict
  for k in list(state_dict.keys()):
    if "norm" in k or "embed" in k:
      print(f"copying {k}")
      qstate_dict[k] = state_dict[k].to("CPU").cast(dtypes.float16).realize()
    elif "rope" in k:
      continue
    else:
      print(f"quantizing {k}")
      qstate_dict[k], qstate_dict[k.replace("weight", "scale")] = absmax_quantize(state_dict[k].to("CPU"))
  print("saving")
  safe_save(qstate_dict, WEIGHTS_DIR/args.model/f"llama_{args.model}_q8a.safetensors")
  exit(0)
