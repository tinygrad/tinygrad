import argparse
import os
import tiktoken
from examples.gpt2 import Transformer, GPT2
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_gguf, load_state_dict
from tinygrad.tensor import Tensor

def load_gpt2_gguf(fn: str):
  gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
  kv_data, state_dict = load_gguf(gguf_tensor)

  gpt2_params = {
    "dim": kv_data["gpt2.embedding_length"], "n_heads": kv_data["gpt2.attention.head_count"],
    "n_layers": kv_data["gpt2.block_count"], "norm_eps": kv_data["gpt2.attention.layer_norm_epsilon"],
    "vocab_size": 50257, "max_seq_len": kv_data["gpt2.context_length"],
  }
  def remap_gguf_key(key: str):
    replaces = [
      ("blk.", "h."), (".attn_qkv.bias", ".attn.c_attn.bias"), (".attn_qkv.weight", ".attn.c_attn.weight"),
      (".ffn_norm.bias", ".ln_2.bias"), (".ffn_norm.weight", ".ln_2.weight"), (".attn_norm.bias", ".ln_1.bias"),
      (".attn_norm.weight", ".ln_1.weight"), (".attn_output.bias", ".attn.c_proj.bias"), (".attn_output.weight", ".attn.c_proj.weight"),
      (".ffn_up.bias", ".mlp.c_fc.bias"), (".ffn_up.weight", ".mlp.c_fc.weight"), (".ffn_down.bias", ".mlp.c_proj.bias"),
      (".ffn_down.weight", ".mlp.c_proj.weight"), ("token_embd.weight", "wte.weight"), ("output.weight", "lm_head.weight"),
      ("output_norm.bias", "ln_f.bias"), ("output_norm.weight", "ln_f.weight"), ("position_embd.weight", "wpe.weight"),
    ]
    for ostr, ns in replaces: key = key.replace(ostr, ns)
    return key
  state_dict = { remap_gguf_key(k): v for k, v in state_dict.items() }
  model = Transformer(**gpt2_params)
  load_state_dict(model, state_dict)
  return GPT2(model, tiktoken.get_encoding("gpt2"))

if __name__ == "__main__":
  default_prompt = "What is the answer to life, the universe, and everything?"

  parser = argparse.ArgumentParser(description='Run GGUF-GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--quant', type=str, default="Q8_0", choices=["Q8_0", "Q4_0", "Q4_1", "Q6_K"], help="Quantization type.")
  args = parser.parse_args()

  fn = fetch(f"https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.{args.quant}.gguf?download=true")
  gpt2 = load_gpt2_gguf(fn)

  print(gpt2.generate(args.prompt, args.count, args.temperature)[0])