# Inspired by https://github.com/karpathy/llm.c
import argparse
import tiktoken
from tinygrad import Tensor, Variable, dtypes, Device
from examples.gpt2 import Transformer
from extra.export_model import export_model
from tinygrad.helpers import getenv, fetch, prod, flatten, _cache_dir
from tinygrad.runtime.ops_clang import ClangCompiler, ClangProgram
from tinygrad.nn.state import torch_load
from ctypes import c_char_p

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)

VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}
def write_fp32(tensor: Tensor, file):
  file.write(tensor.cast(dtypes.float32).to(Device.DEFAULT).numpy().tobytes())

class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    tokenizer = tiktoken.get_encoding("gpt2")
    model = Transformer(**MODEL_PARAMS[model_size])
    return GPT2(model, tokenizer)
  
  @staticmethod
  def write_model(file_path, model_size="gpt2"):
    weights = torch_load(fetch(f"https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin"))
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    weights['lm_head.weight'] = weights['wte.weight']

    with open(file_path, "wb") as file:
      write_fp32(weights['wpe.weight'], file)
      write_fp32(weights['wte.weight'], file)
      # TODO: layernorm weights not used in clang, investigate
      for i in range(12):
        write_fp32(weights[f'h.{i}.attn.c_attn.weight'], file)
        write_fp32(weights[f'h.{i}.attn.c_attn.bias'], file)
        write_fp32(weights[f'h.{i}.attn.c_proj.weight'], file)
        write_fp32(weights[f'h.{i}.attn.c_proj.bias'], file)
        write_fp32(weights[f'h.{i}.mlp.c_fc.weight'], file)
        write_fp32(weights[f'h.{i}.mlp.c_fc.bias'], file)
        write_fp32(weights[f'h.{i}.mlp.c_proj.weight'], file)
        write_fp32(weights[f'h.{i}.mlp.c_proj.bias'], file)
      write_fp32(weights['lm_head.weight'], file)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

if __name__ == "__main__":
  mode = "clang"
  default_prompt = "What is the answer to life, the universe, and everything?"

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--seed', type=int, help="Set the random seed")
  parser.add_argument('--batch_size', type=int, default=1, help="Set the input batch size")
  parser.add_argument('--benchmark', type=int, default=-1, help="Benchmark GPT with the given number of tokens")
  parser.add_argument('--noshow', action='store_true', help="Don't show the output")
  args = parser.parse_args()

  print(f"using {args.model_size}")
  gpt2 = GPT2.build(args.model_size)
  file_path = _cache_dir + f"/tinygrad/downloads/{args.model_size}.bin"
  gpt2.write_model(file_path, args.model_size)
  start_pos = 0
  start_pos_v = Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos)
  prompt_tokens = gpt2.tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"})
  toks = [prompt_tokens[:] for _ in range(args.batch_size)]
  tokens = Tensor([x[start_pos:] for x in toks])

  prg, net_inputs, net_outputs, state = export_model(gpt2.model, mode, tokens, start_pos_v, args.temperature, fread_weights=file_path)
  cprog = [prg]

  inputs = "\n".join([f"{dtype.name} {name}[{sz+args.count}];" for name,sz,dtype,_,_ in net_inputs])
  outputs = "\n".join([f"{dtype.name} {name}[{sz}];" for name,sz,dtype,_,_ in net_outputs])
  cprog.append(inputs)
  cprog.append(outputs)

  cprog.append("""
#include <string.h>
int main(int argc, char* argv[]) {
  int max_length = 100;
  int toks[max_length];
  int start_pos = argc-1;

  for (int i = 0; i < argc-1; i++) {
    input0[i] = atoi(argv[i+1]);
  }
  assert(input0 != NULL);

  fread_net();

  for (int t = 0; t < max_length; t++) {
    printf("generating token %d\\n", t);
    net(start_pos, input0, output0);
    toks[t] = output0[0];
    input0[t+1] = output0[0];
    start_pos += 1;
  }
  for (int t = 0; t < max_length; t++) {
    printf("%d ", toks[t]);
  }
  printf("\\n");
  return 0;
}""")

  # CLANG=1 python3 examples/gpt2c.py --model_size gpt2 | clang -O2 -lm -x c - -o gpt2 && ./gpt2
  src = '\n'.join(cprog)
  # with open("output.c", "w") as f:
  #   f.write(src)
  p = ClangProgram("main", ClangCompiler().compile(src))
  # # NOTE: only works for batch_size 1 right now
  toks = flatten(toks)
  p(1+len(toks), (c_char_p * (1+len(toks)))(b'', *[bytes(str(t), 'utf-8') for t in toks]))
