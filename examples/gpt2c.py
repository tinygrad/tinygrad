# Inspired by https://github.com/karpathy/llm.c
import argparse
import tiktoken
from tinygrad import Tensor, Variable
from examples.gpt2 import Transformer
from extra.export_model import export_model
from tinygrad.helpers import getenv, fetch, prod
from tinygrad.runtime.ops_clang import ClangCompiler, ClangProgram
from tinygrad.nn.state import torch_load

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)

VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}
class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    tokenizer = tiktoken.get_encoding("gpt2")

    model = Transformer(**MODEL_PARAMS[model_size])
    weights = torch_load(fetch(f"https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin", name="gpt2_124M"))
    s = Tensor([prod(t.shape) for t in weights.values()]).sum()
    print("weights", len(weights.keys()), s.numpy())
    # special treatment for the Conv1D weights we need to transpose
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    weights['lm_head.weight'] = weights['wte.weight']

    s = Tensor([prod(t.shape) for t in weights.values()]).sum()
    print("weights", len(weights.keys()), s.numpy())

    # TODO: write model

    return GPT2(model, tokenizer)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

if __name__ == "__main__":
  mode = "clang"
  # default_prompt = "<|endoftext|>"
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

  # print(f"using {args.model_size}")
  gpt2 = GPT2.build(args.model_size)
  start_pos = 0
  prompt_tokens = gpt2.tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"})
  toks = [prompt_tokens[:] for _ in range(args.batch_size)]
  tokens = Tensor([x[start_pos:] for x in toks])
  prg, inp_sizes, out_sizes, state = export_model(gpt2.model, mode, tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), args.temperature, fread_model=True)
  cprog = [prg]

  # print(inp_sizes)
  # print(out_sizes)

  inputs = "\n".join([f"float {inp}[{inp_size}];" for inp,inp_size in inp_sizes.items()])
  outputs = "\n".join([f"float {out}[{out_size}];" for out,out_size in out_sizes.items()])
  cprog.append(inputs)
  cprog.append(outputs)

  cprog.append("""
  int main(int argc, char* argv[]) {
    printf("inside");
    fread_net();
    int max_length = 100;
    int toks[max_length];
    int start_pos = 0;
    float temp = 0.8;

    for (int t = 0; t < max_length; t++) {
      input0[0] = toks[start_pos];
      input0[1] = start_pos;
      input0[2] = temp;
      net(input0, output0);
      toks[start_pos+t] = output0[0];
      start_pos += 1;
    }
    for (int t = 0; t < max_length; t++) {
      printf("%d ", toks[t]);
    }
    return 0;
  }""")

  # CLANG=1 python3 examples/gpt2c.py --model_size gpt2 | clang -O2 -lm -x c - -o gpt2 && ./gpt2
  src = '\n'.join(cprog)
  print(src[-100000:])
  p = ClangProgram("main", ClangCompiler().compile(src))
  p()
