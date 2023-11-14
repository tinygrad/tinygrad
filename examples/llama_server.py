import socket
import numpy as np
import sys
from llama import LLaMa
from pathlib import Path
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
import argparse
from tinygrad.helpers import Timing, getenv, DEBUG, dtypes, CI
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad.helpers import GlobalCounters
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
from tinygrad.shape.symbolic import Variable
import threading


if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description="Run LLaMA in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--personality", type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data to out.prof")
  parser.add_argument("--size", type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
  parser.add_argument("--gen", default="1", help="Generation of the model to use ['1', '2', 'code']")
  parser.add_argument("--quantize", action="store_true", help="Quantize the weights to int8 in memory")
  parser.add_argument("--model", type=Path, default=None, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")

  args = parser.parse_args()

  # *** prompt engineers work here ****

  if args.personality.lower() == "stacy":
    pre_prompt = f"""Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
    examples = {
      "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
      "french revolution was what year?": "The French Revolution started in 1789, and lasted 10 years until 1799.",
      "What is bigger, the moon or the sun?": "The sun is bigger than the moon, except when Mercury is in retrograde.",
    }
    
    user_delim = "\nUser: "
    resp_delim = "Stacy: "
    end_delim = " [EOS]\n"
    pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())
    print(pre_prompt)
  
  LLAMA_SUFFIX = {"1": "", "2": "-2", "code": "-code"}[args.gen]
  MODEL_PATH = args.model or Path(__file__).parents[1] / f"weights/LLaMA{LLAMA_SUFFIX}/{args.size}"
  TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"
  print(f"using LLaMA{LLAMA_SUFFIX}-{args.size} model")
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen=args.gen, model_size=args.size, quantize=args.quantize)

  # encode pre prompt
  toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)

  print(f"Preparing KV cache for chatbot with personality {args.personality}...")
  with Timing():
    llama.model(Tensor([toks]), 0, args.temperature).realize()  # NOTE: outputs are not used
  start_pos = len(toks)

  outputted = llama.tokenizer.decode(toks)
  sys.stdout.write(outputted)
  sys.stdout.flush()

  # Server code because my laptop is shit
  host, port = "0.0.0.0", 5000
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.bind((host, port))
  server.listen(5)
  print(f"Server listening on port {port}...")
  
  while True:
    client, addr = server.accept()
    print(f"Accepted connection from {addr[0]}:{addr[1]}")
    while True:
      user_data = client.recv(1024).decode()
      user_prompt = user_delim + user_data + "\n"
      outputted += user_prompt

      new_toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(outputted)
      assert toks == new_toks[:len(toks)]
      toks = new_toks
      assert outputted == llama.tokenizer.decode(toks)

      last_break = len(outputted)
      for i in range(args.count):
        GlobalCounters.reset()

        st = GlobalCounters.time_sum_s
        with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
          with Timing("ran model in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                      f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                      (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):
            probs = llama.model(Tensor([toks[start_pos:]]), start_pos, args.temperature).realize()
          probs_np = probs.numpy()
          tok = int(np.random.choice(len(probs_np), p=probs_np))

        # use the kv cache
        start_pos = len(toks)

        # add the new token
        toks.append(tok)

        # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
        cur = llama.tokenizer.decode(toks)
        sys.stdout.write(cur[len(outputted):])
        sys.stdout.flush()
        outputted = cur

        if outputted.endswith(end_delim):
          client.sendall(outputted.splitlines()[-1].encode())
    

