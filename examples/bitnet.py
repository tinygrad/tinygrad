#!/usr/bin/env python3

from pathlib import Path
from typing import List
import json, argparse, random, time, os

# Fix ARM64 clang compilation issue
os.environ['CLANG_FLAGS'] = '-O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident'

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinygrad import Device

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.bitnet import BitNetConfig, BitNetForCausalLM, build_transformer, sample
from tinygrad.nn.state import get_parameters
from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.helpers import fetch


class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]

  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode(toks)
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())


# Default settings
TEMPERATURE = 0.85
TOP_K       = 0
TOP_P       = 0.0
ALPHA_F     = 0.0
ALPHA_P     = 0.0


def prefill(model, prompt_ids, past_key_values=None):
    """Prefill model with prompt tokens."""
    if not prompt_ids:
        return past_key_values
    
    prompt_tensor = Tensor([prompt_ids], device=model.device)
    _, past_key_values = model(prompt_tensor, past_key_values)
    
    return past_key_values


if __name__ == "__main__":
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description="Run BitNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--download_model", action="store_true", help="Download the model specified by --model if it doesn't exist")
  parser.add_argument("--model", type=Path, help="Path to the model directory or file")
  parser.add_argument("--size", type=str, default="2B", choices=['2B'], help="Size of model to use")
  parser.add_argument("--shard", type=int, default=1, help="Number of shards to use")
  parser.add_argument("--no_api", action="store_true", help="Do not start the Gradio API")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
  parser.add_argument("--port", type=int, default=7776, help="Web server port")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data")
  parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
  parser.add_argument("--count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--device", type=str, default=Device.DEFAULT, help="Device to use (e.g., METAL, CUDA, CPU)")
  args = parser.parse_args()

  # download_model is the default without a model passed in
  if args.download_model or not args.model:
    # bitnet uses the same tokenizer as llama3
    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="bitnet") 
    args.model = fetch("https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors", "model.safetensors", subdir="bitnet")

  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  # Use absolute path for tokenizer model
  model_dir = args.model if args.model.is_dir() else args.model.parent
  tokenizer_path = (model_dir / "tokenizer.model").resolve()
  if not tokenizer_path.is_file():
    raise FileNotFoundError(f"Tokenizer model not found at expected path: {tokenizer_path}")
  
  tokenizer = Tokenizer(str(tokenizer_path))

  def encode_role(role: str):
    # Flatten the list concatenation
    return [tokenizer.special_tokens["<|start_header_id|>"], ] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"], ] + tokenizer.encode("\n\n")

  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"], ]

  base_device = args.device
  device = tuple(f"{base_device}:{i}" for i in range(args.shard)) if args.shard > 1 else base_device
  model = build_transformer(args.model)[0]

  param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(model))
  print(f"ram used: {param_bytes/1e9:.2f} GB")

  if not args.no_api and not args.benchmark:
    from bottle import Bottle, request, response, HTTPResponse, abort, static_file
    app = Bottle()

    cors_headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
      "Access-Control-Allow-Credentials": "true",
    }
    @app.hook("before_request")
    def handle_options():
      if request.method == "OPTIONS": raise HTTPResponse(headers=cors_headers)
    @app.hook("after_request")
    def enable_cors():
      for key, value in cors_headers.items(): response.set_header(key, value)

    @app.route("/<filename>")
    def server_static(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat").as_posix())
    @app.route("/assets/<filename:path>")
    def server_assets(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat" / "assets").as_posix())
    @app.route("/")
    def index():
      return static_file("index.html", root=(Path(__file__).parent / "tinychat").as_posix())

    @app.get("/v1/models")
    def models():
      return json.dumps([str(args.model)])

    @app.post("/v1/internal/token-count")
    def token_count():
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))

    @app.post("/v1/token/encode")
    def token_encode():
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", "")))

    @app.post("/v1/completions")
    def completions():
      try:
        rjson = json.loads(request.body.read())
        
        prompt = rjson.get("prompt", "")
        max_tokens = rjson.get("max_tokens", 100)
        temperature = rjson.get("temperature", TEMPERATURE)
        top_k = rjson.get("top_k", TOP_K)
        top_p = rjson.get("top_p", TOP_P)
        stream = rjson.get("stream", False)
        
        prompt_ids = tokenizer.encode(prompt)
        
        # Check if we are streaming
        if stream:
          response.content_type = "text/event-stream"
          response.set_header("Cache-Control", "no-cache")
        else:
          abort(400, "streaming required")
        
        # Prefill
        past = prefill(model, prompt_ids)
        
        # Generate with streaming
        last_tok = prompt_ids[-1] if prompt_ids else tokenizer.bos_id
        for i in range(max_tokens):
          GlobalCounters.reset()
          token, past, logits = model(Tensor([[last_tok]], device=Device.DEFAULT), past, temperature, top_k, top_p, ALPHA_F, ALPHA_P)
          last_tok = token
          
          if token in tokenizer.stop_tokens:
            break
          
          # Stream the token
          res = {
            "choices": [{
              "text": tokenizer.decode([token]),
            }]
          }
          yield f"data: {json.dumps(res)}\n\n"
          
      except Exception as e:
        print(f"[ERROR] Error in completions: {e}")
        import traceback
        traceback.print_exc()
        abort(500, f"Internal server error: {e}")

    @app.post("/v1/chat/token/encode")
    def chat_token_encode():
      rjson = json.loads(request.body.read())
      messages = rjson.get("messages", [])
      
      token_sequence = [tokenizer.bos_id]
      for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        token_sequence.extend(encode_message(role, content))
      
      return json.dumps(token_sequence)

    @app.post("/v1/chat/completions")
    def chat_completions():
      try:
        rjson = json.loads(request.body.read())
        
        messages = rjson.get("messages", [])
        max_tokens = rjson.get("max_tokens", 100)
        temperature = rjson.get("temperature", TEMPERATURE)
        top_k = rjson.get("top_k", TOP_K)
        top_p = rjson.get("top_p", TOP_P)
        stream = rjson.get("stream", False)
        
        # Check if we are streaming
        if stream:
          response.content_type = "text/event-stream"
          response.set_header("Cache-Control", "no-cache")
        else:
          abort(400, "streaming required")
        
        # Build token sequence
        token_sequence = [tokenizer.bos_id]
        
        for message in messages:
          role = message.get("role", "user")
          content = message.get("content", "")
          message_tokens = encode_message(role, content)
          token_sequence.extend(message_tokens)
        
        # Add assistant role start
        token_sequence.extend(encode_role("assistant"))
        
        # Prefill with the entire conversation
        past = prefill(model, token_sequence)
        
        # Generate response with streaming
        random_id = random.randbytes(16).hex()
        last_tok = token_sequence[-1]
        
        for i in range(max_tokens):
          GlobalCounters.reset()
          token, past, logits = model(Tensor([[last_tok]], device=Device.DEFAULT), past, temperature, top_k, top_p, ALPHA_F, ALPHA_P)
          last_tok = token
          
          # Check for stop tokens
          if token in tokenizer.stop_tokens:
            break
          
          # Stream the token
          res = {
            "id": random_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(args.model),
            "choices": [{
              "index": 0,
              "delta": {
                "role": "assistant",
                "content": tokenizer.decode([token]),
              },
              "finish_reason": None,
            }]
          }
          yield f"data: {json.dumps(res)}\n\n"
        
        # Send the final chunk
        res = {
          "id": random_id,
          "object": "chat.completion.chunk",
          "created": int(time.time()),
          "model": str(args.model),
          "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
          }]
        }
        yield f"data: {json.dumps(res)}\n\n"
          
      except Exception as e:
        print(f"[ERROR] Error in chat_completions: {e}")
        import traceback
        traceback.print_exc()
        abort(500, f"Internal server error: {e}")

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

  else:
    # Benchmark or direct prompt mode
    if args.benchmark:
      prompt = "The quick brown fox"
    else:
      prompt = args.prompt or "Hello, how are you?"
    
    print(f"Running with prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt)
    print(f"Encoded to {len(prompt_ids)} tokens: {prompt_ids}")
    
    # Prefill
    past = prefill(model, prompt_ids)
    
    # Generate
    generated_tokens = []
    for i in range(args.count):
      if i == 0:
        # First token after prefill
        token, past, logits = model(Tensor([[prompt_ids[-1]]], device=Device.DEFAULT), past, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
      else:
        # Subsequent tokens
        token, past, logits = model(Tensor([[generated_tokens[-1]]], device=Device.DEFAULT), past, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
      
      generated_tokens.append(token)
      
      if token in tokenizer.stop_tokens:
        break
      
      # Print token as we generate
      print(tokenizer.decode([token]), end='', flush=True)
    
    print()  # Final newline
    print(f"Generated {len(generated_tokens)} tokens") 