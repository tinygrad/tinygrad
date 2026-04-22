from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib
from tinygrad import Tensor, nn
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int],
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens),
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode

def _flatten_content(c) -> str:
  return c if isinstance(c, str) else "".join(p["text"] for p in c if p.get("type") == "text")

class Chat:
  """Formats messages into tokens for a given model.

  Two modes:
    - default (simple): uses a small hard-coded preset dispatch keyed on `tokenizer.ggml.pre` for the formatting.
      Covers llama3/qwen2/olmo/kimi-k2/tekken/glm4 chat formats.
    - `use_jinja=True`: renders the GGUF's `tokenizer.chat_template` with the real `jinja2` package.
      Needed for templates using features outside the simple preset set (e.g. Qwen 3.5's macros).
  """
  _PRESETS = ("llama3", "llama-v3", "llama-bpe", "qwen2", "olmo", "kimi-k2", "tekken", "glm4")

  def __init__(self, tok:SimpleTokenizer, template:str|None=None, preset:str="llama3",
               use_jinja:bool=False, extra_stop_ids:typing.Iterable[int]=(), turn_end_id:int|None=None):
    self.tok, self.template, self.use_jinja = tok, template, use_jinja
    self.preset = {"qwen35":"qwen2", "qwen35moe":"qwen2"}.get(preset, preset)
    self.turn_end_id = turn_end_id if turn_end_id is not None else tok.eos_id
    self.stop_ids: set[int] = {x for x in (tok.eos_id, tok.eot_id, self.turn_end_id, *extra_stop_ids) if x is not None}
    if use_jinja:
      if template is None: raise ValueError("use_jinja=True requires tokenizer.chat_template in the GGUF")
    elif self.preset not in self._PRESETS:
      raise ValueError(f"unsupported tokenizer preset {self.preset!r}; pass use_jinja=True to use the GGUF chat_template instead")

  @staticmethod
  def from_gguf_kv(kv:dict, tok:SimpleTokenizer, use_jinja:bool=False) -> 'Chat':
    preset = kv.get('tokenizer.ggml.pre', 'llama3')
    extra: list[int] = []
    turn_end_id: int|None = None
    # OLMo 2: tokenizer.ggml.pre is "dbrx" but the chat format is qwen2-style (<|im_start|>.../<|im_end|>).
    # <|im_end|> terminates turns (not <|endoftext|>) but both should stop generation.
    if kv.get('general.architecture') == 'olmo2':
      preset = 'qwen2'
      im_end = next((i for i,t in enumerate(kv['tokenizer.ggml.tokens']) if t == '<|im_end|>'), None)
      if im_end is not None: extra.append(im_end); turn_end_id = im_end
    return Chat(tok, kv.get('tokenizer.chat_template'), preset, use_jinja, extra, turn_end_id)

  def is_end(self, token_id:int) -> bool: return token_id in self.stop_ids

  def apply(self, messages:list[dict], add_generation_prompt:bool=False, continue_final_message:bool=False) -> list[int]:
    return (self._apply_jinja if self.use_jinja else self._apply_simple)(messages, add_generation_prompt, continue_final_message)

  def _apply_jinja(self, messages, add_generation_prompt, continue_final_message):
    try: import jinja2
    except ImportError as e: raise RuntimeError("use_jinja=True requires the jinja2 package: pip install jinja2") from e
    def tok_str(tid): return self.tok._tok2bytes[tid].decode(errors='replace') if tid is not None and tid in self.tok._tok2bytes else ''
    bos, eos, t = tok_str(self.tok.bos_id), tok_str(self.tok.eos_id), jinja2.Template(self.template)
    if continue_final_message:
      assert messages and messages[-1]["role"] == "assistant", "continue_final_message requires trailing assistant message"
      head = t.render(messages=messages[:-1], add_generation_prompt=True, bos_token=bos, eos_token=eos)
      return self.tok.encode(head + _flatten_content(messages[-1]["content"]))
    return self.tok.encode(t.render(messages=messages, add_generation_prompt=add_generation_prompt, bos_token=bos, eos_token=eos))

  def _apply_simple(self, messages, add_generation_prompt, continue_final_message):
    tok, p, e = self.tok, self.preset, self.turn_end_id

    # role header template (role name interpolated as {0}); llama3 is the default
    role_tmpl = {'qwen2':   "<|im_start|>{0}\n",
                 'olmo':    "<|{0}|>\n",
                 'kimi-k2': "<|im_{0}|>{0}<|im_middle|>",
                 'glm4':    "<|{0}|>"}.get(p, "<|start_header_id|>{0}<|end_header_id|>\n\n")
    def role(r):  # tekken is asymmetric: empty header for assistant
      if p == 'tekken': return tok.encode("[INST]") if r == "user" else []
      return tok.encode(role_tmpl.format(r))

    # end-of-turn token ids; llama3 is the default
    if   p == 'qwen2':   end_turn = [e, *tok.encode("\n")]
    elif p == 'olmo':    end_turn = tok.encode("\n")
    elif p == 'glm4':    end_turn = []
    elif p == 'tekken':  end_turn = tok.encode("[/INST]")
    else:                end_turn = [e]   # llama3, kimi-k2

    prefill = continue_final_message and messages and messages[-1]["role"] == "assistant"
    ids = ([tok.bos_id] if tok.bos_id is not None else []) + (tok.encode("<sop>") if p == 'glm4' else [])
    for i, m in enumerate(messages):
      ids += role(m["role"]) + tok.encode(_flatten_content(m["content"]))
      if not prefill or i < len(messages) - 1: ids += end_turn
    if add_generation_prompt and not prefill: ids += role("assistant")
    return ids

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

# *** simple OpenAI API compatible server with web interface on http://localhost:8000/ ***

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0):
    model, chat = self.server.model, self.server.chat
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = time.perf_counter()
    dec = chat.tok.stream_decoder()
    for next_id in model.generate(ids, temperature=temperature):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if chat.is_end(next_id): break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":dec(next_id)}, "finish_reason":None}], **tmpl}
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    if (tail := dec()): yield {"choices": [{"index":0, "delta":{"content":tail}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def do_POST(self):
    chat = self.server.chat
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens, last assistant message is treated as prefill
      messages = [{"role": m["role"], "content": _flatten_content(m["content"])} for m in body["messages"]]
      prefill = bool(messages) and messages[-1]["role"] == "assistant"
      ids = chat.apply(messages, add_generation_prompt=not prefill, continue_final_message=prefill)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, finish_reason = [], "stop"
        for c in chunks:
          if c["choices"] and c["choices"][0].get("delta", {}).get("content"): out.append(c["choices"][0]["delta"]["content"])
          if c["choices"] and c["choices"][0].get("finish_reason"): finish_reason = c["choices"][0]["finish_reason"]
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, chat:Chat):
    self.model, self.model_name, self.chat = model, model_name, chat
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  parser.add_argument("--jinja", action="store_true", help="Render the GGUF chat_template with the real jinja2 package (needs `pip install jinja2`)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models.get(args.model, args.model))
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  print(f"using model \"{model_name}\" with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # get tokenizer and chat formatter
  tok = SimpleTokenizer.from_gguf_kv(kv)
  chat = Chat.from_gguf_kv(kv, tok, use_jinja=args.jinja)

  # warmup the JIT
  if args.warmup or args.serve:
    # run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, chat).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # interactive chat (falls back to pure completion when the GGUF has no chat template)
  messages: list[dict] = []
  while 1:
    try: user = input('>>> ')
    except EOFError: break
    messages.append({"role": "user", "content": user})
    ids = chat.apply(messages, add_generation_prompt=True)
    dec, assistant = tok.stream_decoder(), []
    for next_id in model.generate(list(ids)):
      sys.stdout.write(dec(next_id) if not chat.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if chat.is_end(next_id): break
      assistant.append(next_id)
    messages.append({"role": "assistant", "content": tok.decode(assistant)})

if __name__ == "__main__": main()
