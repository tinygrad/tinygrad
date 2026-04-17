from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib
from tinygrad import Tensor, nn
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer
from tinygrad.llm.xgrammar_support import ConstraintUnavailableError, build_constraint, constrain_messages, content_to_text, make_token_selector, parse_tool_calls, serialize_assistant_tool_calls

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]: return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
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
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]: return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

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
  "qwen3.5:2b": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

class Handler(HTTPRequestHandler):
  server: 'LLMServer'
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")

  def _message_text(self, msg:dict[str, typing.Any], constraint) -> tuple[str, str]:
    role = msg["role"]
    if role == "assistant" and msg.get("tool_calls"):
      style = getattr(constraint, "tool_style", None) or "qwen"
      text = serialize_assistant_tool_calls(msg["tool_calls"], style)
      content = content_to_text(msg.get("content"))
      return role, (content + ("\n" if content and text else "") + text).strip()
    if role == "tool":
      name = msg.get("name") or msg.get("tool_call_id") or "tool"
      return "user", f"Tool result from {name}:\n{content_to_text(msg.get('content'))}"
    return role, content_to_text(msg.get("content"))

  def build_ids(self, body:dict[str, typing.Any], constraint) -> list[int]:
    tok = self.server.tok
    ids: list[int] = tok.prefix()
    messages = constrain_messages(list(body["messages"]), constraint)
    for i, msg in enumerate(messages):
      role, text = self._message_text(msg, constraint)
      ids += tok.role(role)
      ids += tok.encode(text)
      if role == "assistant" and i == len(messages) - 1: break
      ids += tok.end_turn()
    else: ids += tok.role("assistant")
    return ids

  def _generate_text(self, ids:list[int], model_name:str, constraint=None, max_tokens:int|None=None, temperature:float=0.0):
    model, tok = self.server.model, self.server.tok
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    selector = make_token_selector(self.server, constraint, temperature) if constraint is not None else None
    parts, out = [], []
    finish_reason = "stop"
    st = time.perf_counter()
    dec = tok.stream_decoder()
    pt = st
    for next_id in model.generate(ids, temperature=temperature, token_selector=selector, constraint=constraint):
      if len(out) == 0: pt = time.perf_counter(); stderr_log(f"prefill:{(len(ids)-cache_start_pos)/max(pt-st, 1e-9):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if tok.is_end(next_id): break
      out.append(next_id)
      parts.append(dec(next_id))
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    if (tail := dec()): parts.append(tail)
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/max(et-pt, 1e-9) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")
    return "".join(parts), parts, finish_reason, {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}

  def _tool_stream_chunks(self, tool_calls:list[dict[str, typing.Any]], model_name:str, usage:dict[str, int]|None):
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant"}, "finish_reason":None}], **tmpl}
    for idx, tool_call in enumerate(tool_calls):
      yield {"choices": [{"index":0, "delta":{"tool_calls":[{"index":idx, "id":tool_call["id"], "type":"function", "function":tool_call["function"]}]}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"tool_calls"}], **tmpl}
    if usage is not None: yield {"choices": [], "usage": usage, **tmpl}

  def _text_stream_chunks(self, parts:list[str], finish_reason:str, model_name:str, usage:dict[str, int]|None):
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    for part in parts:
      if part: yield {"choices": [{"index":0, "delta":{"content":part}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if usage is not None: yield {"choices": [], "usage": usage, **tmpl}

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path != "/v1/chat/completions": raise RuntimeError(f"unhandled path {self.path}")
    try: constraint = build_constraint(self.server, body)
    except ConstraintUnavailableError as e: raise RuntimeError(str(e))
    ids = self.build_ids(body, constraint)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    text, parts, finish_reason, usage = self._generate_text(ids, body["model"], constraint=constraint, max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)))
    include_usage = (not body.get("stream")) or body.get("stream_options",{}).get("include_usage", False)
    if constraint is not None and constraint.mode == "tools":
      content, tool_calls = parse_tool_calls(text, constraint.tool_style or "qwen")
      if tool_calls:
        if body.get("stream"): self.stream_json(self._tool_stream_chunks(tool_calls, body["model"], usage if include_usage else None))
        else:
          payload = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion", "created":int(time.time()), "model":body["model"],
            "choices":[{"index":0, "message":{"role":"assistant","content":content, "tool_calls":[{k:v for k,v in tc.items() if k != 'index'} for tc in tool_calls]}, "finish_reason":"tool_calls"}],
            "usage":usage}
          self.send_data(json.dumps(payload).encode())
        return
    if body.get("stream"): self.stream_json(self._text_stream_chunks(parts, finish_reason, body["model"], usage if include_usage else None))
    else:
      payload = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion", "created":int(time.time()), "model":body["model"],
        "choices":[{"index":0, "message":{"role":"assistant","content":text}, "finish_reason":finish_reason}], "usage":usage}
      self.send_data(json.dumps(payload).encode())

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer):
    self.model, self.model_name, self.tok = model, model_name, tok
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()
  raw_model = Tensor.from_url(models.get(args.model, args.model))
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  print(f"using model \"{model_name}\" with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model
  import gc; gc.collect()
  tok = SimpleTokenizer.from_gguf_kv(kv)
  if args.warmup or args.serve:
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))
  if args.serve: LLMServer(('', args.serve), model, model_name, tok).serve_forever()
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)
  ids: list[int] = tok.prefix()
  while 1:
    try: ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn() + tok.role("assistant")
    except EOFError: break
    dec = tok.stream_decoder()
    for next_id in model.generate(ids):
      sys.stdout.write(dec(next_id) if not tok.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if tok.is_end(next_id): break

if __name__ == "__main__": main()
