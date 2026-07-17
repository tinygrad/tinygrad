from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib
from typing import TYPE_CHECKING
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context, fetch, profile_marker, getenv
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer
if TYPE_CHECKING:
  import jinja2

def stream_split(buf:str, tag:str, final:bool) -> tuple[str, str, bool]:
  # split buf on the first full tag into (before, rest); hold back a partial tag at the end unless final
  if tag in buf:
    before, rest = buf.split(tag, 1)
    return before, rest, True
  hold = max((i for i in range(1, min(len(buf), len(tag))+1) if tag.startswith(buf[-i:])), default=0) if not final else 0
  return buf[:len(buf)-hold], buf[len(buf)-hold:], False

def parse_tool_call(s:str) -> tuple[str, typing.Any]|None:
  s = s.strip()
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  if (fm := re.match(r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", s, re.DOTALL)):
    args = {}
    for pm in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", fm.group(2), re.DOTALL):
      try: args[pm.group(1)] = json.loads(pm.group(2))
      except json.JSONDecodeError: args[pm.group(1)] = pm.group(2)
    return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect string content (not null) and tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    if m.get("content") is None: m["content"] = ""
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .text for the final parse
  def __init__(self, parse_tool_calls:bool, prefill_think:bool):
    self.parse_tool_calls, self.text, self.buf, self.fresh = parse_tool_calls, "", "", False
    self.mode = "reasoning" if prefill_think else "undecided"  # output inside a think block is sent as reasoning_content
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.text += piece
    self.buf += piece
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len("<think>") and "<think>".startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len("<think>"):].lstrip("\n")) if self.buf.startswith("<think>") else ("content", self.buf)
    if self.mode == "reasoning":
      emit, self.buf, done = stream_split(self.buf, "</think>", final)
      if emit: yield "reasoning_content", emit
      if not done: return
      self.mode, self.fresh = "content", True
    if self.fresh:  # strip blank lines right after the think block
      self.buf = self.buf.lstrip("\n")
      if not self.buf and not final: return
      self.fresh = False
    if self.mode == "tool": return  # inside a tool_call tag: swallow, self.text has everything for the final parse
    if self.parse_tool_calls:
      emit, self.buf, found = stream_split(self.buf, "<tool_call>", final)
      if emit: yield "content", emit
      if found: self.mode = "tool"
    else:
      yield "content", self.buf
      self.buf = ""

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
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
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
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
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

# *** simple OpenAI API compatible server with web interface on http://localhost:8000/ ***

class FallbackTemplate:
  # minimal jinja2.Template-compatible chat template without jinja2, no tool calling support
  def __init__(self, tok:SimpleTokenizer): self.tok = tok
  def role(self, role:str) -> str:
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"  # OLMoE Instruct format
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset == 'qwen2': return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self) -> str:
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset == 'qwen2': return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True) -> str:
    out = self.tok.decode([] if self.tok.bos_id is None else [self.tok.bos_id]) + ("<sop>" if self.tok.preset == 'glm4' else "")
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                parse_tool_calls=False, prefill_think=False):
    model, tok = self.server.model, self.server.tok
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    yield chunk({"role":"assistant", "content":""})
    out: list[int] = []
    finish_reason = "stop"
    st = time.perf_counter()
    dec = tok.stream_decoder()
    router = StreamRouter(parse_tool_calls, prefill_think)
    for next_id in model.generate(ids, temperature=temperature):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if tok.is_end(next_id): break
      out.append(next_id)
      for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
    if parse_tool_calls:
      tool_calls: list[dict] = []
      for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", router.text, re.DOTALL):
        if (parsed := parse_tool_call(m.group(1))) is None:
          stderr_log(f"failed to parse tool call: {m.group(1)[:200]}")
          yield chunk({"content":m.group(0)})  # don't silently drop output the client can't use
        else:
          name, args = parsed
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      from_template = not isinstance(self.server.template, FallbackTemplate)  # only the jinja path supports tools
      if from_template: normalize_messages(body["messages"])
      rendered = self.server.template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True)
      ids: list[int] = self.server.tok.encode(rendered)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              parse_tool_calls=from_template and bool(body.get("tools")), prefill_think=rendered.rstrip().endswith("<think>"))
      if body.get("stream"): self.stream_json(chunks)
      else:
        chunks = list(chunks)
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"] and c["choices"][0].get("delta")]
        finish_reason = next((c["choices"][0]["finish_reason"] for c in reversed(chunks)
                              if c["choices"] and c["choices"][0].get("finish_reason")), "stop")
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(d.get("content") or "" for d in deltas) or None}
        if (reasoning := "".join(d.get("reasoning_content") or "" for d in deltas)): message["reasoning_content"] = reasoning
        if (tool_calls := [{k:v for k, v in tc.items() if k != "index"} for d in deltas for tc in d.get("tool_calls", [])]):
          message["tool_calls"] = tool_calls
        self.send_data(json.dumps({**chunks[-1], "object":"chat.completion",
          "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:jinja2.Template|FallbackTemplate):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params, "
        f"max context {args.max_context} on {nn.state.get_parameters(model)[0].device}")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if (ct := kv.get('tokenizer.chat_template')) is not None:
    try:
      import jinja2
      env = jinja2.Environment()
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # warmup the JIT
  if args.warmup or args.serve:
    # run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, template).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
    exit(0)

  # interactive chat
  messages: list[dict] = []
  while 1:
    try: messages.append({"role":"user", "content":input('>>> ')})
    except EOFError: break
    ids = tok.encode(template.render(messages=messages, add_generation_prompt=True))
    reply, dec = "", tok.stream_decoder()
    for next_id in model.generate(ids):
      if tok.is_end(next_id):
        sys.stdout.write(dec() + "\n\n")
        break
      reply += (piece := dec(next_id))
      sys.stdout.write(piece)
      sys.stdout.flush()
    messages.append({"role":"assistant", "content":reply})

if __name__ == "__main__": main()
