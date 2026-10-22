from __future__ import annotations
import json, pathlib, queue, re, socketserver, threading, time, traceback, typing, uuid
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log, getenv
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
if TYPE_CHECKING:
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

# the markup of reasoning and tool calls. a model that has a tag as a special token is parsed on the token: the tag decodes to a sentinel (a
# private use character), the same text written by the model (in prose, in code it writes) stays text
TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")
def markup_tags(special_tokens:dict[str, int]) -> tuple[dict[str, str], dict[int, str]]:
  """the string of every tag in the decoded output, and the sentinels of the special token ids"""
  sentinels = {tag: chr(0xE000 + i) for i, tag in enumerate(TAGS) if tag in special_tokens}
  return {tag: sentinels.get(tag, tag) for tag in TAGS}, {special_tokens[tag]: s for tag, s in sentinels.items()}
TEXT_TAGS = {tag: tag for tag in TAGS}
def unmark(s:str) -> str:
  """the sentinels back to their tags"""
  for i, tag in enumerate(TAGS): s = s.replace(chr(0xE000 + i), tag)
  return s

def parse_tool_call(s:str, tags:dict[str, str]=TEXT_TAGS) -> tuple[str, typing.Any]|None:
  s = s.strip()
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  # GLM format: name<arg_key>key</arg_key><arg_value>value</arg_value>...
  # the XML format puts the value on its own lines (the newlines around it aren't part of it), GLM's is exact: a value is sent back as it was
  # generated (the next prompt renders it, the model's cache is reused)
  ak, akc, av, avc = (re.escape(tags[t]) for t in ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"))
  patterns = (
    (r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", r"<parameter=([^>]+)>(.*?)</parameter>", True),
    # a value ends at the </arg_value> before the next <arg_key> or the end (as text tags, a value can hold the markup: code that parses it)
    (rf"([\w.-]+)\s*((?:{ak}.+?{akc}\s*{av}.*?{avc}\s*)*)$", rf"{ak}(.+?){akc}\s*{av}(.*?){avc}(?=\s*(?:{ak}|$))", False))
  for call_pattern, arg_pattern, own_lines in patterns:
    if (fm := re.match(call_pattern, s, re.DOTALL)):
      args = {}
      for pm in re.finditer(arg_pattern, fm.group(2), re.DOTALL):
        value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2)) if own_lines else pm.group(2)
        try: args[pm.group(1)] = json.loads(value)
        except json.JSONDecodeError: args[pm.group(1)] = value
      return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # string-only templates need text parts joined and explicit null content emptied
    if m.get("content", "") is None: m["content"] = ""
    elif isinstance(c := m.get("content"), list) and all(p.get("type") == "text" for p in c): m["content"] = "".join(p["text"] for p in c)

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse. tags: the string of every tag
  def __init__(self, reasoning:bool=False, tags:dict[str, str]=TEXT_TAGS):
    self.buf, self.tags = "", tags
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
  def split(self, tag:str, final:bool) -> tuple[str, bool]:
    # split buf on the first full tag, holding back a partial tag at the end unless final
    if tag in self.buf:
      before, self.buf = self.buf.split(tag, 1)
      return before, True
    hold = max((i for i in range(1, min(len(self.buf), len(tag))+1) if tag.startswith(self.buf[-i:])), default=0) if not final else 0
    emit, self.buf = self.buf[:len(self.buf)-hold], self.buf[len(self.buf)-hold:]
    return emit, False
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    think, think_end, call = self.tags["<think>"], self.tags["</think>"], self.tags["<tool_call>"]
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len(think) and think.startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len(think):]) if self.buf.startswith(think) else ("content", self.buf)
    if self.mode == "reasoning":
      emit, done = self.split(think_end, final)
      if emit: yield "reasoning_content", unmark(emit)
      if not done: return
      self.mode = "content"
    if self.mode == "tool": return
    emit, found = self.split(call, final)
    if emit: yield "content", unmark(emit)
    if found: self.mode, self.buf = "tool", call + self.buf

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, log:str=""):
    tok = self.server.tok
    prompt_tokens = len(ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    dec = tok.stream_decoder()
    tags, sentinels = markup_tags(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {})
    router = StreamRouter(reasoning, tags)
    # the model runs on the server's thread (LLMServer.serve_forever): the request waits there, its tokens come back here
    req = Request(ids, log, max_tokens, temperature)
    self.server.waiting.put(req)
    try:
      yield chunk({"role":"assistant", "content":""})
      while (next_id := req.out.get()) is not None:
        out.append(next_id)
        for field, delta in router.route(sentinels[next_id] if next_id in sentinels else dec(next_id)): yield chunk({field:delta})
      if req.error is not None:
        yield {"error":{"message":f"{type(req.error).__name__}: {req.error}", "type":"server_error", "code":"generation_failed"}}
        return
      finish_reason = req.finish_reason
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      # a call ends at the </tool_call> before the next <tool_call> or the end (as text tags, its arguments can hold the markup)
      call, call_end = re.escape(tags["<tool_call>"]), re.escape(tags["</tool_call>"])
      for m in re.finditer(rf"{call}\s*(.*?)\s*(?:{call_end}(?=\s*(?:{call}|$))|$)", router.buf, re.DOTALL):
        if (parsed := parse_tool_call(m.group(1), tags)) is None:
          stderr_log(f"failed to parse tool call: {unmark(m.group(1))[:200]}\n")
          yield chunk({"content":unmark(m.group(0))})  # don't silently drop output the client can't use
        else:
          name, args = parsed
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":unmark(args) if isinstance(args, str) else
                                         json.dumps(json.loads(unmark(json.dumps(args, ensure_ascii=False))))}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
      yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
      if include_usage:
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out)}, **tmpl}
    finally: req.cancelled = True  # (a client gone: the model stops at its next token)

  def do_POST(self):
    request_st = time.perf_counter()
    log = f"{self.path}  {colored('--', 'BLACK')}  "
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      normalize_messages(body["messages"])
      rendered = self.server.template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, preserve_thinking=True)
      ids: list[int] = self.server.tok.encode(rendered)
      log += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
      if len(ids) >= self.server.model.max_context:
        stderr_log(log + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              reasoning=rendered.rstrip().endswith("<think>"), log=log)
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, reasoning, tool_calls, finish_reason = [], [], [], "stop"
        for c in chunks:
          if "error" in c: return self.send_data(json.dumps(c).encode(), status_code=500)
          if not c["choices"]: continue
          choice = c["choices"][0]
          if (delta := choice.get("delta", {})):
            if delta.get("content"): out.append(delta["content"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            tool_calls += [{k:v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", [])]
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out) or None}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

def _close(g:typing.Iterator) -> None:
  if (close:=getattr(g, "close", None)) is not None: close()  # (a generator: the model stops where it is)

class Request:
  """a request to the model: its prompt, its sampled tokens come back in out (None: the end), it's cancelled when the client goes away"""
  def __init__(self, ids:list[int], log:str, max_tokens:int|None, temperature:float):
    self.ids, self.log, self.max_tokens, self.temperature = ids, log, max_tokens, temperature
    self.cancelled, self.finish_reason = False, "stop"
    self.error: Exception|None = None
    self.out: queue.Queue[int|None] = queue.Queue()
    self.generated: list[int] = []
    self.st, self.pt, self.cache_start, self.steps = time.perf_counter(), 0.0, 0, 0
    self.pos, self.done = 0, False  # batched decoding: the position of its last token (the next step's input)

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  """
  requests come in on threads of their own, the model runs on the server's thread: one request alone is decoded as before (speculative
  decoding, its reply cached for the next turn), many at the same time are batched (Transformer.batch_step): every one has a slot of the
  model's states, a step gives the next token of all of them. a request that may not fit a slot (prompt + max_tokens) waits for the model
  alone, one that fills its slot goes on alone (its tokens prefilled again)
  """
  daemon_threads = True
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    self.waiting: queue.Queue[Request] = queue.Queue()
    # the slots of batched decoding (none: one request at a time)
    self.slots = n if isinstance(n:=getattr(model, "batch_slots", lambda: 0)(), int) else 0
    super().__init__(server_address, Handler)

  def serve_forever(self, poll_interval:float=0.5):
    # the model runs here, not on a thread of its own: BEAM's _try_compile installs a SIGALRM handler, which raises ValueError off the main thread
    threading.Thread(target=super().serve_forever, args=(poll_interval,), daemon=True, name="HTTPServer").start()
    self.schedule()

  # ** the requests' tokens and their end
  def emit(self, req:Request, tok:int) -> bool:
    """a sampled token of req: sent, or the end. returns whether req goes on"""
    if not req.generated and not req.pt:
      req.pt = time.perf_counter()
      stderr_log(req.log + f"in:{colored(f'{req.cache_start:5d}', 'green')} +{len(req.ids)-req.cache_start:5d}  {colored('--', 'BLACK')}  "
                 f"prefill:{(len(req.ids)-req.cache_start)/(req.pt-req.st):5.0f} tok/s\n" if self.verbose_prefill else "")
    if req.cancelled or self.tok.is_end(tok): return self.finish(req)
    req.generated.append(tok)
    req.out.put(tok)
    if req.max_tokens is not None and len(req.generated) >= req.max_tokens:
      req.finish_reason = "length"
      return self.finish(req)
    return True
  verbose_prefill = False
  def finish(self, req:Request, error:Exception|None=None) -> bool:
    if req.done: return False
    req.done, req.error = True, error
    et, n = time.perf_counter(), len(req.generated)
    total = f"total:{et-req.st:6.2f}s"
    pre = (len(req.ids) - req.cache_start) / ((req.pt or et) - req.st)
    spec = f"spec:{n/req.steps:4.2f} tok/step  {colored('--', 'BLACK')}  " if req.steps else ""
    stderr_log(req.log + f"in:{colored(f'{req.cache_start:5d}', 'green')} +{len(req.ids)-req.cache_start:5d}  {colored('--', 'BLACK')}  "
               f"prefill:{pre:5.0f} tok/s  {colored('--', 'BLACK')}  gen:{(n-1)/(et-req.pt) if n > 1 and req.pt else 0:4.0f} tok/s  "
               f"{colored('--', 'BLACK')}  {spec}out:{n:5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if req.cancelled else total}\n")
    req.out.put(None)
    return False

  # ** the model
  def batchable(self, req:Request) -> bool:
    """req can be decoded in a slot: its prompt and room for a reply fit (a reply that fills the slot goes on alone)"""
    if not self.slots: return False
    room = min(1024, req.max_tokens - len(req.generated)) if req.max_tokens is not None else 1024
    return len(req.ids) + len(req.generated) + room < self.model.batch_ctx()

  def prefill(self, req:Request) -> int:
    """the prompt (and what req generated, a continuation) through the model: the first sampled token"""
    ids = req.ids + req.generated
    req.cache_start = self.model.get_start_pos(ids)
    g = self.model.generate(list(ids), temperature=req.temperature)
    try: return next(g)
    finally: _close(g)

  def run_alone(self, req:Request) -> bool:
    """req alone on the model: speculative decoding until its end, or until another request comes (it goes on in slot 0: returns True)"""
    model = self.model
    ids = req.ids + req.generated
    req.cache_start = model.get_start_pos(ids)
    spec_st = list(ss) if isinstance(ss:=getattr(model, "spec_stats", None), list) else []
    g = model.generate(list(ids), temperature=req.temperature)
    try:
      for tok in g:
        if not self.emit(req, tok): return False
        if not self.waiting.empty() and self.batchable(req):
          # another request: this one goes on batched (the model's states after its last token's position are the slot's)
          _close(g)
          req.pos = len(model._cached_tokens)
          model.slot_store(0, req.pos)
          return True
      return self.finish(req)
    finally:
      if spec_st and isinstance(ss:=getattr(model, "spec_stats", None), list): req.steps += ss[0] - spec_st[0]
      _close(g)

  def schedule(self):
    model, slots = self.model, self.slots
    active: dict[int, Request] = {}  # the slot of every request decoded batched
    pending: list[Request] = []  # requests waiting for a slot or for the model alone
    while True:
      try:
        # (a wait of a while at a time: a Ctrl-C stops the server, a lock's wait without a timeout doesn't see it)
        while not active and not pending:
          try: pending.append(self.waiting.get(timeout=0.2))
          except queue.Empty: pass
        while not self.waiting.empty(): pending.append(self.waiting.get_nowait())
        pending = [r for r in pending if not (r.cancelled and not self.finish(r))]
        if not active and pending:
          # one request (or one that doesn't fit a slot): alone
          if len(pending) == 1 or not self.batchable(pending[0]):
            req = pending.pop(0)
            try:
              if self.run_alone(req): active[0] = req
            except Exception as e:
              traceback.print_exc()
              self.finish(req, e)
            continue
        # new requests into free slots: prefilled, their states copied into the slot
        for req in [r for r in pending if self.batchable(r)]:
          if len(active) >= slots: break
          slot = next(i for i in range(slots) if i not in active)
          first = self.prefill(req)
          pending.remove(req)
          req.pos = len(req.ids) + len(req.generated)
          model.slot_store(slot, req.pos)
          if self.emit(req, first): active[slot] = req
        if not active: continue
        # a step of every slot up to the last one used (the free ones in between are rows of nothing)
        rows = [(active[i].generated[-1], active[i].pos, active[i].temperature) if i in active else (0, 0, 0.0) for i in range(max(active)+1)]
        out = model.batch_step(rows)
        for i, req in list(active.items()):
          req.pos += 1
          req.steps += 1
          if not self.emit(req, out[i]): del active[i]
          elif req.pos + 1 >= model.batch_ctx():
            # its slot is full: it goes on alone (prefilled again), when the model is free
            del active[i]
            pending.insert(0, req)
        # the last one batched, nothing waiting: it goes on alone (prefilled again), speculative decoding is faster than a step of one
        if len(active) == 1 and not pending and self.waiting.empty() and getenv("BATCH_LAST_ALONE", 1): pending.append(active.pop(next(iter(active))))
      except Exception as e:
        traceback.print_exc()
        for req in [*active.values(), *pending]: self.finish(req, e)
        active, pending = {}, []
