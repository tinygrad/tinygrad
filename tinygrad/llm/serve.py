from __future__ import annotations
import collections, json, pathlib, queue, re, select, socket, socketserver, sys, threading, time, traceback, typing, uuid
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log
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
  # Kimi-K3 format: tool="name" index="1"<|sep|><|open|>argument key="key" type="string"<|sep|>value<|close|>argument<|sep|>...
  if (m := re.fullmatch(r'tool="([^"]+)"(?: index="\d+")?<\|sep\|>(.*)', s, re.DOTALL)):
    pattern = r'<\|open\|>argument key="([^"]+)" type="([^"]+)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>'
    if re.sub(pattern, '', m[2], flags=re.DOTALL).strip(): return None
    def unescape(v:str) -> str: return v.replace('&quot;', '"').replace('&amp;', '&')
    try: return unescape(m[1]), {unescape(a[1]): a[3] if a[2] == "string" else json.loads(a[3]) for a in re.finditer(pattern, m[2], re.DOTALL)}
    except json.JSONDecodeError: return None
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  # GLM format: name<arg_key>key</arg_key><arg_value>value</arg_value>...
  ak, akc, av, avc = (re.escape(tags[t]) for t in ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"))
  patterns = (
    (r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", r"<parameter=([^>]+)>(.*?)</parameter>"),
    # a value ends at the </arg_value> before the next <arg_key> or the end (as text tags, a value can hold the markup: code that parses it)
    (rf"([\w.-]+)\s*((?:{ak}.+?{akc}\s*{av}.*?{avc}\s*)*)$", rf"{ak}(.+?){akc}\s*{av}(.*?){avc}(?=\s*(?:{ak}|$))"))
  for call_pattern, arg_pattern in patterns:
    if (fm := re.match(call_pattern, s, re.DOTALL)):
      args = {}
      for pm in re.finditer(arg_pattern, fm.group(2), re.DOTALL):
        value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
        try: args[pm.group(1)] = json.loads(value)
        except json.JSONDecodeError: args[pm.group(1)] = value
      return fm.group(1), args
  return None

# (the model's images, video and audio aren't served: a block of them is this note to the model, not an empty placeholder it takes for one it
# saw, so it looks at the file with tools)
MEDIA_NOTE = "[{kind} not shown: this server doesn't give images, video or audio to the model. inspect the file with tools (e.g. code) instead]"
MEDIA_KINDS = {"image": "image", "image_url": "image", "video": "video", "video_url": "video", "audio": "audio", "audio_url": "audio",
               "input_audio": "audio", "document": "document"}
def media_note(part:dict) -> str: return MEDIA_NOTE.format(kind=MEDIA_KINDS.get(part.get("type", ""), part.get("type", "content")))

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # string-only templates need text parts joined and explicit null content emptied
    if m.get("content", "") is None: m["content"] = ""
    elif isinstance(c := m.get("content"), list):
      c = m["content"] = [p if not isinstance(p, dict) or p.get("type") not in MEDIA_KINDS else {"type": "text", "text": media_note(p)} for p in c]
      if all(isinstance(p, dict) and p.get("type") == "text" for p in c): m["content"] = "".join(p["text"] for p in c)

# ** the Anthropic Messages API (/v1/messages): requests as the chat completions' messages and tools, replies as its content blocks and events

def anthropic_messages(body:dict) -> tuple[list[dict], list[dict]|None]:
  """an Anthropic request's system, messages and tools as chat completions' (thinking blocks: reasoning_content, tool results: tool messages)"""
  def text_of(c) -> str:
    if isinstance(c, str): return c
    return "".join(b.get("text", "") if b.get("type") == "text" else media_note(b) if b.get("type") in MEDIA_KINDS else "" for b in c or [])
  msgs:list[dict] = []
  # (Claude Code's system starts with a block of its billing metadata, "x-anthropic-billing-header: cc_version=...;": not the model's prompt)
  if isinstance(system:=body.get("system"), list): system = [b for b in system if not b.get("text", "").startswith("x-anthropic-billing-header:")]
  if system: msgs.append({"role": "system", "content": text_of(system)})
  for m in body.get("messages", []):
    content = m.get("content")
    if isinstance(content, str) or m["role"] == "system":
      # (a system message in the conversation (Claude Code's environment updates): its blocks or string the same message)
      msgs.append({"role": m["role"], "content": text_of(content)})
    elif m["role"] == "assistant":
      msg:dict[str, typing.Any] = {"role": "assistant", "content": "".join(b.get("text", "") for b in content if b.get("type") == "text")}
      if (reasoning:="".join(b.get("thinking", "") for b in content if b.get("type") == "thinking")): msg["reasoning_content"] = reasoning
      if (calls:=[{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": b.get("input") or {}}}
                  for b in content if b.get("type") == "tool_use"]): msg["tool_calls"] = calls
      msgs.append(msg)
    else:
      # (a user turn: its tool results are tool messages, in order, its text a user message after them)
      text:list[str] = []
      for b in content:
        if b.get("type") == "tool_result":
          msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": text_of(b.get("content", ""))})
        else: text.append(text_of([b]))
      if text: msgs.append({"role": "user", "content": "".join(text)})
  tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
           for t in body.get("tools") or [] if "input_schema" in t]
  return msgs, tools or None

STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens"}
# (a reply not streamed and not done in HOLD_JSON seconds: its headers then, and a space every HOLD_JSON seconds until it's done (whitespace
# before JSON is JSON): a proxy that drops a connection quiet for a while doesn't drop it, a client doesn't ask again. a stream: a ping after
# KEEPALIVE seconds without a chunk)
HOLD_JSON, KEEPALIVE = 5.0, 15.0
PING = -1  # (Handler.next_token: no token for a while)

def starts(ids:list[int], t:list[int], n:int|None=None) -> bool:
  """ids starts with the first n (all) tokens of t: a few of them compared first (another conversation differs there), then all (the lists of
  long conversations: a comparison of them all is a copy of every one, with many kept that's the time of a request's prefill)"""
  n = len(t) if n is None else n
  if n > len(ids) or n > len(t): return False
  return all(ids[p] == t[p] for p in (n // 2, (3 * n) // 4, n - 1) if 0 <= p) and ids[:n] == t[:n]

def same_start(a:list[int], b:list[int], n:int=2048) -> bool:
  """a and b share their first n tokens, at most half of either (the system prompt and the task of an agent's conversation: another one's task
  differs)"""
  n = min(n, len(a) // 2, len(b) // 2)
  return a[:n] == b[:n]

def common_prefix(a:list[int], b:list[int]) -> int:
  """the tokens a and b start with (the lists compared in slices: a loop over a long conversation's tokens is slow)"""
  lo, hi = 0, min(len(a), len(b))
  while lo < hi:
    mid = (lo + hi + 1) // 2
    if a[lo:mid] == b[lo:mid]: lo = mid
    else: hi = mid - 1
  return lo

ODD_TOKENS = ("[gMASK]", "[sMASK]", "[MASK]", "<sop>", "<eop>")

def anthropic_events(chunks:typing.Iterator[dict], model:str, input_tokens:int) -> typing.Iterator[dict]:
  """the chat completion chunks of a reply as the Anthropic stream's events (thinking, text and tool_use content blocks)"""
  msg_id = f"msg_{uuid.uuid4().hex[:24]}"
  yield {"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [],
         "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}
  index, finish, output_tokens = -1, "stop", 0
  kind:str|None = None  # the open content block's
  def stop() -> typing.Iterator[dict]:
    if kind == "thinking": yield {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": ""}}
    if kind is not None: yield {"type": "content_block_stop", "index": index}
  for c in chunks:
    if c.get("ping"):
      yield {"type": "ping"}
      continue
    if "error" in c:
      yield {"type": "error", "error": {"type": "api_error", "message": c["error"]["message"]}}
      return
    if not c["choices"]:
      output_tokens = c.get("usage", {}).get("completion_tokens", output_tokens)
      continue
    delta, finish = c["choices"][0].get("delta", {}), c["choices"][0].get("finish_reason") or finish
    for field, block, dtype in (("reasoning_content", "thinking", "thinking_delta"), ("content", "text", "text_delta")):
      if not delta.get(field): continue
      if kind != block:
        yield from stop()
        index, kind = index + 1, block
        yield {"type": "content_block_start", "index": index, "content_block": {"type": block, block: ""}}
      yield {"type": "content_block_delta", "index": index, "delta": {"type": dtype, block: delta[field]}}
    for tc in delta.get("tool_calls", []):
      yield from stop()
      index, kind = index + 1, None
      yield {"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                                                                "input": {}}}
      yield {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": tc["function"]["arguments"]}}
      yield {"type": "content_block_stop", "index": index}
  yield from stop()
  yield {"type": "message_delta", "delta": {"stop_reason": STOP_REASONS.get(finish, "end_turn"), "stop_sequence": None},
         "usage": {"output_tokens": output_tokens}}
  yield {"type": "message_stop"}

def anthropic_message(events:typing.Iterator[dict]) -> dict:
  """the Anthropic stream's events as the reply of a request that doesn't stream"""
  msg:dict[str, typing.Any] = {}
  for e in events:
    if e["type"] == "error": return e
    if e["type"] == "ping": continue
    if e["type"] == "message_start": msg = e["message"]
    elif e["type"] == "content_block_start": msg["content"].append(dict(e["content_block"]))
    elif e["type"] == "content_block_delta":
      block, d = msg["content"][e["index"]], e["delta"]
      if d["type"] == "input_json_delta": block["input"] = json.loads(d["partial_json"] or "{}")
      elif d["type"] == "signature_delta": block["signature"] = d["signature"]
      else: block[block["type"]] += d[block["type"]]
    elif e["type"] == "message_delta":
      msg.update(e["delta"])
      msg["usage"]["output_tokens"] = e["usage"]["output_tokens"]
  return msg

# Kimi-K3's markup (its tags of special tokens and text): start think, end think, end content, a tool call
KIMI_FORMAT = ("<|open|>think<|sep|>", "<|close|>think<|sep|><|open|>response<|sep|>", "<|close|>response<|sep|>",
               r"<\|open\|>call \s*(.*?)\s*(?:<\|close\|>call<\|sep\|>|$)")
THINK_STARTS = ("<think>", KIMI_FORMAT[0])

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse. tags: the string of every tag.
  # special_tokens: the model's (Kimi-K3's markup: KIMI_FORMAT)
  def __init__(self, special_tokens:dict[str, int]|None=None, reasoning:bool=False, tags:dict[str, str]=TEXT_TAGS):
    self.buf = ""
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
    if any(KIMI_FORMAT[2].startswith(t) for t in special_tokens or {}): self.think, self.end_think, self.end_content, self.call = KIMI_FORMAT
    else:
      # a call ends at the </tool_call> before the next <tool_call> or the end (as text tags, its arguments can hold the markup)
      call, call_end = re.escape(tags["<tool_call>"]), re.escape(tags["</tool_call>"])
      self.think, self.end_think, self.end_content = tags["<think>"], tags["</think>"], tags["<tool_call>"]
      self.call = rf"{call}\s*(.*?)\s*(?:{call_end}(?=\s*(?:{call}|$))|$)"
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
    think, think_end, call = self.think, self.end_think, self.end_content
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
    elif self.path.split("?")[0] == "/status": self.send_data(json.dumps(self.server.status(gpus="gpus" in self.path)).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, log:str="", keepalive:float|None=None, top_p:float|None=None):
    tok = self.server.tok
    prompt_tokens = len(ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    dec = tok.stream_decoder()
    tags, sentinels = markup_tags(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {})
    router = StreamRouter(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else None, reasoning, tags)
    # the model runs on the server's thread (LLMServer.serve_forever): the request waits there, its tokens come back here
    # (no top_p: the model's own, general.sampling.top_p)
    req = Request(ids, log, max_tokens, temperature, self.server.top_p if top_p is None else top_p)
    self.server.waiting.put(req)
    try:
      yield chunk({"role":"assistant", "content":""})
      # (special tokens no reply has: the sign of a model gone wrong, logged)
      specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
      odd = {i for t, i in specials.items() if t in ODD_TOKENS}
      # (keepalive: a ping after that long without a chunk, also while tokens come that make none: a tool call is sent when it ends)
      last = time.perf_counter()
      while (next_id := Handler.next_token(self, req, keepalive)) is not None:
        if next_id == PING:
          yield {"ping": True}
          last = time.perf_counter()
          continue
        out.append(next_id)
        if next_id in odd:
          stderr_log(log + colored(f"odd token {next_id} after {len(out)} tokens of the reply\n", "red"))
        for field, delta in router.route(sentinels[next_id] if next_id in sentinels else dec(next_id)):
          yield chunk({field:delta})
          last = time.perf_counter()
        if keepalive is not None and time.perf_counter() - last > keepalive:
          yield {"ping": True}
          last = time.perf_counter()
      if req.error is not None:
        yield {"error":{"message":f"{type(req.error).__name__}: {req.error}", "type":"server_error", "code":"generation_failed"}}
        return
      finish_reason = req.finish_reason
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      for m in re.finditer(router.call, router.buf, re.DOTALL):
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
        # (cached_tokens: the prompt's tokens that were in the caches, not prefilled)
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out),
                                        "prompt_tokens_details": {"cached_tokens": req.cache_start}}, **tmpl}
    finally: req.cancelled = True  # (a client gone: the model stops at its next token)

  def client_gone(self) -> bool:
    """the client closed the connection (a non streaming request writes only at its end: the model would decode for nobody)"""
    try:
      readable, _, _ = select.select([self.connection], [], [], 0)
      return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
    except (OSError, ValueError, TypeError): return False
  def next_token(self, req:Request, keepalive:float|None=None) -> int|None:
    # (the client is checked a second at most apart, also while tokens come: one gone mid reply stops the model, not its end)
    # (keepalive: PING after that long without a token, a request waiting for the model: a stream idle that long the client gives up on)
    st = time.perf_counter()
    while True:
      if keepalive is not None and time.perf_counter() - st > keepalive: return PING
      if time.perf_counter() - req.checked > 1.0:
        req.checked = time.perf_counter()
        if Handler.client_gone(self):
          req.cancelled = True
          return None
      if req.rx: return req.rx.pop(0)
      try: item = req.out.get(timeout=1.0)
      except queue.Empty: continue
      # (a step's tokens come at once, see LLMServer.flush)
      if not isinstance(item, list): return item
      req.rx = item[1:]
      return item[0]

  def bad_request(self, msg:str):
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  {colored('bad request', 'red')}: {msg[:200]}\n")
    return self.send_data(json.dumps({"error": {"message": msg, "type": "invalid_request_error", "code": "invalid_request"}}).encode(),
                          status_code=400)

  def do_POST(self):
    request_st = time.perf_counter()
    log = f"{self.path}  {colored('--', 'BLACK')}  "
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    # (a body that isn't a JSON object, messages the template can't render: a 400, not the handler's exception (the connection dropped))
    try:
      body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
      if not isinstance(body, dict): raise ValueError("the body is not a JSON object")
    except Exception as e: return self.bad_request(f"invalid request body: {e}")
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if (path:=self.path.split("?")[0]) in ("/v1/messages", "/v1/messages/count_tokens"): return self.anthropic(path, body, log, request_st)
    if self.path == "/v1/chat/completions":
      # render and tokenize
      try:
        normalize_messages(body["messages"])
        rendered = self.render(body["messages"], body.get("tools"), body)
      except Exception as e: return self.bad_request(f"{type(e).__name__}: {e}")
      ids: list[int] = self.server.reuse_tokens(rendered, self.server.tok.encode(rendered))
      log += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
      if len(ids) >= self.server.model.max_context:
        stderr_log(log + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      hold = 0.0 if body.get("stream") else HOLD_JSON
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=self.server.temperature if (t:=body.get("temperature")) is None else float(t),
                              reasoning=rendered.rstrip().endswith(THINK_STARTS), log=log,
                              top_p=float(tp) if (tp:=body.get("top_p")) is not None else None, keepalive=hold or None)
      if body.get("stream"): self.stream_json(chunks)
      else:
        held = False
        def reply(data:bytes, status_code:int=200):
          if held: self.wfile.write(data)
          else: self.send_data(data, status_code=status_code)
        out, reasoning, tool_calls, finish_reason, last = [], [], [], "stop", time.perf_counter()
        for c in chunks:
          if hold and time.perf_counter() - last > hold:
            if not held:
              self.send_response(200)
              self.send_header("Content-Type", "application/json")
              self.send_header("Connection", "close")
              self.end_headers()
              self.close_connection, held = True, True
            self.wfile.write(b" ")
            last = time.perf_counter()
          if c.get("ping"): continue
          if "error" in c: return reply(json.dumps(c).encode(), status_code=500)
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
        reply(json.dumps({**c, "object":"chat.completion", "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    else:
      self.send_data(json.dumps({"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error", "code": "not_found"}}).encode(),
                     status_code=404)

  def render(self, messages:list[dict], tools:list[dict]|None, body:dict) -> str:
    # (a reasoning effort asked for is the template's, else its default)
    effort = body.get("reasoning_effort") or (body.get("output_config") or {}).get("effort")
    # (the template's own arguments a request asks for, e.g. enable_thinking: chat_template_kwargs as vLLM's)
    extra = {k: v for k, v in (body.get("chat_template_kwargs") or {}).items() if k not in ("messages", "tools", "add_generation_prompt")}
    text = self.server.template.render(**{"messages": messages, "tools": tools, "add_generation_prompt": True, "preserve_thinking": True,
                                          **({"reasoning_effort": effort} if effort else {}), **extra})
    # (JSON can have lone surrogates ("\ud800"), not text the tokenizer can encode: a replacement character)
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")

  def anthropic(self, path:str, body:dict, log:str, request_st:float):
    """the Anthropic Messages API: the request as chat completions' messages, the reply's chunks as its content blocks (and stream events)"""
    try:
      msgs, tools = anthropic_messages(body)
      normalize_messages(msgs)
      rendered = self.render(msgs, tools, body)
    except Exception as err:
      return self.send_data(json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": f"{type(err).__name__}: {err}"}})
                            .encode(), status_code=400)
    ids: list[int] = self.server.reuse_tokens(rendered, self.server.tok.encode(rendered))
    if path.endswith("count_tokens"): return self.send_data(json.dumps({"input_tokens": len(ids)}).encode())
    log += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
    if len(ids) >= (n:=self.server.model.max_context):
      # (the message Anthropic's clients know: Claude Code compacts its conversation)
      stderr_log(log + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{n:5d}\n")
      return self.send_data(json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                        "message": f"prompt is too long: {len(ids)} tokens > {n-1} maximum"}}).encode(), status_code=400)
    # (the API's default temperature is 1)
    chunks = self.run_model(ids, body.get("model", self.server.model_name), True, max_tokens=body.get("max_tokens"),
                            temperature=float(t if (t:=body.get("temperature")) is not None else 1.0),
                            reasoning=rendered.rstrip().endswith(THINK_STARTS), log=log,
                            keepalive=KEEPALIVE if body.get("stream") else None,
                            top_p=float(tp) if (tp:=body.get("top_p")) is not None else None)
    events = anthropic_events(chunks, body.get("model", self.server.model_name), len(ids))
    try:
      if body.get("stream"):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for e in events:
          self.wfile.write(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode("utf-8"))
          self.wfile.flush()
      else:
        reply = anthropic_message(events)
        self.send_data(json.dumps(reply).encode(), status_code=500 if reply.get("type") == "error" else 200)
    except (BrokenPipeError, ConnectionResetError): pass
    finally: chunks.close()

def _close(g:typing.Iterator) -> None:
  if (close:=getattr(g, "close", None)) is not None: close()  # (a generator: the model stops where it is)

_rids = iter(range(1, 1 << 62))

class Request:
  """a request to the model: its prompt, its sampled tokens come back in out (None: the end), it's cancelled when the client goes away"""
  def __init__(self, ids:list[int], log:str, max_tokens:int|None, temperature:float, top_p:float=1.0):
    self.ids, self.log, self.max_tokens, self.temperature, self.top_p, self.rid = ids, log, max_tokens, temperature, top_p, next(_rids)
    self.cancelled, self.finish_reason, self.checked, self.zeros = False, "stop", 0.0, 0
    self.error: Exception|None = None
    self.out: queue.Queue[int|list[int]|None] = queue.Queue()
    self.held: list[int] = []  # (the tokens of a step not put in out yet: put at once, one wakeup of its thread a step)
    self.rx: list[int] = []  # (the handler's: the tokens of a step it got and hasn't returned yet)
    self.generated: list[int] = []
    self.st, self.pt, self.cache_start, self.steps = time.perf_counter(), 0.0, 0, 0
    self.done = False
    # (the prefill: the prompt, the position of the next slice, whether the first restores a checkpoint. the time it waited for its sequence
    # (queued), the time of its slices)
    self.fill: tuple[list[int], int, bool] = ([], 0, False)
    self.queued, self.prefill_time, self.slices, self.max_gap, self.last_tok = 0.0, 0.0, 0, 0.0, 0.0


class BatchStats:
  """batched decoding's status line every few seconds while it runs: the requests decoding and waiting now, the tokens generated a second and
  the steps (the batch's) and prefills (of the requests that joined it) since the last line"""
  def __init__(self, every:float=5.0):
    self.every, self.st = every, time.perf_counter()
    self.reset()
  def reset(self):
    self.steps, self.tokens, self.step_time, self.prefills, self.prefill_time = 0, 0, 0.0, 0, 0.0
    self.times:dict[str, float|int] = {}  # (more of the time: where it went, and counts)
  def add(self, name:str, t:float|int): self.times[name] = self.times.get(name, 0) + t
  def step(self, running:int, waiting:int):
    if (dt:=time.perf_counter() - self.st) > self.every and self.steps:
      sep = colored('--', 'BLACK')
      stderr_log(f"batch  {sep}  {running:3d} requests decoding, {waiting:3d} waiting  {sep}  {self.tokens/dt:5.0f} tok/s generated  {sep}  "
                 f"{self.steps:4d} steps of {self.step_time/self.steps*1e3:5.1f} ms  {sep}  {self.prefills:3d} prefills ({self.prefill_time:4.1f}s)  "
                 f"{sep}  in the last {dt:.0f}s" + "".join(f"  {sep}  {k} {v:.1f}s" if isinstance(v, float) else f"  {sep}  {k} {v}"
                                                           for k, v in self.times.items()) + "\n")
      self.st = time.perf_counter()
      self.reset()
    self.steps += 1

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  """
  requests come in on threads of their own, the model runs on the server's thread: a model whose sequences share its caches
  (Transformer.seqs_supported) decodes all of them at once (schedule_seqs), another one request at a time (schedule_alone)
  """
  def handle_error(self, request, client_address):
    # (a client gone before its reply: not an error of the server's, no traceback)
    if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)): return
    super().handle_error(request, client_address)
  daemon_threads = True
  request_queue_size = 1024  # the connections waiting to be accepted (many clients at once: the default of 5 refuses them)
  verbose_prefill = False
  generated_total, prefilled_total = 0, 0  # (the tokens generated and prefilled since the start, see status)
  # (the requests that ended (when, ttft, longest wait for a token, total, prompt, cached, generated): the latencies of status)
  recent: collections.deque = collections.deque(maxlen=1000)
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, top_p:float=1.0,
               temperature:float=0.0):
    # top_p: a request's without one (the model's recommended, general.sampling.top_p of its gguf); temperature: a chat completion's
    self.model, self.model_name, self.tok, self.template, self.top_p, self.temperature = model, model_name, tok, template, top_p, temperature
    self.waiting: queue.Queue[Request] = queue.Queue()
    self.inflight = 0  # the requests taken by the model (decoded, or waiting for it): the status line's
    self.stats = BatchStats()
    self.engine: typing.Any = None  # (schedule_seqs': the sequences)
    self.recent = collections.deque(maxlen=1000)
    specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
    self.odd_ids = {i for t, i in specials.items() if t in ODD_TOKENS}
    super().__init__(server_address, Handler)

  def serve_forever(self, poll_interval:float=0.5):
    # (the model's thread polls the devices holding the GIL: the request threads (many streams) take it in turns of 0.5 ms, the default 5 ms
    # delays its steps)
    sys.setswitchinterval(0.0005)
    # the model runs here, not on a thread of its own: BEAM's _try_compile installs a SIGALRM handler, which raises ValueError off the main thread
    threading.Thread(target=super().serve_forever, args=(poll_interval,), daemon=True, name="HTTPServer").start()
    if hasattr(type(self.model), "seqs_supported") and typing.cast(typing.Any, self.model).seqs_supported(): self.schedule_seqs()
    else: self.schedule_alone()

  # ** the requests' tokens and their end
  def flush(self, req:Request):
    """the tokens of req held (emit's hold) sent"""
    if req.held: req.out.put(req.held)
    req.held = []
  def emit(self, req:Request, tok:int, hold:bool=False) -> bool:
    """a sampled token of req: sent (hold: held until flush), or the end. returns whether req goes on"""
    if not req.generated and not req.pt:
      req.pt = time.perf_counter()
      stderr_log(req.log + f"in:{colored(f'{req.cache_start:5d}', 'green')} +{len(req.ids)-req.cache_start:5d}  {colored('--', 'BLACK')}  "
                 f"prefill:{(len(req.ids)-req.cache_start)/(req.pt-req.st):5.0f} tok/s\n" if self.verbose_prefill else "")
    if req.cancelled or self.tok.is_end(tok): return self.finish(req)
    # (the model's states gone wrong (NaN: its argmax is token 0, again and again), or a special token no reply has: the reply fails, the
    # caches are dropped, the client's retry is prefilled from the start)
    req.zeros = req.zeros + 1 if tok == 0 else 0
    if req.zeros >= 16 or tok in self.odd_ids:
      why = "token 0 again and again" if req.zeros else f"token {tok}"
      stderr_log(req.log + colored(f"the model went wrong after {len(req.generated)} tokens ({why}, {req.cache_start} of {len(req.ids)} cached): "
                                   "the reply fails, the caches are dropped\n", "red") + f"  ...{self.tok.decode(req.generated[-60:])!r}\n")
      self.drop_caches()
      return self.finish(req, RuntimeError("the model's states went wrong (dropped: retry)"))
    # (the longest wait for a token after the first: a reply's stall, e.g. behind a prefill)
    if req.generated: req.max_gap = max(req.max_gap, time.perf_counter() - req.last_tok)
    req.last_tok = time.perf_counter()
    req.generated.append(tok)
    if hold: req.held.append(tok)
    else: req.out.put(tok)
    self.stats.tokens += 1
    self.generated_total += 1
    if req.max_tokens is not None and len(req.generated) >= req.max_tokens:
      req.finish_reason = "length"
      return self.finish(req)
    return True
  def finish(self, req:Request, error:Exception|None=None) -> bool:
    if req.done: return False
    req.done, req.error = True, error
    et, n = time.perf_counter(), len(req.generated)
    total, sep = f"total:{et-req.st:6.2f}s", colored('--', 'BLACK')
    # (the phases: waiting for a sequence, the prefill's slices (the decoding ones' steps between them), the first token)
    new = len(req.ids) - req.cache_start
    pre = f"prefill:{req.prefill_time:5.2f}s" + (f" in {req.slices}" if req.slices > 1 else "") + \
          (f" {new/req.prefill_time:5.0f} tok/s" if req.prefill_time else "")
    spec = f"spec:{n/req.steps:4.2f} tok/step  {sep}  " if req.steps else ""
    stderr_log(req.log + f"reqs:{self.inflight:3d}  {sep}  in:{colored(f'{req.cache_start:5d}', 'green')} +{new:5d}  {sep}  "
               f"queue:{req.queued:5.2f}s  {sep}  {pre}  {sep}  ttft:{(req.pt or et)-req.st:5.2f}s  {sep}  "
               f"gen:{(n-1)/(et-req.pt) if n > 1 and req.pt else 0:4.0f} tok/s  {sep}  {spec}out:{n:5d}  {sep}  "
               f"{colored(total, 'red') if req.cancelled else total}\n")
    self.recent.append((time.time(), (req.pt or et) - req.st, req.max_gap, et - req.st, len(req.ids), req.cache_start, n))
    self.flush(req)
    req.out.put(None)
    return False

  def status(self, gpus:bool=False) -> dict:
    """the requests and the caches now (GET /status): the requests decoding, being prefilled (their tokens left) and waiting, the sequences in
    the caches and in the stash (their tokens)"""
    now, ret = time.perf_counter(), typing.cast(dict[str, typing.Any], {"waiting": self.waiting.qsize(), "time": time.time(),
                                                                    "generated_total": self.generated_total, "prefilled_total": self.prefilled_total})
    if (q:=getattr(self, "queues", None)) is not None:
      active, filling, pending = (list(x.values()) if isinstance(x, dict) else list(x) for x in q)
      def req(r:Request) -> dict: return {"prompt": len(r.ids), "cached": r.cache_start, "generated": len(r.generated), "age": round(now - r.st, 2)}
      ret |= {"decoding": [req(r) for r in active], "prefilling": [req(r) | {"left": len(r.fill[0]) - r.fill[1]} for r in filling],
              "waiting": [req(r) for r in pending] + [{}] * self.waiting.qsize()}
    # the latencies of the requests of the last 10 minutes: their ttft, their longest wait for a token after the first, their total
    if rs:=[r for r in list(self.recent) if r[0] > time.time() - 600]:
      def pct(xs:list[float]) -> dict:
        xs = sorted(xs)
        return {f"p{q}": round(xs[min(len(xs)-1, int(q / 100 * len(xs)))], 3) for q in (50, 90, 99)} | {"max": round(xs[-1], 3)}
      ret["last_10m"] = {"requests": len(rs), "ttft": pct([r[1] for r in rs]), "token_gap": pct([r[2] for r in rs]), "total": pct([r[3] for r in rs]),
                         "prompt_tokens": sum(r[4] for r in rs), "cached_tokens": sum(r[5] for r in rs), "generated_tokens": sum(r[6] for r in rs)}
    if gpus:
      # (?gpus: the power and clocks of the devices)
      try:
        from tinygrad.runtime.ops_amd import AMIdleClocks
        ret["gpus"] = AMIdleClocks.metrics()
      except Exception as e: ret["gpus"] = str(e)
    if (eng:=self.engine) is not None:
      seqs, stashed = list(eng.seqs), list(eng.stashed)
      ret |= {"resident": {"sequences": len(seqs), "tokens": sum(s.n for s in seqs), "rows": eng.rows, "rows_taken": sum(s.cap for s in seqs)},
              "stash": {"sequences": len(stashed), "tokens": sum(len(x.tokens) for x in stashed), "chunks_free": len(eng.free_chunks)},
              "evictions": eng.evictions}
    return ret

  # ** the kept conversations
  def kept(self) -> list[list[int]]:
    """the kept conversations' tokens (and their checkpoints'): the model's, the sequences', the stashed ones'"""
    ret = [list(c)] if isinstance(c:=getattr(self.model, "_cached_tokens", None), list) else []
    if self.engine is not None:
      ret += [t for s in list(self.engine.seqs) for t in (s.tokens, s.tokens[:s.ckpt])]
      ret += [t for x in list(self.engine.stashed) for t in (x.tokens, x.tokens[:x.ckpt])]
    return ret
  def reuse_tokens(self, rendered:str, ids:list[int]) -> list[int]:
    """
    the tokens of a prompt that continues a kept conversation as its text but not its tokens (the model's reply tokenized again differently):
    the kept ones and the rest's, so the cache holds it
    """
    cands = [t for t in self.kept() if 1000 <= len(t) < len(ids) + 1000]
    # (one the prompt's tokens continue: the cache has it. a longer one the prompt's text continues: the ones sharing the most tokens with it
    # (an agent's own conversation, not the others' with the same system prompt), however early they differ: a sampled token of an early
    # reply that isn't its text's tokenization differs from then on, every turn after)
    exact = max((len(t) for t in cands if starts(ids, t)), default=0)
    # (an agent's own ones: the others' differ in their task, at the start)
    cands = [t for t in cands if len(t) > exact and same_start(t, ids)]
    for t in sorted(cands, key=lambda t: common_prefix(t, ids), reverse=True)[:2]:
      if rendered.startswith(text:=self.tok.decode(t)): return t + self.tok.encode(rendered[len(text):])
    return ids
  def miss_log(self, ids:list[int], start:int):
    """a request prefilled from 2000 tokens less than it shares with a kept conversation at least: where they differ"""
    kept = [t for t in self.kept() if len(t) >= start + 2000 <= len(ids) and t[start+1999] == ids[start+1999]]
    n, t = max(((common_prefix(t, ids), t) for t in kept if same_start(t, ids)), key=lambda x: x[0], default=(0, []))
    if n - start < 2000: return
    dec = self.tok.decode
    stderr_log(colored(f"cache miss: prefilled from {start}, shares {n} of {len(t)} kept tokens: kept {dec(t[max(0,n-30):n+30])!r} "
                       f"vs {dec(ids[max(0,n-30):n+30])!r}\n", "yellow"))
  def drop_caches(self):
    """the kept conversations forgotten (their states may be wrong): every request is prefilled from the start"""
    if self.engine is not None: self.engine.forget()
    if hasattr(self.model, "_cached_tokens"): self.model._cached_tokens = []

  # ** one request at a time: its prompt prefilled (from the conversation the model has, if it continues it), then decoded to its end
  def schedule_alone(self):
    while True:
      try: req = self.waiting.get(timeout=0.2)
      except queue.Empty: continue
      self.inflight = 1 + self.waiting.qsize()
      if req.cancelled and not self.finish(req): continue
      req.cache_start = self.model.get_start_pos(req.ids)
      g = self.model.generate(list(req.ids), temperature=req.temperature)
      try:
        for tok in g:
          if not self.emit(req, tok): break
        else: self.finish(req)
      except Exception as e:
        traceback.print_exc()
        self.finish(req, e)
      finally: _close(g)

  # ** sequences sharing the model's caches (Transformer.seqs_supported): continuous batching
  # (a prompt is prefilled in slices of PREFILL_SLICE tokens (~1 s). the decoding ones' steps run after MAX_STALL s of slices, for their share
  # of the time (the requests decoding of the ones in flight, DECODE_SHARE at most and 0.1 at least): a long prompt doesn't stop the replies,
  # many prompts at once are prefilled mostly one after another (a step of many is about as fast as one of a few). the next slice is the
  # prompt's with the fewest tokens left to prefill (a short one, an agent's turn, isn't behind a long one), less AGING tokens a second it
  # waited (a long one isn't behind short ones that keep coming))
  PREFILL_SLICE, MAX_STALL, DECODE_SHARE, AGING = 13824, 1.0, 0.5, 1000
  # (the gpus' clocks (AM: their gfx clock's cap, see AMIdleClocks.set_caps): a few users decoding at the full clocks (a user's tokens fast),
  # more at less (the energy a token: ~0.85 J at 2000 MHz on 6x 7900 XTX, ~1.2 J at full), a prefill at a cap it isn't slower at. switched
  # CLOCK_HOLD s apart at most)
  CLOCKS, FAST_USERS, CLOCK_HOLD = {"fast": "", "many": "GFXCLK=2000", "prefill": "GFXCLK=2300"}, 4, 0.5
  def schedule_seqs(self):
    """
    every request has a sequence of its own (the resident one its prompt continues: an agent's turn is prefilled from its last one), prefilled
    when it comes, then decoded with all the others decoding (a step of speculative decoding of every one at once, see Transformer.seq_step)
    """
    from tinygrad.llm.engine import SeqEngine, Seq
    eng = self.engine = SeqEngine(self.model)
    active: dict[Seq, Request] = {}
    filling: dict[Seq, Request] = {}  # (the sequences of the prompts being prefilled: their request's fill is where)
    pending: list[Request] = []
    self.queues = (active, filling, pending)
    k = eng.k
    def prompt(req:Request) -> list[int]: return req.ids + req.generated  # (a paused one (see below) goes on from its tokens so far)
    stall = 0.0  # (the time of the slices since the last steps)
    try:
      from tinygrad.runtime.ops_amd import AMIdleClocks
      set_caps = AMIdleClocks.set_caps if AMIdleClocks.devs else None
    except ImportError: set_caps = None
    clock_mode, clock_at = "", 0.0
    while True:
      try:
        while not active and not pending and not filling:
          try: pending.append(self.waiting.get(timeout=0.2))
          except queue.Empty: pass
        while not self.waiting.empty(): pending.append(self.waiting.get_nowait())
        pending[:] = [r for r in pending if not (r.cancelled and not self.finish(r))]
        for s, req in list(filling.items()):
          if req.cancelled:
            del filling[s]
            self.finish(req)
            eng.stop(s, len(s.tokens))
        self.inflight = len(active) + len(filling) + len(pending)
        # the prompt to prefill a slice of: the one with the fewest tokens left (aged), a waiting one taking a sequence
        now, job = time.perf_counter(), None
        cands:list[tuple[float, int, Seq|Request]] = [(len(r.fill[0]) - r.fill[1] - self.AGING*(now - r.st), 0, s) for s, r in filling.items()]
        if len(active) + len(filling) < eng.max_active:
          cands += [(len(ids) - eng.cached(ids) - self.AGING*(now - r.st), 1, r) for r in pending if (ids:=prompt(r))]
        for _, _, c in sorted(cands, key=lambda x: x[:2]):
          if isinstance(c, Seq):
            job = c
            break
          t0 = time.perf_counter()
          got = eng.take(ids:=prompt(c), max(0, (c.max_tokens or 0) - len(c.generated)))
          self.stats.add("take", time.perf_counter() - t0)
          if got is None: continue
          pending.remove(c)
          job, start, restore = got
          if not c.generated: c.cache_start, c.queued = start, time.perf_counter() - c.st
          c.fill = (ids, start, restore)
          filling[job] = c
          self.miss_log(ids, start)
          break
        for name, t in eng.times.items(): self.stats.add(name, t)
        eng.times.clear()
        if set_caps is not None and (mode:="prefill" if job is not None else "many" if len(active) > self.FAST_USERS else "fast") != clock_mode \
           and time.perf_counter() - clock_at >= self.CLOCK_HOLD:
          set_caps(self.CLOCKS[mode])
          clock_mode, clock_at = mode, time.perf_counter()
        if job is not None:
          req = filling[job]
          ids, pos, restore = req.fill
          end = len(ids) if len(ids) - pos <= self.PREFILL_SLICE * 3 // 2 else pos + self.PREFILL_SLICE
          st = time.perf_counter()
          try:
            if end < len(ids): eng.prefill_slice(job, ids, pos, restore, end)
            else: first = eng.prefill(job, ids, pos, restore, (req.temperature, req.top_p))
          except Exception as e:
            traceback.print_exc()
            del filling[job]
            eng.seqs.remove(job)
            self.finish(req, e)
            continue
          slice_time = time.perf_counter() - st
          stall += slice_time
          req.prefill_time, req.slices = req.prefill_time + slice_time, req.slices + 1
          self.prefilled_total += (end if end < len(ids) else len(ids)) - pos
          self.stats.prefill_time += slice_time
          if end < len(ids): req.fill = (ids, end, False)
          else:
            del filling[job]
            self.stats.prefills += 1
            if self.emit(req, first): active[job] = req
            else: eng.stop(job, len(job.tokens))
        # (requests that can't take a sequence yet, none decoding: not a busy loop)
        if job is None and not active: time.sleep(0.005)
        # the steps of the decoding ones: one, or their share of the time of the slices
        if job is not None and stall < self.MAX_STALL: continue
        if job is not None:
          # (the slice queued last done: its time the prefill's, not the steps')
          st = time.perf_counter()
          eng.settle()
          stall += time.perf_counter() - st
        share = min(self.DECODE_SHARE, max(0.1, len(active) / max(1, len(active) + len(filling) + len(pending))))
        budget, stall = stall * share / (1 - share), 0.0
        st0 = time.perf_counter()
        while active:
          seqs = list(active)
          st = time.perf_counter()
          outs = eng.step(seqs)
          self.stats.step_time += time.perf_counter() - st
          for s, toks in zip(seqs, outs):
            req = active[s]
            req.steps += 1
            for i, t in enumerate(toks):
              if not self.emit(req, t, hold=True):
                # (its tokens up to this one: the ones after it aren't the conversation's)
                eng.stop(s, len(s.tokens) - (len(toks) - 1 - i))
                del active[s]
                break
            else:
              self.flush(req)
              if not eng.room(s, len(s.tokens) + 2*(k+1)):
                del active[s]
                # (the caches full of the decoding ones: it pauses (into the stash), its request goes on when there's room, first. one longer
                # than the caches ends)
                if eng.fits(len(s.tokens) + 2*(k+1)):
                  eng.pause(s)
                  pending.insert(0, req)
                  self.stats.add("paused", 1)
                else:
                  req.finish_reason = "length"
                  self.finish(req)
                  eng.stop(s, len(s.tokens))
          self.stats.step(len(active), len(pending) + len(filling))
          if time.perf_counter() - st0 >= budget: break
      except KeyboardInterrupt: raise
      except Exception as e:
        # (a step that failed: its requests fail, the sequences are dropped)
        traceback.print_exc()
        for s, req in [*active.items(), *filling.items()]:
          self.finish(req, e)
          if s in eng.seqs: eng.seqs.remove(s)
        active.clear()
        filling.clear()
