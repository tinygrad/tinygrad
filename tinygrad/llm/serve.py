from __future__ import annotations
import collections, json, math, os, pathlib, queue, random, re, select, socket, socketserver, sys, threading, time, traceback, typing, uuid
from dataclasses import dataclass, field as DataField
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log, getenv
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
from tinygrad.llm.telemetry import Telemetry, utc_ms
from tinygrad.llm.engine import SessionInFlight
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

class HostConv(typing.NamedTuple):
  """a conversation kept in host memory (HOST_STASH): its tokens, its chunks (Transformer.host_save), the tokens of its checkpoint (the prompt
  of its last reply: a request can continue that one, its reply came back different, or 0), its row of the hidden states (_spec_row)"""
  tokens: list[int]
  chunks: list[typing.Any]
  ckpt: int
  hrow: int
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

def sampling_params(body:dict, temperature:float) -> tuple[int|None, float, float|None]:
  """a request's (max_tokens, temperature, top_p) checked (a ValueError: the request's 400, not a value the model's thread fails on)"""
  def num(k:str, lo:float, hi:float) -> float|None:
    if (v:=body.get(k)) is None: return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
      raise ValueError(f"{k} must be a number from {lo} to {hi}, not {v!r}")
    return float(v)
  if (mt:=body.get("max_completion_tokens")) is None: mt = body.get("max_tokens")
  if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 1):
    raise ValueError(f"max_tokens must be an integer >= 1, not {mt!r}")
  return mt, t if (t:=num("temperature", 0, 100)) is not None else temperature, num("top_p", 0, 1)

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path.split("?")[0] == "/status": self.send_data(json.dumps(self.server.status(gpus="gpus" in self.path)).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, log:str="", keepalive:float|None=None, top_p:float|None=None, continuous_usage:bool=False):
    tok = self.server.tok
    prompt_tokens = len(ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    # (cached_tokens: the prompt's tokens that were in the caches, not prefilled)
    def usage() -> dict: return {"prompt_tokens": prompt_tokens, "completion_tokens": len(out), "total_tokens": prompt_tokens + len(out),
                                 "prompt_tokens_details": {"cached_tokens": req.cache_start}}
    # (continuous_usage: the usage so far in every chunk, stream_options.continuous_usage_stats as vLLM's)
    def chunk(d:dict, finish_reason:str|None=None) -> dict:
      return {"choices": [{"index":0, "delta":d, "finish_reason":finish_reason}], **tmpl, **({"usage": usage()} if continuous_usage else {})}
    out: list[int] = []
    dec = tok.stream_decoder()
    tags, sentinels = markup_tags(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {})
    router = StreamRouter(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else None, reasoning, tags)
    # the model runs on the server's thread (LLMServer.serve_forever): the request waits there, its tokens come back here
    # (no top_p: the model's own, general.sampling.top_p)
    def clean(value: str, limit: int) -> str:
      return re.sub(r"\s+", "_", value)[:limit] if isinstance(value, str) else ""
    sid = clean(self.headers.get("X-Session-ID", ""), 4096)
    bench = clean(self.headers.get("X-Bench-Tag", ""), 96)
    req = Request(ids, log, max_tokens, temperature, self.server.top_p if top_p is None else top_p, session_id=sid, bench_tag=bench)
    # (SUPERSEDE: the same prompt again from the same session (X-Session-ID) while its request runs is a client's retry of it (a client that
    # timed out behind a proxy that doesn't close our connection): the older one stops, it isn't decoded for nobody)
    key = (sid, len(ids), hash(tuple(ids))) if isinstance(sid, str) and sid and getenv("SUPERSEDE", 1) else None
    if key is not None:
      with self.server.keys_lock:
        if (old:=self.server.inflight_keys.get(key)) is not None and not old.done:
          stderr_log(log + colored(f"the same prompt again from session {str(sid)[:40]}: the request before it stops\n", "yellow"))
          old.superseded = old.cancelled = True
        self.server.inflight_keys[key] = req
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
      if req.error is not None or req.superseded:
        msg = f"{type(req.error).__name__}: {req.error}" if req.error is not None else "superseded by the same request again"
        yield {"error":{"message":msg, "type":"server_error", "code":"generation_failed"}}
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
      yield chunk({}, finish_reason)
      if include_usage: yield {"choices": [], "usage": usage(), **tmpl}
    finally:
      req.cancelled = True  # (a client gone: the model stops at its next token)
      if key is not None:
        with self.server.keys_lock:
          if self.server.inflight_keys.get(key) is req: del self.server.inflight_keys[key]

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

  def session_end(self, session_id: str):
    if not isinstance(session_id, str) or not session_id.strip(): return self.bad_request("session_id must be a nonempty string")
    session_id = re.sub(r"\s+", "_", session_id)[:4096]
    expected = os.getenv("SESSION_END_TOKEN", "")
    if not expected:
      return self.send_data(json.dumps({"error": "backend_session_end_disabled", "session_id": session_id}).encode(), status_code=403)
    if self.headers.get("Authorization") != f"Bearer {expected}":
      return self.send_data(json.dumps({"error": "unauthorized", "session_id": session_id}).encode(), status_code=401)
    answer = SessionEndResponse()
    self.server.end_requests.put((session_id, answer))
    wake = Request([], "", 0, self.server.temperature, session_id="session-end-wake")
    wake.cancelled = True  # wake an idle scheduler from its pending queue wait; this is never a generation request
    self.server.waiting.put(wake)
    if not answer.event.wait(30):
      return self.send_data(json.dumps({"error": "session_end_timeout", "session_id": session_id}).encode(), status_code=503)
    result, status = answer.result, answer.status
    stderr_log(f"session_end sid={session_id} status={status} result={json.dumps(result, sort_keys=True)}\n")
    return self.send_data(json.dumps(result).encode(), status_code=status)

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
    if path == "/v1/sessions/end": return self.session_end(body.get("session_id", ""))
    if path.startswith("/v1/sessions/") and path.endswith("/end"):
      from urllib.parse import unquote
      return self.session_end(unquote(path[len("/v1/sessions/"):-len("/end")]))
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
      try: max_tokens, temperature, top_p = sampling_params(body, self.server.temperature)
      except ValueError as e: return self.bad_request(str(e))
      hold = 0.0 if body.get("stream") else HOLD_JSON
      include_usage = not body.get("stream") or (body.get("stream_options") or {}).get("include_usage", False)
      chunks = self.run_model(ids, str(body.get("model") or self.server.model_name), include_usage,
                              continuous_usage=include_usage and bool((body.get("stream_options") or {}).get("continuous_usage_stats", False)),
                              max_tokens=max_tokens, temperature=temperature, reasoning=rendered.rstrip().endswith(THINK_STARTS), log=log,
                              top_p=top_p, keepalive=hold or None)
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
      # (the API's default temperature is 1)
      max_tokens, temperature, top_p = sampling_params(body, 1.0)
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
    chunks = self.run_model(ids, str(body.get("model") or self.server.model_name), True, max_tokens=max_tokens, temperature=temperature,
                            reasoning=rendered.rstrip().endswith(THINK_STARTS), log=log, keepalive=KEEPALIVE if body.get("stream") else None,
                            top_p=top_p)
    events = anthropic_events(chunks, str(body.get("model") or self.server.model_name), len(ids))
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
def trace(msg:str):
  """(SERVE_TRACE: a file) the model's operations of the scheduler, in order: to tell what came before a reply went wrong"""
  if (path:=getenv("SERVE_TRACE", "")):
    with open(path, "a") as f: f.write(f"{time.time():.3f} {msg}\n")

class Request:
  """a request to the model: its prompt, its sampled tokens come back in out (None: the end), it's cancelled when the client goes away"""
  def __init__(self, ids:list[int], log:str, max_tokens:int|None, temperature:float, top_p:float=1.0,
               session_id:str="", bench_tag:str=""):
    self.ids, self.max_tokens, self.temperature, self.top_p, self.rid = ids, max_tokens, temperature, top_p, next(_rids)
    self.session_id, self.bench_tag = session_id, bench_tag
    self.log = f"{utc_ms()} {log}rid:{self.rid} sid:{session_id or '-'} bench:{bench_tag or '-'} --  "
    self.cancelled, self.finish_reason, self.checked, self.zeros, self.superseded = False, "stop", 0.0, 0, False
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
    self.pos, self.done = 0, False  # batched decoding: the position of its last token (the next step's input)
    # batched speculative decoding: the last step's sampled tokens, the position the MTP layer catches up from, the drafts accepted, the parity
    self.y:list[int] = []
    self.mtp, self.j, self.par = 0, 0, 1
    self.no_mix = False  # (MIX) prefilled before its steps, not in them (its cache was less than it seemed)
    # (sequences) the tokens of its prompt in the caches or the stash, and the engine's version then; the version its take last failed at, when
    # (was_hit: it continued a kept conversation when the scheduler first looked: an agent's turn, even if that's lost since)
    self.hit, self.hit_v, self.tried, self.tried_at, self.was_hit = 0, -1, -1, 0.0, False
    self.cold = False  # (sequences) its prefill is of more than COLD_TOKENS: a cold one, see schedule_seqs
    # (decoded in a slot: a copy of the model's checkpoint at the end of its prompt, see LLMServer.ckpt_keep)
    self.ckpt_buf: typing.Any = None
    self.ckpt_tokens: list[int] = []
  def spec_start(self, first:int, pos:int, k:int):
    """the slot's state after a prefill (or the model alone): the token at pos is the next step's input"""
    self.y, self.mtp, self.j, self.par, self.pos = [first] + [0]*k, pos, 0, 1, pos

# short prompts prefilled together (Transformer.batch_prefill) from this many: a step of them costs about what three alone do
BATCH_PREFILL_MIN = getenv("BATCH_PREFILL_MIN", 3)
# (BCP) requests prefilled together from this many: continuations of their conversations, the rest of each prompt in its slot (see Transformer.
# batch_prefill's starts): a step of the model's weights for all of them, not one each. fewer: one at a time (a step is G of P rows)
BCP_MIN = getenv("BCP_MIN", 2)

class BatchStats:
  """batched decoding's status line every few seconds while it runs: the requests decoding and waiting now, the tokens generated a second and
  the steps (the batch's) and prefills (of the requests that joined it) since the last line"""
  def __init__(self, every:float=5.0):
    self.every, self.st = every, time.perf_counter()
    self.reset()
  def reset(self):
    self.steps, self.tokens, self.step_time, self.prefills, self.prefill_time, self.mixed = 0, 0, 0.0, 0, 0.0, 0
    self.times:dict[str, float|int] = {}  # (more of the time: where it went, and counts)
  def add(self, name:str, t:float|int): self.times[name] = self.times.get(name, 0) + t
  def step(self, running:int, waiting:int):
    if (dt:=time.perf_counter() - self.st) > self.every and self.steps:
      sep = colored('--', 'BLACK')
      stderr_log(f"{utc_ms()} batch  {sep}  {running:3d} requests decoding, {waiting:3d} waiting  {sep}  "
                 f"{self.tokens/dt:5.0f} tok/s generated  {sep}  "
                 f"{self.steps:4d} steps of {self.step_time/self.steps*1e3:5.1f} ms  {sep}  {self.prefills:3d} prefills ({self.prefill_time:4.1f}s, "
                 f"{self.mixed} mixed)  "
                 f"{sep}  in the last {dt:.0f}s" + "".join(f"  {sep}  {k} {v:.1f}s" if isinstance(v, float) else f"  {sep}  {k} {v}"
                                                           for k, v in self.times.items()) + "\n")
      self.st = time.perf_counter()
      self.reset()
    self.steps += 1

@dataclass
class SessionEndResponse:
  result:dict = DataField(default_factory=dict)
  status:int = 200
  event:threading.Event = DataField(default_factory=threading.Event)

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  """
  requests come in on threads of their own, the model runs on the server's thread: one request alone is decoded as before (speculative
  decoding, its reply cached for the next turn), many at the same time are batched (Transformer.batch_step): every one has a slot of the
  model's states, a step gives the next token of all of them. a request that may not fit a slot (prompt + max_tokens) waits for the model
  alone, one that fills its slot goes on alone (its tokens prefilled again)
  """
  telemetry = Telemetry("")  # (an instance's own replaces it in __init__; servers made without __init__ (tests) observe into this one)
  def handle_error(self, request, client_address):
    # (a client gone before its reply: not an error of the server's, no traceback)
    if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)): return
    super().handle_error(request, client_address)
  daemon_threads = True
  request_queue_size = 1024  # the connections waiting to be accepted (many clients at once: the default of 5 refuses them)
  verbose_prefill = False
  generated_total, prefilled_total = 0, 0  # (the tokens generated and prefilled since the start, see status)
  # (the requests that ended (when, ttft, longest wait for a token, total, prompt, cached, generated, queued): the latencies of status)
  recent: collections.deque = collections.deque(maxlen=1000)
  temperature = 0.0  # a chat completion's without one
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, top_p:float=1.0,
               temperature:float=0.0):
    # top_p: a request's without one (the model's recommended, general.sampling.top_p of its gguf); temperature: a chat completion's
    # (the model: master's Transformer, or a tensor parallel implementation of its own with more (qwen4exp's sequences, glm5next's batch slots))
    self.model: typing.Any = model
    self.model_name, self.tok, self.template, self.top_p = model_name, tok, template, top_p
    self.temperature = temperature
    self.waiting: queue.Queue[Request] = queue.Queue()
    self.inflight = 0  # the requests taken by the model (decoded, or waiting for it): the status line's
    self.telemetry = Telemetry(getenv("TELEMETRY_LOG", ""))
    self.stats = BatchStats()
    self.engine: typing.Any = None  # (schedule_seqs': the sequences)
    self.end_requests: queue.Queue[tuple[str, SessionEndResponse]] = queue.Queue()
    self.ended_sessions: set[str] = set()
    self.ended_dropped = 0
    self.recent = collections.deque(maxlen=1000)
    # the slots of batched decoding (none: one request at a time)
    # (the slots past them: the landing slots of mixed steps, see Transformer.batch_mixed_step)
    self.slots = n if isinstance(n:=getattr(model, "decode_slots", getattr(model, "batch_slots", lambda: 0))(), int) else 0
    # batched decoding is speculative (the drafts of every sequence verified in the step) when the model drafts
    self.spec = k if self.slots and isinstance(k:=getattr(getattr(model, "config", None), "spec_drafts", 0), int) else 0
    # the conversations kept in the free slots (their tokens, the slot's row of the last one's hidden state): a request that continues one is
    # prefilled from its states (agents taking turns)
    self.stash: dict[int, tuple[list[int], int]] = {}
    # (MIX) the requests prefilled in the steps: (request, landing slot, start, position)
    self.landing:list[tuple[Request, int, int, int]] = []
    # (WIDE) the slot after a sequence's that its positions past its own are in (a sequence spanning two slots): not decoded, kept, taken
    self.partner:dict[int, int] = {}
    # (HOST_STASH: GB of host memory for the ones longer than a slot, their states copied there. the least recently used first out)
    # (see HostConv). the chunks of the ones out are used again (new memory and its devices' mappings are slow, ~1 s a GB)
    self.host: list[HostConv] = []
    self.host_free: list[typing.Any] = []
    self.host_made = 0  # (the chunks made: never freed, used again)
    specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
    self.odd_ids = {i for t, i in specials.items() if t in ODD_TOKENS}
    # (SUPERSEDE: the requests running of every (session, prompt), see Handler.run_model)
    self.inflight_keys:dict[tuple, Request] = {}
    self.keys_lock = threading.Lock()
    super().__init__(server_address, Handler)

  def serve_forever(self, poll_interval:float=0.5):
    # (the model's thread polls the devices holding the GIL: the request threads (many streams) take it in turns of SWITCH_INTERVAL, the
    # default 5 ms delays its steps)
    sys.setswitchinterval(getenv("SWITCH_INTERVAL", 0.0005))
    # the model runs here, not on a thread of its own: BEAM's _try_compile installs a SIGALRM handler, which raises ValueError off the main thread
    threading.Thread(target=super().serve_forever, args=(poll_interval,), daemon=True, name="HTTPServer").start()
    # sequences sharing the caches (qwen4exp), else requests in batch slots (glm5next), else one at a time
    if hasattr(type(self.model), "seqs_supported") and self.model.seqs_supported(): self.schedule_seqs()
    elif self.slots: self.schedule()
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
  verbose_prefill = False
  def finish(self, req:Request, error:Exception|None=None) -> bool:
    if req.done: return False
    req.done, req.error = True, error
    # (its checkpoint's host memory after its slot_done: a reply's end is emitted before it, see schedule)
    if req.ckpt_buf is not None: self.ckpt_later = (self.ckpt_later or []) + [req]
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
    self.recent.append((time.time(), (req.pt or et) - req.st, req.max_gap, et - req.st, len(req.ids), req.cache_start, n, req.queued))
    if (telemetry:=getattr(self, "telemetry", None)) is not None: telemetry.finish(req, et - req.st, error)
    self.flush(req)
    req.out.put(None)
    return False

  def status(self, gpus:bool=False) -> dict:
    """the requests and the caches now (GET /status): the requests decoding, being prefilled (their tokens left) and waiting, the sequences in
    the caches and in the stash (their tokens)"""
    now = time.perf_counter()
    ret = typing.cast(dict[str, typing.Any], {"waiting": self.waiting.qsize(), "time": time.time(),
                                               "generated_total": self.generated_total, "prefilled_total": self.prefilled_total})
    if (telemetry:=getattr(self, "telemetry", None)) is not None: ret["telemetry"] = telemetry.snapshot()
    if (q:=getattr(self, "queues", None)) is not None:
      active, filling, pending = (list(x.values()) if isinstance(x, dict) else list(x) for x in q)
      def req(r:Request) -> dict: return {"request_id": r.rid, "session_id": r.session_id or None,
                                          "bench_tag": r.bench_tag or None, "prompt": len(r.ids), "cached": r.cache_start,
                                          "generated": len(r.generated), "age": round(now - r.st, 2)}
      # (a waiting one's hit: the tokens of its prompt the caches or the stash have)
      ret |= {"decoding": [req(r) for r in active], "prefilling": [req(r) | {"left": len(r.fill[0]) - r.fill[1]} for r in filling],
              "waiting": [req(r) | {"hit": r.hit} for r in pending] + [{}] * self.waiting.qsize()}
    # the latencies of the requests of the last 10 minutes: their ttft, their longest wait for a token after the first, their total
    if rs:=[r for r in list(self.recent) if r[0] > time.time() - 600]:
      def pct(xs:list[float]) -> dict:
        xs = sorted(xs)
        return {f"p{q}": round(xs[min(len(xs)-1, int(q / 100 * len(xs)))], 3) for q in (50, 90, 99)} | {"max": round(xs[-1], 3)}
      # (queue: the wait for a sequence, before its prefill)
      ret["last_10m"] = {"requests": len(rs), "ttft": pct([r[1] for r in rs]), "queue": pct([r[7] for r in rs]), "token_gap": pct([r[2] for r in rs]),
                         "total": pct([r[3] for r in rs]),
                         "prompt_tokens": sum(r[4] for r in rs), "cached_tokens": sum(r[5] for r in rs), "generated_tokens": sum(r[6] for r in rs)}
    if gpus:
      # (?gpus: the power and clocks of the devices)
      try:
        from tinygrad.runtime.ops_amd import AMIdleClocks
        ret["gpus"] = AMIdleClocks.metrics()
      except Exception as e: ret["gpus"] = str(e)
    ret["ended_dropped"] = self.ended_dropped
    if (eng:=self.engine) is not None:
      # (held: the idle conversations of agents mid task, see SeqEngine.STICKY. capacity: the tokens the caches and the stash hold, the
      # conversations the stash holds; lost: the conversations out of both)
      seqs, stashed, C = list(eng.seqs), list(eng.stashed), getattr(self.model, "SEQ_MOVE", 0)
      ret |= {"resident": {"sequences": len(seqs), "tokens": sum(s.n for s in seqs), "rows": eng.rows, "rows_taken": sum(s.cap for s in seqs),
                           "held": sum(not s.active and eng.held(s.used) for s in seqs)},
              "stash": {"sequences": len(stashed), "tokens": sum(len(x.tokens) for x in stashed), "chunks_free": len(eng.free_chunks),
                        "held": sum(eng.held(x.used) for x in stashed)},
              "capacity": {"tokens": eng.rows + C * getattr(self.model, "stash_chunks", 0), "slots": eng.nslots,
                           "stash_entries": getattr(self.model, "stash_entries", 0), "max_active": eng.max_active,
                           "conversations": eng.nslots + getattr(self.model, "stash_entries", 0)},
              "evictions": eng.evictions, "lost": eng.lost}
      # (the memory of the devices: weights, caches, slots, workspace, free)
      if (mr:=getattr(self.model, "memory_report", None)) is not None:
        try: ret["memory"] = mr()
        except Exception as e: ret["memory"] = str(e)
    return ret

  # ** the kept conversations
  # ** the model
  def batchable(self, req:Request) -> bool:
    """req can be decoded in a slot: its prompt and room for a reply fit (a reply that fills the slot goes on alone)"""
    # (BATCH_DECODE=0: every request alone, in turn, the slots keep the conversations)
    if not self.slots or not getenv("BATCH_DECODE", 1): return False
    room = min(1024, req.max_tokens - len(req.generated)) if req.max_tokens is not None else 1024
    return len(req.ids) + len(req.generated) + room < self.span_max() * self.model.batch_ctx() - 2*(self.spec + 2)
  def span_max(self) -> int:
    """(WIDE) the slots a sequence spans at most (the model's)"""
    return n if self.spec and isinstance(n:=getattr(self.model, "max_span", lambda: 1)(), int) else 1
  def wide(self, req:Request) -> bool:
    """(WIDE) req spans two slots: its prompt and room for a reply past one"""
    room = min(1024, req.max_tokens - len(req.generated)) if req.max_tokens is not None else 1024
    return self.span_max() > 1 and len(req.ids) + len(req.generated) + room >= self.model.batch_ctx() - self.spec - 2
  def wide_limit(self, i:int) -> int:
    """the positions of the sequence in slot i at most: its slot's, (WIDE) the next one's too but the last ones (its row of nothing writes them)"""
    return self.model.batch_ctx() * (2 if i in self.partner else 1) - (self.spec + 2 if i in self.partner else 0)
  def make_pair(self, active:dict) -> int|None:
    """(WIDE) two free slots next to each other: a sequence moved from between two free ones. the first of them, None: no two free"""
    taken = self.taken(active)
    free = [i for i in range(self.slots) if i not in taken]
    if len(free) < 2: return None
    # (the one after the first free slot, if it's a sequence alone in its slot, into the last free one)
    for a in free:
      if a + 1 < self.slots and a + 1 in active and a + 1 not in self.partner and (b:=free[-1]) != a:
        trace(f"slot_move req {active[a+1].rid} {a+1} -> {b} (a pair for a wide one)")
        self.slot_taken(b)
        self.model.slot_move(a + 1, b)
        active[b] = active.pop(a + 1)
        return a
    return None
  def taken(self, active:dict) -> set[int]:
    """the slots taken: decoded, and (WIDE) the ones after them a sequence's positions are in"""
    return set(active) | set(self.partner.values())
  def free_slots(self, active:dict) -> int:
    """the slots free: not taken, not the ones the requests landing (MIX) go into"""
    return self.slots - len(self.taken(active)) - len(self.landing)
  def free_slot(self, active:dict, wide:bool=False) -> int|None:
    """a free slot (wide: one with the next free too, not the last), None: none"""
    taken = self.taken(active)
    return next((i for i in range(self.slots - wide) if i not in taken and not (wide and i+1 in taken)), None)

  # ** the conversations kept in slots (STASH): the model's states hold one conversation, a request of another one overwrites them. the one
  # there is stored in a free slot first, a request that continues a stored one takes its states back (an agent's turn isn't prefilled again)
  # (SLOT_HOST: the model's conversation is a prompt decoded in a slot: its whole conversation is saved when done, see slot_done)
  main_moved = False
  ckpt_pool: list|None = None  # (host memory for requests' checkpoints, see ckpt_keep)
  ckpt_later: list|None = None  # (the requests done, their checkpoints' host memory freed next)
  STASH_MIN = getenv("STASH_MIN", 512)  # the tokens a conversation has at least to be kept, a request continues one by at least to load it
  def stash_main(self, busy:set[int]):
    """
    the model's conversation into a free slot (an empty one, else the oldest kept) past the busy ones: a batched step writes the free slots
    before its last one (rows of nothing), a request takes the first free one
    """
    model, tokens = self.model, list(getattr(self.model, "_cached_tokens", []))
    if not self.keeps() or not getenv("STASH", 1) or len(tokens) < self.STASH_MIN or any(t == tokens for t, _ in self.stash.values()) or \
       any(e.tokens == tokens for e in self.host) or (self.main_moved and self.slot_host()): return
    if not self.slots or len(tokens) >= model.batch_ctx(): return self.host_save(tokens)
    busy = busy | set(self.partner) | set(self.partner.values())
    free = list(range(max(busy, default=-1)+1, self.slots))
    # (an empty one, else host memory, else a random kept one: agents taking turns in a cycle longer than the slots, the oldest is always the
    # next one used)
    kept = [i for i in self.stash if i in free]
    if not [i for i in free if i not in self.stash] and getenv("HOST_STASH", 0.0) > 0: return self.host_save(tokens)
    if not (victims := [i for i in free if i not in self.stash] or ([random.choice(kept)] if kept else [])): return
    trace(f"stash_main victim {victims[0]} len {len(tokens)} busy {sorted(busy)}")
    model.slot_store(victims[0], len(tokens), getattr(model, "_spec_row", 0))
    self.stash.pop(victims[0], None)
    self.stash[victims[0]] = (tokens, 0)  # (the newest last: the oldest is the next victim)
  def slot_host(self) -> bool:
    """
    (SLOT_HOST, with slots and HOST_STASH) a request decoded in a slot has its whole conversation saved to host memory when done: with more
    agents than slots the slot is taken before its next turn, the prompt alone kept would have the turn prefill the reply again
    """
    return bool(self.slots) and getenv("HOST_STASH", 0.0) > 0 and bool(getenv("SLOT_HOST", 1))
  def keeps(self) -> bool:
    """conversations are kept past the model's (in slots, or in host memory: HOST_STASH, or the sequences sharing its caches)"""
    return bool(self.slots) or getenv("HOST_STASH", 0.0) > 0 or hasattr(self, "engine")
  def host_save(self, tokens:list[int]):
    """(HOST_STASH) the model's conversation into host memory: an older turn of it out, the least recently used out past the GB"""
    model = self.model
    if (most:=int(getenv("HOST_STASH", 0.0) * 1e9) // model.HOST_CHUNK) <= 0: return
    # (a short one is prefilled again about as fast as it's saved and loaded, ~0.5 s)
    if len(tokens) < getenv("HOST_MIN", 4096): return
    st = time.perf_counter()
    ckpt = model.host_ckpt(tokens) if getenv("HOST_CKPT", 1) else 0
    k = model.host_chunks(len(tokens), bool(ckpt))
    # (an older turn of it out first: its chunks are this one's, in their order (its copies the same program). then free ones, new ones up to
    # the GB, the least recently used one's. one this continues from its checkpoint is an older turn too: its client sent the reply back
    # without its reasoning, that reply isn't coming back)
    older = [e for e in self.host if starts(tokens, e.tokens) or (0 < e.ckpt < len(tokens) and starts(tokens, e.tokens, e.ckpt))]
    chunks = max((e.chunks for e in older), key=len, default=[])[:k]
    self.host_free += [c for e in older for c in e.chunks if not any(c is x for x in chunks)]
    self.host = [e for e in self.host if not any(e is o for o in older)]
    made = 0
    while len(self.host_free) < k - len(chunks):
      if self.host_made < most:
        self.host_free.append(model.host_alloc())
        self.host_made, made = self.host_made + 1, made + 1
      elif self.host: self.host_free += self.host.pop(0).chunks
      else: return
    chunks = chunks + [self.host_free.pop() for _ in range(k - len(chunks))]
    model.host_save(len(tokens), chunks, bool(ckpt))
    self.host.append(HostConv(tokens, chunks, ckpt, getattr(model, "_spec_row", 0)))
    trace(f"host_save len {len(tokens)} ckpt {ckpt} {k} chunks ({made} new) in {time.perf_counter()-st:.2f} s, {len(self.host)} kept, "
          f"{len(self.host_free)} free")
  def host_prepare(self):
    """(HOST_STASH) idle: a chunk of host memory made ahead (its memory and its mappings on the devices, ~0.7 s a GB: not a request's wait)"""
    if (gb:=getenv("HOST_STASH", 0.0)) > 0 and getenv("HOST_PREPARE", 1) and self.host_made < int(gb * 1e9) // self.model.HOST_CHUNK:
      self.host_free.append(self.model.host_alloc())
      self.host_made += 1
  @staticmethod
  def host_hit(e:HostConv, ids:list[int]) -> int:
    """the tokens of ids a conversation in host memory has: all of it, its checkpoint's (the prompt of its last reply), or none"""
    return next((n for n in (len(e.tokens), e.ckpt) if 0 < n < len(ids) and starts(ids, e.tokens, n)), 0)
  def kept(self) -> list[tuple[list[int], int]]:
    """
    the kept conversations' tokens and their lengths (and their checkpoints': the same tokens, the checkpoint's length. not a copy of every
    conversation's prefix a request): the model's, the sequences', the stashed ones', the batch slots' and the host's
    """
    ret = [(list(c), len(c))] if isinstance(c:=getattr(self.model, "_cached_tokens", None), list) else []
    if (eng:=getattr(self, "engine", None)) is not None:
      ret += [p for s in list(eng.seqs) for p in ((s.tokens, len(s.tokens)), (s.tokens, s.ckpt))]
      ret += [p for x in list(eng.stashed) for p in ((x.tokens, len(x.tokens)), (x.tokens, x.ckpt))]
    return ret + [(t, len(t)) for t, _ in list(getattr(self, "stash", {}).values())] + \
      [(e.tokens, len(e.tokens)) for e in list(getattr(self, "host", []))]
  def reuse_tokens(self, rendered:str, ids:list[int]) -> list[int]:
    """
    the tokens of a prompt that continues a kept conversation as its text but not its tokens (the model's reply tokenized again differently):
    the kept ones and the rest's, so the cache holds it
    """
    cands = [(t, n) for t, n in self.kept() if 1000 <= n < len(ids) + 1000 and n <= len(t)]
    # (one the prompt's tokens continue: the cache has it. a longer one the prompt's text continues: the ones sharing the most tokens with it
    # (an agent's own conversation, not the others' with the same system prompt), however early they differ: a sampled token of an early
    # reply that isn't its text's tokenization differs from then on, every turn after)
    exact = max((n for t, n in cands if starts(ids, t, n)), default=0)
    # (an agent's own ones: the others' differ in their task, at the start)
    own = [t[:n] for t, n in cands if n > exact and t[:(m:=min(2048, n // 2, len(ids) // 2))] == ids[:m]]
    for t in sorted(own, key=lambda t: common_prefix(t, ids), reverse=True)[:2]:
      if rendered.startswith(text:=self.tok.decode(t)): return t + self.tok.encode(rendered[len(text):])
    return ids

  def miss_log(self, ids:list[int], start:int):
    """a request prefilled from 2000 tokens less than it shares with a kept conversation at least: where they differ"""
    kept = [t for t, n in self.kept() if n == len(t) >= start + 2000 <= len(ids) and t[start+1999] == ids[start+1999]]
    n, t = max(((common_prefix(t, ids), t) for t in kept if same_start(t, ids)), key=lambda x: x[0], default=(0, []))
    if n - start < 2000: return
    dec = self.tok.decode
    stderr_log(colored(f"cache miss: prefilled from {start}, shares {n} of {len(t)} kept tokens: kept {dec(t[max(0,n-30):n+30])!r} "
                       f"vs {dec(ids[max(0,n-30):n+30])!r}\n", "yellow"))

  def stash_load(self, ids:list[int], busy:set[int], target:int|None=None):
    """
    before a prefill of ids (into slot target): the kept conversation it continues the most into the model (if more than the model's own), the
    model's kept. (the target's own is loaded before the prefill writes the slot)
    """
    if not self.keeps() or not getenv("STASH", 1): return
    model, have = self.model, self.model.get_start_pos(ids)
    best = max(((len(t), i) for i, (t, _) in self.stash.items() if (i not in busy or i == target) and len(t) < len(ids) and starts(ids, t)),
               default=(0, -1))
    host = max(((self.host_hit(e, ids), j) for j, e in enumerate(self.host)), default=(0, -1))
    if host[0] > best[0] and host[0] >= have + self.STASH_MIN:
      # (out of the list while the model's is saved: not the one it frees memory of. back as the most recently used)
      e = self.host.pop(host[1])
      self.stash_main(busy)
      self.host.append(e)
      st = time.perf_counter()
      model.host_load(e.chunks, len(e.tokens), e.hrow, bool(e.ckpt))
      self.main_writes(n if callable(f:=getattr(model, "_host_len", None)) and isinstance(n:=f(len(e.tokens)), int) else len(e.tokens))
      model._cached_tokens, model._ckpt_tokens = list(e.tokens), list(e.tokens[:e.ckpt])
      trace(f"host_load len {len(e.tokens)} ckpt {e.ckpt} for {host[0]} in {time.perf_counter()-st:.2f} s")
    elif best[0] >= have + self.STASH_MIN:
      self.stash_main(busy | {best[1]})
      tokens, hrow = self.stash[best[1]]
      trace(f"stash_load slot {best[1]} len {best[0]} hrow {hrow} busy {sorted(busy)} target {target}")
      model.slot_load(best[1], best[0], hrow)
      model._cached_tokens = list(tokens)
    elif have < self.STASH_MIN: self.stash_main(busy)  # (ids overwrite the model's conversation)
  def stash_hit(self, ids:list[int], busy:set[int], target:int|None=None) -> int:
    """the tokens of ids the model or a kept conversation has (where its prefill starts)"""
    if not self.keeps(): return 0
    return max([self.model.get_start_pos(ids)] + [len(t) for i, (t, _) in self.stash.items() if (i not in busy or i == target) and
                                                  len(t) < len(ids) and starts(ids, t)] +
               [self.host_hit(e, ids) for e in self.host])
  # requests waiting this long are taken in turn, before the others the cache has more of
  STARVE = getenv("STARVE", 120.0)
  STARVE_LONG = getenv("STARVE_LONG", 30.0)  # (a request that doesn't fit a slot: the batch drains for it after this long)
  PAUSE_LONG = getenv("PAUSE_LONG", 1)  # (aliased slots: the batch waits for it instead, after PAUSE_WAIT, see pause_long)
  PAUSE_WAIT = getenv("PAUSE_WAIT", 0.0)
  def next_alone(self, pending:list[Request]) -> int:
    """
    the next request alone: the one whose conversation the model or a slot has the most of (agents taking turns: one's next request often comes
    while the others wait, taking it first keeps its cache), but the first one if it has waited too long
    """
    if len(pending) == 1 or not self.keeps() or time.perf_counter() - pending[0].st > self.STARVE: return 0
    return max(range(len(pending)), key=lambda i: (self.stash_hit(pending[i].ids + pending[i].generated, set()), -i))

  LONG_FIRST = getenv("LONG_FIRST", 1)
  def admit_order(self, reqs:list[Request]) -> list[Request]:
    """
    (LONG_FIRST) the requests waiting for a slot, the longest conversation first (agents' longest tasks: their turns one after the other are the
    end of a run of many, a turn's wait for a slot is on that path every turn), the ones that waited past STARVE before them in turn
    """
    if not self.LONG_FIRST or len(reqs) < 2: return reqs
    now = time.perf_counter()
    return sorted(reqs, key=lambda r: (now - r.st <= self.STARVE, -(len(r.ids) + len(r.generated)) if now - r.st <= self.STARVE else r.st))

  def main_writes(self, length:int):
    """
    the model's cache written up to `length` positions: the conversations kept in the slots it overwrites are gone (SLOT_ALIAS: the slots are
    the model's cache past its first batch_ctx positions, a sequence that long is alone)
    """
    if callable(f:=getattr(self.model, "main_overlaps", None)) and isinstance(gone:=f(length), list) and gone:
      trace(f"main_writes {length}: slots {gone} gone")
      self.slot_taken(*gone)
  def drop_caches(self):
    """the kept conversations forgotten (their states may be wrong): every request is prefilled from the start"""
    if (eng:=getattr(self, "engine", None)) is not None: eng.forget()
    if getattr(self, "stash", None): self.stash.clear()
    if getattr(self, "host", None):
      self.host_free += [c for e in self.host for c in e.chunks]
      self.host.clear()
    if hasattr(self.model, "_cached_tokens"): self.model._cached_tokens = []
    if hasattr(self.model, "_ckpt_tokens"): self.model._ckpt_tokens = []
  def step_writes(self, active:dict):
    """
    a batched step of the active slots writes every slot below its size (the next of batch_sizes past the last active one): the free ones
    there are rows of nothing at position 0, the conversations kept in them are gone (their first positions and their states overwritten)
    """
    m = next(s for s in self.model.batch_sizes() if s >= max(active) + 1)
    self.slot_taken(*[i for i in range(m) if i not in active])
  def slot_taken(self, *slots:int):
    for i in slots: self.stash.pop(i, None)
  def slot_done(self, i:int, req:Request, active:dict):
    """
    a request decoded batched in slot i is done (or goes on alone): its conversation is kept there, if no step writes it (past the active ones).
    its last step's rows were its tokens at start..start+k, the first j+1 real ones: the states of its tokens up to the last of them it has
    """
    try: self._slot_done(i, req, active)
    finally: self.partner.pop(i, None)
  def _slot_done(self, i:int, req:Request, active:dict):
    k = self.spec
    if not k or not getenv("STASH", 1) or req.error is not None: return
    start = req.mtp - 1
    n = min(len(req.ids) + len(req.generated), req.pos)
    hrow = (1 - req.par) * (k + 1) + n - start - 1
    if self.slot_host() and n >= max(getenv("HOST_MIN", 4096), start + 1):
      # (its states are the slot's until the next step: into the model's (the model's conversation kept first if it isn't), from it to host)
      st = time.perf_counter()
      if not self.main_moved: self.stash_main(set(active) | {i})
      self.model.slot_load(i, n, hrow)
      self.model._cached_tokens, self.model._ckpt_tokens, self.main_moved = (req.ids + req.generated)[:n], [], False
      # (its prompt's checkpoint too: a next turn without its reasoning goes on from there)
      if req.ckpt_buf is not None and len(ck:=req.ckpt_tokens) < n and starts(self.model._cached_tokens, ck):
        self.model.ckpt_copy(False, req.ckpt_buf)
        self.model._ckpt_tokens = list(ck)
      self.host_save(self.model._cached_tokens)
      trace(f"slot_host req {req.rid} slot {i} len {n} in {time.perf_counter()-st:.2f} s")
    if i <= max(active, default=-1) or n < max(self.STASH_MIN, start + 1) or n >= self.model.batch_ctx() - k - 2: return
    self.stash.pop(i, None)
    self.stash[i] = ((req.ids + req.generated)[:n], hrow)
    trace(f"slot_done req {req.rid} slot {i} len {n} hrow {self.stash[i][1]} active {sorted(active)}")

  # ** one request at a time: its prompt prefilled (from the conversation the model has, if it continues it), then decoded to its end
  def schedule_alone(self):
    while True:
      try: req = self.waiting.get(timeout=0.2)
      except queue.Empty: continue
      self.inflight = 1 + self.waiting.qsize()
      if req.cancelled and not self.finish(req): continue
      req.cache_start = self.model.get_start_pos(req.ids)
      spec_st = list(ss) if isinstance(ss:=getattr(self.model, "spec_stats", None), list) else []
      def collect_spec():
        nonlocal spec_st
        if spec_st and isinstance(ss:=getattr(self.model, "spec_stats", None), list):
          req.steps += ss[0] - spec_st[0]
          spec_st = list(ss)
      g = self.model.generate(list(req.ids), temperature=req.temperature)
      try:
        for tok in g:
          collect_spec()  # emit can finish immediately at max_tokens, before request telemetry
          if not self.emit(req, tok): break
        else:
          collect_spec()
          self.finish(req)
      except Exception as e:
        traceback.print_exc()
        self.finish(req, e)
      finally:
        collect_spec()
        _close(g)
  def ckpt_keep(self, req:Request):
    """
    (SLOT_HOST) req goes on in a slot after its prompt: the model's checkpoint (the states at its prompt's end) copied to host memory, its
    conversation's host_save has it (the model's own checkpoint is another prompt's by then). a client that doesn't send the model's reasoning
    back (its next turn: the reply without it, a tool's result) has the conversation differ right after the prompt: without the checkpoint
    it's prefilled from the start every turn
    """
    model = self.model
    if not self.slot_host() or not getenv("SLOT_CKPT", 1) or req.ckpt_buf is not None or len(req.ids) < getenv("HOST_MIN", 4096): return
    if not callable(getattr(model, "ckpt_copy", None)) or getattr(model, "_ckpt_tokens", None) != req.ids: return
    pool = self.ckpt_pool = self.ckpt_pool or []
    if not pool and (buf:=model.ckpt_alloc()) is not None: pool.append(buf)
    if not pool: return
    req.ckpt_buf, req.ckpt_tokens = pool.pop(), list(req.ids)
    model.ckpt_copy(True, req.ckpt_buf)
    trace(f"ckpt_keep req {req.rid} len {len(req.ids)}")
  def ckpt_free(self, req:Request):
    if req.ckpt_buf is not None:
      self.ckpt_pool = self.ckpt_pool or []
      self.ckpt_pool.append(req.ckpt_buf)
    req.ckpt_buf, req.ckpt_tokens = None, []

  def mixing(self, active:dict) -> int:
    """(MIX) the prompts of the next step (mixed steps: requests' new tokens prefilled in a step of decoding, its sizes the model's)"""
    model = self.model
    if not active or not self.spec or not isinstance(land:=getattr(model, "mix_landing", lambda: 0)(), int) or not land: return 0
    return land if next(m for m in model.batch_sizes() if m > max(active)) in model.mix_sizes() else 0
  def mix_ok(self, req:Request, busy:set[int]) -> bool:
    """(MIX) req's prompt in mixed steps: its new tokens (past what's kept of its conversation) MIX_MAX at most, in chunks of a step's"""
    n = len(req.ids) + len(req.generated)
    new = n - self.stash_hit(req.ids + req.generated, busy)
    return self.batchable(req) and not self.wide(req) and not req.no_mix and 0 < new <= self.MIX_MAX
  MIX_MAX = getenv("MIX_MAX", 4096)
  def mix_take(self, mixes:list[Request], active:dict, pending:list[Request], land:int):
    """
    (MIX) the prompts of the next mixed steps (as many as free landing slots, free slots): each one's conversation from the model's or a kept
    one into its landing slot (self.landing: request, landing slot, start, position)
    """
    model, busy = self.model, {x[1] for x in self.landing}
    for req in self.admit_order(mixes):
      if len(busy) >= land or len(self.taken(active)) + len(self.landing) >= self.slots: break
      ids, landing = req.ids + req.generated, next(self.slots + i for i in range(land) if self.slots + i not in busy)
      _p0 = time.perf_counter()
      trace(f"mix req {req.rid} ids {len(ids)} landing {landing}")
      self.miss_log(ids, self.stash_hit(ids, set(active)))
      self.stash_load(ids, set(active))
      start = model.prefix_to_slot(ids, landing)
      self.main_moved = True
      self.stats.prefill_time += time.perf_counter() - _p0
      # (a cache that has less of it: prefilled before a step)
      if len(ids) - start > self.MIX_MAX:
        req.no_mix = True
        continue
      pending.remove(req)
      self.landing.append((req, landing, start, start))
      busy.add(landing)
  def mix_prompts(self) -> list[tuple[list[int], int, float|tuple[float, float], int]]:
    """(MIX) the prompts of a mixed step: the next chunk of every landing one (as many tokens as the model's largest prompt of a step)"""
    most = max(self.model.MIX_PS)
    return [((req.ids + req.generated)[pos:pos+most], landing, (req.temperature, req.top_p), pos) for req, landing, _, pos in self.landing]
  def mix_done(self, firsts:list[int], active:dict):
    """(MIX) the chunks of a mixed step done: the prompts done from their landing slots into free ones, decoded from their first tokens"""
    model, k, most, left = self.model, self.spec, max(self.model.MIX_PS), []
    for (req, landing, start, pos), first in zip(self.landing, firsts):
      n = len(req.ids) + len(req.generated)
      if (pos:=min(n, pos + most)) < n:
        left.append((req, landing, start, pos))
        continue
      # (its prompt's checkpoint from its slot, as ckpt_keep's: the model's own is another prompt's)
      if self.slot_host() and getenv("SLOT_CKPT", 1) and not req.generated and n >= getenv("HOST_MIN", 4096) and \
         callable(getattr(model, "ckpt_from_slot", None)):
        model.ckpt_from_slot(landing, n)
        model._ckpt_tokens = list(req.ids)
        self.ckpt_keep(req)
        model._ckpt_tokens = []
      # (a free slot is kept for every one landing, see free_slots)
      if (slot:=self.free_slot(active)) is None: raise RuntimeError(f"mixed step: no free slot for req {req.rid}")
      self.slot_taken(slot)
      model.slot_move(landing, slot)
      req.cache_start, req.pos = start, n
      trace(f"mix done req {req.rid} landing {landing} slot {slot} start {start} pos {n}")
      self.stats.prefills += 1
      req.spec_start(first, n, k)
      if self.emit(req, first): active[slot] = req
    self.landing = left

  def bcp_batch(self, cands:list[Request], most:int, busy:set[int]) -> list[Request]:
    """
    (BCP) the requests of a batch_continue: the oldest, then the ones of the fewest new tokens, as many as a step of the rows of the class of
    the most new tokens of them takes (BPF_PS: a step's prompts padded to its rows)
    """
    model = self.model
    if not cands or most <= 0 or not callable(getattr(model, "bpf_class", None)): return cands[:most]
    def new(r:Request) -> int: return len(r.ids) + len(r.generated) - self.stash_hit(r.ids + r.generated, busy)
    batch = [cands[0]] + sorted(cands[1:], key=new)[:most - 1]
    while len(batch) > 1 and len(batch) > max(model.bpf_sizes(model.bpf_class(max(new(r) for r in batch)) or model.BPF_P)): batch.pop()
    return batch

  def batch_continue(self, batch:list[Request], active:dict, pending:list[Request]):
    """
    (BCP) requests into free slots in steps of the model for all of them: each one's conversation (the model's, a kept one, its checkpoint)
    into its slot, the rest of its prompt prefilled there (batch_prefill from its start, in chunks of P), its first token. its prompt's
    checkpoint (in its slot: the model's isn't its) kept with it (ckpt_keep)
    """
    model, k = self.model, self.spec
    _p0 = time.perf_counter()
    free = [i for i in range(self.slots) if i not in self.taken(active)][:len(batch)]
    starts, lens = [], []
    for req, slot in zip(batch, free):
      ids = req.ids + req.generated
      busy = set(active) | set(free)
      trace(f"batch_continue req {req.rid} ids {len(ids)} slot {slot}")
      self.miss_log(ids, self.stash_hit(ids, busy, slot))
      self.stash_load(ids, busy, slot)
      starts.append(model.prefix_to_slot(ids, slot))
      lens.append(len(ids))
      # (the model's conversation is in the slot: not saved again)
      self.main_moved = True
      self.slot_taken(slot)
    pos, firsts = list(starts), [0] * len(batch)
    # (the steps' rows: the class of the most new tokens, BPF_P if past them (chunks of it))
    P = c if callable(getattr(model, "bpf_class", None)) and (c:=model.bpf_class(max(l - s for l, s in zip(lens, starts)))) else model.BPF_P
    while (todo:=[g for g in range(len(batch)) if pos[g] < lens[g]]):
      chunks:list[tuple[list[int], int, float|tuple[float, float]]] = \
        [((batch[g].ids + batch[g].generated)[pos[g]:pos[g] + P], free[g], (batch[g].temperature, batch[g].top_p)) for g in todo]
      outs = model.batch_prefill(chunks, starts=[pos[g] for g in todo], P=P)
      for g, out in zip(todo, outs): pos[g], firsts[g] = min(lens[g], pos[g] + P), out
    for g in range(len(batch)):
      if starts[g] == lens[g]: firsts[g] = model.slot_first(free[g], (batch[g].temperature, batch[g].top_p))
    prefill_time = time.perf_counter() - _p0
    self.stats.prefills, self.stats.prefill_time = self.stats.prefills + len(batch), self.stats.prefill_time + prefill_time
    if (telemetry:=getattr(self, "telemetry", None)) is not None:
      telemetry.prefill(sum(l - s for l, s in zip(lens, starts)), prefill_time, any(s or r.generated for s, r in zip(starts, batch)), len(batch))
    for g, (req, slot) in enumerate(zip(batch, free)):
      pending.remove(req)
      req.cache_start, req.pos = starts[g], lens[g]
      # (its prompt's checkpoint from its slot, as ckpt_keep's: the model's own is another prompt's)
      if self.slot_host() and getenv("SLOT_CKPT", 1) and not req.generated and lens[g] >= getenv("HOST_MIN", 4096) and \
         callable(getattr(model, "ckpt_from_slot", None)):
        model.ckpt_from_slot(slot, lens[g])
        model._ckpt_tokens = list(req.ids)
        self.ckpt_keep(req)
        model._ckpt_tokens = []
      trace(f"batch_continue done req {req.rid} slot {slot} start {starts[g]} pos {req.pos}")
      req.spec_start(firsts[g], req.pos, k)
      if self.emit(req, firsts[g]): active[slot] = req

  def preempt(self, active:dict, pending:list[Request], only:set[int]|None=None):
    """
    (PREEMPT_LONG) the batch drains for a request that doesn't fit a slot: the requests decoding stop now, not after their replies (thousands
    of tokens: the server mostly idle meanwhile). each goes on after it as one that fills its slot does, its conversation kept (in host memory).
    only: the ones in these slots
    """
    stopped = []
    for i, req in sorted(active.items(), reverse=True):
      if only is not None and i not in only: continue
      del active[i]
      self.slot_done(i, req, active)
      stopped.append(req)
    trace(f"preempt reqs {[r.rid for r in stopped]}")
    pending[:0] = sorted(stopped, key=lambda r: r.st)

  def prefill(self, req:Request, busy:set[int]|None=None, target:int|None=None) -> int:
    """the prompt (and what req generated, a continuation) through the model (into slot target): the first sampled token"""
    ids = req.ids + req.generated
    trace(f"prefill req {req.rid} ids {len(ids)} target {target} busy {sorted(busy or set())}")
    self.miss_log(ids, self.stash_hit(ids, busy or set(), target))
    self.stash_load(ids, busy or set(), target)
    self.main_moved = False
    req.cache_start = self.model.get_start_pos(ids)
    g = self.model.generate(list(ids), temperature=req.temperature, top_p=req.top_p)
    try: return next(g)
    finally: _close(g)

  def pause_long(self, req:Request, active:dict, pending:list[Request]):
    """
    (PAUSE_LONG) a request that doesn't fit a slot alone while the batch waits (aliased: the model's cache past batch_ctx is the first slots'):
    the requests in the slots it overwrites stop (see preempt), the others go on from their slots after it, nothing of theirs prefilled again
    """
    model = self.model
    self.preempt(active, pending, only=set(model.main_overlaps(model.max_context)))
    pending.remove(req)
    trace(f"pause_long req {req.rid} active {sorted(active)}")
    try: self.run_alone(req, set(active))
    except Exception as e:
      traceback.print_exc()
      self.finish(req, e)
    # (its conversation past batch_ctx is the slots', the next step writes them: kept in host memory first)
    self.stash_main(set(active))

  def run_alone(self, req:Request, busy:set[int]|None=None) -> bool:
    """
    req alone on the model: speculative decoding until its end, or until another request comes (it goes on in slot 0: returns True). busy: the
    slots of a batch that waits for it (it goes on to its end)
    """
    model = self.model
    ids = req.ids + req.generated
    trace(f"alone req {req.rid} ids {len(ids)}")
    self.miss_log(ids, self.stash_hit(ids, busy or set()))
    self.stash_load(ids, busy or set())
    self.main_moved = False
    req.cache_start = model.get_start_pos(ids)
    spec_st = list(ss) if isinstance(ss:=getattr(model, "spec_stats", None), list) else []
    def collect_spec():
      nonlocal spec_st
      if spec_st and isinstance(ss:=getattr(model, "spec_stats", None), list):
        req.steps += ss[0] - spec_st[0]
        spec_st = list(ss)
    g = model.generate(list(ids), temperature=req.temperature, top_p=req.top_p)
    try:
      for tok in g:
        collect_spec()  # emit can finish immediately at max_tokens, before the finally block
        if not self.emit(req, tok): return False
        if busy is None and not self.waiting.empty() and self.batchable(req):
          # another request: this one goes on batched (the model's states after its last token's position are the slot's)
          _close(g)
          req.pos = len(model._cached_tokens)
          self.ckpt_keep(req)
          self.slot_taken(0)
          trace(f"alone to batch req {req.rid} slot 0 pos {req.pos} row {getattr(model, '_spec_row', 0)}")
          model.slot_store(0, req.pos, getattr(model, "_spec_row", 0))
          self.main_moved = True
          req.spec_start(req.generated[-1], req.pos, self.spec)
          return True
      collect_spec()
      return self.finish(req)
    finally:
      collect_spec()
      _close(g)
      # (its positions and the drafts past them)
      self.main_writes(max(len(ids) + len(req.generated), len(getattr(model, "_cached_tokens", []))) + 64)

  # ** sequences sharing the model's caches (Transformer.seqs_supported): continuous batching
  # (a prompt is prefilled in slices of PREFILL_SLICE tokens (~1 s). the decoding ones' steps run after MAX_STALL s of slices (0.3: 1.0 had
  # replies wait 1.5/2.3 s p50/p90 for a token under agent load, 0.3: 0.8/1.5 s, and more tokens), for their share
  # of the time (the requests decoding of the ones in flight, DECODE_SHARE at most and 0.1 at least): a long prompt doesn't stop the replies,
  # many prompts at once are prefilled mostly one after another (a step of many is about as fast as one of a few). the next slice is the
  # prompt's with the fewest tokens left to prefill (a short one, an agent's turn, isn't behind a long one), less AGING tokens a second it
  # waited. a long one isn't behind short ones that keep coming forever: one waiting past STARVE_SEQ goes first, and takes a sequence even if
  # a held conversation is lost for it (see SeqEngine.STICKY))
  PREFILL_SLICE, MAX_STALL, DECODE_SHARE = getenv("PREFILL_SLICE", 13824), getenv("MAX_STALL", 0.3), getenv("DECODE_SHARE", 0.5)
  AGING, STARVE_SEQ, TURN_WAIT = getenv("AGING", 0), getenv("SEQ_STARVE", 600.0), getenv("SEQ_TURN_WAIT", 10.0)
  # (SEQ_STARVE_ALT: after a slice of a starving prompt the next one is the best of the others (a waiting one admitted, a warm turn's short
  # suffix): a long cold prompt past STARVE takes every other slice, not all of them while every other request waits behind it)
  STARVE_ALT = getenv("SEQ_STARVE_ALT", 0)
  COLD_MAX, COLD_TOKENS = getenv("SEQ_COLD_MAX", 1), getenv("SEQ_COLD_TOKENS", 16384)
  # (the gpus' clocks (AM: their gfx clock's cap, see AMIdleClocks.set_caps): a few users decoding at the full clocks (a user's tokens fast),
  # more at less (the energy a token: ~0.85 J at 2000 MHz on 6x 7900 XTX, ~1.2 J at full), a prefill at a cap it isn't slower at. switched
  # CLOCK_HOLD s apart at most)
  # (CLOCK_FAST, CLOCK_MANY, CLOCK_PREFILL: a mode's caps from the env (e.g. "GFXCLK=1900"; "" none), a box's thermals without a code change)
  CLOCKS = {"fast": getenv("CLOCK_FAST", ""), "many": getenv("CLOCK_MANY", "GFXCLK=2000"), "prefill": getenv("CLOCK_PREFILL", "GFXCLK=2300")}
  FAST_USERS, CLOCK_HOLD = getenv("CLOCK_FAST_USERS", 4), 0.5
  def process_session_ends(self):
    """Execute end requests only on the model's scheduler thread; a current HTTP response is never cancelled."""
    if not hasattr(self, "end_requests"): self.end_requests = queue.Queue()
    if not hasattr(self, "ended_sessions"): self.ended_sessions = set()
    if not hasattr(self, "ended_dropped"): self.ended_dropped = 0
    while not self.end_requests.empty():
      session_id, answer = self.end_requests.get_nowait()
      if session_id in self.ended_sessions:
        result = {"session_id": session_id, "already_ended": True, "dropped": 0, "resident_dropped": 0, "stash_dropped": 0}
      else:
        try:
          result = self.engine.end_session(session_id)
          self.ended_dropped += result["dropped"]
          self.ended_sessions.add(session_id)
          result.update({"session_id": session_id, "already_ended": False})
        except SessionInFlight:
          result = {"session_id": session_id, "already_ended": False, "active": True,
                    "dropped": 0, "resident_dropped": 0, "stash_dropped": 0}
      result.setdefault("active", False)
      result["ended_dropped"] = self.ended_dropped
      answer.result, answer.status = result, 409 if result["active"] else 200
      answer.event.set()

  def schedule_seqs(self):
    """
    every request has a sequence of its own (the resident one its prompt continues: an agent's turn is prefilled from its last one), prefilled
    when it comes, then decoded with all the others decoding (a step of speculative decoding of every one at once, see Transformer.seq_step)
    """
    from tinygrad.llm.engine import SeqEngine, Seq
    eng = self.engine = SeqEngine(self.model)
    eng.telemetry = self.telemetry
    active: dict[Seq, Request] = {}
    filling: dict[Seq, Request] = {}  # (the sequences of the prompts being prefilled: their request's fill is where)
    pending: list[Request] = []
    self.queues = (active, filling, pending)
    k = eng.k
    mix_tokens = int(getattr(self.model, "SEQ_MIX_TOKENS", 0))
    # (a paused one (see below) goes on from its tokens so far)
    def prompt(req:Request) -> list[int]: return req.ids + req.generated if req.generated else req.ids
    stall = 0.0  # (the time of the slices since the last steps)
    starved_last = False  # (the last slice was a starving prompt's, see STARVE_ALT)
    clocks = f(self.CLOCKS) if callable(f:=getattr(self.model, "clock_caps", None)) else self.CLOCKS
    try:
      from tinygrad.runtime.ops_amd import AMIdleClocks
      set_caps = AMIdleClocks.set_caps if AMIdleClocks.devs else None
    except ImportError: set_caps = None
    clock_mode, clock_at = "", 0.0
    while True:
      try:
        self.process_session_ends()
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
        # the prompt to prefill a slice of: the ones waiting past STARVE first (in turn), then the one with the fewest tokens left (aged: AGING),
        # a waiting one taking a sequence (one that would lose a held conversation waits, see SeqEngine.take: an agent's next turn first)
        now, job = time.perf_counter(), None
        def starving(r:Request) -> bool: return now - r.st > self.STARVE_SEQ
        cands:list[tuple[bool, float, int, Seq|Request]] = \
          [(not starving(r), r.st if starving(r) else len(r.fill[0]) - r.fill[1] - self.AGING*(now - r.st), 0, s) for s, r in filling.items()]
        # (what the caches have of a waiting one's prompt, and whether it fits: looked at again when the engine's sequences change, its take
        # tried again then or a second later (a held conversation's hold ends))
        if len(active) + len(filling) < eng.max_active:
          t0 = time.perf_counter()
          for r in pending:
            if r.hit_v != eng.version: r.hit, r.hit_v = eng.cached(prompt(r), touch=True), eng.version
            r.was_hit = r.was_hit or r.hit > 0
          self.stats.add("cached", time.perf_counter() - t0)
          # (cold ones: SEQ_COLD_MAX being prefilled at most (a cold prefill of an agent's conversation is seconds: many at once and the warm
          # turns wait behind them, starve, lose theirs and come back cold))
          cold = sum(r.cold for r in filling.values()) >= self.COLD_MAX
          cands += [(not starving(r), (r.st if starving(r) else len(r.ids) + len(r.generated) - r.hit - self.AGING*(now - r.st)), 1, r)
                    for r in pending if (r.tried != eng.version or now - r.tried_at > 1.0) and
                    not (cold and len(r.ids) + len(r.generated) - r.hit > self.COLD_TOKENS)]
        order = sorted(cands, key=lambda x: x[:3])
        if self.STARVE_ALT and starved_last: order = [x for x in order if x[0]] + [x for x in order if not x[0]]
        starved_last = False
        for fresh, _, _, c in order:
          if isinstance(c, Seq):
            job, starved_last = c, not fresh
            break
          t0 = time.perf_counter()
          # (an agent's turn (it continues (continued) a conversation) waiting past SEQ_TURN_WAIT s: the held conversations not waited for go to
          # the stash for its rows, the least recently used ones there lost if it's full (an agent in a long tool call: one re-prefill later,
          # not every waiting turn blocked by idle ones' rows). a new conversation past the caches' and the stash's capacity waits for one to
          # end (unless starving and nothing else runs))
          force = (c.was_hit and now - c.st > self.TURN_WAIT) or (starving(c) and not (active or filling))
          got = eng.take(ids:=prompt(c), max(0, (c.max_tokens or 0) - len(c.generated)), force=force, session_id=c.session_id)
          take_time = time.perf_counter() - t0
          self.stats.add("take", take_time)
          trace(f"seq_take at={t0:.6f} ms={take_time*1e3:.3f} success={got is not None} "
                f"active={len(active)} pending={len(pending)} filling={len(filling)}")
          if got is None:
            c.tried, c.tried_at = eng.version, now
            continue
          pending.remove(c)
          job, start, restore = got
          if not c.generated: c.cache_start, c.queued = start, time.perf_counter() - c.st
          self.telemetry.cache_observe(c.session_id, len(ids), self.model.max_context, start, eng.last_take_stash)
          c.fill, c.cold = (ids, start, restore), len(ids) - start > self.COLD_TOKENS
          if active and 0 < len(ids) - start <= mix_tokens:
            # (SEQ_MIX_TOKENS: force this short suffix in the draft slots of the ordinary steps, beside the other sequences)
            eng.mix_start(job, ids, start, restore, (c.temperature, c.top_p))
            active[job] = c
            trace(f"seq_mix_admit req={c.rid} rows={len(ids)-start} cached={start} active={len(active)}")
            job = None
            break
          filling[job] = c
          self.miss_log(ids, start)
          starved_last = not fresh
          break
        for name, t in eng.times.items(): self.stats.add(name, t)
        eng.times.clear()
        if set_caps is not None and (mode:="prefill" if job is not None else "many" if len(active) > self.FAST_USERS else "fast") != clock_mode \
           and time.perf_counter() - clock_at >= self.CLOCK_HOLD:
          set_caps(clocks[mode])
          clock_mode, clock_at = mode, time.perf_counter()
        if job is not None:
          req = filling[job]
          ids, pos, restore = req.fill
          # (a model's slices in whole prefill chunks (GLM: its greedy tokens generate's, see Transformer.seq_slice))
          sl = f(self.PREFILL_SLICE) if callable(f:=getattr(self.model, "seq_slice", None)) else self.PREFILL_SLICE
          end = len(ids) if len(ids) - pos <= sl * 3 // 2 else pos + sl
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
          new_slice = (end if end < len(ids) else len(ids)) - pos
          trace(f"seq_prefill at={st:.6f} ms={slice_time*1e3:.3f} rows={new_slice} cached={pos} "
                f"active={len(active)} pending={len(pending)} filling={len(filling)}")
          self.telemetry.prefill(new_slice, slice_time, bool(req.cache_start or req.generated))
          self.prefilled_total += new_slice
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
          settle_time = time.perf_counter() - st
          stall += settle_time
          trace(f"seq_settle at={st:.6f} ms={settle_time*1e3:.3f} active={len(active)}")
        # (the waiting ones that can't take a sequence now (no room) don't count: their prefill isn't the steps' time)
        can = len(filling) + sum(r.tried != eng.version for r in pending)
        share = min(self.DECODE_SHARE, max(0.1, len(active) / max(1, len(active) + can)))
        budget, stall = stall * share / (1 - share), 0.0
        st0 = time.perf_counter()
        while active:
          seqs = list(active)
          mixing = {s for s in seqs if s.mix}
          st = time.perf_counter()
          outs = eng.step(seqs)
          step_time = time.perf_counter() - st
          for s in mixing:
            if s.mix: continue
            req, (ids, start, _) = active[s], active[s].fill
            new = len(ids) - start
            self.stats.prefills, self.stats.mixed = self.stats.prefills + 1, self.stats.mixed + 1
            self.prefilled_total += new
            trace(f"seq_mixed req={req.rid} rows={new} cached={start} active={len(active)}")
          self.stats.step_time += step_time
          step_tokens = sum(len(toks) for toks in outs)
          self.telemetry.step(len(seqs), max(len(r.ids) + len(r.generated) for r in active.values()), step_time, step_tokens)
          trace(f"seq_step at={st:.6f} ms={step_time*1e3:.3f} rows={len(seqs)} tokens={step_tokens} "
                f"active={len(active)} pending={len(pending)} filling={len(filling)}")
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

  def schedule(self):
    model, slots, k = self.model, self.slots, self.spec
    active: dict[int, Request] = {}  # the slot of every request decoded batched
    pending: list[Request] = []  # requests waiting for a slot or for the model alone
    self.landing = []
    while True:
      try:
        # (a wait of a while at a time: a Ctrl-C stops the server, a lock's wait without a timeout doesn't see it)
        while not active and not pending and not self.landing:
          try: pending.append(self.waiting.get(timeout=0.2))
          except queue.Empty: self.host_prepare()
        while not self.waiting.empty(): pending.append(self.waiting.get_nowait())
        pending = [r for r in pending if not (r.cancelled and not self.finish(r))]
        for r in self.ckpt_later or []: self.ckpt_free(r)
        self.ckpt_later = []
        self.inflight = len(active) + len(pending) + len(self.landing)
        # (a request that doesn't fit a slot waiting past STARVE_LONG: the slots take no more requests (the batch drains, see preempt), then it
        # goes on alone. else it waits for no request to be decoded: while agents take turns that's never)
        # (PAUSE_LONG: the batch waits for it, see pause_long)
        if active and self.PAUSE_LONG and callable(getattr(model, "main_overlaps", None)) and \
           (long_req:=next((r for r in pending if not self.batchable(r) and time.perf_counter() - r.st > self.PAUSE_WAIT), None)) is not None:
          self.pause_long(long_req, active, pending)
          continue
        drain = any(not self.batchable(r) and time.perf_counter() - r.st > self.STARVE_LONG for r in pending)
        if drain and active and getenv("PREEMPT_LONG", 1): self.preempt(active, pending)
        if not active and pending:
          # one request (or one that doesn't fit a slot): alone
          if len(pending) == 1 or not self.batchable(pending[0]) or drain:
            long = next((i for i, r in enumerate(pending) if not self.batchable(r)), None) if drain else None
            req = pending.pop(long if long is not None else self.next_alone(pending))
            try:
              if self.run_alone(req): active[0] = req
            except Exception as e:
              traceback.print_exc()
              self.finish(req, e)
            continue
        # (BCP) new requests into free slots, prefilled together: each one's conversation from the model's or a kept one into its slot, the rest
        # of its prompt (in chunks of P) continued there
        if (G:=model.batch_prefills()) and getenv("BCP", 0) and callable(getattr(model, "prefix_to_slot", None)) and not drain:
          room, most = model.batch_ctx() - model.BPF_P, getenv("BCP_MAX_NEW", 1536)
          # (a prompt of many new tokens: alone, in big chunks (the prefill's chunks of up to 1536 rows run faster than a batch's of P))
          def cont(r:Request) -> bool:
            n = len(r.ids) + len(r.generated)
            return self.batchable(r) and n <= room and n - self.stash_hit(r.ids + r.generated, set(active)) <= most
          # (fewer than a step's worth while others decode: they wait BCP_WAIT seconds at most for more, agents' turns come one at a time)
          while len(batch:=self.bcp_batch([r for r in pending if cont(r)], min(G, self.free_slots(active)), set(active))) >= BCP_MIN and \
                (len(batch) >= G or not active or time.perf_counter() - min(r.st for r in batch) > getenv("BCP_WAIT", 1.5)):
            self.batch_continue(batch, active, pending)
          # (the ones that wait for more: not prefilled alone meanwhile. one that waited long enough is, if no others came)
          waiting_bcp = {r for r in pending if cont(r) and time.perf_counter() - r.st <= getenv("BCP_WAIT", 1.5)} if active else set()
        else: waiting_bcp = set()
        # new requests into free slots: short prompts prefilled together into theirs, (the others) prefilled, their states copied into the slot
        def shorts() -> list[Request]: return [r for r in pending if self.batchable(r) and not r.generated and len(r.ids) <= model.BPF_P]
        # (with a step's worth waiting, they wait for that many free slots)
        defer = bool(G:=model.batch_prefills()) and bool(active) and len(shorts()) >= G and self.free_slots(active) < G
        while G and not defer and not drain and len(short:=shorts()[:min(G, self.free_slots(active))]) >= BATCH_PREFILL_MIN:
          _p0 = time.perf_counter()
          free = [i for i in range(slots) if i not in self.taken(active)][:len(short)]
          self.slot_taken(*free)
          firsts = model.batch_prefill([(r.ids, slot, (r.temperature, r.top_p)) for r, slot in zip(short, free)])
          prefill_time = time.perf_counter() - _p0
          self.stats.prefills, self.stats.prefill_time = self.stats.prefills + len(short), self.stats.prefill_time + prefill_time
          self.telemetry.prefill(sum(len(r.ids) for r in short), prefill_time, False, len(short))
          for req, slot, first in zip(short, free, firsts):
            pending.remove(req)
            req.cache_start, req.pos = 0, len(req.ids)
            req.spec_start(first, req.pos, k)
            if self.emit(req, first): active[slot] = req
        # (MIX: the requests of mixed steps wait for theirs)
        mixes = [r for r in pending if self.mix_ok(r, set(active))] if (land:=self.mixing(active)) and not drain else []
        if mixes: self.mix_take(mixes, active, pending, land)
        for req in self.admit_order([r for r in pending if self.batchable(r) and not (defer and r in shorts()) and r not in waiting_bcp and
                                     r not in mixes]):
          if len(self.taken(active)) + len(self.landing) >= slots or drain: break
          wide = self.wide(req)
          to = self.free_slot(active, wide) if self.free_slots(active) >= 1 + wide else None
          # (WIDE: two free slots apart: a sequence moved for them to be next to each other. else it waits for them, before the others)
          if to is None and wide and self.free_slots(active) >= 2: to = self.make_pair(active)
          if to is None: break
          slot = to
          _p0 = time.perf_counter()
          req.queued = _p0 - req.st
          first = self.prefill(req, self.taken(active) | {slot} | ({slot+1} if wide else set()), slot)
          self.slot_taken(slot)
          prefill_time = time.perf_counter() - _p0
          self.stats.prefills, self.stats.prefill_time = self.stats.prefills + 1, self.stats.prefill_time + prefill_time
          self.telemetry.prefill(len(req.ids) + len(req.generated) - req.cache_start, prefill_time, bool(req.cache_start))
          pending.remove(req)
          req.pos = len(req.ids) + len(req.generated)
          self.ckpt_keep(req)
          trace(f"slot_store req {req.rid} slot {slot} pos {req.pos}" + (f" wide {slot+1}" if wide else ""))
          model.slot_store(slot, req.pos)
          if wide:
            self.partner[slot] = slot + 1
            self.slot_taken(slot + 1)
          self.main_moved = True
          req.spec_start(first, req.pos, k)
          if self.emit(req, first): active[slot] = req
        if not active and not self.landing: continue
        self.stats.step(len(active), len(pending) + self.waiting.qsize())
        step_n = len(active)
        step_context = max(len(r.ids) + len(r.generated) for r in active.values())
        _t0 = time.perf_counter()
        # a step of every slot up to the last one used (the free ones in between are rows of nothing)
        self.inflight = len(active) + len(pending) + self.waiting.qsize() + len(self.landing)
        trace("step " + " ".join(f"{i}:{r.rid}@{r.pos}" for i, r in sorted(active.items())))
        self.step_writes(active)
        if k:
          # (WIDE: a partner slot's row of nothing at its last positions, past the ones of the sequence spanning it, see wide_limit)
          # (every row of the step's size: a mixed step's is the largest)
          hi = max(active, default=-1) + 1
          if self.partner: hi = next(m for m in model.batch_sizes() if m >= max(hi, max(self.partner.values()) + 1))
          if self.landing and self.partner: hi = max(model.mix_sizes())
          seqs:list[tuple[list[int], int, int, int, float|tuple[float, float]]] = \
            [(active[i].y, active[i].mtp, active[i].j, active[i].par, (active[i].temperature, active[i].top_p)) if i in active else
             ([0]*(k+1), model.batch_ctx() - k - 2 if i in self.partner.values() else 0, 0, 1, 0.0) for i in range(hi)]
          if self.landing:
            prompts = self.mix_prompts()
            trace("mix step " + " ".join(f"{r.rid}@{pos}+{len(t)}" for (r, _, _, pos), (t, *_) in zip(self.landing, prompts)))
            spec, firsts = model.batch_mixed_step(seqs, prompts)
          else: spec = model.batch_spec_step(seqs)
        else: out = model.batch_step([(active[i].generated[-1], active[i].pos, (active[i].temperature, active[i].top_p)) if i in active else
                                      (0, 0, 0.0)
                                      for i in range(max(active)+1)])
        step_time = time.perf_counter() - _t0
        self.stats.step_time += step_time
        step_tokens = 0
        for i, req in list(active.items()):
          req.steps += 1
          if k:
            # the drafts accepted: up to the first that isn't the sampled token, the sampled tokens after them (see Transformer._spec_steps)
            sampled, drafts = spec[i]
            j = next((t for t in range(k) if drafts[t] != sampled[t]), k)
            start = req.mtp + req.j
            req.y, req.mtp, req.j, req.par, req.pos = sampled, start + 1, j, 1 - req.par, start + j + 1
            step_tokens += j + 1
            if not all(self.emit(req, t) for t in sampled[:j+1]):
              del active[i]
              self.slot_done(i, req, active)
              continue
          else:
            step_tokens += 1
            req.pos += 1
            if not self.emit(req, out[i]):
              del active[i]
              continue
          if req.pos + k + 2 >= model.batch_ctx() and i not in self.partner and self.span_max() > 1 and i + 1 < slots and \
             i + 1 not in self.taken(active) and self.free_slots(active) > 0:
            # (WIDE) its slot is full, the next is free: its positions go on there
            self.partner[i] = i + 1
            self.slot_taken(i + 1)
            trace(f"wide req {req.rid} slot {i} -> {i+1} pos {req.pos}")
          if req.pos + k + 2 >= self.wide_limit(i):
            # its slot is full: it goes on alone (from its kept states, or prefilled again), when the model is free
            del active[i]
            self.slot_done(i, req, active)
            pending.insert(0, req)
        if self.landing: self.mix_done(firsts, active)
        self.telemetry.step(step_n, step_context, step_time, step_tokens)
        # the sequences in the first slots: a step of fewer rows (the one of the last slot used is the size of every step)
        while active and (hi:=max(active)) >= len(active) and hi not in self.partner and \
              next(m for m in model.batch_sizes() if m >= hi+1) > next(m for m in model.batch_sizes() if m >= len(self.taken(active))):
          if (lo:=self.free_slot(active)) is None or lo > hi: break
          self.slot_taken(lo)
          trace(f"slot_move req {active[hi].rid} {hi} -> {lo}")
          model.slot_move(hi, lo)
          active[lo] = active.pop(hi)
        # the last one batched (not speculative), nothing waiting: it goes on alone (prefilled again), speculative decoding is faster
        if not k and len(active) == 1 and not pending and self.waiting.empty() and getenv("BATCH_LAST_ALONE", 1):
          pending.append(active.pop(next(iter(active))))
      except Exception as e:
        traceback.print_exc()
        for req in [*active.values(), *pending, *(x[0] for x in self.landing)]: self.finish(req, e)
        active, pending, self.landing = {}, [], []
