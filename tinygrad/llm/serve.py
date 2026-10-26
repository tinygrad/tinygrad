from __future__ import annotations
import json, pathlib, queue, random, re, select, socket, socketserver, threading, time, traceback, typing, uuid
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

# ** the Anthropic Messages API (/v1/messages): requests as the chat completions' messages and tools, replies as its content blocks and events

def anthropic_messages(body:dict) -> tuple[list[dict], list[dict]|None]:
  """an Anthropic request's system, messages and tools as chat completions' (thinking blocks: reasoning_content, tool results: tool messages)"""
  def text_of(c) -> str:
    if isinstance(c, str): return c
    return "".join(b.get("text", "") if b.get("type") == "text" else "[image]" if b.get("type") == "image" else "" for b in c or [])
  msgs:list[dict] = []
  if (system:=body.get("system")): msgs.append({"role": "system", "content": text_of(system)})
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
PING = -1  # (Handler.next_token: no token for a while)

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
                reasoning:bool=False, log:str="", keepalive:float|None=None, top_p:float|None=None):
    tok = self.server.tok
    prompt_tokens = len(ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    dec = tok.stream_decoder()
    tags, sentinels = markup_tags(sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {})
    router = StreamRouter(reasoning, tags)
    # the model runs on the server's thread (LLMServer.serve_forever): the request waits there, its tokens come back here
    # (no top_p: the model's own, general.sampling.top_p)
    req = Request(ids, log, max_tokens, temperature, self.server.top_p if top_p is None else top_p)
    self.server.waiting.put(req)
    try:
      yield chunk({"role":"assistant", "content":""})
      # (special tokens no reply has: the sign of a model gone wrong, logged)
      specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
      odd = {i for t, i in specials.items() if t in ODD_TOKENS}
      while (next_id := Handler.next_token(self, req, keepalive)) is not None:
        if next_id == PING:
          yield {"ping": True}
          continue
        out.append(next_id)
        if next_id in odd:
          stderr_log(log + colored(f"odd token {next_id} after {len(out)} tokens of the reply\n", "red"))
          trace(f"odd token req {req.rid} after {len(out)}")
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
      try: return req.out.get(timeout=1.0)
      except queue.Empty: pass

  def do_POST(self):
    request_st = time.perf_counter()
    log = f"{self.path}  {colored('--', 'BLACK')}  "
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if (path:=self.path.split("?")[0]) in ("/v1/messages", "/v1/messages/count_tokens"): return self.anthropic(path, body, log, request_st)
    if self.path == "/v1/chat/completions":
      # render and tokenize
      normalize_messages(body["messages"])
      rendered = self.render(body["messages"], body.get("tools"), body)
      ids: list[int] = self.server.reuse_tokens(rendered, self.server.tok.encode(rendered))
      log += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
      if len(ids) >= self.server.model.max_context:
        stderr_log(log + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      # (a reply not streamed and not done in HOLD_JSON seconds: its headers then, and a space every HOLD_JSON seconds until it's done
      # (whitespace before JSON is JSON): a proxy that drops a connection quiet for a while doesn't drop it, a client doesn't ask again)
      hold = 0.0 if body.get("stream") else getenv("HOLD_JSON", 5.0)
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              reasoning=rendered.rstrip().endswith("<think>"), log=log,
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
      raise RuntimeError(f"unhandled path {self.path}")

  def render(self, messages:list[dict], tools:list[dict]|None, body:dict) -> str:
    # (a reasoning effort asked for is the template's, else its default)
    effort = body.get("reasoning_effort") or (body.get("output_config") or {}).get("effort")
    return self.server.template.render(messages=messages, tools=tools, add_generation_prompt=True, preserve_thinking=True,
                                       **({"reasoning_effort": effort} if effort else {}))

  def anthropic(self, path:str, body:dict, log:str, request_st:float):
    """the Anthropic Messages API: the request as chat completions' messages, the reply's chunks as its content blocks (and stream events)"""
    msgs, tools = anthropic_messages(body)
    normalize_messages(msgs)
    rendered = self.render(msgs, tools, body)
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
                            temperature=float(body.get("temperature", 1.0)), reasoning=rendered.rstrip().endswith("<think>"), log=log,
                            keepalive=getenv("KEEPALIVE", 15.0) if body.get("stream") else None,
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
def trace(msg:str):
  """(SERVE_TRACE: a file) the model's operations of the scheduler, in order: to tell what came before a reply went wrong"""
  if (path:=getenv("SERVE_TRACE", "")):
    with open(path, "a") as f: f.write(f"{time.time():.3f} {msg}\n")

class Request:
  """a request to the model: its prompt, its sampled tokens come back in out (None: the end), it's cancelled when the client goes away"""
  def __init__(self, ids:list[int], log:str, max_tokens:int|None, temperature:float, top_p:float=1.0):
    self.ids, self.log, self.max_tokens, self.temperature, self.top_p, self.rid = ids, log, max_tokens, temperature, top_p, next(_rids)
    self.cancelled, self.finish_reason, self.checked, self.zeros = False, "stop", 0.0, 0
    self.error: Exception|None = None
    self.out: queue.Queue[int|None] = queue.Queue()
    self.generated: list[int] = []
    self.st, self.pt, self.cache_start, self.steps = time.perf_counter(), 0.0, 0, 0
    self.pos, self.done = 0, False  # batched decoding: the position of its last token (the next step's input)
    # batched speculative decoding: the last step's sampled tokens, the position the MTP layer catches up from, the drafts accepted, the parity
    self.y:list[int] = []
    self.mtp, self.j, self.par = 0, 0, 1
  def spec_start(self, first:int, pos:int, k:int):
    """the slot's state after a prefill (or the model alone): the token at pos is the next step's input"""
    self.y, self.mtp, self.j, self.par, self.pos = [first] + [0]*k, pos, 0, 1, pos

# short prompts prefilled together (Transformer.batch_prefill) from this many: a step of them costs about what three alone do
BATCH_PREFILL_MIN = getenv("BATCH_PREFILL_MIN", 3)

class BatchStats:
  """batched decoding's status line every few seconds while it runs: the requests decoding and waiting now, the tokens generated a second and
  the steps (the batch's) and prefills (of the requests that joined it) since the last line"""
  def __init__(self, every:float=5.0):
    self.every, self.st = every, time.perf_counter()
    self.reset()
  def reset(self): self.steps, self.tokens, self.step_time, self.prefills, self.prefill_time = 0, 0, 0.0, 0, 0.0
  def step(self, running:int, waiting:int):
    if (dt:=time.perf_counter() - self.st) > self.every and self.steps:
      sep = colored('--', 'BLACK')
      stderr_log(f"batch  {sep}  {running:3d} requests decoding, {waiting:3d} waiting  {sep}  {self.tokens/dt:5.0f} tok/s generated  {sep}  "
                 f"{self.steps:4d} steps of {self.step_time/self.steps*1e3:5.1f} ms  {sep}  {self.prefills:3d} prefills ({self.prefill_time:4.1f}s)  "
                 f"{sep}  in the last {dt:.0f}s\n")
      self.st = time.perf_counter()
      self.reset()
    self.steps += 1

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  """
  requests come in on threads of their own, the model runs on the server's thread: one request alone is decoded as before (speculative
  decoding, its reply cached for the next turn), many at the same time are batched (Transformer.batch_step): every one has a slot of the
  model's states, a step gives the next token of all of them. a request that may not fit a slot (prompt + max_tokens) waits for the model
  alone, one that fills its slot goes on alone (its tokens prefilled again)
  """
  daemon_threads = True
  request_queue_size = 1024  # the connections waiting to be accepted (many clients at once: the default of 5 refuses them)
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, top_p:float=1.0):
    # top_p: a request's without one (the model's recommended, general.sampling.top_p of its gguf)
    self.model, self.model_name, self.tok, self.template, self.top_p = model, model_name, tok, template, top_p
    self.waiting: queue.Queue[Request] = queue.Queue()
    self.inflight = 0  # the requests taken by the model (decoded, or waiting for it): the status line's
    self.stats = BatchStats()
    # the slots of batched decoding (none: one request at a time)
    self.slots = n if isinstance(n:=getattr(model, "batch_slots", lambda: 0)(), int) else 0
    # batched decoding is speculative (the drafts of every sequence verified in the step) when the model drafts
    self.spec = k if self.slots and isinstance(k:=getattr(getattr(model, "config", None), "spec_drafts", 0), int) else 0
    # the conversations kept in the free slots (their tokens, the slot's row of the last one's hidden state): a request that continues one is
    # prefilled from its states (agents taking turns)
    self.stash: dict[int, tuple[list[int], int]] = {}
    # (HOST_STASH: GB of host memory for the ones longer than a slot, their states copied there. the least recently used first out)
    # (see HostConv). the chunks of the ones out are used again (new memory and its devices' mappings are slow, ~1 s a GB)
    self.host: list[HostConv] = []
    self.host_free: list[typing.Any] = []
    self.host_made = 0  # (the chunks made: never freed, used again)
    specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
    self.odd_ids = {i for t, i in specials.items() if t in ODD_TOKENS}
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
    # (the model's states gone wrong (NaN: its argmax is token 0, again and again), or a special token no reply has: the reply fails, the
    # caches are dropped, the client's retry is prefilled from the start)
    req.zeros = req.zeros + 1 if tok == 0 else 0
    if req.zeros >= getenv("CORRUPT_ZEROS", 16) or tok in self.odd_ids:
      why = "token 0 again and again" if req.zeros else f"token {tok}"
      stderr_log(req.log + colored(f"the model went wrong after {len(req.generated)} tokens ({why}, {req.cache_start} of {len(req.ids)} cached): "
                                   "the reply fails, the caches are dropped\n", "red") + f"  ...{self.tok.decode(req.generated[-60:])!r}\n")
      trace(f"corrupt req {req.rid} after {len(req.generated)}")
      self.drop_caches()
      return self.finish(req, RuntimeError("the model's states went wrong (dropped: retry)"))
    req.generated.append(tok)
    req.out.put(tok)
    self.stats.tokens += 1
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
    stderr_log(req.log + f"reqs:{self.inflight:3d}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{req.cache_start:5d}', 'green')} +{len(req.ids)-req.cache_start:5d}  {colored('--', 'BLACK')}  "
               f"prefill:{pre:5.0f} tok/s  {colored('--', 'BLACK')}  gen:{(n-1)/(et-req.pt) if n > 1 and req.pt else 0:4.0f} tok/s  "
               f"{colored('--', 'BLACK')}  {spec}out:{n:5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if req.cancelled else total}\n")
    req.out.put(None)
    return False

  # ** the model
  def batchable(self, req:Request) -> bool:
    """req can be decoded in a slot: its prompt and room for a reply fit (a reply that fills the slot goes on alone)"""
    # (BATCH_DECODE=0: every request alone, in turn, the slots keep the conversations)
    if not self.slots or not getenv("BATCH_DECODE", 1): return False
    room = min(1024, req.max_tokens - len(req.generated)) if req.max_tokens is not None else 1024
    return len(req.ids) + len(req.generated) + room < self.model.batch_ctx()

  # ** the conversations kept in slots (STASH): the model's states hold one conversation, a request of another one overwrites them. the one
  # there is stored in a free slot first, a request that continues a stored one takes its states back (an agent's turn isn't prefilled again)
  STASH_MIN = getenv("STASH_MIN", 512)  # the tokens a conversation has at least to be kept, a request continues one by at least to load it
  def stash_main(self, busy:set[int]):
    """
    the model's conversation into a free slot (an empty one, else the oldest kept) past the busy ones: a batched step writes the free slots
    before its last one (rows of nothing), a request takes the first free one
    """
    model, tokens = self.model, list(getattr(self.model, "_cached_tokens", []))
    if not self.keeps() or not getenv("STASH", 1) or len(tokens) < self.STASH_MIN or any(t == tokens for t, _ in self.stash.values()) or \
       any(e.tokens == tokens for e in self.host): return
    if not self.slots or len(tokens) >= model.batch_ctx(): return self.host_save(tokens)
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
  def keeps(self) -> bool:
    """conversations are kept past the model's (in slots, or in host memory: HOST_STASH)"""
    return bool(self.slots) or getenv("HOST_STASH", 0.0) > 0
  def host_save(self, tokens:list[int]):
    """(HOST_STASH) the model's conversation into host memory: an older turn of it out, the least recently used out past the GB"""
    model = self.model
    if (most:=int(getenv("HOST_STASH", 0.0) * 1e9) // model.HOST_CHUNK) <= 0: return
    # (a short one is prefilled again about as fast as it's saved and loaded, ~0.5 s)
    if len(tokens) < getenv("HOST_MIN", 4096): return
    st = time.perf_counter()
    ckpt = model.host_ckpt(tokens) if getenv("HOST_CKPT", 1) else 0
    k = model.host_chunks(len(tokens))
    # (an older turn of it out first: its chunks are this one's, in their order (its copies the same program). then free ones, new ones up to
    # the GB, the least recently used one's)
    older = [e for e in self.host if tokens[:len(e.tokens)] == e.tokens]
    chunks = max((e.chunks for e in older), key=len, default=[])[:k]
    self.host_free += [c for e in older for c in e.chunks if not any(c is x for x in chunks)]
    self.host = [e for e in self.host if tokens[:len(e.tokens)] != e.tokens]
    made = 0
    while len(self.host_free) < k - len(chunks):
      if self.host_made < most:
        self.host_free.append(model.host_alloc())
        self.host_made, made = self.host_made + 1, made + 1
      elif self.host: self.host_free += self.host.pop(0).chunks
      else: return
    chunks = chunks + [self.host_free.pop() for _ in range(k - len(chunks))]
    model.host_save(len(tokens), chunks)
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
    return next((n for n in (len(e.tokens), e.ckpt) if 0 < n < len(ids) and ids[:n] == e.tokens[:n]), 0)
  def kept(self) -> list[list[int]]:
    """the kept conversations' tokens: the model's, the slots', the host's"""
    model = c if isinstance(c:=getattr(self.model, "_cached_tokens", []), list) else []
    return [list(model)] + [t for t, _ in list(self.stash.values())] + [e.tokens for e in list(self.host)]
  def reuse_tokens(self, rendered:str, ids:list[int]) -> list[int]:
    """
    the tokens of a prompt that continues a kept conversation (the model's, or one in a slot) as its text but not its tokens (the model's reply
    tokenized again differently): the kept ones and the rest's, so the cache holds it
    """
    if not getenv("REUSE_TOKENS", 1): return ids
    # (the conversations, and their checkpoints: the text after one can be different, a reply before it tokenized differently)
    ck = c if isinstance(c:=getattr(self.model, "_ckpt_tokens", None), list) else []
    for t in sorted(self.kept() + [list(ck)] + [e.tokens[:e.ckpt] for e in list(self.host) if e.ckpt], key=len, reverse=True):
      if len(t) < 1000 or len(t) >= len(ids) + 1000: continue
      if ids[:len(t)] == t: return ids
      if next((i for i, (a, b) in enumerate(zip(t, ids)) if a != b), len(t)) < len(t) // 2: continue
      if rendered.startswith(text:=self.tok.decode(t)): return t + self.tok.encode(rendered[len(text):])
    return ids

  def miss_log(self, ids:list[int], start:int):
    """(CACHE_LOG) a request prefilled from much less than it shares with a kept conversation: where they differ"""
    if not self.keeps() or not getenv("CACHE_LOG", 1): return
    kept = self.kept()
    def lcp(t:list[int]) -> int: return next((i for i, (a, b) in enumerate(zip(t, ids)) if a != b), min(len(t), len(ids)))
    n, t = max(((lcp(t), t) for t in kept), key=lambda x: x[0], default=(0, []))
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
    best = max(((len(t), i) for i, (t, _) in self.stash.items() if (i not in busy or i == target) and len(t) < len(ids) and ids[:len(t)] == t),
               default=(0, -1))
    host = max(((self.host_hit(e, ids), j) for j, e in enumerate(self.host)), default=(0, -1))
    if host[0] > best[0] and host[0] >= have + self.STASH_MIN:
      # (out of the list while the model's is saved: not the one it frees memory of. back as the most recently used)
      e = self.host.pop(host[1])
      self.stash_main(busy)
      self.host.append(e)
      st = time.perf_counter()
      model.host_load(e.chunks, len(e.tokens), e.hrow)
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
                                                  len(t) < len(ids) and ids[:len(t)] == t] +
               [self.host_hit(e, ids) for e in self.host])
  # requests waiting this long are taken in turn, before the others the cache has more of
  STARVE = getenv("STARVE", 120.0)
  def next_alone(self, pending:list[Request]) -> int:
    """
    the next request alone: the one whose conversation the model or a slot has the most of (agents taking turns: one's next request often comes
    while the others wait, taking it first keeps its cache), but the first one if it has waited too long
    """
    if len(pending) == 1 or not self.keeps() or time.perf_counter() - pending[0].st > self.STARVE: return 0
    return max(range(len(pending)), key=lambda i: (self.stash_hit(pending[i].ids + pending[i].generated, set()), -i))

  def drop_caches(self):
    """the model's conversation and the kept ones forgotten (their states may be wrong): every request is prefilled from the start"""
    self.stash.clear()
    self.host_free += [c for e in self.host for c in e.chunks]
    self.host.clear()
    if hasattr(self.model, "_cached_tokens"): self.model._cached_tokens, self.model._ckpt_tokens = [], []
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
    k = self.spec
    if not k or not getenv("STASH", 1) or i <= max(active, default=-1) or req.error is not None: return
    start = req.mtp - 1
    n = min(len(req.ids) + len(req.generated), req.pos)
    if n < max(self.STASH_MIN, start + 1) or n >= self.model.batch_ctx() - k - 2: return
    self.stash.pop(i, None)
    self.stash[i] = ((req.ids + req.generated)[:n], (1 - req.par) * (k + 1) + n - start - 1)
    trace(f"slot_done req {req.rid} slot {i} len {n} hrow {self.stash[i][1]} active {sorted(active)}")

  def prefill(self, req:Request, busy:set[int]|None=None, target:int|None=None) -> int:
    """the prompt (and what req generated, a continuation) through the model (into slot target): the first sampled token"""
    ids = req.ids + req.generated
    trace(f"prefill req {req.rid} ids {len(ids)} target {target} busy {sorted(busy or set())}")
    self.miss_log(ids, self.stash_hit(ids, busy or set(), target))
    self.stash_load(ids, busy or set(), target)
    req.cache_start = self.model.get_start_pos(ids)
    g = self.model.generate(list(ids), temperature=req.temperature, top_p=req.top_p)
    try: return next(g)
    finally: _close(g)

  def run_alone(self, req:Request) -> bool:
    """req alone on the model: speculative decoding until its end, or until another request comes (it goes on in slot 0: returns True)"""
    model = self.model
    ids = req.ids + req.generated
    trace(f"alone req {req.rid} ids {len(ids)}")
    self.miss_log(ids, self.stash_hit(ids, set()))
    self.stash_load(ids, set())
    req.cache_start = model.get_start_pos(ids)
    spec_st = list(ss) if isinstance(ss:=getattr(model, "spec_stats", None), list) else []
    g = model.generate(list(ids), temperature=req.temperature, top_p=req.top_p)
    try:
      for tok in g:
        if not self.emit(req, tok): return False
        if not self.waiting.empty() and self.batchable(req):
          # another request: this one goes on batched (the model's states after its last token's position are the slot's)
          _close(g)
          req.pos = len(model._cached_tokens)
          self.slot_taken(0)
          trace(f"alone to batch req {req.rid} slot 0 pos {req.pos} row {getattr(model, '_spec_row', 0)}")
          model.slot_store(0, req.pos, getattr(model, "_spec_row", 0))
          req.spec_start(req.generated[-1], req.pos, self.spec)
          return True
      return self.finish(req)
    finally:
      if spec_st and isinstance(ss:=getattr(model, "spec_stats", None), list): req.steps += ss[0] - spec_st[0]
      _close(g)

  def schedule(self):
    model, slots, k = self.model, self.slots, self.spec
    active: dict[int, Request] = {}  # the slot of every request decoded batched
    pending: list[Request] = []  # requests waiting for a slot or for the model alone
    while True:
      try:
        # (a wait of a while at a time: a Ctrl-C stops the server, a lock's wait without a timeout doesn't see it)
        while not active and not pending:
          try: pending.append(self.waiting.get(timeout=0.2))
          except queue.Empty: self.host_prepare()
        while not self.waiting.empty(): pending.append(self.waiting.get_nowait())
        pending = [r for r in pending if not (r.cancelled and not self.finish(r))]
        self.inflight = len(active) + len(pending)
        if not active and pending:
          # one request (or one that doesn't fit a slot): alone
          if len(pending) == 1 or not self.batchable(pending[0]):
            req = pending.pop(self.next_alone(pending))
            try:
              if self.run_alone(req): active[0] = req
            except Exception as e:
              traceback.print_exc()
              self.finish(req, e)
            continue
        # new requests into free slots: short prompts prefilled together into theirs, (the others) prefilled, their states copied into the slot
        def shorts() -> list[Request]: return [r for r in pending if self.batchable(r) and not r.generated and len(r.ids) <= model.BPF_P]
        # (with a step's worth waiting, they wait for that many free slots)
        defer = bool(G:=model.batch_prefills()) and bool(active) and len(shorts()) >= G and slots - len(active) < G
        while G and not defer and len(short:=shorts()[:min(G, slots - len(active))]) >= BATCH_PREFILL_MIN:
          _p0 = time.perf_counter()
          free = [i for i in range(slots) if i not in active][:len(short)]
          self.slot_taken(*free)
          firsts = model.batch_prefill([(r.ids, slot, (r.temperature, r.top_p)) for r, slot in zip(short, free)])
          self.stats.prefills, self.stats.prefill_time = self.stats.prefills + len(short), self.stats.prefill_time + time.perf_counter() - _p0
          for req, slot, first in zip(short, free, firsts):
            pending.remove(req)
            req.cache_start, req.pos = 0, len(req.ids)
            req.spec_start(first, req.pos, k)
            if self.emit(req, first): active[slot] = req
        for req in [r for r in pending if self.batchable(r) and not (defer and r in shorts())]:
          if len(active) >= slots: break
          slot = next(i for i in range(slots) if i not in active)
          _p0 = time.perf_counter()
          first = self.prefill(req, set(active) | {slot}, slot)
          self.slot_taken(slot)
          self.stats.prefills, self.stats.prefill_time = self.stats.prefills + 1, self.stats.prefill_time + time.perf_counter() - _p0
          pending.remove(req)
          req.pos = len(req.ids) + len(req.generated)
          trace(f"slot_store req {req.rid} slot {slot} pos {req.pos}")
          model.slot_store(slot, req.pos)
          req.spec_start(first, req.pos, k)
          if self.emit(req, first): active[slot] = req
        if not active: continue
        self.stats.step(len(active), len(pending) + self.waiting.qsize())
        _t0 = time.perf_counter()
        # a step of every slot up to the last one used (the free ones in between are rows of nothing)
        self.inflight = len(active) + len(pending) + self.waiting.qsize()
        trace("step " + " ".join(f"{i}:{r.rid}@{r.pos}" for i, r in sorted(active.items())))
        self.step_writes(active)
        if k:
          spec = model.batch_spec_step([(active[i].y, active[i].mtp, active[i].j, active[i].par, (active[i].temperature, active[i].top_p))
                                        if i in active else
                                        ([0]*(k+1), 0, 0, 1, 0.0) for i in range(max(active)+1)])
        else: out = model.batch_step([(active[i].generated[-1], active[i].pos, (active[i].temperature, active[i].top_p)) if i in active else
                                      (0, 0, 0.0)
                                      for i in range(max(active)+1)])
        self.stats.step_time += time.perf_counter() - _t0
        for i, req in list(active.items()):
          req.steps += 1
          if k:
            # the drafts accepted: up to the first that isn't the sampled token, the sampled tokens after them (see Transformer._spec_steps)
            sampled, drafts = spec[i]
            j = next((t for t in range(k) if drafts[t] != sampled[t]), k)
            start = req.mtp + req.j
            req.y, req.mtp, req.j, req.par, req.pos = sampled, start + 1, j, 1 - req.par, start + j + 1
            if not all(self.emit(req, t) for t in sampled[:j+1]):
              del active[i]
              self.slot_done(i, req, active)
              continue
          else:
            req.pos += 1
            if not self.emit(req, out[i]):
              del active[i]
              continue
          if req.pos + k + 2 >= model.batch_ctx():
            # its slot is full: it goes on alone (from its kept states, or prefilled again), when the model is free
            del active[i]
            self.slot_done(i, req, active)
            pending.insert(0, req)
        # the sequences in the first slots: a step of fewer rows (the one of the last slot used is the size of every step)
        while active and (hi:=max(active)) >= len(active) and next(m for m in model.batch_sizes() if m >= hi+1) > \
                         next(m for m in model.batch_sizes() if m >= len(active)):
          lo = next(i for i in range(slots) if i not in active)
          self.slot_taken(lo)
          trace(f"slot_move req {active[hi].rid} {hi} -> {lo}")
          model.slot_move(hi, lo)
          active[lo] = active.pop(hi)
        # the last one batched (not speculative), nothing waiting: it goes on alone (prefilled again), speculative decoding is faster
        if not k and len(active) == 1 and not pending and self.waiting.empty() and getenv("BATCH_LAST_ALONE", 1):
          pending.append(active.pop(next(iter(active))))
      except Exception as e:
        traceback.print_exc()
        for req in [*active.values(), *pending]: self.finish(req, e)
        active, pending = {}, []
