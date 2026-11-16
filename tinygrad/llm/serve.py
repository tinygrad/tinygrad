from __future__ import annotations
import collections, functools, json, math, os, pathlib, queue, re, select, socket, socketserver, sys, threading, time, traceback, typing, uuid
from dataclasses import dataclass, field as DataField
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log, getenv
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
from tinygrad.llm.telemetry import Telemetry, utc_ms
from tinygrad.llm.engine import SessionInFlight
if typing.TYPE_CHECKING: from tinygrad.llm.engine import SeqEngine, Seq
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

# ** VISION (VISION_MMPROJ, see cli.py): a chat request's image_url parts (data: URLs) are pictures the model sees: their parts the template's
# image parts, its one image pad of each the picture's rows (their ids: qwen_vision.picture_ids), the tower's rows of them in the prefill
# (a picture is resized down to at most VISION_MAX_TOKENS rows of 32 x 32 pixels: 1,024 by default = about one megapixel, the details of a
# larger one are lost; the usage's prompt_tokens count its rows)

def extract_pictures(messages:list[dict]) -> list:
  """the pictures of a request's messages in order, each part made the template's image part; a qwen_vision.PictureError (a 400) for one it can't
  have (not a data: URL, not an image, too large or too many, in a system message), PictureBusy (a 503) if the process's decoders were busy for
  VISION_DECODE_WAIT seconds"""
  from tinygrad.llm.qwen_vision import PictureError, PictureBusy, VISION_MAX_IMAGES, VISION_DECODE_WAIT, decode_picture
  pics:list = []
  deadline = time.monotonic() + VISION_DECODE_WAIT
  for m in messages:
    if not isinstance(c:=m.get("content"), list): continue
    for i, part in enumerate(c):
      if not isinstance(part, dict) or part.get("type") != "image_url": continue
      if m.get("role") == "system": raise PictureError("a system message cannot contain images")
      if len(pics) == VISION_MAX_IMAGES: raise PictureError(f"more than {VISION_MAX_IMAGES} images in a request")
      url = u.get("url") if isinstance(u:=part.get("image_url"), dict) else u
      try: pics.append(decode_picture(url, deadline))
      except PictureBusy: raise
      except PictureError as e: raise type(e)(f"image {len(pics) + 1}: {e}") from None
      c[i] = {"type": "image"}
  return pics

def expand_pictures(ids:list[int], pics:list, pad:int) -> tuple[list[int], dict[int, typing.Any]]:
  """the prompt's image pads (one a picture, in order) made the pictures' rows' ids, and the pictures by their first row's id"""
  from tinygrad.llm.qwen_vision import PictureError
  slots = positions(ids, pad)
  if len(slots) != len(pics): raise PictureError(f"the prompt has {len(slots)} image slots for {len(pics)} images (an image token in the text?)")
  out:list[int] = []
  at, by_id = 0, {}
  for i, pic in zip(slots, pics):
    out.extend(ids[at:i])
    out.extend(pid:=pic.ids())
    by_id[pid[0]], at = pic, i + 1
  out.extend(ids[at:])
  return out, by_id

def positions(ids:list[int], x:int) -> list[int]:
  """the indices of x in ids (list.index's scan: a Pokemon prompt of 300K tokens in ~1 ms, not ~10 ms a Python loop)"""
  ret, i = [], -1
  try:
    while True: ret.append(i:=ids.index(x, i + 1))
  except ValueError: return ret

def redacted(body:typing.Any) -> typing.Any:
  """a request body to print: every data: URL a placeholder of its length (no picture's bytes in a log)"""
  if isinstance(body, dict): return {k: redacted(v) for k, v in body.items()}
  if isinstance(body, list): return [redacted(v) for v in body]
  if isinstance(body, str) and body.startswith("data:"): return f"data:...({len(body)} chars)"
  return body

class RenderCache:
  """(SERVE_RENDER_CACHE=1, off by default) a conversation's rendered text up to its last assistant message, kept: a request that repeats
  those messages and adds no user message renders only the ones after it (a Pokemon step at 250K: the template over ~2,500 messages, ~80 ms).
  The text after the cut is the template's over the conversation's start (its system messages and first user message) and the new messages,
  less the start's own text: the same text where a message renders from itself, its neighbours' roles and the last user message (Qwen 3.x,
  GLM 5.x). A new user message (a template may then render older turns differently: preserve_thinking=False, clear_thinking): the full
  render, unless the template with these arguments renders an assistant turn the same with a user message after it (probed once: pi sends a
  tool's screenshot as a user message after the tool's, every Pokemon step), or the kept text's request had a user message after its cut too
  (those turns were rendered as turns with a user message after them; a user message whose text is a tool's response wrapped counts apart:
  Qwen does not count it, GLM does), and the template renders such a turn the same whatever follows it (probed once). A template argument
  not in ARGS (add_vision_id: pictures numbered over the conversation): the full render. Checked: a cut is made only where the full text
  ends with that tail, and every SERVE_RENDER_CHECK-th (64) cached render is compared with the full one (a difference: logged, the cache
  off). Its renders run under its lock (the template holds the GIL anyway)"""
  ARGS = frozenset({"tools", "preserve_thinking", "reasoning_effort", "enable_thinking", "clear_thinking"})
  def __init__(self, template:typing.Any, size:int=64, check:int=64):
    self.template, self.size, self.check, self.on, self.hits = template, size, check, True, 0
    self.entries:list[tuple[str, list[dict], int, int, str]] = []  # (arguments' key, messages, start's length, cut, text before the cut)
    self.lock, self.user_stable, self.user_settled = threading.Lock(), typing.cast(dict[str, bool], {}), typing.cast(dict[str, bool], {})

  def stable(self, key:str, kw:dict) -> bool:
    """the template with these arguments renders an assistant turn (its reasoning, a tool call) the same when a user message follows it"""
    if (ret:=self.user_stable.get(key)) is None:
      a:dict = {"role": "assistant", "content": "a", "reasoning_content": "r",
           "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}]}
      ms:list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, a, {"role": "tool", "tool_call_id": "c", "content": "t"}]
      try: ret = self.full(ms + [{"role": "user", "content": "v"}], kw).startswith(self.full(ms, kw, False))
      except Exception: ret = False
      self.user_stable[key] = ret
    return ret

  def settled(self, key:str, kw:dict) -> bool:
    """the template with these arguments renders an assistant turn with a user message after it the same whatever messages come later"""
    if (ret:=self.user_settled.get(key)) is None:
      def turn(i:int) -> list[dict]:
        return [{"role": "assistant", "content": f"a{i}", "reasoning_content": f"r{i}",
                 "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "f", "arguments": {}}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"t{i}"}, {"role": "user", "content": f"u{i}"}]
      ms:list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}] + turn(0)
      try: ret = self.full(ms + turn(1), kw).startswith(self.full(ms, kw, False))
      except Exception: ret = False
      self.user_settled[key] = ret
    return ret

  @staticmethod
  def users(messages:list[dict]) -> tuple[bool, bool]:
    """(a user message, a user message that is not a tool's response wrapped) among messages"""
    def text(c) -> str:
      return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else ""
    us = [text(m.get("content")).strip() for m in messages if m.get("role") == "user"]
    return bool(us), any(not (t.startswith("<tool_response>") and t.endswith("</tool_response>")) for t in us)

  def full(self, messages:list[dict], kw:dict, gen:bool=True) -> str:
    return self.template.render(messages=messages, add_generation_prompt=gen, **kw)

  def tail(self, start:list[dict], new:list[dict], kw:dict) -> str|None:
    """the text the messages new add after a conversation's messages that begin with start and end with an assistant message"""
    base, text = self.full(start, kw, False), self.full(start + new, kw)
    return text[len(base):] if text.startswith(base) else None

  def entry(self, key:str, kw:dict, messages:list[dict], s:int, text:str) -> tuple|None:
    """the entry of messages (rendered: text) at their last cut (after an assistant message, past the start), if the text's end is its tail"""
    k = next((k for k in range(len(messages)-1, s, -1) if messages[k-1].get("role") == "assistant"), None)
    if k is None or (t:=self.tail(messages[:s], messages[k:], kw)) is None or not text.endswith(t): return None
    return (key, list(messages), s, k, text[:len(text)-len(t)])

  def render(self, messages:list[dict], template:typing.Any=None, /, **kw) -> str:
    # (another template (the server's changed): its own entries)
    if template is not None and template is not self.template: self.template, self.entries = template, []
    if not self.on or not self.ARGS.issuperset(kw): return self.full(messages, kw)
    key = json.dumps(kw, sort_keys=True, default=str)
    # (the start: the leading system messages and the first user message)
    s = next((i+1 for i, m in enumerate(messages) if m.get("role") not in ("system", "developer")), 0)
    if not s or messages[s-1].get("role") != "user": return self.full(messages, kw)
    with self.lock:
      # (the entry with the longest kept text this request repeats whose turns before the cut it renders the same: the messages after the
      # cut have a user message here as there (none, or one and the template settled), or the template is stable)
      def same(e:tuple) -> bool:
        return (u:=self.users(messages[e[3]:])) == self.users(e[1][e[3]:]) and (not u[0] or self.settled(key, kw)) or self.stable(key, kw)
      hit = max((e for e in self.entries if e[0] == key and e[2] == s and e[3] < len(messages) and messages[:e[3]] == e[1][:e[3]] and same(e)),
                key=lambda e: e[3], default=None)
      text = None
      if hit is not None and (t:=self.tail(messages[:s], messages[hit[3]:], kw)) is not None:
        text, self.hits = hit[4] + t, self.hits + 1
        if self.check and self.hits % self.check == 0 and text != (want:=self.full(messages, kw)):
          stderr_log(colored(f"render cache: a cached render differs from the full one ({len(messages)} messages): cache off\n", "red"))
          self.on, text = False, want
      if text is None: text = self.full(messages, kw)
      if self.on and (e:=self.entry(key, kw, messages, s, text)) is not None:
        self.entries = [e] + [x for x in self.entries if x is not hit][:self.size-1]
    return text

def normalize_messages(messages:list[dict], images:bool=False) -> None:
  # (images: VISION's image parts (see extract_pictures) stay the template's)
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # string-only templates need text parts joined and explicit null content emptied
    if m.get("content", "") is None: m["content"] = ""
    elif isinstance(c := m.get("content"), list):
      c = m["content"] = [p if not isinstance(p, dict) or p.get("type") not in MEDIA_KINDS or (images and p.get("type") == "image") else
                          {"type": "text", "text": media_note(p)} for p in c]
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
# KEEPALIVE seconds without a chunk (env KEEPALIVE, default 15))
HOLD_JSON, KEEPALIVE = 5.0, getenv("KEEPALIVE", 15.0)
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

def collapse_pictures(t:list[int], pad:int) -> list[int]:
  """VISION: a token list with a picture's rows (negative ids, qwen_vision.picture_ids) as the template's text has it: each picture one image pad"""
  if min(t, default=0) >= 0: return t
  from tinygrad.llm.qwen_vision import ID_BITS
  mask = (1 << ID_BITS) - 1
  return [x if x >= 0 else pad for x in t if x >= 0 or (-x - 1) & mask == 0]

def common_prefix_pictures(t:list[int], ids:list[int], pad:int, pads:list[int]|None=None) -> tuple[int, int]:
  """
  VISION: the tokens t (with pictures' rows) and ids (a pad a picture) start with as text: (the count in ids, the count in t). pads: the
  positions of the pads in ids (made once for many t). compared a stretch between two pictures at a time; a picture's rows skipped by a
  binary search (its ids are consecutive)
  """
  from tinygrad.llm.qwen_vision import ID_BITS
  if pads is None: pads = positions(ids, pad)
  mask, i, j = (1 << ID_BITS) - 1, 0, 0
  for p in pads + [len(ids)]:
    if (a:=t[j:j + p - i]) != (b:=ids[i:p]): return i + (k:=common_prefix(a, b)), j + k
    i, j = p, j + p - i
    if p == len(ids) or j >= len(t) or (x:=t[j]) >= 0 or (-x - 1) & mask: return i, j
    lo, hi = 1, min(len(t) - j, 1 << ID_BITS)
    while lo < hi:
      mid = (lo + hi + 1) // 2
      if t[j + mid - 1] == x - (mid - 1): lo = mid
      else: hi = mid - 1
    i, j = i + 1, j + lo
  return i, j

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

def sampling_top_k(body:dict) -> int|None:
  """a request's top_k checked (None: the server's, TOP_K; 0 or -1: none, as vLLM's -1)"""
  if (v:=body.get("top_k")) is None: return None
  if isinstance(v, bool) or not isinstance(v, int) or v < -1: raise ValueError(f"top_k must be an integer >= -1, not {v!r}")
  return max(v, 0)

class StopFilter:
  """
  a request's stop strings over the reply's text (reasoning too, as the reference API): feed() returns the text safe to send (a tail that
  could start a stop string held back) and the stop string found (the text before it sent, the rest not)
  """
  def __init__(self, stops:list[str]): self.stops, self.held = [x for x in stops if x], ""
  def feed(self, piece:str, final:bool=False) -> tuple[str, str|None]:
    t = self.held + piece
    if (hits:=sorted((t.find(x), x) for x in self.stops if x in t)):
      self.held = ""
      return t[:hits[0][0]], hits[0][1]
    hold = 0 if final else max((k for k in range(1, len(t)+1) if any(x.startswith(t[-k:]) and x != t[-k:] for x in self.stops)), default=0)
    self.held = t[len(t)-hold:]
    return t[:len(t)-hold], None

def stop_strings(body:dict) -> list[str]:
  """a request's stop (a string or a list of up to 16): a ValueError the request's 400"""
  if (v:=body.get("stop")) is None: return []
  if isinstance(v, str): v = [v]
  if not isinstance(v, list) or len(v) > 16 or not all(isinstance(x, str) for x in v): raise ValueError("stop must be a string or a list of strings")
  return [x for x in v if x]

def strip_json_fence(s:str) -> str:
  """the reply of a JSON response_format as a bare document: a Markdown code fence around it removed (```json ... ```)"""
  return m[1] if (m:=re.fullmatch(r"\s*```[a-zA-Z]*\s*\n?(.*?)\s*```\s*", s, re.DOTALL)) else s.strip()

def partial_tool_call(s:str, tags:dict[str, str]=TEXT_TAGS) -> tuple[str, str]|None:
  """
  a tool call the token cap cut (no end tag): its name and its arguments as far as they came, a JSON text cut there (as the reference API
  returns one), None without a name. GLM's name<arg_key>k</arg_key><arg_value>v..., hermes' {"name": ..., "arguments": ..., XML's <function=
  """
  s = s.strip()
  if s.startswith("{"):
    if not (m:=re.match(r'\{\s*"name"\s*:\s*"([^"]+)"', s)): return None
    return m[1], a[1] if (a:=re.search(r'"(?:arguments|parameters)"\s*:\s*(.*)$', s, re.DOTALL)) else "{"
  ak, akc, av, avc = (re.escape(tags[t]) for t in ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"))
  if (fm:=re.match(r"<function=([^>]+)>(.*)$", s, re.DOTALL)): name, body, pair = fm[1], fm[2], r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)"
  elif (fm:=re.match(r"([\w.-]+)(.*)$", s, re.DOTALL)): name, body, pair = fm[1], fm[2], rf"{ak}(.+?){akc}\s*{av}(.*?)(?:{avc}|$)"
  else: return None
  parts = []
  for pm in re.finditer(pair, body, re.DOTALL):
    value = re.sub(r"^\r?\n|\r?\n\Z", "", unmark(pm[2]))
    try: text = json.dumps(json.loads(value), ensure_ascii=False)
    except json.JSONDecodeError: text = value if pm.end() == len(body) and value[:1] in '{["0123456789-tfn' else json.dumps(value, ensure_ascii=False)
    parts.append(f"{json.dumps(unmark(pm[1]), ensure_ascii=False)}: {text}")
  return unmark(name), "{" + ", ".join(parts)

def api_keys() -> list[str]:
  """API_KEY (comma separated) and the keys in the file API_KEY_FILE (one a line, or comma separated; read at every request: a key is changed
  without a restart, and is not in the server's argv or environment listing): the keys a request to /v1/ must carry (Authorization: Bearer
  <key>, or x-api-key); none set: no check"""
  text = os.getenv("API_KEY", "")
  if kf:=os.getenv("API_KEY_FILE"): text += "," + pathlib.Path(kf).read_text().replace("\n", ",")
  return [k.strip() for k in text.split(",") if k.strip()]

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def end_headers(self):
    # (every response carries the request's id, X-Request-Id, as the reference API's: a client quotes it in a report)
    self.send_header("X-Request-Id", getattr(self, "request_id", "") or uuid.uuid4().hex)
    super().end_headers()
  def api_error(self, status:int, message:str, code:str|None, param:str|None=None):
    err = {"message": message, "type": "invalid_request_error", "code": code, **({"param": param} if param else {})}
    return self.send_data(json.dumps({"error": err}).encode(), status_code=status)
  def authorized(self) -> bool:
    """a request to /v1/ (but the session end, its own token) with one of the API keys, when the server has any (API_KEY)"""
    if not (keys:=api_keys()) or not self.path.startswith("/v1/") or self.path.startswith("/v1/sessions/"): return True
    auth = self.headers.get("Authorization", "")
    got = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else (self.headers.get("x-api-key") or "").strip()
    return got in keys
  def model_ids(self) -> list[str]:
    """the model's names a request may ask for: the server's and MODEL_ALIASES (comma separated, e.g. the reference API's zai-org/GLM-5.3)"""
    return [self.server.model_name] + [a.strip() for a in os.getenv("MODEL_ALIASES", "").split(",") if a.strip()]
  def known_model(self, name:typing.Any) -> bool:
    """MODEL_CHECK=1 (default off: our harnesses send any name): only the model's names (any case) or none"""
    if os.getenv("MODEL_CHECK", "0") in ("", "0"): return True
    return not name or (isinstance(name, str) and name.lower() in {m.lower() for m in self.model_ids()})
  def stream_json(self, source:typing.Generator):
    """the chunks as server-sent events, a keepalive ping as an SSE comment (': keepalive': a client with a read timeout stays)"""
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      for r in source:
        data = json.dumps({k:v for k,v in r.items() if not k.startswith("_")})
        self.wfile.write(b": keepalive\n\n" if r.get("ping") else f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()
      self.wfile.write(b"data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError): source.close()
  def do_GET(self):
    self.request_id = uuid.uuid4().hex
    if not self.authorized(): return self.api_error(401, "Invalid API key provided.", "invalid_api_key")
    if self.path == "/v1/models":
      owner = os.getenv("MODEL_OWNED_BY", "tinygrad")
      self.send_data(json.dumps({"object":"list", "data":[{"id":m, "object":"model", "created":0, "owned_by":owner} for m in self.model_ids()]})
                     .encode())
    elif self.path.split("?")[0] == "/status": self.send_data(json.dumps(self.server.status(gpus="gpus" in self.path)).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, log:str="", keepalive:float|None=None, top_p:float|None=None, continuous_usage:bool=False,
                top_k:int|None=None, stop:list[str]|None=None, json_mode:bool=False, pictures:dict|None=None):
    tok = self.server.tok
    prompt_tokens = len(ids)
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    # (cached_tokens: the prompt's tokens that were in the caches, not prefilled)
    # (reasoning_tokens: the reply's tokens of its reasoning, as the reference API's usage (and OpenAI's completion_tokens_details))
    def usage() -> dict: return {"prompt_tokens": prompt_tokens, "completion_tokens": len(out), "total_tokens": prompt_tokens + len(out),
                                 "prompt_tokens_details": {"cached_tokens": req.cache_start}, "reasoning_tokens": n_reasoning[0],
                                 "completion_tokens_details": {"reasoning_tokens": n_reasoning[0]}}
    n_reasoning = [0]
    # (continuous_usage: the usage so far in every chunk, stream_options.continuous_usage_stats as vLLM's)
    def chunk(d:dict, finish_reason:str|None=None, **extra) -> dict:
      return {"choices": [{"index":0, "delta":d, "finish_reason":finish_reason, **extra}], **tmpl,
              **({"usage": usage()} if continuous_usage else {})}
    # (stop: the reply ends at a stop string, matched_stop it; json_mode: the content held to its end, sent as a bare JSON document)
    stopf, matched, json_parts = StopFilter(stop or []), typing.cast(str|None, None), typing.cast(list[str], [])
    def emit(field:str, delta:str) -> typing.Iterator[dict]:
      if json_mode and field == "content": json_parts.append(delta)
      else: yield chunk({field:delta})
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
    if top_k and not self.server.sample_top_k:
      if not self.server._top_k_said: stderr_log(f"top_k {top_k} ignored: the server runs without SAMPLE_TOP_K (or its model has no top_k)\n")
      self.server._top_k_said, top_k = True, 0
    req = Request(ids, log, max_tokens, temperature, self.server.top_p if top_p is None else top_p, session_id=sid, bench_tag=bench,
                  top_k=self.server.top_k if top_k is None else top_k)
    # (VISION: the request holds its pictures until its prompt is prefilled, nothing else does: their pixels freed then)
    if pictures: req.pictures, req.pics = {k: functools.partial(self.server.picture_rows, p) for k, p in pictures.items()}, dict(pictures)
    pictures = None
    # (SUPERSEDE: the same prompt again from the same session (X-Session-ID) while its request runs is a client's retry of it (a client that
    # timed out behind a proxy that doesn't close our connection): the older one stops, it isn't decoded for nobody)
    key = (sid, len(ids), hash(tuple(ids))) if isinstance(sid, str) and sid else None
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
        text, matched = stopf.feed(sentinels[next_id] if next_id in sentinels else dec(next_id))
        routed = list(router.route(text, final=matched is not None))
        if any(f == "reasoning_content" for f, _ in routed) or (not routed and router.mode == "reasoning"): n_reasoning[0] += 1
        for field, delta in routed:
          yield from emit(field, delta)
          last = time.perf_counter()
        if matched is not None:
          req.cancelled = True  # (the model stops at its next token)
          break
        if keepalive is not None and time.perf_counter() - last > keepalive:
          yield {"ping": True}
          last = time.perf_counter()
      if req.error is not None or req.superseded:
        msg = f"{type(req.error).__name__}: {req.error}" if req.error is not None else "superseded by the same request again"
        yield {"error":{"message":msg, "type":"server_error", "code":"generation_failed"}}
        return
      finish_reason = "stop" if matched is not None else req.finish_reason
      if matched is None:
        for field, delta in router.route(stopf.feed(dec(), final=True)[0], final=True): yield from emit(field, delta)
      if json_parts: yield chunk({"content":strip_json_fence("".join(json_parts))})
      tool_calls: list[dict] = []
      for m in re.finditer(router.call, router.buf, re.DOTALL):
        # (a call the token cap cut: its name and arguments as far as they came, not its markup in the content)
        if finish_reason == "length" and not m.group(0).rstrip().endswith(tags["</tool_call>"]):
          if (cut := partial_tool_call(m.group(1), tags)) is not None:
            tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                               "function":{"name":cut[0], "arguments":cut[1]}})
          continue
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
      yield chunk({}, finish_reason, **({"matched_stop": matched} if matched is not None else {}))
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
      try: item = req.out.get(timeout=min(1.0, keepalive or 1.0))
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
    self.request_id = uuid.uuid4().hex
    log = f"{self.path}  {colored('--', 'BLACK')}  "
    # (VISION: a body larger than VISION_MAX_BODY_MB isn't kept: 413, read away in pieces of 1 MB (at most 256 MB: then the connection closes) so
    # the client gets the answer, not a broken pipe)
    n = int(self.headers.get("Content-Length", "0"))
    if getattr(self.server, "vision", None) is not None and n > getenv("VISION_MAX_BODY_MB", 32) * 2**20:
      left = min(n, 2**28)
      while left > 0 and (got:=len(self.rfile.read(min(2**20, left)))): left -= got
      self.close_connection = True
      return self.api_error(413, f"request body larger than {getenv('VISION_MAX_BODY_MB', 32)} MB", "request_too_large")
    raw_body = self.rfile.read(n)
    path = self.path.split("?")[0]
    # (the path-form session end needs no request body. The body-form alias accepts its session_id in JSON.)
    try:
      if not raw_body.strip() and path.startswith("/v1/sessions/"):
        body: dict[str, typing.Any] = {}
      else:
        body = json.loads(raw_body.decode("utf-8"))
      del raw_body
      if not isinstance(body, dict): raise ValueError("the body is not a JSON object")
    except Exception as e: return self.bad_request(f"invalid request body: {e}")
    if DEBUG >= 1: print(json.dumps(redacted(body), indent=2))
    if not self.authorized(): return self.api_error(401, "Invalid API key provided.", "invalid_api_key")
    if path in ("/v1/messages", "/v1/messages/count_tokens"): return self.anthropic(path, body, log, request_st)
    if path == "/v1/sessions/end": return self.session_end(body.get("session_id", ""))
    if path.startswith("/v1/sessions/") and path.endswith("/end"):
      from urllib.parse import unquote
      return self.session_end(unquote(path[len("/v1/sessions/"):-len("/end")]))
    if self.path == "/v1/chat/completions":
      # (a model the server doesn't serve: 404, as the reference API (MODEL_CHECK: its name or an alias, any case, or none: the server's))
      if not self.known_model(body.get("model")):
        return self.api_error(404, f"The model '{body.get('model')}' does not exist.", "model_not_found", "model")
      # render and tokenize (VISION: the pictures decoded first, their image pads then their rows' ids)
      from tinygrad.llm.qwen_vision import PictureError, PictureBusy, PictureNotCached
      pictures:list = []
      by_id:dict[int, typing.Any] = {}
      try:
        if getattr(self.server, "vision", None) is not None: pictures = extract_pictures(body["messages"])
        normalize_messages(body["messages"], images=bool(pictures))
        rendered = self.render(body["messages"], body.get("tools"), body)
      except PictureBusy as e: return self.api_error(503, str(e), "server_busy")
      except PictureNotCached as e: return self.api_error(400, str(e), "picture_not_cached")
      except PictureError as e: return self.bad_request(str(e))
      except Exception as e: return self.bad_request(f"{type(e).__name__}: {e}")
      ids: list[int] = self.server.reuse_tokens(rendered, self.server.tok.encode(rendered))
      if pictures:
        try: ids, by_id = expand_pictures(ids, pictures, self.server.model.image_pad_id)
        except PictureError as e: return self.bad_request(str(e))
        self.server.pictures_served = True  # (reuse_tokens: kept lists may hold pictures from now on)
      log += f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  "
      if len(ids) >= self.server.model.max_context:
        stderr_log(log + f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      try: (max_tokens, temperature, top_p), top_k, stops = sampling_params(body, self.server.temperature), sampling_top_k(body), \
        stop_strings(body)
      except ValueError as e: return self.bad_request(str(e))
      json_mode = isinstance(rf:=body.get("response_format"), dict) and rf.get("type") in ("json_object", "json_schema")
      hold = 0.0 if body.get("stream") else HOLD_JSON
      include_usage = not body.get("stream") or (body.get("stream_options") or {}).get("include_usage", False)
      chunks = self.run_model(ids, str(body.get("model") or self.server.model_name), include_usage,
                              continuous_usage=include_usage and bool((body.get("stream_options") or {}).get("continuous_usage_stats", False)),
                              max_tokens=max_tokens, temperature=temperature, reasoning=rendered.rstrip().endswith(THINK_STARTS), log=log,
                              top_p=top_p, top_k=top_k, keepalive=hold or (KEEPALIVE if body.get("stream") else None), stop=stops,
                              json_mode=json_mode, pictures=by_id)
      del pictures, by_id
      if body.get("stream"): self.stream_json(chunks)
      else:
        held = False
        def reply(data:bytes, status_code:int=200):
          if held: self.wfile.write(data)
          else: self.send_data(data, status_code=status_code)
        out, reasoning, tool_calls, finish_reason, last, extra = [], [], [], "stop", time.perf_counter(), {}
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
            tool_calls += delta.get("tool_calls", [])
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
          if "matched_stop" in choice: extra["matched_stop"] = choice["matched_stop"]
        # (no content: "", as the reference API's (not null))
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out)}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        choice = {"index":0, "message":message, "finish_reason":finish_reason, **extra}
        reply(json.dumps({**c, "object":"chat.completion", "choices":[choice]}).encode())
    else:
      self.send_data(json.dumps({"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error", "code": "not_found"}}).encode(),
                     status_code=404)

  def render(self, messages:list[dict], tools:list[dict]|None, body:dict) -> str:
    # (a reasoning effort asked for is the template's, else its default)
    effort = body.get("reasoning_effort") or (body.get("output_config") or {}).get("effort")
    # (the template's own arguments a request asks for, e.g. enable_thinking: chat_template_kwargs as vLLM's)
    extra = {k: v for k, v in (body.get("chat_template_kwargs") or {}).items() if k not in ("messages", "tools", "add_generation_prompt")}
    kw = {"tools": tools, "preserve_thinking": True, **({"reasoning_effort": effort} if effort else {}), **extra}
    text = self.server.render_cache.render(messages, self.server.template, **kw)
    # (JSON can have lone surrogates ("\ud800"), not text the tokenizer can encode: a replacement character)
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")

  def anthropic(self, path:str, body:dict, log:str, request_st:float):
    """the Anthropic Messages API: the request as chat completions' messages, the reply's chunks as its content blocks (and stream events)"""
    try:
      msgs, tools = anthropic_messages(body)
      normalize_messages(msgs)
      rendered = self.render(msgs, tools, body)
      # (the API's default temperature is 1)
      (max_tokens, temperature, top_p), top_k = sampling_params(body, 1.0), sampling_top_k(body)
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
                            top_p=top_p, top_k=top_k)
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
               session_id:str="", bench_tag:str="", top_k:int=0):
    self.ids, self.max_tokens, self.temperature, self.top_p, self.rid = ids, max_tokens, temperature, top_p, next(_rids)
    self.top_k = top_k
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
    self.done = False
    # (sequences) the tokens of its prompt in the caches or the stash, and the engine's version then; the version its take last failed at, when
    # (was_hit: it continued a kept conversation when the scheduler first looked: an agent's turn, even if that's lost since)
    self.hit, self.hit_v, self.tried, self.tried_at, self.was_hit = 0, -1, -1, 0.0, False
    self.cold = False  # (sequences) its prefill is of more than COLD_TOKENS: a cold one, see schedule_seqs
    self.pictures: dict[int, typing.Callable] = {}  # (VISION: its pictures' rows by their first row's id, see Transformer.seq_prefill)
    self.pics: dict[int, typing.Any] = {}  # (VISION: its pictures (qwen_vision.Picture) by their first row's id: the tower's slices, see tower_slice)
  @property
  def sampling(self) -> tuple[float, ...]:
    """its temperature, top_p and (if it has one) top_k: the model's row of them (the seq steps take a top_k, the batch steps don't)"""
    return (self.temperature, self.top_p, float(self.top_k)) if self.top_k > 0 else (self.temperature, self.top_p)

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
  def ckpt_tail(self) -> str:
    """
    the text of the generation prompt after the role header an assistant turn renders with (Qwen 3.x: '<think>\\n' of
    '<|im_start|>assistant\\n<think>\\n': an old turn without its reasoning renders '<|im_start|>assistant\\n' + its text)
    """
    # (the turn as it renders with a user message after it, its reasoning not kept (Qwen 3.x preserve_thinking, GLM 5.x clear_thinking))
    ms:list[dict] = [{"role": "user", "content": "u"}]
    kw = {"preserve_thinking": False, "clear_thinking": True}
    gen = self.template.render(messages=ms, add_generation_prompt=True, **kw)
    start = self.template.render(messages=ms, add_generation_prompt=False, **kw)
    turn = self.template.render(messages=ms + [{"role": "assistant", "content": "\x00"}, {"role": "user", "content": "v"}],
                                add_generation_prompt=False, **kw)
    if not (gen.startswith(start) and turn.startswith(start)): return ""
    gen, turn = gen[len(start):], turn[len(start):]
    n = next((i for i in range(min(len(gen), len(turn))) if gen[i] != turn[i]), min(len(gen), len(turn)))
    return gen[n:]

  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, top_p:float=1.0,
               temperature:float=0.0, vision:typing.Any=None):
    # top_p: a request's without one (the model's recommended, general.sampling.top_p of its gguf); temperature: a chat completion's
    # (the model: master's Transformer, or a tensor parallel implementation of its own with more (the sequences of qwen4exp, glm5next, glm_dsa))
    self.model: typing.Any = model
    self.model_name, self.tok, self.template, self.top_p = model_name, tok, template, top_p
    self.temperature = temperature
    self.render_cache = RenderCache(template, check=getenv("SERVE_RENDER_CHECK", 64))
    self.render_cache.on = bool(getenv("SERVE_RENDER_CACHE", 0))
    # (VISION: the vision tower (qwen_vision.VisionTower), the rows of the last pictures it made (VISION_CACHE_MB))
    self.vision, self._pic_rows = vision, collections.OrderedDict[bytes, typing.Any]()
    self._tower_job: tuple[typing.Any, typing.Any]|None = None  # (VISION: the picture whose tower runs in slices, and its TowerJob)
    self.pictures_served = False  # (VISION: a request with pictures was served: kept lists may hold pictures, see reuse_tokens)
    self.top_k = max(getenv("TOP_K", 0), 0)  # a request's without one (TOP_K: the config's, e.g. Qwen's 20; 0: none)
    # (the models' graphs take a top_k with SAMPLE_TOP_K (on with TOP_K): without it a request's top_k is ignored, said once)
    self.sample_top_k, self._top_k_said = bool(getenv("SAMPLE_TOP_K", int(self.top_k > 0))), False
    # (a model without a top_k in its sampler (glm_dsa): every top_k ignored, said once)
    if self.sample_top_k and not getattr(model, "SAMPLE_TOP_K", True): self.sample_top_k = False
    if self.top_k and not self.sample_top_k: stderr_log(f"TOP_K {self.top_k} ignored: this model's sampler has no top_k\n")
    if not self.sample_top_k: self.top_k = 0
    self.waiting: queue.Queue[Request] = queue.Queue()
    self.inflight = 0  # the requests taken by the model (decoded, or waiting for it): the status line's
    self.telemetry = Telemetry(getenv("TELEMETRY_LOG", ""))
    self.stats = BatchStats()
    self.engine: typing.Any = None  # (schedule_seqs': the sequences)
    self.end_requests: queue.Queue[tuple[str, SessionEndResponse]] = queue.Queue()
    self.ended_sessions: set[str] = set()
    self.ended_dropped = 0
    self.recent = collections.deque(maxlen=1000)
    specials = sp if isinstance(sp:=getattr(tok, "_special_tokens", None), dict) else {}
    self.odd_ids = {i for t, i in specials.items() if t in ODD_TOKENS}
    # (SUPERSEDE: the requests running of every (session, prompt), see Handler.run_model)
    self.inflight_keys:dict[tuple, Request] = {}
    self.keys_lock = threading.Lock()
    super().__init__(server_address, Handler)

  def serve_forever(self, poll_interval:float=0.5):
    # (the model's thread polls the devices holding the GIL: the request threads (many streams) take it in turns of SWITCH_INTERVAL, the
    # default 5 ms delays its steps)
    sys.setswitchinterval(0.0005)
    # the model runs here, not on a thread of its own: BEAM's _try_compile installs a SIGALRM handler, which raises ValueError off the main thread
    threading.Thread(target=super().serve_forever, args=(poll_interval,), daemon=True, name="HTTPServer").start()
    # sequences sharing the caches (qwen4exp, glm5next, glm_dsa), else one at a time
    if hasattr(type(self.model), "seqs_supported") and self.model.seqs_supported(): self.schedule_seqs()
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
  def kept(self) -> list[tuple[list[int], int]]:
    """
    the kept conversations' tokens and their lengths (and their checkpoints': the same tokens, the checkpoint's length. not a copy of every
    conversation's prefix a request): the model's, the sequences' and the stashed ones'
    """
    ret = [(list(c), len(c))] if isinstance(c:=getattr(self.model, "_cached_tokens", None), list) else []
    if (eng:=getattr(self, "engine", None)) is not None:
      ret += [p for s in list(eng.seqs) for p in ((s.tokens, len(s.tokens)), (s.tokens, s.ckpt))]
      ret += [p for x in list(eng.stashed) for p in ((x.tokens, len(x.tokens)), (x.tokens, x.ckpt))]
    return ret
  def tower_slice(self, req:Request) -> float|None:
    """
    VISION: one slice of the tower (seconds) for the pictures the next prefill of req reads whose rows aren't kept: the job in progress
    (whoever's), else the first such picture's. None: none left (its prefill reads kept rows, the tower doesn't run inside it).
    (the pictures ahead of the prefill are made up to half of VISION_CACHE_MB, at least one, and kept at the young end of the cache: a prompt
    with more pictures than the cache holds gets the rest as its prefill moves on. making them all first evicted the first ones and made them
    again: 1,100 screenshots of 2.2 MB in 2,048 MB never prefilled)
    """
    if self._tower_job is None:
      from tinygrad.llm.qwen_vision import picture_runs
      ids, pos, _ = req.fill
      budget, ahead, pic = getenv("VISION_CACHE_MB", 512) * 2**20 // 2, 0, None
      row = (r.nbytes // max(1, r.shape[0])) if (r:=next(iter(self._pic_rows.values()), None)) is not None else 0
      for start, n, _, first in (picture_runs(ids) if req.pics else []):
        if start + n <= pos or (p:=req.pics.get(first)) is None: continue
        if ahead and (ahead + n) * row > budget: break
        ahead += n
        if p.digest not in self._pic_rows:
          pic = p
          break
        self._pic_rows.move_to_end(p.digest)
      if pic is None: return None
      self._tower_job = (pic, self.vision.start(*self.vision.patches(pic.rgb)))
    st = time.perf_counter()
    pic, job = self._tower_job
    if (rows:=job.step()) is not None:
      self._tower_job = None
      self._keep_rows(pic.digest, rows)
    return time.perf_counter() - st

  def _keep_rows(self, digest:bytes, rows:typing.Any):
    self._pic_rows[digest] = rows
    while sum(x.nbytes for x in self._pic_rows.values()) > getenv("VISION_CACHE_MB", 512) * 2**20 and len(self._pic_rows) > 1:
      self._pic_rows.popitem(last=False)

  def picture_rows(self, pic:typing.Any) -> typing.Any:
    """VISION: a picture's rows (n, dim) float32 by the tower (the model's thread, between its prefill chunks); the last ones' kept"""
    if (r:=self._pic_rows.get(pic.digest)) is not None:
      self._pic_rows.move_to_end(pic.digest)
      return r
    # (not kept (tower_slice made them first, an LRU dropped them since): the whole tower now, in the prefill. a sliced job in progress (another
    # picture's, maybe of the same bucket: its inputs and carries) is finished first and its rows kept)
    if self._tower_job is not None:
      jp, job = self._tower_job
      while (rows:=job.step()) is None: pass
      self._tower_job = None
      self._keep_rows(jp.digest, rows)
      if (r:=self._pic_rows.get(pic.digest)) is not None: return r
    self._keep_rows(pic.digest, r:=self.vision(*self.vision.patches(pic.rgb)))
    return r

  def reuse_tokens(self, rendered:str, ids:list[int]) -> list[int]:
    """
    the tokens of a prompt that continues a kept conversation as its text but not its tokens (the model's reply tokenized again differently):
    the kept ones and the rest's, so the cache holds it
    """
    # (VISION: a request with a picture, or one whose candidates hold a kept list with pictures that starts as it does (only if a picture was
    # ever served: a text fleet scans nothing; another conversation's pictures don't send every text request the slower way: g2 Nov 14, 0.1 s
    # a request): _reuse_pictures; every other one (a text-only request on a VISION server) the path of before, the same code)
    if getattr(self, "vision", None) is not None and \
       (getattr(getattr(self, "model", None), "image_pad_id", -1) in ids or (getattr(self, "pictures_served", False) and
        any(same_start(t, ids) and min(t, default=0) < 0 for t, n in self.kept() if 1000 <= n < len(ids) + 1000 and n <= len(t)))):
      return self._reuse_pictures(rendered, ids)
    return self._reuse_text(rendered, ids)

  def _reuse_text(self, rendered:str, ids:list[int]) -> list[int]:
    """reuse_tokens of a request without pictures (and no kept list with pictures among its candidates): see reuse_tokens"""
    cands = [(t, n) for t, n in self.kept() if 1000 <= n < len(ids) + 1000 and n <= len(t)]
    # (one the prompt's tokens continue: the cache has it. a longer one the prompt's text continues: the ones sharing the most tokens with it
    # (an agent's own conversation, not the others' with the same system prompt), however early they differ: a sampled token of an early
    # reply that isn't its text's tokenization differs from then on, every turn after)
    exact = max((n for t, n in cands if starts(ids, t, n)), default=0)
    # (an agent's own ones: the others' differ in their task, at the start)
    own = [t[:n] for t, n in cands if n > exact and t[:(m:=min(2048, n // 2, len(ids) // 2))] == ids[:m]]
    for t in sorted(own, key=lambda t: common_prefix(t, ids), reverse=True)[:2]:
      # (VISION: not one with pictures (their rows' ids aren't text): its prompt is the tokens of its text and pictures as they are)
      if getattr(self, "vision", None) is not None and min(t, default=0) < 0: continue
      if rendered.startswith(text:=self.tok.decode(t)): return t + self.tok.encode(rendered[len(text):])
    return ids

  def _reuse_pictures(self, rendered:str, ids:list[int]) -> list[int]:
    """
    VISION: reuse_tokens with the kept conversations that have pictures, compared as the template's text has them (a picture one image pad:
    ids are the request's tokens before its pads are expanded; a kept list returned so is expanded again to the same rows' ids by the same
    pictures). a short one too (a two-turn picture chat). the kept lists compared in slices between their pictures, collapsed only to return.
    """
    pad, info, walks, exact = getattr(getattr(self, "model", None), "image_pad_id", -1), [], {}, 0
    if not ids: return ids
    pads = positions(ids, pad)
    for t, n in self.kept():
      if not 0 < n <= len(t) or not (t[0] == ids[0] or (t[0] < 0 and ids[0] == pad)): continue
      if id(t) not in walks: walks[id(t)] = common_prefix_pictures(t, ids, pad, pads)
      i, j = walks[id(t)]
      # (one that goes on past the whole prompt (i: all of ids shared) is longer than its text: never its prefix. Not decoded (g5 Nov 14: a
      # finished run of the same game kept, every request of the next run decoded its 260K tokens to fail: +0.08 s prep a step))
      if i == len(ids) and j < n: continue
      # (text ones as reuse_tokens takes them: 1,000 tokens at least and not 1,000 more than the prompt)
      if min(t[:max(j, 1)]) >= 0 and not 1000 <= n < len(ids) + 1000: continue
      if j >= n: exact = max(exact, n)
      info.append((t, n, i, j))
    # (n - (j - i): its length as text, the rows of its pictures in the shared part one pad each)
    own = [(t, n, i, j) for t, n, i, j in info if n > exact and i >= min(2048, (n - (j - i)) // 2, len(ids) // 2)]
    for t, n, i, j in sorted(own, key=lambda x: x[2], reverse=True)[:2]:
      if (ret:=self._reuse_rest(rendered, ids, i, collapse_pictures(t[j:n], pad))) is False: continue
      if ret is not None: return ret
      if rendered.startswith(text:=self.tok.decode(c:=collapse_pictures(t[:n], pad))): return c + self.tok.encode(rendered[len(text):])
    return ids

  def _reuse_rest(self, rendered:str, ids:list[int], i:int, rest:list[int]) -> list[int]|typing.Literal[False]|None:
    """
    _reuse_pictures' check of a kept list without decoding its shared part (a Pokemon step at 230K: two decodes, ~30-80 ms of the prep, and
    with the Qwen line's preserve_thinking=False never a match): ids[:i] are its first tokens as the text has them (common_prefix_pictures),
    ids the text's tokens (their bytes its bytes: the encoder splits every character into words, every word into byte tokens), so only rest
    (its tokens after them, collapsed) is compared, as bytes, where ids[i:] begin: its tokens (the full check's), False (the full check's no:
    a text without a replacement character continues a decoded text only if it continues its bytes), None (the full check decides)
    """
    if (tb:=getattr(self.tok, "_tok2bytes", None)) is None: return None
    try: rb = rendered.encode()
    except UnicodeEncodeError: return None
    at = len(rb) - len(tail:=b"".join(tb[x] for x in ids[i:]))
    if at < 0 or not rb.endswith(tail) or b"\xef\xbf\xbd" in rb: return None
    end = at + len(b:=b"".join(tb[x] for x in rest))
    # (the kept rest follows the shared part and ends on a character's start)
    if rb[at:end] != b or (end < len(rb) and rb[end] & 0xC0 == 0x80): return False
    return ids[:i] + rest + self.tok.encode(rb[end:].decode())

  def miss_log(self, ids:list[int], start:int):
    """a request prefilled from 2000 tokens less than it shares with a kept conversation at least: where they differ"""
    kept = [t for t, n in self.kept() if n == len(t) >= start + 2000 <= len(ids) and t[start+1999] == ids[start+1999]]
    n, t = max(((common_prefix(t, ids), t) for t in kept if same_start(t, ids)), key=lambda x: x[0], default=(0, []))
    if n - start < 2000: return
    def dec(t:list[int]) -> str: return self.tok.decode([x for x in t if x >= 0])  # (VISION: a picture's rows aren't text)
    stderr_log(colored(f"cache miss: prefilled from {start}, shares {n} of {len(t)} kept tokens: kept {dec(t[max(0,n-30):n+30])!r} "
                       f"vs {dec(ids[max(0,n-30):n+30])!r}\n", "yellow"))

  # requests waiting this long are taken in turn, before the others the cache has more of

  def seq_end(self, eng:"SeqEngine", s:"Seq", req:Request, kept:int):
    """the end of req's decoding in sequence s: kept for a next turn, unless it failed (emit's guard: its states went wrong, the retry is
    prefilled from the start, not continued from them)"""
    if req.error is not None: eng.drop(s)
    else: eng.stop(s, kept)
  def drop_caches(self):
    """the kept conversations forgotten (their states may be wrong): every request is prefilled from the start"""
    if (eng:=getattr(self, "engine", None)) is not None: eng.forget()
    if hasattr(self.model, "_cached_tokens"): self.model._cached_tokens = []
    if hasattr(self.model, "_ckpt_tokens"): self.model._ckpt_tokens = []

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

  # ** sequences sharing the model's caches (Transformer.seqs_supported): continuous batching
  # (a prompt is prefilled in slices of PREFILL_SLICE tokens (~1 s). the decoding ones' steps run after MAX_STALL s of slices (0.3: 1.0 had
  # replies wait 1.5/2.3 s p50/p90 for a token under agent load, 0.3: 0.8/1.5 s, and more tokens), for their share
  # of the time (the requests decoding of the ones in flight, DECODE_SHARE at most and 0.1 at least): a long prompt doesn't stop the replies,
  # many prompts at once are prefilled mostly one after another (a step of many is about as fast as one of a few). the next slice is the
  # prompt's with the fewest tokens left to prefill (a short one, an agent's turn, isn't behind a long one), less AGING tokens a second it
  # waited. a long one isn't behind short ones that keep coming forever: one waiting past STARVE_SEQ goes first, and takes a sequence even if
  # a held conversation is lost for it (see SeqEngine.STICKY))
  # (PREFILL_SLICE, AGING, SEQ_STARVE and SEQ_STARVE_ALT are live profile overrides: C5 uses 1536 and STARVE_ALT 1.)
  PREFILL_SLICE, MAX_STALL, DECODE_SHARE = getenv("PREFILL_SLICE", 13824), 0.3, 0.5
  AGING, STARVE_SEQ, TURN_WAIT = getenv("AGING", 0), getenv("SEQ_STARVE", 600.0), 10.0
  # (SEQ_STARVE_ALT: after a slice of a starving prompt the next one is the best of the others (a waiting one admitted, a warm turn's short
  # suffix): a long cold prompt past STARVE takes every other slice, not all of them while every other request waits behind it)
  STARVE_ALT = getenv("SEQ_STARVE_ALT", 0)
  COLD_MAX, COLD_TOKENS = 1, 16384
  # (the gpus' clocks (AM: their gfx clock's cap, see AMIdleClocks.set_caps): a few users decoding at the full clocks (a user's tokens fast),
  # more at less (the energy a token: ~0.85 J at 2000 MHz on 6x 7900 XTX, ~1.2 J at full), a prefill at a cap it isn't slower at. switched
  # CLOCK_HOLD s apart at most)
  # (CLOCK_FAST, CLOCK_MANY, CLOCK_PREFILL: a mode's caps from the env (e.g. "GFXCLK=1900"; "" none), a box's thermals without a code change)
  CLOCKS = {"fast": getenv("CLOCK_FAST", ""), "many": getenv("CLOCK_MANY", "GFXCLK=2000"), "prefill": getenv("CLOCK_PREFILL", "GFXCLK=2300")}
  FAST_USERS, CLOCK_HOLD = 4, 0.5
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
    # (SEQ_CKPT_TAIL=1: a prompt's checkpoint kept before the tail of the generation prompt, see SeqEngine.TAIL and ckpt_tail)
    if getenv("SEQ_CKPT_TAIL", 0): eng.TAIL = self.tok.encode(self.ckpt_tail())
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
          if active and 0 < len(ids) - start <= mix_tokens and min(ids[start:], default=0) >= 0:
            # (SEQ_MIX_TOKENS: force this short suffix in the draft slots of the ordinary steps, beside the other sequences)
            eng.mix_start(job, ids, start, restore, c.sampling)
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
        # (VISION: the rows of the pictures a prefill will read are made before it, one slice of the tower (qwen_vision.TowerJob) a turn of
        # this loop instead of a prefill slice: the others' decoding steps between them as between prefill slices)
        if job is not None and getattr(self, "vision", None) is not None and filling[job].pics and (tsl:=self.tower_slice(filling[job])) is not None:
          stall += tsl
          trace(f"tower_slice ms={tsl*1e3:.3f} active={len(active)}")
          job = None
        if job is not None:
          req = filling[job]
          ids, pos, restore = req.fill
          # (a model's slices in whole prefill chunks (GLM: its greedy tokens generate's, see Transformer.seq_slice))
          sl = f(self.PREFILL_SLICE) if callable(f:=getattr(self.model, "seq_slice", None)) else self.PREFILL_SLICE
          end = len(ids) if len(ids) - pos <= sl * 3 // 2 else pos + sl
          st = time.perf_counter()
          # (PREFILL_BIG_ALONE: the biggest prefill chunks only while nobody decodes (=2: and no other request waits or prefills), see _seq_chunks)
          self.model._pf_alone = not active and (getenv("PREFILL_BIG_ALONE", 0) != 2 or (not pending and len(filling) <= 1))
          try:
            if req.pictures: (eng.prefill_slice(job, ids, pos, restore, end, pictures=req.pictures) if end < len(ids) else
                              (first:=eng.prefill(job, ids, pos, restore, req.sampling, pictures=req.pictures)))
            elif end < len(ids): eng.prefill_slice(job, ids, pos, restore, end)
            else: first = eng.prefill(job, ids, pos, restore, req.sampling)
            if end == len(ids): req.pictures, req.pics = {}, {}
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
            else: self.seq_end(eng, job, req, len(job.tokens))
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
                self.seq_end(eng, s, req, len(s.tokens) - (len(toks) - 1 - i))
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

