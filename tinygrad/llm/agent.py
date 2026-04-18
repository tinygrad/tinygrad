import json, re, uuid

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

# compact tool signatures keep the prompt small
def _tool_sig(t):
  fn = t.get("function", t)
  ps, req = fn.get("parameters", {}).get("properties", {}), set(fn.get("parameters", {}).get("required", []))
  args = ", ".join(f"{k}{'' if k in req else '?'}:{'|'.join(map(json.dumps, e)) if (e:=v.get('enum')) else '|'.join(t) if isinstance((t:=v.get('type')), list) else t or 'any'}" for k,v in ps.items())
  return f"{fn['name']}({args})"

# normalize model output into OpenAI-style tool_calls
def _tool_call(obj: str|dict) -> dict|None:
  try:
    if isinstance(obj, str): obj = json.loads(obj.strip())
    if isinstance(obj, dict) and 'name' in obj:
      args = obj.get('arguments', {})
      if isinstance(args, str): args = json.loads(args)
      return {'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}}
  except (json.JSONDecodeError, TypeError):
    pass
  return None

def format_tools(tools: list|None) -> str:
  if not tools: return ""
  return "<tools>\n" + "\n".join(_tool_sig(t) for t in tools) + "\n</tools>\nReply only: " + TOOL_CALL_OPEN + '{"name":"...","arguments":{...}}' + TOOL_CALL_CLOSE

# strip streamed tool_call blocks while collecting them
class StreamingToolParser:
  def __init__(self): self.hold, self.buf, self.in_tag, self.tool_calls = "", "", False, []
  def process(self, chunk: str) -> str:
    self.hold += chunk
    out = ""
    while True:
      if self.in_tag:
        if (i:=self.hold.find(TOOL_CALL_CLOSE)) == -1:
          if len(self.hold) >= len(TOOL_CALL_CLOSE): self.buf, self.hold = self.buf + self.hold[:1-len(TOOL_CALL_CLOSE)], self.hold[1-len(TOOL_CALL_CLOSE):]
          break
        if (tc:=_tool_call(self.buf + self.hold[:i])) is not None: self.tool_calls.append(tc)
        self.buf, self.hold, self.in_tag = "", self.hold[i+len(TOOL_CALL_CLOSE):], False
      else:
        if (i:=self.hold.find(TOOL_CALL_OPEN)) == -1:
          if len(self.hold) >= len(TOOL_CALL_OPEN): out, self.hold = out + self.hold[:1-len(TOOL_CALL_OPEN)], self.hold[1-len(TOOL_CALL_OPEN):]
          break
        out, self.hold, self.in_tag = out + self.hold[:i], self.hold[i+len(TOOL_CALL_OPEN):], True
    return out
  def finalize(self) -> str:
    if self.in_tag and (tc:=_tool_call(self.buf + self.hold)) is not None: self.tool_calls.append(tc)
    out = "" if self.in_tag else self.hold
    self.hold, self.buf, self.in_tag = "", "", False
    return out

# parse tagged tool calls, then fall back to bare JSON
def parse_tool_calls(text: str) -> list[dict]:
  out = [tc for m in re.finditer(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL) if (tc:=_tool_call(m.group(1))) is not None]
  if out: return out
  try: raw = json.loads(text.strip())
  except json.JSONDecodeError: return []
  return [tc for obj in (raw if isinstance(raw, list) else [raw]) if (tc:=_tool_call(obj)) is not None]
