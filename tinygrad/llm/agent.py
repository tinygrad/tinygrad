import json, re, uuid

TOOL_CALL_OPEN = "\u16ec"
TOOL_CALL_CLOSE = "\u16ed"

def format_tools(tools: list|None, preset: str) -> str:
  if not tools: return ""
  if preset == 'qwen2':
    return "\n# Tools\n\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures in <tools></tools> XML tags:\n<tools>\n" + \
      "\n".join(json.dumps(t.get("function", t), ensure_ascii=False) for t in tools) + \
      "\n</tools>\n\nFor each function call, return a json object with function name and arguments within " + TOOL_CALL_OPEN + " " + TOOL_CALL_CLOSE + " XML tags:\n" + TOOL_CALL_OPEN + '{"name": <function-name>, "arguments": <args-json>}' + TOOL_CALL_CLOSE + "\n"
  return ""

def format_tool_call(tc: dict, preset: str) -> str:
  if preset == 'qwen2':
    args = tc["function"]["arguments"]
    if isinstance(args, str):
      try: args = json.loads(args)
      except json.JSONDecodeError: pass
    return TOOL_CALL_OPEN + json.dumps({"name": tc["function"]["name"], "arguments": args}, ensure_ascii=False) + TOOL_CALL_CLOSE
  return json.dumps(tc)

def format_tool_result(content: str, preset: str) -> str:
  if preset == 'qwen2':
    return "\n" + content + "\n"
  return content

class StreamingToolParser:
  def __init__(self):
    self.hold = ""
    self.in_tag = False
    self._tool_buf = ""
    self.tool_calls: list[dict] = []

  def _parse_tool_buf(self):
    try:
      obj = json.loads(self._tool_buf.strip())
      if isinstance(obj, dict) and 'name' in obj:
        args = obj.get('arguments', {})
        if isinstance(args, str): args = json.loads(args)
        self.tool_calls.append({'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}})
    except (json.JSONDecodeError, TypeError): pass
    self._tool_buf = ""

  def process(self, chunk: str) -> str:
    self.hold += chunk
    content = ""
    while True:
      if self.in_tag:
        i = self.hold.find(TOOL_CALL_CLOSE)
        if i == -1:
          hold_back = len(TOOL_CALL_CLOSE) - 1
          if len(self.hold) > hold_back: self._tool_buf += self.hold[:-hold_back]; self.hold = self.hold[-hold_back:]
          break
        self._tool_buf += self.hold[:i]
        self._parse_tool_buf()
        self.hold = self.hold[i + len(TOOL_CALL_CLOSE):]
        self.in_tag = False
      else:
        i_tool = self.hold.find(TOOL_CALL_OPEN)
        if i_tool != -1:
          content += self.hold[:i_tool]
          self.hold = self.hold[i_tool + len(TOOL_CALL_OPEN):]
          self.in_tag = True
        else:
          hold_back = len(TOOL_CALL_OPEN) - 1
          if len(self.hold) > hold_back: content += self.hold[:-hold_back]; self.hold = self.hold[-hold_back:]
          break
    return content

  def finalize(self) -> str:
    if self.in_tag:
      self._tool_buf += self.hold
      self._parse_tool_buf()
      result = ""
    else:
      result = self.hold
    self.hold, self.in_tag, self._tool_buf = "", False, ""
    return result

def parse_tool_calls(text: str) -> list[dict]:
  tool_calls = []
  for m in re.finditer(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL):
    try:
      obj = json.loads(m.group(1))
      if isinstance(obj, dict) and 'name' in obj:
        args = obj.get('arguments', {})
        if isinstance(args, str): args = json.loads(args)
        tool_calls.append({'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}})
    except (json.JSONDecodeError, TypeError): pass
  return tool_calls
