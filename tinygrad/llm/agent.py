import json, re, uuid

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

def format_tools(tools: list|None, preset: str) -> str:
  if not tools: return ""
  if preset in ('qwen2', 'qwen3', 'qwen35', 'qwen35moe'):
    return "# Tools\n<tools>\n" + "\n".join(json.dumps(t) for t in tools) + "\n</tools>\nUse <tool_call>{\"name\": ..., \"arguments\": ...}</tool_call>"
  if preset in ('llama3', 'llama-v3', 'llama-bpe'):
    return "# Tools\n" + "".join(f"{t.get('function', {}).get('name', '')}: {t.get('function', {}).get('description', '')}\n" for t in tools) + "Call: {\"name\": ..., \"arguments\": {...}}"
  return ""

def format_tool_response(content: str, preset: str) -> str:
  if preset in ('qwen2', 'qwen3', 'qwen35', 'qwen35moe'): return f"<tool_response>\n{content}\n</tool_response>\n"
  if preset in ('llama3', 'llama-v3', 'llama-bpe'): return f"  \n\n{content}<|eot_id|>"
  return content

THINK_TAGS = ("think", "thinking")
THINK_OPENS = [f'<{tag}>' for tag in THINK_TAGS]
THINK_CLOSES = [f'</{tag}>' for tag in THINK_TAGS]

class StreamingToolParser:
  def __init__(self): self.hold, self.in_tag, self.in_think = "", False, False
  def process(self, chunk: str) -> str:
    self.hold += chunk
    content = ""
    while True:
      for tag in THINK_TAGS: self.hold = re.sub(r'<' + tag + r'>\s*.*?\s*</' + tag + r'>', '', self.hold, flags=re.DOTALL)
      if self.in_think:
        close_idx, close_tag = -1, None
        for ct in THINK_CLOSES:
          i = self.hold.find(ct)
          if i != -1 and (close_idx == -1 or i < close_idx): close_idx, close_tag = i, ct
        if close_idx != -1:
          self.hold = self.hold[close_idx + len(close_tag):]
          self.in_think = False
          continue
        hold_back = max(len(ct) for ct in THINK_CLOSES) - 1
        self.hold = self.hold[-hold_back:] if len(self.hold) > hold_back else self.hold
        break
      elif self.in_tag:
        i = self.hold.find(TOOL_CALL_CLOSE)
        if i == -1: self.hold = self.hold[-(len(TOOL_CALL_CLOSE)-1):] if len(self.hold) >= len(TOOL_CALL_CLOSE) else self.hold; break
        self.hold, self.in_tag = self.hold[i+len(TOOL_CALL_CLOSE):], False
      else:
        best_idx, best_tag, best_type = -1, None, None
        for ot in THINK_OPENS:
          i = self.hold.find(ot)
          if i != -1 and (best_idx == -1 or i < best_idx): best_idx, best_tag, best_type = i, ot, 'think'
        i_tool = self.hold.find(TOOL_CALL_OPEN)
        if i_tool != -1 and (best_idx == -1 or i_tool < best_idx): best_idx, best_tag, best_type = i_tool, TOOL_CALL_OPEN, 'tool'
        if best_idx != -1:
          content += self.hold[:best_idx]
          self.hold = self.hold[best_idx + len(best_tag):]
          if best_type == 'think': self.in_think = True
          else: self.in_tag = True
        else:
          hold_back = max(max(len(ot) for ot in THINK_OPENS), len(TOOL_CALL_OPEN)) - 1
          if len(self.hold) > hold_back: content += self.hold[:-hold_back]; self.hold = self.hold[-hold_back:]
          break
    for tag in THINK_TAGS: content = re.sub(r'<' + tag + r'>\s*.*?\s*</' + tag + r'>', '', content, flags=re.DOTALL)
    return content
  def finalize(self) -> str:
    for tag in THINK_TAGS: self.hold = re.sub(r'<' + tag + r'>\s*.*?\s*</' + tag + r'>', '', self.hold, flags=re.DOTALL)
    result = self.hold if not self.in_tag and not self.in_think else ""
    self.hold, self.in_tag, self.in_think = "", False, False
    return result

def parse_tool_calls(text: str) -> tuple[list[dict], list[str], str]:
  thinking = [m.group(1) for m in re.finditer(r'<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>', text, re.DOTALL)]
  tool_calls = []
  for m in re.finditer(rf'{TOOL_CALL_OPEN}\s*(.*?)\s*{TOOL_CALL_CLOSE}', text, re.DOTALL):
    try:
      obj = json.loads(m.group(1))
      if isinstance(obj, dict) and 'name' in obj:
        args = obj.get('arguments', {})
        if isinstance(args, str): args = json.loads(args)
        tool_calls.append({'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}})
    except (json.JSONDecodeError, TypeError): pass
  if not tool_calls:
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\}|\{\})[^{}]*\}', text):
      try:
        args = json.loads(m.group(2)) if m.group(2) != '{}' else {}
        tool_calls.append({'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': m.group(1), 'arguments': json.dumps(args)}})
      except json.JSONDecodeError: pass
  clean = re.sub(r'<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>', '', re.sub(rf'{TOOL_CALL_OPEN}\s*(.*?)\s*{TOOL_CALL_CLOSE}', '', text, flags=re.DOTALL), flags=re.DOTALL)
  return tool_calls, thinking, clean.strip()
