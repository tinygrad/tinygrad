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

class StreamingToolParser:
  def __init__(self): self.hold, self.in_tag = "", False
  def process(self, chunk: str) -> str:
    self.hold += chunk; content = ""
    while True:
      if self.in_tag:
        i = self.hold.find(TOOL_CALL_CLOSE)
        if i == -1: self.hold = self.hold[-(len(TOOL_CALL_CLOSE)-1):] if len(self.hold) >= len(TOOL_CALL_CLOSE) else self.hold; break
        self.hold, self.in_tag = self.hold[i+len(TOOL_CALL_CLOSE):], False
      else:
        i = self.hold.find(TOOL_CALL_OPEN)
        if i == -1: content += self.hold[:-(len(TOOL_CALL_OPEN)-1)] if len(self.hold) >= len(TOOL_CALL_OPEN) else ""; self.hold = self.hold[-(len(TOOL_CALL_OPEN)-1):] if len(self.hold) >= len(TOOL_CALL_OPEN) else self.hold; break
        content += self.hold[:i]; self.hold, self.in_tag = self.hold[i+len(TOOL_CALL_OPEN):], True
    return content
  def finalize(self) -> str:
    result = self.hold if not self.in_tag else ""
    self.hold, self.in_tag = "", False
    return result

def parse_tool_calls(text: str) -> tuple[list[dict], list[str], str]:
  thinking = [m.group(1) for m in re.finditer(r'<thinking>\s*(.*?)\s*</thinking>', text, re.DOTALL)]
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
  clean = re.sub(r'<thinking>\s*(.*?)\s*</thinking>', '', re.sub(rf'{TOOL_CALL_OPEN}\s*(.*?)\s*{TOOL_CALL_CLOSE}', '', text, flags=re.DOTALL), flags=re.DOTALL)
  return tool_calls, thinking, clean.strip()
