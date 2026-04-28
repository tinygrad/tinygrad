import json, re, uuid

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

# normalize model output into OpenAI-style tool_calls
def _tool_call(obj: str|dict) -> dict|None:
  try:
    if isinstance(obj, str): obj = json.loads(obj.strip())
    if isinstance(obj, dict) and 'name' in obj:
      args = obj.get('arguments', {})
      if isinstance(args, str): args = json.loads(args)
      return {'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}}
  except (json.JSONDecodeError, TypeError): pass
  return None

# prompt the model with full tool schemas, keeping the response format explicit
def format_tools(tools: list|None) -> str:
  if not tools: return ""
  return "<tools>\n" + "\n".join(json.dumps(t.get('function', t), ensure_ascii=False) for t in tools) + \
    "\n</tools>\nThe \"name\" field must exactly match one tool name listed in <tools>.\nReply only with: " + \
    TOOL_CALL_OPEN + '{"name":"...","arguments":{...}}' + TOOL_CALL_CLOSE

# parse the last tool_call block
def parse_tool_calls(text: str) -> list[dict]:
  matches = re.findall(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL)
  return [tc] if matches and (tc:=_tool_call(matches[-1])) is not None else []
