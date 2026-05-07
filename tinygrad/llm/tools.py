import json, re, uuid

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

# compact tool signatures keep the prompt small
def _tool_sig(fn: dict) -> str:
  params = fn.get("parameters", {})
  props, req_set = params.get("properties", {}), set(params.get("required", []))
  return f"{fn['name']}({','.join(k for k in props.keys() if k in req_set)})"

# normalize model output into OpenAI-style tool_calls
def _tool_call(text: str) -> dict|None:
  try:
    obj = json.loads(text.strip())
    args = obj.get('arguments', {})
    if isinstance(args, str): args = json.loads(args)
    return {'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}}
  except json.JSONDecodeError: pass
  return None

# prompt the model with compact signatures, keeping the response format explicit
def format_tools(tools: list|None) -> str:
  if not tools: return ""
  functions = [tool.get('function', {}) for tool in tools]
  return "<tools>\n" + "\n".join(_tool_sig(fn) for fn in functions) + "\n</tools>\n" \
    + "Use exactly one listed tool name in \"name\" and include all arguments in \"arguments\".\n" \
    + "Reply with a complete tool call: " + TOOL_CALL_OPEN + '{"name":"...","arguments":{..}}' + TOOL_CALL_CLOSE \
    + "\nExample bash call: " + TOOL_CALL_OPEN + '{"name":"bash","arguments":{"command":"ls","description":""}}' + TOOL_CALL_CLOSE

# parse the last tool_call block
def parse_tool_calls(text: str) -> list[dict]:
  matches = re.findall(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL)
  return [tc] if matches and (tc:=_tool_call(matches[-1])) is not None else []
