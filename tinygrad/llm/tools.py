import json, re, uuid

TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"

# compact tool signatures keep the prompt small
def _tool_sig(t):
  fn = t.get("function", t)
  params = fn.get("parameters", {})
  props, req = params.get("properties", {}), set(params.get("required", []))
  def arg_sig(k, v):
    if e:=v.get("enum"): typ = "|".join(map(json.dumps, e))
    elif isinstance((t:=v.get("type")), list): typ = "|".join(t)
    else: typ = v.get("type") or "any"
    return f"{k}{'' if k in req else '?'}:{typ}"
  args = ", ".join(arg_sig(k, v) for k,v in props.items())
  return f"{fn['name']}({args})"

def _tool_requires_description(name: str, tools: list|None) -> bool:
  for t in tools or []:
    fn = t.get("function", t)
    if fn.get("name") == name and "description" in fn.get("parameters", {}).get("required", []): return True
  return False

# normalize model output into OpenAI-style tool_calls
def _tool_call(obj: str|dict, tools: list|None=None) -> dict|None:
  try:
    if isinstance(obj, str): obj = json.loads(obj.strip())
    if isinstance(obj, dict) and 'name' in obj:
      args = obj.get('arguments', {})
      if isinstance(args, str): args = json.loads(args)
      if "description" not in args and _tool_requires_description(obj['name'], tools): args["description"] = ""
      return {'id': f'call_{uuid.uuid4().hex[:24]}', 'type': 'function', 'function': {'name': obj['name'], 'arguments': json.dumps(args)}}
  except (json.JSONDecodeError, TypeError): pass
  return None

# prompt the model with compact signatures, keeping the response format explicit
def format_tools(tools: list|None) -> str:
  if not tools: return ""
  tool_names = [t.get('function', t)['name'] for t in tools]
  prompt = "<tools>\n" + "\n".join(_tool_sig(t) for t in tools) + "\n</tools>\nTool names: " + ", ".join(tool_names) + \
    "Use exactly one listed tool name in \"name\".\nReply only with a complete tool call: "
  return prompt + TOOL_CALL_OPEN + '{"name":"...","arguments":{...}}' + TOOL_CALL_CLOSE

# parse the last tool_call block
def parse_tool_calls(text: str, tools: list|None=None) -> list[dict]:
  matches = re.findall(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL)
  return [tc] if matches and (tc:=_tool_call(matches[-1], tools)) is not None else []
