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

# prompt the model with compact signatures, but keep the response format explicit
def format_tools(tools: list|None) -> str:
  if not tools: return ""
  return ("<tools>\n" +
          "\n".join(_tool_sig(t) for t in tools) +
          "\n</tools>\nReply only: " + TOOL_CALL_OPEN + '{"name":"...","arguments":{...}}' + TOOL_CALL_CLOSE)

# parse tagged tool calls and fill missing string descriptions when needed
def parse_tool_calls(text: str, tools: list|None=None) -> list[dict]:
  out = [tc for m in re.finditer(rf'{re.escape(TOOL_CALL_OPEN)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE)}', text, re.DOTALL)
         if (tc:=_tool_call(m.group(1))) is not None]
  if not out and (i:=text.rfind(TOOL_CALL_OPEN)) != -1:
    out = [tc] if (tc:=_tool_call(text[i+len(TOOL_CALL_OPEN):])) is not None else []
  tool_map = {t.get("function", t).get("name"): t.get("function", t) for t in tools or []}
  for tc in out:
    fn = tc.get("function", {})
    params = tool_map.get(fn.get("name"), {}).get("parameters", {})
    if "description" not in params.get("properties", {}): continue
    try: args = json.loads(fn["arguments"])
    except (json.JSONDecodeError, KeyError, TypeError): continue
    if not isinstance(args.get("description"), str): args["description"] = ""
    fn["arguments"] = json.dumps(args)
  return out
