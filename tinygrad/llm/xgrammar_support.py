from __future__ import annotations
import json, re, uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
  import xgrammar as xgr
  from xgrammar.structural_tag import AnyTextFormat, JSONSchemaFormat, SequenceFormat, StructuralTag, TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat
except ImportError:  # pragma: no cover - optional dependency
  xgr = None

@dataclass(frozen=True)
class ConstraintConfig:
  mode: str
  prompt_prefix: str
  compiled_grammar: Any
  tool_style: str|None = None
  tools: tuple[dict[str, Any], ...] = ()
  parallel_tool_calls: bool = False
  tool_choice: Any = None

class ConstraintUnavailableError(RuntimeError): pass

_VOCAB_TYPE_MAP = {
  "llama3": "BYTE_LEVEL",
  "llama-v3": "BYTE_LEVEL",
  "llama-bpe": "BYTE_LEVEL",
  "qwen2": "RAW",
  "olmo": "RAW",
  "kimi-k2": "RAW",
  "tekken": "BYTE_LEVEL",
  "glm4": "RAW",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def is_available() -> bool:
  return xgr is not None


def _require_xgrammar() -> None:
  if xgr is None: raise ConstraintUnavailableError("xgrammar is not installed")


def _model_vocab_size(server) -> int:
  if hasattr(getattr(getattr(server, "model", None), "output", None), "weight"): return int(server.model.output.weight.shape[0])
  if hasattr(server.tok, "_tok2bytes") and server.tok._tok2bytes: return max(server.tok._tok2bytes) + 1
  return 256


def _tokenizer_info(tok, vocab_size:int):
  _require_xgrammar()
  if hasattr(tok, "_xgr_tokenizer_info") and getattr(tok, "_xgr_vocab_size", None) == vocab_size: return tok._xgr_tokenizer_info
  encoded_vocab = [tok._tok2bytes.get(i, b"") for i in range(vocab_size)] if hasattr(tok, "_tok2bytes") else [bytes([i % 256]) for i in range(vocab_size)]
  vocab_type = getattr(xgr.VocabType, _VOCAB_TYPE_MAP.get(tok.preset, "RAW"))
  stop_token_ids = [tid for tid in (tok.eos_id, tok.eot_id) if tid is not None]
  info = xgr.TokenizerInfo(encoded_vocab, vocab_type=vocab_type, vocab_size=vocab_size, stop_token_ids=stop_token_ids or None)
  tok._xgr_tokenizer_info, tok._xgr_vocab_size = info, vocab_size
  return info


def _compiler(server):
  _require_xgrammar()
  vocab_size = _model_vocab_size(server)
  if hasattr(server, "_xgr_compiler") and getattr(server, "_xgr_vocab_size", None) == vocab_size: return server._xgr_compiler
  compiler = xgr.GrammarCompiler(_tokenizer_info(server.tok, vocab_size))
  server._xgr_compiler, server._xgr_vocab_size = compiler, vocab_size
  return compiler


def _tool_style_for_tokenizer(tok) -> str:
  return {"qwen2": "qwen", "llama3": "llama", "llama-v3": "llama", "llama-bpe": "llama", "kimi-k2": "kimi", "glm4": "glm47"}.get(tok.preset, "qwen")


def _tool_prompt(style:str, tools:list[dict[str, Any]], tool_choice:Any, parallel_tool_calls:bool) -> str:
  lines = ["# Tools", "You may call tools when helpful."]
  if tool_choice == "required": lines.append("You must call at least one tool.")
  elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function": lines.append(f"You must call the tool named {tool_choice['function']['name']}.")
  else: lines.append("You may either answer normally or call tool(s).")
  lines.append("Parallel tool calls are allowed." if parallel_tool_calls else "Call at most one tool in this response.")
  if style == "qwen":
    lines.append("Use Qwen tool call syntax exactly: <tool_call> then a JSON object with keys name and arguments, then </tool_call>.")
  for tool in tools:
    fn = tool.get("function", {})
    lines.append(json.dumps({"name": fn.get("name"), "parameters": fn.get("parameters", True)}, ensure_ascii=False))
  return "\n".join(lines)


def _forced_tool_name(tool_choice:Any) -> str|None:
  return tool_choice.get("function", {}).get("name") if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" else None


def _build_qwen_tool_structural_tag(tools:list[dict[str, Any]], tool_choice:Any, parallel_tool_calls:bool) -> StructuralTag:
  forced_name = _forced_tool_name(tool_choice)
  tags = []
  for tool in tools:
    function = tool.get("function", {})
    name = function.get("name")
    if forced_name is not None and name != forced_name: continue
    parameters = function.get("parameters", True) if function.get("strict", True) else True
    tags.append(TagFormat(begin=f'<tool_call>\n{{"name": "{name}", "arguments": ', content=JSONSchemaFormat(json_schema=parameters), end='}\n</tool_call>'))
  if not tags: raise ValueError("No compatible tools available for tool calling")
  if tool_choice == "required" or forced_name is not None:
    if parallel_tool_calls:
      return StructuralTag(format=TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True))
    return StructuralTag(format=TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True, stop_after_first=True))
  return StructuralTag(format=TriggeredTagsFormat(triggers=["<tool_call>"], tags=tags, stop_after_first=not parallel_tool_calls))


def build_constraint(server, body:dict[str, Any]) -> ConstraintConfig|None:
  response_format = body.get("response_format")
  tools = [tool for tool in body.get("tools", []) if tool.get("type", "function") == "function" and "function" in tool]
  if response_format is None and not tools: return None
  _require_xgrammar()
  compiler = _compiler(server)
  if response_format is not None:
    rtype = response_format.get("type")
    if rtype == "json_schema":
      schema = response_format.get("json_schema", {})
      if isinstance(schema, dict) and "schema" in schema: schema = schema["schema"]
      prompt = "Return ONLY valid JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
      return ConstraintConfig("json_schema", prompt, compiler.compile_json_schema(schema, strict_mode=True))
    if rtype == "json_object":
      return ConstraintConfig("json_schema", "Return ONLY valid JSON.", compiler.compile_builtin_json_grammar())
  if tools and body.get("tool_choice") != "none":
    style = _tool_style_for_tokenizer(server.tok)
    if style != "qwen": raise ConstraintUnavailableError(f"Tool calling is currently only implemented for qwen-style tokenizers, got {style}")
    parallel_tool_calls = bool(body.get("parallel_tool_calls", True))
    tool_choice = body.get("tool_choice", "auto")
    tag = _build_qwen_tool_structural_tag(tools, tool_choice, parallel_tool_calls)
    return ConstraintConfig("tools", _tool_prompt(style, tools, tool_choice, parallel_tool_calls), compiler.compile_structural_tag(tag), tool_style=style, tools=tuple(tools), parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice)
  return None


def make_token_selector(server, constraint:ConstraintConfig, temperature:float):
  _require_xgrammar()
  matcher = xgr.GrammarMatcher(constraint.compiled_grammar)
  bitmask = xgr.allocate_token_bitmask(1, _model_vocab_size(server))

  def token_selector(logits, tokens):
    if matcher.is_terminated():
      return server.tok.eot_id if getattr(server.tok, 'eot_id', None) is not None else server.tok.eos_id
    xgr.reset_token_bitmask(bitmask)
    matcher.fill_next_token_bitmask(bitmask)
    vocab_size = logits.shape[-1]
    row = bitmask[0].numpy().astype(np.uint32)
    allowed = np.array([bool((row[i // 32] >> np.uint32(i % 32)) & np.uint32(1)) for i in range(vocab_size)], dtype=bool)
    values = logits.numpy()[0].astype(np.float64)
    values[~allowed] = -np.inf
    if not np.isfinite(values).any(): raise RuntimeError("xgrammar masked out the entire vocabulary")
    if temperature < 1e-6:
      next_token = int(values.argmax())
    else:
      shifted = (values - np.nanmax(values)) / max(temperature, 1e-12)
      probs = np.exp(shifted)
      probs[~np.isfinite(probs)] = 0
      probs_sum = probs.sum()
      probs = probs / probs_sum if probs_sum > 0 else allowed.astype(np.float64) / allowed.sum()
      next_token = int(np.random.choice(np.arange(vocab_size), p=probs))
    matcher.accept_token(next_token)
    return next_token

  return token_selector


def constrain_messages(messages:list[dict[str, Any]], constraint:ConstraintConfig|None) -> list[dict[str, Any]]:
  if constraint is None or not constraint.prompt_prefix: return messages
  return [{"role": "system", "content": constraint.prompt_prefix}, *messages]


def serialize_assistant_tool_calls(tool_calls:list[dict[str, Any]], style:str) -> str:
  if style != "qwen": return json.dumps(tool_calls, ensure_ascii=False)
  parts = []
  for tool_call in tool_calls:
    fn = tool_call.get("function", {})
    args = fn.get("arguments", "{}")
    if not isinstance(args, str): args = json.dumps(args, ensure_ascii=False)
    parts.append(f'<tool_call>\n{{"name": "{fn.get("name", "")}", "arguments": {args}}}\n</tool_call>')
  return "\n".join(parts)


def content_to_text(content:Any) -> str:
  if isinstance(content, str): return content
  if isinstance(content, list): return ''.join(part.get('text', '') for part in content if part.get('type') == 'text')
  return '' if content is None else str(content)


def parse_tool_calls(text:str, style:str) -> tuple[str|None, list[dict[str, Any]]]:
  if style != "qwen": return text or None, []
  matches = list(_TOOL_CALL_RE.finditer(text))
  if not matches: return text or None, []
  tool_calls = []
  for idx, match in enumerate(matches):
    payload = json.loads(match.group(1))
    arguments = payload.get("arguments", payload.get("parameters", {}))
    tool_calls.append({
      "id": f"call_{uuid.uuid4().hex[:24]}",
      "type": "function",
      "index": idx,
      "function": {
        "name": payload["name"],
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
      },
    })
  text_without_tools = _TOOL_CALL_RE.sub('', text).strip()
  return text_without_tools or None, tool_calls
