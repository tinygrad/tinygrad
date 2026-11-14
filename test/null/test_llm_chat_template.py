# cli.chat_template: its environment reads a plain dict's field at once (no AttributeError first): the rendered text must be byte for byte
# jinja2's own (a stock Environment with the same settings) on the model's template (Qwen 3.8 Flash's, qwen38_chat_template.jinja: from the
# GGUF's tokenizer.chat_template) for a text-only agent conversation (DeepSWE's shape) and a picture one (Pokemon's), and on a template that
# reads a dict's own attribute names (items / get / keys / values / pop: the attribute first, as jinja2) and missing fields (undefined)
import unittest, json, time, pathlib, types, copy
from tinygrad.llm.cli import chat_template
from tinygrad.llm.serve import normalize_messages

QWEN = (pathlib.Path(__file__).parent / "qwen38_chat_template.jinja").read_text()
TOK = types.SimpleNamespace(bos_id=None, eos_id=0, decode=lambda ids: "<|im_end|>")

def stock(ct:str):
  import jinja2
  env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
  env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
  env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
  env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
  env.globals['bos_token'], env.globals['eos_token'] = "", "<|im_end|>"
  return env.from_string(ct)

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command", "parameters": {"type": "object", "properties":
          {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "press", "description": "press a button", "parameters": {"type": "object", "properties": {}}}}]

def agent_conversation(turns:int) -> list[dict]:
  """DeepSWE's shape: a system and a task, then assistant turns (reasoning, text, tool calls with JSON-string arguments) and tool results"""
  ms:list[dict] = [{"role": "system", "content": "You are a helpful assistant that can interact with a computer."},
                   {"role": "user", "content": "<pr_description>\nFix the bug in parse() <&> \"quotes\" \u00e9\n</pr_description>"}]
  for i in range(turns):
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"grep -n 'x{i}' a.py | head"})}}]
    if i % 3 == 0: calls.append({"id": f"d{i}", "type": "function", "function": {"name": "press", "arguments": "{}"}})
    ms.append({"role": "assistant", "content": "" if i % 2 else f"Step {i}: look.", "reasoning_content": f"  think {i}\n", "tool_calls": calls})
    ms += [{"role": "tool", "tool_call_id": c["id"], "content": f"out {i}\n<tag>&" if i % 4 else [{"type": "text", "text": f"part {i}"}]}
           for c in calls]
    if i % 5 == 4: ms.append({"role": "user", "content": f"keep going {i}", "items": "a field named as a dict method", "get": 1})
  return ms

def picture_conversation(steps:int) -> list[dict]:
  """Pokemon's shape: one task, then a tool call and its result (text + a screenshot: the image part VISION makes of it) a step"""
  ms:list[dict] = [{"role": "user", "content": "Play Pokemon Red."}]
  for i in range(steps):
    ms.append({"role": "assistant", "content": "", "reasoning_content": f"I see the map {i}.",
               "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "press", "arguments": json.dumps({"button": "up"})}}]})
    ms.append({"role": "tool", "tool_call_id": f"t{i}", "content": [{"type": "text", "text": f"pressed up ({i})"}, {"type": "image"}]})
  return ms

class TestChatTemplate(unittest.TestCase):
  def same(self, ct:str, messages:list[dict], **kw):
    a, b = copy.deepcopy(messages), copy.deepcopy(messages)
    for m in (a, b): normalize_messages(m, images=True)
    args = {"tools": TOOLS, "add_generation_prompt": True, "preserve_thinking": True, **kw}
    got, want = chat_template(ct, TOK).render(messages=a, **args), stock(ct).render(messages=b, **args)
    self.assertEqual(got, want)
    return got

  def test_qwen_agent_text(self):
    text = self.same(QWEN, agent_conversation(60))
    self.assertIn("<tool_call>\n<function=bash>", text)
    self.same(QWEN, agent_conversation(7), reasoning_effort="high")
    self.same(QWEN, agent_conversation(7), enable_thinking=False)
    self.same(QWEN, agent_conversation(7), preserve_thinking=False)

  def test_qwen_pictures(self):
    text = self.same(QWEN, picture_conversation(200))
    self.assertEqual(text.count("<|image_pad|>"), 200)
    self.same(QWEN, picture_conversation(5), add_vision_id=True)

  def test_qwen_errors_alike(self):
    for bad in ([{"role": "user", "content": "a"}, {"role": "system", "content": "late"}], []):
      for t in (chat_template(QWEN, TOK), stock(QWEN)):
        with self.assertRaises(Exception): t.render(messages=bad, add_generation_prompt=True)

  def test_dict_attributes_and_missing_fields(self):
    ct = ("{% for m in messages %}[{{ m.role }}|{{ m.items is defined }}|{{ m.get is callable }}|{{ m.keys is callable }}|{{ m.pop is callable }}"
          "|{{ m.missing is defined }}|{{ m.missing }}|{{ m['items'] if 'items' in m else '-' }}"
          "|{{ m.content.text if m.content is mapping else m.content }}"
          "|{{ m.__class__ is defined }}|{{ m | tojson }}]{% endfor %}{{ ns_test }}")
    ms = [{"role": "user", "content": "x", "items": [1, 2], "get": "g", "keys": "k", "pop": None},
          {"role": "tool", "content": {"text": "mapped", "values": 3}}, {"content": None}]
    self.same(ct, ms)

if __name__ == "__main__":
  unittest.main()
