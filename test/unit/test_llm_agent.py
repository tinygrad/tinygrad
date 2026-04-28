import json, unittest
from tinygrad.llm.agent import TOOL_CALL_OPEN, TOOL_CALL_CLOSE, format_tools, parse_tool_calls

BASH_TOOL = {"function": {"name": "bash", "parameters": {
  "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}}}
READ_TOOL = {"function": {"name": "read", "parameters": {
  "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}

class TestLLMAgent(unittest.TestCase):
  def test_format_tools_uses_full_schemas(self):
    out = format_tools([BASH_TOOL, READ_TOOL])
    self.assertIn("<tools>", out)
    self.assertIn('"name": "bash"', out)
    self.assertIn('"name": "read"', out)
    self.assertIn('"properties"', out)
    self.assertIn('The "name" field must exactly match one tool name listed in <tools>.', out)
    self.assertIn(TOOL_CALL_OPEN, out)
    self.assertIn(TOOL_CALL_CLOSE, out)

  def test_parse_tool_calls_complete_tag(self):
    text = TOOL_CALL_OPEN + '{"name":"read","arguments":{"filePath":"x.py"}}' + TOOL_CALL_CLOSE
    calls = parse_tool_calls(text)
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["function"]["name"], "read")
    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"filePath": "x.py"})

  def test_parse_tool_calls_returns_empty_on_bad_json(self):
    text = TOOL_CALL_OPEN + '{"name":"bash","arguments":' + TOOL_CALL_CLOSE
    self.assertEqual(parse_tool_calls(text), [])

if __name__ == "__main__":
  unittest.main()
