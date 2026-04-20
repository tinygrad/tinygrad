import json, unittest
from tinygrad.llm.agent import TOOL_CALL_OPEN, format_tools, parse_tool_calls

BASH_TOOL = {"function": {"name": "bash", "parameters": {
  "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}}}
READ_TOOL = {"function": {"name": "read", "parameters": {
  "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}

class TestLLMAgent(unittest.TestCase):
  def test_format_tools_uses_compact_signatures(self):
    out = format_tools([BASH_TOOL, READ_TOOL])
    self.assertIn("<tools>", out)
    self.assertIn("bash(command:string, description:string[brief summary]", out)
    self.assertIn("read(filePath:string)", out)
    self.assertIn(TOOL_CALL_OPEN, out)
    self.assertNotIn('"properties"', out)

  def test_parse_tool_calls_open_tag(self):
    text = TOOL_CALL_OPEN + '{"name":"read","arguments":{"filePath":"x.py"}}'
    calls = parse_tool_calls(text)
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["function"]["name"], "read")
    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"filePath": "x.py", "description": ""})

  def test_parse_tool_calls_returns_empty_on_bad_json(self):
    text = TOOL_CALL_OPEN + '{"name":"bash","arguments":'
    self.assertEqual(parse_tool_calls(text), [])

if __name__ == "__main__":
  unittest.main()
