import json, unittest
from tinygrad.llm.agent import TOOL_CALL_OPEN, format_tools, parse_tool_calls

BASH_TOOL = {
  "type": "function",
  "function": {
    "name": "bash",
    "description": "run shell command",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {"type": "string"},
        "description": {"type": "string"},
        "timeout": {"type": "integer"},
      },
      "required": ["command", "description"],
    },
  },
}

READ_TOOL = {
  "type": "function",
  "function": {
    "name": "read",
    "parameters": {
      "type": "object",
      "properties": {"filePath": {"type": "string"}},
      "required": ["filePath"],
    },
  },
}

class TestLLMAgent(unittest.TestCase):
  def test_format_tools_uses_compact_signatures(self):
    out = format_tools([BASH_TOOL, READ_TOOL])
    self.assertIn("<tools>", out)
    self.assertIn("bash(command:string, description:string", out)
    self.assertIn("read(filePath:string)", out)
    self.assertIn(TOOL_CALL_OPEN, out)
    self.assertIn("Include all required arguments.", out)
    self.assertNotIn('"properties"', out)

  def test_parse_tool_calls_open_tag(self):
    text = TOOL_CALL_OPEN + '{"name":"read","arguments":{"filePath":"x.py"}}'
    calls = parse_tool_calls(text)
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["function"]["name"], "read")
    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"filePath": "x.py"})

  def test_parse_tool_calls_recovers_trailing_open_tag(self):
    text = TOOL_CALL_OPEN + '{"name":"bash","arguments":{"command":"ls -la"}}'
    calls = parse_tool_calls(text)
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["function"]["name"], "bash")

  def test_parse_tool_calls_leaves_missing_description_unchanged(self):
    text = TOOL_CALL_OPEN + '{"name":"bash","arguments":{"command":"ls -la"}}'
    calls = parse_tool_calls(text)
    args = json.loads(calls[0]["function"]["arguments"])
    self.assertEqual(args["command"], "ls -la")
    self.assertNotIn("description", args)

  def test_parse_tool_calls_returns_empty_on_bad_json(self):
    text = TOOL_CALL_OPEN + '{"name":"bash","arguments":'
    self.assertEqual(parse_tool_calls(text), [])

if __name__ == "__main__":
  unittest.main()
