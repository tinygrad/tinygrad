import json, unittest
from tinygrad.llm.tools import TOOL_CALL_OPEN, TOOL_CALL_CLOSE, format_tools, parse_tool_calls

BASH_TOOL = {"function": {"name": "bash", "parameters": {"properties": {"command": {}, "description": {}}, "required": ["command", "description"]}}}
READ_TOOL = {"function": {"name": "read", "parameters": {"properties": {"filePath": {}}, "required": ["filePath"]}}}
def tool_call(name, arguments): return TOOL_CALL_OPEN + json.dumps({"name": name, "arguments": arguments}) + TOOL_CALL_CLOSE

class TestLLMTools(unittest.TestCase):
  def test_format_tools(self):
    out = format_tools([BASH_TOOL, READ_TOOL])
    self.assertIn("bash(command,description)", out)
    self.assertIn("read(filePath)", out)
    self.assertIn(TOOL_CALL_OPEN + '{"name":"...","arguments":{...}}' + TOOL_CALL_CLOSE, out)
    self.assertNotIn('"properties"', out)

  def test_parse_tool_calls(self):
    calls = parse_tool_calls(tool_call("read", {"filePath": "old.py"}) + tool_call("bash", {"command": "ls"}))
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0]["function"]["name"], "bash")
    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls"})

  def test_parse_tool_calls_bad_json(self):
    self.assertEqual(parse_tool_calls(TOOL_CALL_OPEN + '{"name":"bash","arguments":' + TOOL_CALL_CLOSE), [])
