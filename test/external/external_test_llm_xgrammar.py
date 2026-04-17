import json, os, subprocess, tempfile, time, unittest
from pathlib import Path

import requests

PORT = int(os.getenv("QWEN35_XGRAMMAR_PORT", "8017"))
REPO = Path(__file__).resolve().parents[2]

class TestRealLLMXGrammar(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if os.getenv("RUN_REAL_QWEN35_XGRAMMAR") != "1":
      raise unittest.SkipTest("set RUN_REAL_QWEN35_XGRAMMAR=1 to run the real Qwen3.5 2B xgrammar smoke test")
    cls.proc = subprocess.Popen([
      str(REPO / ".venv/bin/python"), "-m", "tinygrad.llm", "--model", os.getenv("QWEN35_XGRAMMAR_MODEL", "qwen3.5:2b"), "--serve", str(PORT), "--warmup",
    ], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 600
    while time.time() < deadline:
      try:
        requests.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).raise_for_status()
        break
      except Exception:
        time.sleep(2)
    else:
      raise RuntimeError("tinygrad llm server failed to start")

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, "proc"):
      cls.proc.terminate()
      try: cls.proc.wait(timeout=10)
      except subprocess.TimeoutExpired: cls.proc.kill()

  def chat(self, payload):
    resp = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

  def test_json_schema(self):
    data = self.chat({
      "model": "qwen3.5:2b",
      "messages": [{"role": "user", "content": "Return the capital of France and the integer 7."}],
      "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"capital": {"type": "string"}, "n": {"type": "integer"}}, "required": ["capital", "n"], "additionalProperties": False}}},
      "temperature": 0,
      "max_tokens": 120,
    })
    parsed = json.loads(data["choices"][0]["message"]["content"])
    self.assertEqual(sorted(parsed.keys()), ["capital", "n"])
    self.assertIsInstance(parsed["n"], int)

  def test_required_tool_call(self):
    data = self.chat({
      "model": "qwen3.5:2b",
      "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
      "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}],
      "tool_choice": "required",
      "temperature": 0,
      "max_tokens": 120,
    })
    tool_calls = data["choices"][0]["message"]["tool_calls"]
    self.assertGreaterEqual(len(tool_calls), 1)
    self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")

  def test_parallel_tool_calls(self):
    data = self.chat({
      "model": "qwen3.5:2b",
      "messages": [{"role": "user", "content": "Use both tools: weather in Paris and time in Europe/Paris."}],
      "tools": [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"], "additionalProperties": False}}},
      ],
      "tool_choice": "required",
      "parallel_tool_calls": True,
      "temperature": 0,
      "max_tokens": 180,
    })
    self.assertGreaterEqual(len(data["choices"][0]["message"]["tool_calls"]), 1)

if __name__ == "__main__":
  unittest.main()
