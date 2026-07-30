"""Real-model OpenCode regression.

Run against an existing server:
  RUN_LLM_OPENCODE_REGRESSION=1 LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    python -m pytest test/external/external_test_llm_opencode.py -v

Or set LLM_GGUF and let the test start the tinygrad server.
"""
from __future__ import annotations

import json, os, pathlib, re, shutil, socket, subprocess, sys, tempfile, time, unittest, urllib.request

RUN_REGRESSION = os.getenv("RUN_LLM_OPENCODE_REGRESSION") == "1"

def _server_ready(base_url:str) -> bool:
  try:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=1) as response: return response.status == 200
  except OSError: return False

@unittest.skipUnless(RUN_REGRESSION, "set RUN_LLM_OPENCODE_REGRESSION=1 to run the OpenCode regression")
class TestLLMOpenCode(unittest.TestCase):
  server:subprocess.Popen|None = None
  server_log:tempfile._TemporaryFileWrapper|None = None

  @classmethod
  def setUpClass(cls):
    if shutil.which("opencode") is None: raise unittest.SkipTest("opencode is not installed")
    if base_url := os.getenv("LLM_BASE_URL"):
      cls.base_url = base_url.rstrip("/")
      if not cls.base_url.endswith("/v1"): cls.base_url += "/v1"
      if not _server_ready(cls.base_url): raise RuntimeError(f"LLM server is not responding at {cls.base_url}")
      return

    model = pathlib.Path(os.environ["LLM_GGUF"])
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      port = sock.getsockname()[1]
    cls.base_url = f"http://127.0.0.1:{port}/v1"
    cls.server_log = tempfile.NamedTemporaryFile(mode="w+", prefix="tinygrad-llm-")
    cls.server = subprocess.Popen(
      [sys.executable, "-m", "tinygrad.llm", "--model", str(model), "--serve", str(port), "--max_context", "262144"],
      stdout=cls.server_log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and cls.server.poll() is None:
      if _server_ready(cls.base_url): return
      time.sleep(0.25)
    cls.server_log.seek(0)
    raise RuntimeError(f"LLM server failed to start:\n{cls.server_log.read()[-8000:]}")

  @classmethod
  def tearDownClass(cls):
    if cls.server is not None:
      cls.server.terminate()
      try: cls.server.wait(timeout=10)
      except subprocess.TimeoutExpired:
        cls.server.kill()
        cls.server.wait(timeout=10)
    if cls.server_log is not None: cls.server_log.close()

  def run_opencode(self, prompt:str, cwd:pathlib.Path) -> str:
    config = cwd / "opencode.json"
    config.write_text(json.dumps({
      "$schema": "https://opencode.ai/config.json", "permission": {"*": "allow"}, "formatter": False, "lsp": False,
      "provider": {"regression": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": self.base_url},
                                  "models": {"tinygrad": {"name": "tinygrad"}}}},
    }))
    env = os.environ | {"OPENCODE_CONFIG": str(config)}
    result = subprocess.run(
      ["opencode", "run", "--pure", "--auto", "--dir", str(cwd), "-m", "regression/tinygrad", prompt],
      cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    self.assertEqual(result.returncode, 0, result.stdout)
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout)

  def test_read_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd, marker = pathlib.Path(directory), "tinygrad-opencode-regression-7f3a91c2"
      (cwd / "exact.txt").write_text(marker + "\n")
      output = self.run_opencode("Read exact.txt with a tool and reply with its exact contents, with no other text.", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:→|>)\s*Read\s+exact\.txt\s*$")
      self.assertIn(marker, output)

  def test_shell_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      output = self.run_opencode(
        "Use the shell tool to run `printf tinygrad-shell-regression > shell-regression.txt`, then report completion.", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:\$|→|>)\s*.*printf\s+tinygrad-shell-regression")
      self.assertEqual((cwd / "shell-regression.txt").read_text(), "tinygrad-shell-regression")

if __name__ == "__main__": unittest.main()
