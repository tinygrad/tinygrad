"""End-to-end regression tests for tinygrad's OpenCode integration.

Run against an existing server:
  RUN_LLM_OPENCODE_REGRESSION=1 LLM_BASE_URL=http://127.0.0.1:9000/v1 \
    python -m pytest test/external/external_test_llm_opencode.py -v

Or let the test start the server:
  RUN_LLM_OPENCODE_REGRESSION=1 \
    LLM_GGUF=/raid/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    python -m pytest test/external/external_test_llm_opencode.py -v
"""
from __future__ import annotations

import fcntl, json, os, pathlib, re, shutil, socket, subprocess, sys, tempfile, time, unittest, urllib.request


RUN_REGRESSION = os.getenv("RUN_LLM_OPENCODE_REGRESSION") == "1"
DEFAULT_GGUF = "/raid/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"

SORT_C = r"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#define N 1000000

/* Optimized LSD Radix sort using 8-bit chunks (256 buckets) with loop unrolling and prefetching */
static void radix_sort(int *a, int *buf, int n) {
    const int BITS = 8;
    const int MASK = (1 << BITS) - 1;
    const int BUCKETS = (1 << BITS);
    uint32_t count[BUCKETS];
    uint32_t *u = (uint32_t *)a;
    uint32_t *ubuf = (uint32_t *)buf;
    uint32_t *src = u;
    uint32_t *dst = ubuf;

    /* Convert signed to unsigned by flipping sign bit */
    for (int i = 0; i < n; i++)
        u[i] ^= (uint32_t)1U << 31;

    for (int shift = 0; shift < 32; shift += BITS) {
        memset(count, 0, sizeof(count));

        /* Counting pass - unrolled by 8 with prefetching */
        int i = 0;
        for (; i + 7 < n; i += 8) {
            count[(src[i] >> shift) & MASK]++;
            count[(src[i+1] >> shift) & MASK]++;
            count[(src[i+2] >> shift) & MASK]++;
            count[(src[i+3] >> shift) & MASK]++;
            count[(src[i+4] >> shift) & MASK]++;
            count[(src[i+5] >> shift) & MASK]++;
            count[(src[i+6] >> shift) & MASK]++;
            count[(src[i+7] >> shift) & MASK]++;
        }
        for (; i < n; i++)
            count[(src[i] >> shift) & MASK]++;

        /* Prefix sums in-place */
        uint32_t total = 0;
        for (int i = 0; i < BUCKETS; i++) {
            uint32_t c = count[i];
            count[i] = total;
            total += c;
        }

        /* Distribution pass - unrolled by 8 with prefetching */
        i = 0;
        for (; i + 7 < n; i += 8) {
            dst[count[(src[i] >> shift) & MASK]++] = src[i];
            dst[count[(src[i+1] >> shift) & MASK]++] = src[i+1];
            dst[count[(src[i+2] >> shift) & MASK]++] = src[i+2];
            dst[count[(src[i+3] >> shift) & MASK]++] = src[i+3];
            dst[count[(src[i+4] >> shift) & MASK]++] = src[i+4];
            dst[count[(src[i+5] >> shift) & MASK]++] = src[i+5];
            dst[count[(src[i+6] >> shift) & MASK]++] = src[i+6];
            dst[count[(src[i+7] >> shift) & MASK]++] = src[i+7];
        }
        for (; i < n; i++)
            dst[count[(src[i] >> shift) & MASK]++] = src[i];

        /* Swap src/dst pointers */
        uint32_t *tmp = src;
        src = dst;
        dst = tmp;
    }

    /* Copy back if needed */
    if (src != u)
        memcpy(u, src, n * sizeof(uint32_t));

    /* Convert back to signed */
    for (int i = 0; i < n; i++)
        u[i] ^= (uint32_t)1U << 31;
}

int main() {
    int *arr = malloc(N * sizeof(int));
    int *buf = malloc(N * sizeof(int));
    if (!arr || !buf) { perror("malloc"); return 1; }

    srand(42);
    for (int i = 0; i < N; i++)
        arr[i] = rand() | (rand() << 15);

    clock_t start = clock();
    radix_sort(arr, buf, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Sorted %d integers in %.3f ms\n", N, elapsed);

    for (int i = 1; i < N; i++) {
        if (arr[i] < arr[i - 1]) {
            printf("ERROR: not sorted at index %d\n", i);
            free(arr);
            free(buf);
            return 1;
        }
    }
    printf("Verification passed.\n");

    free(arr);
    free(buf);
    return 0;
}
"""


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def server_ready(base_url:str) -> bool:
  try:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=1) as response:
      return response.status == 200
  except (OSError, urllib.error.URLError):
    return False

def wait_server_ready(base_url:str, timeout:float=120) -> bool:
  # The inference server is intentionally single-request. Under xdist another worker can be running a completion while
  # this worker's setUpClass probes /models, so a single one-second probe is not evidence that the server is down.
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if server_ready(base_url): return True
    time.sleep(0.25)
  return False


@unittest.skipUnless(RUN_REGRESSION, "set RUN_LLM_OPENCODE_REGRESSION=1 to run the real-model OpenCode regression")
class TestLLMOpenCode(unittest.TestCase):
  server: subprocess.Popen|None = None
  server_log: tempfile._TemporaryFileWrapper|None = None

  @classmethod
  def setUpClass(cls):
    if shutil.which("opencode") is None: raise unittest.SkipTest("opencode is not installed")
    if (base_url := os.getenv("LLM_BASE_URL")) is not None:
      cls.base_url = base_url.rstrip("/")
      if not cls.base_url.endswith("/v1"): cls.base_url += "/v1"
      # Coordinate class-level probes with test requests too. Otherwise xdist workers can queue one-second /models
      # probes behind a completion, time out, and later make the server write to already-closed sockets.
      with open("/tmp/tinygrad-llm-opencode-regression.lock", "w") as server_lock:
        fcntl.flock(server_lock, fcntl.LOCK_EX)
        if not wait_server_ready(cls.base_url): raise RuntimeError(f"LLM server is not responding at {cls.base_url}")
      return

    model = pathlib.Path(os.getenv("LLM_GGUF", DEFAULT_GGUF))
    if not model.is_file(): raise unittest.SkipTest(f"model not found: {model}")
    port = free_port()
    cls.base_url = f"http://127.0.0.1:{port}/v1"
    cls.server_log = tempfile.NamedTemporaryFile(mode="w+", prefix="tinygrad-llm-")
    cls.server = subprocess.Popen(
      [sys.executable, "-m", "tinygrad.llm", "--model", str(model), "--serve", str(port), "--max_context", "262144"],
      stdout=cls.server_log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and cls.server.poll() is None:
      if server_ready(cls.base_url): return
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

  def setUp(self):
    # The model and its KV/recurrent caches are stateful. Keep xdist workers from interleaving independent OpenCode
    # conversations, which changes cache reuse and can hide precisely the incremental path this suite exercises.
    self._server_lock = open("/tmp/tinygrad-llm-opencode-regression.lock", "w")
    fcntl.flock(self._server_lock, fcntl.LOCK_EX)

  def tearDown(self):
    fcntl.flock(self._server_lock, fcntl.LOCK_UN)
    self._server_lock.close()

  def run_opencode(self, prompt:str, cwd:pathlib.Path, timeout:int=120) -> str:
    config = cwd / "opencode.json"
    config.write_text(json.dumps({
      "$schema": "https://opencode.ai/config.json",
      "permission": {"*": "allow"},
      "formatter": False,
      "lsp": False,
      "provider": {"regression": {
        "npm": "@ai-sdk/openai-compatible",
        "options": {"baseURL": self.base_url},
        "models": {"tinygrad": {"name": "tinygrad"}},
      }},
    }))
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(config)
    result = subprocess.run(["opencode", "run", "--pure", "--auto", "--dir", str(cwd), "-m", "regression/tinygrad", prompt], cwd=cwd,
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    self.assertEqual(result.returncode, 0, result.stdout)
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout)

  def chat(self, messages:list[dict], max_tokens:int=4) -> dict:
    request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps({
      "model":"tinygrad", "messages":messages, "max_tokens":max_tokens, "temperature":0,
    }).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request, timeout=120) as response: return json.load(response)

  def test_same_session_reuses_prompt_cache_after_changed_generated_tail(self):
    # OpenCode reconstructs assistant tool calls from structured deltas, so their rendered tail need not be token-identical
    # to what the model generated. The long stable prompt before that tail must still be restored from a recurrent checkpoint.
    messages = [{"role":"system", "content":"You are a concise assistant. " * 400}, {"role":"user", "content":"Reply OK."}]
    first = self.chat(messages)
    content = first["choices"][0]["message"].get("content") or "OK"
    content = content[:-1] + ("x" if content[-1:] != "x" else "y")
    messages += [{"role":"assistant", "content":content},
                 {"role":"user", "content":"Reply OK again."}]
    second = self.chat(messages)
    self.assertGreater(second["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

  def test_reads_and_correctly_explains_valid_c(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      source = cwd / "sort.c"
      source.write_text(SORT_C)
      subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only", str(source)], check=True)

      output = self.run_opencode("sort.c?", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:→|>)\s*Read\s+sort\.c\s*$", "OpenCode did not execute the read tool")
      self.assertRegex(output, r"(?i)radix sort")
      self.assertNotRegex(output, r"(?i)(corrupt|mangled|not valid C|invalid C|syntax error|does not compile|won't compile)")
      self.assertEqual(source.read_text(), SORT_C)
      subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only", str(source)], check=True)

  def test_read_tool_preserves_exact_contents(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      marker = "tinygrad-read-regression-7f3a91c2"
      source = cwd / "exact.txt"
      source.write_text(marker + "\n")

      output = self.run_opencode("Read exact.txt with a tool and reply with its exact contents, with no other text.", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:→|>)\s*Read\s+exact\.txt\s*$", "OpenCode did not execute the read tool")
      self.assertIn(marker, output)
      self.assertEqual(source.read_text(), marker + "\n")

  def test_executes_shell_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      marker = cwd / "shell-regression.txt"
      output = self.run_opencode(
        "Use the shell tool to run `printf tinygrad-shell-regression > shell-regression.txt`, then report completion.", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:\$|→|>)\s*.*printf\s+tinygrad-shell-regression",
                       "OpenCode did not execute the shell tool")
      self.assertTrue(marker.is_file(), output)
      self.assertEqual(marker.read_text(), "tinygrad-shell-regression")

  def test_does_not_repeat_identical_failed_shell_call(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      command = "clang -x c /dev/null -fsyntax-only -mllvm -tinygrad-definitely-invalid-option=1"
      output = self.run_opencode(
        f"Run `{command}` exactly once with the shell tool. After it fails, do not retry it; explain that the option is unsupported.", cwd)
      self.assertIn("Unknown command line argument", output)
      self.assertLessEqual(output.count(command), 1, output)

  def test_stops_when_benchmark_goal_is_met(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      benchmark = cwd / "benchmark.sh"
      benchmark.write_text("#!/bin/sh\necho 'Sorted 1000000 integers in 8.300 ms'\n")
      benchmark.chmod(0o755)
      output = self.run_opencode(
        "Use the shell tool to run ./benchmark.sh. Keep optimizing until it reports under 10 ms, then stop immediately.", cwd)
      self.assertIn("8.300 ms", output)
      self.assertLessEqual(output.count("$ ./benchmark.sh"), 1, output)

  def test_multiline_tool_argument_preserves_trailing_newline(self):
    with tempfile.TemporaryDirectory() as directory:
      cwd = pathlib.Path(directory)
      target = cwd / "numbers.txt"
      target.write_text("replace me\n")
      output = self.run_opencode(
        "Read numbers.txt, then use the write tool to replace it with the numbers 1 through 300, one number per line. Do not use bash.", cwd)
      self.assertRegex(output, r"(?im)^\s*(?:←|→|>)\s*Write\s+numbers\.txt\s*$", "OpenCode did not execute the write tool")
      self.assertEqual(target.read_text(), "".join(f"{i}\n" for i in range(1, 301)))


if __name__ == "__main__": unittest.main()
