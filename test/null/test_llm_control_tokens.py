# A GLM reply's role marker ends it and a CONTROL token is never its text (ts5 issue 3, Nov 16: at reasoning_effort low the model wrote
# '<|assistant|>' after a short answer and a second turn after it; the reply showed both). The fixtures are the two greedy cases.
import threading, time, unittest
from unittest.mock import Mock
from tinygrad.llm.cli import SimpleTokenizer

SPECIAL = [("<|endoftext|>", 3), ("<|user|>", 3), ("<|assistant|>", 3), ("<|observation|>", 3), ("<|system|>", 3), ("<think>", 4),
           ("</think>", 4), ("<|begin_of_box|>", 3)]
CHARS = [chr(c) for c in range(33, 127)] + ["Ġ"]  # (byte-level: Ġ is the space)

def glm_tok() -> SimpleTokenizer:
  kv = {"tokenizer.ggml.tokens": [t for t, _ in SPECIAL] + CHARS, "tokenizer.ggml.token_type": [k for _, k in SPECIAL] + [1] * len(CHARS),
        "tokenizer.ggml.pre": "glm4", "tokenizer.ggml.eos_token_id": 0, "tokenizer.ggml.eot_token_id": 1, "tokenizer.ggml.eom_token_id": 3,
        "tokenizer.ggml.add_bos_token": False}
  return SimpleTokenizer.from_gguf_kv(kv)

class TestTokenizerEnds(unittest.TestCase):
  def test_role_markers_end(self):
    tok = glm_tok()
    self.assertEqual([tok.is_end(i) for i in range(5)], [True] * 5)  # eos, <|user|>, <|assistant|>, <|observation|>, <|system|>
    self.assertFalse(tok.is_end(6))  # </think>
    self.assertEqual(tok.control_ids, frozenset({0, 1, 2, 3, 4, 7}))  # type 3 only: <think> / </think> (4) are markup, not control

  def test_no_role_markers(self):
    tok = SimpleTokenizer.from_gguf_kv({"tokenizer.ggml.tokens": ["<|im_end|>", "a"], "tokenizer.ggml.token_type": [3, 1],
                                        "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.eos_token_id": 0})
    self.assertTrue(tok.is_end(0))
    self.assertFalse(tok.is_end(1))

class TestControlTokensServed(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    import jinja2
    from tinygrad.llm.serve import LLMServer
    cls.tok = glm_tok()
    cls.mock_model = Mock()
    cls.mock_model.max_context = 4096
    cls.mock_model.get_start_pos = Mock(return_value=0)
    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "glm", cls.tok, jinja2.Template("{{ messages[-1].content }}<|assistant|><think>"))
    threading.Thread(target=cls.server.serve_forever, daemon=True).start()
    time.sleep(0.1)
    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.server.server_address[1]}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def reply(self, generated:str, stream:bool) -> tuple[str, str, str|None]:
    ids = self.tok.encode(generated)
    self.mock_model.generate = Mock(side_effect=lambda *a, **k: iter(ids))
    msgs = [{"role": "user", "content": "q"}]
    if not stream:
      r = self.client.chat.completions.create(model="glm", messages=msgs, max_tokens=200)
      m = r.choices[0].message
      return m.content or "", getattr(m, "reasoning_content", None) or "", r.choices[0].finish_reason
    chunks = [c for c in self.client.chat.completions.create(model="glm", messages=msgs, max_tokens=200, stream=True) if c.choices]
    return ("".join(c.choices[0].delta.content or "" for c in chunks),
            "".join(getattr(c.choices[0].delta, "reasoning_content", None) or "" for c in chunks), chunks[-1].choices[0].finish_reason)

  def test_knights(self):
    for stream in (False, True):
      with self.subTest(stream=stream):
        self.assertEqual(self.reply("</think>A: knave, B: knight<|assistant|></think>A: knave, B: knight<|user|>", stream),
                         ("A: knave, B: knight", "", "stop"))

  def test_runner_order(self):
    for stream in (False, True):
      with self.subTest(stream=stream):
        self.assertEqual(self.reply("order.</think>DABC<|assistant|>Related user asks.</think>DABC<|user|>", stream),
                         ("DABC", "order.", "stop"))

  def test_control_token_not_shown(self):
    self.assertEqual(self.reply("</think> <|begin_of_box|>42<|user|>", True), (" 42", "", "stop"))

if __name__ == '__main__':
  unittest.main()
