import unittest
from tinygrad.llm.cli import SimpleTokenizer, Chat

class TestChatSimple(unittest.TestCase):
  def test_tekken_preset(self):
    # Tekken (Mistral): role(user)=[INST], role(assistant)=[], end_turn=[/INST].
    kv = {
      "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "hello", "sure"],
      "tokenizer.ggml.token_type": [3, 3, 3, 3, 3, 1, 1],
      "tokenizer.ggml.pre": "tekken",
      "tokenizer.ggml.eos_token_id": 2,
    }
    tok = SimpleTokenizer.from_gguf_kv(kv)
    chat = Chat.from_gguf_kv(kv, tok)
    # single user turn: [INST] hello [/INST]
    self.assertEqual(chat.apply([{"role": "user", "content": "hello"}]), [3, 5, 4])
    # user + assistant: [INST] hello [/INST] sure [/INST]
    self.assertEqual(chat.apply([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "sure"}]),
                     [3, 5, 4, 6, 4])
    # add_generation_prompt on tekken appends role("assistant") which is []
    self.assertEqual(chat.apply([{"role": "user", "content": "hello"}], add_generation_prompt=True), [3, 5, 4])

  def test_is_end_basic(self):
    kv = {"tokenizer.ggml.tokens": ["<unk>", "<eos>"], "tokenizer.ggml.token_type": [3, 3],
          "tokenizer.ggml.pre": "llama3", "tokenizer.ggml.eos_token_id": 1}
    tok = SimpleTokenizer.from_gguf_kv(kv)
    chat = Chat(tok, preset="llama3")
    self.assertTrue(chat.is_end(1))
    self.assertFalse(chat.is_end(0))

  def test_olmo2_simple_mode(self):
    # OLMo 2: pre='dbrx' (unsupported) but arch override maps it to qwen2-style chat with <|im_end|> as turn-end.
    # tokenizer's eos_id must be untouched; Chat widens stop_ids and uses <|im_end|> for end_turn.
    kv = {
      "tokenizer.ggml.tokens": ["<|endoftext|>", "hello", "<|im_end|>", "<|im_start|>"],
      "tokenizer.ggml.token_type": [3, 1, 3, 3],
      "tokenizer.ggml.pre": "dbrx",
      "tokenizer.ggml.eos_token_id": 0,
      "general.architecture": "olmo2",
    }
    tok = SimpleTokenizer.from_gguf_kv(kv)
    chat = Chat.from_gguf_kv(kv, tok)
    self.assertEqual(tok.eos_id, 0)                # raw GGUF eos (<|endoftext|>) untouched
    self.assertEqual(chat.preset, "qwen2")         # arch-overridden preset
    self.assertEqual(chat.turn_end_id, 2)          # <|im_end|>
    self.assertTrue(chat.is_end(0))                # <|endoftext|>
    self.assertTrue(chat.is_end(2))                # <|im_end|>
    self.assertFalse(chat.is_end(1))

  def test_assistant_prefill_no_end_turn(self):
    # continue_final_message should drop the trailing end_turn after the prefill assistant message.
    kv = {
      "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "hello", "sure"],
      "tokenizer.ggml.token_type": [3, 3, 3, 3, 3, 1, 1],
      "tokenizer.ggml.pre": "tekken",
      "tokenizer.ggml.eos_token_id": 2,
    }
    tok = SimpleTokenizer.from_gguf_kv(kv)
    chat = Chat.from_gguf_kv(kv, tok)
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "sure"}]
    self.assertEqual(chat.apply(msgs), [3, 5, 4, 6, 4])
    self.assertEqual(chat.apply(msgs, continue_final_message=True), [3, 5, 4, 6])

  def test_unsupported_preset_without_jinja(self):
    kv = {"tokenizer.ggml.tokens": ["<unk>"], "tokenizer.ggml.token_type": [3], "tokenizer.ggml.pre": "dbrx"}
    tok = SimpleTokenizer.from_gguf_kv(kv)
    with self.assertRaises(ValueError):
      Chat.from_gguf_kv(kv, tok)   # "dbrx" not in _PRESETS and not olmo2 arch
    # works with use_jinja=True (no simple-preset check), as long as a chat_template is provided
    kv2 = {**kv, "tokenizer.chat_template": "x"}
    tok2 = SimpleTokenizer.from_gguf_kv(kv2)
    Chat.from_gguf_kv(kv2, tok2, use_jinja=True)

class TestChatJinja(unittest.TestCase):
  def setUp(self):
    try: import jinja2  # noqa: F401
    except ImportError: self.skipTest("jinja2 not installed")

  def test_render_simple_template(self):
    template = ("{%- for m in messages %}{%- if m.role == 'user' %}{{- '[INST]' + m.content + '[/INST]' }}"
                "{%- elif m.role == 'assistant' %}{{- m.content + '</s>' }}{%- endif %}{%- endfor %}")
    kv = {
      "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "hello"],
      "tokenizer.ggml.token_type": [3, 3, 3, 3, 3, 1],
      "tokenizer.ggml.pre": "tekken",
      "tokenizer.ggml.eos_token_id": 2,
      "tokenizer.chat_template": template,
    }
    tok = SimpleTokenizer.from_gguf_kv(kv)
    chat = Chat.from_gguf_kv(kv, tok, use_jinja=True)
    self.assertEqual(chat.apply([{"role": "user", "content": "hello"}]), [3, 5, 4])

  def test_jinja_requires_template(self):
    kv = {"tokenizer.ggml.tokens": ["<unk>"], "tokenizer.ggml.token_type": [3], "tokenizer.ggml.pre": "llama3"}
    tok = SimpleTokenizer.from_gguf_kv(kv)
    with self.assertRaises(ValueError):
      Chat.from_gguf_kv(kv, tok, use_jinja=True)   # no tokenizer.chat_template

if __name__ == '__main__':
  unittest.main()
