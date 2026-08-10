import unittest
from tinygrad.llm.cli import KimiK3Template
from tinygrad.llm.serve import StreamRouter

class TestKimiK3Template(unittest.TestCase):
  def test_simple_text_chat(self):
    template = KimiK3Template()
    got = template.render([{"role":"system", "content":"Be concise."}, {"role":"user", "content":"Hello"}])
    self.assertTrue(got.startswith('<|open|>message role="system" type="thinking-effort"<|sep|>'))
    self.assertIn('<|open|>message role="user"<|sep|>Hello<|close|>message<|sep|><|end_of_msg|>', got)
    self.assertTrue(got.endswith('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'))

  def test_preserves_assistant_thinking(self):
    got = KimiK3Template().render([{"role":"assistant", "reasoning_content":"why", "content":"answer"}], add_generation_prompt=False)
    self.assertIn('<|open|>think<|sep|>why<|close|>think<|sep|>', got)
    self.assertIn('<|open|>response<|sep|>answer<|close|>response<|sep|>', got)

  def test_rejects_unimplemented_modalities(self):
    with self.assertRaisesRegex(ValueError, "text-only"):
      KimiK3Template().render([{"role":"user", "content":[{"type":"image", "url":"x"}]}])
    with self.assertRaisesRegex(ValueError, "tool rendering"):
      KimiK3Template().render([{"role":"user", "content":"x"}], tools=[{"type":"function"}])

  def test_xtml_stream_router(self):
    router, routed = StreamRouter(reasoning=True, xtml=True), []
    for piece in ("rea", "son<|close|>thi", "nk<|sep|><|open|>response<|sep|>ans", "wer<|close|>response<|sep|>"):
      routed.extend(router.route(piece))
    self.assertEqual(routed, [("reasoning_content", "rea"), ("reasoning_content", "son"), ("content", "ans"), ("content", "wer")])

if __name__ == "__main__": unittest.main()
