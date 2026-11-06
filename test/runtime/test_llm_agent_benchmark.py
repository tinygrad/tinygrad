import unittest
from types import SimpleNamespace
from test.external.external_benchmark_llm_agent import AgentBenchmark, coding_prompt


class TestAgentBenchmark(unittest.TestCase):
  def make_bench(self, first, following):
    def remaining(tokens, temperature):
      yield from following
    model = SimpleNamespace(max_context=32, _token_host=[-1, -1, first], spec_stats=[0, 0],
                            _checkpoint=lambda *a, **kw: None, _wait_host=lambda f, what: f(), _spec_steps=remaining)
    tokenizer = SimpleNamespace(is_end=lambda token: token == 99, decode=lambda tokens: str(tokens))
    bench = AgentBenchmark(model, tokenizer)
    bench.prompt, bench.position, bench.gen = [1, 2], 2, None
    bench.output, bench.ended, bench.end_reason = [], False, None
    return bench

  def test_eos_is_not_counted_or_followed(self):
    def tokens():
      yield 8
      yield 9
      yield 99
      raise AssertionError("consumed a token after EOS")
    bench = self.make_bench(7, tokens())
    result = bench.decode_step(128)
    self.assertEqual((result["tokens"], result["end_reason"]), (2, "eos"))
    self.assertEqual(bench.output, [7, 8, 9, 99])
    self.assertEqual(bench.decode_step(128)["tokens"], 0)

  def test_first_token_eos(self):
    result = self.make_bench(99, iter([7])).decode_step()
    self.assertEqual((result["tokens"], result["end_reason"]), (0, "eos"))

  def test_context_limit_is_not_eos(self):
    result = self.make_bench(7, iter([8, 9])).decode_step()
    self.assertEqual((result["tokens"], result["end_reason"]), (2, "context_limit"))

  def test_windows_do_not_drop_tokens(self):
    bench = self.make_bench(7, iter([8, 9, 10, 99]))
    self.assertEqual(bench.decode_step(2)["tokens"], 2)
    self.assertEqual(bench.decode_step(2)["tokens"], 1)
    self.assertEqual(bench.output, [7, 8, 9, 10, 99])

  def test_prompt_size_and_source(self):
    tokenizer = SimpleNamespace(encode=lambda text: list(map(ord, text)))
    prompt = coding_prompt(tokenizer, "sampling-review", 4096)
    self.assertEqual(len(prompt), 4096)
    text = ''.join(map(chr, prompt))
    self.assertIn("tinygrad/llm/model.py", text)
    self.assertIn("def _sample", text)
    self.assertIn("END REPOSITORY SNAPSHOT", text)
    with self.assertRaises(ValueError): coding_prompt(tokenizer, "sampling-review", 1)


if __name__ == "__main__": unittest.main()
