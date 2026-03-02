import unittest
from unittest.mock import patch
from tinygrad import Tensor

class TestTransformerGenerate(unittest.TestCase):
  def test_start_pos_parameter_is_used(self):
    """Test that start_pos parameter is not ignored (regression test for always resetting to 0)."""
    from tinygrad.apps.llm import Transformer
    from tinygrad.uop.ops import UOp
    # Create a minimal transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      s = tokens.shape[-1]
      shape_val = s.val if isinstance(s, UOp) else s
      sp_val = start_pos.val if isinstance(start_pos, UOp) else start_pos
      captured_inputs.append((shape_val, sp_val))
      return Tensor([[42]])  # return a fake next token

    with patch.object(Transformer, '__call__', mock_call):
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens, start_pos=3)
      next(gen)  # get first token

    # With start_pos=3, the initial tensor should only have tokens[3:] = [4, 5] (length 2)
    # If the bug existed (start_pos always reset to 0), it would have all 5 tokens
    self.assertEqual(captured_inputs[0][0], 2)  # shape should have 2 tokens
    self.assertEqual(captured_inputs[0][1], 3)  # start_pos should be 3, not 0

class TestPrefillCorrectness(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.apps.llm import Transformer
    cls.model = Transformer(num_blocks=2, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                            norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=128)

  def _reset_caches(self, model):
    for block in model.blk:
      if hasattr(block, "cache_kv"): del block.cache_kv
      if hasattr(block, "causal_mask"): del block.causal_mask

  def test_prefill_matches_forward(self):
    """Test that prefill_forward produces the same output as forward for various lengths."""
    model = self.__class__.model
    v_prefill = model.v_prefill
    token_buf = Tensor.zeros(1, model.max_context, dtype="int32").contiguous().realize()
    for length in [2, 4, 8, 16]:
      tokens = Tensor.randint(length, high=100, dtype="int32")
      token_buf[0, :length].assign(tokens).realize()

      self._reset_caches(model)
      expected = model.forward(tokens.reshape(1, -1), 0).realize()

      self._reset_caches(model)
      actual = model.prefill_forward(token_buf[:, :v_prefill.bind(length)], 0).realize()

      self.assertEqual(expected.item(), actual.item(), f"mismatch at length={length}: forward={expected.item()}, prefill={actual.item()}")

  def test_prefill_then_decode_matches_sequential(self):
    """Test that prefill + decode produces the same sequence as token-by-token forward."""
    from tinygrad import UOp, getenv
    model = self.__class__.model
    tokens = Tensor.randint(6, high=100, dtype="int32").tolist()
    num_decode = 3

    # reference: feed all tokens one at a time, then decode
    self._reset_caches(model)
    ref_ids = list(tokens)
    t = Tensor([ref_ids], dtype="int32")
    t = model.forward(t, 0)
    ref_ids.append(int(t.item()))
    for i in range(num_decode - 1):
      sp = len(ref_ids) - 1
      t = model.forward(t, sp)
      ref_ids.append(int(t.item()))

    # test: prefill then decode
    self._reset_caches(model)
    test_ids = list(tokens)
    token_buf = Tensor.zeros(1, model.max_context, dtype="int32").contiguous().realize()
    token_buf[0, :len(test_ids)].assign(Tensor(test_ids, dtype="int32")).realize()
    t = model.prefill_forward(token_buf[:, :model.v_prefill.bind(len(test_ids))], 0)
    test_ids.append(int(t.item()))
    for i in range(num_decode - 1):
      sp = len(test_ids) - 1
      t = model.forward(t, sp)
      test_ids.append(int(t.item()))

    self.assertEqual(ref_ids, test_ids, f"sequence mismatch: ref={ref_ids[-num_decode:]}, test={test_ids[-num_decode:]}")

  def test_prefill_with_start_pos(self):
    """Test that prefill with start_pos > 0 matches sequential forward."""
    model = self.__class__.model
    first_tokens = Tensor.randint(4, high=100, dtype="int32").tolist()
    second_tokens = Tensor.randint(3, high=100, dtype="int32").tolist()
    all_tokens = first_tokens + second_tokens

    # reference: forward all at once from pos 0
    self._reset_caches(model)
    expected = model.forward(Tensor([all_tokens], dtype="int32"), 0).realize()

    # test: forward first chunk, then prefill second chunk at start_pos
    self._reset_caches(model)
    model.forward(Tensor([first_tokens], dtype="int32"), 0).realize()
    token_buf = Tensor.zeros(1, model.max_context, dtype="int32").contiguous().realize()
    token_buf[0, :len(second_tokens)].assign(Tensor(second_tokens, dtype="int32")).realize()
    actual = model.prefill_forward(token_buf[:, :model.v_prefill.bind(len(second_tokens))],
                                   model.v_start_pos.bind(len(first_tokens))).realize()

    self.assertEqual(expected.item(), actual.item(),
                     f"start_pos mismatch: all-at-once={expected.item()}, split={actual.item()}")

  def test_multi_turn_generate(self):
    """Test that multi-turn generate produces same tokens as reference (token-by-token forward)."""
    model = self.__class__.model
    num_decode = 3

    # simulate 2 turns of conversation
    turn1_tokens = Tensor.randint(5, high=100, dtype="int32").tolist()
    turn2_tokens = Tensor.randint(4, high=100, dtype="int32").tolist()

    # reference: token-by-token forward (no prefill, no JIT)
    self._reset_caches(model)
    ref_ids = list(turn1_tokens)
    # prefill turn 1
    t = model.forward(Tensor([ref_ids], dtype="int32"), 0)
    ref_ids.append(int(t.item()))
    # decode turn 1
    for _ in range(num_decode - 1):
      t = model.forward(t, len(ref_ids) - 1)
      ref_ids.append(int(t.item()))
    # prefill turn 2
    start_pos_2 = len(ref_ids)
    ref_ids += turn2_tokens
    t = model.forward(Tensor([ref_ids[start_pos_2:]], dtype="int32"), start_pos_2)
    ref_ids.append(int(t.item()))
    # decode turn 2
    for _ in range(num_decode - 1):
      t = model.forward(t, len(ref_ids) - 1)
      ref_ids.append(int(t.item()))

    # test: using prefill_forward with symbolic variable
    self._reset_caches(model)
    test_ids = list(turn1_tokens)
    token_buf = Tensor.zeros(1, model.max_context, dtype="int32").contiguous().realize()
    # prefill turn 1
    token_buf[0, :len(test_ids)].assign(Tensor(test_ids, dtype="int32")).realize()
    t = model.prefill_forward(token_buf[:, :model.v_prefill.bind(len(test_ids))], 0)
    test_ids.append(int(t.item()))
    # decode turn 1
    for _ in range(num_decode - 1):
      t = model.forward(t, len(test_ids) - 1)
      test_ids.append(int(t.item()))
    # prefill turn 2
    start_pos_2 = len(test_ids)
    test_ids += turn2_tokens
    new_tokens = test_ids[start_pos_2:]
    token_buf[0, :len(new_tokens)].assign(Tensor(new_tokens, dtype="int32")).realize()
    t = model.prefill_forward(token_buf[:, :model.v_prefill.bind(len(new_tokens))],
                              model.v_start_pos.bind(start_pos_2))
    test_ids.append(int(t.item()))
    # decode turn 2
    for _ in range(num_decode - 1):
      t = model.forward(t, len(test_ids) - 1)
      test_ids.append(int(t.item()))

    self.assertEqual(ref_ids, test_ids, f"multi-turn mismatch:\nref ={ref_ids}\ntest={test_ids}")

  def test_generate_matches_forward(self):
    """Test that generate() (with symbolic prefill + JIT) matches token-by-token forward for 3 turns."""
    from tinygrad import TinyJit
    model = self.__class__.model
    num_decode = 3
    turns = [Tensor.randint(5, high=100, dtype="int32").tolist() for _ in range(3)]

    # reference: token-by-token forward
    self._reset_caches(model)
    ref_ids: list[int] = []
    for turn_tokens in turns:
      start_pos = len(ref_ids)
      ref_ids += turn_tokens
      t = model.forward(Tensor([ref_ids[start_pos:]], dtype="int32"), start_pos)
      ref_ids.append(int(t.item()))
      for _ in range(num_decode - 1):
        t = model.forward(t, len(ref_ids) - 1)
        ref_ids.append(int(t.item()))

    # test: using generate() with symbolic prefill
    self._reset_caches(model)
    model.forward_jit = TinyJit(model.forward)
    model.prefill_jit = TinyJit(model.prefill_forward)
    test_ids: list[int] = []
    for turn_tokens in turns:
      start_pos = len(test_ids)
      test_ids += turn_tokens
      gen = model.generate(test_ids, start_pos)
      for _ in range(num_decode): next(gen)

    self.assertEqual(ref_ids, test_ids, f"generate mismatch:\nref ={ref_ids}\ntest={test_ids}")

if __name__ == '__main__':
  unittest.main()
