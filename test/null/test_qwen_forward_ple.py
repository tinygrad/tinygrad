# qwen4exp Transformer._forward's prelude (the embedding rows, then the n-gram rows of the token buffer): the assertion that the tokens are a view
# of the token buffer must hold for the seq prefill chunk (a symbolic slice of the buffer at a variable base) and for a plain slice. DEV=NULL
import types, unittest
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.llm.qwen4exp import Transformer

class Reached(Exception): pass

def _model(hc:bool) -> Transformer:
  m = Transformer.__new__(Transformer)
  m.token_embd = nn.Embedding(512, 64)
  m.config = types.SimpleNamespace(hc=types.SimpleNamespace(count=4) if hc else None, ple=types.SimpleNamespace(multipliers=(1, 2, 3), eos=0))
  m._token_buf = Tensor.empty(1, 4096, dtype=dtypes.int32)
  def window(buf, base, start, n): raise Reached  # past the assertion: the rows of the n-gram table are looked up here
  m._token_window = window
  return m

class TestForwardPle(unittest.TestCase):
  def test_prefill_chunk_symbolic_view(self):
    # (_seq_chunk: self(token_buf[:, vb+sp:vb+sp+nt], sp, temp, vb, part))
    for hc in (False, True):
      m = _model(hc)
      vb, sp = UOp.variable("seq_base", 0, 4096).bind(100), UOp.variable("start_pos", 0, 4095).bind(8)
      nt = UOp.variable("toks", 1, 64).bind(40)
      with self.subTest(hc=hc), self.assertRaises(Reached):
        m._forward(m._token_buf[:, vb+sp:vb+sp+nt], sp, None, vb)

  def test_plain_view(self):
    m = _model(False)
    with self.assertRaises(Reached): m._forward(m._token_buf[:, 0:64], 0, None, 0)

  def test_not_a_view_of_the_buffer_is_refused(self):
    m = _model(False)
    with self.assertRaisesRegex(AssertionError, "n-gram rows"): m._forward(Tensor.empty(1, 64, dtype=dtypes.int32), 0, None, 0)

if __name__ == "__main__": unittest.main()
