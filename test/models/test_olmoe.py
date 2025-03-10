import unittest
from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported
from examples.olmoe import Transformer, functools, MixtureFeedForward, fetch_weights, convert_from_huggingface, Tensor, nn

class TestOLMoe(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "need dtypes.bfloat16")
  def test_olmoe(self):
    model = Transformer(n_layers=16, dim=2048, hidden_dim=1024, n_heads=16, norm_eps=1e-5, qk_norm=1e-5, max_context=1024,
                        vocab_size=50304, feed_forward=functools.partial(MixtureFeedForward, 64, 8))
    model_state_dict = nn.state.get_state_dict(model)
    del model_state_dict['freqs_cis']

    state = fetch_weights()
    nhf_state = convert_from_huggingface(state, model, 16, 16)
    # NOTE: i'm not sure this actually needs float32, it may just change the type of things downstream from it. but doesn't match torch w/o this
    for needs_float32 in ['tok_embeddings.weight']: nhf_state[needs_float32] = nhf_state[needs_float32].float()
    nn.state.load_state_dict(model, nhf_state, verbose=False, strict=False, consume=True, realize=False)
    assert len(nhf_state) == 0

    count = 30
    temperature = 0

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    toks = [12092]
    start_pos = 0
    for i in range(count):
      tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
      toks.append(tok)
      start_pos += 1
      print(toks)
      print(tokenizer.decode(toks))

    # Hello, I am a newbie to this forum and I am trying to get a better understanding of the different types of data that can be stored in a
    assert toks == [12092, 13, 309, 717, 247, 747, 17782, 281, 436, 12209, 285, 309, 717, 2820, 281, 755,
                    247, 1805, 4685, 273, 253, 1027, 3510, 273, 941, 326, 476, 320, 7141, 275, 247], "BAD OUTPUT!"