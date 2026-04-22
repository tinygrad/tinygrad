# Reproduces the bug from err_log.txt:
#   ValueError: size mismatch, can't reshape ((4105,)) -> ((1, 4096))
#
# The bug: when the chat history grows larger than max_context, model.generate()
# blows up at tinygrad/llm/model.py:409 because `[0] * (max_context - len(tokens))`
# is empty (len negative), so the final Tensor has len(tokens) elements instead
# of max_context, and the reshape to (1, max_context) fails.
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig

MAX_CTX = 32
cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0,
                        rope_dim=32, v_head_dim=32, max_context=MAX_CTX)
model = Transformer(cfg)

# simulate a long chat: tokens exceed max_context (mirrors the 4105 > 4096 case)
tokens = [1] * (MAX_CTX + 9)
print(f"tokens={len(tokens)} max_context={MAX_CTX}")
next(model.generate(tokens))  # ValueError: size mismatch, can't reshape ((41,)) -> ((1, 32))
