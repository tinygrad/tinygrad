import os
os.environ["WQKV"] = "1"
import unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from examples.mlperf.models.llama import Transformer
from examples.mlperf.models.flat_llama import FlatTransformer

def copy_weights(flat:FlatTransformer, ref:Transformer):
  n_layers = flat.n_layers
  Tensor.realize(*nn.state.get_state_dict(ref).values())
  flat.wqkv.assign(Tensor(np.stack([ref.layers[i].attention.wqkv.weight.numpy() for i in range(n_layers)])))
  flat.wo.assign(Tensor(np.stack([ref.layers[i].attention.wo.weight.numpy() for i in range(n_layers)])))
  flat.w13.assign(Tensor(np.stack([np.concatenate([ref.layers[i].feed_forward.w1.weight.numpy(),
                                                    ref.layers[i].feed_forward.w3.weight.numpy()], axis=0) for i in range(n_layers)])))
  flat.w2.assign(Tensor(np.stack([ref.layers[i].feed_forward.w2.weight.numpy() for i in range(n_layers)])))
  flat.attention_norm.assign(Tensor(np.stack([ref.layers[i].attention_norm.weight.numpy() for i in range(n_layers)])))
  flat.ffn_norm.assign(Tensor(np.stack([ref.layers[i].ffn_norm.weight.numpy() for i in range(n_layers)])))
  flat.norm.weight.assign(Tensor(ref.norm.weight.numpy()))
  flat.tok_embeddings.weight.assign(Tensor(ref.tok_embeddings.weight.numpy()))
  flat.output.weight.assign(Tensor(ref.output.weight.numpy()))

class TestFlatLlama(unittest.TestCase):
  def test_forward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens).realize()
    flat_logits = flat(tokens).realize()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    diff = (ref_logits - flat_logits).abs().max().item()
    self.assertLess(diff, 1e-5, f"forward mismatch: max abs diff {diff}")

  def test_backward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)

    for p in get_parameters(ref): p.requires_grad_(True)
    for p in get_parameters(flat): p.requires_grad_(True)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2, 10]])

    ref_loss = ref(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    ref_loss.backward()
    ref_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(ref).items() if v.grad is not None}

    flat_loss = flat(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    flat_loss.backward()
    flat_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(flat).items() if v.grad is not None}

    # check loss matches
    self.assertAlmostEqual(ref_loss.item(), flat_loss.item(), places=4)

    # check output weight grad matches
    diff = abs(ref_grads["output.weight"] - flat_grads["output.weight"]).max()
    self.assertLess(diff, 1e-4, f"output.weight grad mismatch: max abs diff {diff}")

    # check per-layer weight grads match
    for i in range(params["n_layers"]):
      for flat_key, ref_key in [
        ("wqkv", f"layers.{i}.attention.wqkv.weight"),
        ("wo", f"layers.{i}.attention.wo.weight"),
        ("w2", f"layers.{i}.feed_forward.w2.weight"),
      ]:
        diff = abs(ref_grads[ref_key] - flat_grads[flat_key][i]).max()
        self.assertLess(diff, 1e-4, f"layer {i} {flat_key} grad mismatch: max abs diff {diff}")

      # w13 grad = cat(w1.grad, w3.grad)
      ref_w13_grad = np.concatenate([ref_grads[f"layers.{i}.feed_forward.w1.weight"], ref_grads[f"layers.{i}.feed_forward.w3.weight"]], axis=0)
      diff = abs(ref_w13_grad - flat_grads["w13"][i]).max()
      self.assertLess(diff, 1e-4, f"layer {i} w13 grad mismatch: max abs diff {diff}")

if __name__ == "__main__":
  unittest.main()
