import math
import sys
import unittest
from pathlib import Path

if __name__ == "__main__": sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters

from examples.mlperf.models.flux import EmbedND, Flux, QKNorm, apply_rope, mse_loss, rectified_flow_inputs, rope, timestep_embedding


ATOL_EXACT = 1e-6
ATOL_MODEL = 1e-5
ATOL_BF16 = 8e-3

TORCHTITAN_BF16_OUTPUT = Tensor([[[1.1484375, 0.0177001953125, -0.58984375, -0.4296875, 0.5234375, -0.0230712890625, -0.51171875, -0.482421875],
                                  [0.8046875, 0.09423828125, -0.3359375, -0.3359375, 0.205078125, -0.12109375, -0.322265625, -0.26171875],
                                  [0.66015625, 0.1025390625, -0.25, -0.2890625, 0.0966796875, -0.1572265625, -0.2578125, -0.1689453125]]])
TORCHTITAN_BF16_GRAD_FIRST8 = (
  Tensor([-0.408203125, 0.1181640625, -0.373046875, 0.10546875, -0.2265625, 0.130859375, -0.2119140625, -0.0135498046875]),
  Tensor([-0.00970458984375, -0.00848388671875, -0.0072021484375, -0.005950927734375,
          -0.00469970703125, -0.0034027099609375, -0.002166748046875, -0.00093841552734375]),
  Tensor([-4.586763679981232e-08, 1.659827830735594e-11, -4.400499165058136e-08, 2.735760062932968e-09,
          -3.632158041000366e-08, 5.005858838558197e-09, -2.7939677238464355e-08, 7.159542292356491e-09]),
)
TORCHTITAN_BF16_GRAD_MAX = (0.52734375, 0.15234375, 0.004364013671875)


def small_flux(guidance_embed:bool=False) -> Flux:
  return Flux(guidance_embed=guidance_embed, in_channels=8, out_channels=8, vec_in_dim=12, context_in_dim=10,
              hidden_size=64, mlp_ratio=2.0, num_heads=4, depth=1, depth_single_blocks=1, axes_dim=(4, 6, 6))


def inputs(batch:int=1):
  img = Tensor.arange(batch * 3 * 8, dtype=dtypes.float32).reshape(batch, 3, 8) / 20
  txt = Tensor.arange(batch * 2 * 10, dtype=dtypes.float32).reshape(batch, 2, 10) / 30
  img_ids = Tensor([[[0, 0, 0], [0, 1, 0], [0, 1, 1]]] * batch)
  txt_ids = Tensor([[[0, 0, 0], [1, 0, 0]]] * batch)
  timesteps = Tensor([0.25] * batch)
  y = Tensor.arange(batch * 12, dtype=dtypes.float32).reshape(batch, 12) / 10
  return img, img_ids, txt, txt_ids, timesteps, y


class TestFluxPrimitives(unittest.TestCase):
  def test_rope_orientation_and_embed_shape(self):
    pe = rope(Tensor([[0.0, math.pi / 2]]), dim=2, theta=10_000)
    self.assertEqual(pe.shape, (1, 2, 1, 2, 2))
    self.assertTrue(pe[0, 0, 0].allclose(Tensor([[1.0, 0.0], [0.0, 1.0]]), atol=ATOL_EXACT, rtol=0).item())
    q = Tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    qr, kr = apply_rope(q, q, pe.unsqueeze(1))
    expected = Tensor([[[[[1.0, 0.0], [-1.0, 0.0]]]]])
    self.assertTrue(qr.allclose(expected, atol=ATOL_EXACT, rtol=0).item())
    self.assertTrue(kr.allclose(expected, atol=ATOL_EXACT, rtol=0).item())

    ids = Tensor([[[0, 0, 0], [1, 2, 3]]])
    embedded = EmbedND(dim=16, theta=10_000, axes_dim=(4, 6, 6))(ids)
    self.assertEqual(embedded.shape, (1, 1, 2, 8, 2, 2))
    self.assertTrue(embedded[0, 0, 0].allclose(Tensor.eye(2).reshape(1, 2, 2).expand(8, 2, 2),
                                                        atol=ATOL_EXACT, rtol=0).item())

  def test_timestep_embedding_analytic(self):
    got = timestep_embedding(Tensor([0.0, 0.001]), 5, max_period=100, time_factor=1000)
    self.assertEqual(got.shape, (2, 5))
    self.assertTrue(got[0].allclose(Tensor([1, 1, 0, 0, 0]), atol=ATOL_EXACT, rtol=0).item())
    expected = Tensor([math.cos(1), math.cos(0.1), math.sin(1), math.sin(0.1), 0])
    self.assertTrue(got[1].allclose(expected, atol=ATOL_EXACT, rtol=0).item())

  def test_qknorm_casts_to_value_dtype(self):
    q, k = QKNorm(4)(Tensor([[1.0, 2.0, 3.0, 4.0]]), Tensor([[4.0, 3.0, 2.0, 1.0]]), Tensor.zeros(1, 4, dtype=dtypes.float16))
    self.assertEqual(q.dtype, dtypes.float16)
    self.assertEqual(k.dtype, dtypes.float16)

  def test_invalid_configurations(self):
    with self.assertRaisesRegex(ValueError, "divisible"):
      Flux(False, hidden_size=62, num_heads=4, axes_dim=(4, 6, 6))
    with self.assertRaisesRegex(ValueError, "even"):
      Flux(False, hidden_size=64, num_heads=4, axes_dim=(3, 7, 6))
    with self.assertRaisesRegex(ValueError, "positional dim"):
      Flux(False, hidden_size=64, num_heads=4, axes_dim=(4, 4, 4))

  def test_rectified_flow_and_fp32_mse(self):
    data, noise, t = Tensor([[[2.0, -2.0]]]), Tensor([[[6.0, 2.0]]]), Tensor([0.25])
    x_t, target = rectified_flow_inputs(data, noise, t)
    self.assertTrue(x_t.allclose(Tensor([[[3.0, -1.0]]]), atol=ATOL_EXACT, rtol=0).item())
    self.assertTrue(target.allclose(Tensor([[[4.0, 4.0]]]), atol=ATOL_EXACT, rtol=0).item())
    loss = mse_loss(Tensor([[[2.0, 0.0]]], dtype=dtypes.float16), target.cast(dtypes.float16))
    self.assertEqual(loss.dtype, dtypes.float32)
    self.assertAlmostEqual(loss.item(), 10.0, delta=ATOL_EXACT)


class TestFluxArchitecture(unittest.TestCase):
  def test_validation_guidance_and_mlperf_zero_output(self):
    model = small_flux()
    args = inputs()
    out = model(*args)
    self.assertEqual(out.shape, (1, 3, 8))
    self.assertEqual(out.abs().max().item(), 0.0)
    with self.assertRaisesRegex(ValueError, "guidance"):
      small_flux(guidance_embed=True)(*args)
    guided = small_flux(guidance_embed=True)(*args, guidance=Tensor([3.5]))
    self.assertEqual(guided.abs().max().item(), 0.0)
    bad = list(args)
    bad[1] = Tensor.zeros(1, 2, 3)
    with self.assertRaisesRegex(ValueError, "token"):
      model(*bad)
    bad = list(args)
    bad[3] = Tensor.zeros(1, 2, 2)
    with self.assertRaisesRegex(ValueError, "axes"):
      model(*bad)

  def test_zero_init_masks_upstream_but_not_final_projection_gradient(self):
    model = small_flux()
    args = inputs()
    target = Tensor.ones(1, 3, 8)
    loss = mse_loss(model(*args), target)
    loss.backward()
    final_grad, upstream_grad = model.final_layer.linear.weight.grad, model.img_in.weight.grad
    self.assertIsNotNone(final_grad)
    self.assertIsNotNone(upstream_grad)
    assert final_grad is not None and upstream_grad is not None
    self.assertTrue(math.isfinite(loss.item()))
    self.assertGreater(final_grad.abs().max().item(), 0.0)
    self.assertEqual(upstream_grad.abs().max().item(), 0.0)

  def test_unmasked_deterministic_output_and_upstream_gradients(self):
    Tensor.manual_seed(7)
    model = small_flux()
    params = get_parameters(model)
    for i, param in enumerate(params):
      vals = ((Tensor.arange(param.numel(), dtype=dtypes.float32).reshape(param.shape) + i) % 17 - 8) / 200
      param.assign(vals.cast(param.dtype)).realize()
    args = inputs()
    out1, out2 = model(*args).realize(), model(*args).realize()
    self.assertTrue(out1.allclose(out2, atol=ATOL_MODEL, rtol=0).item())
    self.assertGreater(out1.abs().max().item(), 0.0)
    # Build the gradient graph after the deterministic inference realizations.
    loss = mse_loss(model(*args), Tensor.zeros(out1.shape))
    final_grad, img_grad, block_grad = loss.gradient(model.final_layer.linear.weight, model.img_in.weight,
                                                     model.double_blocks[0].img_attn.qkv.weight)
    for name, grad in (("final", final_grad), ("image input", img_grad), ("double block", block_grad)):
      self.assertTrue(math.isfinite(grad.abs().max().item()), name)
      self.assertGreater(grad.abs().max().item(), 0.0, name)

  def test_pinned_torchtitan_bfloat16_artifact(self):
    # Generated with torch 2.9.1+cpu from pinned TorchTitan commit 9603aa83, using shape-only deterministic weights.
    model = small_flux()
    params = get_parameters(model)
    for param in params: param.replace(param.cast(dtypes.bfloat16)).realize()
    for param in params:
      vals = ((Tensor.arange(param.numel(), dtype=dtypes.float32).reshape(param.shape) % 17) - 8) / 200
      param.assign(vals.cast(param.dtype)).realize()

    args = inputs()
    out = model(*args)
    loss = mse_loss(out, Tensor.zeros(out.shape))
    grads = loss.gradient(model.final_layer.linear.weight, model.img_in.weight, model.double_blocks[0].img_attn.qkv.weight)
    self.assertTrue(out.float().allclose(TORCHTITAN_BF16_OUTPUT, atol=ATOL_BF16, rtol=0).item())
    self.assertAlmostEqual(loss.item(), 0.18556304275989532, delta=1.5e-3)
    for name, grad, expected, expected_max, atol in zip(("final", "image input", "double block"), grads,
                                                        TORCHTITAN_BF16_GRAD_FIRST8, TORCHTITAN_BF16_GRAD_MAX,
                                                        (4e-3, 8e-4, 4e-5)):
      self.assertTrue(grad.float().flatten()[:8].allclose(expected, atol=atol, rtol=0).item(), name)
      self.assertAlmostEqual(grad.float().abs().max().item(), expected_max, delta=atol, msg=name)

    pe = rope(Tensor([[0, 1, 2]], dtype=dtypes.bfloat16), 4, 10_000)
    expected_pe = Tensor([[[[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                           [[[0.5390625, -0.83984375], [0.83984375, 0.5390625]], [[1, -0.010009765625], [0.010009765625, 1]]],
                           [[[-0.416015625, -0.91015625], [0.91015625, -0.416015625]], [[1, -0.02001953125], [0.02001953125, 1]]]]])
    self.assertTrue(pe.allclose(expected_pe, atol=3e-4, rtol=0).item())
    norm = QKNorm(4)
    norm.query_norm.weight = Tensor.ones(4, dtype=dtypes.bfloat16)
    norm.key_norm.weight = Tensor.ones(4, dtype=dtypes.bfloat16)
    q = Tensor([[0.001, 0.002, 0.003, 0.004]], dtype=dtypes.bfloat16)
    q, _ = norm(q, q, Tensor.zeros(1, 4, dtype=dtypes.bfloat16))
    self.assertTrue(q.float().allclose(Tensor([[0.361328125, 0.72265625, 1.0859375, 1.4453125]]), atol=0, rtol=0).item())


if __name__ == "__main__":
  unittest.main()
