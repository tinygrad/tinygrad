import unittest
from tinygrad import Tensor, Device, nn, GlobalCounters
from tinygrad.helpers import CI
from tinygrad.nn.state import get_parameters
from extra.lr_scheduler import OneCycleLR
from extra.models.llama import RMSNorm, Attention
import numpy as np

d_zero = f"{Device.DEFAULT}:0"
d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
d2, d3 = f"{Device.DEFAULT}:3", f"{Device.DEFAULT}:4"
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

@unittest.skipIf(CI and Device.DEFAULT in {"GPU", "CUDA", "METAL"}, "no GPU CI")
class TestMultiTensor(unittest.TestCase):
  def test_shard(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d1), 0)
    for lb in X.lazydata.lbs:
      assert lb.shape == (128,)

  def test_shard_same_device(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, X.device), 0)
    (X + X).realize()

  def test_shard_plus_one_sum(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_([d0, d1], 0)
    (X + 1).sum().realize()

  def test_shard_plus_one_sum_d_zero(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_([d_zero, d1], 0)
    (X + 1).sum().realize()

  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d0, d1), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def _test_simple_add_axis(self, shard_x, shard_w):
    X = Tensor.ones(256).contiguous().realize()
    W = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    W.shard_((d0, d1), shard_w)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_simple_add(self): return self._test_simple_add_axis(None, None)
  def test_simple_add_X(self): return self._test_simple_add_axis(0, None)
  def test_simple_add_W(self): return self._test_simple_add_axis(None, 0)
  def test_simple_add_XW(self): return self._test_simple_add_axis(0, 0)

  def test_four_add(self):
    X = Tensor.ones(256, 256).contiguous().realize()
    W = Tensor.ones(256, 256).contiguous().realize()
    X.shard_((d0, d1, d2, d3), 1)
    W.shard_((d0, d1, d2, d3), None)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def _test_simple_reduce_axis(self, shard_x):
    X = Tensor.ones(256, 256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    O = X.sum(axis=1)
    np.testing.assert_allclose(O.numpy(), 256)

  def test_simple_reduce(self): return self._test_simple_reduce_axis(None)
  def test_simple_reduce_0(self): return self._test_simple_reduce_axis(0)
  def test_simple_reduce_1(self): return self._test_simple_reduce_axis(1)

  def _test_matmul_shard_axis(self, shard_x, shard_w):
    X = Tensor.kaiming_uniform(N, N).realize()
    W = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard((d0, d1), shard_x)
    Ws = W.shard((d0, d1), shard_w)
    O = (Xs@Ws)
    np.testing.assert_allclose(X.numpy() @ W.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def _test_double_matmul_shard_axis(self, shard_x, shard_w):
    X = Tensor.kaiming_uniform(N, N).realize()
    W1 = Tensor.kaiming_uniform(N, N).realize()
    W2 = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard((d0, d1), shard_x)
    W1s = W1.shard((d0, d1), shard_w)
    W2s = W2.shard((d0, d1), shard_w)
    O = (Xs@W1s)@W2s
    np.testing.assert_allclose((X.numpy() @ W1.numpy()) @ W2.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def test_matmul_shard_none(self): return self._test_matmul_shard_axis(None, None)
  def test_matmul_shard_X_0(self): return self._test_matmul_shard_axis(0, None)
  def test_matmul_shard_X_1(self): return self._test_matmul_shard_axis(1, None)
  def test_matmul_shard_W_0(self): return self._test_matmul_shard_axis(None, 0)
  def test_matmul_shard_W_1(self): return self._test_matmul_shard_axis(None, 1)

  def test_matmul_shard_0_0(self): return self._test_matmul_shard_axis(0, 0)
  def test_matmul_shard_0_1(self): return self._test_matmul_shard_axis(0, 1)
  def test_matmul_shard_1_0(self): return self._test_matmul_shard_axis(1, 0)
  def test_matmul_shard_1_1(self): return self._test_matmul_shard_axis(1, 1)

  def test_double_matmul_shard_X_0(self): return self._test_double_matmul_shard_axis(0, None)
  def test_double_matmul_shard_X_1(self): return self._test_double_matmul_shard_axis(1, None)
  def test_double_matmul_shard_W_0(self): return self._test_double_matmul_shard_axis(None, 0)
  def test_double_matmul_shard_W_1(self): return self._test_double_matmul_shard_axis(None, 1)

  def test_conv_data_shard(self):
    conv = nn.Conv2d(3, 16, 3, bias=False)
    for p in get_parameters(conv): p.shard_((d0, d1))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_conv_bias_data_shard(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_backprop_conv(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    optim = nn.optim.Adam(get_parameters(conv))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    optim.zero_grad()
    out.mean().backward()
    #for p in get_parameters(conv): p.grad.realize()
    optim.step()

  def test_lr_scheduler_OneCycleLR(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    optim = nn.optim.SGD(get_parameters(conv))
    lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
    lr_sched.step()

  def test_embedding(self):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    layer = nn.Embedding(vocab_size, embed_size)
    x = Tensor(np.random.randint(0, vocab_size, (B, T)))
    z = layer(x)

    layer_sharded = nn.Embedding(vocab_size, embed_size)
    layer_sharded.weight.assign(layer.weight.shard((d0, d1), axis=1)).realize()
    x_sharded = x.shard((d0, d1), axis=None)
    z_shard = layer_sharded(x_sharded)

    np.testing.assert_allclose(z.numpy(), z_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_rmsnorm(self):
    B, T, embed_size = 4, 10, 20

    layer_norm = RMSNorm(embed_size)
    x = Tensor.rand((B, T, embed_size)).contiguous().realize()
    y = layer_norm(x)

    # for norm layers, the correct way to shard weights is duplication
    layer_norm_sharded = RMSNorm(embed_size)
    layer_norm_sharded.weight.shard_((d0, d1), axis=None).realize()

    # if x is being sharded, then all-reduce is involved
    x_sharded = x.shard((d0, d1), axis=2).realize()
    y_shard = layer_norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

    # if x is being duplicated, then the operations remain inside each GPU
    # which is the common case
    x_sharded = x.shard((d0, d1), axis=None).realize()
    y_shard = layer_norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

  @unittest.skipIf(Device.DEFAULT == "LLVM", "LLVM segmentation fault")
  @unittest.skipIf(Device.DEFAULT == "GPU", "GPU requires cl_khr_fp16")
  def test_llama_attention(self):
    bs = 1
    seq_len = 1
    dim = 128
    n_heads = 4
    n_kv_heads = 4
    max_context = 32

    freqs_cis = Tensor.rand(1, seq_len, 1, (dim//n_heads)//2, 2).half()
    mask = None
    start_pos = 0

    layer = Attention(dim, n_heads, n_kv_heads, max_context, linear=nn.Linear)
    x = Tensor.rand(bs, seq_len, dim).half()
    y = layer(x, start_pos, freqs_cis, mask).realize()

    layer_sharded = Attention(dim, n_heads, n_kv_heads, max_context, linear=nn.Linear)
    layer_sharded.wq.weight.assign(layer.wq.weight.shard((d0, d1), axis=0)).realize()
    layer_sharded.wk.weight.assign(layer.wk.weight.shard((d0, d1), axis=0)).realize()
    layer_sharded.wv.weight.assign(layer.wv.weight.shard((d0, d1), axis=0)).realize()
    layer_sharded.wo.weight.assign(layer.wo.weight.shard((d0, d1), axis=0)).realize()
    x_sharded = x.shard((d0, d1), axis=None).realize()
    freqs_cis_sharded = freqs_cis.shard((d0, d1), axis=None).realize()
    y_sharded = layer_sharded(x_sharded, start_pos, freqs_cis_sharded, mask)

    np.testing.assert_allclose(y.numpy(), y_sharded.numpy(), atol=1e-6, rtol=1e-6)

  def test_data_parallel_resnet(self):
    import sys, pathlib
    sys.path.append((pathlib.Path(__file__).parent.parent / "extra" / "models").as_posix())
    from resnet import ResNet18

    fake_image = Tensor.rand((2, 3, 224, 224))
    fake_image_sharded = fake_image.shard((d0, d1), axis=0)
    print(fake_image_sharded.shape)
    m = ResNet18()
    m.load_from_pretrained()
    real_output = m(fake_image).numpy()
    for p in get_parameters(m): p.shard_((d0, d1)).realize()
    GlobalCounters.reset()
    shard_output = m(fake_image_sharded).realize()
    assert shard_output.lazydata.lbs[0].shape == (1, 1000)
    assert shard_output.lazydata.lbs[1].shape == (1, 1000)
    shard_output_np = shard_output.numpy()
    np.testing.assert_allclose(real_output, shard_output_np, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  unittest.main()