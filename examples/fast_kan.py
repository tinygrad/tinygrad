# model based off https://arxiv.org/abs/2405.06721
from tinygrad import Device, TinyJit, Tensor, nn, GlobalCounters
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import getenv, colored, trange
import tinygrad.function as F

import numpy as np

from typing import *

class SplineLinearFunction:
  def __init__(
    self,
    in_features: int, out_features: int, init_scale: float = 0.1
  ):
    self.init_scale = init_scale
    self.linear_function = nn.Linear(in_features, out_features, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.linear_function(x)

class RadialBasisFunction:
  def __init__(
    self,
    grid_min = -2.,
    grid_max = 2.,
    num_grids = 8,
    denominator = None
  ):
    self.grid_min = grid_min
    self.grid_max = grid_max
    self.num_grids = num_grids
    # You don't need a special Parameter initialization here.
    # You can initialize a Tensor later with this
    self.grid = Tensor(np.linspace(grid_min, grid_max, num_grids, dtype=np.float32), requires_grad=True)
    self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

  def __call__(self, x:Tensor) -> Tensor:
    return (-(((x[..., None] - self.grid) / self.denominator).pow(2))).exp()

class FastKANLayer:
  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    grid_min: float = -2.,
    grid_max: float = 2.,
    num_grids: int = 8,
    use_base_update: bool = True,
    use_layernorm: bool = True,
    base_activation = Tensor.silu,
    spline_weight_init_scale: float = 0.1
  ) -> None:
    self.input_dim = input_dim
    self.output_dim = output_dim
    # normally you'd init layernorm here.
    # but because layernorm *isn't* a layer in tinygrad,
    # it's a function, I'm gonna hold off until the call
    self.layernorm = None
    if use_layernorm:
      assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
      self.layernorm = nn.LayerNorm(input_dim)
    self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
    self.spline_linear = SplineLinearFunction(input_dim * num_grids, output_dim, spline_weight_init_scale)
    self.use_base_update = use_base_update
    if use_base_update:
      self.base_activation = base_activation
      self.base_linear = nn.Linear(input_dim, output_dim)

  def __call__(
    self, x: Tensor, use_layernorm=True
  ) -> Tensor:
    if self.layernorm is not None and use_layernorm:
      spline_basis = self.rbf(self.layernorm(x))
    else:
      spline_basis = self.rbf(x)
    spline_basis_view = spline_basis.view(*spline_basis.shape[:-2], -1)
    ret = self.spline_linear(spline_basis_view)
    if self.use_base_update:
      base = self.base_linear(self.base_activation(x))
      ret = ret + base
    return ret

class FastKAN:
  def __init__(
    self,
    layers_hidden: List[int],
    grid_min: float = -2.,
    grid_max: float = 2.,
    num_grids: int = 8,
    use_base_update: bool = True,
    base_activation = Tensor.silu,
    spline_weight_init_scale: float = 0.1,
  ) -> None:
    self.layers = [
      FastKANLayer(
        in_dim, out_dim,
        grid_min=grid_min,
        grid_max=grid_max,
        num_grids=num_grids,
        use_base_update=use_base_update,
        base_activation=base_activation,
        spline_weight_init_scale=spline_weight_init_scale,
      ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
    ]

  def __call__(self, x:Tensor) -> Tensor:
    for layer in self.layers:
      x = layer(x)
    return x

class AttentionWithFastKANTransform:
  def __init__(
    self,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    head_dim: int,
    num_heads: int,
    gating: bool = True,
  ):
    self.num_heads = num_heads
    total_dim = head_dim * self.num_heads
    self.gating = gating
    self.linear_q = FastKANLayer(q_dim, total_dim)
    self.linear_k = FastKANLayer(k_dim, total_dim)
    self.linear_v = FastKANLayer(v_dim, total_dim)
    self.linear_o = FastKANLayer(total_dim, q_dim)
    self.linear_g = None
    if self.gating:
      self.linear_g = FastKANLayer(q_dim, total_dim)
    # precompute the 1/sqrt(head_dim)
    self.norm = head_dim**-0.5

  def __call__(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor = None,      # additive attention bias
  ) -> Tensor:

    wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
    wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
    att = (wq * wk).sum(-1).softmax(-2)     # *qkh
    del wq, wk
    if bias is not None:
      att = att + bias[..., None]

    wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
    o = (att[..., None] * wv).sum(-3)        # *qhc
    del att, wv

    o = o.view(*o.shape[:-2], -1)           # *q(hc)

    if self.linear_g is not None:
      # gating, use raw query input
      g = self.linear_g(q)
      o = Tensor.sigmoid(g) * o

    # merge heads
    o = self.linear_o(o)
    return o


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  # instantiate the model
  model = FastKAN([28 * 28, 64, 10])
  opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-3, weight_decay=1e-4)

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 128), high=X_train.shape[0])
    loss = (model(X_train[samples].view(-1, 28 * 28))).sparse_categorical_crossentropy(Y_train[samples]).backward()
    opt.step()
    return loss

  @TinyJit
  @Tensor.test()
  def get_test_acc() -> Tensor: return (model(X_test.view(-1, 28 * 28)).argmax(axis=1) == Y_test).mean() * 100.0

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
